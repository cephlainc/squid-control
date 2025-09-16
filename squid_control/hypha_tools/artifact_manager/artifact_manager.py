"""
This module provides the ArtifactManager class, which manages artifacts for the application.
It includes methods for creating vector collections, adding vectors, searching vectors,
and handling file uploads and downloads.
"""

import asyncio
import base64
import io
import math
import time
import uuid
import zipfile
import aiohttp
import dotenv
import httpx
import numcodecs
import numpy as np
from hypha_rpc import connect_to_server
from hypha_rpc.rpc import RemoteException
from PIL import Image

dotenv.load_dotenv()
ENV_FILE = dotenv.find_dotenv()
if ENV_FILE:
    dotenv.load_dotenv(ENV_FILE)

class SquidArtifactManager:
    """
    Manages artifacts for the application.
    """

    def __init__(self):
        self._svc = None
        self.server = None
        # Upload queue infrastructure for background uploads during scanning
        self.upload_queue = None  # Will be initialized when needed
        self.upload_worker_task = None
        self.upload_worker_running = False
        self.current_dataset_id = None
        self.current_gallery_id = None
        self.upload_frozen = False  # For handling upload failures
        self.microscope_service_id = None
        self.experiment_id = None
        self.acquisition_settings = None
        self.description = None

    async def connect_server(self, server):
        """
        Connect to the server.

        Args:
            server (Server): The server instance.
        """
        self.server = server
        self._svc = await server.get_service("public/artifact-manager")


    def _artifact_id(self, workspace, name):
        """
        Generate the artifact ID.

        Args:
            workspace (str): The workspace.
            name (str): The artifact name.

        Returns:
            str: The artifact ID.
        """
        return f"{workspace}/{name}"

    async def create_vector_collection(
        self, workspace, name, manifest, config, overwrite=False, exists_ok=False
    ):
        """
        Create a vector collection.

        Args:
            workspace (str): The workspace.
            name (str): The collection name.
            manifest (dict): The collection manifest.
            config (dict): The collection configuration.
            overwrite (bool, optional): Whether to overwrite the existing collection.
        """
        art_id = self._artifact_id(workspace, name)
        try:
            await self._svc.create(
                alias=art_id,
                type="vector-collection",
                manifest=manifest,
                config=config,
                overwrite=overwrite,
            )
        except RemoteException as e:
            if not exists_ok:
                raise e

    async def add_vectors(self, workspace, coll_name, vectors):
        """
        Add vectors to the collection.

        Args:
            workspace (str): The workspace.
            coll_name (str): The collection name.
            vectors (list): The vectors to add.
        """
        art_id = self._artifact_id(workspace, coll_name)
        await self._svc.add_vectors(artifact_id=art_id, vectors=vectors)

    async def search_vectors(self, workspace, coll_name, vector, top_k=None):
        """
        Search for vectors in the collection.

        Args:
            workspace (str): The workspace.
            coll_name (str): The collection name.
            vector (ndarray): The query vector.
            top_k (int, optional): The number of top results to return.

        Returns:
            list: The search results.
        """
        art_id = self._artifact_id(workspace, coll_name)
        return await self._svc.search_vectors(
            artifact_id=art_id, query={"cell_image_vector": vector}, limit=top_k
        )

    async def add_file(self, workspace, coll_name, file_content, file_path):
        """
        Add a file to the collection.

        Args:
            workspace (str): The workspace.
            coll_name (str): The collection name.
            file_content (bytes): The file content.
            file_path (str): The file path.
        """
        art_id = self._artifact_id(workspace, coll_name)
        await self._svc.edit(artifact_id=art_id, version="stage")
        put_url = await self._svc.put_file(art_id, file_path, download_weight=1.0)
        async with httpx.AsyncClient() as client:
            response = await client.put(put_url, data=file_content, timeout=500)
        response.raise_for_status()
        await self._svc.commit(art_id)

    async def get_file(self, workspace, coll_name, file_path):
        """
        Retrieve a file from the collection.

        Args:
            workspace (str): The workspace.
            coll_name (str): The collection name.
            file_path (str): The file path.

        Returns:
            bytes: The file content.
        """
        art_id = self._artifact_id(workspace, coll_name)
        get_url = await self._svc.get_file(art_id, file_path)

        async with httpx.AsyncClient() as client:
            response = await client.get(get_url, timeout=500)
        response.raise_for_status()

        return response.content

    async def remove_vectors(self, workspace, coll_name, vector_ids=None):
        """
        Clear the vectors in the collection.

        Args:
            workspace (str): The workspace.
            coll_name (str): The collection name.
        """
        art_id = self._artifact_id(workspace, coll_name)
        if vector_ids is None:
            all_vectors = await self._svc.list_vectors(art_id)
            while len(all_vectors) > 0:
                vector_ids = [vector["id"] for vector in all_vectors]
                await self._svc.remove_vectors(art_id, vector_ids)
                all_vectors = await self._svc.list_vectors(art_id)
        else:
            await self._svc.remove_vectors(art_id, vector_ids)

    async def list_files_in_dataset(self, dataset_id):
        """
        List all files in a dataset.

        Args:
            dataset_id (str): The ID of the dataset.

        Returns:
            list: A list of files in the dataset.
        """
        files = await self._svc.list_files(dataset_id)
        return files

    async def navigate_collections(self, parent_id=None):
        """
        Navigate through collections and datasets.

        Args:
            parent_id (str, optional): The ID of the parent collection. Defaults to None for top-level collections.

        Returns:
            list: A list of collections and datasets under the specified parent.
        """
        collections = await self._svc.list(artifact_id=parent_id)
        return collections

    async def get_file_details(self, dataset_id, file_path):
        """
        Get details of a specific file in a dataset.

        Args:
            dataset_id (str): The ID of the dataset.
            file_path (str): The path to the file in the dataset.

        Returns:
            dict: Details of the file, including size, type, and last modified date.
        """
        files = await self._svc.list_files(dataset_id)
        for file in files:
            if file['name'] == file_path:
                return file
        return None

    async def download_file(self, dataset_id, file_path, local_path):
        """
        Download a file from a dataset.

        Args:
            dataset_id (str): The ID of the dataset.
            file_path (str): The path to the file in the dataset.
            local_path (str): The local path to save the downloaded file.
        """
        get_url = await self._svc.get_file(dataset_id, file_path)
        async with httpx.AsyncClient() as client:
            response = await client.get(get_url)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)

    async def search_datasets(self, keywords=None, filters=None):
        """
        Search and filter datasets based on keywords and filters.

        Args:
            keywords (list, optional): A list of keywords for searching datasets.
            filters (dict, optional): A dictionary of filters to apply.

        Returns:
            list: A list of datasets matching the search criteria.
        """
        datasets = await self._svc.list(keywords=keywords, filters=filters)
        return datasets

    async def list_subfolders(self, dataset_id, dir_path=None):
        """
        List all subfolders in a specified directory within a dataset.

        Args:
            dataset_id (str): The ID of the dataset.
            dir_path (str, optional): The directory path within the dataset to list subfolders. Defaults to None for the root directory.

        Returns:
            list: A list of subfolders in the specified directory.
        """
        try:
            print(f"Listing files for dataset_id={dataset_id}, dir_path={dir_path}")
            files = await self._svc.list_files(dataset_id, dir_path=dir_path)
            print(f"Files received, length: {len(files)}")
            subfolders = [file for file in files if file.get('type') == 'directory']
            print(f"Subfolders filtered, length: {len(subfolders)}")
            return subfolders
        except Exception as e:
            print(f"Error listing subfolders for {dataset_id}: {e}")
            import traceback
            print(traceback.format_exc())
            return []

    async def create_or_get_microscope_gallery(self, microscope_service_id, experiment_id=None):
        """
        Create or get a gallery for a specific microscope in the agent-lens workspace.
        
        Args:
            microscope_service_id (str): The hypha service ID of the microscope
            experiment_id (str, optional): The experiment ID for gallery naming. Required if microscope_service_id ends with a number
            
        Returns:
            dict: The gallery artifact information
        """
        workspace = "agent-lens"

        # Determine gallery naming based on microscope service ID
        # Check if microscope service ID ends with a number (e.g., '-1', '-2', etc.)
        import re
        number_match = re.search(r'-(\d+)$', microscope_service_id)

        if number_match:
            # Special case: microscope ID ends with a number, use experiment-based gallery
            if experiment_id is None:
                raise ValueError("experiment_id is required when microscope_service_id ends with a number")
            gallery_number = number_match.group(1)
            # FIXED: Always use the base experiment_id for gallery naming, not folder-specific names
            # This ensures all datasets from the same experiment go into the same gallery
            gallery_alias = f"{gallery_number}-{experiment_id}"
        else:
            # Standard case: use microscope-based gallery
            gallery_alias = f"microscope-gallery-{microscope_service_id}"

        try:
            # Try to get existing gallery
            gallery = await self._svc.read(artifact_id=f"{workspace}/{gallery_alias}")
            print(f"Found existing gallery: {gallery_alias}")
            return gallery
        except Exception as e:
            # Handle both RemoteException and RemoteError, and check for various error patterns
            error_str = str(e).lower()
            if ("not found" in error_str or
                "does not exist" in error_str or
                "keyerror" in error_str or
                "artifact with id" in error_str):
                # Gallery doesn't exist, create it
                print(f"Creating new gallery: {gallery_alias}")

                # Determine gallery name and description based on type
                if number_match:
                    gallery_name = f"Experiment Gallery - {experiment_id}"
                    gallery_description = f"Dataset collection for experiment {experiment_id}"
                    gallery_type = "experiment-gallery"
                else:
                    gallery_name = f"Microscope Gallery - {microscope_service_id}"
                    gallery_description = f"Dataset collection for microscope service {microscope_service_id}"
                    gallery_type = "microscope-gallery"

                gallery_manifest = {
                    "name": gallery_name,
                    "description": gallery_description,
                    "microscope_service_id": microscope_service_id,
                    "experiment_id": experiment_id,
                    "created_by": "squid-control-system",
                    "type": gallery_type
                }

                gallery_config = {
                    "permissions": {"*": "r", "@": "r+"},
                    "collection_schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "record_type": {"type": "string", "enum": ["zarr-dataset"]},
                            "microscope_service_id": {"type": "string"},
                            "experiment_id": {"type": "string"},
                            "acquisition_settings": {"type": "object"},
                            "timestamp": {"type": "string"}
                        },
                        "required": ["name", "description", "record_type"]
                    }
                }

                gallery = await self._svc.create(
                    alias=f"{workspace}/{gallery_alias}",
                    type="collection",
                    manifest=gallery_manifest,
                    config=gallery_config
                )
                print(f"Created gallery: {gallery_alias}")
                return gallery
            else:
                raise e

    async def upload_multiple_zip_files_to_dataset(self, microscope_service_id, experiment_id,
                                                   zarr_files_info, acquisition_settings=None,
                                                   description=None, dataset_name=None):
        """
        Upload multiple zarr files to a single dataset within a gallery.
        
        Args:
            microscope_service_id (str): The hypha service ID of the microscope
            experiment_id (str): The experiment ID for gallery naming
            zarr_files_info (list): List of dicts with 'name', 'content'/'file_path', 'size_mb' for each file
                                   - 'content': file data in memory (original behavior)
                                   - 'file_path': path to file on disk (new streaming behavior)
            acquisition_settings (dict, optional): Acquisition settings metadata
            description (str, optional): Description of the dataset
            dataset_name (str, optional): Custom dataset name. If None, uses experiment_id-timestamp
            
        Returns:
            dict: Information about the uploaded dataset
        """
        workspace = "agent-lens"

        # Generate dataset name with timestamp if not provided
        if dataset_name is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            dataset_name = f"{experiment_id}-{timestamp}"

        # Validate all ZIP files with streaming validation to avoid memory exhaustion
        total_size_mb = 0
        print(f"🔍 Validating {len(zarr_files_info)} ZIP files with streaming validation...")
        
        # Use sequential validation to avoid loading multiple large files into memory
        for i, file_info in enumerate(zarr_files_info):
            print(f"  Validating file {i+1}/{len(zarr_files_info)}: {file_info['name']} ({file_info['size_mb']:.2f} MB)")
            
            # Handle both file_path and content for validation
            if 'file_path' in file_info:
                # Load ONE file at a time, validate, then release memory immediately
                with open(file_info['file_path'], 'rb') as f:
                    file_content = f.read()
                await self._validate_zarr_zip_content(file_content)
                # Explicitly delete to free memory immediately
                del file_content
                print(f"    ✅ File validated and memory freed")
            else:
                # Original behavior with content in memory (for backward compatibility)
                await self._validate_zarr_zip_content(file_info['content'])
            total_size_mb += file_info['size_mb']
        
        print(f"✅ All {len(zarr_files_info)} ZIP files validated successfully")

        # Run detailed ZIP integrity test on first file as representative
        if zarr_files_info:
            first_file = zarr_files_info[0]
            print(f"🔍 Running detailed integrity test on first file: {first_file['name']}")
            
            if 'file_path' in first_file:
                # Load ONE file, test integrity, then release memory immediately
                with open(first_file['file_path'], 'rb') as f:
                    file_content = f.read()
                zip_test_results = await self.test_zip_file_integrity(
                    file_content, f"Upload: {dataset_name} (first file)"
                )
                # Explicitly delete to free memory immediately
                del file_content
                print(f"    ✅ Integrity test completed and memory freed")
            else:
                # Original behavior with content in memory
                zip_test_results = await self.test_zip_file_integrity(
                    first_file['content'], f"Upload: {dataset_name} (first file)"
                )
            if not zip_test_results["valid"]:
                raise ValueError(f"ZIP file integrity test failed: {', '.join(zip_test_results['issues'])}")
                
            print(f"✅ Integrity test passed for first file")

        # Ensure gallery exists
        gallery = await self.create_or_get_microscope_gallery(microscope_service_id, experiment_id)

        # Check name availability
        name_check = await self.check_dataset_name_availability(microscope_service_id, dataset_name)
        if not name_check["available"]:
            raise ValueError(f"Dataset name '{dataset_name}' is not available: {name_check['reason']}")

        # Create dataset manifest
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        dataset_manifest = {
            "name": dataset_name,
            "description": description or f"Zarr dataset from microscope {microscope_service_id}",
            "record_type": "zarr-dataset",
            "microscope_service_id": microscope_service_id,
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "acquisition_settings": acquisition_settings or {},
            "file_format": "ome-zarr",
            "upload_method": "squid-control-api",
            "total_size_mb": total_size_mb,
            "file_count": len(zarr_files_info),
            "zip_format": "ZIP64-compatible"
        }

        # Create dataset in staging mode
        dataset_alias = f"{workspace}/{dataset_name}"
        dataset = await self._svc.create(
            parent_id=gallery["id"],
            alias=dataset_alias,
            manifest=dataset_manifest,
            stage=True
        )

        uploaded_files = []

        try:
            # Upload each zarr zip file with retry logic
            for i, file_info in enumerate(zarr_files_info):
                file_name = file_info['name']
                file_size_mb = file_info['size_mb']

                print(f"Uploading file {i+1}/{len(zarr_files_info)}: {file_name} ({file_size_mb:.2f} MB)")

                # Handle both file_path and content for upload
                if 'file_path' in file_info:
                    # Load ONE file at a time to avoid memory exhaustion
                    with open(file_info['file_path'], 'rb') as f:
                        file_content = f.read()
                    print(f"  📁 Loaded file into memory: {file_info['file_path']} ({file_size_mb:.2f} MB)")
                else:
                    # Original behavior: content already in memory
                    file_content = file_info['content']

                # Upload zarr zip file with retry logic
                await self._upload_large_zip_multipart(
                    dataset["id"],
                    file_content,
                    max_retries=3,
                    file_path=f"{file_name}.zip"
                )
                
                # Clear file content from memory immediately after upload
                if 'file_path' in file_info:
                    del file_content
                    print(f"  🧹 Cleared {file_size_mb:.2f} MB from memory")

                uploaded_files.append({
                    "name": file_name,
                    "size_mb": file_size_mb,
                    "file_index": i+1
                })

                print(f"Successfully uploaded file: {file_name}")

            # Commit the dataset only after all files are uploaded
            await self._svc.commit(dataset["id"])

            print(f"Successfully uploaded zarr dataset: {dataset_name} ({total_size_mb:.2f} MB, {len(uploaded_files)} files)")
            return {
                "success": True,
                "dataset_id": dataset["id"],
                "dataset_name": dataset_name,
                "gallery_id": gallery["id"],
                "experiment_id": experiment_id,
                "upload_timestamp": timestamp,
                "total_size_mb": total_size_mb,
                "uploaded_files": uploaded_files,
                "file_count": len(uploaded_files)
            }

        except Exception as e:
            # If upload fails, try to clean up the staged dataset
            try:
                await self._svc.discard(dataset["id"])
            except:
                pass
            raise e

    async def start_upload_worker(self, microscope_service_id, experiment_id, acquisition_settings=None, description=None):
        """
        Start the background upload worker for uploading wells during scanning.
        
        Args:
            microscope_service_id (str): The hypha service ID of the microscope
            experiment_id (str): The experiment ID for dataset naming
            acquisition_settings (dict, optional): Acquisition settings metadata
            description (str, optional): Description of the dataset
        """
        if self.upload_worker_running:
            print("Upload worker already running")
            return

        # Initialize upload queue
        self.upload_queue = asyncio.Queue()
        self.upload_worker_running = True
        self.upload_frozen = False
        self.microscope_service_id = microscope_service_id
        self.experiment_id = experiment_id
        self.acquisition_settings = acquisition_settings
        self.description = description

        # Create dataset for this upload session
        await self._create_upload_dataset()

        # Start background worker
        self.upload_worker_task = asyncio.create_task(self._upload_worker_loop())
        print(f"Started upload worker for experiment: {experiment_id}")

    async def stop_upload_worker(self):
        """
        Stop the background upload worker and commit the dataset.
        """
        if not self.upload_worker_running:
            print("Upload worker not running")
            return

        print("Stopping upload worker - waiting for queue to empty...")
        self.upload_worker_running = False

        # Wait for upload worker to complete and process remaining items
        if self.upload_worker_task:
            try:
                # Give extra time for the worker to process remaining items in queue
                await asyncio.wait_for(self.upload_worker_task, timeout=60.0)
            except asyncio.TimeoutError:
                print("Upload worker did not stop gracefully, cancelling")
                self.upload_worker_task.cancel()
                try:
                    await self.upload_worker_task
                except asyncio.CancelledError:
                    pass

        # Commit the dataset if it exists
        if self.current_dataset_id:
            try:
                await self._svc.commit(self.current_dataset_id)
                print(f"Committed dataset: {self.current_dataset_id}")
            except Exception as e:
                print(f"Failed to commit dataset: {e}")

        # Reset state
        self.upload_queue = None
        self.upload_worker_task = None
        self.current_dataset_id = None
        self.current_gallery_id = None
        self.microscope_service_id = None
        self.experiment_id = None
        self.acquisition_settings = None
        self.description = None
        print("Upload worker stopped")

    async def add_well_to_upload_queue(self, well_name, well_zip_content, well_size_mb):
        """
        Add a well to the upload queue for background processing.
        
        Args:
            well_name (str): Name of the well (e.g., "well_A1_96")
            well_zip_content (bytes): ZIP content of the well canvas
            well_size_mb (float): Size of the ZIP content in MB
        """
        if not self.upload_worker_running or self.upload_frozen:
            print(f"Upload worker not running or frozen, skipping upload for {well_name}")
            return

        try:
            await self.upload_queue.put({
                'name': well_name,
                'content': well_zip_content,
                'size_mb': well_size_mb
            })
            print(f"Added {well_name} to upload queue")
        except Exception as e:
            print(f"Failed to add {well_name} to upload queue: {e}")

    async def _create_upload_dataset(self):
        """
        Create a dataset for the current upload session.
        """
        workspace = "agent-lens"

        # Generate dataset name with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        dataset_name = f"{self.experiment_id}-{timestamp}"

        # Ensure gallery exists
        gallery = await self.create_or_get_microscope_gallery(self.microscope_service_id, self.experiment_id)
        self.current_gallery_id = gallery["id"]

        # Create dataset manifest
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        dataset_manifest = {
            "name": dataset_name,
            "description": self.description or f"Zarr dataset from microscope {self.microscope_service_id}",
            "record_type": "zarr-dataset",
            "microscope_service_id": self.microscope_service_id,
            "experiment_id": self.experiment_id,
            "timestamp": timestamp,
            "acquisition_settings": self.acquisition_settings or {},
            "file_format": "ome-zarr",
            "upload_method": "squid-control-api-background",
            "zip_format": "ZIP64-compatible"
        }

        # Create dataset in staging mode
        dataset_alias = f"{workspace}/{dataset_name}"
        dataset = await self._svc.create(
            parent_id=gallery["id"],
            alias=dataset_alias,
            manifest=dataset_manifest,
            stage=True
        )

        self.current_dataset_id = dataset["id"]
        print(f"Created upload dataset: {dataset_name}")

    async def _upload_worker_loop(self):
        """
        Background loop that processes the upload queue.
        """
        while self.upload_worker_running:
            try:
                # Get well from queue with timeout
                well_info = await asyncio.wait_for(self.upload_queue.get(), timeout=1.0)

                # Upload with retry logic
                success = await self._upload_single_well_with_retry(well_info)

                if not success:
                    # Freeze upload queue after 3 failed attempts
                    self.upload_frozen = True
                    print(f"Upload failed after 3 retries for {well_info['name']}, freezing upload queue")
                    break

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Upload worker error: {e}")
                # Don't break on general errors, continue processing
                continue

        # Process any remaining items in the queue before stopping
        print("Upload worker stopping - processing remaining items in queue...")
        while not self.upload_queue.empty():
            try:
                well_info = self.upload_queue.get_nowait()
                print(f"Processing remaining item: {well_info['name']}")

                # Upload with retry logic
                success = await self._upload_single_well_with_retry(well_info)

                if not success:
                    print(f"Failed to upload remaining item {well_info['name']}")

            except Exception as e:
                print(f"Error processing remaining upload item: {e}")
                break

        print("Upload worker loop completed")

    async def _upload_single_well_with_retry(self, well_info):
        """
        Upload a single well with retry logic.
        
        Args:
            well_info (dict): Well information with 'name', 'content', 'size_mb'
            
        Returns:
            bool: True if upload succeeded, False otherwise
        """
        for attempt in range(3):
            try:
                await self._upload_large_zip_multipart(
                    self.current_dataset_id,
                    well_info['content'],
                    max_retries=1,  # Single attempt per retry cycle
                    file_path=f"{well_info['name']}.zip"
                )
                print(f"Successfully uploaded well: {well_info['name']} (attempt {attempt + 1})")
                return True
            except Exception as e:
                print(f"Upload attempt {attempt + 1} failed for {well_info['name']}: {e}")
                if attempt < 2:
                    wait_time = 2 ** attempt
                    print(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)

        return False

    async def check_dataset_name_availability(self, microscope_service_id, dataset_name):
        """
        Check if a dataset name is available in the microscope's gallery.
        
        Args:
            microscope_service_id (str): The hypha service ID of the microscope
            dataset_name (str): The proposed dataset name
            
        Returns:
            dict: Information about name availability and suggestions
        """
        workspace = "agent-lens"
        gallery_alias = f"microscope-gallery-{microscope_service_id}"
        dataset_alias = f"{workspace}/{dataset_name}"

        # Validate dataset name format
        import re
        if not re.match(r'^[a-z0-9][a-z0-9\-:]*[a-z0-9]$', dataset_name):
            return {
                "available": False,
                "reason": "Invalid name format. Use lowercase letters, numbers, hyphens, and colons only. Must start and end with alphanumeric character.",
                "suggestions": []
            }

        try:
            # Check if dataset already exists
            existing_dataset = await self._svc.read(artifact_id=dataset_alias)

            # Generate alternative suggestions
            suggestions = []
            timestamp = int(time.time())
            base_suggestions = [
                f"{dataset_name}-v2",
                f"{dataset_name}-{timestamp}",
                f"{dataset_name}-copy",
                f"{dataset_name}-new"
            ]

            for suggestion in base_suggestions:
                try:
                    await self._svc.read(artifact_id=f"{workspace}/{suggestion}")
                except Exception:
                    suggestions.append(suggestion)
                    if len(suggestions) >= 3:
                        break

            return {
                "available": False,
                "reason": "Dataset name already exists",
                "existing_dataset": existing_dataset,
                "suggestions": suggestions
            }

        except Exception as e:
            # Handle both RemoteException and RemoteError, and check for various error patterns
            error_str = str(e).lower()
            if ("not found" in error_str or
                "does not exist" in error_str or
                "keyerror" in error_str or
                "artifact with id" in error_str):
                return {
                    "available": True,
                    "reason": "Name is available",
                    "suggestions": []
                }
            else:
                raise e

    async def upload_zarr_dataset(self, microscope_service_id, dataset_name, zarr_zip_content,
                                 acquisition_settings=None, description=None, experiment_id=None):
        """
        Upload a zarr fileset as a zip file to the microscope's gallery.
        
        Args:
            microscope_service_id (str): The hypha service ID of the microscope
            dataset_name (str): The name for the dataset (will be overridden if experiment_id is provided)
            zarr_zip_content (bytes): The zip file content containing the zarr fileset
            acquisition_settings (dict, optional): Acquisition settings metadata
            description (str, optional): Description of the dataset
            experiment_id (str, optional): The experiment ID for dataset naming. If provided, dataset_name will be overridden with '{experiment_id}-{date and time}'
            
        Returns:
            dict: Information about the uploaded dataset
        """
        workspace = "agent-lens"

        # Generate dataset name if experiment_id is provided
        if experiment_id is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            dataset_name = f"{experiment_id}-{timestamp}"

        # Validate ZIP file before upload
        await self._validate_zarr_zip_content(zarr_zip_content)

        # Run detailed ZIP integrity test
        zip_test_results = await self.test_zip_file_integrity(zarr_zip_content, f"Upload: {dataset_name}")
        if not zip_test_results["valid"]:
            raise ValueError(f"ZIP file integrity test failed: {', '.join(zip_test_results['issues'])}")

        # Ensure gallery exists
        gallery = await self.create_or_get_microscope_gallery(microscope_service_id, experiment_id)

        # Check name availability
        name_check = await self.check_dataset_name_availability(microscope_service_id, dataset_name)
        if not name_check["available"]:
            raise ValueError(f"Dataset name '{dataset_name}' is not available: {name_check['reason']}")

        # Create dataset manifest
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        zip_size_mb = len(zarr_zip_content) / (1024 * 1024)

        dataset_manifest = {
            "name": dataset_name,
            "description": description or f"Zarr dataset from microscope {microscope_service_id}",
            "record_type": "zarr-dataset",
            "microscope_service_id": microscope_service_id,
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "acquisition_settings": acquisition_settings or {},
            "file_format": "ome-zarr",
            "upload_method": "squid-control-api",
            "zip_size_mb": zip_size_mb,
            "zip_format": "ZIP64-compatible"
        }

        # Create dataset in staging mode
        dataset_alias = f"{workspace}/{dataset_name}"
        dataset = await self._svc.create(
            parent_id=gallery["id"],
            alias=dataset_alias,
            manifest=dataset_manifest,
            stage=True
        )

        try:
            # Upload zarr zip file with retry logic
            await self._upload_large_zip_multipart(dataset["id"], zarr_zip_content, max_retries=3)

            # Commit the dataset
            await self._svc.commit(dataset["id"])

            print(f"Successfully uploaded zarr dataset: {dataset_name} ({zip_size_mb:.2f} MB)")
            return {
                "success": True,
                "dataset_id": dataset["id"],
                "dataset_name": dataset_name,
                "gallery_id": gallery["id"],
                "experiment_id": experiment_id,
                "upload_timestamp": timestamp,
                "zip_size_mb": zip_size_mb
            }

        except Exception as e:
            # If upload fails, try to clean up the staged dataset
            try:
                await self._svc.discard(dataset["id"])
            except:
                pass
            raise e

    async def _validate_zarr_zip_content(self, zarr_zip_content: bytes) -> None:
        """
        Validate that the ZIP content is properly formatted and not corrupted.
        
        Args:
            zarr_zip_content (bytes): The ZIP file content to validate
            
        Raises:
            ValueError: If ZIP file is invalid or corrupted
        """
        # Move CPU-intensive ZIP validation to thread pool to avoid blocking asyncio loop
        def _validate_zip_sync(zip_content: bytes) -> None:
            import io

            zip_buffer = io.BytesIO(zip_content)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                # Test the ZIP file structure
                file_list = zip_file.namelist()
                if not file_list:
                    raise ValueError("ZIP file is empty")

                # Check for expected zarr structure
                zarr_files = [f for f in file_list if f.startswith('data.zarr/')]
                if not zarr_files:
                    raise ValueError("ZIP file does not contain expected 'data.zarr/' structure")

                # Test that we can read the ZIP file entries
                test_count = min(5, len(file_list))
                for i in range(test_count):
                    try:
                        with zip_file.open(file_list[i]) as f:
                            f.read(1024)  # Read a small chunk
                    except Exception as e:
                        raise ValueError(f"Cannot read file {file_list[i]} from ZIP: {e}")

                print(f"ZIP validation passed: {len(file_list)} files, {len(zip_content) / (1024*1024):.2f} MB")

        try:
            await asyncio.to_thread(_validate_zip_sync, zarr_zip_content)
        except zipfile.BadZipFile as e:
            raise ValueError(f"Invalid ZIP file format: {e}")
        except Exception as e:
            raise ValueError(f"ZIP file validation failed: {e}")

    def _calculate_optimal_part_size(self, file_size_bytes: int) -> tuple[int, int]:
        """
        Calculate optimal part size and count for multipart upload.
        
        Args:
            file_size_bytes (int): Size of the file in bytes
            
        Returns:
            tuple[int, int]: (part_size_bytes, part_count)
        """
        # Target part size: 100MB for optimal performance
        target_part_size = 100 * 1024 * 1024  # 100MB

        # Minimum part size required by S3: 5MB
        min_part_size = 5 * 1024 * 1024  # 5MB

        # Maximum part count allowed: 10,000
        max_parts = 10000

        if file_size_bytes <= target_part_size:
            # Small file: use single part
            return file_size_bytes, 1

        # Calculate part count based on target size
        part_count = math.ceil(file_size_bytes / target_part_size)

        # Ensure we don't exceed maximum parts
        if part_count > max_parts:
            # Recalculate with larger part size
            part_size = math.ceil(file_size_bytes / max_parts)
            # Ensure minimum part size
            if part_size < min_part_size:
                raise ValueError(f"File too large for multipart upload: {file_size_bytes / (1024*1024):.1f} MB")
            return part_size, max_parts

        # Ensure minimum part size
        actual_part_size = math.ceil(file_size_bytes / part_count)
        if actual_part_size < min_part_size:
            # Recalculate with minimum part size
            part_count = math.ceil(file_size_bytes / min_part_size)
            if part_count > max_parts:
                raise ValueError(f"File too large for multipart upload: {file_size_bytes / (1024*1024):.1f} MB")
            return min_part_size, part_count

        return target_part_size, part_count

    async def _upload_large_zip_multipart(self, dataset_id: str, zarr_zip_content: bytes, max_retries: int = 3, file_path: str = "zarr_dataset.zip") -> None:
        """
        Upload large ZIP file using multipart upload with retry logic and appropriate timeouts.
        
        Args:
            dataset_id (str): The dataset ID
            zarr_zip_content (bytes): The ZIP file content
            max_retries (int): Maximum number of retry attempts
            file_path (str): The file path within the dataset
            
        Raises:
            Exception: If upload fails after all retries
        """
        zip_size_mb = len(zarr_zip_content) / (1024 * 1024)

        # Calculate optimal part size and count
        try:
            part_size_bytes, part_count = self._calculate_optimal_part_size(len(zarr_zip_content))
        except ValueError as e:
            raise Exception(f"Cannot upload file: {e}")

        # Calculate timeout based on file size (minimum 5 minutes, add 1 minute per 50MB)
        timeout_seconds = max(300, int(zip_size_mb / 50) * 60 + 300)

        for attempt in range(max_retries):
            try:
                print(f"Multipart upload attempt {attempt + 1}/{max_retries} for {zip_size_mb:.2f} MB ZIP file to {file_path}")
                print(f"  Using {part_count} parts, {part_size_bytes / (1024*1024):.1f} MB per part, timeout: {timeout_seconds}s")

                # Step 1: Start multipart upload
                multipart_info = await self._svc.put_file_start_multipart(
                    artifact_id=dataset_id,
                    file_path=file_path,
                    part_count=part_count,
                    expires_in=3600  # 1 hour expiration
                )

                upload_id = multipart_info["upload_id"]
                part_urls = multipart_info["parts"]

                print(f"  Started multipart upload with ID: {upload_id}")

                # Step 2: Upload all parts concurrently
                async def upload_part(part_info):
                    part_number = part_info["part_number"]
                    url = part_info["url"]

                    # Calculate start and end positions for this part
                    start_pos = (part_number - 1) * part_size_bytes
                    end_pos = min(start_pos + part_size_bytes, len(zarr_zip_content))

                    # Extract the specific part from the ZIP content
                    part_data = zarr_zip_content[start_pos:end_pos]

                    # Upload the part with appropriate timeout
                    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
                        response = await client.put(
                            url,
                            content=part_data,
                            headers={
                                'Content-Type': 'application/octet-stream',
                                'Content-Length': str(len(part_data))
                            }
                        )
                        response.raise_for_status()

                        return {
                            "part_number": part_number,
                            "etag": response.headers["ETag"].strip('"')
                        }

                # Upload parts with controlled concurrency (max 5 concurrent uploads)
                semaphore = asyncio.Semaphore(5)

                async def upload_with_semaphore(part_info):
                    async with semaphore:
                        return await upload_part(part_info)

                print(f"  Uploading {len(part_urls)} parts with controlled concurrency...")
                uploaded_parts = await asyncio.gather(*[
                    upload_with_semaphore(part) for part in part_urls
                ])

                print("  All parts uploaded successfully")

                # Step 3: Complete multipart upload
                try:
                    # Validate that we have all parts before completing
                    if len(uploaded_parts) != part_count:
                        raise Exception(f"Expected {part_count} parts but got {len(uploaded_parts)}")

                    # Sort parts by part number to ensure correct order
                    uploaded_parts.sort(key=lambda x: x["part_number"])

                    # Verify part numbers are sequential
                    for i, part in enumerate(uploaded_parts):
                        if part["part_number"] != i + 1:
                            raise Exception(f"Missing or out-of-order part: expected {i + 1}, got {part['part_number']}")

                    result = await self._svc.put_file_complete_multipart(
                        artifact_id=dataset_id,
                        upload_id=upload_id,
                        parts=uploaded_parts
                    )

                    if result["success"]:
                        print(f"Multipart upload completed successfully on attempt {attempt + 1}")
                        return
                    else:
                        raise Exception(f"Multipart upload completion failed: {result['message']}")

                except Exception as completion_error:
                    print(f"Error completing multipart upload: {completion_error}")
                    # Try to abort the multipart upload to clean up
                    try:
                        # Note: The current API doesn't have abort_multipart, but we could add it later
                        print(f"Multipart upload {upload_id} failed, parts may need manual cleanup")
                    except Exception as abort_error:
                        print(f"Could not abort multipart upload: {abort_error}")

                    # Re-raise the completion error to trigger retry
                    raise completion_error

            except httpx.TimeoutException as e:
                print(f"Upload timeout on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Upload failed after {max_retries} attempts due to timeout")

            except httpx.HTTPStatusError as e:
                print(f"Upload HTTP error on attempt {attempt + 1}: {e.response.status_code} - {e.response.text}")
                if e.response.status_code == 413:  # Payload too large
                    raise Exception(f"ZIP file is too large ({zip_size_mb:.2f} MB) for upload")
                elif e.response.status_code >= 500:  # Server errors - retry
                    if attempt == max_retries - 1:
                        raise Exception(f"Server error after {max_retries} attempts: {e}")
                else:  # Client errors - don't retry
                    raise Exception(f"Upload failed with HTTP {e.response.status_code}: {e.response.text}")

            except Exception as e:
                print(f"Upload error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Upload failed after {max_retries} attempts: {e}")

            # Wait before retry (exponential backoff)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)

    async def test_zip_file_integrity(self, zip_content: bytes, description: str = "ZIP file") -> dict:
        """
        Test ZIP file integrity and provide detailed diagnostics.
        
        Args:
            zip_content (bytes): The ZIP file content to test
            description (str): Description of the ZIP file for logging
            
        Returns:
            dict: Detailed test results including file count, size, and any issues
        """
        # Move CPU-intensive ZIP integrity testing to thread pool to avoid blocking asyncio loop
        def _test_zip_integrity_sync(zip_content: bytes, description: str) -> dict:
            import io

            results = {
                "valid": False,
                "size_mb": len(zip_content) / (1024 * 1024),
                "file_count": 0,
                "zip64_required": False,
                "zip64_enabled": False,
                "compression_method": None,
                "issues": [],
                "sample_files": []
            }

            try:
                zip_buffer = io.BytesIO(zip_content)

                # Test basic ZIP file opening
                with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                    file_list = zip_file.namelist()
                    results["file_count"] = len(file_list)

                    if not file_list:
                        results["issues"].append("ZIP file is empty")
                        return results

                    # Check if ZIP64 is required based on size and file count
                    results["zip64_required"] = results["size_mb"] > 200 or results["file_count"] > 65535

                    # Test central directory access
                    try:
                        info_list = zip_file.infolist()

                        # Check for ZIP64 format indicators
                        zip64_indicators = []

                        # Check for large individual files (>= 4GB)
                        large_files = any(info.file_size >= 0xFFFFFFFF or info.compress_size >= 0xFFFFFFFF for info in info_list)
                        if large_files:
                            zip64_indicators.append("large_files")

                        # Check total archive size vs ZIP32 limits (>= 4GB)
                        if results["size_mb"] * 1024 * 1024 >= 0xFFFFFFFF:
                            zip64_indicators.append("archive_size")

                        # Check file count vs ZIP32 limits (>= 65535 files)
                        if results["file_count"] >= 0xFFFF:
                            zip64_indicators.append("file_count")

                        # For large files, check if ZIP64 format is actually being used
                        # The key insight: if ZIP64 is required but the file can be read successfully,
                        # then ZIP64 format is likely being used correctly
                        if results["zip64_required"]:
                            # Try to access the central directory - if this succeeds for large files,
                            # it means ZIP64 format is working
                            try:
                                # Test reading a few files to verify ZIP64 access works
                                test_files = min(3, len(info_list))
                                for i in range(test_files):
                                    if info_list[i].file_size > 0:
                                        with zip_file.open(info_list[i]) as f:
                                            f.read(1)  # Read just one byte to test access

                                # If we can successfully read files from a large ZIP, ZIP64 is working
                                results["zip64_enabled"] = True
                                zip64_indicators.append("successful_access")

                            except Exception as e:
                                # If we can't read files from a large ZIP, ZIP64 might not be working
                                results["zip64_enabled"] = False
                                results["issues"].append(f"ZIP64 access test failed: {e}")
                        else:
                            # For smaller files, ZIP64 is not required
                            results["zip64_enabled"] = False

                        results["zip64_indicators"] = zip64_indicators

                        # Get compression method from first file
                        if info_list:
                            results["compression_method"] = info_list[0].compress_type

                    except Exception as e:
                        results["issues"].append(f"Cannot access central directory: {e}")
                        return results

                    # Test reading sample files
                    sample_count = min(5, len(file_list))
                    for i in range(sample_count):
                        try:
                            file_info = zip_file.getinfo(file_list[i])
                            with zip_file.open(file_list[i]) as f:
                                data = f.read(1024)  # Read first 1KB
                                results["sample_files"].append({
                                    "name": file_list[i],
                                    "size": file_info.file_size,
                                    "compressed_size": file_info.compress_size,
                                    "readable": True
                                })
                        except Exception as e:
                            results["issues"].append(f"Cannot read file {file_list[i]}: {e}")
                            results["sample_files"].append({
                                "name": file_list[i],
                                "readable": False,
                                "error": str(e)
                            })

                    # Final validation: if ZIP64 is required but not enabled, that's an issue
                    if results["zip64_required"] and not results["zip64_enabled"]:
                        results["issues"].append("Large file requires ZIP64 format but ZIP64 is not enabled")

                    # Mark as valid if no issues found
                    results["valid"] = len(results["issues"]) == 0

            except zipfile.BadZipFile as e:
                results["issues"].append(f"Invalid ZIP file format: {e}")
            except Exception as e:
                results["issues"].append(f"ZIP file test failed: {e}")

            # Log results
            status = "VALID" if results["valid"] else "INVALID"
            print(f"ZIP Test [{description}]: {status}")
            print(f"  Size: {results['size_mb']:.2f} MB, Files: {results['file_count']}")
            print(f"  ZIP64 Required: {results['zip64_required']}, Enabled: {results['zip64_enabled']}")
            if "zip64_indicators" in results and results["zip64_indicators"]:
                print(f"  ZIP64 Indicators: {', '.join(results['zip64_indicators'])}")
            print(f"  Compression: {results['compression_method']}")

            if results["issues"]:
                print(f"  Issues: {', '.join(results['issues'])}")

            return results

        return await asyncio.to_thread(_test_zip_integrity_sync, zip_content, description)

# Constants
SERVER_URL = "https://hypha.aicell.io"
ARTIFACT_ALIAS = "20250824-example-data-20250824-221822"
DEFAULT_CHANNEL = "BF_LED_matrix_full"

# New class to replace TileManager using Zarr for efficient access
class ZarrImageManager:
    def __init__(self):
        self.artifact_manager = None
        self.artifact_manager_server = None
        self.workspace = "agent-lens"  # Default workspace
        self.chunk_size = 256  # Default chunk size for Zarr
        self.channels = [
            "BF_LED_matrix_full",
            "Fluorescence_405_nm_Ex",
            "Fluorescence_488_nm_Ex",
            "Fluorescence_561_nm_Ex",
            "Fluorescence_638_nm_Ex"
        ]
        self.is_running = True
        self.session = None
        self.default_timestamp = "20250824-example-data-20250824-221822"  # Set a default timestamp
        self.scale_key = 'scale0'

        # New attributes for HTTP-based access
        self.metadata_cache = {}  # Cache for .zarray and .zgroup metadata
        self.metadata_cache_lock = asyncio.Lock()
        self.processed_tile_cache = {}  # Cache for processed tiles
        self.processed_tile_ttl = 40 * 60  # 40 minutes in seconds
        self.processed_tile_cache_size = 1000  # Max number of tiles to cache
        self.empty_regions_cache = {}  # Cache for known empty regions
        self.empty_regions_cache_size = 500  # Max number of empty regions to cache
        self.http_session_lock = asyncio.Lock()
        self.server_url = "https://hypha.aicell.io"

    async def _get_http_session(self):
        """Get or create an aiohttp.ClientSession with increased connection pool."""
        async with self.http_session_lock:
            if self.session is None or self.session.closed:
                connector = aiohttp.TCPConnector(
                    limit_per_host=50,  # Max connections per host
                    limit=100,          # Total max connections
                    ssl=False if "localhost" in self.server_url else True
                )
                self.session = aiohttp.ClientSession(connector=connector)
            return self.session

    async def _fetch_zarr_metadata(self, dataset_alias, metadata_path_in_dataset, use_cache=True):
        """
        Fetch and cache Zarr metadata (.zgroup or .zarray) for a given dataset alias.
        Args:
            dataset_alias (str): The alias of the dataset (e.g., "agent-lens/20250824-example-data-20250824-221822")
            metadata_path_in_dataset (str): Path within the dataset (e.g., "Channel/scaleN/.zarray")
            use_cache (bool): Whether to use cached metadata. Defaults to True.
        """
        cache_key = (dataset_alias, metadata_path_in_dataset)
        if use_cache:
            async with self.metadata_cache_lock:
                if cache_key in self.metadata_cache:
                    print(f"Using cached metadata for {cache_key}")
                    return self.metadata_cache[cache_key]

        if not self.artifact_manager:
            print("Artifact manager not available in ZarrImageManager for metadata fetch.")
            # Attempt to connect if not already
            await self.connect()
            if not self.artifact_manager:
                raise ConnectionError("Artifact manager connection failed.")

        try:
            print(f"Fetching metadata: dataset_alias='{dataset_alias}', path='{metadata_path_in_dataset}'")

            metadata_content_bytes = await self.artifact_manager.get_file(
                self.workspace,
                dataset_alias.split('/')[-1],  # Extract artifact name from full path
                metadata_path_in_dataset
            )
            metadata_str = metadata_content_bytes.decode('utf-8')
            import json
            metadata = json.loads(metadata_str)

            async with self.metadata_cache_lock:
                self.metadata_cache[cache_key] = metadata
            print(f"Fetched and cached metadata for {cache_key}")
            return metadata
        except Exception as e:
            print(f"Error fetching metadata for {dataset_alias} / {metadata_path_in_dataset}: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    async def connect(self,server_url="https://hypha.aicell.io"):
        """Connect to the Artifact Manager service and initialize http session."""
        try:
            self.server_url = server_url.rstrip('/') # Ensure no trailing slash

            self.artifact_manager_server = await connect_to_server({
                "client_id": f"zarr-image-client-for-squid-{uuid.uuid4()}",
                "server_url": server_url,
            })

            self.artifact_manager = SquidArtifactManager()
            await self.artifact_manager.connect_server(self.artifact_manager_server)

            # Initialize aiohttp session
            await self._get_http_session()  # Ensures session is created

            # Prime metadata for a default dataset if needed, or remove if priming is dynamic
            # Example: await self.prime_metadata("agent-lens/20250824-example-data-20250824-221822", self.channels[0], scale=0)

            print("ZarrImageManager connected successfully")
            return True
        except Exception as e:
            print(f"Error connecting to artifact manager: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    async def close(self):
        """Close the image manager and cleanup resources"""
        self.is_running = False

        # Clear all caches
        self.processed_tile_cache.clear()
        async with self.metadata_cache_lock:
            self.metadata_cache.clear()
        self.empty_regions_cache.clear()

        # Close the aiohttp session
        async with self.http_session_lock:
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None

        # Disconnect from the server
        if self.artifact_manager_server:
            await self.artifact_manager_server.disconnect()
            self.artifact_manager_server = None
            self.artifact_manager = None

    def _add_to_empty_regions_cache(self, key):
        """Add a region key to the empty regions cache."""
        # Add to cache
        self.empty_regions_cache[key] = True # Store True instead of expiry_time

        # Basic FIFO size control if cache exceeds max size
        if len(self.empty_regions_cache) > self.empty_regions_cache_size:
            try:
                # Get the first key inserted (FIFO)
                first_key = next(iter(self.empty_regions_cache))
                del self.empty_regions_cache[first_key]
                print(f"Cleaned up oldest entry {first_key} from empty regions cache due to size limit.")
            except StopIteration: # pragma: no cover
                pass # Cache might have been cleared by another operation concurrently

    async def get_chunk_np_data(self, dataset_id, channel, scale, x, y, well_id="F5"):
        """
        Get a chunk as numpy array using new OME-Zarr well-based format.
        Args:
            dataset_id (str): The alias of the dataset.
            channel (str): Channel name
            scale (int): Scale level
            x (int): X coordinate of the chunk for this scale.
            y (int): Y coordinate of the chunk for this scale.
            well_id (str): Well ID (e.g., "F5") - defaults to F5 for backward compatibility
        Returns:
            np.ndarray or None: Chunk data as numpy array, or None if not found/empty/error.
        """
        start_time = time.time()
        # Key for processed_tile_cache and empty_regions_cache
        tile_cache_key = f"{dataset_id}:{well_id}:{channel}:{scale}:{x}:{y}"

        # 1. Check processed tile cache
        if tile_cache_key in self.processed_tile_cache:
            cached_data = self.processed_tile_cache[tile_cache_key]
            if time.time() - cached_data['timestamp'] < self.processed_tile_ttl:
                print(f"Using cached processed tile data for {tile_cache_key}")
                return cached_data['data']
            else:
                del self.processed_tile_cache[tile_cache_key]

        # 2. Check empty regions cache
        if tile_cache_key in self.empty_regions_cache:
            # No TTL check, if it's in the cache, it's considered empty
            print(f"Skipping known empty tile: {tile_cache_key}")
            return None

        # NEW FORMAT: Get well metadata from OME-Zarr structure
        artifact_name_only = dataset_id.split('/')[-1]
        well_zip_path = f"well_{well_id}_96.zip"
        zarray_path_in_well = f"data.zarr/{scale}/.zarray"
        
        # Construct URL to access .zarray metadata in the well ZIP
        zarray_metadata_url = f"{self.server_url}/{self.workspace}/artifacts/{artifact_name_only}/zip-files/{well_zip_path}?path={zarray_path_in_well}"
        
        # Fetch .zarray metadata
        http_session = await self._get_http_session()
        try:
            async with http_session.get(zarray_metadata_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    print(f"Failed to get .zarray metadata from {zarray_metadata_url}: HTTP {response.status}")
                    self._add_to_empty_regions_cache(tile_cache_key)
                    return None
                # Read as text and parse JSON manually to avoid MIME type issues
                response_text = await response.text()
                import json
                zarray_metadata = json.loads(response_text)
        except Exception as e:
            print(f"Error fetching .zarray metadata from {zarray_metadata_url}: {e}")
            self._add_to_empty_regions_cache(tile_cache_key)
            return None
        
        if not zarray_metadata:
            print(f"Failed to get .zarray metadata for {dataset_id}/well_{well_id}/{scale}")
            self._add_to_empty_regions_cache(tile_cache_key)
            return None

        try:
            # OME-Zarr format: shape is [T, C, Z, Y, X]
            z_shape = zarray_metadata["shape"]         # [T, C, Z, total_height, total_width]
            z_chunks = zarray_metadata["chunks"]       # [t_chunk, c_chunk, z_chunk, chunk_height, chunk_width]
            z_dtype_str = zarray_metadata["dtype"]
            z_dtype = np.dtype(z_dtype_str)
            z_compressor_meta = zarray_metadata.get("compressor")  # Can be null
            z_fill_value = zarray_metadata.get("fill_value", 0)  # Important for empty/partial chunks

        except KeyError as e:
            print(f"Incomplete .zarray metadata for {dataset_id}/well_{well_id}/{scale}: Missing key {e}")
            return None

        # Get channel index from well metadata
        channel_index = await self._get_channel_index_from_well(dataset_id, well_id, channel)
        if channel_index is None:
            print(f"Channel '{channel}' not found in well {well_id}")
            self._add_to_empty_regions_cache(tile_cache_key)
            return None

        # Check chunk coordinates are within bounds (using Y, X dimensions from 5D array)
        image_height, image_width = z_shape[3], z_shape[4]  # Y, X dimensions
        chunk_height, chunk_width = z_chunks[3], z_chunks[4]  # Y, X chunk sizes
        
        num_chunks_y_total = (image_height + chunk_height - 1) // chunk_height
        num_chunks_x_total = (image_width + chunk_width - 1) // chunk_width

        if not (0 <= y < num_chunks_y_total and 0 <= x < num_chunks_x_total):
            print(f"Chunk coordinates ({x}, {y}) out of bounds for {dataset_id}/well_{well_id}/scale{scale} (max: {num_chunks_x_total-1}, {num_chunks_y_total-1})")
            self._add_to_empty_regions_cache(tile_cache_key)
            return None

        # NEW FORMAT: Construct chunk path in OME-Zarr format
        # Chunk filename format: t.c.z.y.x (all coordinates needed)
        t_coord = 0  # Default timepoint
        c_coord = channel_index
        z_coord = 0  # Default Z slice
        chunk_filename = f"{t_coord}.{c_coord}.{z_coord}.{y}.{x}"
        
        chunk_path_in_well = f"data.zarr/{scale}/{chunk_filename}"
        
        # Construct the full chunk download URL
        chunk_download_url = f"{self.server_url}/{self.workspace}/artifacts/{artifact_name_only}/zip-files/{well_zip_path}?path={chunk_path_in_well}"

        print(f"Attempting to fetch chunk: {chunk_download_url}")

        raw_chunk_bytes = None
        try:
            async with http_session.get(chunk_download_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    raw_chunk_bytes = await response.read()
                elif response.status == 404:
                    print(f"Chunk not found (404) at {chunk_download_url}. Treating as empty.")
                    self._add_to_empty_regions_cache(tile_cache_key)
                    # Create an empty tile using fill_value
                    empty_tile_data = np.full((chunk_height, chunk_width), z_fill_value, dtype=z_dtype)
                    return empty_tile_data[:self.chunk_size, :self.chunk_size]  # Ensure correct output size
                else:
                    error_text = await response.text()
                    print(f"Error fetching chunk {chunk_download_url}: HTTP {response.status} - {error_text}")
                    return None  # Indicate error
        except asyncio.TimeoutError:
            print(f"Timeout fetching chunk: {chunk_download_url}")
            return None
        except aiohttp.ClientError as e:  # More specific aiohttp errors
            print(f"ClientError fetching chunk {chunk_download_url}: {e}")
            return None
        except Exception as e:  # Catch-all for other unexpected errors during fetch
            print(f"Unexpected error fetching chunk {chunk_download_url}: {e}")
            import traceback
            print(traceback.format_exc())
            return None

        if not raw_chunk_bytes:  # Should be caught by 404 or other errors, but as a safeguard
            print(f"No data received for chunk: {chunk_download_url}, though HTTP status was not an error.")
            self._add_to_empty_regions_cache(tile_cache_key)
            empty_tile_data = np.full((chunk_height, chunk_width), z_fill_value, dtype=z_dtype)
            return empty_tile_data[:self.chunk_size, :self.chunk_size]

        # 4. Decompress and decode chunk data
        try:
            if z_compressor_meta is None:  # Raw, uncompressed data
                decompressed_data = raw_chunk_bytes
            else:
                codec = numcodecs.get_codec(z_compressor_meta)  # Handles filters too if defined in compressor object
                decompressed_data = codec.decode(raw_chunk_bytes)

            # Convert to NumPy array and reshape. OME-Zarr chunk shape is [chunk_height, chunk_width]
            chunk_data = np.frombuffer(decompressed_data, dtype=z_dtype).reshape((chunk_height, chunk_width))

            # The Zarr chunk might be smaller than self.chunk_size if it's a partial edge chunk.
            # We need to return a tile of self.chunk_size.
            final_tile_data = np.full((self.chunk_size, self.chunk_size), z_fill_value, dtype=z_dtype)

            # Determine the slice to copy from chunk_data and where to place it in final_tile_data
            copy_height = min(chunk_data.shape[0], self.chunk_size)
            copy_width = min(chunk_data.shape[1], self.chunk_size)

            final_tile_data[:copy_height, :copy_width] = chunk_data[:copy_height, :copy_width]

        except Exception as e:
            print(f"Error decompressing/decoding chunk from {chunk_download_url}: {e}")
            print(f"Metadata: dtype={z_dtype_str}, compressor={z_compressor_meta}, chunk_shape={z_chunks}")
            import traceback
            print(traceback.format_exc())
            return None  # Indicate error

        # 5. Check if tile is effectively empty (e.g., all fill_value or zeros)
        # Use a small threshold for non-zero values if fill_value is 0 or not defined
        is_empty_threshold = 10
        if z_fill_value is not None:
            if np.all(final_tile_data == z_fill_value):
                print(f"Tile data is all fill_value ({z_fill_value}), treating as empty: {tile_cache_key}")
                self._add_to_empty_regions_cache(tile_cache_key)  # Cache as empty
                return None  # Return None for empty tiles based on fill_value
        elif np.count_nonzero(final_tile_data) < is_empty_threshold:
            print(f"Tile data is effectively empty (few non-zeros), treating as empty: {tile_cache_key}")
            self._add_to_empty_regions_cache(tile_cache_key)  # Cache as empty
            return None

        # 6. Cache the processed tile
        self.processed_tile_cache[tile_cache_key] = {
            'data': final_tile_data,
            'timestamp': time.time()
        }

        total_time = time.time() - start_time
        print(f"Total tile processing time for {tile_cache_key}: {total_time:.3f}s, size: {final_tile_data.nbytes/1024:.1f}KB")

        return final_tile_data

    async def _get_channel_index_from_well(self, dataset_id, well_id, channel_name):
        """
        Get channel index from well's OME-Zarr metadata.
        Args:
            dataset_id (str): The alias of the dataset.
            well_id (str): Well ID (e.g., "F5")
            channel_name (str): Channel name to look up
        Returns:
            int or None: Channel index or None if not found
        """
        try:
            artifact_name_only = dataset_id.split('/')[-1]
            well_zip_path = f"well_{well_id}_96.zip"
            zattrs_path_in_well = "data.zarr/.zattrs"
            
            # Construct URL to access .zattrs metadata in the well ZIP
            zattrs_metadata_url = f"{self.server_url}/{self.workspace}/artifacts/{artifact_name_only}/zip-files/{well_zip_path}?path={zattrs_path_in_well}"
            
            # Fetch .zattrs metadata
            http_session = await self._get_http_session()
            async with http_session.get(zattrs_metadata_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    print(f"Failed to get .zattrs metadata from {zattrs_metadata_url}: HTTP {response.status}")
                    return None
                # Read as text and parse JSON manually to avoid MIME type issues
                response_text = await response.text()
                import json
                zattrs_metadata = json.loads(response_text)
            
            # Check squid_canvas channel mapping first (new format)
            if "squid_canvas" in zattrs_metadata and "channel_mapping" in zattrs_metadata["squid_canvas"]:
                channel_mapping = zattrs_metadata["squid_canvas"]["channel_mapping"]
                if channel_name in channel_mapping:
                    return channel_mapping[channel_name]
            
            # Fallback to OMERO channels
            if "omero" in zattrs_metadata and "channels" in zattrs_metadata["omero"]:
                channels = zattrs_metadata["omero"]["channels"]
                for idx, channel_info in enumerate(channels):
                    if channel_info.get("label") == channel_name:
                        return idx
            
            # Fallback mapping for common channel names
            channel_name_map = {
                'BF_LED_matrix_full': 0,
                'BF LED matrix full': 0,
                'Fluorescence_405_nm_Ex': 1,
                'Fluorescence 405 nm Ex': 1,
                'Fluorescence_488_nm_Ex': 2,
                'Fluorescence 488 nm Ex': 2,
                'Fluorescence_561_nm_Ex': 4,
                'Fluorescence 561 nm Ex': 4,
                'Fluorescence_638_nm_Ex': 3,
                'Fluorescence 638 nm Ex': 3,
                'Fluorescence_730_nm_Ex': 5,
                'Fluorescence 730 nm Ex': 5
            }
            
            if channel_name in channel_name_map:
                print(f"Using fallback channel mapping: {channel_name} → {channel_name_map[channel_name]}")
                return channel_name_map[channel_name]
            
            print(f"Channel '{channel_name}' not found in well {well_id}")
            return None
            
        except Exception as e:
            print(f"Error getting channel index for {channel_name} in well {well_id}: {e}")
            return None

    # Legacy methods for backward compatibility - now use chunk-based access
    async def get_zarr_group(self, dataset_id, channel):
        """Legacy method - now returns None as we use direct chunk access instead. Timestamp is ignored."""
        print("Warning: get_zarr_group is deprecated, using direct chunk access instead. Timestamp parameter is ignored.")

    async def prime_metadata(self, dataset_alias, channel_name, scale, use_cache=True):
        """Pre-fetch .zarray metadata for a given dataset, channel, and scale."""
        print(f"Priming metadata for {dataset_alias}/{channel_name}/scale{scale} (use_cache={use_cache})")
        try:
            zarray_path = f"{channel_name}/scale{scale}/.zarray"
            await self._fetch_zarr_metadata(dataset_alias, zarray_path, use_cache=use_cache)

            zgroup_channel_path = f"{channel_name}/.zgroup"
            await self._fetch_zarr_metadata(dataset_alias, zgroup_channel_path, use_cache=use_cache)

            zgroup_root_path = ".zgroup"
            await self._fetch_zarr_metadata(dataset_alias, zgroup_root_path, use_cache=use_cache)
            print(f"Metadata priming complete for {dataset_alias}/{channel_name}/scale{scale}")
            return True
        except Exception as e:
            print(f"Error priming metadata: {e}")
            return False

    async def get_region_np_data(self, dataset_id, channel, scale, x, y, direct_region=None, width=None, height=None):
        """
        Get a region as numpy array using new HTTP chunk access
        
        Args:
            dataset_id (str): The dataset ID (e.g., "agent-lens/scan-time-lapse-...")
            channel (str): Channel name
            scale (int): Scale level
            x (int): X coordinate (chunk coordinates)
            y (int): Y coordinate (chunk coordinates)
            direct_region (tuple, optional): A tuple of (y_start, y_end, x_start, x_end) for direct region extraction.
                If provided, x and y are ignored and this region is used directly.
            width (int, optional): Desired width of the output image. If specified, the output will be resized/padded to this width.
            height (int, optional): Desired height of the output image. If specified, the output will be resized/padded to this height.
            
        Returns:
            np.ndarray: Region data as numpy array
        """
        try:
            # Determine the output dimensions
            output_width = width if width is not None else self.chunk_size
            output_height = height if height is not None else self.chunk_size

            # For direct region access, we need to fetch multiple chunks and stitch them together
            if direct_region is not None:
                y_start, y_end, x_start, x_end = direct_region

                # Get metadata to determine chunk size
                # dataset_id is now the full path like "agent-lens/scan-time-lapse-..."
                zarray_path_in_dataset = f"{channel}/scale{scale}/.zarray"
                zarray_metadata = await self._fetch_zarr_metadata(dataset_id, zarray_path_in_dataset)

                if not zarray_metadata:
                    print("Failed to get .zarray metadata for direct region access")
                    return np.zeros((output_height, output_width), dtype=np.uint8)

                z_chunks = zarray_metadata["chunks"]  # [chunk_height, chunk_width]
                z_dtype = np.dtype(zarray_metadata["dtype"])

                # Calculate which chunks we need
                chunk_y_start = y_start // z_chunks[0]
                chunk_y_end = (y_end - 1) // z_chunks[0] + 1
                chunk_x_start = x_start // z_chunks[1]
                chunk_x_end = (x_end - 1) // z_chunks[1] + 1

                # Create result array
                result_height = y_end - y_start
                result_width = x_end - x_start
                result = np.zeros((result_height, result_width), dtype=z_dtype)

                # Fetch and stitch chunks
                for chunk_y in range(chunk_y_start, chunk_y_end):
                    for chunk_x in range(chunk_x_start, chunk_x_end):
                        chunk_data = await self.get_chunk_np_data(dataset_id, channel, scale, chunk_x, chunk_y)

                        if chunk_data is not None:
                            # Calculate where this chunk fits in the result
                            chunk_y_offset = chunk_y * z_chunks[0]
                            chunk_x_offset = chunk_x * z_chunks[1]

                            # Calculate the slice within the chunk
                            chunk_y_slice_start = max(0, y_start - chunk_y_offset)
                            chunk_y_slice_end = min(z_chunks[0], y_end - chunk_y_offset)
                            chunk_x_slice_start = max(0, x_start - chunk_x_offset)
                            chunk_x_slice_end = min(z_chunks[1], x_end - chunk_x_offset)

                            # Calculate the slice within the result
                            result_y_slice_start = max(0, chunk_y_offset - y_start + chunk_y_slice_start)
                            result_y_slice_end = result_y_slice_start + (chunk_y_slice_end - chunk_y_slice_start)
                            result_x_slice_start = max(0, chunk_x_offset - x_start + chunk_x_slice_start)
                            result_x_slice_end = result_x_slice_start + (chunk_x_slice_end - chunk_x_slice_start)

                            # Copy the data
                            if (chunk_y_slice_end > chunk_y_slice_start and chunk_x_slice_end > chunk_x_slice_start and
                                result_y_slice_end > result_y_slice_start and result_x_slice_end > result_x_slice_start):
                                result[result_y_slice_start:result_y_slice_end, result_x_slice_start:result_x_slice_end] = \
                                    chunk_data[chunk_y_slice_start:chunk_y_slice_end, chunk_x_slice_start:chunk_x_slice_end]

                # Resize to requested dimensions if needed
                if width is not None or height is not None:
                    final_result = np.zeros((output_height, output_width), dtype=result.dtype)
                    copy_height = min(result.shape[0], output_height)
                    copy_width = min(result.shape[1], output_width)
                    final_result[:copy_height, :copy_width] = result[:copy_height, :copy_width]
                    result = final_result

                # Ensure data is in the right format (uint8)
                if result.dtype != np.uint8:
                    if result.dtype == np.float32 or result.dtype == np.float64:
                        # Normalize floating point data
                        if result.max() > 0:
                            result = (result / result.max() * 255).astype(np.uint8)
                        else:
                            result = np.zeros(result.shape, dtype=np.uint8)
                    else:
                        # For other integer types, scale appropriately
                        result = result.astype(np.uint8)

                return result

            else:
                # Single chunk access
                # dataset_id is the full path like "agent-lens/scan-time-lapse-..."
                chunk_data = await self.get_chunk_np_data(dataset_id, channel, scale, x, y)

                if chunk_data is None:
                    return np.zeros((output_height, output_width), dtype=np.uint8)

                # Resize to requested dimensions if needed
                if width is not None or height is not None:
                    result = np.zeros((output_height, output_width), dtype=chunk_data.dtype)
                    copy_height = min(chunk_data.shape[0], output_height)
                    copy_width = min(chunk_data.shape[1], output_width)
                    result[:copy_height, :copy_width] = chunk_data[:copy_height, :copy_width]
                    chunk_data = result

                # Ensure data is in the right format (uint8)
                if chunk_data.dtype != np.uint8:
                    if chunk_data.dtype == np.float32 or chunk_data.dtype == np.float64:
                        # Normalize floating point data
                        if chunk_data.max() > 0:
                            chunk_data = (chunk_data / chunk_data.max() * 255).astype(np.uint8)
                        else:
                            chunk_data = np.zeros(chunk_data.shape, dtype=np.uint8)
                    else:
                        # For other integer types, scale appropriately
                        chunk_data = chunk_data.astype(np.uint8)

                return chunk_data

        except Exception as e:
            print(f"Error getting region data: {e}")
            import traceback
            print(traceback.format_exc())
            return np.zeros((output_height, output_width), dtype=np.uint8)

    async def get_region_bytes(self, dataset_id, channel, scale, x, y):
        """Serve a region as PNG bytes. Timestamp is ignored."""
        try:
            # Get region data as numpy array
            region_data = await self.get_region_np_data(dataset_id, channel, scale, x, y)

            if region_data is None:
                print(f"No numpy data for region {dataset_id}/{channel}/{scale}/{x}/{y}, returning blank image.")
                # Create a blank image
                pil_image = Image.new("L", (self.chunk_size, self.chunk_size), color=0)
            else:
                try:
                    # Ensure data is in a suitable range for image conversion if necessary
                    if region_data.dtype == np.uint16:
                        # Basic windowing for uint16: scale to uint8
                        scaled_data = (region_data / 256).astype(np.uint8)
                        pil_image = Image.fromarray(scaled_data)
                    elif region_data.dtype == np.float32 or region_data.dtype == np.float64:
                        # Handle float data: normalize to 0-255 for PNG
                        min_val, max_val = np.min(region_data), np.max(region_data)
                        if max_val > min_val:
                            normalized_data = ((region_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        else: # Flat data
                            normalized_data = np.zeros_like(region_data, dtype=np.uint8)
                        pil_image = Image.fromarray(normalized_data)
                    else: # Assume uint8 or other directly compatible types
                        pil_image = Image.fromarray(region_data)
                except Exception as e:
                    print(f"Error converting numpy region to PIL Image: {e}. Data type: {region_data.dtype}, shape: {region_data.shape}")
                    pil_image = Image.new("L", (self.chunk_size, self.chunk_size), color=0) # Fallback to blank

            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG") # Default PNG compression
            return buffer.getvalue()
        except Exception as e:
            print(f"Error in get_region_bytes: {str(e)}")
            blank_image = Image.new("L", (self.chunk_size, self.chunk_size), color=0)
            buffer = io.BytesIO()
            blank_image.save(buffer, format="PNG")
            return buffer.getvalue()

    async def get_region_base64(self, dataset_id, channel, scale, x, y):
        """Serve a region as base64 string. Timestamp is ignored."""
        region_bytes = await self.get_region_bytes(dataset_id, channel, scale, x, y)
        return base64.b64encode(region_bytes).decode('utf-8')

    async def test_zarr_access(self, dataset_id=None, channel=None, bypass_cache=False):
        """
        Test function to verify Zarr chunk access is working correctly.
        Attempts to access a known chunk.
        
        Args:
            dataset_id (str, optional): The dataset ID to test. Defaults to a standard test dataset.
            channel (str, optional): The channel to test. Defaults to a standard test channel.
            bypass_cache (bool, optional): If True, bypasses metadata cache for this test. Defaults to False.
            
        Returns:
            dict: A dictionary with status, success flag, and additional info.
        """
        try:
            # Use default values if not provided
            dataset_id = dataset_id or "agent-lens/20250824-example-data-20250824-221822"
            channel = channel or "BF_LED_matrix_full"

            print(f"Testing Zarr chunk access for dataset: {dataset_id}, channel: {channel}, bypass_cache: {bypass_cache}")

            scale = 0 # Typically testing scale0
            print(f"Attempting to prime metadata for dataset: {dataset_id}, channel: {channel}, scale: {scale}")
            # Pass use_cache=!bypass_cache
            metadata_primed = await self.prime_metadata(dataset_id, channel, scale, use_cache=not bypass_cache)

            if not metadata_primed: # prime_metadata now returns True/False
                return {
                    "status": "error",
                    "success": False,
                    "message": "Failed to prime metadata for test chunk."
                }

            return {
                "status": "ok",
                "success": True,
                "message": f"Successfully primed metadata for test chunk (bypass_cache={bypass_cache})."
            }

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"Error in test_zarr_access: {str(e)}")
            print(error_traceback)

            return {
                "status": "error",
                "success": False,
                "message": f"Error accessing Zarr: {str(e)}",
                "error": str(e),
                "traceback": error_traceback
            }
