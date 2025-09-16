"""
Offline processing module for time-lapse experiment data.
Handles stitching and uploading of stored microscopy data.
"""

import asyncio
import io
import json
import logging
import os
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import List

import cv2
import pandas as pd

# Use print statements for debugging since logger configuration seems problematic
logger = logging.getLogger(__name__)


class OfflineProcessor:
    """Handles offline stitching and uploading of time-lapse data."""

    def __init__(self, squid_controller, zarr_artifact_manager=None, service_id=None, 
                 max_concurrent_wells=3, image_batch_size=5):
        print("🔧 OfflineProcessor.__init__ called")
        self.squid_controller = squid_controller
        self.zarr_artifact_manager = zarr_artifact_manager
        self.service_id = service_id
        self.logger = logger
        
        # Performance configuration
        self.max_concurrent_wells = max_concurrent_wells
        self.image_batch_size = image_batch_size
        
        # Ensure configuration is loaded
        from squid_control.control.config import CONFIG
        self._ensure_config_loaded()

    def _ensure_config_loaded(self):
        """Ensure the configuration is properly loaded."""
        from squid_control.control.config import CONFIG, load_config
        import os
        
        # Check if DEFAULT_SAVING_PATH is already loaded
        if not CONFIG.DEFAULT_SAVING_PATH:
            print("Configuration not loaded, attempting to load HCS_v2 config...")
            try:
                # Try to load the config file
                current_dir = Path(__file__).parent
                config_path = current_dir / "config" / "configuration_HCS_v2.ini"
                if config_path.exists():
                    load_config(str(config_path), None)
                    print(f"Configuration loaded: DEFAULT_SAVING_PATH = {CONFIG.DEFAULT_SAVING_PATH}")
                else:
                    print(f"Config file not found at {config_path}")
            except Exception as e:
                print(f"Failed to load configuration: {e}")
        else:
            print(f"Configuration already loaded: DEFAULT_SAVING_PATH = {CONFIG.DEFAULT_SAVING_PATH}")

    def find_experiment_folders(self, experiment_id: str) -> List[Path]:
        """
        Find all experiment folders matching experiment_id prefix.
        
        Args:
            experiment_id: Experiment ID to search for (e.g., 'test-drug')
            
        Returns:
            Sorted list of Path objects like:
            [experiment_id-20250822T143055, experiment_id-20250822T163022, ...]
        """

        from squid_control.control.config import CONFIG
        print(f"CONFIG.DEFAULT_SAVING_PATH = {CONFIG.DEFAULT_SAVING_PATH}")
        base_path = Path(CONFIG.DEFAULT_SAVING_PATH)
        print(f"Searching in base path: {base_path}")
        if not base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")

        pattern = f"{experiment_id}-*"
        print(f"Using pattern: {pattern}")
        folders = sorted(base_path.glob(pattern))
        print(f"Found {len(folders)} folders matching pattern: {[f.name for f in folders]}")

        # Filter to only directories that contain a '0' subfolder
        valid_folders = []
        for folder in folders:
            print(f"Checking folder: {folder.name}")
            if folder.is_dir():
                zero_folder = folder / "0"
                print(f"  Looking for '0' subfolder: {zero_folder}")
                if zero_folder.exists() and zero_folder.is_dir():
                    valid_folders.append(folder)
                    print(f"  ✓ Valid folder: {folder.name}")
                else:
                    print(f"  ✗ Skipping {folder.name}: no '0' subfolder found")
            else:
                print(f"  ✗ Skipping {folder.name}: not a directory")

        print(f"Found {len(valid_folders)} valid experiment folders for '{experiment_id}': {[f.name for f in valid_folders]}")
        return valid_folders

    def parse_acquisition_parameters(self, experiment_folder: Path) -> dict:
        """
        Parse acquisition parameters.json from the experiment folder.
        
        Args:
            experiment_folder: Path to experiment folder (e.g., test-drug-20250822T143055)
            
        Returns:
            Dictionary with acquisition parameters
        """
        # Try the main experiment folder first
        json_file = experiment_folder / "acquisition parameters.json"
        if not json_file.exists():
            # Fallback to the '0' subfolder
            json_file = experiment_folder / "0" / "acquisition parameters.json"
            if not json_file.exists():
                raise FileNotFoundError(f"No acquisition parameters.json found in {experiment_folder}/ or {experiment_folder}/0/")

        with open(json_file) as f:
            params = json.load(f)

        self.logger.info(f"Loaded acquisition parameters from {json_file}: Nx={params.get('Nx')}, Ny={params.get('Ny')}, dx={params.get('dx(mm)')}, dy={params.get('dy(mm)')}")
        return params

    def parse_configurations_xml(self, experiment_folder: Path) -> dict:
        """
        Parse configurations.xml and extract channel settings.
        
        Args:
            experiment_folder: Path to experiment folder
            
        Returns:
            Dictionary with channel configurations
        """
        # Try the main experiment folder first
        xml_file = experiment_folder / "configurations.xml"
        if not xml_file.exists():
            # Fallback to the '0' subfolder
            xml_file = experiment_folder / "0" / "configurations.xml"
            if not xml_file.exists():
                raise FileNotFoundError(f"No configurations.xml found in {experiment_folder}/ or {experiment_folder}/0/")

        return self._parse_xml_channels(xml_file)

    def _parse_xml_channels(self, xml_file: Path) -> dict:
        """Extract channel information from XML configuration file."""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        channels = {}
        for mode in root.findall('mode'):
            if mode.get('Selected') == '1':  # Only selected channels
                channel_name = mode.get('Name')
                channels[channel_name] = {
                    'exposure_time': float(mode.get('ExposureTime', 0)),
                    'intensity': float(mode.get('IlluminationIntensity', 0)),
                    'illumination_source': mode.get('IlluminationSource'),
                    'analog_gain': float(mode.get('AnalogGain', 0)),
                    'mode_id': mode.get('ID')
                }

        self.logger.info(f"Found {len(channels)} selected channels: {list(channels.keys())}")
        return channels

    def parse_coordinates_csv(self, experiment_folder: Path) -> dict:
        """
        Parse coordinates.csv and group by well.
        
        Args:
            experiment_folder: Path to experiment folder
            
        Returns:
            Dictionary grouped by well: {well_id: [coordinate_data]}
        """
        csv_file = experiment_folder / "0" / "coordinates.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"No coordinates.csv found in {experiment_folder}/0/")

        df = pd.read_csv(csv_file)

        # Group by 'region' column (which contains well IDs like 'G9')
        wells_data = {}

        for region in df['region'].unique():
            well_data = df[df['region'] == region].copy()
            # Only process k=0 (single focal plane)
            well_data = well_data[well_data['k'] == 0]

            if len(well_data) > 0:
                wells_data[region] = well_data.to_dict('records')

        self.logger.info(f"Found {len(wells_data)} wells with data: {list(wells_data.keys())}")
        return wells_data

    def create_xml_to_channel_mapping(self, xml_channels: dict) -> dict:
        """
        Create mapping from filename channel names to ChannelMapper human names.
        
        Args:
            xml_channels: Dictionary of channel configurations from XML
            
        Returns:
            Dictionary mapping filename channel names (zarr format) to human names (expected by canvas)
        """

        from squid_control.control.config import ChannelMapper
        
        # Create mapping from zarr names (used in filenames) to human names (expected by canvas)
        filename_to_human_mapping = {}
        
        # Get all channel info and create zarr_name -> human_name mapping
        for channel_info in ChannelMapper.CHANNELS.values():
            filename_to_human_mapping[channel_info.zarr_name] = channel_info.human_name
        
        print(f"Filename to human name mapping: {filename_to_human_mapping}")
        return filename_to_human_mapping

    def create_temp_experiment_manager(self, temp_path: str):
        """
        Create temporary experiment manager for offline stitching.
        
        Args:
            temp_path: Path for temporary zarr storage
            
        Returns:
            ExperimentManager instance
        """

        # Ensure temp directory exists
        Path(temp_path).mkdir(parents=True, exist_ok=True)

        from squid_control.stitching.zarr_canvas import ExperimentManager
        temp_exp_manager = ExperimentManager(
            base_path=temp_path,
            pixel_size_xy_um=self.squid_controller.pixel_size_xy
        )

        return temp_exp_manager

    def _load_and_stitch_well_images_sync(self, well_data: List[dict],
                                         experiment_folder: Path,
                                         canvas, channel_mapping: dict) -> None:
        """
        Load BMP images and add them to well canvas with optimized batch processing.
        
        Args:
            well_data: List of coordinate records for this well
            experiment_folder: Path to experiment folder
            canvas: WellZarrCanvas instance
            channel_mapping: XML to ChannelMapper name mapping
        """
        data_folder = experiment_folder / "0"

        # Get available channels for this canvas
        available_channels = list(canvas.channel_to_zarr_index.keys())
        print(f"Available channels for canvas: {available_channels}")

        # Pre-filter and group images by position for batch processing
        position_images = {}
        for coord_record in well_data:
            i = int(coord_record['i'])
            j = int(coord_record['j'])
            k = int(coord_record['k'])
            x_mm = float(coord_record['x (mm)'])
            y_mm = float(coord_record['y (mm)'])
            well_id = coord_record['region']

            # Skip if not k=0 (single focal plane only)
            if k != 0:
                continue

            # Find all image files for this position
            pattern = f"{well_id}_{i}_{j}_{k}_*.bmp"
            image_files = list(data_folder.glob(pattern))
            
            if image_files:
                position_images[(i, j, x_mm, y_mm)] = image_files
                print(f"Well {well_id} position ({i},{j},{k}): found {len(image_files)} images")
            else:
                print(f"⚠️ Well {well_id} position ({i},{j},{k}): NO IMAGES FOUND!")
                print(f"  🔍 Pattern used: {pattern}")
                print(f"  🔍 Data folder: {data_folder}")

        # Process images sequentially - one by one
        images_added = 0
        total_images = sum(len(files) for files in position_images.values())
        
        for (i, j, x_mm, y_mm), image_files in position_images.items():
            print(f"Processing position ({i},{j}) with {len(image_files)} images...")
            
            # Process each image one by one (no batching, no parallel)
            for img_index, img_file in enumerate(image_files):
                print(f"  Loading image {img_index + 1}/{len(image_files)}: {img_file.name}")
                
                # Load and process single image synchronously
                success = self._load_and_process_single_image_sync(
                    img_file, x_mm, y_mm, canvas, channel_mapping, available_channels
                )
                
                if success:
                    images_added += 1
                else:
                    print(f"  ❌ Failed to add image {img_file.name}")
        
        print(f"✅ Total images added to canvas: {images_added}/{total_images}")
        
    def _load_and_process_single_image_sync(self, img_file: Path, x_mm: float, y_mm: float,
                                           canvas, channel_mapping: dict, available_channels: list) -> bool:
        """
        Load and process a single image sequentially (no async operations).
        
        Returns:
            True if image was successfully added, False otherwise
        """
        try:
            # Extract channel name from filename
            filename_parts = img_file.stem.split('_')
            if len(filename_parts) < 5:
                return False
                
            # Channel name is everything after the position indices (in zarr format)
            channel_name = '_'.join(filename_parts[4:])
            mapped_channel_name = channel_mapping.get(channel_name, channel_name)

            # Check if this channel is available in the canvas
            if mapped_channel_name not in available_channels:
                print(f"    ❌ Channel {mapped_channel_name} not available in canvas, skipping")
                print(f"    🔍 Available channels: {available_channels}")
                print(f"    🔍 Channel mapping: {channel_mapping}")
                return False

            # Load image synchronously (no thread pool)
            image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"    Failed to load image file: {img_file}")
                return False

            # Get zarr channel index
            zarr_channel_idx = canvas.get_zarr_channel_index(mapped_channel_name)

            # Add image to canvas using stitching queue (same pattern as normal_scan_with_stitching)
            import asyncio
            import concurrent.futures
            
            # Get the current event loop from the canvas's context
            try:
                # Try to run in current thread's event loop if available
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we need to use run_coroutine_threadsafe
                    future = asyncio.run_coroutine_threadsafe(
                        self._add_image_to_stitching_queue(
                            canvas, image, x_mm, y_mm, zarr_channel_idx, 0, 0
                        ), loop
                    )
                    future.result(timeout=30)  # Wait for completion with timeout
                else:
                    # If no running loop, run directly
                    asyncio.run(self._add_image_to_stitching_queue(
                        canvas, image, x_mm, y_mm, zarr_channel_idx, 0, 0
                    ))
            except RuntimeError:
                # No event loop in this thread, create one
                asyncio.run(self._add_image_to_stitching_queue(
                    canvas, image, x_mm, y_mm, zarr_channel_idx, 0, 0
                ))

            return True
            
        except Exception as e:
            print(f"    ❌ Failed to process image {img_file}: {e}")
            return False

    async def _load_and_process_single_image(self, img_file: Path, x_mm: float, y_mm: float,
                                           canvas, channel_mapping: dict, available_channels: list) -> bool:
        """
        Load and process a single image asynchronously.
        
        Returns:
            True if image was successfully added, False otherwise
        """
        try:
            # Extract channel name from filename
            filename_parts = img_file.stem.split('_')
            if len(filename_parts) < 5:
                return False
                
            # Channel name is everything after the position indices (in zarr format)
            channel_name = '_'.join(filename_parts[4:])
            mapped_channel_name = channel_mapping.get(channel_name, channel_name)

            # Check if this channel is available in the canvas
            if mapped_channel_name not in available_channels:
                return False

            # Load image in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(
                None, cv2.imread, str(img_file), cv2.IMREAD_GRAYSCALE
            )
            
            if image is None:
                return False

            # Get zarr channel index
            zarr_channel_idx = canvas.get_zarr_channel_index(mapped_channel_name)

            # Add image to canvas using stitching queue (same pattern as normal_scan_with_stitching)
            await self._add_image_to_stitching_queue(canvas, image, x_mm, y_mm, zarr_channel_idx)

            return True
            
        except Exception as e:
            print(f"✗ Failed to process image {img_file}: {e}")
            return False

    def extract_timestamp_from_folder(self, folder_path: Path) -> str:
        """
        Extract timestamp identifier from folder name.
        
        Handles both old and new timestamp formats:
        - Old format: 20250718-U2OS-FUCCI-Eto-ER-20250720T095000_2025-07-20_09-51-13.673205
        - New format: 20250718-U2OS-FUCCI-Eto-ER_2025-07-20_09-51-13.673205
        
        Args:
            folder_path: Path to experiment folder
            
        Returns:
            Normalized timestamp string (e.g., '2025-07-20_09-51-13')
        """
        import re
        
        folder_name = folder_path.name
        
        # Pattern to match the normalized timestamp format: _YYYY-MM-DD_HH-MM-SS
        # This pattern will match both old and new formats
        timestamp_pattern = r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:\.\d+)?)'
        
        match = re.search(timestamp_pattern, folder_name)
        if match:
            # Extract the timestamp part (without the leading underscore)
            timestamp = match.group(1)
            # Remove microseconds if present for consistency
            if '.' in timestamp:
                timestamp = timestamp.split('.')[0]
            return timestamp
        else:
            # Fallback: try to find any timestamp-like pattern
            # Look for patterns like YYYYMMDDTHHMMSS or YYYY-MM-DD_HH-MM-SS
            fallback_patterns = [
                r'(\d{8}T\d{6})',  # YYYYMMDDTHHMMSS
                r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})',  # YYYY-MM-DD_HH-MM-SS
                r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})',  # YYYY-MM-DD_HH:MM:SS
            ]
            
            for pattern in fallback_patterns:
                match = re.search(pattern, folder_name)
                if match:
                    return match.group(1)
            
            # Final fallback: use folder name
            return folder_name

    def create_normalized_dataset_name(self, experiment_folder: Path, experiment_id: str = None) -> str:
        """
        Create a normalized dataset name using the extracted timestamp.
        
        This helps group related experiments together by using a consistent
        timestamp format regardless of the original folder naming.
        
        Args:
            experiment_folder: Path to experiment folder
            experiment_id: Optional experiment ID prefix
            
        Returns:
            Normalized dataset name (e.g., 'experiment-20250720-095113')
        """
        # Extract the normalized timestamp
        timestamp = self.extract_timestamp_from_folder(experiment_folder)
        
        # Get folder name for reference
        folder_name = experiment_folder.name
        
        # Use experiment_id as prefix if provided, otherwise use folder name base
        if experiment_id:
            # Extract base name from experiment_id (before any timestamps)
            base_name = experiment_id.split('_')[0] if '_' in experiment_id else experiment_id
        else:
            # Extract base name from folder (everything before the first timestamp pattern)
            # Remove timestamp patterns to get base name
            import re
            base_name = re.sub(r'_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:\.\d+)?.*$', '', folder_name)
            base_name = re.sub(r'-\d{8}T\d{6}.*$', '', base_name)
        
        # Convert timestamp to old format (YYYYMMDD-HHMMSS) if we have a valid timestamp
        if timestamp and timestamp != folder_name:
            # Check if timestamp is in the new format (YYYY-MM-DD_HH-MM-SS)
            if '_' in timestamp and '-' in timestamp:
                # Convert from YYYY-MM-DD_HH-MM-SS to YYYYMMDD-HHMMSS
                date_part, time_part = timestamp.split('_')
                date_compact = date_part.replace('-', '')
                time_compact = time_part.replace('-', '')
                old_format_timestamp = f"{date_compact}-{time_compact}"
            else:
                # Already in old format or some other format, use as-is
                old_format_timestamp = timestamp
            normalized_name = f"{base_name}-{old_format_timestamp}"
        else:
            normalized_name = base_name
        
        # Sanitize the final name
        return self.sanitize_dataset_name(normalized_name)

    def sanitize_dataset_name(self, name: str) -> str:
        """
        Sanitize dataset name to meet artifact manager requirements.
        
        Requirements: lowercase letters, numbers, hyphens, and colons only.
        Must start and end with alphanumeric character.
        
        Args:
            name: Original dataset name
            
        Returns:
            Sanitized dataset name
        """
        import re
        
        # Convert to lowercase
        sanitized = name.lower()
        
        # Replace invalid characters with hyphens
        # Keep only: lowercase letters, numbers, hyphens, colons
        sanitized = re.sub(r'[^a-z0-9\-:]', '-', sanitized)
        
        # Remove multiple consecutive hyphens
        sanitized = re.sub(r'-+', '-', sanitized)
        
        # Remove leading/trailing hyphens and colons
        sanitized = sanitized.strip('-:')
        
        # Ensure it starts and ends with alphanumeric
        if not sanitized[0].isalnum():
            sanitized = 'run-' + sanitized
        if not sanitized[-1].isalnum():
            sanitized = sanitized + '-1'
        
        # Ensure minimum length
        if len(sanitized) < 3:
            sanitized = f"run-{sanitized}-{int(time.time())}"
        
        return sanitized


    async def _process_single_well(self, well_id: str, well_row: str, well_column: int,
                                 well_data: List[dict], experiment_folder: Path,
                                 temp_exp_manager, channel_mapping: dict) -> dict:
        """
        Process a single well with optimized stitching.
        
        Returns:
            Dictionary with well zarr file info, or None if failed
        """
        try:
            self.logger.info(f"Processing well {well_id} with {len(well_data)} positions")

            # Create well canvas using the standard approach
            canvas = temp_exp_manager.get_well_canvas(
                well_row, well_column, '96', 100.0  # Very large padding for absolute coordinates
            )

            # Start stitching
            await canvas.start_stitching()

            try:
                # Load and stitch images for this well - use to_thread for blocking operations
                await asyncio.to_thread(
                    self._load_and_stitch_well_images_sync,
                    well_data, experiment_folder, canvas, channel_mapping
                )

                # Wait for stitching to complete properly
                await self._wait_for_stitching_completion(canvas)

                # CRITICAL: Check which channels have data and activate them
                logger.info(f"Running post-stitching channel activation check for well {well_row}{well_column}")
                canvas.activate_channels_with_data()

                # Export as zip file to disk with proper naming - use to_thread
                well_zip_filename = f"well_{well_row}{well_column}_96.zip"
                well_zip_path = await asyncio.to_thread(self._export_well_to_zip_direct, canvas, well_zip_filename)
            finally:
                await canvas.stop_stitching()

            # Get file size
            import os
            file_size_bytes = os.path.getsize(well_zip_path)
                
            well_info = {
                'name': f"well_{well_row}{well_column}_96",
                'file_path': well_zip_path,
                'size_mb': file_size_bytes / (1024 * 1024)
            }

            self.logger.info(f"Exported well {well_id} as {file_size_bytes/(1024*1024):.2f} MB zip to {well_zip_path}")
            return well_info

        except Exception as e:
            self.logger.error(f"Error processing well {well_id}: {e}")
            # Clean up any partial Zarr files that might cause issues
            try:
                if hasattr(canvas, 'zarr_path') and canvas.zarr_path.exists():
                    import glob
                    import os
                    partial_files = glob.glob(str(canvas.zarr_path / "**" / "*.partial"), recursive=True)
                    for partial_file in partial_files:
                        try:
                            os.remove(partial_file)
                        except:
                            pass
            except:
                pass
            return None

    async def stitch_and_upload_timelapse(self, experiment_id: str,
                                        upload_immediately: bool = True,
                                        cleanup_temp_files: bool = True,
                                        max_concurrent_runs: int = 1,
                                        use_parallel_wells: bool = True) -> dict:
        """
        Parallel stitching and uploading - one folder at a time, 3 wells at a time.
        
        Args:
            experiment_id: Experiment ID to search for
            upload_immediately: Whether to upload each well after stitching
            cleanup_temp_files: Whether to delete temporary files after upload
            max_concurrent_runs: Not used (kept for compatibility)
            use_parallel_wells: Whether to process wells in parallel (3 at a time)
            
        Returns:
            Dictionary with processing results
        """
        results = {
            "success": True,
            "experiment_id": experiment_id,
            "processed_runs": [],
            "failed_runs": [],
            "total_datasets": 0,
            "total_size_mb": 0,
            "start_time": time.time(),
            "processing_mode": "parallel_wells" if use_parallel_wells else "sequential_wells"
        }

        try:
            print("=" * 60)
            if use_parallel_wells:
                print(f"PARALLEL PROCESSING STARTED for experiment_id: {experiment_id}")
                print(f"Mode: One folder at a time, 3 wells at a time")
            else:
                print(f"SEQUENTIAL PROCESSING STARTED for experiment_id: {experiment_id}")
                print(f"Mode: One folder at a time, one well at a time")
            print("=" * 60)
            
            # Find all experiment folders
            print("Searching for experiment folders...")
            experiment_folders = self.find_experiment_folders(experiment_id)

            if not experiment_folders:
                results["success"] = False
                results["message"] = f"No experiment folders found for ID: {experiment_id}"
                results["processing_time_seconds"] = time.time() - results["start_time"]
                self.logger.warning(f"No experiment folders found for ID: {experiment_id}")
                return results

            # Process each experiment folder (each folder = one dataset)
            for folder_index, exp_folder in enumerate(experiment_folders):
                print(f"\n📁 Processing folder {folder_index + 1}/{len(experiment_folders)}: {exp_folder.name}")
                
                # Choose processing method based on use_parallel_wells flag
                if use_parallel_wells:
                    run_result = await self.process_experiment_run_parallel(
                        exp_folder, upload_immediately, cleanup_temp_files, experiment_id
                    )
                else:
                    run_result = await self.process_experiment_run_sequential(
                        exp_folder, upload_immediately, cleanup_temp_files, experiment_id
                    )
                
                if run_result["success"]:
                    results["processed_runs"].append(run_result)
                    results["total_datasets"] += 1  # Each folder = one dataset
                    results["total_size_mb"] += run_result.get("total_size_mb", 0)
                    print(f"✅ Folder {exp_folder.name} completed successfully")
                    print(f"   Dataset: {run_result.get('dataset_name', 'Unknown')}")
                    print(f"   Wells: {run_result.get('wells_processed', 0)}")
                else:
                    results["failed_runs"].append(run_result)
                    print(f"❌ Folder {exp_folder.name} failed: {run_result.get('error', 'Unknown error')}")
                    
                    # Stop processing if upload failed and upload_immediately is True
                    if upload_immediately and "upload" in run_result.get('error', '').lower():
                        print(f"🛑 Stopping processing due to upload failure in folder {exp_folder.name}")
                        results["success"] = False
                        results["message"] = f"Processing stopped due to upload failure in folder {exp_folder.name}"
                        break

            results["processing_time_seconds"] = time.time() - results["start_time"]

            processing_mode = "Parallel" if use_parallel_wells else "Sequential"
            self.logger.info(f"{processing_mode} processing completed: {results['total_datasets']} datasets created, "
                           f"{len(results['failed_runs'])} folder failures, "
                           f"{results['total_size_mb']:.2f} MB total")

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            self.logger.error(f"Processing failed: {e}")

        return results
    
    def _cleanup_existing_temp_folders(self, experiment_folder_name: str):
        """
        Clean up any existing temporary offline_stitch folders for this experiment.
        
        Args:
            experiment_folder_name: Name of the experiment folder to clean up temp folders for
        """
        try:
            from squid_control.control.config import CONFIG
            
            if CONFIG.DEFAULT_SAVING_PATH and Path(CONFIG.DEFAULT_SAVING_PATH).exists():
                base_temp_path = Path(CONFIG.DEFAULT_SAVING_PATH)
                
                # Find all offline_stitch folders for this experiment
                pattern = f"offline_stitch_{experiment_folder_name}_*"
                temp_folders = list(base_temp_path.glob(pattern))
                
                if temp_folders:
                    print(f"🧹 Found {len(temp_folders)} temporary folders to clean up:")
                    for temp_folder in temp_folders:
                        try:
                            if temp_folder.is_dir():
                                shutil.rmtree(temp_folder, ignore_errors=True)
                                print(f"  🗑️ Cleaned up: {temp_folder.name}")
                        except Exception as e:
                            print(f"  ⚠️ Failed to cleanup {temp_folder.name}: {e}")
                else:
                    print(f"  ✅ No temporary folders found for {experiment_folder_name}")
            else:
                print(f"  ⚠️ Cannot cleanup temp folders: CONFIG.DEFAULT_SAVING_PATH not available")
                
        except Exception as e:
            print(f"  ⚠️ Error during temp folder cleanup: {e}")

    async def process_experiment_run_parallel(self, experiment_folder: Path,
                                            upload_immediately: bool = True,
                                            cleanup_temp_files: bool = True,
                                            experiment_id: str = None) -> dict:
        """
        Process a single experiment run - all wells in parallel (3 at a time).
        
        Args:
            experiment_folder: Path to experiment folder
            upload_immediately: Whether to upload the dataset after processing
            cleanup_temp_files: Whether to cleanup temp files
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing experiment folder: {experiment_folder.name}")

        try:
            # 0. Check for .done file in well_zips directory - skip processing if found
            from squid_control.control.config import CONFIG
            well_zips_path = Path(CONFIG.DEFAULT_SAVING_PATH) / "well_zips"
            done_file = well_zips_path / ".done"
            
            if done_file.exists():
                print(f"🎯 Found .done file at {done_file} - SKIPPING PROCESSING, going directly to upload!")
                return await self._upload_existing_wells_from_directory(
                    well_zips_path, experiment_folder, experiment_id, upload_immediately, cleanup_temp_files
                )
            
            # 1. Parse metadata from this folder
            print(f"📋 Reading metadata from {experiment_folder.name}...")
            acquisition_params = self.parse_acquisition_parameters(experiment_folder)
            xml_channels = self.parse_configurations_xml(experiment_folder)
            channel_mapping = self.create_xml_to_channel_mapping(xml_channels)
            coordinates_data = self.parse_coordinates_csv(experiment_folder)
            
            print(f"Found {len(coordinates_data)} wells to process: {list(coordinates_data.keys())}")

            # 2. Create temporary experiment for this run
            from squid_control.control.config import CONFIG
            
            if CONFIG.DEFAULT_SAVING_PATH and Path(CONFIG.DEFAULT_SAVING_PATH).exists():
                base_temp_path = Path(CONFIG.DEFAULT_SAVING_PATH)
                temp_path = base_temp_path / f"offline_stitch_{experiment_folder.name}_{int(time.time())}"
                temp_path.mkdir(parents=True, exist_ok=True)
                print(f"Using configured saving path for temporary stitching: {temp_path}")
            else:
                temp_path = Path(tempfile.mkdtemp(prefix=f"offline_stitch_{experiment_folder.name}_"))
                print(f"Using system temp directory for stitching: {temp_path}")
                
            temp_exp_manager = self.create_temp_experiment_manager(str(temp_path))

            # 3. Process all wells in parallel (3 at a time)
            print(f"🚀 Starting parallel processing of {len(coordinates_data)} wells (max 3 concurrent)...")
            
            # Create semaphore to limit concurrent well processing to 3
            semaphore = asyncio.Semaphore(self.max_concurrent_wells)
            
            # Create tasks for all wells
            well_tasks = []
            for well_index, (well_id, well_data) in enumerate(coordinates_data.items()):
                # Extract well row and column
                if len(well_id) >= 2:
                    well_row = well_id[0]
                    well_column = int(well_id[1:]) if well_id[1:].isdigit() else 1
                    
                    # Create task for this well
                    task = self._process_well_with_semaphore(
                        semaphore, well_id, well_row, well_column, well_data,
                        experiment_folder, temp_exp_manager, channel_mapping,
                        well_index + 1, len(coordinates_data)
                    )
                    well_tasks.append(task)
                else:
                    print(f"⚠️ Invalid well ID format: {well_id}, skipping...")
            
            # Wait for all wells to complete
            print(f"⏳ Waiting for {len(well_tasks)} wells to complete...")
            well_results = await asyncio.gather(*well_tasks, return_exceptions=True)
            
            # Process results
            wells_processed = 0
            total_size_mb = 0.0
            well_zip_files = []  # Store all well ZIP files for combined upload
            
            for i, result in enumerate(well_results):
                if isinstance(result, Exception):
                    well_id = list(coordinates_data.keys())[i]
                    print(f"  ❌ Well {well_id} failed with exception: {result}")
                    continue
                
                if result is None:
                    well_id = list(coordinates_data.keys())[i]
                    print(f"  ❌ Well {well_id} returned None")
                    continue
                
                wells_processed += 1
                total_size_mb += result['size_mb']
                well_zip_files.append(result)
                print(f"  ✅ Well {result['name']} completed: {result['size_mb']:.2f} MB")
            
            print(f"🎉 Parallel processing complete: {wells_processed}/{len(coordinates_data)} wells processed successfully")
            
            # Create .done file to mark processing completion
            if wells_processed > 0:
                well_zips_path = Path(CONFIG.DEFAULT_SAVING_PATH) / "well_zips"
                done_file = well_zips_path / ".done"
                try:
                    done_file.touch()
                    print(f"✅ Created .done file at {done_file} to mark processing completion")
                except Exception as e:
                    print(f"⚠️ Failed to create .done file: {e}")
            
            # Create dataset name using normalized timestamp extraction
            # This ensures consistent naming regardless of folder timestamp format
            dataset_name = self.create_normalized_dataset_name(experiment_folder, experiment_id)
            print(f"📝 Using dataset name: {dataset_name}")

            # 4. Upload all wells to a single dataset (like upload_zarr_dataset does)
            upload_result = None
            if well_zip_files and upload_immediately and self.zarr_artifact_manager:
                print(f"\n📦 Uploading {len(well_zip_files)} wells to single dataset...")
                
                try:
                    # Prepare zarr_files_info for upload_multiple_zip_files_to_dataset
                    # Use file_path instead of content to prevent memory exhaustion
                    zarr_files_info = []
                    
                    # Add all well ZIP files with file paths (streaming upload)
                    for well_info in well_zip_files:
                        zarr_files_info.append({
                            'name': well_info['name'],  # e.g., "well_A1_96"
                            'file_path': well_info['file_path'],  # Use file path instead of content
                            'size_mb': well_info['size_mb']
                        })
                    
                    # Use the original experiment_id for gallery creation, dataset_name for dataset naming
                    # This ensures all datasets from the same experiment go into the same gallery
                    gallery_experiment_id = experiment_id if experiment_id else dataset_name
                    
                    # Upload all wells to a single dataset
                    upload_result = await self.zarr_artifact_manager.upload_multiple_zip_files_to_dataset(
                        microscope_service_id=self.service_id,
                        experiment_id=gallery_experiment_id,
                        zarr_files_info=zarr_files_info,
                        dataset_name=dataset_name,
                        acquisition_settings={
                            "microscope_service_id": self.service_id,
                            "experiment_name": experiment_folder.name,
                            "total_wells": len(well_zip_files),
                            "total_size_mb": total_size_mb,
                            "offline_processing": True
                        },
                        description=f"Offline processed experiment: {experiment_folder.name} with {len(well_zip_files)} wells"
                    )
                    
                    print(f"  ✅ Dataset upload complete: {upload_result.get('dataset_name')}")
                    
                    # Clean up individual well ZIP files
                    if cleanup_temp_files:
                        for well_info in well_zip_files:
                            try:
                                import os
                                os.unlink(well_info['file_path'])
                                print(f"    🗑️ Cleaned up {well_info['name']}.zip")
                            except Exception as e:
                                print(f"    ⚠️ Failed to cleanup {well_info['name']}.zip: {e}")
                        
                        # Also remove the .done file after successful upload
                        try:
                            well_zips_path = Path(CONFIG.DEFAULT_SAVING_PATH) / "well_zips"
                            done_file = well_zips_path / ".done"
                            done_file.unlink()
                            print(f"    🗑️ Cleaned up .done file")
                        except Exception as e:
                            print(f"    ⚠️ Failed to cleanup .done file: {e}")
                    
                    # Clean up any existing temporary offline_stitch folders after successful upload
                    self._cleanup_existing_temp_folders(experiment_folder.name)
                    
                except Exception as upload_error:
                    print(f"  ❌ Dataset upload failed: {upload_error}")
                    upload_result = None

            # 5. Cleanup temporary files
            if cleanup_temp_files:
                shutil.rmtree(temp_path, ignore_errors=True)
                self.logger.debug(f"Cleaned up temporary files: {temp_path}")

            return {
                "success": True,
                "experiment_folder": experiment_folder.name,
                "wells_processed": wells_processed,
                "total_size_mb": total_size_mb,
                "dataset_name": dataset_name,
                "upload_result": upload_result
            }

        except Exception as e:
            self.logger.error(f"Error in processing {experiment_folder.name}: {e}")
            return {
                "success": False,
                "experiment_folder": experiment_folder.name,
                "error": str(e)
            }

    async def _process_well_with_semaphore(self, semaphore: asyncio.Semaphore, well_id: str, 
                                         well_row: str, well_column: int, well_data: List[dict],
                                         experiment_folder: Path, temp_exp_manager, 
                                         channel_mapping: dict, well_index: int, total_wells: int) -> dict:
        """
        Process a single well with semaphore-controlled concurrency.
        
        Args:
            semaphore: Asyncio semaphore to limit concurrent processing
            well_id: Well identifier (e.g., 'A1')
            well_row: Well row letter (e.g., 'A')
            well_column: Well column number (e.g., 1)
            well_data: List of coordinate records for this well
            experiment_folder: Path to experiment folder
            temp_exp_manager: Temporary experiment manager
            channel_mapping: Channel mapping dictionary
            well_index: Current well index (1-based)
            total_wells: Total number of wells
            
        Returns:
            Dictionary with well zarr file info, or None if failed
        """
        async with semaphore:  # Acquire semaphore (limits to 3 concurrent wells)
            print(f"🧪 Processing well {well_index}/{total_wells}: {well_id} (acquired semaphore)")
            
            try:
                # Process the well (stitch images)
                print(f"  📸 Stitching {len(well_data)} positions for well {well_id}...")
                well_info = await self._process_single_well(
                    well_id, well_row, well_column, well_data,
                    experiment_folder, temp_exp_manager, channel_mapping
                )
                
                if well_info is None:
                    print(f"  ❌ Failed to process well {well_id}")
                    return None
                
                print(f"  ✅ Stitching complete for well {well_id}: {well_info['size_mb']:.2f} MB")
                return well_info
                
            except Exception as e:
                print(f"  ❌ Exception processing well {well_id}: {e}")
                return None
            finally:
                print(f"  🔓 Released semaphore for well {well_id}")

    async def process_experiment_run_sequential(self, experiment_folder: Path,
                                              upload_immediately: bool = True,
                                              cleanup_temp_files: bool = True,
                                              experiment_id: str = None) -> dict:
        """
        Process a single experiment run - all wells in one dataset (sequential mode).
        
        This method is kept for backward compatibility and testing.
        
        Args:
            experiment_folder: Path to experiment folder
            upload_immediately: Whether to upload the dataset after processing
            cleanup_temp_files: Whether to cleanup temp files
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing experiment folder (sequential mode): {experiment_folder.name}")

        try:
            # 0. Check for .done file in well_zips directory - skip processing if found
            from squid_control.control.config import CONFIG
            well_zips_path = Path(CONFIG.DEFAULT_SAVING_PATH) / "well_zips"
            done_file = well_zips_path / ".done"
            
            if done_file.exists():
                print(f"🎯 Found .done file at {done_file} - SKIPPING PROCESSING, going directly to upload!")
                return await self._upload_existing_wells_from_directory(
                    well_zips_path, experiment_folder, experiment_id, upload_immediately, cleanup_temp_files
                )
            
            # 1. Parse metadata from this folder
            print(f"📋 Reading metadata from {experiment_folder.name}...")
            acquisition_params = self.parse_acquisition_parameters(experiment_folder)
            xml_channels = self.parse_configurations_xml(experiment_folder)
            channel_mapping = self.create_xml_to_channel_mapping(xml_channels)
            coordinates_data = self.parse_coordinates_csv(experiment_folder)
            
            print(f"Found {len(coordinates_data)} wells to process: {list(coordinates_data.keys())}")

            # 2. Create temporary experiment for this run
            from squid_control.control.config import CONFIG
            
            if CONFIG.DEFAULT_SAVING_PATH and Path(CONFIG.DEFAULT_SAVING_PATH).exists():
                base_temp_path = Path(CONFIG.DEFAULT_SAVING_PATH)
                temp_path = base_temp_path / f"offline_stitch_{experiment_folder.name}_{int(time.time())}"
                temp_path.mkdir(parents=True, exist_ok=True)
                print(f"Using configured saving path for temporary stitching: {temp_path}")
            else:
                temp_path = Path(tempfile.mkdtemp(prefix=f"offline_stitch_{experiment_folder.name}_"))
                print(f"Using system temp directory for stitching: {temp_path}")
                
            temp_exp_manager = self.create_temp_experiment_manager(str(temp_path))

            # 3. Process all wells in this folder sequentially
            wells_processed = 0
            total_size_mb = 0.0
            well_zip_files = []  # Store all well ZIP files for combined upload
            
            # Create dataset name using normalized timestamp extraction
            # This ensures consistent naming regardless of folder timestamp format
            dataset_name = self.create_normalized_dataset_name(experiment_folder, experiment_id)
            print(f"📝 Using dataset name: {dataset_name}")
            
            for well_index, (well_id, well_data) in enumerate(coordinates_data.items()):
                print(f"\n🧪 Processing well {well_index + 1}/{len(coordinates_data)}: {well_id}")
                
                # Extract well row and column
                if len(well_id) >= 2:
                    well_row = well_id[0]
                    well_column = int(well_id[1:]) if well_id[1:].isdigit() else 1
                else:
                    print(f"⚠️ Invalid well ID format: {well_id}, skipping...")
                    continue
                
                # Step 1: Process the well (stitch images)
                print(f"  📸 Step 1: Stitching {len(well_data)} positions for well {well_id}...")
                well_info = await self._process_single_well(
                    well_id, well_row, well_column, well_data,
                    experiment_folder, temp_exp_manager, channel_mapping
                )
                
                if well_info is None:
                    print(f"  ❌ Failed to process well {well_id}")
                    continue
                
                print(f"  ✅ Stitching complete for well {well_id}: {well_info['size_mb']:.2f} MB")
                wells_processed += 1
                total_size_mb += well_info['size_mb']
                well_zip_files.append(well_info)
                print(f"  ✅ Well {well_id} completed successfully")

            # Create .done file to mark processing completion
            if wells_processed > 0:
                well_zips_path = Path(CONFIG.DEFAULT_SAVING_PATH) / "well_zips"
                done_file = well_zips_path / ".done"
                try:
                    done_file.touch()
                    print(f"✅ Created .done file at {done_file} to mark processing completion")
                except Exception as e:
                    print(f"⚠️ Failed to create .done file: {e}")

            # 4. Upload all wells to a single dataset (like upload_zarr_dataset does)
            upload_result = None
            if well_zip_files and upload_immediately and self.zarr_artifact_manager:
                print(f"\n📦 Uploading {len(well_zip_files)} wells to single dataset...")
                
                try:
                    # Prepare zarr_files_info for upload_multiple_zip_files_to_dataset
                    # Use file_path instead of content to prevent memory exhaustion
                    zarr_files_info = []
                    
                    # Add all well ZIP files with file paths (streaming upload)
                    for well_info in well_zip_files:
                        zarr_files_info.append({
                            'name': well_info['name'],  # e.g., "well_A1_96"
                            'file_path': well_info['file_path'],  # Use file path instead of content
                            'size_mb': well_info['size_mb']
                        })
                    
                    # Use the original experiment_id for gallery creation, dataset_name for dataset naming
                    # This ensures all datasets from the same experiment go into the same gallery
                    gallery_experiment_id = experiment_id if experiment_id else dataset_name
                    
                    # Upload all wells to a single dataset
                    upload_result = await self.zarr_artifact_manager.upload_multiple_zip_files_to_dataset(
                        microscope_service_id=self.service_id,
                        experiment_id=gallery_experiment_id,
                        zarr_files_info=zarr_files_info,
                        dataset_name=dataset_name,
                        acquisition_settings={
                            "microscope_service_id": self.service_id,
                            "experiment_name": experiment_folder.name,
                            "total_wells": len(well_zip_files),
                            "total_size_mb": total_size_mb,
                            "offline_processing": True
                        },
                        description=f"Offline processed experiment: {experiment_folder.name} with {len(well_zip_files)} wells"
                    )
                    
                    print(f"  ✅ Dataset upload complete: {upload_result.get('dataset_name')}")
                    
                    # Clean up individual well ZIP files
                    if cleanup_temp_files:
                        for well_info in well_zip_files:
                            try:
                                import os
                                os.unlink(well_info['file_path'])
                                print(f"    🗑️ Cleaned up {well_info['name']}.zip")
                            except Exception as e:
                                print(f"    ⚠️ Failed to cleanup {well_info['name']}.zip: {e}")
                        
                        # Also remove the .done file after successful upload
                        try:
                            well_zips_path = Path(CONFIG.DEFAULT_SAVING_PATH) / "well_zips"
                            done_file = well_zips_path / ".done"
                            done_file.unlink()
                            print(f"    🗑️ Cleaned up .done file")
                        except Exception as e:
                            print(f"    ⚠️ Failed to cleanup .done file: {e}")
                    
                    # Clean up any existing temporary offline_stitch folders after successful upload
                    self._cleanup_existing_temp_folders(experiment_folder.name)
                    
                except Exception as upload_error:
                    print(f"  ❌ Dataset upload failed: {upload_error}")
                    upload_result = None

            # 5. Cleanup temporary files
            if cleanup_temp_files:
                shutil.rmtree(temp_path, ignore_errors=True)
                self.logger.debug(f"Cleaned up temporary files: {temp_path}")

            return {
                "success": True,
                "experiment_folder": experiment_folder.name,
                "wells_processed": wells_processed,
                "total_size_mb": total_size_mb,
                "dataset_name": dataset_name,
                "upload_result": upload_result
            }

        except Exception as e:
            self.logger.error(f"Error in processing {experiment_folder.name}: {e}")
            return {
                "success": False,
                "experiment_folder": experiment_folder.name,
                "error": str(e)
            }


    async def _wait_for_stitching_completion(self, canvas, timeout_seconds=60):
        """
        Wait for stitching to complete properly with timeout and progress monitoring.
        
        Args:
            canvas: WellZarrCanvas instance
            timeout_seconds: Maximum time to wait for stitching completion
        """
        start_time = time.time()
        last_queue_size = -1
        empty_queue_count = 0  # Count consecutive empty queue checks
        
        while time.time() - start_time < timeout_seconds:
            # Check if stitching is still active
            if not canvas.is_stitching:
                break
                
            # Check queue size
            current_queue_size = canvas.stitch_queue.qsize()
            
            # Log progress if queue size changed
            if current_queue_size != last_queue_size:
                if current_queue_size > 0:
                    print(f"Stitching queue has {current_queue_size} images remaining...")
                    empty_queue_count = 0  # Reset counter when queue has items
                last_queue_size = current_queue_size
            
            # If queue is empty, wait longer to ensure all zarr writes complete
            if current_queue_size == 0:
                empty_queue_count += 1
                print(f"Queue empty, waiting for zarr writes to complete... (check {empty_queue_count})")
                
                # Wait longer for zarr writes to complete
                await asyncio.sleep(2.0)  # Increased from 0.5 to 2.0 seconds
                
                # Only exit after 3 consecutive empty checks (6 seconds total)
                if empty_queue_count >= 3:
                    print("Queue empty for 3 consecutive checks, stitching should be complete")
                    break
            else:
                empty_queue_count = 0  # Reset counter when queue has items
            
            # Wait before checking again
            await asyncio.sleep(0.5)  # Increased from 0.1 to 0.5 seconds
        
        # Final check
        final_queue_size = canvas.stitch_queue.qsize()
        if final_queue_size > 0:
            print(f"Warning: {final_queue_size} images still in stitching queue after timeout")
        else:
            print("Stitching queue is completely empty")
        
        # Additional wait for zarr writes to complete
        print("Waiting additional 3 seconds for zarr writes to complete...")
        await asyncio.sleep(3.0)
        
        elapsed_time = time.time() - start_time
        print(f"Stitching completion wait took {elapsed_time:.2f} seconds")

    async def _add_image_to_stitching_queue(self, canvas, image, x_mm, y_mm, zarr_channel_idx, z_idx=0, timepoint=0):
        """
        Add image to stitching queue using the exact same pattern as normal_scan_with_stitching.
        This ensures proper processing with all scales and coordinate conversion.
        
        Args:
            canvas: WellZarrCanvas instance
            image: Image array
            x_mm, y_mm: Absolute coordinates (like normal_scan_with_stitching)
            zarr_channel_idx: Local zarr channel index
            z_idx: Z-slice index (default 0)
            timepoint: Timepoint index (default 0)
        """
        try:
            # Add to stitching queue with normal scan flag (all scales) - same as normal_scan_with_stitching
            queue_item = {
                'image': image.copy(),
                'x_mm': x_mm,  # Use absolute coordinates - WellZarrCanvas will convert to well-relative
                'y_mm': y_mm,  # Use absolute coordinates - WellZarrCanvas will convert to well-relative
                'channel_idx': zarr_channel_idx,
                'z_idx': z_idx,
                'timepoint': timepoint,
                'timestamp': time.time(),
                'quick_scan': False  # Flag to indicate this is normal scan (all scales)
            }

            # Check queue size before adding
            queue_size_before = canvas.stitch_queue.qsize()
            await canvas.stitch_queue.put(queue_item)
            queue_size_after = canvas.stitch_queue.qsize()


            return f"Queued for stitching (all scales)"

        except Exception as e:
            print(f"    ❌ Failed to add image to stitching queue: {e}")
            return f"Failed to queue: {e}"

    async def _add_image_with_backpressure(self, canvas, image, x_mm, y_mm, zarr_channel_idx, max_queue_size=100):
        """
        Add image to canvas with backpressure to prevent queue overflow.
        
        Args:
            canvas: WellZarrCanvas instance
            image: Image array
            x_mm, y_mm: Coordinates
            zarr_channel_idx: Channel index
            max_queue_size: Maximum queue size before applying backpressure
        """
        # Check queue size and apply backpressure if needed
        queue_size = canvas.stitch_queue.qsize()
        
        if queue_size > max_queue_size:
            # Wait for queue to drain a bit
            print(f"Stitching queue full ({queue_size} items), waiting for processing...")
            while canvas.stitch_queue.qsize() > max_queue_size // 2:
                await asyncio.sleep(0.1)
        
        # Add image to canvas
        await canvas.add_image_async(
            image, x_mm - canvas.well_center_x, y_mm - canvas.well_center_y,
            zarr_channel_idx, 0, 0  # z_idx=0, timepoint=0
        )

    def _export_well_to_zip_direct(self, canvas, filename: str) -> str:
        """
        Export a well canvas to a ZIP file on disk using CONFIG.DEFAULT_SAVING_PATH.
        
        Args:
            canvas: WellZarrCanvas instance
            filename: Desired filename (e.g., 'well_A1_96.zip')
            
        Returns:
            str: Path to the created ZIP file
        """
        from pathlib import Path
        from squid_control.control.config import CONFIG
        
        # Use CONFIG.DEFAULT_SAVING_PATH for output directory
        try:
            output_dir = Path(CONFIG.DEFAULT_SAVING_PATH) / "well_zips"
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Using output directory: {output_dir}")
        except Exception as e:
            # Fallback to temp directory if CONFIG not available
            import tempfile
            output_dir = Path(tempfile.gettempdir()) / "well_zips"
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.warning(f"Could not use CONFIG.DEFAULT_SAVING_PATH, using temp: {e}")
        
        # Create full path for the ZIP file
        zip_file_path = output_dir / filename
        
        # Clean up any partial Zarr files before export
        if hasattr(canvas, 'zarr_path') and canvas.zarr_path.exists():
            import glob
            import os
            partial_files = glob.glob(str(canvas.zarr_path / "**" / "*.partial"), recursive=True)
            for partial_file in partial_files:
                try:
                    os.remove(partial_file)
                except:
                    pass
        
        # Export canvas to this specific file path
        self.logger.info(f"Exporting well canvas to: {zip_file_path}")
        canvas.export_to_zip(str(zip_file_path))
        
        return str(zip_file_path)

    async def _upload_existing_wells_from_directory(self, well_zips_path: Path, experiment_folder: Path,
                                                   experiment_id: str, upload_immediately: bool,
                                                   cleanup_temp_files: bool) -> dict:
        """
        Upload existing well ZIP files from well_zips directory (when .done file detected).
        
        Args:
            well_zips_path: Path to directory containing well ZIP files
            experiment_folder: Original experiment folder for metadata
            experiment_id: Experiment ID for upload
            upload_immediately: Whether to upload immediately
            cleanup_temp_files: Whether to cleanup temp files
            
        Returns:
            Dictionary with processing results
        """
        try:
            print(f"📁 Scanning for existing well ZIP files in {well_zips_path}...")
            
            # Find all well ZIP files matching the pattern well_*_96.zip
            well_zip_files = []
            zip_pattern = "well_*_96.zip"
            
            for zip_file in well_zips_path.glob(zip_pattern):
                if zip_file.is_file():
                    file_size_bytes = zip_file.stat().st_size
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    
                    # Extract well name from filename (e.g., "well_A1_96.zip" -> "well_A1_96")
                    well_name = zip_file.stem
                    
                    well_zip_files.append({
                        'name': well_name,
                        'file_path': str(zip_file),
                        'size_mb': file_size_mb
                    })
                    
                    print(f"  📦 Found: {well_name} ({file_size_mb:.2f} MB)")
            
            if not well_zip_files:
                print(f"⚠️ No well ZIP files found matching pattern {zip_pattern}")
                return {
                    "success": False,
                    "experiment_folder": experiment_folder.name,
                    "error": f"No well ZIP files found in {well_zips_path}"
                }
            
            wells_processed = len(well_zip_files)
            total_size_mb = sum(well_info['size_mb'] for well_info in well_zip_files)
            
            print(f"🎉 Found {wells_processed} existing well ZIP files, total size: {total_size_mb:.2f} MB")
            
            # Create dataset name using normalized timestamp extraction
            dataset_name = self.create_normalized_dataset_name(experiment_folder, experiment_id)
            print(f"📝 Using dataset name: {dataset_name}")

            # Upload all wells to a single dataset (if upload requested)
            upload_result = None
            if upload_immediately and self.zarr_artifact_manager:
                print(f"\n📦 Uploading {len(well_zip_files)} existing wells to single dataset...")
                
                try:
                    # Use the original experiment_id for gallery creation, dataset_name for dataset naming
                    gallery_experiment_id = experiment_id if experiment_id else dataset_name
                    
                    # Upload all wells to a single dataset
                    upload_result = await self.zarr_artifact_manager.upload_multiple_zip_files_to_dataset(
                        microscope_service_id=self.service_id,
                        experiment_id=gallery_experiment_id,
                        zarr_files_info=well_zip_files,  # Already has file_path instead of content
                        dataset_name=dataset_name,
                        acquisition_settings={
                            "microscope_service_id": self.service_id,
                            "experiment_name": experiment_folder.name,
                            "total_wells": len(well_zip_files),
                            "total_size_mb": total_size_mb,
                            "offline_processing": True,
                            "from_existing_zips": True  # Flag to indicate this was from existing files
                        },
                        description=f"Upload of existing processed wells: {experiment_folder.name} with {len(well_zip_files)} wells (detected .done file)"
                    )
                    
                    print(f"  ✅ Dataset upload complete: {upload_result.get('dataset_name')}")
                    
                    # Clean up individual well ZIP files if requested
                    if cleanup_temp_files:
                        for well_info in well_zip_files:
                            try:
                                import os
                                os.unlink(well_info['file_path'])
                                print(f"    🗑️ Cleaned up {well_info['name']}.zip")
                            except Exception as e:
                                print(f"    ⚠️ Failed to cleanup {well_info['name']}.zip: {e}")
                        
                        # Also remove the .done file
                        try:
                            done_file = well_zips_path / ".done"
                            done_file.unlink()
                            print(f"    🗑️ Cleaned up .done file")
                        except Exception as e:
                            print(f"    ⚠️ Failed to cleanup .done file: {e}")
                    
                    # Clean up any existing temporary offline_stitch folders after successful upload
                    self._cleanup_existing_temp_folders(experiment_folder.name)
                    
                except Exception as upload_error:
                    print(f"  ❌ Dataset upload failed: {upload_error}")
                    upload_result = None
            else:
                print(f"⏭️ Upload skipped (upload_immediately={upload_immediately}, zarr_artifact_manager available: {self.zarr_artifact_manager is not None})")

            return {
                "success": True,
                "experiment_folder": experiment_folder.name,
                "wells_processed": wells_processed,
                "total_size_mb": total_size_mb,
                "dataset_name": dataset_name,
                "upload_result": upload_result,
                "from_existing_zips": True  # Flag to indicate this was from existing files
            }

        except Exception as e:
            self.logger.error(f"Error uploading existing wells from {well_zips_path}: {e}")
            return {
                "success": False,
                "experiment_folder": experiment_folder.name,
                "error": str(e)
            }



def create_optimized_processor(squid_controller, zarr_artifact_manager=None, service_id=None,
                              max_concurrent_wells=3, image_batch_size=5):
    """
    Create an optimized OfflineProcessor instance with performance tuning.
    
    Args:
        squid_controller: SquidController instance
        zarr_artifact_manager: ZarrArtifactManager instance
        service_id: Service ID for uploads
        max_concurrent_wells: Maximum number of wells to process concurrently (default: 3)
        image_batch_size: Number of images to process in each batch (default: 10)
        
    Returns:
        Optimized OfflineProcessor instance
    """
    return OfflineProcessor(
        squid_controller=squid_controller,
        zarr_artifact_manager=zarr_artifact_manager,
        service_id=service_id,
        max_concurrent_wells=max_concurrent_wells,
        image_batch_size=image_batch_size
    )


def create_high_performance_processor(squid_controller, zarr_artifact_manager=None, service_id=None):
    """
    Create a high-performance OfflineProcessor instance optimized for speed.
    
    This configuration prioritizes speed over memory usage.
    
    Args:
        squid_controller: SquidController instance
        zarr_artifact_manager: ZarrArtifactManager instance
        service_id: Service ID for uploads
        
    Returns:
        High-performance OfflineProcessor instance
    """
    return OfflineProcessor(
        squid_controller=squid_controller,
        zarr_artifact_manager=zarr_artifact_manager,
        service_id=service_id,
        max_concurrent_wells=4,  # Higher concurrency
        image_batch_size=10      # Larger batches
    )


def create_memory_efficient_processor(squid_controller, zarr_artifact_manager=None, service_id=None):
    """
    Create a memory-efficient OfflineProcessor instance optimized for low memory usage.
    
    This configuration prioritizes memory usage over speed.
    
    Args:
        squid_controller: SquidController instance
        zarr_artifact_manager: ZarrArtifactManager instance
        service_id: Service ID for uploads
        
    Returns:
        Memory-efficient OfflineProcessor instance
    """
    return OfflineProcessor(
        squid_controller=squid_controller,
        zarr_artifact_manager=zarr_artifact_manager,
        service_id=service_id,
        max_concurrent_wells=1,  # Lower concurrency
        image_batch_size=5       # Smaller batches
    )
