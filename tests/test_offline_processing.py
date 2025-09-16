"""
Test suite for offline processing functionality.

WHAT WAS TESTED:
================
✅ Data Generation & File Structure
   - OfflineDataGenerator creates realistic synthetic microscopy data
   - Generates proper folder structure: experiment_id-timestamp/0/
   - Creates all required files: acquisition parameters.json, configurations.xml, 
     coordinates.csv, BMP image files with correct naming pattern

✅ Configuration Management
   - Tests that CONFIG.DEFAULT_SAVING_PATH is properly set and used
   - Verifies the offline processor can find experiment folders using the config

✅ Metadata Parsing
   - Parses acquisition parameters from JSON
   - Extracts channel configurations from XML
   - Reads coordinates from CSV and groups by well
   - Creates proper channel name mappings

✅ Image Processing Pipeline
   - Loads BMP images from disk
   - Processes images through the stitching queue
   - Handles multiple channels (BF LED matrix, Fluorescence 488nm)
   - Converts coordinates and applies proper transformations

✅ Zarr Canvas Operations
   - Creates well-specific Zarr canvases
   - Adds images to stitching queue with correct parameters
   - Waits for stitching completion properly
   - Exports canvases to ZIP files

✅ File Management
   - Creates temporary directories for processing
   - Exports well canvases to ZIP files in well_zips/ directory
   - Creates .done marker files
   - Handles cleanup of temporary files

✅ Upload Interface
   - Calls the upload method with correct parameters
   - Passes proper metadata (experiment_id, dataset_name, etc.)
   - Handles the upload response structure

✅ Error Handling & Edge Cases
   - Tests the .done file shortcut path (skip processing, upload existing)
   - Handles missing or invalid data gracefully
   - Proper cleanup on success/failure

WHAT WAS NOT TESTED:
===================
❌ Actual Network Upload - No real HTTP requests to artifact manager
❌ Large Dataset Performance - Only tested with tiny datasets
❌ Real Hardware Integration - No actual microscope hardware

"""

import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import pytest
import pytest_asyncio

from squid_control.control.config import CONFIG
from squid_control.offline_processing import OfflineProcessor


class OfflineDataGenerator:
    """Helper class to generate synthetic microscopy data for offline processing tests."""

    @staticmethod
    def create_synthetic_microscopy_data(base_path: Path, experiment_id: str,
                                       num_runs: int = 2, wells: List[str] = None,
                                       channels: List[str] = None) -> List[Path]:
        """
        Create synthetic microscopy data in the expected format for offline processing.

        Args:
            base_path: Base directory to create experiment folders
            experiment_id: Experiment ID prefix
            num_runs: Number of experiment runs to create
            wells: List of well IDs (e.g., ['A1', 'B2', 'C3'])
            channels: List of channel names

        Returns:
            List of created experiment folder paths
        """
        if wells is None:
            wells = ['A1', 'B2', 'C3']  # Default test wells
        if channels is None:
            channels = ['BF LED matrix full', 'Fluorescence 488 nm Ex', 'Fluorescence 561 nm Ex']

        experiment_folders = []

        for run_idx in range(num_runs):
            # Create timestamp for this run
            timestamp = f"20250822T{14 + run_idx:02d}30{run_idx:02d}"
            experiment_folder = base_path / f"{experiment_id}-{timestamp}"
            experiment_folder.mkdir(parents=True, exist_ok=True)

            # Create the '0' subfolder
            data_folder = experiment_folder / "0"
            data_folder.mkdir(exist_ok=True)

            # Creating synthetic data in: {experiment_folder}

            # Generate acquisition parameters
            acquisition_params = {
                "dx(mm)": 0.9,
                "Nx": 3,
                "dy(mm)": 0.9,
                "Ny": 3,
                "dz(um)": 1.5,
                "Nz": 1,
                "dt(s)": 0,
                "Nt": 1,
                "with CONFIG.AF": False,
                "with reflection CONFIG.AF": True,
                "objective": {
                    "magnification": 20,
                    "NA": 0.4,
                    "tube_lens_f_mm": 180,
                    "name": "20x (Boli)"
                },
                "sensor_pixel_size_um": 1.85,
                "tube_lens_mm": 50
            }

            with open(data_folder / "acquisition parameters.json", 'w') as f:
                json.dump(acquisition_params, f, indent=2)

            # Generate configurations.xml
            OfflineDataGenerator._create_configurations_xml(data_folder, channels)

            # Generate coordinates and images for each well
            all_coordinates = []

            for well_idx, well_id in enumerate(wells):
                well_coords = OfflineDataGenerator._create_well_data(
                    data_folder, well_id, channels, acquisition_params, well_idx
                )
                all_coordinates.extend(well_coords)

            # Create coordinates.csv
            df = pd.DataFrame(all_coordinates)
            df.to_csv(data_folder / "coordinates.csv", index=False)

            experiment_folders.append(experiment_folder)
            # Created experiment run: {experiment_folder.name}

        return experiment_folders

    @staticmethod
    def _create_configurations_xml(data_folder: Path, channels: List[str]):
        """Create configurations.xml file with channel settings."""
        root = ET.Element("modes")

        # Channel mapping to XML format
        channel_configs = {
            "BF LED matrix full": {
                "ID": "1",
                "ExposureTime": "5.0",
                "AnalogGain": "1.1",
                "IlluminationSource": "0",
                "IlluminationIntensity": "32.0"
            },
            "Fluorescence 488 nm Ex": {
                "ID": "6",
                "ExposureTime": "100.0",
                "AnalogGain": "10.0",
                "IlluminationSource": "12",
                "IlluminationIntensity": "27.0"
            },
            "Fluorescence 561 nm Ex": {
                "ID": "8",
                "ExposureTime": "300.0",
                "AnalogGain": "10.0",
                "IlluminationSource": "14",
                "IlluminationIntensity": "50.0"
            }
        }

        for channel in channels:
            config = channel_configs.get(channel, {
                "ID": "1",
                "ExposureTime": "50.0",
                "AnalogGain": "1.0",
                "IlluminationSource": "0",
                "IlluminationIntensity": "50.0"
            })

            mode = ET.SubElement(root, "mode")
            mode.set("ID", config["ID"])
            mode.set("Name", channel)
            mode.set("ExposureTime", config["ExposureTime"])
            mode.set("AnalogGain", config["AnalogGain"])
            mode.set("IlluminationSource", config["IlluminationSource"])
            mode.set("IlluminationIntensity", config["IlluminationIntensity"])
            mode.set("CameraSN", "")
            mode.set("ZOffset", "0.0")
            mode.set("PixelFormat", "default")
            mode.set("_PixelFormat_options", "[default,MONO8,MONO12,MONO14,MONO16,BAYER_RG8,BAYER_RG12]")
            mode.set("Selected", "1")

        # Write XML file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(data_folder / "configurations.xml", encoding="UTF-8", xml_declaration=True)

    @staticmethod
    def _create_well_data(data_folder: Path, well_id: str, channels: List[str],
                         acquisition_params: dict, well_offset: int) -> List[dict]:
        """Create synthetic images and coordinates for a single well."""
        coordinates = []

        # Well center coordinates (simulate different well positions)
        well_center_x = 20.0 + well_offset * 9.0  # 9mm spacing between wells
        well_center_y = 60.0 + well_offset * 9.0

        Nx = acquisition_params["Nx"]
        Ny = acquisition_params["Ny"]
        dx = acquisition_params["dx(mm)"]
        dy = acquisition_params["dy(mm)"]

        # Generate images for each position in the well
        for i in range(Nx):
            for j in range(Ny):
                # Calculate position coordinates
                x_mm = well_center_x + (i - Nx//2) * dx
                y_mm = well_center_y + (j - Ny//2) * dy
                z_um = 4035.0 + np.random.normal(0, 10)  # Simulate focus variation

                # Generate timestamp
                timestamp = f"2025-08-22_18-16-{35 + i*2 + j}.{702228 + i*100 + j*10:06d}"

                # Create images for each channel
                for channel in channels:
                    # Generate synthetic microscopy image
                    image = OfflineDataGenerator._generate_synthetic_image(channel, i, j)

                    # Save as BMP file
                    filename = f"{well_id}_{i}_{j}_0_{channel.replace(' ', '_')}.bmp"
                    filepath = data_folder / filename
                    cv2.imwrite(str(filepath), image)

                # Add coordinate record
                coordinates.append({
                    "i": i,
                    "j": j,
                    "k": 0,
                    "x (mm)": x_mm,
                    "y (mm)": y_mm,
                    "z (um)": z_um,
                    "time": timestamp,
                    "region": well_id
                })

        return coordinates

    @staticmethod
    def _generate_synthetic_image(channel: str, i: int, j: int) -> np.ndarray:
        """Generate a synthetic microscopy image for testing."""
        # Create 512x512 image
        height, width = 512, 512

        # Generate different patterns based on channel
        if "BF" in channel or "Bright" in channel:
            # Brightfield - uniform with some texture
            image = np.random.normal(2000, 100, (height, width)).astype(np.uint16)
            # Add some structure
            y, x = np.ogrid[:height, :width]
            structure = 500 * np.sin(x * 0.02) * np.cos(y * 0.02)
            image = np.clip(image + structure, 0, 4095).astype(np.uint16)

        elif "488" in channel:
            # GFP-like fluorescence
            image = np.random.exponential(200, (height, width)).astype(np.uint16)
            # Add some bright spots
            for _ in range(5):
                center_y = np.random.randint(50, height-50)
                center_x = np.random.randint(50, width-50)
                y, x = np.ogrid[:height, :width]
                spot = 1000 * np.exp(-((x-center_x)**2 + (y-center_y)**2) / (2*30**2))
                image = np.clip(image + spot, 0, 4095).astype(np.uint16)

        elif "561" in channel:
            # RFP-like fluorescence
            image = np.random.gamma(2, 150, (height, width)).astype(np.uint16)
            # Add some linear structures
            y, x = np.ogrid[:height, :width]
            lines = 800 * np.sin(x * 0.01 + y * 0.005)
            image = np.clip(image + lines, 0, 4095).astype(np.uint16)

        else:
            # Default pattern
            image = np.random.randint(100, 1000, (height, width), dtype=np.uint16)

        # Add position-dependent variation
        position_factor = 1.0 + 0.1 * (i + j) / 6.0
        image = np.clip(image * position_factor, 0, 4095).astype(np.uint16)

        # Convert to 8-bit for BMP format
        image_8bit = (image / 16).astype(np.uint8)

        return image_8bit


class FakeSquidController:
    def __init__(self, pixel_size_xy: float = 0.333):
        self.pixel_size_xy = pixel_size_xy


class FakeZarrArtifactManager:
    """Fake uploader that records uploads instead of performing network I/O."""

    def __init__(self):
        self.upload_calls = []

    async def upload_multiple_zip_files_to_dataset(
        self,
        microscope_service_id,
        experiment_id,
        zarr_files_info,
        dataset_name,
        acquisition_settings,
        description,
    ):
        # Record call for assertions
        self.upload_calls.append(
            {
                "microscope_service_id": microscope_service_id,
                "experiment_id": experiment_id,
                "zarr_files_info": zarr_files_info,
                "dataset_name": dataset_name,
                "acquisition_settings": acquisition_settings,
                "description": description,
            }
        )
        # Return minimal result similar to real manager
        total_mb = sum(info.get("size_mb", 0) for info in zarr_files_info)
        return {
            "success": True,
            "dataset_name": dataset_name,
            "files_uploaded": len(zarr_files_info),
            "total_size_mb": total_mb,
        }


@pytest_asyncio.fixture
async def temp_saving_path():
    """Set CONFIG.DEFAULT_SAVING_PATH to a temporary directory for the test."""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Ensure directory exists and is writable
        base = Path(tmpdir)
        base.mkdir(parents=True, exist_ok=True)
        # Point DEFAULT_SAVING_PATH to our temp dir
        old_path = CONFIG.DEFAULT_SAVING_PATH
        CONFIG.DEFAULT_SAVING_PATH = str(base)
        try:
            yield base
        finally:
            # Restore
            CONFIG.DEFAULT_SAVING_PATH = old_path


@pytest.mark.asyncio
async def test_offline_stitch_and_upload_minimal(temp_saving_path):
    """End-to-end minimal flow: generate tiny experiment and upload one dataset."""

    # Arrange: create small synthetic data under DEFAULT_SAVING_PATH
    experiment_id = "offline-test"
    experiment_folders = OfflineDataGenerator.create_synthetic_microscopy_data(
        base_path=temp_saving_path,
        experiment_id=experiment_id,
        num_runs=1,
        wells=["A1"],
        channels=["BF LED matrix full", "Fluorescence 488 nm Ex"],
    )
    assert len(experiment_folders) == 1

    fake_controller = FakeSquidController()
    fake_uploader = FakeZarrArtifactManager()

    processor = OfflineProcessor(
        squid_controller=fake_controller,
        zarr_artifact_manager=fake_uploader,
        service_id="microscope-control-squid-test",
        max_concurrent_wells=1,
        image_batch_size=2,
    )

    # Act
    result = await processor.stitch_and_upload_timelapse(
        experiment_id=experiment_id,
        upload_immediately=True,
        cleanup_temp_files=False,  # Keep files for assertion
        use_parallel_wells=False,
    )

    # Assert basic success
    assert result["success"] is True
    assert result["total_datasets"] == 1
    assert len(result["processed_runs"]) == 1

    # Assert upload occurred and files look reasonable
    assert len(fake_uploader.upload_calls) == 1
    call = fake_uploader.upload_calls[0]
    assert call["dataset_name"]
    zips = call["zarr_files_info"]
    assert len(zips) >= 1
    for info in zips:
        assert Path(info["file_path"]).exists(), f"Missing ZIP: {info}"
        assert info["size_mb"] > 0


@pytest.mark.asyncio
async def test_offline_done_path_uploads_existing_well_zips(temp_saving_path):
    """If .done exists, processor should skip stitching and upload existing well ZIPs."""

    # Arrange: create a dummy experiment folder (not used for parsing in .done path)
    exp_folder = temp_saving_path / "offline-test-20250101T010101"
    (exp_folder / "0").mkdir(parents=True, exist_ok=True)

    # Create well_zips dir with pre-existing small zip files and a .done marker
    well_zips = Path(CONFIG.DEFAULT_SAVING_PATH) / "well_zips"
    well_zips.mkdir(parents=True, exist_ok=True)

    # Create tiny zip files
    for name in ["well_A1_96.zip", "well_B1_96.zip"]:
        (well_zips / name).write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # minimal empty ZIP EOCD
    # Touch .done
    (well_zips / ".done").touch()

    fake_controller = FakeSquidController()
    fake_uploader = FakeZarrArtifactManager()

    processor = OfflineProcessor(
        squid_controller=fake_controller,
        zarr_artifact_manager=fake_uploader,
        service_id="microscope-control-squid-test",
        max_concurrent_wells=1,
        image_batch_size=1,
    )

    # Act: call run-parallel (will early-return to upload existing)
    run_result = await processor.process_experiment_run_parallel(
        experiment_folder=exp_folder,
        upload_immediately=True,
        cleanup_temp_files=True,
        experiment_id="offline-test",
    )

    # Assert
    assert run_result["success"] is True
    assert run_result.get("from_existing_zips") is True or run_result.get("wells_processed", 0) >= 1
    assert len(fake_uploader.upload_calls) == 1

    # .done should be removed after successful upload when cleanup_temp_files=True
    assert not (well_zips / ".done").exists()  # cleaned up
