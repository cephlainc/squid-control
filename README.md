# Squid Control

The Squid Control software is a Python package that provides a simple interface to control the Squid microscope, integrated with the [Hypha platform](https://hypha.aicell.io/) for remote access and distributed control.

## Installation and Usage

### Quick Start

**Install from source (recommended for development)**
```bash
# Clone the repository
git clone https://github.com/aicell-lab/squid-control.git
cd squid-control

# Install in development mode
pip install -e .[dev]
```

### Environment Setup

For development, we recommend using conda:

```bash
# Create conda environment
conda create -n squid python=3.11

# Activate environment
conda activate squid

# Install in development mode
pip install -e .[dev]
```

### Usage

**Command Line Interface:**

The Squid Control system provides a unified command-line interface with subcommands:

```bash
# Main microscope service
python -m squid_control microscope [--simulation] [--local] [--verbose]

# Mirror service for cloud-to-local proxy
python -m squid_control mirror [--cloud-service-id ID] [--local-service-id ID] [--verbose]

# Examples:
# Run microscope in simulation mode
python -m squid_control microscope --simulation

# Run microscope in local mode
python -m squid_control microscope --local

# Run microscope with verbose logging
python -m squid_control microscope --simulation --verbose

# Get help
python -m squid_control --help
python -m squid_control microscope --help
python -m squid_control mirror --help
```

### Simulation Mode

To start simulation mode, use the following command:
```bash
python -m squid_control microscope --simulation
```

The simulation mode includes a **virtual microscope sample** using Zarr data archives, allowing you to test the microscope software without physical hardware. The simulated sample data is uploaded on **Artifact Manager**, which is a feature on the Hypha platform for managing and sharing large datasets.

## Mirror Service

The **Mirror Service** is a sophisticated proxy system that bridges cloud and local microscope control systems, enabling remote control of microscopes while maintaining full WebRTC video streaming capabilities.

### Why Do We Need Mirror Service?

The public Hypha server (`hypha.aicell.io`) may not always be stable for critical device control. The mirror service provides a solution:

1. **Setup local Hypha server** on your workstation for stable device control
2. **Register local microscope service** on your local Hypha server
3. **Run mirror service** on the same workstation to mirror hardware control to remote servers
4. **Result**: You get both stability (local control) and remote access (cloud availability)

### How to Use Mirror Service

```bash
# Run mirror service with default settings
python -m squid_control mirror

# Run with custom service IDs
python -m squid_control mirror \
  --cloud-service-id "mirror-microscope-control-squid-2" \
  --local-service-id "microscope-control-squid-2"

# Run with custom server URLs
python -m squid_control mirror \
  --cloud-server-url "https://hypha.aicell.io" \
  --cloud-workspace "reef-imaging" \
  --local-server-url "http://localhost:9527" \
  --local-service-id "microscope-control-squid-1"
```

### Mirror Service Features

- **Dynamic Method Mirroring**: Automatically mirrors all available methods from local services to cloud
- **WebRTC Video Streaming**: Real-time video with metadata transmission via data channels
- **Health Monitoring**: Automatic health checks with exponential backoff reconnection
- **Configurable Service IDs**: Customizable cloud and local service identifiers
- **Automatic Illumination Control**: Manages illumination based on WebRTC connection state

## Zarr Canvas & Image Stitching

The Squid Control system features advanced **Zarr Canvas & Image Stitching** capabilities that enable real-time creation of large field-of-view images from multiple microscope acquisitions.

### Key Features

#### **Multi-Scale Canvas Architecture**
- **OME-Zarr Compliance**: Full OME-Zarr 0.4 specification support with proper metadata
- **Multi-Scale Pyramid**: 4x downsampling between levels for efficient storage
- **Well-Based Organization**: Individual well canvases for precise control
- **Real-Time Stitching**: Background processing for non-blocking operation
- **Quick Scan Mode**: High-speed continuous scanning (up to 10fps)

### Configuration

#### **Environment Variables**
- `ZARR_PATH`: Base directory for zarr storage (default: `/tmp/zarr_canvas`)
- Authentication token for cloud Hypha server
- Authentication token for local Hypha server (if user have)

---

## About

<img style="width:60px;" src="./docs/assets/aicell-lab.jpeg"> AICell Lab  
<img style="width:60px;" src="./docs/assets/cephla_logo.svg"> Cephla Inc.

---

## Note

The current branch is a fork from https://github.com/hongquanli/octopi-research/ at the following commit:
```
commit dbb49fc314d82d8099d5e509c0e1ad9a919245c9 (HEAD -> master, origin/master, origin/HEAD)
Author: Hongquan Li <hqlisu@gmail.com>
Date:   Thu Apr 4 18:07:51 2024 -0700
    add laser af characterization mode for saving images from laser af camera
```