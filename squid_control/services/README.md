# Squid Control Services

This directory contains various services for the Squid microscope control system.

## Directory Structure

```
services/
├── __init__.py              # Services module initialization
├── mirror/                  # Mirror service for cloud-to-local proxy
│   ├── __init__.py         # Mirror service module initialization
│   ├── mirror_service.py   # Main mirror service class
│   ├── video_track.py      # WebRTC video track component
│   └── cli.py              # Command-line interface
└── README.md               # This file
```

## Mirror Service

The mirror service acts as a proxy between cloud and local microscope control systems, allowing remote control of microscopes while maintaining WebRTC video streaming capabilities.

### Features

- **Dynamic Method Mirroring**: Automatically mirrors all available methods from local services
- **WebRTC Video Streaming**: Real-time video streaming with metadata transmission
- **Health Monitoring**: Automatic health checks and reconnection handling
- **Configurable Service IDs**: Customizable cloud and local service identifiers

### Usage

#### Method 1: Using the main module (Recommended)

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



#### Method 2: Backward compatibility script

```bash
# Use the legacy runner script
python squid_control/run_mirror_service.py \
  --cloud-service-id "mirror-microscope-control-squid-2" \
  --local-service-id "microscope-control-squid-2"
```

### Configuration

The mirror service can be configured through:

1. **Environment Variables**:
   - `REEF_WORKSPACE_TOKEN`: Cloud service authentication token
   - `REEF_LOCAL_TOKEN`: Local service authentication token

2. **Command-Line Arguments**:
   - `--cloud-service-id`: ID for the cloud service
   - `--local-service-id`: ID for the local service
   - `--cloud-server-url`: Cloud server URL
   - `--cloud-workspace`: Cloud workspace name
   - `--local-server-url`: Local server URL
   - `--log-file`: Log file path
   - `--verbose`: Enable verbose logging

### Architecture

The mirror service consists of several components:

1. **MirrorMicroscopeService**: Main service class that handles:
   - Cloud and local service connections
   - Dynamic method mirroring
   - WebRTC service management
   - Health monitoring and reconnection

2. **MicroscopeVideoTrack**: WebRTC video track that:
   - Streams real-time microscope images
   - Handles frame processing and timing
   - Transmits metadata via data channels
   - Manages FPS and quality settings

3. **CLI Interface**: Command-line interface that:
   - Parses command-line arguments
   - Configures the service
   - Handles startup and shutdown
   - Provides user feedback

### Health Monitoring

The service includes comprehensive health monitoring:

- **Automatic Reconnection**: Reconnects to lost services automatically
- **Health Checks**: Regular ping operations to verify service health
- **Exponential Backoff**: Intelligent retry logic for failed connections
- **Graceful Degradation**: Continues operation with available services

### WebRTC Integration

The WebRTC service provides:

- **Real-time Video**: Live microscope video streaming
- **Metadata Transmission**: Stage position and other data via data channels
- **Automatic Illumination**: Turns on/off illumination based on connection state
- **ICE Server Management**: Automatic STUN/TURN server configuration

### Error Handling

The service implements robust error handling:

- **Connection Failures**: Automatic retry with exponential backoff
- **Service Unavailability**: Graceful degradation and fallback
- **Resource Cleanup**: Proper cleanup of resources on shutdown
- **Logging**: Comprehensive logging for debugging and monitoring

## Development

### Adding New Services

To add a new service:

1. Create a new directory under `services/`
2. Implement the service class
3. Create a CLI interface if needed
4. Update the main module entry point
5. Add tests and documentation

### Testing

```bash
# Run tests for the services module
python -m pytest tests/ -k "services"

# Run specific service tests
python -m pytest tests/ -k "mirror"
```

### Contributing

When contributing to the services:

1. Follow the established code structure
2. Add proper error handling and logging
3. Include comprehensive tests
4. Update documentation
5. Follow the project's coding standards
