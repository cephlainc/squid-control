"""
Squid Microscope Control System

A Python-based control system for the Squid microscope (by Cephla Inc.), featuring:
- Real-time microscope hardware control and automation
- Web-based API service using Hypha RPC
- Camera integration with multiple vendors (ToupCam, FLIR, TIS)
- Well plate scanning and image acquisition
- WebRTC video streaming for remote microscope viewing
- AI-powered chatbot integration for natural language microscope control
- Simulation mode with Zarr-based virtual samples
- Multi-channel fluorescence imaging capabilities
- Mirror services for cloud-to-local proxy control

Usage:
    # Run main microscope service
    python -m squid_control microscope [--simulation] [--local] [--verbose]
    
    # Run mirror service
    python -m squid_control mirror --cloud-service-id "mirror-microscope-control-squid-2" --local-service-id "microscope-control-squid-2"
    
    # Import in Python code
    from squid_control.start_hypha_service import MicroscopeHyphaService
    from squid_control.squid_controller import SquidController
    from squid_control.services.mirror import MirrorMicroscopeService
"""

__version__ = "0.1.0"
__author__ = "Cephla Inc."

# Use lazy imports to avoid installation-time failures
def _import_main_classes():
    """Lazy import to avoid import errors during installation"""
    try:
        from .squid_controller import SquidController
        from .start_hypha_service import MicroscopeHyphaService
        return MicroscopeHyphaService, SquidController
    except ImportError:
        return None, None

def _import_mirror_services():
    """Lazy import of mirror services"""
    try:
        from .services.mirror import MicroscopeVideoTrack, MirrorMicroscopeService
        return MirrorMicroscopeService, MicroscopeVideoTrack, True
    except ImportError:
        return None, None, False

# Lazy load classes
MicroscopeHyphaService, SquidController = _import_main_classes()
MirrorMicroscopeService, MicroscopeVideoTrack, MIRROR_SERVICES_AVAILABLE = _import_mirror_services()

__all__ = [
    "MicroscopeHyphaService",
    "SquidController",
]

# Add mirror services if available
if MIRROR_SERVICES_AVAILABLE:
    __all__.extend([
        "MirrorMicroscopeService",
        "MicroscopeVideoTrack",
    ])
