"""
Services module for squid_control.

This module contains various services including:
- Mirror services for cloud-to-local proxy
- Main microscope control services
- WebRTC video streaming services
"""

__version__ = "0.1.0"

# Import main service classes
from .mirror.mirror_service import MirrorMicroscopeService
from .mirror.video_track import MicroscopeVideoTrack

__all__ = [
    "MirrorMicroscopeService",
    "MicroscopeVideoTrack",
]
