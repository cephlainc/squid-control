"""
Mirror services for squid_control.

This module provides proxy services that bridge cloud and local microscope control systems.
"""

__version__ = "0.1.0"

from .mirror_service import MirrorMicroscopeService
from .video_track import MicroscopeVideoTrack

__all__ = [
    "MirrorMicroscopeService",
    "MicroscopeVideoTrack",
]
