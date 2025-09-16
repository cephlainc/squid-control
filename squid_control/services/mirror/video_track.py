"""
Microscope video track for WebRTC streaming.

This module provides the MicroscopeVideoTrack class that handles real-time
video streaming from the microscope to WebRTC clients.
"""

import asyncio
import fractions
import json
import logging
import time

import cv2
import numpy as np
from aiortc import MediaStreamTrack
from av import VideoFrame

logger = logging.getLogger(__name__)


class MicroscopeVideoTrack(MediaStreamTrack):
    """
    A video stream track that provides real-time microscope images.
    """

    kind = "video"

    def __init__(self, local_service, parent_service=None):
        super().__init__()
        if local_service is None:
            raise ValueError("local_service cannot be None when initializing MicroscopeVideoTrack")
        self.local_service = local_service
        self.parent_service = parent_service  # Reference to MirrorMicroscopeService for data channel access
        self.count = 0
        self.running = True
        self.start_time = None
        self.fps = 5  # Target FPS for WebRTC stream
        self.frame_width = 750
        self.frame_height = 750
        logger.info("MicroscopeVideoTrack initialized with local_service")

    def draw_crosshair(self, img, center_x, center_y, size=20, color=[255, 255, 255]):
        """Draw a crosshair at the specified position"""
        height, width = img.shape[:2]

        # Horizontal line
        if 0 <= center_y < height:
            start_x = max(0, center_x - size)
            end_x = min(width, center_x + size)
            img[center_y, start_x:end_x] = color

        # Vertical line
        if 0 <= center_x < width:
            start_y = max(0, center_y - size)
            end_y = min(height, center_y + size)
            img[start_y:end_y, center_x] = color

    async def recv(self):
        if not self.running:
            logger.warning("MicroscopeVideoTrack: recv() called but track is not running")
            raise Exception("Track stopped")

        try:
            if self.start_time is None:
                self.start_time = time.time()

            # Time the entire frame processing (including sleep)
            frame_start_time = time.time()

            # Calculate and perform FPS throttling sleep
            next_frame_time = self.start_time + (self.count / self.fps)
            sleep_duration = next_frame_time - time.time()
            sleep_start = time.time()
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
            sleep_end = time.time()
            actual_sleep_time = (sleep_end - sleep_start) * 1000  # Convert to ms

            # Start timing actual processing after sleep
            processing_start_time = time.time()

            # Check if local_service is still available
            if self.local_service is None:
                logger.error("MicroscopeVideoTrack: local_service is None")
                raise Exception("Local service not available")

            # Time getting the video frame from local service
            get_frame_start = time.time()
            frame_data = await self.local_service.get_video_frame(
                frame_width=self.frame_width,
                frame_height=self.frame_height
            )
            get_frame_end = time.time()
            get_frame_latency = (get_frame_end - get_frame_start) * 1000  # Convert to ms

            # Extract stage position from frame metadata
            stage_position = None
            if isinstance(frame_data, dict) and 'metadata' in frame_data:
                stage_position = frame_data['metadata'].get('stage_position')
                logger.debug(f"Frame {self.count}: Found stage_position in metadata: {stage_position}")
            else:
                logger.debug(f"Frame {self.count}: No metadata found in frame_data, keys: {list(frame_data.keys()) if isinstance(frame_data, dict) else 'not dict'}")

            # Handle new JPEG format returned by get_video_frame
            if isinstance(frame_data, dict) and 'data' in frame_data:
                # New format: dictionary with JPEG data
                jpeg_data = frame_data['data']
                frame_size_bytes = frame_data.get('size_bytes', len(jpeg_data))

                # Decode JPEG data to numpy array
                decode_start = time.time()
                if isinstance(jpeg_data, bytes):
                    # Convert bytes to numpy array for cv2.imdecode
                    jpeg_np = np.frombuffer(jpeg_data, dtype=np.uint8)
                    # Decode JPEG to BGR format (OpenCV default)
                    processed_frame_bgr = cv2.imdecode(jpeg_np, cv2.IMREAD_COLOR)
                    if processed_frame_bgr is None:
                        raise Exception("Failed to decode JPEG data")
                    # Convert BGR to RGB for VideoFrame
                    processed_frame = cv2.cvtColor(processed_frame_bgr, cv2.COLOR_BGR2RGB)
                else:
                    raise Exception(f"Unexpected JPEG data type: {type(jpeg_data)}")
                decode_end = time.time()
                decode_latency = (decode_end - decode_start) * 1000  # Convert to ms
                print(f"Frame {self.count} decode time: {decode_latency:.2f}ms")
            else:
                # Fallback for old format (numpy array)
                processed_frame = frame_data
                if hasattr(processed_frame, 'nbytes'):
                    frame_size_bytes = processed_frame.nbytes
                else:
                    import sys  # noqa: PLC0415
                    frame_size_bytes = sys.getsizeof(processed_frame)

                frame_size_kb = frame_size_bytes / 1024
                print(f"Frame {self.count} raw data size: {frame_size_kb:.2f} KB ({frame_size_bytes} bytes)")

            # Time processing the frame
            process_start = time.time()
            current_time = time.time()
            # Use a 90kHz timebase, common for video, to provide accurate frame timing.
            # This prevents video from speeding up if frame acquisition is slow.
            time_base = fractions.Fraction(1, 90000)
            pts = int((current_time - self.start_time) * time_base.denominator)

            new_video_frame = VideoFrame.from_ndarray(processed_frame, format="rgb24")
            new_video_frame.pts = pts
            new_video_frame.time_base = time_base
            process_end = time.time()
            process_latency = (process_end - process_start) * 1000  # Convert to ms

            # SEND METADATA VIA WEBRTC DATA CHANNEL
            # Send metadata through data channel instead of embedding in video frame
            if stage_position and self.parent_service:
                try:
                    # Create frame metadata including stage position
                    frame_metadata = {
                        'stage_position': stage_position,
                        'timestamp': current_time,
                        'frame_count': self.count
                    }
                    # Add any additional metadata from frame_data if available
                    if isinstance(frame_data, dict) and 'metadata' in frame_data:
                        frame_metadata.update(frame_data['metadata'])

                    metadata_json = json.dumps(frame_metadata)
                    # Send metadata via WebRTC data channel
                    asyncio.create_task(self._send_metadata_via_datachannel(metadata_json))
                    logger.debug(f"Sent metadata via data channel: {len(metadata_json)} bytes")
                except Exception as e:
                    logger.warning(f"Failed to send metadata via data channel: {e}")

            # Calculate processing and total latencies
            processing_end_time = time.time()
            processing_latency = (processing_end_time - processing_start_time) * 1000  # Convert to ms
            total_frame_latency = (processing_end_time - frame_start_time) * 1000  # Convert to ms

            # Print timing information every frame (you can adjust frequency as needed)
            if isinstance(frame_data, dict) and 'data' in frame_data:
                print(f"Frame {self.count} timing: sleep={actual_sleep_time:.2f}ms, get_video_frame={get_frame_latency:.2f}ms, decode={decode_latency:.2f}ms, process={process_latency:.2f}ms, processing_total={processing_latency:.2f}ms, total_with_sleep={total_frame_latency:.2f}ms")
            else:
                print(f"Frame {self.count} timing: sleep={actual_sleep_time:.2f}ms, get_video_frame={get_frame_latency:.2f}ms, process={process_latency:.2f}ms, processing_total={processing_latency:.2f}ms, total_with_sleep={total_frame_latency:.2f}ms")

            if self.count % (self.fps * 5) == 0:  # Log every 5 seconds
                duration = current_time - self.start_time
                if duration > 0:
                    actual_fps = (self.count + 1) / duration
                    logger.info(f"MicroscopeVideoTrack: Sent frame {self.count}, actual FPS: {actual_fps:.2f}")
                else:
                    logger.info(f"MicroscopeVideoTrack: Sent frame {self.count}")

            self.count += 1
            return new_video_frame

        except Exception as e:
            logger.error(f"MicroscopeVideoTrack: Error in recv(): {e}", exc_info=True)
            self.running = False
            raise

    def stop(self):
        logger.info("MicroscopeVideoTrack stop() called.")
        self.running = False

    async def _send_metadata_via_datachannel(self, metadata_json):
        """Send metadata via WebRTC data channel"""
        try:
            if (self.parent_service and
                hasattr(self.parent_service, 'metadata_data_channel') and
                self.parent_service.metadata_data_channel):
                if self.parent_service.metadata_data_channel.readyState == 'open':
                    self.parent_service.metadata_data_channel.send(metadata_json)
                    logger.debug(f"Metadata sent via data channel: {len(metadata_json)} bytes")
                else:
                    logger.debug(f"Data channel not ready, state: {self.parent_service.metadata_data_channel.readyState}")
        except Exception as e:
            logger.warning(f"Error sending metadata via data channel: {e}")
