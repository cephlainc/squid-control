"""
Mirror microscope service for cloud-to-local proxy.

This module provides the MirrorMicroscopeService class that acts as a proxy
between cloud and local microscope control systems.
"""

import asyncio
import logging
import logging.handlers
import os

# WebRTC imports
import aiohttp

# Image processing imports
import dotenv
from hypha_rpc import connect_to_server, register_rtc_service

from .video_track import MicroscopeVideoTrack

dotenv.load_dotenv()
ENV_FILE = dotenv.find_dotenv()
if ENV_FILE:
    dotenv.load_dotenv(ENV_FILE)

# Set up logging
from squid_control.utils.logging_utils import setup_logging

logger = setup_logging("mirror_squid_control_service.log")


class MirrorMicroscopeService:
    """
    Mirror service that proxies requests between cloud and local microscope systems.
    
    This service allows remote control of microscopes by mirroring local service
    methods to the cloud while maintaining WebRTC video streaming capabilities.
    """

    def __init__(self):
        self.login_required = True
        # Connection to cloud service
        self.cloud_server_url = "https://hypha.aicell.io"
        self.cloud_workspace = "reef-imaging"
        self.cloud_token = os.environ.get("REEF_WORKSPACE_TOKEN")
        self.cloud_service_id = "mirror-microscope-control-squid-1"
        self.cloud_server = None
        self.cloud_service = None  # Add reference to registered cloud service

        # Connection to local service
        self.local_server_url = "http://reef.dyn.scilifelab.se:9527"
        self.local_token = os.environ.get("REEF_LOCAL_TOKEN")
        self.local_service_id = "microscope-control-squid-1"
        self.local_server = None
        self.local_service = None
        self.video_track = None

        # Video streaming state
        self.is_streaming = False
        self.webrtc_service_id = None
        self.webrtc_connected = False
        self.metadata_data_channel = None

        # Setup task tracking
        self.setup_task = None

        # Store dynamically created mirror methods
        self.mirrored_methods = {}

    async def connect_to_local_service(self):
        """Connect to the local microscope service"""
        try:
            logger.info(f"Connecting to local service at {self.local_server_url}")
            self.local_server = await connect_to_server({
                "server_url": self.local_server_url,
                "token": self.local_token,
                "ping_interval": None
            })

            # Connect to the local service
            self.local_service = await self.local_server.get_service(self.local_service_id)
            logger.info(f"Successfully connected to local service {self.local_service_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to local service: {e}")
            self.local_service = None
            self.local_server = None
            return False

    async def cleanup_cloud_service(self):
        """Clean up the cloud service registration"""
        try:
            if self.cloud_service:
                logger.info(f"Unregistering cloud service {self.cloud_service_id}")
                # Try to unregister the service
                try:
                    await self.cloud_server.unregister_service(self.cloud_service_id)
                    logger.info(f"Successfully unregistered cloud service {self.cloud_service_id}")
                except Exception as e:
                    logger.warning(f"Failed to unregister cloud service {self.cloud_service_id}: {e}")

                self.cloud_service = None

            # Clear mirrored methods
            self.mirrored_methods.clear()
            logger.info("Cleared mirrored methods")

        except Exception as e:
            logger.error(f"Error during cloud service cleanup: {e}")

    def _create_mirror_method(self, method_name, local_method):
        """Create a mirror method that forwards calls to the local service"""
        async def mirror_method(*args, **kwargs):
            try:
                if self.local_service is None:
                    logger.warning(f"Local service is None when calling {method_name}, attempting to reconnect")
                    success = await self.connect_to_local_service()
                    if not success or self.local_service is None:
                        raise Exception("Failed to connect to local service")

                # Forward the call to the local service
                result = await local_method(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Failed to call {method_name}: {e}")
                raise e

        # Check if the original method has schema information
        if hasattr(local_method, '__schema__'):
            # Preserve the schema information from the original method
            original_schema = getattr(local_method, '__schema__')

            # Handle case where schema might be None
            if original_schema is not None:
                logger.info(f"Preserving schema for method {method_name}: {original_schema}")

                # Create a new function with the same signature and schema
                # We need to manually copy the schema information since we can't use the decorator directly
                mirror_method.__schema__ = original_schema
                mirror_method.__doc__ = original_schema.get('description', f"Mirror of {method_name}")
            else:
                logger.debug(f"Schema is None for method {method_name}, using basic mirror")
        else:
            # No schema information available, return the basic mirror method
            logger.debug(f"No schema information found for method {method_name}, using basic mirror")

        return mirror_method

    def _get_mirrored_methods(self):
        """Dynamically create mirror methods for all callable methods in local_service"""
        if self.local_service is None:
            logger.warning("Cannot create mirror methods: local_service is None")
            return {}

        logger.info(f"Creating mirror methods for local service {self.local_service_id}")
        logger.info(f"Local service type: {type(self.local_service)}")
        logger.info(f"Local service attributes: {list(dir(self.local_service))}")

        mirrored_methods = {}

        # Methods to exclude from mirroring (these are handled specially)
        excluded_methods = {
            'name', 'id', 'config', 'type',  # Service metadata
            '__class__', '__doc__', '__dict__', '__module__',  # Python internals
        }

        # Get all attributes from the local service
        for attr_name in dir(self.local_service):
            if attr_name.startswith('_') or attr_name in excluded_methods:
                logger.debug(f"Skipping attribute: {attr_name} (excluded or private)")
                continue

            attr = getattr(self.local_service, attr_name)

            # Check if it's callable (a method)
            if callable(attr):
                logger.info(f"Creating mirror method for: {attr_name}")
                mirrored_methods[attr_name] = self._create_mirror_method(attr_name, attr)
            else:
                logger.debug(f"Skipping non-callable attribute: {attr_name}")

        logger.info(f"Total mirrored methods created: {len(mirrored_methods)}")
        logger.info(f"Mirrored method names: {list(mirrored_methods.keys())}")
        return mirrored_methods

    async def check_service_health(self):
        """Check if the service is healthy and rerun setup if needed"""
        logger.info("Starting service health check task")
        while True:
            try:
                # Try to get the service status
                if self.cloud_service_id and self.cloud_server:
                    try:
                        service = await self.cloud_server.get_service(self.cloud_service_id)
                        # Try a simple operation to verify service is working
                        ping_result = await asyncio.wait_for(service.ping(), timeout=10)
                        if ping_result != "pong":
                            logger.error(f"Cloud service health check failed: {ping_result}")
                            raise Exception("Cloud service not healthy")
                    except Exception as e:
                        logger.error(f"Cloud service health check failed: {e}")
                        raise Exception(f"Cloud service not healthy: {e}")
                else:
                    logger.info("Cloud service ID or server not set, waiting for service registration")

                # Always check local service regardless of whether it's None
                try:
                    if self.local_service is None:
                        logger.info("Local service connection lost, attempting to reconnect")
                        success = await self.connect_to_local_service()
                        if not success or self.local_service is None:
                            raise Exception("Failed to connect to local service")

                    #logger.info("Checking local service health...")
                    local_ping_result = await asyncio.wait_for(self.local_service.ping(), timeout=10)
                    #logger.info(f"Local service response: {local_ping_result}")

                    if local_ping_result != "pong":
                        logger.error(f"Local service health check failed: {local_ping_result}")
                        raise Exception("Local service not healthy")

                    #logger.info("Local service health check passed")
                except Exception as e:
                    logger.error(f"Local service health check failed: {e}")
                    self.local_service = None  # Reset connection so it will reconnect next time
                    raise Exception(f"Local service not healthy: {e}")
            except Exception as e:
                logger.error(f"Service health check failed: {e}")
                logger.info("Attempting to clean up and rerun setup...")

                # Clean up everything properly
                try:
                    # First, clean up the cloud service
                    await self.cleanup_cloud_service()

                    # Then disconnect from servers
                    if self.cloud_server:
                        await self.cloud_server.disconnect()
                    if self.local_server:
                        await self.local_server.disconnect()
                    if self.setup_task:
                        self.setup_task.cancel()  # Cancel the previous setup task
                except Exception as disconnect_error:
                    logger.error(f"Error during cleanup: {disconnect_error}")
                finally:
                    self.cloud_server = None
                    self.cloud_service = None
                    self.local_server = None
                    self.local_service = None
                    self.mirrored_methods.clear()

                # Retry setup with exponential backoff
                retry_count = 0
                max_retries = 50
                base_delay = 10

                while retry_count < max_retries:
                    try:
                        delay = base_delay * (2 ** min(retry_count, 5))  # Cap at 32 * base_delay
                        logger.info(f"Retrying setup in {delay} seconds (attempt {retry_count + 1}/{max_retries})")
                        await asyncio.sleep(delay)

                        # Rerun the setup method
                        self.setup_task = asyncio.create_task(self.setup())
                        await self.setup_task
                        logger.info("Setup successful after reconnection")
                        break  # Exit the loop if setup is successful
                    except Exception as setup_error:
                        retry_count += 1
                        logger.error(f"Failed to rerun setup (attempt {retry_count}/{max_retries}): {setup_error}")
                        if retry_count >= max_retries:
                            logger.error("Max retries reached, giving up on setup")
                            await asyncio.sleep(60)  # Wait longer before next health check cycle
                            break

            await asyncio.sleep(10)  # Check more frequently (was 30)

    async def start_hypha_service(self, server):
        """Start the Hypha service with dynamically mirrored methods"""
        self.cloud_server = server

        # Ensure we have a connection to the local service
        if self.local_service is None:
            logger.info("Local service not connected, attempting to connect before creating mirror methods")
            success = await self.connect_to_local_service()
            if not success:
                raise Exception("Cannot start Hypha service without local service connection")

        # Get the mirrored methods from the current local service
        self.mirrored_methods = self._get_mirrored_methods()

        # Base service configuration with core methods
        service_config = {
            "name": "Mirror Microscope Control Service",
            "id": self.cloud_service_id,
            "config": {
                "visibility": "protected",
                "run_in_executor": True
            },
            "type": "echo",
            "ping": self.ping,
        }

        # Add all mirrored methods to the service configuration
        service_config.update(self.mirrored_methods)

        # Register the service
        self.cloud_service = await server.register_service(service_config)

        logger.info(
            f"Mirror service (service_id={self.cloud_service_id}) started successfully with {len(self.mirrored_methods)} mirrored methods, available at {self.cloud_server_url}/services"
        )

        logger.info(f'You can use this service using the service id: {self.cloud_service.id}')
        id = self.cloud_service.id.split(":")[1]

        logger.info(f"You can also test the service via the HTTP proxy: {self.cloud_server_url}/{server.config.workspace}/services/{id}")

    async def start_webrtc_service(self, server, webrtc_service_id_arg):
        self.webrtc_service_id = webrtc_service_id_arg

        async def on_init(peer_connection):
            logger.info("WebRTC peer connection initialized")
            # Mark as connected when peer connection starts
            self.webrtc_connected = True

            # Create data channel for metadata transmission
            self.metadata_data_channel = peer_connection.createDataChannel("metadata", ordered=True)
            logger.info("Created metadata data channel")

            @self.metadata_data_channel.on("open")
            def on_data_channel_open():
                logger.info("Metadata data channel opened")

            @self.metadata_data_channel.on("close")
            def on_data_channel_close():
                logger.info("Metadata data channel closed")

            @self.metadata_data_channel.on("error")
            def on_data_channel_error(error):
                logger.error(f"Metadata data channel error: {error}")

            @peer_connection.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"WebRTC connection state changed to: {peer_connection.connectionState}")
                if peer_connection.connectionState in ["closed", "failed", "disconnected"]:
                    # Mark as disconnected
                    self.webrtc_connected = False
                    self.metadata_data_channel = None
                    self.local_service.off_illumination()
                    logger.info("Illumination closed")
                    if self.video_track and self.video_track.running:
                        logger.info(f"Connection state is {peer_connection.connectionState}. Stopping video track.")
                        self.video_track.stop()
                elif peer_connection.connectionState in ["connected"]:
                    # Mark as connected
                    self.webrtc_connected = True

            @peer_connection.on("track")
            def on_track(track):
                logger.info(f"Track {track.kind} received from client")

                if self.video_track and self.video_track.running:
                    self.video_track.stop()

                # Ensure local_service is available before creating video track
                if self.local_service is None:
                    logger.error("Cannot create video track: local_service is not available")
                    return

                try:
                    self.local_service.on_illumination()
                    logger.info("Illumination opened")
                    self.video_track = MicroscopeVideoTrack(self.local_service, self)
                    peer_connection.addTrack(self.video_track)
                    logger.info("Added MicroscopeVideoTrack to peer connection")
                except Exception as e:
                    logger.error(f"Failed to create video track: {e}")
                    return

                @track.on("ended")
                def on_ended():
                    logger.info(f"Client track {track.kind} ended")
                    self.local_service.off_illumination()
                    logger.info("Illumination closed")
                    if self.video_track:
                        logger.info("Stopping MicroscopeVideoTrack.")
                        self.video_track.stop()  # Now synchronous
                        self.video_track = None
                    self.metadata_data_channel = None

        ice_servers = await self.fetch_ice_servers()
        if not ice_servers:
            logger.warning("Using fallback ICE servers")
            ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

        try:
            await register_rtc_service(
                server,
                service_id=self.webrtc_service_id,
                config={
                    "visibility": "protected",
                    "ice_servers": ice_servers,
                    "on_init": on_init,
                },
            )
            logger.info(f"WebRTC service registered with id: {self.webrtc_service_id}")
        except Exception as e:
            logger.error(f"Failed to register WebRTC service ({self.webrtc_service_id}): {e}")
            if "Service already exists" in str(e):
                logger.info(f"WebRTC service {self.webrtc_service_id} already exists. Attempting to retrieve it.")
                try:
                    _ = await server.get_service(self.webrtc_service_id)
                    logger.info(f"Successfully retrieved existing WebRTC service: {self.webrtc_service_id}")
                except Exception as get_e:
                    logger.error(f"Failed to retrieve existing WebRTC service {self.webrtc_service_id}: {get_e}")
                    raise
            else:
                raise

    async def setup(self):
        # Connect to cloud workspace
        logger.info(f"Connecting to cloud workspace {self.cloud_workspace} at {self.cloud_server_url}")
        server = await connect_to_server({
            "server_url": self.cloud_server_url,
            "token": self.cloud_token,
            "workspace": self.cloud_workspace,
            "ping_interval": None
        })

        # Connect to local service first (needed to get available methods)
        logger.info("Connecting to local service before setting up mirror service")
        success = await self.connect_to_local_service()
        if not success or self.local_service is None:
            raise Exception("Failed to connect to local service during setup")

        # Verify local service is working
        try:
            ping_result = await asyncio.wait_for(self.local_service.ping(), timeout=10)
            if ping_result != "pong":
                raise Exception(f"Local service verification failed: {ping_result}")
            logger.info("Local service connection verified successfully")
        except Exception as e:
            logger.error(f"Local service verification failed: {e}")
            raise Exception(f"Local service not responding properly: {e}")

        # Small delay to ensure local service is fully ready
        await asyncio.sleep(1)

        # Start the cloud service with mirrored methods
        logger.info("Starting cloud service with mirrored methods")
        await self.start_hypha_service(server)

        # Start the WebRTC service
        self.webrtc_service_id = f"video-track-{self.local_service_id}"
        logger.info(f"Starting WebRTC service with id: {self.webrtc_service_id}")
        await self.start_webrtc_service(server, self.webrtc_service_id)

        logger.info("Setup completed successfully")

    def ping(self):
        """Ping function for health checks"""
        return "pong"

    async def fetch_ice_servers(self):
        """Fetch ICE servers from the coturn service"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://ai.imjoy.io/public/services/coturn/get_rtc_ice_servers') as response:
                    if response.status == 200:
                        ice_servers = await response.json()
                        logger.info("Successfully fetched ICE servers")
                        return ice_servers
                    else:
                        logger.warning(f"Failed to fetch ICE servers, status: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching ICE servers: {e}")
            return None

    def start_video_streaming(self, context=None):
        """Start WebRTC video streaming"""
        try:
            if not self.is_streaming:
                self.is_streaming = True
                logger.info("Video streaming started")
                return {"status": "streaming_started", "message": "WebRTC video streaming has been started"}
            else:
                return {"status": "already_streaming", "message": "Video streaming is already active"}
        except Exception as e:
            logger.error(f"Failed to start video streaming: {e}")
            raise e

    def stop_video_streaming(self, context=None):
        """Stop WebRTC video streaming"""
        try:
            if self.is_streaming:
                self.is_streaming = False
                if self.video_track:
                    self.video_track.running = False
                logger.info("Video streaming stopped")
                return {"status": "streaming_stopped", "message": "WebRTC video streaming has been stopped"}
            else:
                return {"status": "not_streaming", "message": "Video streaming is not currently active"}
        except Exception as e:
            logger.error(f"Failed to stop video streaming: {e}")
            raise e

    async def set_video_fps(self, fps=5, context=None):
        """Special method to set video FPS for both WebRTC and local service"""
        try:
            if self.local_service is None:
                await self.connect_to_local_service()

            # Update WebRTC video track FPS if active
            if self.video_track and self.video_track.running:
                old_webrtc_fps = self.video_track.fps
                self.video_track.fps = fps
                logger.info(f"WebRTC video track FPS updated from {old_webrtc_fps} to {fps}")

            # Forward call to local service if it has this method
            if hasattr(self.local_service, 'set_video_fps'):
                result = await self.local_service.set_video_fps(fps)
                return result
            else:
                logger.warning("Local service does not have set_video_fps method")
                return {"status": "webrtc_only", "message": f"WebRTC FPS set to {fps}, local service method not available"}

        except Exception as e:
            logger.error(f"Failed to set video FPS: {e}")
            raise e
