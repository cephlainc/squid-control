import asyncio
import os
import uuid

import pytest
import pytest_asyncio
from hypha_rpc import connect_to_server

# Import the mirror service components
from squid_control.services.mirror.mirror_service import MirrorMicroscopeService
from squid_control.services.mirror.video_track import MicroscopeVideoTrack

# Import the real microscope service for testing
from squid_control.start_hypha_service import MicroscopeHyphaService

# Mark all tests in this module as asyncio and integration tests
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

# Test configuration
TEST_SERVER_URL = "https://hypha.aicell.io"
TEST_WORKSPACE = "agent-lens"
TEST_TIMEOUT = 120  # seconds

# Test constants
TEST_FPS = 10
TEST_STAGE_X = 15
TEST_STAGE_Y = 25
TEST_STAGE_Z = 10
TEST_VIDEO_WIDTH = 750
TEST_VIDEO_HEIGHT = 750
TEST_DEFAULT_FPS = 5


class SimpleTestDataStore:
    """Simple test datastore that doesn't require external services."""

    def __init__(self):
        self.storage = {}
        self.counter = 0

    def put(self, file_type, data, filename, description=""):
        self.counter += 1
        file_id = f"test_file_{self.counter}"
        self.storage[file_id] = {
            'type': file_type,
            'data': data,
            'filename': filename,
            'description': description
        }
        return file_id

    def get_url(self, file_id):
        if file_id in self.storage:
            return f"https://test-storage.example.com/{file_id}"
        return None


@pytest_asyncio.fixture(scope="function")
async def test_server_connection():
    """Create a connection to the test server."""
    token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
    if not token:
        pytest.skip("AGENT_LENS_WORKSPACE_TOKEN not set - skipping integration test")

    # Use the existing workspace since the token is tied to it
    server = await connect_to_server({
        "server_url": TEST_SERVER_URL,
        "token": token,
        "workspace": TEST_WORKSPACE,
        "ping_interval": None
    })

    yield server

    # Cleanup
    try:
        await server.disconnect()
    except Exception:
        pass  # Ignore cleanup errors


async def _create_test_microscope(test_id):
    """Helper function to create and configure a test microscope."""
    microscope = MicroscopeHyphaService(is_simulation=True, is_local=False)
    microscope.service_id = test_id
    microscope.login_required = False  # Disable auth for tests
    microscope.authorized_emails = None

    # Create a simple datastore for testing
    microscope.datastore = SimpleTestDataStore()

    # Disable similarity search service to avoid OpenAI costs
    microscope.similarity_search_svc = None

    # Override setup method to avoid connecting to external services during tests
    async def mock_setup():
        pass
    microscope.setup = mock_setup

    return microscope


async def _cleanup_microscope_service(microscope, server):
    """Helper function to clean up microscope service."""
    # Stop video buffering if it's running to prevent event loop errors
    if microscope and hasattr(microscope, 'stop_video_buffering'):
        try:
            if microscope.video_buffering_active:
                microscope.stop_video_buffering()
                await asyncio.sleep(0.5)  # Give time for cleanup
        except Exception:
            pass  # Ignore cleanup errors

    # Unregister the service
    if microscope and hasattr(microscope, 'service_id'):
        try:
            await server.unregister_service(microscope.service_id)
        except Exception:
            pass  # Ignore cleanup errors


@pytest_asyncio.fixture(scope="function")
async def real_microscope_service(test_server_connection):
    """Create a real microscope service for testing."""
    server = test_server_connection
    microscope = None

    try:
        # Create unique service ID for this test
        test_id = f"test-mirror-local-microscope-{uuid.uuid4().hex[:8]}"

        # Create real microscope instance in simulation mode
        microscope = await _create_test_microscope(test_id)

        # Register the service
        await microscope.start_hypha_service(server, test_id)

        # Get the registered service to test against
        microscope_service = await server.get_service(test_id)

        # Verify service is working
        ping_result = await microscope_service.ping()
        assert ping_result == "pong"

        yield microscope, microscope_service, test_id

    finally:
        # Comprehensive cleanup
        await _cleanup_microscope_service(microscope, server)


@pytest_asyncio.fixture(scope="function")
async def real_mirror_service(test_server_connection, real_microscope_service):
    """Create a real mirror service connected to a real microscope service."""
    server = test_server_connection
    microscope, microscope_service, local_service_id = real_microscope_service

    # Create mirror service
    mirror_service = MirrorMicroscopeService()

    # Configure for testing
    mirror_service.cloud_server_url = TEST_SERVER_URL
    mirror_service.cloud_workspace = server.config.workspace
    mirror_service.cloud_token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
    mirror_service.cloud_service_id = f"test-mirror-{uuid.uuid4().hex[:8]}"

    # Set up the local service connection to point to our real microscope service
    mirror_service.local_service = microscope_service
    mirror_service.local_server = server  # Use same server for simplicity in testing
    mirror_service.local_service_id = local_service_id

    yield mirror_service

    # Cleanup
    try:
        await mirror_service.cleanup_cloud_service()
        if mirror_service.cloud_server and mirror_service.cloud_server != server:
            await mirror_service.cloud_server.disconnect()
    except Exception:
        pass  # Ignore cleanup errors


class TestMirrorService:
    """Test cases for the MirrorMicroscopeService."""

    async def test_service_initialization(self):
        """Test that the service initializes correctly."""
        service = MirrorMicroscopeService()

        assert service.cloud_server_url == "https://hypha.aicell.io"
        assert service.cloud_workspace == "reef-imaging"
        assert service.local_service_id == "microscope-control-squid-1"
        assert service.cloud_service_id == "mirror-microscope-control-squid-1"
        assert service.mirrored_methods == {}
        assert not service.is_streaming
        assert not service.webrtc_connected

    async def test_ping_method(self):
        """Test the ping health check method."""
        service = MirrorMicroscopeService()
        result = service.ping()
        assert result == "pong"

    async def test_local_service_connection_real(self, real_microscope_service):
        """Test local service connection with real service."""
        microscope, microscope_service, service_id = real_microscope_service

        service = MirrorMicroscopeService()
        service.local_service = microscope_service

        # Test ping through the mirror service's local connection
        result = await service.local_service.ping()
        assert result == "pong"

        # Test a real method call
        config_result = await service.local_service.get_microscope_configuration()
        assert isinstance(config_result, dict)
        assert "success" in config_result
        assert config_result["success"] is True

    async def test_mirror_method_creation_real(self, real_microscope_service):
        """Test that mirror methods are created correctly from real service."""
        microscope, microscope_service, service_id = real_microscope_service

        service = MirrorMicroscopeService()
        service.local_service = microscope_service

        mirrored_methods = service._get_mirrored_methods()

        # Check that expected methods are mirrored
        expected_methods = [
            "ping",

            "set_illumination",

            "get_video_frame",
            "get_microscope_configuration"
        ]
        for method in expected_methods:
            expected_error = f"Expected method '{method}' not found in mirrored methods"
            assert method in mirrored_methods, expected_error

        # Test that a mirrored method works
        mirror_ping = mirrored_methods["ping"]
        result = await mirror_ping()
        assert result == "pong"

        # Test a more complex mirrored method
        mirror_config = mirrored_methods["get_microscope_configuration"]
        config_result = await mirror_config()
        assert isinstance(config_result, dict)
        assert "success" in config_result
        assert config_result["success"] is True

    async def test_schema_preservation_real(self, real_microscope_service):
        """Test that schema information is properly preserved in mirror methods."""
        microscope, microscope_service, service_id = real_microscope_service

        service = MirrorMicroscopeService()
        service.local_service = microscope_service

        mirrored_methods = service._get_mirrored_methods()

        # Test schema preservation for key methods
        test_methods = ['move_by_distance', 'get_status', 'ping', 'set_illumination']

        for method_name in test_methods:
            if method_name in mirrored_methods:
                mirror_method = mirrored_methods[method_name]
                original_method = getattr(microscope_service, method_name)

                # Check if original method has schema
                if hasattr(original_method, '__schema__'):
                    original_schema = getattr(original_method, '__schema__')

                    # Check if mirror method has schema
                    assert hasattr(mirror_method, '__schema__'), f"Mirror method {method_name} missing __schema__ attribute"

                    mirror_schema = getattr(mirror_method, '__schema__')

                    # Schema should be preserved (not None)
                    if original_schema is not None:
                        assert mirror_schema is not None, f"Mirror method {method_name} has None schema when original has schema"

                        # Check that key schema elements are preserved
                        assert mirror_schema.get('name') == original_schema.get('name'), f"Schema name mismatch for {method_name}"
                        assert mirror_schema.get('description') == original_schema.get('description'), f"Schema description mismatch for {method_name}"

                        # Check that parameters are preserved
                        original_params = original_schema.get('parameters', {})
                        mirror_params = mirror_schema.get('parameters', {})
                        assert mirror_params == original_params, f"Schema parameters mismatch for {method_name}"

                        # Check that docstring is preserved
                        if original_schema.get('description'):
                            assert mirror_method.__doc__ == original_schema.get('description'), f"Docstring mismatch for {method_name}"

                        print(f"✅ Schema preserved for {method_name}: {mirror_schema.get('name')} - {mirror_schema.get('description')[:50]}...")
                    else:
                        print(f"⚠️ Original method {method_name} has None schema")
                else:
                    print(f"ℹ️ Original method {method_name} has no schema attribute")
            else:
                print(f"❌ Method {method_name} not found in mirrored methods")

        # Test specific schema details for a well-known method
        if 'move_by_distance' in mirrored_methods:
            mirror_method = mirrored_methods['move_by_distance']
            assert hasattr(mirror_method, '__schema__')

            schema = getattr(mirror_method, '__schema__')
            assert schema is not None

            # Check specific parameter details
            parameters = schema.get('parameters', {})
            properties = parameters.get('properties', {})

            # Check that x, y, z parameters are preserved
            assert 'x' in properties, "x parameter missing from schema"
            assert 'y' in properties, "y parameter missing from schema"
            assert 'z' in properties, "z parameter missing from schema"

            # Check parameter descriptions
            x_param = properties.get('x', {})
            assert 'description' in x_param, "x parameter missing description"
            assert 'unit: milimeter' in x_param['description'], "x parameter description incomplete"

            print("✅ Detailed schema verification passed for move_by_distance")

    async def test_video_streaming_controls(self):
        """Test video streaming start/stop controls."""
        service = MirrorMicroscopeService()

        # Test starting video streaming
        result = service.start_video_streaming()
        assert result["status"] == "streaming_started"
        assert service.is_streaming

        # Test starting when already streaming
        result = service.start_video_streaming()
        assert result["status"] == "already_streaming"

        # Test stopping video streaming
        result = service.stop_video_streaming()
        assert result["status"] == "streaming_stopped"
        assert not service.is_streaming

        # Test stopping when not streaming
        result = service.stop_video_streaming()
        assert result["status"] == "not_streaming"

    async def test_hypha_service_registration_real(self, real_mirror_service, test_server_connection):
        """Test registering the mirror service with Hypha using real services."""
        service = real_mirror_service
        server = test_server_connection

        # Start the Hypha service
        await service.start_hypha_service(server)

        # Verify service is registered
        assert service.cloud_service is not None
        assert service.cloud_server == server
        assert len(service.mirrored_methods) > 0

        # Get the registered service from the server to test it
        registered_service = await server.get_service(service.cloud_service_id)

        # Test that we can call the service
        ping_result = await registered_service.ping()
        assert ping_result == "pong"

    async def test_mirrored_method_calls_real(self, real_mirror_service, test_server_connection):
        """Test that mirrored methods work through the cloud service with real services."""
        service = real_mirror_service
        server = test_server_connection

        # Start the Hypha service
        await service.start_hypha_service(server)

        # Get the registered service from the server to test it
        registered_service = await server.get_service(service.cloud_service_id)

        # Test calling a mirrored method through the cloud service
        config_result = await registered_service.get_microscope_configuration()
        assert isinstance(config_result, dict)
        assert "success" in config_result
        assert config_result["success"] is True

        # Test stage status through the mirror service
        status = await registered_service.get_status()
        assert isinstance(status, dict)
        assert "current_x" in status and "current_y" in status and "current_z" in status

        # Test moving stage (small movement in simulation)
        move_result = await registered_service.move_by_distance(x=0.1, y=0.1, z=0.0)
        assert isinstance(move_result, dict)
        # In simulation mode, move_by_distance typically returns a status


class TestMicroscopeVideoTrack:
    """Test cases for the MicroscopeVideoTrack."""

    async def test_video_track_initialization_real(self, real_microscope_service):
        """Test video track initialization with real service."""
        microscope, microscope_service, service_id = real_microscope_service

        track = MicroscopeVideoTrack(microscope_service)

        assert track.local_service == microscope_service
        assert track.fps == TEST_DEFAULT_FPS
        assert track.frame_width == TEST_VIDEO_WIDTH
        assert track.frame_height == TEST_VIDEO_HEIGHT
        assert track.running
        assert track.count == 0

    async def test_video_track_initialization_error(self):
        """Test video track initialization with None service."""
        with pytest.raises(ValueError, match="local_service cannot be None"):
            MicroscopeVideoTrack(None)

    async def test_video_track_stop_real(self, real_microscope_service):
        """Test stopping the video track with real service."""
        microscope, microscope_service, service_id = real_microscope_service

        track = MicroscopeVideoTrack(microscope_service)

        assert track.running
        track.stop()
        assert not track.running


class TestMirrorServiceIntegration:
    """Integration tests for real mirror service functionality."""

    async def test_end_to_end_mirror_functionality(self, real_mirror_service, test_server_connection):
        """Test complete end-to-end functionality of mirror service."""
        service = real_mirror_service
        server = test_server_connection

        # Start the mirror service
        await service.start_hypha_service(server)

        # Get the cloud-facing service
        cloud_service = await server.get_service(service.cloud_service_id)

        # Test basic functionality
        ping_result = await cloud_service.ping()
        assert ping_result == "pong"

        # Test microscope configuration retrieval
        config = await cloud_service.get_microscope_configuration()
        assert isinstance(config, dict)
        assert "success" in config
        assert config["success"] is True

        # Test stage operations
        status = await cloud_service.get_status()
        assert isinstance(status, dict)

        # Test illumination control
        await cloud_service.set_illumination(channel=0, intensity=50)
        # The exact return format may vary, but it should not raise an exception

        # Test video frame acquisition
        frame_data = await cloud_service.get_video_frame(frame_width=512, frame_height=512)
        assert isinstance(frame_data, dict)
        assert "data" in frame_data  # Should contain image data

    async def test_mirror_service_error_handling(self, real_mirror_service, test_server_connection):
        """Test error handling in mirror service."""
        service = real_mirror_service
        server = test_server_connection

        # Start the mirror service
        await service.start_hypha_service(server)

        # Get the cloud-facing service
        cloud_service = await server.get_service(service.cloud_service_id)

        # Test calling a method that might fail (depends on implementation)
        try:
            # This should work in simulation mode, but test the error path
            result = await cloud_service.get_status()
            assert isinstance(result, dict)
        except Exception:
            # If it fails, make sure the error propagates properly
            pass

        # The service should still be responsive after any errors
        ping_result = await cloud_service.ping()
        assert ping_result == "pong"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
