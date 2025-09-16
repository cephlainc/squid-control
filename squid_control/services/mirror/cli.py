"""
Command-line interface for the mirror service.

This module provides the main entry point for running the mirror service
with command-line arguments.
"""

import argparse
import asyncio
import traceback

from .mirror_service import MirrorMicroscopeService


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Mirror service for Squid microscope control.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default service IDs
  python -m squid_control.services.mirror
  
  # Run with custom service IDs
  python -m squid_control.services.mirror \\
    --cloud-service-id "mirror-microscope-control-squid-2" \\
    --local-service-id "microscope-control-squid-2"
  
  # Run with custom local server URL
  python -m squid_control.services.mirror \\
    --local-server-url "http://localhost:9527" \\
    --local-service-id "microscope-control-squid-1"
        """
    )

    parser.add_argument(
        "--cloud-service-id",
        default="mirror-microscope-control-squid-1",
        help="ID for the cloud service (default: mirror-microscope-control-squid-1)"
    )

    parser.add_argument(
        "--local-service-id",
        default="microscope-control-squid-1",
        help="ID for the local service (default: microscope-control-squid-1)"
    )

    parser.add_argument(
        "--cloud-server-url",
        default="https://hypha.aicell.io",
        help="Cloud server URL (default: https://hypha.aicell.io)"
    )

    parser.add_argument(
        "--cloud-workspace",
        default="reef-imaging",
        help="Cloud workspace name (default: reef-imaging)"
    )

    parser.add_argument(
        "--local-server-url",
        default="http://reef.dyn.scilifelab.se:9527",
        help="Local server URL (default: http://reef.dyn.scilifelab.se:9527)"
    )

    parser.add_argument(
        "--log-file",
        default="mirror_squid_control_service.log",
        help="Log file path (default: mirror_squid_control_service.log)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser


def main():
    """Main entry point for the mirror service"""
    parser = create_parser()
    args = parser.parse_args()

    # Create and configure the mirror service
    mirror_service = MirrorMicroscopeService()

    # Override configuration with command-line arguments
    mirror_service.cloud_service_id = args.cloud_service_id
    mirror_service.local_service_id = args.local_service_id
    mirror_service.cloud_server_url = args.cloud_server_url
    mirror_service.cloud_workspace = args.cloud_workspace
    mirror_service.local_server_url = args.local_server_url

    # Set up logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    print("Starting mirror service:")
    print(f"  Cloud Service ID: {mirror_service.cloud_service_id}")
    print(f"  Local Service ID: {mirror_service.local_service_id}")
    print(f"  Cloud Server: {mirror_service.cloud_server_url}")
    print(f"  Cloud Workspace: {mirror_service.cloud_workspace}")
    print(f"  Local Server: {mirror_service.local_server_url}")
    print(f"  Log File: {args.log_file}")
    print()

    # Run the service
    loop = asyncio.get_event_loop()

    async def run_service():
        try:
            mirror_service.setup_task = asyncio.create_task(mirror_service.setup())
            await mirror_service.setup_task

            # Start the health check task
            asyncio.create_task(mirror_service.check_service_health())

            # Keep the service running
            while True:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            print("\nShutting down mirror service...")
        except Exception as e:
            print(f"Error running mirror service: {e}")
            traceback.print_exc()
        finally:
            # Cleanup
            try:
                if mirror_service.cloud_service:
                    await mirror_service.cleanup_cloud_service()
                if mirror_service.cloud_server:
                    await mirror_service.cloud_server.disconnect()
                if mirror_service.local_server:
                    await mirror_service.local_server.disconnect()
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")

    try:
        loop.run_until_complete(run_service())
    except KeyboardInterrupt:
        print("\nMirror service stopped by user")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
