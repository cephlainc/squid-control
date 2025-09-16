#!/usr/bin/env python3
"""
Main entry point for the squid_control module.
This allows users to run: python -m squid_control [options]
"""

import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands"""
    parser = argparse.ArgumentParser(
        description="Squid Microscope Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run main microscope service
  python -m squid_control microscope --simulation --verbose
  
  # Run mirror service
  python -m squid_control mirror --cloud-service-id "mirror-microscope-control-squid-2" --local-service-id "microscope-control-squid-2"
  
  # Run specific service directly
  python -m squid_control.services.mirror --cloud-service-id "mirror-microscope-control-squid-2"
        """
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    # Microscope service subcommand
    microscope_parser = subparsers.add_parser(
        "microscope",
        help="Run the main microscope control service"
    )
    microscope_parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode"
    )
    microscope_parser.add_argument(
        "--local",
        action="store_true",
        help="Run in local mode only"
    )
    microscope_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    # Mirror service subcommand
    mirror_parser = subparsers.add_parser(
        "mirror",
        help="Run the mirror service for cloud-to-local proxy"
    )
    mirror_parser.add_argument(
        "--cloud-service-id",
        default="mirror-microscope-control-squid-1",
        help="ID for the cloud service"
    )
    mirror_parser.add_argument(
        "--local-service-id",
        default="microscope-control-squid-1",
        help="ID for the local service"
    )
    mirror_parser.add_argument(
        "--cloud-server-url",
        default="https://hypha.aicell.io",
        help="Cloud server URL"
    )
    mirror_parser.add_argument(
        "--cloud-workspace",
        default="reef-imaging",
        help="Cloud workspace name"
    )
    mirror_parser.add_argument(
        "--local-server-url",
        default="http://reef.dyn.scilifelab.se:9527",
        help="Local server URL"
    )
    mirror_parser.add_argument(
        "--log-file",
        default="mirror_squid_control_service.log",
        help="Log file path"
    )
    mirror_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser


def main():
    """Main entry point with subcommand routing"""
    parser = create_parser()
    args = parser.parse_args()

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "microscope":
            # Import locally to avoid circular imports
            # Create a new argument parser for the microscope service
            # that matches what start_hypha_service.py expects
            import argparse as ap

            from .start_hypha_service import main as microscope_main
            microscope_parser = ap.ArgumentParser()
            microscope_parser.add_argument("--simulation", action="store_true", default=False)
            microscope_parser.add_argument("--local", action="store_true", default=False)
            microscope_parser.add_argument("--verbose", "-v", action="count")

            # Convert our args to the format expected by start_hypha_service.py
            microscope_args = []
            if args.simulation:
                microscope_args.append("--simulation")
            if args.local:
                microscope_args.append("--local")
            if args.verbose:
                microscope_args.append("--verbose")

            # Temporarily replace sys.argv to pass arguments to microscope_main
            original_argv = sys.argv
            sys.argv = ["start_hypha_service.py"] + microscope_args

            try:
                microscope_main()
            finally:
                # Restore original sys.argv
                sys.argv = original_argv

        elif args.command == "mirror":
            # Import locally to avoid circular imports
            import asyncio
            import traceback

            from .services.mirror.cli import MirrorMicroscopeService

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

        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)

    except ImportError as e:
        print(f"Error importing required module: {e}")
        print("Make sure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running {args.command} service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
