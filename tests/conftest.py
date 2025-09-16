import asyncio
import os
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

# Configure matplotlib environment BEFORE any matplotlib import
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Configure matplotlib to use non-interactive backend for testing
try:
    # Import matplotlib with explicit backend configuration
    import matplotlib
    matplotlib.use('Agg', force=True)  # Use non-interactive backend
    matplotlib.interactive(False)

    # Additional configuration to prevent type registration issues
    matplotlib.rcParams['backend'] = 'Agg'
    matplotlib.rcParams['interactive'] = False

    # Clear any existing type registrations that might cause conflicts
    try:
        import matplotlib.colors
        # Force re-registration of types to prevent conflicts
        matplotlib.colors._colors_full_map.clear()
    except Exception:
        pass

except Exception as e:
    print(f"Warning: Could not configure matplotlib backend: {e}")
    # If matplotlib fails to import, we'll continue without it for testing

def pytest_collection_modifyitems(config, items):
    """Modify test collection to ensure matplotlib is configured."""
    # No need to reconfigure matplotlib here since it's already done above
    pass

# Configure asyncio policy for better event loop management
def pytest_configure(config):
    """Configure pytest with asyncio settings."""
    # Register custom marks
    config.addinivalue_line("markers", "integration: mark test as integration test requiring external services")
    config.addinivalue_line("markers", "asyncio: mark test as asyncio-based test")
    config.addinivalue_line("markers", "cleanup: mark test as requiring cleanup of test directories")

    # Add command line options
    config.addinivalue_line("addopts", "--cleanup-docs: Clean up Documents folder before and after tests")

    # Suppress deprecation warnings from websockets and other libraries
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Suppress matplotlib warnings that commonly occur in CI
    warnings.filterwarnings("ignore", message=".*matplotlib.*")
    warnings.filterwarnings("ignore", message=".*_InterpolationType.*")

    # Matplotlib is already configured at module level, no need to reconfigure here

    # Set asyncio policy for consistent event loop handling
    if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    # Create a new event loop for the test session
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Configure the loop with proper executor
    loop.set_default_executor(ThreadPoolExecutor(max_workers=4))

    yield loop

    # Cleanup
    try:
        # Cancel all remaining tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()

        # Run until all tasks are cancelled
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

        # Close the loop
        loop.close()
    except Exception as e:
        print(f"Error during event loop cleanup: {e}")

@pytest.fixture(autouse=True, scope="function")
def cleanup_tasks():
    """Auto-cleanup fixture to ensure tasks are cleaned up after each test."""
    yield

    # Clean up any remaining tasks after each test
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
            if tasks:
                for task in tasks:
                    if not task.cancelled():
                        task.cancel()
    except RuntimeError:
        # No event loop available, nothing to clean up
        pass

def clean_test_experiments():
    """Clean up test experiment directories from the Documents folder."""
    # Check multiple possible locations for test experiments
    possible_paths = [
        Path.home() / "Documents",  # Actual Documents folder
        Path("~/Documents").expanduser(),  # Expanded ~/Documents
        Path("./~/Documents"),  # Relative path that might be used
        Path("/tmp/zarr_canvas"),  # Default ZARR_PATH
    ]
    
    # List of test experiment patterns to clean up
    test_patterns = [
        "test_experiment_*",
        "temp_exp_*", 
        "test_well_canvas_experiment*",
        "test_reset_experiment*",
        "project_*",
        "experiment_*",
        "invalid*",
        "default"
    ]
    
    cleaned_count = 0
    total_size = 0
    
    for base_path in possible_paths:
        if not base_path.exists():
            continue
            
        print(f"   🔍 Checking: {base_path}")
        
        for pattern in test_patterns:
            for item in base_path.glob(pattern):
                if item.is_dir():
                    try:
                        # Calculate directory size before deletion
                        dir_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                        total_size += dir_size
                        
                        shutil.rmtree(item)
                        cleaned_count += 1
                        size_mb = dir_size / (1024 * 1024)
                        print(f"   🧹 Cleaned up: {item.name} ({size_mb:.2f} MB) from {base_path}")
                    except Exception as e:
                        print(f"   ⚠️  Warning: Could not remove {item}: {e}")
    
    if cleaned_count > 0:
        total_mb = total_size / (1024 * 1024)
        print(f"   ✅ Cleaned up {cleaned_count} test experiment directories ({total_mb:.2f} MB)")
    else:
        print("   ℹ️  No test experiment directories found to clean up")

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_experiments_session(request):
    """Session-level cleanup: clean up before and after all tests."""
    # Check if cleanup is enabled via command line option
    cleanup_enabled = request.config.getoption("--cleanup-docs", default=False)
    
    if cleanup_enabled:
        print("\n🧹 Starting test session cleanup...")
        clean_test_experiments()
    
    yield
    
    if cleanup_enabled:
        print("\n🧹 Ending test session cleanup...")
        clean_test_experiments()

@pytest.fixture(autouse=True, scope="function")
def cleanup_test_experiments_function(request):
    """Function-level cleanup: clean up after each test function."""
    yield
    
    # Check if cleanup is enabled via command line option
    cleanup_enabled = request.config.getoption("--cleanup-docs", default=False)
    
    if cleanup_enabled:
        # Only clean up if we're running experiment-related tests
        # This prevents unnecessary cleanup for non-experiment tests
        import sys
        if any(keyword in sys.argv for keyword in ['experiment', 'well_canvas', 'test_squid_controller']):
            clean_test_experiments()

def pytest_addoption(parser):
    """Add command line options for test configuration."""
    parser.addoption(
        "--cleanup-docs",
        action="store_true",
        default=False,
        help="Clean up Documents folder before and after tests"
    )

def pytest_runtest_setup(item):
    """Setup for each test item."""
    # Skip integration tests if environment variable not set
    if "integration" in item.keywords:
        import os
        if not os.environ.get("AGENT_LENS_WORKSPACE_TOKEN"):
            pytest.skip("AGENT_LENS_WORKSPACE_TOKEN not set - skipping integration test")
