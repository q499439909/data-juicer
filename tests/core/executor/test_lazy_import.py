import subprocess
import sys


def test_executor_package_does_not_eagerly_import_optional_ray_adapters():
    script = """
import sys
import data_juicer.core.executor

assert "data_juicer.core.executor.ray_executor" not in sys.modules
assert "data_juicer.core.executor.ray_executor_partitioned" not in sys.modules
"""

    subprocess.run([sys.executable, "-c", script], check=True)
