"""Fast-profile launch capture must accept the actual reviewed recipe."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def test_fast_launch_has_explicit_recipe_argument():
    file=ROOT/'scripts/drama3/capture_fast_launch.py'
    assert file.is_file(), 'fast-profile launch collector is missing'
    result=subprocess.run([sys.executable,str(file),'--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--recipe' in result.stdout
