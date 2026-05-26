"""Script relocated to Data_Cleaned/pipeline_tools/add_transcripts.py.

This stub preserves the original CLI entrypoint and delegates execution to the
copy in `Data_Cleaned/pipeline_tools/`.
"""
import runpy
from pathlib import Path


if __name__ == "__main__":
    new_path = Path(__file__).parent / "pipeline_tools" / "add_transcripts.py"
    runpy.run_path(str(new_path), run_name="__main__")
