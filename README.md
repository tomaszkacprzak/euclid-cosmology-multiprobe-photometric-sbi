# euclid-cosmology-multiprobe-photometric-sbi
Project 448


## Installation

1. Select your target platform. Currently we have scripts that work on Clariden and Calculon. Copy `pipeline.py` and `.toml` files to the workspace directory. On Clariden, you may also want to work on an interactive job to set things up; use `interactive.sh`.
2. Copy `clone.sh`, config `.yaml` files to your workspace.
3. Clone the member repositories to `repos` directory, run `sh clone.sh`.
4. Install the `uv` environment with all dependencies of workspace members:
  
   ```
   uv lock
   uv sync --all-extras --all-groups --all-packages
   ```

## Running jobs:

The main submission script is `pipeline.py`, which is meant to be edited freely by the user. It contains different steps of the analysis. Make sure to submit them one at the time, or explicitly chain them in the `pipeline.py`.
Then submit jobs like this:

```
uv run pipeline.py
```


