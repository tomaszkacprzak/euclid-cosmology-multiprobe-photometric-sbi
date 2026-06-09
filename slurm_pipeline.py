#!/usr/bin/env python3
import os
from pathlib import Path
from simple_slurm import Slurm


# Equivalent to workflow.launchDir
LAUNCH_DIR = Path(os.getcwd())
LOG_DIR = LAUNCH_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = LAUNCH_DIR / "config_euclidtr1v2p0multi.yaml"
DIR_OUT = LAUNCH_DIR / "euclidtr1v2p0multi"

def submit_probemaps_array(
    *,
    name,
    array,
    **kwargs,
):
    """
    Submit one Slurm job array for cosmogridv11.apps.run_probemaps.
    """
    slurm_args = {
        "job_name": name,
        "array": array,
        "cpus_per_task": 8,
        "mem": "15600M",
        "time": "12:00:00",
        "partition": "performance",
        "clusters": "cluster",
        "output": str(LOG_DIR / f"{name}.%A_%a.out"),
        "error": str(LOG_DIR / f"{name}.%A_%a.err"),
    }
    slurm_args.update(kwargs)
    slurm = Slurm(**slurm_args)

    
    command = f"""
    pixi run uv run python -m cosmogridv11.apps.run_probemaps main \\
        --config="{CONFIG_FILE}" \\
        --dir_out="{DIR_OUT}" \\
        --num_maps_per_index=10 \\
        --indices="$SLURM_ARRAY_TASK_ID" \\
        --verbosity=info
    """
    
    

    print(f'Submitting {name} command: {command}')
    job_id = slurm.sbatch(command)
    print(f"Submitted {name} jobid: {job_id}")
    return job_id



# pixi run uv run python -m cosmogridv11.apps.run_probemaps main --config=config_euclidtr1v2p0multi.yaml --dir_out=euclidtr1v2p0multi --num_maps_per_index=10 --indices="17" --verbosity=info
grid_job_id = submit_probemaps_array(name="euclid_proj_grid", array=range(17, 32))

# pixi run uv run python -m cosmogridv11.apps.run_probemaps main --config=config_euclidtr1v2p0multi.yaml --dir_out=euclidtr1v2p0multi --num_maps_per_index=10 --indices="0" --verbosity=info
fidu_job_id = submit_probemaps_array(name="euclid_proj_fidu", array=range(0, 10), dependency={"afterok": grid_job_id})

