prev_home=$(pwd)
srun --time=0:30:0 -n1 -c32 --mem-per-cpu=1950 --gpus-per-task=1 -A a0186 --mpi=pmix --network=disable_rdzv_get --environment=./edf.toml --partition=debug --pty bash -c "cd $prev_home; exec bash --rcfile /capstor/scratch/cscs/tomaszk/.bashrc -i"
