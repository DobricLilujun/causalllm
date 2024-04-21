#!/bin/bash -l
#SBATCH -J Singularity_Jupyter_parallel_cuda
#SBATCH -N 1 # Nodes
#SBATCH -n 1 # Tasks
#SBATCH -c 4 # Cores assigned to each tasks
#SBATCH --time=0-01:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --qos=normal
#SBATCH --mail-user=<firstname>.<lastname>@uni.lu
#SBATCH --mail-type=BEGIN,END


# module load tools/Singularity

export MY_ENV="causalLLM"
export VENV="$HOME/.envs/venv_cuda_${MY_ENV}"
export JUPYTER_CONFIG_DIR="$HOME/jupyter_singularity/$MY_ENV/"
export JUPYTER_PATH="$VENV/share/jupyter":"$HOME/jupyter_singularity/$MY_ENV/jupyter_path"
export JUPYTER_DATA_DIR="$HOME/jupyter_singularity/$MY_ENV/jupyter_data"
export JUPYTER_RUNTIME_DIR="$HOME/jupyter_singularity/$MY_ENV/jupyter_runtime"
export IPYTHONDIR="$HOME/ipython_singularity/$MY_ENV"

mkdir -p $JUPYTER_CONFIG_DIR
mkdir -p $IPYTHONDIR

export IP_ADDRESS=$(hostname -I | awk '{print $1}')
export XDG_RUNTIME_DIR=""

#create a new ipython profile appended with the job id number
profile=job_${causalLLM}


# echo "On your laptop: ssh -p 8022 ${USER}@access-${ULHPC_CLUSTER}.uni.lu ${USER}@<node-name> -NL 8889:localhost:8889" 
# echo "Replace <node-name> with the compute node name received by slurm. Command squeue -u ${USER} to see it. Example: "aion-0085" or "iris-0112".

if [ ! -d "$VENV" ];then
    # For some reasons, there is an issue with venv -- using virtualenv instead
    singularity exec --nv jupyter_kernel_cuda.sif python3 -m virtualenv $VENV --system-site-packages
    singularity run --nv jupyter_kernel_cuda.sif $VENV "python3 -m pip install --upgrade pip" 
    singularity run --nv jupyter_kernel_cuda.sif $VENV "python3 -m pip install --upgrade pip fastapi transformers torch pandas accelerate bitsandbytes"
    singularity run --nv jupyter_kernel_cuda.sif $VENV "python3 -m ipykernel install --sys-prefix --name $MY_ENV --display-name $MY_ENV"

fi

singularity run --nv jupyter_kernel_cuda.sif $VENV "ipython profile create --parallel ${profile}"
singularity run --nv jupyter_kernel_cuda.sif $VENV "jupyter nbextension enable --py ipyparallel"
singularity run --nv jupyter_kernel_cuda.sif $VENV "jupyter notebook --ip localhost --no-browser --port 8889" &
sleep 5s
singularity run --nv jupyter_kernel_cuda.sif $VENV "jupyter notebook list"
singularity run --nv jupyter_kernel_cuda.sif $VENV "jupyter --paths"
singularity run --nv jupyter_kernel_cuda.sif $VENV "jupyter kernelspec list"

singularity run --nv jupyter_kernel_cuda.sif $VENV "python script/preparation.py"

wait