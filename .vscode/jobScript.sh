!/bin/bash -l
#SBATCH -J Singularity_Jupyter
# SBATCH -N 1 # vim Nodes
#SBATCH -n 1 # Tasks
#SBATCH -c 2 # Cores assigned to each task
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH --mail-user=<firstname>.<lastname>@uni.lu
#SBATCH --mail-type=BEGIN,END
si
module load tools/Singularity
# Avoid to modify your current jupyter config
export JUPYTER_CONFIG_DIR="$HOME/jupyter_sing/$SLURM_JOBID/"
export JUPYTER_PATH="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_path"
export JUPYTER_DATA_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_data"
export JUPYTER_RUNTIME_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_runtime"
export IPYTHONDIR="$HOME/ipython_sing/$SLURM_JOBID"
mkdir -p $IPYTHONDIR

export IP_ADDRESS=$(hostname -I | awk '{print $1}')

echo "On your laptop: ssh -p 8022 -NL 8889:${IP_ADDRESS}:8889 ${USER}@access-${ULHPC_CLUSTER}.uni.lu " 
singularity instance start jupyter.sif jupyter
singularity exec instance://jupyter jupyter \
    notebook --ip ${IP_ADDRESS} --no-browser --port 8889 &
pid=$!
sleep 5s
singularity exec instance://jupyter  jupyter notebook list
wait $pid
echo "Stopping instance"
singularity instance stop jupyter

# export VERSION="v3.8.1"
export VERSION="v3.6.4"
git clone https://github.com/hpcng/singularity.git
cd singularity
git checkout ${VERSION}

./mconfig && \
    make -C builddir && \
    sudo make -C builddir uninstall

可能的问题：

    singularity 版本不兼容，导致