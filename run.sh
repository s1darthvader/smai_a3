#!/bin/bash
#SBATCH --job-name=Bangla_Train
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --constraint=2080ti
#SBATCH --time=96:00:00
#SBATCH --output=bangla_%j.out
#SBATCH --error=bangla_%j.err

# --- 1. SETTINGS & ENV SETUP ---
PROJECT_DIR=$HOME/smai_a3
ZIP_PATH=$PROJECT_DIR/dataset.zip
SCRATCH_DIR=/ssd_scratch/$USER/bangla_job_$SLURM_JOB_ID

module purge
module load u22/cuda/11.8

# Fixed path: environment is in your home directory
source $HOME/bangla_env/bin/activate

# --- 2. SETUP SCRATCH ---
echo ">>> Setting up scratch workspace..."
mkdir -p $SCRATCH_DIR

echo ">>> Copying dataset and script to scratch..."
# Fixed filenames
cp $ZIP_PATH $SCRATCH_DIR/
cp $PROJECT_DIR/main.py $SCRATCH_DIR/

cd $SCRATCH_DIR

echo ">>> Unzipping dataset..."
# Fixed zip name
unzip -q dataset.zip

# It extracts as 'BanglaLekha-Isolated', rename it to 'Dataset' so the python code finds 'Dataset/Images'
mv BanglaLekha-Isolated Dataset

# --- 3. EXECUTION ---
echo ">>> Starting BanglaLekha training on 4 GPUs with DDP..."

# CRITICAL: Prevent CPU thread contention in DDP. 40 total CPUs / 4 processes = 10 threads per process.
export OMP_NUM_THREADS=10

# Launch with torchrun instead of python3
torchrun --standalone --nproc_per_node=4 main.py

# --- 4. SALVAGE & CLEANUP ---
echo ">>> Training complete. Salvaging results..."
cp *.pth $PROJECT_DIR/
cp *.png $PROJECT_DIR/

echo ">>> Cleaning up scratch..."
cd $PROJECT_DIR
rm -rf $SCRATCH_DIR

echo ">>> Pipeline complete."
