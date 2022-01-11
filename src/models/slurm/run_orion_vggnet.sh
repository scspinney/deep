#!/bin/bash
#SBATCH --account=def-patricia
#SBATCH --array=1-10
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=187G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-10:00
#SBATCH --output=../../reports/vggnet-%N-%j.out



module purge

module load python/3.7.9

module load cuda/11.2.2 cudnn/8.2.0

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$EBROOTCUDA
export XLA_PYTHON_CLIENT_PREALLOCATE=false

VIRTUAL_ENV=/scratch/spinney/env


source $VIRTUAL_ENV/bin/activate


# python main.py --lr~'loguniform(1e-5, 1.0)'
orion hunt -n parallel-exp --worker-trials 1 python /home/spinney/project/spinney/deep/src/models/vggnet.py --cfg~"choices(['A','B','C','D','E'])" --classifier_cfg~"choices(['A','B','C','D','E'])" --name vggnet --data_dir $SCRATCH/enigma_drug/data --batch_size 16  --max_epochs 100 --num_workers 6 --num_samples -1 --learning_rate 0.0001 --cropped True --seed 42 --optim adamw --num_classes 5


