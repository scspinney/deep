
module purge

module load python/3.7.9

module load cuda/11.2.2 cudnn/8.2.0

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$EBROOTCUDA
export XLA_PYTHON_CLIENT_PREALLOCATE=false

VIRTUAL_ENV=/scratch/spinney/env

if [ -d "$VIRTUAL_ENV" ]; then
    source $VIRTUAL_ENV/bin/activate
else

virtualenv --no-download $VIRTUAL_ENV
source $VIRTUAL_ENV/bin/activate
pip install --no-index --upgrade pip

pip install --no-index torch
pip install --no-index pytorch_lightning
pip install --no-index tensorflow
pip install --no-index matplotlib
pip install --no-index wandb
pip install --no-index nilearn
pip install --no-index nibabel
pip install --no-index pandas
pip install --no-index torchvision
#pip install --no-index glob2
pip install --no-index torchio
pip install jupyter
echo -e '#!/bin/bash\nunset XDG_RUNTIME_DIR\njupyter notebook --ip $(hostname -f) --no-browser' > $VIRTUAL_ENV/bin/notebook.sh
chmod u+x $VIRTUAL_ENV/bin/notebook.sh
pip install jupyterlab jupyter-server-proxy nbserverproxy ipykernel dask jedi==0.17.2 jupyterlmod ipykernel

jupyter nbextension enable --py jupyterlmod --sys-prefix
jupyter serverextension enable --py jupyterlmod --sys-prefix
python -m ipykernel install --user --name cedar --display-name 'Python 3.7.9 DEEP'
fi

echo "Running the jupyter notebook in interactive node:"

#salloc --time=2:0:0  --ntasks=$NTASKS --cpus-per-task=$CPUSPERTASK --mem-per-cpu=$MEMPERCPU --account=$GROUP srun $VIRTUAL_ENV/bin/notebook.sh
salloc --account=def-patricia --cpus-per-task=6 --mem=187G --gres=gpu:v100l:1 --time=2:0:0 srun $VIRTUAL_ENV/bin/notebook.sh &
