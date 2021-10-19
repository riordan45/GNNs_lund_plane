# lund_plane_DNN
Lund Plane DNN studies.

### Requirements
- numpy, matplotlib
- tensorflow v2.3
- (keras, now included in tensorflow)
- ROOT
- root_numpy
- uproot

## Contents
- Jupyter Notebooks
    - `nn_train_on_root_file.ipynb` trains a neural network, taking a root file containing the training set as input. Performs pre-processing on the fly. First step of new workflow.
    - `nn_evaluate_on_root_file.ipynb` evaluates the tagger performance on root files containing the testing set, performing pre-processing steps on the fly. Saves root files containing jet kinematics and tagger score. Second step of new workflow.
    - `tagger_performance_loop.ipyn` loops over root files containing jet variables and tagger scores. Calculates and plots tagger efficiencies, ROC curves and other performance metrics. Final stage of new workflow.
    - `plot_images.ipynb` plots jet images and Lund Planes from `.npz` inputs to NNs.

- Python scripts
    - `combine_nn_tagger_scores.py` script to merge root files containing tagger scores created using `nn_evaluate_on_root_file.ipynb`. Required for plotting the performance of multiple taggers using `tagger_performance_loop.ipynb`.
    - `train_test_split_rootfiles.py` splits root files into a training and a testing set. Part of the new workflow.

- Python utitlities
    - `keras_nets` contains keras models for neural network architectures.
    - `util` utility functions for neural network preprocessing/plotting.
    
- Output directories
    - `plots` directory for storing plots (ignored by git).
    - `save` directory for storing saved models (ignored by git).
    
- Bash scripts etc.
    - `launch_notebooks.sh` launches anaconda and the Jupyter notebook server.
    - `.condarc_sample` sample config for anaconda.
        - When setting up your own envrionment change env and pkgs locations to set up virtual evironments in your own directory on disk (`/mnt/storage`). 

- Old Jupyter Notebooks
    - `old_workflow/nn_preprocessing.ipynb` makes '.npz' files containing jet images / Lund Planes. Takes `.root` ntuples as inputs.
    - `old_workflow/neural_nets.ipynp` trains and evaluates neural networks.
    - `old_workflow/recurrent_neural_nets.ipynb` trains and evaluates recurrent neural networks, eg. LSTMs.

- Old Scripts
    - `old_workflow/merge_npz_files.py` script to merge `.npz` compressed python array files.
    - `old_workflow/merge_hdf_files.py` script to merge `.hdf5` python array files.
    - `old_workflow/batch_run_notebook.sh` bash script for running jupyter notebooks in batch mode. Use eg. the `screen` command to keep jobs running while logged out. 
    - `old_workflow/qsub_batch_run_notebook.sh` script for running `nn_preprocessing.ipynb` on the UCL HEP batch farm using `qsub`. Required for pre-processing the full dataset.
    - `old_workflow/submit_jobs.sh` submits `qsub_batch_run_notebook.sh` to the batch farm.


### Note on workflow for 2020
The ususal strategy for training and evaluating the neural networks has been to first produce `.npz` or `.hdf5` files containing pre-processed images or arrays of Lund Plane coordinates, that can be fed directly into the neural network. This required running an additional pre-processing step, which was time consuming. Tagger evaluation and plotting was done in the same notebook as tagger training, which made the results hard to reproduce.

To combat these issues the preferred strategy is now instead to read the data as Cern `.root` files and pre-process it directly before feeding it into the neural networks. This is done in `nn_train_on_root_file.ipynb`. Evaluating the tagger performance is also done on `.root` files using `nn_evaluate_on_root_file.ipynb`, which procudes new `.root` files containing tagger scores in addition to jet kinematics ($p_T$, $\eta$, $\phi$, $m$, etc. ). The advantage of this workflow is that tagger performance can be evaluated for both our taggers and other ATLAS taggers using `tagger_performance_loop.ipynb`, which only requires `.root` files containing tagger scores and ntuples as input. All tagger performance plots should be made in `tagger_performance_loop.ipynb` using the variables and tagger scores that have been saved to the `.root` files.

- - -
    
### Getting Started: How to run jupyter notebooks on remote

- Connnect to plus1 using ssh with X11 and port forwarding:
    - `ssh -Y -L localhost:8888:localhost:8080 <username>@plus1.hep.ucl.ac.uk`
    
- Then to <hostname> (gpu02, gpu03 or pcXXX):
    - `ssh -Y -L localhost:8080:localhost:8888 <hostname>`
    
- **Optional** Follow the instructions below to set up an anaconda environment.

- Run the launch script:
    - `sh launch_notebooks.sh`
    
- Open link to notebook shown in shell in local browser. The Jupyter notebook server should be forwarded to `localhost:8888` -- the same location as on the remote machine.

- Try running eg. `neural_nets.ipynb` or`recurrent_neural_nets.ipynb` on one of the gpu machines. Currently, the preprocessing and plotting notebooks which use pyROOT will not run on the gpu machines. Set up a py2.7 environment in a shared directory to run these scripts (see instructions below). 


### To set up a new anaconda environment on gpu01, gpu02 or gpu03

- Activate the base anaconda environment on disk: `source /mnt/storage/anaconda3.XXX/bin/activate` where `XXX` is whatever version exists on this machine.

- Copy the file `.condarc_sample` into your home directory and rename it `.condarc`. Open the file in a text editor and replace the paths for the `envs` and `pkgs` directories to one of your directories on disk eg. `/mnt/storage/<username>/conda`. If you cannot access `/mnt/storage` email `support@hep.ucl.ac.uk` to get a directory set up for you.

- Create the environment: `conda create --name <environment name>` 

- `<environment name>` should be `gpuX_conda` for the launch script to run.

- Activate the environment: `conda activate <environment name>`

- Install necessary packages:
   - `conda install jupyter scikit-learn matplotlib pandas uproot`
   - `pip install tensorflow` (this should install the latest vesion, 2.3)
   - `conda install -c anaconda cudnn` (this will upgrade CuDNN to the latest version, as required by TensorFlow)


### To set up a new anaconda environment for all other machines (pcXXX)

- **As of September 2020, all notebooks/scripts can be run on the gpu machines. It should therefore not be required to set up a conda environment on the pcXXX machines.** However, I sometimes still encounter problems when running certain ROOT functions with ROOT v. 6.18 and python v. 3.7, since ROOT does not officially support python 3.7. The instructions below are only necessary when ROOT crashes cannot be avoided and a python 3.6 enviroment is required. 

- This is required for pre-processing data on the batch farm using qsub. 

- To set up your own instance of anaconda on the machines in the UCL cluster one can use miniconda which can be downloaded [here](https://docs.conda.io/en/latest/miniconda.html). 

- Download the py3.6 version and follow the installation instructions for linux.

- You will need some space in a shared directory (eg. `unix/atlas2`). The anaconda/miniconda installations are very large (~3GB) which would quickly fill up your home directory. Email `support@hep.ucl.ac.uk` to get a directory set up for you.

- Copy the file `.condarc_sample` into your home directory and rename in `.condarc`. Open the file in a text editor and replace the paths for the `envs` and `pkgs` directories to one of your directories on disk eg. `/unix/atlas2/<username>/conda`.

- Create the environment: `conda create --name <environment name>` 

- `<environment name>` should be `rootenv` for the launch script to run without errors.

- Activate the environment: `conda activate <environment name>`

### To use PyTorch and PyTorch Geometric

The installation guide can be found here: https://pytorch.org/
I personally did it with pip instead of conda because it was faster. Just run the command:
- pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

Note that this is for CUDA version 11.1 and 11.2. If you have 10.2 just change cu111 to cu102.

Now the installation of PyTorch Geometric is dependent on both the PyTorch version and the CUDA version you have. The installation guide is found here: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
In my case I had to do:
- pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
- pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
- pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
- pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
- pip install torch-geometric

You should be good to go when it comes to creating Graph Neural Networks now. Take a look at the specific Jupyter Notebook to see how one can train and test a model on a GPU.
