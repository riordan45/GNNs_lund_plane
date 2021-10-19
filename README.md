# Graph Neural Networks for tagging jets with the Lund Plane
Lund Plane DNN studies.

### Setup
Use conda and import the environment with the .yml file. 

### Workflow
All your datasets need to be in the `.root` format. This is preferred over `.npz` and `.hdf5` due to the irregular shape of the Lund variables arrays. This representation also allows the processing of the data on the fly which is needed due to some `.root` files have emppty components for certain jets.
The testing portion was written mainly for the LXPLUS cluster but you can use the testing scripts on your own machine by removing the slice of testable `.root` files.

## Contents
- Python scripts
    - `train_test_split_rootfiles.py` splits root files into a training and a testing set. Part of the new workflow.
    - `gnn_train.py` trains your GNN models without the adversarial network (which is needed for mass decorrelation). Just copy and paste your own GNN here.
    - `adv_script.py` decorrelates your GNN model (or any Neural Network). You need to have a trained model already for this.
    - `test_1out.py` tests your model based on your `*test.root` files assuming the models output is 1 in the final linear layer.
    - `test_2ut.py` tests your model based on your `*test.root` files assuming the models output is 2 in the final linear layer.
    
- For LXPLUS
    - `test.sh` bash script for the `.sub` file. Change how you activate your environment and access your scripts.
    - `test.sub` file for HTcondor submission. Change the number of queues to the number of your test files.    
    
- Jupyter Notebooks
    - `pytorch_graph_neural_networks.ipynb` Notebook generally used for debugging a gnn model before you include it in the gnn_train.py script to make sure it works on your machine.
    - `pytorch_adversarial_network.ipynb` Notebook that was used for debugging the adversarial network.  



    

