import os.path as osp
import os
import glob
import torch
import awkward as ak
import time
import uproot
import uproot3
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataListLoader, DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import GatedGraphConv, SplineConv, global_mean_pool, DataParallel, EdgeConv, GATConv, GINConv
from torch_geometric.data import Data
import scipy.sparse as ss
from datetime import datetime, timedelta
import argparse

parser = argparse.ArgumentParser(description='Index helper')
parser.add_argument('index', type=int, help='Beginning of slice')
args = parser.parse_args()

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def create_train_dataset_prmy(z, k, d, label):
    graphs = []
    for i in range(len(z)):
        vec = []
        vec.append(np.array([d[i], z[i], k[i]]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)
        ya = kneighbors_graph(vec, n_neighbors=int(np.floor(vec.shape[0]/2)))
        edges = np.array([ya.nonzero()[0], ya.nonzero()[1]])
        edge = torch.tensor(edges, dtype=torch.long)
        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float), edge_index=edge, y=torch.tensor(label[i], dtype=torch.float)))
    return graphs


def create_train_dataset_fulld(z, k, d, p1, p2, label):
    graphs = []
    for i in range(len(z)):
        #if i%1000 == 0: 
            #print("Processing event {}/{}".format(i, len(z)), end="\r")
        vec = []
        vec.append(np.array([d[i], z[i], k[i]]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)

        v1 = [[ind, x] for ind, x in enumerate(p1[i]) if x > -1]
        v2 = [[ind, x] for ind, x in enumerate(p2[i]) if x > -1]

        a1 = np.reshape(v1,(len(v1),2)).T
        a2 = np.reshape(v2,(len(v2),2)).T
        edge1 = np.concatenate((a1[0], a2[0], a1[1], a2[1]),axis = 0)
        edge2 = np.concatenate((a1[1], a2[1], a1[0], a2[0]),axis = 0)
        edge = torch.tensor(np.array([edge1, edge2]), dtype=torch.long)
        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float), edge_index=edge, y=torch.tensor(label[i], dtype=torch.float)))
    return graphs


def create_test_dataset_prmy(z, k, d):
    graphs = []
    for i in range(len(z)):
        vec = []
        vec.append(np.array([d[i], z[i], k[i]]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)
        ya = kneighbors_graph(vec, n_neighbors=int(np.floor(vec.shape[0]/2)))
        edges = np.array([ya.nonzero()[0], ya.nonzero()[1]])
        edge = torch.tensor(edges, dtype=torch.long)
        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float), edge_index=edge))
    return graphs

def create_test_dataset_fulld(z, k, d, p1, p2):
    graphs = []
    for i in range(len(z)):
        vec = []
        vec.append(np.array([d[i], z[i], k[i]]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)
        v1 = [[ind, x] for ind, x in enumerate(p1[i]) if x > -1]
        v2 = [[ind, x] for ind, x in enumerate(p2[i]) if x > -1]

        a1 = np.reshape(v1,(len(v1),2)).T
        a2 = np.reshape(v2,(len(v2),2)).T
        edge1 = np.concatenate((a1[0], a2[0], a1[1], a2[1]),axis = 0)
        edge2 = np.concatenate((a1[1], a2[1], a1[0], a2[0]),axis = 0)
        edge = torch.tensor(np.array([edge1, edge2]), dtype=torch.long)
        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float), edge_index=edge))
    return graphs

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 64),
                                  nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(128, 128),
                                  nn.ReLU(), nn.Linear(128, 128),nn.ReLU(), nn.Linear(128, 128)),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(256,256,),
                                  nn.ReLU(), nn.Linear(256, 256),nn.ReLU(), nn.Linear(256, 256)),aggr='add')
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(256, 64)
        self.lin3 = torch.nn.Linear(64, 2)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.1)
        x = self.lin2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin3(x)
        #print(x.shape)
        return torch.sigmoid(x)


#Configuration from bash script
if "INFILE" in os.environ:
    infile_path = os.environ["INFILE"]
    model_name = os.environ["MODEL"]
    epochs = int(os.environ["EPOCHS"])

#Configuration in notebook
else:    
#    files = glob.glob("/mnt/storage/asopio/train_test_split_20210305/training_set_jz2to9.root")
    files = glob.glob("/eos/user/r/riordan/data/wprime/*_train.root") + glob.glob("/eos/user/r/riordan/data/j3to9/*_train.root")
    model_name = "GNN_full"
#    model_name = "LSTM"
#    model_name = "1DCNN"
#    model_name = "2DCNN"
#    model_name = "ImgCNN"
#    model_name = "GNN"
#    model_name = "Transformer"
#    nb_epochs = 20


#Load tf keras model
# jet_type = "Akt10RecoChargedJet" #track jets


intreename = "lundjets_InDetTrackParticles"

model_filename = "save/models/"+model_name+".hdf5"

def pad_aux(arr, l):
    """
    This function pads empty auxiliary variables lists such as jet pt and jet mass.

    It should be applied where there is a scalar value per jet.
    """
    arr = ak.pad_none(arr, l)
    arr = ak.fill_none(arr, 0)
    arr = ak.to_numpy(arr)
    return arr

def pad_lundvar(arr, l):
    """
    This function pads empty Lund variables lists such as Z and the graph connectivity information.

    It should be applied where there is a list of said values per jet.
    """
    arr = ak.pad_none(arr, 1)
    arr = ak.fill_none(arr, [0])
    arr = arr[:,0]
    arr = ak.pad_none(arr, l)
    arr = ak.fill_none(arr, 0)
    #arr = ak.to_numpy(arr)
    return arr

@torch.no_grad()
def get_accuracy(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]/2),2))
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(new_y[:,1]).sum().item()
    return correct / len(loader.dataset)
    
@torch.no_grad()
def get_scores(loader):
    model.eval()
    total_output = np.array([[1,1]])
    for data in loader:
        data = data.to(device)
        pred = model(data)
        total_output = np.append(total_output, pred.cpu().detach().numpy(), axis=0)
    return total_output[1:]

class LundNet(torch.nn.Module):
    def __init__(self):
        super(LundNet, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='mean')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='mean')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='mean')
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='mean')
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),aggr='mean')
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),aggr='mean')
        self.sc1 = nn.Sequential(nn.Linear(3, 32, bias=False), nn.BatchNorm1d(num_features=32))
        self.sc2 = nn.Sequential(nn.ReLU())
        self.sc3 = nn.Sequential(nn.Linear(32, 64, bias=False), nn.BatchNorm1d(num_features=64))
        self.sc5 = nn.Sequential(nn.Linear(64, 128, bias=False), nn.BatchNorm1d(num_features=128))
        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x1 = self.sc2(self.sc1(x) + x1)
        x2 = self.conv2(x1, edge_index)
        x2 = self.sc2(x2 + x1)
        x3 = self.conv3(x2, edge_index)
        x3 = self.sc2(self.sc3(x2) + x3)
        x4 = self.conv4(x3, edge_index)
        x4 = self.sc2(x4 + x3)
        x5 = self.conv5(x4, edge_index)
        x5 = self.sc2(self.sc5(x4) + x5)
        x6 = self.conv6(x5, edge_index)
        x6 = self.sc2(x6 + x5)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        return torch.softmax(x)
                                

class GATNet(torch.nn.Module):
    def __init__(self, in_channels):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, 16, heads=8, dropout=0.1)
        self.conv2 = GATConv(16 * 8, 16, heads=8, dropout=0.1)
        self.conv3 = GATConv(16 * 8, 32, heads=8, dropout=0.1)
        self.conv4 = GATConv(32 * 8, 32, heads=16, dropout=0.1)
        self.conv5 = GATConv(16 * 32, 64, heads=16, dropout=0.1)
        self.conv6 = GATConv(64 * 16, 64, heads=16, dropout=0.1)
        self.seq1 = nn.Sequential(nn.Linear(3072, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        #self.lin1 = torch.nn.Linear(128, 128)
        #self.lin2 = torch.nn.Linear(448, 64)
        self.lin = nn.Linear(256, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return torch.sigmoid(x)

class GGNet(torch.nn.Module):
    def __init__(self):
        super(GGNet, self).__init__()
        self.conv1 = GatedGraphConv(8, 5)
        self.conv2 = GatedGraphConv(16, 6)
        self.conv3 = GatedGraphConv(32, 2)
        self.conv4 = GatedGraphConv(64, 2)
        self.conv5 = GatedGraphConv(126, 4)
        self.conv6 = GatedGraphConv(256, 4)
        self.seq1 = nn.Sequential(nn.Linear(502, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        self.lin = nn.Linear(256, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return torch.sigmoid(x)

class GINNet(torch.nn.Module):
    def __init__(self):
        super(GINNet, self).__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(3, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),train_eps=True)
        self.conv2 = GINConv(nn.Sequential(nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),train_eps=True)
        self.conv3 = GINConv(nn.Sequential(nn.Linear(32,64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),train_eps=True)
        self.conv4 = GINConv(nn.Sequential(nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),train_eps=True)
        self.conv5 = GINConv(nn.Sequential(nn.Linear(64, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),train_eps=True)
        self.conv6 = GINConv(nn.Sequential(nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),train_eps=True)

        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        #self.lin1 = torch.nn.Linear(128, 128)
        #self.lin2 = torch.nn.Linear(448, 64)
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return torch.sigmoid(x)



path = "/your/path/to/models/gatnet.pt"
#torch.save(model.state_dict(), path)
model = GATNet(3)
model.load_state_dict(torch.load(path))
device = torch.device('cuda')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

infiles_list = glob.glob("/your/path/to/signal/*_test.root") + glob.glob("/your/path/to/jzslices/*_test.root")
infiles_list = infiles_list[args.index:args.index+1]

#EVALUATING
#output ntuple directory
jet_type = "Akt10UFOJet" #UFO jets
outdir = "/your/path/to/scores"
model_name = "trial_phase"
#input TTree name
intreename = "lundjets_InDetTrackParticles"
nentries_total = 0
for infile_name in infiles_list: 
    nentries_total += uproot3.numentries(infile_name, intreename)

print("Evaluating on {} files with {} entries in total.".format(len(infiles_list), nentries_total))


nentries = 0
print(model_name)
t_start = time.time()
for i_file, infile_name in enumerate(infiles_list):
    
    print("Loading file {}/{}".format(i_file+1, len(infiles_list)))
    t_filestart = time.time()
    
    #Open file 
    file = uproot.open(infile_name)
    tree = file[intreename]    
    
    #Get weights info
    dsids = tree["DSID"].array()
    mcweights = tree["mcWeight"].array()
    NBHadrons = pad_ak(tree["Akt10TruthJet_inputJetGABHadrons"].array(), 30)[:,0]
    #### 
    #For Full decluster
    
    parent1 = pad_lundvar(tree["{}_jetLundIDParent1".format(jet_type)].array(), 2)
    parent2 = pad_lundvar(tree["{}_jetLundIDParent2".format(jet_type)].array(), 2)
    #print(len(parent1))
    ####
    #Get jet kinematics
    jet_pts = pad_aux(tree["{}_jetPt".format(jet_type)].array(), 30)[:,0]
    jet_etas = pad_aux(tree["{}_jetEta".format(jet_type)].array(), 30)[:,0]
    jet_phis = pad_aux(tree["{}_jetPhi".format(jet_type)].array(), 30)[:,0]
    jet_ms = pad_aux(tree["{}_jetM".format(jet_type)].array(), 30)[:,0]
    
    #Get lund values
    all_lund_zs = pad_lundvar(tree["{}_jetLundZ".format(jet_type)].array(), 2)
    all_lund_kts = pad_lundvar(tree["{}_jetLundKt".format(jet_type)].array(), 2)
    all_lund_drs = pad_lundvar(tree["{}_jetLundDeltaR".format(jet_type)].array(), 2)
    #print(len(all_lund_zs))
    delta_t_fileax = time.time() - t_filestart
    #print("Opened data in {:.4f} seconds.".format(delta_t_fileax))
     
    #### 
    #For Full decluster
    
    test_data = create_test_dataset_fulld(all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2)
    
    ####
    
    #test_data = create_test_dataset_prmy(all_lund_zs, all_lund_kts, all_lund_drs)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)
    
    #Predict scores
    y_pred = get_scores(test_loader)
    print(y_pred)
    tagger_scores = y_pred[:,1]
    inv_tagger_scores = y_pred[:,0]
    delta_t_pred = time.time() - t_filestart - delta_t_fileax
    print("Calculated predicitions in {:.4f} seconds,".format(delta_t_pred))

    #Save root files containing model scores
    filename = infile_name.split("/")[-1]
    outfile_path = os.path.join(outdir, filename) 
    
    
    with uproot3.recreate("{}_score_{}.root".format(outfile_path, model_name)) as f:

        treename = "FlatSubstructureJetTree"

        #Declare branch data types
        f[treename] = uproot3.newtree({"EventInfo_mcChannelNumber": "int32",
                                      "EventInfo_mcEventWeight": "float32",
                                      "EventInfo_NBHadrons": "int32",   # I doubt saving the parents is necessary here
                                      "fjet_nnscore": "float32",        # which is why I didn't include them
                                      "fjet_invnnscore": "float32", 
                                      "fjet_pt": "float32",
                                      "fjet_eta": "float32",
                                      "fjet_phi": "float32",
                                      "fjet_m": "float32",
                                      })

        #Save branches
        f[treename].extend({"EventInfo_mcChannelNumber": dsids,
                            "EventInfo_mcEventWeight": mcweights,
                            "EventInfo_NBHadrons": NBHadrons,
                            "fjet_nnscore": tagger_scores, 
                            "fjet_invnnscore": inv_tagger_scores, 
                            "fjet_pt": jet_pts,
                            "fjet_eta": jet_etas,
                            "fjet_phi": jet_phis,
                            "fjet_m": jet_ms,
                            })

    delta_t_save = time.time() - t_start - delta_t_fileax - delta_t_pred
    print("Saved data in {:.4f} seconds.".format(delta_t_save))
    
    
    #Time statistics
    nentries += uproot3.numentries(infile_name, intreename)
    time_per_entry = (time.time() - t_start)/nentries
    eta = time_per_entry * (nentries_total - nentries)
    
    print("Evaluated on {} out of {} events".format(nentries, nentries_total))    
    print("Estimated time until completion: {}".format(str(timedelta(seconds=eta))))
    
    
print("Total evaluation time: {:.4f} seconds.".format(time.time()-t_start))









































