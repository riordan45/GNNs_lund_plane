import awkward
import os.path as osp
import os
import glob
import torch
import awkward as ak
import time
import uproot
import uproot3
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataListLoader, DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel, EdgeConv, GATConv, GINConv, PNAConv
from torch_geometric.data import Data
import scipy.sparse as ss
from datetime import datetime, timedelta

print("Libraries loaded!")

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
        self.lin3 = torch.nn.Linear(64, 1)
        
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

class LundNet_modified(torch.nn.Module):
    def __init__(self):
        super(LundNet, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(), 
                                            nn.Linear(32, 32), 
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(), 
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(), 
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='add')
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(), 
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='add')
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(), 
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),aggr='add')
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(), 
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),aggr='add')
        
        self.seq1 = nn.Sequential(nn.Linear(448, 384), 
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

class EdgeGinNet(torch.nn.Module):
    def __init__(self):
        super(EdgeGinNet, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(), 
                                            nn.Linear(32, 32), 
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(), 
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(), 
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='add')
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

#Configuration from bash script
if "INFILE" in os.environ:
    infile_path = os.environ["INFILE"]
    model_name = os.environ["MODEL"]
    epochs = int(os.environ["EPOCHS"])

#Configuration in notebook
else:    
#    files = glob.glob("/mnt/storage/asopio/train_test_split_20210305/training_set_jz2to9.root")
    files = glob.glob("/your/path/to/signal/*_train.root") + glob.glob("/your/path/to/jzslicess/*_train.root")

#Load tf keras model
# jet_type = "Akt10RecoChargedJet"
jet_type = "Akt10UFOJet" #UFO jets
save_trained_model = True
intreename = "lundjets_InDetTrackParticles"


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

print("Training tagger on files", len(files))
t_start = time.time()

dsids = np.array([])
NBHadrons = np.array([])
trial = np.ones((1,30))
all_lund_zs = ak.from_numpy(trial)
all_lund_kts = ak.from_numpy(trial)
all_lund_drs = ak.from_numpy(trial)
parent1 = ak.from_numpy(trial)
parent2 = ak.from_numpy(trial)
vector = []
for file in files:   
    print("Loading file",file)
    
    infile = uproot.open(file)
    tree = infile[intreename]
    dsids = np.append( dsids, np.array(tree["DSID"].array()) )
    #mcweights = tree["mcWeight"].array()
    NBHadrons = np.append( NBHadrons, pad_ak(tree["Akt10TruthJet_inputJetGABHadrons"].array(), 30)[:,0] )
    parent1 = ak.concatenate((parent1, pad_ak3(tree["{}_jetLundIDParent1".format(jet_type)].array(), 2)), axis = 0)
    parent2 = ak.concatenate((parent2, pad_ak3(tree["{}_jetLundIDParent2".format(jet_type)].array(), 2)), axis = 0)

    #Get Lund Variables

    all_lund_zs = ak.concatenate((all_lund_zs, pad_ak3(tree["{}_jetLundZ".format(jet_type)].array(), 2)), axis=0)
    all_lund_kts = ak.concatenate((all_lund_kts, pad_ak3(tree["{}_jetLundKt".format(jet_type)].array(), 2)), axis=0)
    all_lund_drs = ak.concatenate((all_lund_drs, pad_ak3(tree["{}_jetLundDeltaR".format(jet_type)].array(), 2)), axis=0)
    
all_lund_zs = all_lund_zs[1:]    
all_lund_kts = all_lund_kts[1:]
all_lund_drs = all_lund_drs[1:]
parent1 = parent1[1:]
parent2 = parent2[1:]

delta_t_fileax = time.time() - t_start
print("Opened data in {:.4f} seconds.".format(delta_t_fileax))

labels = ( dsids > 370000 ) # Depends on your signal and background definitions


#print(labels)
labels = to_categorical(labels, 2)

# The following is if you want the classifier to output one value instead of two
labels = np.reshape(labels[:,1], (len(all_lund_zs), 1))


#W bosons
# It will take about 30 minutes to finish
dataset = create_train_dataset_fulld(all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, labels) 
train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
print("Dataset created!")

# This is for PNANet just in case
deg = torch.zeros(10, dtype=torch.long)
for data in dataset:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())

class PNANet(torch.nn.Module):
    def __init__(self,in_channels):
        aggregators = ['sum','mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation',"linear",'inverse_linear']

        super(PNANet, self).__init__()
        self.conv1 = PNAConv(in_channels, out_channels=32, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv2 = PNAConv(in_channels=32, out_channels=64, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv3 = PNAConv(in_channels=64, out_channels=128, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv4 = PNAConv(in_channels=128, out_channels=256, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv5 = PNAConv(in_channels=256, out_channels=256, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv6 = PNAConv(in_channels=256, out_channels=256, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.seq1 = nn.Sequential(nn.Linear(992, 384),
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
        x = torch.cat((x1, x2,x3,x4,x5,x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        return torch.sigmoid(x)


model = EdgeGinNet()
device = torch.device('cuda')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

def train(epoch):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]/2),2))
        loss = F.binary_cross_entropy(output, new_y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(dataset)

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
    
for epoch in range(1, 21):
    print("Epoch:{}".format(epoch))
    loss = train(epoch)
    train_acc = get_accuracy(train_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}'.format(epoch, loss, train_acc))

path = "/your/output/path/model.pt"
torch.save(model.state_dict(), path)
