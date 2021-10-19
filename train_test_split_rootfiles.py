#!/bin/python
#Splits root files into training and testing sets
import ROOT
import numpy as np
import sys, os
import glob

files = glob.glob("/your/path/to/root_files/*ANALYSIS.root")
treename = "lundjets_InDetTrackParticles"
outpath_name = "/your/path/for/train_test_files"

#print(files)
test_size = 0.90 # Percentage of data used for testing, the remaining is train data, 10% here

for file in files:
    
    print("Opening file:", file)
    filename = file.split('/')[-1]
    f = ROOT.TFile(file)
    
    t = f.Get(treename)
    nentries = t.GetEntries()
    test_entries = int(nentries * test_size)

    print("training set entries: {:20} | testing set entries: {:20}".format(nentries-test_entries, 
                                                                            test_entries))
    
    print("Saving testing set")
    testfile = ROOT.TFile(outpath_name+filename+"_test.root", "recreate");
    
    tcopy = t.CopyTree("", "", test_entries, 0)
        
    testfile.Write()
    testfile.Close()

    
    print("Saving training set")
    trainfile = ROOT.TFile(outpath_name+filename+"_train.root", "recreate");
    
    tcopy = t.CopyTree("", "", nentries, test_entries)
   
    trainfile.Write()
    trainfile.Close()

    f.Close()
