import numpy
import json
import os
import sys

params_name = "parameters.txt"
save_folder = "Model0"


model = "Architectures/DoubleLinked.py"
name = "DLs200_64"

lrate = 0.0001
reg = 0
batch_size = 5
kernel_size = 5
filter_list =  [64,128,256, 512, 1024]
#loss = "fancy"
loss = "categorical_crossentropy"
#loss = "jaccard"
#data = "Humans_CT_Phantom_s224.hdf5"
data = "cv_dataset_s200.hdf5"
no_epochs = 5000
duplicate = True


d = {"name":name,
     "model_path": model,
     "data_path": data,
     "save_folder": save_folder,
     "kernel_size": kernel_size,
     "batch_size": batch_size,
     "filters": filter_list,
     "lrate": lrate,
     "reg":reg,
     "loss": loss,
     "no_epochs": no_epochs,
     "duplicate": duplicate}


if (os.path.isfile(params_name)):
    confirm_metada = input("Warning params file exists, continue? (y/n) ")
    if(confirm_metada == "y"):
        os.remove(params_name)
    else:
        sys.exit()
        

with open(params_name, 'w') as fp:
    json.dump(d, fp)
    
    
#models = ["DoubleLinked_s224", "DoubleLinked_s200"]
#lrates = [1,2,3]
#regs = [1,2,3]
#batch_sizes = [5,20]
#kernel_sizes = [3,5]
#filter_list = [[64,128,256,512,1024,2048], [32,64,128,256,512,1024]]
#losses = ["categorical_crossentropy", "fancy"]
#im_size = ["Humans_CT_Phantom_s224.hdf5", "Humans_CT_Phantom_s200.hdf5"]
#
#counter = 0
#for size in im_size:
#    for loss in losses:
#        for filters in filter_list:
#            for lrate in lrates:
#                for reg in regs:
#                    for batch in batch_sizes:
#                        for kernel in kernel_sizes:
#                            
#                            d["name"] = "Model_" + counter
#                            d["save_folder"] = "Model_" + counter
#                            d["data_path"] = size
#                            d["lrate"] = lrate
#                            d["reg"] = reg
#                            d["batch_size"] = batch
#                            d["filters"] = filters
#                            d["kernel_size"] = kernel
#                            d["loss"] = loss
#                            d["name"] = name
#
#                            counter += 1
#
                    
