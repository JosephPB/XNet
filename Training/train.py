import numpy as np
import TrainingClass
import json
import os
import sys
from PostProcessing import PostProcessing
import glob
import shutil
import webbrowser
import time
#from Killer import GracefulKiller
#killer = GracefulKiller()

param_files = glob.glob("aug*.txt")
print("I will train on all these parameter files:\n")
print(*param_files, sep = "\n")

#print("Opening tensorboard... \n")
#tb_url = "http://127.0.0.1:7007/"
#webbrowser.open(tb_url)

for file in param_files:
    params = json.load(open(file,'r'))

    save_folder = params["save_folder"]
    if(os.path.isdir(save_folder)):
        rm_folder = input("Warning, folder exists! Delete? (y/n) ")
        if(rm_folder == "y"):
            shutil.rmtree(save_folder)
        else:
            sys.exit()
            
    os.mkdir(save_folder)

    #tensorboard stuff
    #tbdir = os.path.join(save_folder, "tboard")
    #os.mkdir(tbdir)
    #os.system("killall tensorboard")
    #os.system("tensorboard --logdir=" + tbdir + " --port=7007 &")
    training = TrainingClass.TrainingClass(**params)
    try:
        training.fit()
    except:
        print("\n Dying... \n")
        
    print("Running post training analysis...\n")
        
    h5_files = np.sort(glob.glob(os.path.join(params["save_folder"], "*.h5")))
    try:
        pp = PostProcessing( h5_files[-1], params["data_path"], device = "gpu")
    except:
        print("You haven't trained anything?")
        continue

    pfile = open(os.path.join(params["save_folder"],  "results.txt"), "w")
    pfile.write("Overall perfomance: \n")
    accuracy_test, trainable_count = pp.evaluate_overall(device = "gpu")
    pfile.write("Accuracy: {} \nTrainable parameters: {} \n\n".format(round(accuracy_test,2)*100, trainable_count) )
    
    pfile.write("Performance per class:\n")
    beam_accuracy, tissue_accuracy, bone_accuracy = pp.evaluate_perclass()
    pfile.write(" Open beam: {} \n Soft Tissue: {} \n Bone: {}\n\n".format(round(beam_accuracy,2)*100, round(tissue_accuracy,2)*100, round(bone_accuracy,2)*100))
    
    pfile.write("True Positives and False Positives:\n")
    tp, fp = pp.tpfp()
    pfile.write(" TP: {} \n FP {} \n\n ".format(round(tp,2)*100, round(fp,2)*100))
    
    pfile.write("Threshold 90% \n")
    thresh90 = pp.thresholding(0.9)
    tp90, fp90 = pp.tpfp(thresh90)
    pfile.write(" TP90: {} \n FP90 {}\n \n ".format(round(tp90,2)*100, round(fp90,2)*100))
    
    pfile.write("Threshold 99% \n")
    thresh99 = pp.thresholding(0.99)
    tp99, fp99 = pp.tpfp(thresh99)
    pfile.write(" TP99: {} \n FP99 {}\n \n ".format(round(tp99,2)*100, round(fp99,2)*100))
    
    pfile.close()
   
    lc_fig, lc_ax = pp.learning_curve(os.path.join(params["save_folder"], params["name"] + ".csv"))
    lc_fig.savefig(os.path.join(params["save_folder"], "learning_curve.png"))
    
    rc_fig, rc_ax = pp.ROC_curve()
    rc_fig.savefig( os.path.join(params["save_folder"] , "roc_curve.png") )
    
    
