import os
import time
import shutil

PATH = "./Results/RVB/burgers"

def mkdir(Nf, act, iter):
    new_path = PATH + "/Nf%d_%s_iter%d"%(Nf, act, iter)
    folder = os.path.exists(new_path)
    if not folder:
        os.makedirs(new_path)
        print("mkdir done!")
    else:
        print("folder exists!")
        print("make folder by time")
        new_path = new_path + str(time.time()).split(".")[0]
        os.makedirs(new_path)
        print("mkdir done!")
    return new_path

def movefiles(old, new):
    filelist = os.listdir(old)
    for file in filelist:
        src = old + "/" + file
        if os.path.isfile(src):
            dst = new + "/" + file
            shutil.move(src, dst)
    print("files moved done!")

def arrangeFiles(model, iter):
    new = mkdir(model.Nf, model.activation._tf_api_names[-1], iter)
    movefiles(PATH, new)
    print("File arranged done!")
