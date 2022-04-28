def logTime(model):
    if hasattr(model, 'training_time'):
        with open("./Results/RVB/burgers/Nf%d_logs.txt"%(model.Nf), 'a') as fout:
            fout.write("Nf=%d\n"%(model.Nf))
            fout.write("training_time=%f\n"%(model.training_time))
    print("training time appended done!")

def logRelativeError(model, error):
    with open("./Results/RVB/burgers/Nf%d_logs.txt"%(model.Nf), 'a') as fout:
        fout.write("L2 error=%f\n"%(error))
    print("L2 error appended done!")