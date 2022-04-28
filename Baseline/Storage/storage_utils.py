import numpy as np

def dumpTotalLoss(model):
    if hasattr(model, 'loss_log'):
        np.save('./Results/RVB/burgers/Nf%d_loss.npy' %(model.Nf), model.loss_log)
    if hasattr(model, 'loss_b_log'):
        np.save('./Results/RVB/burgers/Nf%d_loss_b.npy'%(model.Nf), model.loss_b_log)
    if hasattr(model, 'loss_r_log'):
        np.save('./Results/RVB/burgers/Nf%d_loss_r.npy'%(model.Nf), model.loss_r_log)
    if hasattr(model, 'loss_g_log'):
        np.save('./Results/RVB/burgers/Nf%d_loss_g.npy'%(model.Nf), model.loss_g_log)
    if hasattr(model, 'loss_v_log'):
        np.save('./Results/RVB/burgers/Nf%d_loss_v.npy'%(model.Nf), model.loss_v_log)
    print("dump to .npy done!")

