import matplotlib.pyplot as plt
import numpy as np

def plotting(X, T, Exact, u_pred):
    fig = plt.figure(figsize=(6, 5))
    plt.pcolor(T, X, Exact, cmap='jet')
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Exact $u(x,t)$')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('./Results/RVB/burgers/Exact u.pdf')

    fig = plt.figure(figsize=(6, 5))
    plt.pcolor(T, X, u_pred, cmap='jet')
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Predict $u(x,y)$')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('./Results/RVB/burgers/Pred u.pdf')

    fig = plt.figure(figsize=(6, 5))
    plt.pcolor(T, X, np.abs(u_pred - Exact), cmap='jet')
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Absolute error')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('./Results/RVB/burgers/Absolute error.pdf')

    print("plotting done!")
