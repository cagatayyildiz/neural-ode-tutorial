import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
import glob

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch


def plot_vdp_trajectories(t, Y, ode_rhs):
    N_ = 10
    Y = Y.transpose(0,1)
    min_x,min_y = Y.min(dim=0)[0].min(dim=0)[0].detach().cpu().numpy()
    max_x,max_y = Y.max(dim=0)[0].max(dim=0)[0].detach().cpu().numpy()
    xs1_,xs2_ = np.meshgrid(np.linspace(min_x, max_x, N_),np.linspace(min_y, max_y, N_))
    Z  = np.array([xs1_.T.flatten(), xs2_.T.flatten()]).T
    Z = torch.from_numpy(Z).float().to(Y.device)
    F = ode_rhs(None,Z).detach().cpu().numpy()
    F /= ((F**2).sum(1,keepdims=True))**(0.25)
    Z  = Z.detach().cpu().numpy()
    
    t = t.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    fig = plt.figure(1,[15,7.5],constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[:, 0])
    
    N = Y.shape[0]
    if N>3:
        print('Plotting the first 3 data sequences.')
        N = 3
    ax1.set_xlabel('State $x_1$',fontsize=17)
    ax1.set_ylabel('State $x_2$',fontsize=17)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    h1 = ax1.quiver(xs1_, xs2_, F[:,0].reshape(N_,N_).T, F[:,1].reshape(N_,N_).T, \
                cmap=plt.cm.Blues)
    for n in range(N):
        h2, = ax1.plot(Y[n,0,0],Y[n,0,1],'o', fillstyle='none', \
                 markersize=11.0, linewidth=2.0)
        h3, = ax1.plot(Y[n,:,0],Y[n,:,1],'-', color=h2.get_color(), linewidth=3.0)
    plt.legend([h1,h2,h3],['Vector field','Initial value','Observed sequence'],
        loc='lower right', fontsize=20, bbox_to_anchor=(1.5, 0.05))
    
    
    ax2 = fig.add_subplot(gs[0, 1:])
    for n in range(N):
        h4, = ax2.plot(t,Y[n,:,0])
    ax2.set_xlabel('time',fontsize=17)
    ax2.set_ylabel('State $x_1$',fontsize=17)

    ax3 = fig.add_subplot(gs[1, 1:])
    for n in range(N):
        h5, = ax3.plot(t,Y[n,:,1])
    ax3.set_xlabel('time',fontsize=17)
    ax3.set_ylabel('State $x_2$',fontsize=17)
    
    plt.show()
    plt.close()
    

def plot_ode(t, X, ode_rhs, Xhat=None, L=1, return_fig=False):
    N_ = 10
    X = X.transpose(0,1)
    if Xhat is not None:
        Xhat = Xhat.transpose(0,1)
    min_x,min_y = X.min(dim=0)[0].min(dim=0)[0].detach().cpu().numpy()
    max_x,max_y = X.max(dim=0)[0].max(dim=0)[0].detach().cpu().numpy()
    xs1_,xs2_ = np.meshgrid(np.linspace(min_x, max_x, N_),np.linspace(min_y, max_y, N_))
    Z = np.array([xs1_.T.flatten(), xs2_.T.flatten()]).T
    Z = torch.from_numpy(Z).float().to(X.device)
    Z = torch.stack([Z]*L)
    F = ode_rhs(None,Z).detach().cpu().numpy()
    F /= ((F**2).sum(-1,keepdims=True))**(0.25)
    Z  = Z.detach().cpu().numpy()

    t = t.detach().cpu().numpy()
    X = X.detach().cpu().numpy()
    fig = plt.figure(1,[15,7.5],constrained_layout=True)
    gs  = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[:, 0])

    ax1.set_xlabel('State $x_1$',fontsize=17)
    ax1.set_ylabel('State $x_2$',fontsize=17)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    for F_ in F:
        h1 = ax1.quiver(xs1_, xs2_, F_[:,0].reshape(N_,N_).T, F_[:,1].reshape(N_,N_).T, \
                    cmap=plt.cm.Blues)
    if Xhat is None: # only plotting data
        for X_ in X:
            h2, = ax1.plot(X_[0,0],X_[0,1],'o', fillstyle='none', \
                     markersize=11.0, linewidth=2.0)
            h3, = ax1.plot(X_[:,0],X_[:,1],'-',color=h2.get_color(),linewidth=3.0)
    else: # plotting data and fits, set the color correctly!
        h2, = ax1.plot(X[0,0,0],X[0,0,1],'o',color='firebrick', fillstyle='none', \
                 markersize=11.0, linewidth=2.0)
        h3, = ax1.plot(X[0,:,0],X[0,:,1],'-',color='firebrick',linewidth=3.0)
    if Xhat is not None and Xhat.ndim==3:
        Xhat = Xhat.unsqueeze(0)
    if Xhat is None:
        plt.legend([h1,h2,h3],['Vector field','Initial value','State trajectory'],
            loc='lower right', fontsize=20, bbox_to_anchor=(1.5, 0.05))
    else:
        Xhat = Xhat.detach().cpu()
        for xhat in Xhat:
            h4, = ax1.plot(xhat[0,:,0],xhat[0,:,1],'-',color='royalblue',linewidth=3.0)
        if Xhat.shape[0]>1:
            ax1.plot(X[0,:,0],X[0,:,1],'-',color='firebrick',linewidth=5.0)
        plt.legend([h1,h2,h3,h4],['Vector field','Initial value','Data sequence', 'Forward simulation'],
            loc='lower right', fontsize=20, bbox_to_anchor=(1.5, 0.05))

    ax2 = fig.add_subplot(gs[0, 1:])
    if Xhat is None: # only plotting data
        for X_ in X:
            h4, = ax2.plot(t,X_[:,0],linewidth=3.0)
    else: # plotting data and fits, set the color correctly!
        h4, = ax2.plot(t,X[0,:,0],color='firebrick',linewidth=3.0)
    if Xhat is not None:
        for xhat in Xhat:
            ax2.plot(t,xhat[0,:,0],color='royalblue',linewidth=3.0)
        if Xhat.shape[0]>1:
            ax2.plot(t,X[0,:,0],color='firebrick',linewidth=5.0)
    ax2.set_xlabel('time',fontsize=17)
    ax2.set_ylabel('State $x_1$',fontsize=17)

    ax3 = fig.add_subplot(gs[1, 1:])
    
    if Xhat is None: # only plotting data
        for X_ in X:
            h5, = ax3.plot(t,X_[:,1],linewidth=3.0)
    else: # plotting data and fits, set the color correctly!
        h5, = ax3.plot(t,X[0,:,1],color='firebrick',linewidth=3.0)
    if Xhat is not None:
        for xhat in Xhat:
            ax3.plot(t,xhat[0,:,1],color='royalblue',linewidth=3.0)
        if Xhat.shape[0]>1:
            ax3.plot(t,X[0,:,1],color='firebrick',linewidth=5.0)
    ax3.set_xlabel('time',fontsize=17)
    ax3.set_ylabel('State $x_2$',fontsize=17)
    
    if return_fig:
        return fig,ax1,h3,h4,h5
    else:
        plt.show()


def plot_vdp_animation(t,X,ode_rhs):
    t,X = t.cpu(), X.detach().cpu()
    fig,ax1,h3,h4,h5 = plot_ode(t, X, ode_rhs, return_fig=True)
    def animate(i):
        h3.set_data(X[:(i+1)*5,0,0],X[:(i+1)*5,0,1])
        h4.set_data(t[:(i+1)*5],X[:(i+1)*5,0,0])
        h5.set_data(t[:(i+1)*5],X[:(i+1)*5,0,1])
        ax1.set_title('State trajectory until t={:.2f}'.format(5*t[i].item()), fontsize=17)
        return (h3,h4,h5,)
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=100, blit=True)
    plt.close()
    return anim

def plot_cnf_data(tr_data):
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax1 = fig.gca()
    ax1.set_title('Target samples')
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])

    ax1.hist2d(*tr_data.detach().cpu().numpy().T, bins=300, density=True,
               range=[[-1.5, 1.5], [-1.5, 1.5]]);

def plot_cnf_animation(target_sample, t0, t1, viz_timesteps, p_z0, z_t1, z_t_samples, z_t_density, logp_diff_t):
    img_path = os.path.join('etc', 'cnf')

    for (t, z_sample, z_density, logp_diff) in zip(
            np.linspace(t0, t1, viz_timesteps),
            z_t_samples, z_t_density, logp_diff_t):
        fig = plt.figure(figsize=(12, 4), dpi=200)
        plt.tight_layout()
        plt.axis('off')
        plt.margins(0, 0)
        fig.suptitle(f'{t:.2f}s')

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title('Target')
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title('Samples')
        ax2.get_xaxis().set_ticks([])
        ax2.get_yaxis().set_ticks([])
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title('Log Probability')
        ax3.get_xaxis().set_ticks([])
        ax3.get_yaxis().set_ticks([])

        ax1.hist2d(*target_sample.detach().cpu().numpy().T, bins=300, density=True,
                   range=[[-1.5, 1.5], [-1.5, 1.5]])

        ax2.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                   range=[[-1.5, 1.5], [-1.5, 1.5]])

        logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
        ax3.tricontourf(*z_t1.detach().cpu().numpy().T,
                        np.exp(logp.detach().cpu().numpy()), 200)

        plt.savefig(os.path.join(img_path, f"cnf-viz-{int(t*1000):05d}.jpg"),
                   pad_inches=0.2, bbox_inches='tight')
        plt.close()

    imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(img_path, f"cnf-viz-*.jpg")))]

    fig = plt.figure(figsize=(18,6))
    ax  = fig.gca()
    img = ax.imshow(imgs[0])

    def animate(i):
        img.set_data(imgs[i])
        return img,

    anim = animation.FuncAnimation(fig, animate, frames=41, interval=200)
    plt.close()
    return anim


def plot_mnist_sequences(X, N=5):
    N = min(5,N)
    print(f'Plotting {N} rotating MNIST sequences.')
    X = X.transpose(0,1)
    T = X.shape[1]
    X_ = X[torch.randint(0,X.shape[0],[N])].permute(0,1,3,4,2).cpu().numpy()
    plt.figure(1,(T,N))
    for n in range(N):
        for t in range(T):
            plt.subplot(N,T,n*T+t+1)
            plt.imshow(X_[n,t].squeeze(-1), cmap='gray')
            plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.show()
    plt.close()

    
def plot_mnist_latent_trajectories(zt):
    zt = zt.transpose(0,1)
    N,T,q = zt.shape
    zt = zt.detach() # N,T,q
    zt = zt.reshape(-1,q) # NT,q
    U,S,V = torch.pca_lowrank(zt)
    zt_pca = zt@V[:,:2] 
    zt_pca =  zt_pca.reshape(N,T,2).cpu().numpy() # N,T,2
    plt.figure(1,(5,5))
    for n in range(N):
        p, = plt.plot(zt_pca[n,0,0],zt_pca[n,0,1],'o',markersize=10)
        plt.plot(zt_pca[n,:,0],zt_pca[n,:,1],'-*', color=p.get_color())
    plt.xlabel('PCA-1',fontsize=15)
    plt.ylabel('PCA-2',fontsize=15)
    plt.title('Latent trajectories',fontsize=18)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    
def plot_mnist_predictions(X, zt, Xhat, N=5):
    plot_mnist_latent_trajectories(zt)
    N = min(5,N)
    print(f'Plotting {N} rotating MNIST sequences (top rows) and corresponding predictions (bottom).')
    Xhat = Xhat[0] if Xhat.ndim==6 else Xhat
    T  = X.shape[1]
    X_ = X.permute(0,1,3,4,2).cpu().numpy()
    Xhat_ = Xhat.permute(0,1,3,4,2).detach().cpu().numpy()
    plt.figure(1,(T,3*N))
    for n in range(N):
        for t in range(T):
            plt.subplot(3*N,T,3*n*T+t+1)
            plt.imshow(X_[t,n].squeeze(-1), cmap='gray')
            plt.xticks([]); plt.yticks([])
            plt.subplot(3*N,T,3*n*T+T+t+1)
            plt.imshow(Xhat_[t,n].squeeze(-1), cmap='gray')
            plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.show()
    plt.close()

