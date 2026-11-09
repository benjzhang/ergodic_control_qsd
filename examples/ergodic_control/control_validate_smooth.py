"""Goal 1: does the INS-learned control produce the RIGHT control? (smoothed score)

Train analytic-baseline score (h ~ grad log phi) on INS samples, build the controlled
drift, run the controlled process, check (a) rho*-match and (b) ergodic cost ~ lambda.
Score is smoothed (few harmonics, larger DSM sigma, weight decay) to cut the pointwise
noise that inflates the control energy |u*|^2.
"""
import argparse, os
import numpy as np
import scipy.ndimage as ndi
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import problem_smooth as P

L=P.L; a=P.a_coeff; SW=P.sigma; HERE=os.path.dirname(__file__)
CENT=torch.tensor(P.centers, dtype=torch.float32)

def ff(x,n):
    f=[]
    for k in range(1,n+1): f+=[torch.cos(2*np.pi*k*x/L),torch.sin(2*np.pi*k*x/L)]
    return torch.cat(f,-1)
def DV_torch(x):
    d=x[:,None,:]-CENT; d=(d+L/2)%L-L/2
    q=-(d**2).sum(-1)/(2*SW**2); w=torch.softmax(q,1)
    return (w[...,None]*d/SW**2).sum(1)
class PNet(nn.Module):
    def __init__(s,n=3,h=256,d=3):
        super().__init__(); s.n=n
        ly=[nn.Linear(4*n,h),nn.SiLU()]
        for _ in range(d-1): ly+=[nn.Linear(h,h),nn.SiLU()]
        ly+=[nn.Linear(h,1)]; s.net=nn.Sequential(*ly)
    def phi(s,x): return s.net(ff(x,s.n)).squeeze(-1)
    def hgrad(s,x):
        x=x.detach().requires_grad_(True)
        (g,)=torch.autograd.grad(s.phi(x).sum(),x,create_graph=s.training); return g
def train(samp,epochs,sigma,n,wd,batch=16384,lr=1e-3,seed=0):
    torch.manual_seed(seed); X=torch.tensor(samp,dtype=torch.float32)
    m=PNet(n=n); m.train(); opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=wd); M=X.shape[0]
    for ep in range(epochs):
        i=torch.randint(0,M,(batch,)); x=X[i]; eps=torch.randn_like(x)
        xt=torch.remainder(x+sigma*eps,L)
        s=m.hgrad(xt)-DV_torch(xt).detach()
        loss=((s-(-eps/sigma))**2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1)%max(1,epochs//6)==0: print(f"  ep {ep+1}/{epochs} loss {loss.item():.2f}",flush=True)
    m.eval(); return m

def interp(field,X,h):
    c=np.stack([X[:,0]/h-0.5,X[:,1]/h-0.5])
    d0=ndi.map_coordinates(field[...,0],c,order=1,mode="grid-wrap")
    d1=ndi.map_coordinates(field[...,1],c,order=1,mode="grid-wrap")
    return np.stack([d0,d1],1)
def simulate(field,h,walkers,steps,dt,burn,rng):
    X=rng.uniform(0,L,(walkers,2)); sq=np.sqrt(a*dt); keep=[]
    for k in range(steps):
        X=np.mod(X+interp(field,X,h)*dt+sq*rng.standard_normal(X.shape),L)
        if k>burn*steps and k%10==0: keep.append(X.copy())
    return np.concatenate(keep,0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=8000)
    ap.add_argument("--sigma",type=float,default=0.10)
    ap.add_argument("--harmonics",type=int,default=3)
    ap.add_argument("--wd",type=float,default=1e-4)
    ap.add_argument("--walkers",type=int,default=8000)
    args=ap.parse_args()
    rng=np.random.default_rng(0)
    d=np.load(os.path.join(HERE,"qsd_samples_smooth.npy"),allow_pickle=True).item()
    fd=np.load(os.path.join(HERE,"fd_smooth.npy"),allow_pickle=True).item()
    lam=fd["lam"]; Nfd=fd["N"]
    print(f"training smoothed score (harm={args.harmonics}, sigma={args.sigma}, wd={args.wd}) ...",flush=True)
    model=train(d["samples"],args.epochs,args.sigma,args.harmonics,args.wd)

    Ng=160; h=L/Ng; g=(np.arange(Ng)+0.5)*h
    X1,X2=np.meshgrid(g,g,indexing="ij"); pts=np.stack([X1.ravel(),X2.ravel()],1)
    hh=model.hgrad(torch.tensor(P.wrap(pts),dtype=torch.float32)).detach().numpy()
    gV=P.DV(pts); s=hh-gV
    drift_rho=(gV+a*s).reshape(Ng,Ng,2)
    u=(a*s+2*gV)

    print("running controlled process ...",flush=True)
    rho_s=simulate(drift_rho,h,args.walkers,15000,1e-3,0.3,rng)

    ui=interp(u.reshape(Ng,Ng,2),rho_s,h)
    J=np.mean(P.c(rho_s)+np.sum(ui**2,1)/(2*a)); Ec=np.mean(P.c(rho_s))
    print(f"achieved ergodic cost J = {J:.4f}   (exact lambda = {lam:.4f})   [E_rho[c]={Ec:.4f}, energy={J-Ec:.4f}]")

    Hh,_,_=np.histogram2d(rho_s[:,0],rho_s[:,1],bins=Nfd,range=[[0,L],[0,L]],density=True)
    rho_fd=fd["rho"].reshape(Nfd,Nfd); rho_fd=rho_fd/(rho_fd.sum()*(L/Nfd)**2)
    from scipy.ndimage import gaussian_filter
    Hs=gaussian_filter(Hh,3,mode="wrap")
    corr=np.corrcoef(Hs.ravel(),rho_fd.ravel())[0,1]
    print(f"controlled rho* vs FD rho*: corr = {corr:.3f}")

    fig,ax=plt.subplots(1,2,figsize=(11,5.3),sharex=True,sharey=True)
    ax[0].imshow(rho_fd.T,origin="lower",extent=[0,L,0,L],cmap="gray")
    ax[0].set_title(r"exact optimal $\rho^*$ (FD)")
    ax[1].imshow(Hs.T,origin="lower",extent=[0,L,0,L],cmap="gray")
    ax[1].set_title("controlled process under INS-learned $u^*$")
    for a_ in ax: a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"INS-learned control drives the system to the optimal $\\rho^*$ (corr {corr:.2f}); "
                 f"cost J={J:.3f} vs optimal $\\lambda$={lam:.3f} ({100*(J-lam)/lam:+.0f}%)",fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(os.path.join(HERE,"figures","control_validate_smooth.png"),dpi=150,bbox_inches="tight")
    print("saved figures/control_validate_smooth.png")
    open(os.path.join(HERE,"control_validate_smooth.txt"),"w").write(
        f"harm={args.harmonics} sigma={args.sigma} wd={args.wd}: J={J:.4f} lambda={lam:.4f} "
        f"gap={(J-lam)/lam:.4f} rho_corr={corr:.4f} Ec={Ec:.4f} energy={J-Ec:.4f}\n")

if __name__=="__main__": main()
