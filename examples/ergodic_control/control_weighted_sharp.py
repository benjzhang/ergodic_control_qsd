"""Control from WEIGHTED-DSM: score matching on the full fwd/bwd weighted empirical
measure (per-batch draws prop. to dt*rho / dt*(1-rho)), NO pre-resampling.

Same smoothing config as the 12%-gap resampled run (harmonics=3, sigma=0.10, wd=1e-4)
so the only change is weighted-vs-resampled.  Reports rho*-match + cost gap for
direct A/B comparison.
"""
import argparse, os
import numpy as np
import scipy.ndimage as ndi
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import fleming_viot
import problem as P

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

def train_weighted(pool_t, Wcum, epochs, sigma, n, wd, batch=8192, lr=1e-3, seed=0):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    m=PNet(n=n); m.train(); opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=wd)
    tot=Wcum[-1]
    for ep in range(epochs):
        u=rng.random(batch)*tot
        idx=np.searchsorted(Wcum,u)                    # draw ~ dt*rho / dt*(1-rho) from FULL pool
        x=pool_t[idx]; eps=torch.randn_like(x)
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
    ap.add_argument("--T",type=float,default=50.0); ap.add_argument("--N",type=int,default=20)
    ap.add_argument("--h0",type=float,default=0.1); ap.add_argument("--burn",type=float,default=0.3)
    ap.add_argument("--epochs",type=int,default=8000); ap.add_argument("--sigma",type=float,default=0.10)
    ap.add_argument("--harmonics",type=int,default=3); ap.add_argument("--wd",type=float,default=1e-4)
    ap.add_argument("--walkers",type=int,default=8000); ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args()
    rng=np.random.default_rng(args.seed); adiff=P.diffusion(P.a_coeff); N=args.N
    fd=np.load(os.path.join(HERE,"fd_reference.npy"),allow_pickle=True).item(); lam=fd["lam"]; Nfd=fd["N"]

    print("running INS ...",flush=True)
    x0=rng.uniform(0,L,(N,P.dim)); y0=rng.uniform(0,L,(N,P.dim))
    px,py,ti,rho=fleming_viot(args.T,P.V,P.DV,P.D2V,adiff,P.a_coeff,P.c,args.h0,x0.copy(),y0.copy())
    mburn=ti>args.burn*args.T; px,py,ti,rho=px[mburn],py[mburn],ti[mburn],rho[mburn]
    dt=np.append(np.diff(ti),0.0)

    # FULL weighted fwd/bwd pool -- NO resampling
    pool=np.concatenate([P.wrap(px[:,i,:]) for i in range(N)]+[P.wrap(py[:,i,:]) for i in range(N)]).astype(np.float32)
    w=np.concatenate([dt*rho[:,i] for i in range(N)]+[dt*(1-rho[:,i]) for i in range(N)])
    w=np.clip(w,0,None); Wcum=np.cumsum(w)
    print(f"weighted pool: {len(pool)} points (vs 40k resampled)",flush=True)
    pool_t=torch.tensor(pool)

    print("training weighted DSM ...",flush=True)
    model=train_weighted(pool_t,Wcum,args.epochs,args.sigma,args.harmonics,args.wd)

    Ng=160; h=L/Ng; g=(np.arange(Ng)+0.5)*h
    X1,X2=np.meshgrid(g,g,indexing="ij"); pts=np.stack([X1.ravel(),X2.ravel()],1)
    hh=model.hgrad(torch.tensor(P.wrap(pts),dtype=torch.float32)).detach().numpy()
    gV=P.DV(pts); s=hh-gV
    drift_rho=(gV+a*s).reshape(Ng,Ng,2); u=(a*s+2*gV)

    print("running controlled process ...",flush=True)
    rho_s=simulate(drift_rho,h,args.walkers,15000,1e-3,0.3,rng)
    ui=interp(u.reshape(Ng,Ng,2),rho_s,h)
    J=np.mean(P.c(rho_s)+np.sum(ui**2,1)/(2*a)); Ec=np.mean(P.c(rho_s))
    Hh,_,_=np.histogram2d(rho_s[:,0],rho_s[:,1],bins=Nfd,range=[[0,L],[0,L]],density=True)
    rho_fd=fd["rho"].reshape(Nfd,Nfd); rho_fd=rho_fd/(rho_fd.sum()*(L/Nfd)**2)
    from scipy.ndimage import gaussian_filter
    Hs=gaussian_filter(Hh,3,mode="wrap"); corr=np.corrcoef(Hs.ravel(),rho_fd.ravel())[0,1]
    print(f"WEIGHTED-DSM: J={J:.4f} lambda={lam:.4f} gap={100*(J-lam)/lam:+.1f}%  rho_corr={corr:.3f}  energy={J-Ec:.4f}")

    # ---- save data for "how the method works" figures (no re-run needed later) ----
    E=px.shape[0]
    fwd_pool=pool[:N*E]; bwd_pool=pool[N*E:]; fwd_w=w[:N*E]; bwd_w=w[N*E:]
    def _rs(pp,ww,k):
        ww=np.clip(ww,0,None); cc=np.cumsum(ww); uu=rng.random(k)*cc[-1]; return pp[np.searchsorted(cc,uu)]
    subr=rho_s[rng.choice(len(rho_s),min(12000,len(rho_s)),replace=False)]
    np.savez(os.path.join(HERE,"howitworks_data_sharp.npz"),
             g=g, u=u.reshape(Ng,Ng,2), drift_rho=drift_rho,
             rho_samples=subr,
             nu_fwd=_rs(fwd_pool,fwd_w,4000), nu_bwd=_rs(bwd_pool,bwd_w,4000),
             lam=lam, sigma_smooth=P.sigma)
    torch.save(model.state_dict(), os.path.join(HERE,"wdsm_model_sharp.pt"))
    print("saved howitworks_data_sharp.npz + wdsm_model_sharp.pt")

    fig,ax=plt.subplots(1,2,figsize=(11,5.3),sharex=True,sharey=True)
    ax[0].imshow(rho_fd.T,origin="lower",extent=[0,L,0,L],cmap="gray"); ax[0].set_title(r"exact optimal $\rho^*$ (FD)")
    ax[1].imshow(Hs.T,origin="lower",extent=[0,L,0,L],cmap="gray")
    ax[1].set_title("controlled under weighted-DSM $u^*$")
    for a_ in ax: a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"Weighted-DSM control (full fwd/bwd measure): $\\rho^*$ corr {corr:.2f}, "
                 f"cost J={J:.3f} vs $\\lambda$={lam:.3f} ({100*(J-lam)/lam:+.0f}%)",fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(os.path.join(HERE,"figures","control_weighted_sharp.png"),dpi=150,bbox_inches="tight")
    open(os.path.join(HERE,"control_weighted_sharp.txt"),"w").write(
        f"WEIGHTED: J={J:.4f} lambda={lam:.4f} gap={(J-lam)/lam:.4f} rho_corr={corr:.4f} energy={J-Ec:.4f}\n")
    print("saved figures/control_weighted_smooth.png")

if __name__=="__main__": main()
