"""'How the method works' — 3-stage pipeline in samples/fields.
Backdrops are the MEANINGFUL densities (nu, killing c, rho*), faint, so the samples
and the concentration pop -- not the busy 16-well potential.
"""
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import problem_smooth as P

L=P.L; HERE=os.path.dirname(__file__)
d=np.load(os.path.join(HERE,"howitworks_data_sharp.npz"))
g=d["g"]; u=d["u"]; rho_s=d["rho_samples"]; nu_f=d["nu_fwd"]; nu_b=d["nu_bwd"]
fd=np.load(os.path.join(HERE,"fd_reference.npy"),allow_pickle=True).item(); N=fd["N"]
nu=fd["nu"].reshape(N,N); rho=fd["rho"].reshape(N,N); cc=fd["c"].reshape(N,N)
rng=np.random.default_rng(0)

def back(ax,F,alpha=0.85):
    ax.imshow(F.T,origin="lower",extent=[0,L,0,L],cmap="Greys",alpha=alpha)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0,L); ax.set_ylim(0,L)

fig,ax=plt.subplots(1,3,figsize=(15.6,5.5))

# (1) samples of nu over the nu density (broad)
nus=np.concatenate([nu_f,nu_b]); nus=nus[rng.choice(len(nus),min(4000,len(nus)),replace=False)]
back(ax[0],nu); ax[0].scatter(nus[:,0],nus[:,1],s=5,c="crimson",alpha=0.5,linewidths=0)
ax[0].set_title("1.  Sample the QSD $\\nu$\n(forward/backward Fleming–Viot)",fontsize=12)

# (2) learned control field over the killing c (high center -> arrows push mass out)
Ng=u.shape[0]; gu=g; XU,YU=np.meshgrid(gu,gu,indexing="ij")
us=gaussian_filter(u,(1.0,1.0,0),mode="wrap")     # light display de-noise
st=max(1,Ng//20); sub=(slice(None,None,st),slice(None,None,st))
back(ax[1],cc,alpha=0.7)
ax[1].quiver(XU[sub],YU[sub],us[...,0][sub],us[...,1][sub],color="red",scale=95,width=0.005)
ax[1].set_title("2.  Learn the control $u^*=a\\,\\nabla\\log\\phi$\n(score matching on the samples)",fontsize=12)

# (3) controlled samples over rho* (concentrated in the low-cost corners)
rs=rho_s[rng.choice(len(rho_s),min(4000,len(rho_s)),replace=False)]
back(ax[2],rho); ax[2].scatter(rs[:,0],rs[:,1],s=5,c="crimson",alpha=0.5,linewidths=0)
ax[2].set_title("3.  Apply the control\n(process driven to the optimal $\\rho^*$)",fontsize=12)

fig.suptitle("Sample the killed system's QSD  →  learn the control from the samples  →  the controlled process reaches $\\rho^*$",
             fontsize=13)
fig.tight_layout(rect=[0,0,1,0.94])
out=os.path.join(HERE,"figures","howitworks_sharp.png")
fig.savefig(out,dpi=150,bbox_inches="tight")
print("saved",out)
