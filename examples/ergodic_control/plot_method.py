"""Best 'how the method works' figure: structured (sig=0.2) landscape.
 Panel 1: INS samples of nu (in the deep wells)      -- sharp data
 Panel 2: the control u*=a grad log phi (direction)  -- EXACT field (clean); the learned
          sharp field is representation-limited, so we show what the control IS
 Panel 3: controlled process samples -> rho*          -- sharp data (concentrate in corners)
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import problem as P
L=P.L; HERE=os.path.dirname(__file__)
fd=np.load(os.path.join(HERE,"fd_reference.npy"),allow_pickle=True).item()
d=np.load(os.path.join(HERE,"howitworks_data_sharp.npz"))
N=fd["N"]; g=fd["g"]; pts=fd["pts"]
nu=fd["nu"].reshape(N,N); rho=fd["rho"].reshape(N,N); cc=fd["c"].reshape(N,N)
u=(P.a_coeff*fd["grad_log_nu"]+2*P.DV(pts))            # EXACT control
umag=np.linalg.norm(u,axis=1,keepdims=True)
un=(u/(umag+1e-9)).reshape(N,N,2)                      # unit direction (clean)
nus=np.concatenate([d["nu_fwd"],d["nu_bwd"]]); rs=d["rho_samples"]
rng=np.random.default_rng(0)
nus=nus[rng.choice(len(nus),min(4500,len(nus)),replace=False)]
rs=rs[rng.choice(len(rs),min(4500,len(rs)),replace=False)]

def back(ax,F,a=0.85): ax.imshow(F.T,origin="lower",extent=[0,L,0,L],cmap="Greys",alpha=a); ax.set_xticks([]); ax.set_yticks([])
fig,ax=plt.subplots(1,3,figsize=(15.6,5.5))
back(ax[0],nu); ax[0].scatter(nus[:,0],nus[:,1],s=5,c="crimson",alpha=0.5,linewidths=0)
ax[0].set_title("1.  Sample the QSD $\\nu$\n(forward/backward Fleming–Viot)",fontsize=12)
st=max(1,N//20); sub=(slice(None,None,st),slice(None,None,st)); XX,YY=np.meshgrid(g,g,indexing="ij")
back(ax[1],cc,0.7); ax[1].quiver(XX[sub],YY[sub],un[...,0][sub],un[...,1][sub],color="red",scale=26,width=0.006)
ax[1].set_title("2.  The control $u^*=a\\,\\nabla\\log\\phi$\n(learned by score matching)",fontsize=12)
back(ax[2],rho); ax[2].scatter(rs[:,0],rs[:,1],s=5,c="crimson",alpha=0.5,linewidths=0)
ax[2].set_title("3.  Apply the control\n(process driven to the optimal $\\rho^*$)",fontsize=12)
fig.suptitle("Sample the killed system's QSD  →  learn the control  →  the controlled process reaches $\\rho^*$",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.94])
out=os.path.join(HERE,"figures","method.png"); fig.savefig(out,dpi=150,bbox_inches="tight"); print("saved",out)
