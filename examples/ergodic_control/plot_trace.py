"""Plot goal-2 convergence: running control-cost estimate vs simulation time,
INS vs plain FV, several seeds, against exact lambda."""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(__file__)
fd=np.load(os.path.join(HERE,"fd_smooth.npy"),allow_pickle=True).item(); lam=fd["lam"]

def load(s):
    rows=[json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE,"trace_out",f"{s}_s*.json")))]
    return rows
ins=load("ins"); fv=load("fv")
fig,ax=plt.subplots(figsize=(7.6,5.0))
tmin=4.0
def band(rows,color,label):
    t=np.array(rows[0]["tgrid"]); E=np.array([r["est"] for r in rows])
    m=t>=tmin
    for e in E: ax.plot(t[m], e[m], color=color, alpha=0.25, lw=1.0)
    mean=np.nanmean(E,0)
    ax.plot(t[m], mean[m], color=color, lw=2.4, label=label)
band(fv,"0.55","plain FV (no swap)")
band(ins,"0.15","INS (infinite swap)")
ax.axhline(lam,color="red",ls="--",lw=1.5,label=f"exact $\\lambda$ = {lam:.3f}")
ax.set_xlabel("simulation time  T  (budget)")
ax.set_ylabel(r"running control-cost estimate  $E_\nu[c]$")
ax.set_title("INS converges to the optimal control cost; plain FV plateaus at its bias\n"
             "(N=20, h0=0.1, %d seeds each)"%len(ins),fontsize=11)
ax.legend(fontsize=9,loc="upper right")
ax.set_ylim(lam-0.15, lam+0.55)
fig.tight_layout()
out=os.path.join(HERE,"figures","convergence_smooth.png")
fig.savefig(out,dpi=150,bbox_inches="tight")
# report final plateaus
fi=np.nanmean([r["est"][-1] for r in ins]); ff=np.nanmean([r["est"][-1] for r in fv])
print("saved",out)
print("final plateau: INS=%.4f (%.1f%%)  FV=%.4f (%.1f%%)  lambda=%.4f"
      %(fi,100*(fi-lam)/lam, ff,100*(ff-lam)/lam, lam))
