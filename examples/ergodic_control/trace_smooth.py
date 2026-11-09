"""Goal 2: running estimate of the control cost E_nu[c] vs simulation time, so INS's
convergence to the exact lambda can be watched against plain FV plateauing at its
finite-N bias.  One (sampler, seed) per call -> saves the running trace.

    python trace_smooth.py --sampler ins --seed 0 --T 50 --N 20 --h0 0.1
"""
import argparse, json, os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import fleming_viot, fleming_viot_vanilla
import problem_smooth as P
L=P.L; HERE=os.path.dirname(__file__)

def running_estimate(tgrid, ti, a_k, b_k):
    """cumulative num/den (a_k,b_k per-event) sampled onto tgrid."""
    A=np.cumsum(a_k); B=np.cumsum(b_k)
    # for each grid time, index of last event <= t
    idx=np.searchsorted(ti, tgrid, side="right")-1
    est=np.full(len(tgrid), np.nan)
    ok=idx>=0
    est[ok]=A[idx[ok]]/np.clip(B[idx[ok]],1e-300,None)
    return est

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sampler",choices=["ins","fv"],required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--T",type=float,default=50.0)
    ap.add_argument("--N",type=int,default=20)
    ap.add_argument("--h0",type=float,default=0.1)
    args=ap.parse_args()
    rng=np.random.default_rng(args.seed); a=P.diffusion(P.a_coeff); N=args.N
    if args.sampler=="ins":
        x0=rng.uniform(0,L,(N,P.dim)); y0=rng.uniform(0,L,(N,P.dim))
        px,py,ti,rho=fleming_viot(args.T,P.V,P.DV,P.D2V,a,P.a_coeff,P.c,args.h0,x0.copy(),y0.copy())
        dt=np.append(np.diff(ti),0.0)
        cpx=np.stack([P.c(px[:,i,:]) for i in range(N)],1)   # (E,N)
        cpy=np.stack([P.c(py[:,i,:]) for i in range(N)],1)
        per=np.sum(cpx*rho+cpy*(1-rho),1)                    # sum over particles (E,)
        a_k=dt*per; b_k=N*dt
    else:
        x0=rng.uniform(0,L,(N,P.dim))
        pv,ti=fleming_viot_vanilla(args.T, lambda x:-P.DV(x), a, P.c, args.h0, x0.copy())
        dt=np.append(np.diff(ti),0.0)
        per=np.sum(np.stack([P.c(pv[:,i,:]) for i in range(N)],1),1)
        a_k=dt*per; b_k=N*dt
    tgrid=np.linspace(0,args.T,300)
    est=running_estimate(tgrid, ti, a_k, b_k)
    out=dict(sampler=args.sampler,seed=args.seed,T=args.T,N=args.N,h0=args.h0,
             tgrid=tgrid.tolist(), est=est.tolist(), n_events=int(len(ti)))
    d=os.path.join(HERE,"trace_out"); os.makedirs(d,exist_ok=True)
    fn=os.path.join(d,f"{args.sampler}_s{args.seed}.json")
    json.dump(out, open(fn,"w"))
    print(f"[{args.sampler} s{args.seed}] events={len(ti)}  final est={est[-1]:.4f}  -> {fn}")

if __name__=="__main__": main()
