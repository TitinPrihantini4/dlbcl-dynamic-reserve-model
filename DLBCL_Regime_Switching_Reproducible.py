import numpy as np, pandas as pd
from scipy.stats import qmc, rankdata
BASE=dict(g=.22,K=1.,e=.62,rh=.18,rf=.15,th=.17,tf=.15,dh=.10,df=.08,ch=.04,cf=.05,C=.35)
def RI(x,p):
 L,H,F,D=x; return np.sqrt(max(H,1e-12)*max(F,1e-12))/(1+.25*max(L,0)+.25*p["C"])
def ui(r): return 1. if r>=.62 else (.75 if r>=.42 else .55)
def deriv(x,p):
 L,H,F,D=x;u=ui(RI(x,p));return np.array([p["g"]*L*(1-L/p["K"])-p["e"]*u*L,p["rh"]*(1-H)*F-p["th"]*u*H-p["dh"]*L*H-p["ch"]*p["C"]*H,p["rf"]*(1-F)*H-p["tf"]*u*F-p["df"]*L*F-p["cf"]*p["C"]*F,u/6])
def sim(x0=(.75,.72,.68,0),mods=None,n=600):
 p=BASE.copy();p.update(mods or {});dt=6/n;x=np.array(x0,float)
 for _ in range(n):
  k1=deriv(x,p);k2=deriv(x+dt*k1/2,p);k3=deriv(x+dt*k2/2,p);k4=deriv(x+dt*k3,p);x=x+dt*(k1+2*k2+2*k3+k4)/6;x[:3]=np.clip(x[:3],0,1)
 return x,p
print(sim())
