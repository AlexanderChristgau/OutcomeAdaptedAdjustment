import numpy as np

expit = lambda x: 1/(1+np.exp(-x))
eps_m0 = 0.01

m0 = lambda w,_: eps_m0 + (1 - 2*eps_m0)*(w[:,0]>0.5)

bump = lambda w,idx: (w[:,:len(idx)//2]@idx[:len(idx)//2])>0.5*(idx[:len(idx)//2]).sum()
m1 = lambda w,idx: eps_m0 + (1 - 2*eps_m0)*(bump(w,idx))

g_lin = lambda t,z: t+3*z
g_sqr = lambda t,z: z**(1+t)
g_sin = lambda t,z: (3+t)*np.sin(np.pi*z)
g_sinc = lambda t,z: (3+t)*np.sinc(np.pi*z)
g_cbrt = lambda t,z: (2+t)*np.cbrt(z)

link_dic = {'lin':g_lin,'sin':g_sin,'sqr':g_sqr,'cbrt':g_cbrt,'sinc':g_sinc}

def sample_idx(rng,d=10):
    indexY = np.hstack([np.ones(1),rng.normal(size=(d-1))/np.sqrt(d-1)])
    #indexY = rng.normal(size=d)/np.sqrt(d)
    return indexY


def compute_IM(rng,indexY,link='sin',N=25000,t=1):
    d = len(indexY)
    g = link_dic[link]
    W = rng.uniform(0,1,size=(N,d))
    
    return g(t*np.ones(N),W@indexY).mean()


def compute_ATE(rng,indexY,link='sin',N=25000):
    d = len(indexY)
    g = link_dic[link]
    W = rng.uniform(0,1,size=(N,d)) 
    
    return g(np.ones(N),W@indexY).mean() - g(np.zeros(N),W@indexY).mean()


def sample_SI(rng,indexY,n=500,link='sin',Ystd=1):
    d = len(indexY)
    g = link_dic[link]
    W = rng.uniform(0,1,size=(n,d)) 
    T = rng.binomial(1,m0(W,indexY))
    Y = g(T,W@indexY) + Ystd*rng.normal(size=(n,))
    
    return W,T,Y
