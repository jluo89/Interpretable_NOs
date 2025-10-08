import math
import tqdm
import numpy as np

from scipy.io import savemat

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--t', type=int, default=1, help="Specify the time")
args = parser.parse_args()

def u_function(aij, x, y, t, k=36, c=0.1, r=1):
    
    i = np.linspace(1,k,k)
    j = np.linspace(1,k,k)

    sinix = np.sin(x*math.pi*i)
    sinjy = np.sin(y*math.pi*j)

    i = i[:, np.newaxis]
    j = j[np.newaxis, :]

    matrix = i**2 + j**2
    sqrt_matrix = np.sqrt(i**2 + j**2)
    r_matrix = np.power(matrix, -r)

    cosijt = np.cos(c*math.pi*t*sqrt_matrix)

    sinix = np.expand_dims(sinix,axis=1)
    sinjy = np.expand_dims(sinjy,axis=0)

    sumup = sinix@sinjy

    init = aij*sumup*r_matrix

    result = math.pi*np.sum(init*cosijt)/(k**2)
    init = math.pi*np.sum(init)/(k**2)
    
    return (init, result)


t = args.t
K = 24
r = 64
num_example = 1000
aij = np.random.uniform(-1, 1, (num_example, K, K))

file_name = f'wave_{r}x{r}_in_{t}.mat'
x = np.linspace(0,1,r)
y = np.linspace(0,1,r)
init = np.zeros((num_example,r,r))
wave = np.zeros((num_example,r,r))

p_bar = tqdm.tqdm(desc=f"Generating {num_example}",position=0,total=num_example)

for n in range(num_example):
    p_bar.update()
    aij_sample = aij[n,:,:]

    for m in range(len(x)):
        x_prime = x[m]
        for i in range(len(y)):
            init[n,m,i], wave[n,m,i] = u_function(aij[n],x[m],y[i],t,k=K)
            # test = u_function(aij=aij_sample,x=x_prime,y=y[i],t=t)
p_bar.close()

savemat(file_name,{'x':init,'y':wave})