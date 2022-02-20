import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

d = 10
K = 5
A = np.zeros((K,d,d))
b = np.zeros((K,d))

for k in range(1,K+1):
    for i in range(1,d+1):
        for j in range(1,d+1):
            if (i < j):
                A[k-1,i-1,j-1] = math.exp(i/j) * math.cos(i*j) * math.sin(k)
            elif (i > j):
                A[k-1,i-1,j-1] = math.exp(j/i) * math.cos(j*i) * math.sin(k)
        b[k-1,i-1] = math.exp(i/k) * math.sin(i*k)
        A[k-1,i-1,i-1] = i/10 * abs(math.sin(k)) + np.sum(np.abs(A[k-1,i-1]))

def f_k(k, x):
    return np.dot(x, np.dot(A[k], x)) - np.dot(b[k], x)

def f(x):
    return max([f_k(k, x) for k in range(K)])

T = 100000
C = 0.5
x = np.ones(10)
f_opt = -0.8411902729911584

def sub_gradient(x):
    k = np.argmax([f_k(k, x) for k in range(K)])
    g = 2 * np.dot(A[k],x) - b[k]
    return g

fs = [f(x)]

for t in range(T-1):
    g = sub_gradient(x) 
    # eta = C/math.sqrt(t+1)            # (b)
    eta = (f(x) - f_opt) / LA.norm(g)   # (c)
    x = x - eta * g / LA.norm(g)
    fs.append(min(f(x), fs[-1]))

time = np.arange(1,T+1,1)

plt.loglog(time, [i - f_opt for i in fs])
plt.xlabel('iteration number')
plt.ylabel('sub-optimality gap')
plt.show()