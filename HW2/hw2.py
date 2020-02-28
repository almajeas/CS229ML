import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

def calc_t(x):
    return -0.3 + 0.5 * x

def prior(W, m, cov):
    return np.exp((-1.0/2)*(W-m).T.dot(np.dot(np.linalg.inv(cov),(W-m))))

def plot_sampled_points(X, t):
    points = pd.DataFrame({'x': X[:, 0], 'y': t[:, 0]})
    sns.lmplot(x='x', y='y', data=points , height=40)
    lim = 1.3
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.show()

#data range
lim = 1.0
r = np.arange(-lim, lim, 0.05)
num = len(r)

#plot configuration initialization
rows = num 
cols = 3
fig, axes = plt.subplots(rows,cols, figsize=(cols*4 , rows*4 ), dpi=100)
fig.subplots_adjust(hspace = 0.3, wspace=0.3)
axs = axes.ravel()
pltctr = 0

#Parameter Initialization
alpha = 2.0
beta = 25
mu = 0 
sigma = 0.2
covariance_matrix = (1/alpha) * np.identity(2, dtype=float)

#Matrix population
W = np.zeros((2, 1))
X = np.random.uniform(-lim, lim, len(r))
X = np.reshape(X,(len(X), 1))
t = np.empty(X.shape)

#Calculating t and addition noise
for i in range(len(X)):
    t[i] = calc_t(X[i])+ np.random.normal(mu, 1/beta)
posterior_matrix = np.ones((len(X), len(t)))
likelihood_matrix = np.ones((len(X), len(t)))


for ctr in range(num):
    for i, w0 in enumerate(r):
        for j, w1 in enumerate(r):
            if ctr == 0:
                posterior_matrix[j][i] = prior(np.array([[w0], [w1]]), 0, covariance_matrix)
            likelihood_matrix[j][i] = np.exp((-beta/2.0) * (t[ctr] - (w0 + w1*X[ctr]))**2)
    
    ## Plotting Prior/Posterior
    pltctr += 1
    axs[pltctr].set_xlim([-lim, lim])
    axs[pltctr].set_ylim([-lim, lim])
    axs[pltctr].contourf(r, r, posterior_matrix, cmap='jet')
    axs[pltctr].set_title("Prior/Posterior")
    axs[pltctr].set(xlabel='w0', ylabel='w1')
    axs[pltctr].plot(-0.3, 0.5, marker="+", color='w')
    
    ##Sampling and plotting dataspace
    pltctr += 1
    for i in range(6):
        xs, ys = np.unravel_index(np.random.choice(posterior_matrix.size, p=posterior_matrix.ravel()/float(posterior_matrix.sum())), posterior_matrix.shape)
        w1 = 2 * ((xs)/(len(r))) - 1 #Normalize index to -1, 1
        w0 = 2 * ((ys)/(len(r))) - 1 #Normalize index to -1, 1
        x = np.linspace(-1,1,len(r))
        y = w1*x + w0
        axs[pltctr].plot(x, y, '-r')
    markersX = []
    markersY = []
    for k in range(ctr+1):
        markersX.append(X[k])
        markersY.append(t[k])
    axs[pltctr].plot(markersX, markersY, 'o', markerfacecolor='none')
    axs[pltctr].set_xlim([-lim, lim])
    axs[pltctr].set_ylim([-lim, lim])
    axs[pltctr].set(xlabel='x', ylabel='y')
    axs[pltctr].set_title("Data space")

    posterior_matrix = posterior_matrix * likelihood_matrix
    if ctr == num - 1:
        break
    ##Plotting likelihood
    pltctr += 1
    axs[pltctr].set_xlim([-lim, lim])
    axs[pltctr].set_ylim([-lim, lim])
    axs[pltctr].contourf(r, r, likelihood_matrix, cmap='jet')
    axs[pltctr].set_title("Likelihood")
    axs[pltctr].set(xlabel='w0', ylabel='w1')

plt.savefig('./plt.png')
# plt.show()