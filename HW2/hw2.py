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

mu = 0 
sigma = 0.2
r = np.arange(-1, 1, 0.05)
beta = 25
alpha = 2.0
covariance_matrix = (1/alpha) * np.identity(2, dtype=float)
W = np.zeros((2, 1))
X = np.random.uniform(-1, 1, 40)
X = np.reshape(X,(len(X), 1))
t = np.empty(X.shape)
for i in range(len(X)):
    t[i] = calc_t(X[i])+ np.random.normal(mu, 1/beta)
posterior_matrix = np.ones((len(X), len(t)))
likelihood_matrix = np.ones((len(X), len(t)))

rows = len(r) + 1
cols = 3
fig, axes = plt.subplots(rows,cols, figsize=(cols*2 , rows*2 ), dpi=80)
fig.subplots_adjust(hspace = 1, wspace=1)

axs = axes.ravel()
pltctr = 0
axs[pltctr].plot(0 ,0)
for ctr in range(len(r)):
    for i, w0 in enumerate(r):
        for j, w1 in enumerate(r):
            if ctr == 0:
                posterior_matrix[j][i] = prior(np.array([[w0], [w1]]), 0, covariance_matrix)
            likelihood_matrix[j][i] = np.exp((-beta/2.0) * (t[ctr] - (w0 + w1*X[ctr]))**2)
    posterior_matrix = posterior_matrix * likelihood_matrix
    pltctr += 1
    axs[pltctr].contourf(r, r, posterior_matrix, cmap='jet')
    axs[pltctr].set_title("Prior/Posterior")
    axs[pltctr].set(xlabel='w0', ylabel='w1')
    # x = np.linspace(-1,1,len(r))
    # y = 0.5*x-0.3
    # plt.plot(x, y, '-r', label='y=2x+1')
    axs[pltctr].plot(-0.3, 0.5, marker="+")
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
    for k in range(ctr):
        markersX.append(X[k])
        markersY.append(t[k])
    axs[pltctr].plot(markersX, markersY, 'o', color='blue')
    lim = 1.0
    axs[pltctr].set_xlim([-lim, lim])
    axs[pltctr].set_ylim([-lim, lim])
    axs[pltctr].set(xlabel='x', ylabel='y')
    axs[pltctr].set_title("Data space")


    # plt.show()
    pltctr += 1
    axs[pltctr].contourf(r, r, likelihood_matrix, cmap='jet')
    axs[pltctr].set_title("Likelihood")
    axs[pltctr].set(xlabel='w0', ylabel='w1')
    # plt.show()
    
    # plt.show()
plt.savefig('/home/almajea/code/python/CS229ML/HW2/plt.png')
# plt.show()