import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from sklearn.model_selection import train_test_split


# Sigmoid
def sigmoidMatrix(x, j):
    X = np.empty((len(x), j+1))
    i = 0
    min = np.min(x)
    max = np.max(x)
    step = (max-min) / j
    std = np.std(x)/50
    for xi in x:
        row = np.empty(j+1)
        for n in range(j+1):
            u = n*step
            row[n] = 1 / (1 + np.exp(-1 * (xi-u)/std))
        X[i] = row
        i = i + 1
    return X

# Gaussian
def gaussianMatrix(x, j):
    X = np.empty((len(x), j+1))
    i = 0
    min = np.min(x)
    max = np.max(x)
    step = (max-min) / j
    s = np.std(x)
    s = np.mean(x)
    # s = (max-min) / j
    for xi in x:
        row = np.empty(j+1)
        for n in range(j+1):
            u = n*step
            row[n] = np.exp( -1 * ((xi - u)**2 / (2 * s**2)))
        X[i] = row
        i = i + 1
    return X

# Polynomial
def polynomialMatrix(x, degree):
    X = np.empty((len(x), degree+1))
    i = 0
    for xi in x:
        row = np.empty(degree+1)
        for n in range(degree+1):
            row[n] = xi**n
        X[i] = row
        i = i + 1
    return X

def sum_square_error(Y, T):
    return 0.5  * np.sum(np.square(np.subtract(Y, T)))


# Read file
x,t = [], []
with open('regression_x_t.txt') as file:
  i = 0
  for line in file:
    xtemp, ttemp = line.split(' ')
    i = i + 1
    x.append(float(xtemp))
    t.append(float(ttemp))

# Split file into training and Test Sets

x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.20, random_state=42)

# Convert python list to vector (Matrix)
t_train = np.array(t_train)
t_train = np.expand_dims(t_train, axis=1)

# W Coefficients
j = 20
polynomial_degree = j
learning_rate =  1e-2
# X = polynomialMatrix(x_train, polynomial_degree)
# X = gaussianMatrix(x_train, polynomial_degree)
##############################################################################
#BGD Sigmoid
X = sigmoidMatrix(x_train, j)
W = np.zeros((j+1, 1))
Y = np.dot(X, W)

err = sum_square_error(Y, t_train)
errs = []
errs.append(err)
for itr in range(20000):
    # W = np.subtract(W , (1/len(t_train)) * learning_rate * np.dot(X.T,np.subtract(Y, t_train)))
    W = np.subtract(W , (1/len(t_train)) * learning_rate * np.dot(X.T,np.subtract(Y, t_train)))
    Y = np.dot(X, W)
    err = sum_square_error(Y, t_train)
    errs.append(err)


plt.title("HW1.1.a decreasing of error function with the increasing of iteration numbers")
plt.plot(errs)
plt.xlabel("x")
plt.ylabel("Y")
plt.show()
print("HW1.1.b SGD obtained coefficient W")
print(W)

XTest = sigmoidMatrix(x_test, j)
YTestPredict = np.dot(XTest, W)
plt.title("HW1.1.c predicted  f(x) vs. Actual Target t")
plt.scatter(x_test, YTestPredict)
plt.scatter(x_test, t_test)
plt.show()
rms = np.sqrt(2 * sum_square_error(YTestPredict, t_test) / len(t_test))
print("HW1.1.d SGD Test Set RMS: ")
print(rms)



#############################################################################
#Stochastic 
randomIndexes = []
for i in range(len(t_train)):
    randomIndexes.append(i)

rnd.shuffle(randomIndexes)
W = np.zeros((j+1, 1))
Y = np.dot(X, W)
err = sum_square_error(Y, t_train)
errs = []
errs.append(err)
for itr in range(400):
    for stoc in randomIndexes:
        #get random point
        i = stoc # rnd.randint(0, len(t_train)-1)
        Xi = X[i]
        Xi = np.reshape(Xi, (1,j+1))
        Yi = np.dot(Xi, W)
        grad = Xi.T * np.subtract(Yi, t_train[i])
        W = np.subtract(W , (1.0/j) * learning_rate * grad)
    Y = np.dot(X, W)
    err = sum_square_error(Y, t_train)
    errs.append(err)

plt.title("HW1.2.a decreasing of error function with the increasing of iteration numbers")
plt.plot(errs)
plt.xlabel("x")
plt.ylabel("Y")
plt.show()

print("HW1.2.b Stochastic obtained coefficient W")
print(W)

YTestPredict = np.dot(XTest, W)
plt.title("HW1.2.c predicted  f(x) vs. Actual Target t")
plt.scatter(x_test, YTestPredict)
plt.scatter(x_test, t_test)
plt.show()
rms = np.sqrt(2 * sum_square_error(YTestPredict, t_test) / len(t_test))

print("Stochastic  RMS: ")
print(rms)


#############################################################################
#Maximum likelihood 

MLHW =  np.dot( np.dot( np.linalg.inv( np.dot(X.T, X)), X.T), t_train)
YTestPredict = np.dot(XTest, MLHW)
plt.title("HW1.3.c predicted  f(x) vs. Actual Target t")
plt.scatter(x_test, YTestPredict)
plt.scatter(x_test, t_test)
plt.show()
rms = np.sqrt(2 * sum_square_error(YTestPredict, t_test) / len(t_test))
print("HW1.3.a Maximum Likelihood W")
print(MLHW)

print("HW1.3.d Maximum Likelihood RMS: ")
print(rms)

