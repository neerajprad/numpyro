from __future__ import print_function, division
import numpy as vnp
import jax.numpy as np
from jax import grad, value_and_grad, vmap
import time

from implicit import perturb_theta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter("ignore")

vnp.random.seed(11)

def predict(theta, X):
    return np.dot(X, theta)


# Build a toy dataset.
N, d = 66, 3
sigma = 0.30
X_train = 2.0 * np.arange(N, dtype=np.float32) / N - 1.0
X_train = np.power(X_train[:, None], np.arange(d))
W_true = vnp.random.randn(d) / np.power(np.arange(d) + 1, 2.0)
Y_train_noiseless = np.dot(X_train, W_true)
noise = vnp.random.randn(N)
Y_train = Y_train_noiseless + sigma * np.power(np.abs(noise), 2.7) * np.sign(noise)

N_test = 400
X_test = 2.0 * np.arange(N_test) / N_test - 1.0
X_test = np.power(X_test[:, None], np.arange(d))
Y_test_noiseless = np.dot(X_test, W_true)
noise = vnp.random.randn(N_test)
Y_test = Y_test_noiseless + sigma * np.power(np.abs(noise), 2.7) * np.sign(noise)


# vector of losses
def loss(theta):
    preds = predict(theta, X_train)
    squared_error = np.power(Y_train - preds, 2.0)
    return squared_error

def summed_loss(theta):
    return np.sum(loss(theta)) / N

theta = vnp.random.randn(d)
N_steps = 1000
lr = 0.5
print("Doing MAP....")

for step in range(N_steps):
    loss_value, theta_grad = value_and_grad(summed_loss)(theta)
    theta -= lr * theta_grad
    if step % 500 == 0 and step > 1000:
        lr /= 3.0
    if step % 100 == 0 or step==N_steps - 1:
        print("[%03d]  loss: %.11f" % (step, loss_value))

print("Final theta: ", theta)
print("Gradient norm at nominal minimum: ", vnp.linalg.norm(grad(summed_loss)(theta)))


def RMSE(theta, X, Y):
    return np.sqrt(np.sum(np.power(Y - predict(theta, X), 2.0)) / X.shape[0])

def LPPD(thetas, X, Y):
    preds = vmap(lambda theta: predict(theta, X))(thetas)
    error = np.power(Y - preds, 2.0)
    log_prob = -0.5 * error / (sigma * sigma)
    log_prob_max = np.max(log_prob, axis=0)
    return np.mean(np.log(np.mean(np.exp(log_prob - log_prob_max), axis=0))) + np.mean(log_prob_max)


print("MAP Train RMSE:  %.6f  %.6f" % (RMSE(theta, X_train, Y_train_noiseless), RMSE(theta, X_train, Y_train)))
print("MAP Test  RMSE:  %.6f  %.6f" % (RMSE(theta, X_test, Y_test_noiseless), RMSE(theta, X_test, Y_test)))
print("MAP Train LPPD:  %.6f" % LPPD(theta[None, :], X_train, Y_train))
print("MAP Test  LPPD:  %.6f" % LPPD(theta[None, :], X_test, Y_test))

print("Doing NPL...")

#orders = [1, 2, 3, 4]
orders = [1, 2]
grad_norms = {}
thetas = {}
predictions = {}
timings = {}

for order in orders:
    grad_norms[order], thetas[order], predictions[order], timings[order] = [], [], [], 0.0

N_thetas = 120
N_thetas_chunk = 30

for _ in range(N_thetas // N_thetas_chunk):
    omegas = vnp.random.dirichlet(np.ones(N), size=N_thetas_chunk)
    domegas = omegas - 1.0 / N

    for order in orders:
        t0 = time.time()
        thetas[order].append(vmap(lambda domega: perturb_theta(loss, theta, np.ones(N) / N, domega, order=order))(domegas))
        timings[order] += time.time() - t0
        grad_norms[order].append(vmap(lambda omega, theta: \
                                 np.linalg.norm(grad(lambda th: np.sum(omega * loss(th)))(theta)))(omegas, thetas[order][-1]))
        predictions[order].append(vmap(lambda th: np.dot(X_test, th))(thetas[order][-1]))

for order in orders:
    print("\n[order %d] grad norms:" % order,
          vnp.mean(vnp.concatenate(grad_norms[order])), " +- ", vnp.std(vnp.concatenate(grad_norms[order])),
          " timing: %.4f sec" % timings[order])

    thetas[order] = vnp.concatenate(thetas[order])
    theta_mean = np.mean(thetas[order], axis=0)
    print("[order %d] NPL Train RMSE:  %.6f  %.6f" % (order, RMSE(theta_mean, X_train, Y_train_noiseless),
                                                      RMSE(theta_mean, X_train, Y_train)))
    print("[order %d] NPL Test  RMSE:  %.6f  %.6f" % (order, RMSE(theta_mean, X_test, Y_test_noiseless),
                                                      RMSE(theta_mean, X_test, Y_test)))
    print("[order %d] NPL Train LPPD:  %.6f" % (order, LPPD(thetas[order], X_train, Y_train)))
    print("[order %d] NPL Test  LPPD:  %.6f" % (order, LPPD(thetas[order], X_test, Y_test)))

    preds = vnp.concatenate(predictions[order])
    percentiles = vnp.percentile(preds, [10.0, 90.0], axis=0)

    plt.figure(figsize=(14,8))
    plt.plot(X_train[:, 1], Y_train, 'kx')
    plt.plot(X_test[:, 1], np.dot(X_test, theta) , 'red', linewidth=3.0)
    plt.plot(X_test[:, 1], np.dot(X_test, theta_mean) , 'purple', linewidth=2.0, linestyle='dotted')
    for k in range(0, N_thetas, 20):
        plt.plot(X_test[:, 1], np.dot(X_test, thetas[order][k, :]), color='k', linestyle='dashed', linewidth=0.5)
    plt.fill_between(X_test[:, 1], percentiles[0, :], percentiles[1, :], color='lightblue')
    plt.savefig('out_order%d.pdf' % order)
    plt.close()
