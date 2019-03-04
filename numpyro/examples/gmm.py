import argparse

import jax.numpy as np
from jax import random
from jax.experimental import optimizers
from jax.random import PRNGKey

import numpyro.distributions as dist

from numpyro.handlers import sample, param
from numpyro.svi import SVI, elbo

K = 2  # num components
def softplus(data):
    return np.log(1. + np.exp(data))


def model(data):
#     weights = sample('weights', dist.Dirichlet(0.5 * np.ones(K)))
    weights = sample('weights', dist.uniform(np.zeros(K), np.ones(K)))
    weights = np.concatenate([weights, 1-weights], axis=0)
    scale = sample('scale', dist.lognorm(np.array([0.]), np.array([2.])))
    locs = sample('locs', dist.norm(np.zeros(K), 10. * np.ones(K)))

    assignment = sample('assignment', dist.categorical(weights))
    sample('obs', dist.norm(locs[assignment], scale), obs=data)


def guide(data):
    key = random.PRNGKey(0)
    # delta guide
    weight_param = param('weight_param', np.ones(K))
    scale_param = param('scale_param', random.normal(key, shape=(K,)))
    loc_loc = param('loc_loc', random.normal(key, shape=(K,)))
    loc_scale = param('loc_scale', random.normal(key, shape=(K,)))
#     weights = sample('weights', dist.Dirichlet(K))
    weights = sample('weights', dist.uniform(np.zeros(K), np.ones(K)))
    scale = sample('scale', dist.lognorm(softplus(scale_param)))
    locs = sample('locs', dist.norm(loc_loc, softplus(loc_scale)))


def main(args):
    data = np.array([0., 1., 1.5, 10., 11., 12.])
    opt_init, opt_update = optimizers.adam(args.learning_rate)
    svi = SVI(model, guide, opt_init, opt_update, elbo)
    opt_state = None
    for i in range(args.num_steps):
        loss, opt_state = svi.step(i, data, opt_state=opt_state)
        print("step {} loss = {}".format(i, loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian Mixture Model")
    parser.add_argument("-n", "--num-steps", default=100, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.02, type=float)
    args = parser.parse_args()
    main(args)
