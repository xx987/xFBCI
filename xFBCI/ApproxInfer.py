import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from jax.scipy.stats import multivariate_normal
from numpy import linalg as LA


import jax.random as random
import jax.numpy as jnp
import numpy as np

#Likelihood function for all dataset (logistics)
def likelihood_p_k(theta,X,W):
    #Normalization as the distribution of dataset input
    # X = scaler.fit_transform(X) #normalize the data if the code has Nan or gradient vanishing/explosion
    linear_pred = X @ theta
    sigmoid = 1 / (1 + jnp.exp(-linear_pred))

    log_likelihood = np.sum(
        W * jnp.log(sigmoid) +
        (1 - W) * jnp.log(1 - sigmoid)
    )

    return log_likelihood


class SG_MCMC_Infer:
    def __init__(self, p_k,
                 eta,
                 Lambda,
                 learning_rate,
                 num_samples,
                 theta,
                 X,
                 W,
                 batch_size):
        """
        p_k: Likelihood function
        eta: q_{-k}'s mean (Gaussian)
        Lambda: q_{-k}'s covariance (Gaussian)
        num_samples: number of samples be collected
        """
        self.p_k = p_k
        self.eta = eta
        self.Lambda = Lambda
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.theta = theta
        self.samples = []
        self.current_s = []
        self.X = X
        self.W = W
        self.batch_size = batch_size


    #SGLD sample
    def sgld_sample(self):
        samples = []
        current_position = jnp.array(self.theta)
        v = 0.5
        prng_key = random.PRNGKey(42)
        prng_key, prng_key_sg, prng_key_noise = random.split(prng_key, 3)
        momentum = 0.2
        for _ in range(self.num_samples):

            indices = np.random.choice(self.X.shape[0], self.batch_size, replace=False)
            grad = ((len(self.X)/self.batch_size)*jax.grad(lambda x:-self.p_k(x,self.X[indices],self.W[indices]))(current_position)
                    -jnp.dot(self.Lambda,jnp.array(current_position)))
            noise = 0.5 * random.normal(prng_key_noise, current_position.shape)
            grad = grad+noise*jnp.sqrt(0.5 * self.learning_rate)
            current_position-=self.learning_rate*grad#v
            final_sampl = jnp.array(current_position)+ jnp.array(self.eta)
            samples.append(final_sampl)

        return jnp.array(samples)

