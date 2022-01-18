import argparse
import os

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import jax.tools.colab_tpu
import numpy as np
import numpyro
import numpyro.distributions as dist
import tensorflow.compat.v2 as tf
from numpyro.contrib.module import random_flax_module
from numpyro.infer import MCMC, NUTS, Predictive, init_to_feasible, init_to_value

from utils.load_data import load_cifar10_dataset

# jax.tools.colab_tpu.setup_tpu()

def run_conv_bnn(train_index=50000, num_warmup=100, num_samples=100, gpu=False):

    # Administrative stuff

    print(jax.default_backend())
    print(jax.device_count())

    # Disable tensorflow from using GPU

    tf.enable_v2_behavior()

    if gpu:

        physical_devices = tf.config.list_physical_devices('GPU')

        try:
            # Disable first GPU
            tf.config.set_visible_devices(physical_devices[1:], 'TPU')
            logical_devices = tf.config.list_logical_devices('TPU')
            # Logical device was not created for first GPU
            assert len(logical_devices) == len(physical_devices) - 1
        except:
            pass

        # Enable JAX/NumPyro to use GPU

        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"
        numpyro.set_platform("gpu")

    # Set numpy seeds

    np.random.seed(42)

    # Declare constants for easy checks

    TRAIN_IDX = train_index
    NUM_WARMUP = num_warmup
    NUM_SAMPLES = num_samples

    print("Training samples: ", train_index)
    print("Warmup samples: ", num_warmup)
    print("Number of samples: ", num_warmup)

    # Create keys for numpyro

    rng_key, rng_key_predict = random.split(random.PRNGKey(0))

    # Load CIFAR-10 datasets

    train_x, test_x, y_train, y_test, temp_ds, test_ds = load_cifar10_dataset(
        train_index=TRAIN_IDX, flatten=False)
    # print(y_train)
    y_train = jnp.argmax(y_train, axis=1)
    y_test = jnp.argmax(y_test, axis=1)

    # Define model

    class CNN(nn.Module):

        @nn.compact
        def __call__(self, x):

            x = nn.Conv(features=16, kernel_size=(3, 3))(x)
            x = nn.softplus(x)
            # x = nn.Conv(features=16, kernel_size=(3, 3))(x)
            # x = nn.softplus(x)
            x = nn.max_pool(x, window_shape=(2, 2))
            x = nn.Conv(features=16, kernel_size=(3, 3))(x)
            x = nn.softplus(x)
            # x = nn.Conv(features=32, kernel_size=(3, 3))(x)
            # x = nn.softplus(x)
            x = nn.max_pool(x, window_shape=(2, 2))
            x = x.reshape((x.shape[0], -1))  # flatten
            x = nn.Dense(features=32)(x)
            x = nn.softplus(x)
            x = nn.Dense(features=10)(x)
            # x = nn.softmax(x)

            return x

    def model(x, y):

        module = CNN()

        net = random_flax_module(
            "CNN",
            module,
            prior = dist.Normal(0, 100),
            # prior={
            #     "Conv_0.bias": dist.Normal(0, 100),
            #     "Conv_0.kernel": dist.Normal(0, 100),
            #     "Conv_1.bias": dist.Normal(0, 100),
            #     "Conv_1.kernel": dist.Normal(0, 100),
            #     "Dense_0.bias": dist.Normal(0, 100),
            #     "Dense_0.kernel": dist.Normal(0, 100),
            #     "Dense_1.bias": dist.Normal(0, 100),
            #     "Dense_1.kernel": dist.Normal(0, 100),
            # },
            input_shape=(1, 32, 32, 3)
        )

        # numpyro.sample("y_pred", dist.Multinomial(total_count=1, probs=net(x)), obs=y)
        # y1 = jnp.argmax(y, axis=0)
        numpyro.sample("y_pred", dist.Categorical(logits=net(x)), obs=y)

    # Initialize parameters

    model2 = CNN()
    batch = train_x[0:1, ]  # (N, H, W, C) format
    print("Batch shape: ", batch.shape)
    variables = model2.init(jax.random.PRNGKey(42), batch)
    output = model2.apply(variables, batch)
    print("Output shape: ", output.shape)
    init = flax.core.unfreeze(variables)["params"]

    # Create more reasonable initial values by sampling from the prior

    prior_dist = dist.Normal(0, 10)
    init_new = init.copy()
    total_params = 0

    for i, high in enumerate(init_new.keys()):
        for low in init_new[high].keys():
            print(init_new[high][low].shape)
            init_new[high][low] = prior_dist.sample(
                jax.random.PRNGKey(i), init_new[high][low].shape)

            # increment count of total_params
            layer_params = np.prod(
                np.array([j for j in init_new[high][low].shape]))
            total_params += layer_params

    print("Total parameters: ", total_params)

    # Initialize MCMC

    # kernel = NUTS(model, init_strategy=init_to_value(values=init_new), target_accept_prob=0.70)
    kernel = NUTS(model, 
                  init_strategy=init_to_feasible(), 
                  target_accept_prob=0.80,
                  max_tree_depth=10)
    mcmc = MCMC(
        kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=1,
        progress_bar=True,
    )

    # Run MCMC

    mcmc.run(rng_key, train_x, y_train)

    # for i in range(NUM_SAMPLES):
    #    mcmc.run(random.PRNGKey(i), temp_ds['image'], y_train)
    #    batches = [mcmc.get_samples()]
    #    mcmc._warmup_state = mcmc._last_state

    # mcmc.print_summary()

    # Prediction Utilities

    # TODO:

    # 1) Accuracy metrics on test and train
    # 2) Trace plots for parameters, or summary of R_hats across multiple chains
    # 3) Parameter posterior statistics (R_hat, n_eff)

    # TODO: convert the train_preds to probabilities over class, averaged by uncertainties?

    # Train accuracy calculation

    train_preds = Predictive(model, mcmc.get_samples())(
        jax.random.PRNGKey(2), train_x, y=None)["y_pred"]
    train_preds_ave = jnp.mean(train_preds, axis=0)
    # train_preds_index = jnp.argmax(train_preds_ave, axis=1)
    # accuracy = (temp_ds["label"] == train_preds_index).mean()*100
    accuracy = (y_train == train_preds_ave).mean()*100
    print("Train accuracy: ", accuracy)

    # Test accuracy calculation

    test_preds = Predictive(model, mcmc.get_samples())(
        jax.random.PRNGKey(2), test_x, y=None)["y_pred"]
    test_preds_ave = jnp.mean(test_preds, axis=0)
    accuracy = (y_test == test_preds_ave).mean()*100
    print("Test accuracy: ", accuracy)

    # all_samples = mcmc.get_samples()
    # plt.plot(all_samples["CNN/Conv_0.kernel"][:, 3,3,3,16], "o")
    # plt.plot(all_samples["CNN/Dense_0.kernel"][:, 10], "o")

    # mcmc.print_summary()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convolutional Bayesian Neural Networks for CIFAR-10")
    parser.add_argument("--train_index", type=int, default=50000)
    parser.add_argument("--num_warmup", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--gpu", type=bool, default=False)
    args = parser.parse_args()

    # Run main function

    run_conv_bnn(args.train_index, args.num_warmup, args.num_samples, args.gpu)
