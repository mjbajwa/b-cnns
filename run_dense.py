# -*- coding: utf-8 -*-

# !pip install --upgrade pip
# !pip install numpyro jax jaxlib flax scikit-learn
# Run the following in shell before executing: export XLA_PYTHON_CLIENT_MEM_FRACTION=.7

# !export XLA_PYTHON_CLIENT_MEM_FRACTION=.5

import matplotlib.pyplot as plt
import numpyro.distributions as dist
import numpyro
import flax.linen as nn
import tqdm
import numpy as np
import os
import jax.numpy as jnp
import flax
import jax
import jax.random as random
import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
import jax.tools.colab_tpu
import argparse

from sklearn.preprocessing import LabelBinarizer
from numpyro.infer import init_to_value, Predictive
from numpyro.infer import init_to_feasible, NUTS, MCMC, HMC
from numpyro.contrib.module import random_flax_module, flax_module
from numpyro.infer import init_to_feasible

# jax.tools.colab_tpu.setup_tpu()

def run_dense_bnn(gpu=True):

    # Administrative stuff

    print(jax.default_backend())
    # Disable tensorflow from using GPU

    tf.enable_v2_behavior()
    
    if gpu:

        physical_devices = tf.config.list_physical_devices('GPU')
    
        try:
            # Disable first GPU
            tf.config.set_visible_devices(physical_devices[1:], 'GPU')
            logical_devices = tf.config.list_logical_devices('GPU')
            # Logical device was not created for first GPU
            assert len(logical_devices) == len(physical_devices) - 1
        except:
            pass

        # Enable JAX/NumPyro to use GPU

        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" 
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".8"
        numpyro.set_platform("gpu")
        numpyro.set_host_device_count(11)
    
    np.random.seed(0)

    # CONSTANTS 

    TRAIN_IDX = 50000
    NUM_WARMUP = 1000
    NUM_SAMPLES = 1000

    # Create keys for numpyro

    rng_key, rng_key_predict = random.split(random.PRNGKey(0))

    # Load CIFAR-10 datasets

    # Training dataset

    train_ds = tfds.as_numpy(tfds.load(
        'cifar10', split=tfds.Split.TRAIN, batch_size=-1))
    train_ds = {'image': train_ds['image'].astype(jnp.float32) / 255.,
                'label': train_ds['label'].astype(jnp.int32)}
    y_train_all = LabelBinarizer().fit_transform(train_ds['label'])

    # Test dataset

    test_ds = tfds.as_numpy(tfds.load('cifar10', split=tfds.Split.TEST, batch_size=-1))
    test_ds = {'image': test_ds['image'].astype(jnp.float32) / 255.,
            'label': test_ds['label'].astype(jnp.int32)}
    y_test = LabelBinarizer().fit_transform(test_ds['label'])

    # Filter to the first 1000 images for configuration

    temp_ds = {}
    temp_ds['image'] = train_ds['image'][0:TRAIN_IDX]
    temp_ds['label'] = train_ds['label'][0:TRAIN_IDX]
    y_train = y_train_all[0:TRAIN_IDX]

    # Flatten 

    train_x_flat = np.zeros(shape=(temp_ds['image'].shape[0], np.prod(temp_ds['image'].shape[1:])))
    test_x_flat = np.zeros(shape=(test_ds['image'].shape[0], np.prod(test_ds['image'].shape[1:])))

    for i, im in tqdm.tqdm(enumerate(temp_ds['image'])):
        train_x_flat[i, :] = im.flatten()
        
    for i, im in tqdm.tqdm(enumerate(test_ds['image'])):
        test_x_flat[i, :] = im.flatten()

    train_x_flat.shape

    test_x_flat.shape

    # idx = 10
    # plt.figure(figsize=(5, 5))
    # plt.imshow(temp_ds['image'][idx])
    # plt.title(temp_ds['label'][idx])
    # plt.show()

    # Define model

    class CNN(nn.Module):
            
        @nn.compact
        def __call__(self, x):
            
            # x = x.reshape(x.shape[0], -1)  # flatten
            x = nn.Dense(features=64)(x)
            x = nn.softplus(x) # TODO: check tanh vs softplus
            x = nn.Dense(features=128)(x)
            x = nn.softplus(x) # TODO: check tanh vs softplus
            x = nn.Dense(features=10)(x)
            x = nn.softmax(x)
            
            return x
        

    def model(x, y):
        
        module = CNN()
        
        net = random_flax_module(
            "CNN", 
            module, 
            prior = {
            # "Conv_0.bias": dist.Normal(0, 100), 
            # "Conv_0.kernel": dist.Normal(0, 100), 
            # "Conv_1.bias": dist.Normal(0, 100), 
            # "Conv_1.kernel": dist.Normal(0, 100), 
            "Dense_0.bias": dist.Normal(0, 100), 
            "Dense_0.kernel": dist.Normal(0, 100), 
            "Dense_1.bias": dist.Normal(0, 50), 
            "Dense_1.kernel": dist.Normal(0, 50),
            "Dense_2.bias": dist.Normal(0, 10), 
            "Dense_2.kernel": dist.Normal(0, 10)
            },
            
            input_shape=(3072, )
        
        )
                
        numpyro.sample("y_pred", dist.Multinomial(total_count=1, probs=net(x)), obs=y)

    # model2 = CNN()
    # batch = train_x_flat[0]  # (N, H, W, C) format
    # variables = model2.init(jax.random.PRNGKey(42), batch)
    # output = model2.apply(variables, batch)      
    # print(output.shape)

    batch = train_x_flat[0, :]

    print(batch.shape)

    # Initialize parameters 

    model2 = CNN()
    batch = train_x_flat[0]  # (N, H, W, C) format
    variables = model2.init(jax.random.PRNGKey(42), batch)
    output = model2.apply(variables, batch)      
    print(output.shape)
    init = flax.core.unfreeze(variables)["params"]

    # Create more reasonable initial values by sampling from the prior

    prior_dist = dist.Normal(0, 10)
    init_new = init.copy()
    total_params = 0

    for i, high in enumerate(init_new.keys()):
        for low in init_new[high].keys():
            print(init_new[high][low].shape)
            init_new[high][low] = prior_dist.sample(jax.random.PRNGKey(i), init_new[high][low].shape)
            
            # increment count of total_params
            layer_params = np.prod(np.array([j for j in init_new[high][low].shape]))
            total_params += layer_params

    print("Total parameters: ", total_params)

    init_new["Dense_0"]["kernel"].shape

    init_new["Dense_1"]["kernel"].shape

    # Initialize MCMC

    # kernel = NUTS(model, init_strategy=init_to_value(values=init_new))
    kernel = NUTS(model)
    mcmc = MCMC(  
        kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=1,
        progress_bar=True,
        # jit_model_args=True,
    )

    # Run MCMC

    # mcmc.run(rng_key, temp_ds['image'], y_train, init_params=init)
    # mcmc.run(rng_key, temp_ds['image'], y_train, init_params=init_new)
    # mcmc.run(rng_key, temp_ds['image'], y_train, init_params=init_new)
    # mcmc.run(rng_key, temp_ds['image'], y_train)

    mcmc.run(rng_key, train_x_flat, y_train)

    # for i in range(NUM_SAMPLES):
    #    mcmc.run(random.PRNGKey(i), temp_ds['image'], y_train)
    #    batches = [mcmc.get_samples()]
    #    mcmc._warmup_state = mcmc._last_state

    # mcmc.print_summary()

    ### Prediction Utilities

    # TODO:

    # 1) Accuracy metrics on test and train
    # 2) Trace plots for parameters, or summary of R_hats across multiple chains
    # 3) Parameter posterior statistics (R_hat, n_eff)

    # TODO: convert the train_preds to probabilities over class, averaged by uncertainties?

    # Train accuracy calculation

    train_preds = Predictive(model, mcmc.get_samples())(jax.random.PRNGKey(2), train_x_flat, y=None)["y_pred"]
    train_preds_ave = jnp.mean(train_preds, axis=0)
    train_preds_index = jnp.argmax(train_preds_ave, axis=1)
    accuracy = (temp_ds["label"] == train_preds_index).mean()*100
    print(accuracy)

    # Test accuracy calculation

    test_preds = Predictive(model, mcmc.get_samples())(jax.random.PRNGKey(2), test_x_flat, y=None)["y_pred"]
    test_preds_ave = jnp.mean(test_preds, axis=0)
    test_preds_index = jnp.argmax(test_preds_ave, axis=1)
    accuracy = (test_ds["label"] == test_preds_index).mean()*100
    print(accuracy)

    all_samples = mcmc.get_samples()

    # plt.plot(all_samples["CNN/Conv_0.kernel"][:, 3,3,3,16], "o")
    # plt.plot(all_samples["CNN/Dense_0.kernel"][:, 10], "o")

    # mcmc.print_summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convolutional Bayesian Neural Networks for CIFAR-10")
    parser.add_argument("--gpu", type=bool, default=False)
    args = parser.parse_args()
    run_dense_bnn(args.gpu)