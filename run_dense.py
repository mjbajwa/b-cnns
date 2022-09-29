# -*- coding: utf-8 -*-

import os
 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import logging
from pathlib import Path

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import jax.tools.colab_tpu
import numpy as np
import numpyro
import numpyro.distributions as dist
# import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tqdm
from numpyro.contrib.module import random_flax_module
from numpyro.infer import MCMC, NUTS, Predictive, init_to_feasible
from sklearn.preprocessing import LabelBinarizer

from utils.load_data import load_cifar10_dataset
from utils.misc import make_output_folder, mcmc_summary_to_dataframe, plot_extra_fields, plot_traces, rhat_histogram, print_extra_fields

def run_dense_bnn(train_index=50000, num_warmup=100, num_samples=100, gpu=False):

    # Administrative stuff

    print(jax.default_backend())
    print(jax.device_count())
    print(jax.lib.xla_bridge.get_backend().platform)
    
    # Disable tensorflow from using GPU

    tf.enable_v2_behavior()
    
    if gpu:

        # physical_devices = tf.config.list_physical_devices('GPU')
        # tf.config.experimental.set_visible_devices([], 'GPU')
    
        try:
            # Disable first GPU
            tf.config.set_visible_devices(physical_devices[1:], 'GPU')
            logical_devices = tf.config.list_logical_devices('GPU')
            # Logical device was not created for first GPU
            assert len(logical_devices) == len(physical_devices) - 1
        except:
            pass

        # Enable JAX/NumPyro to use GPU

        # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" 
       #  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".8"
        numpyro.set_platform("gpu")
        numpyro.set_host_device_count(11)
    
    np.random.seed(0)

    # Declare constants for easy checks

    TRAIN_IDX = train_index
    NUM_WARMUP = num_warmup
    NUM_SAMPLES = num_samples

    # Create keys for numpyro

    rng_key, rng_key_predict = random.split(random.PRNGKey(0))

    # Load CIFAR-10 datasets

    train_x, test_x, y_train, y_test, temp_ds, test_ds = load_cifar10_dataset(train_index=TRAIN_IDX, flatten=True)
    print(train_x.shape)
    print(test_x.shape)
    
    # Define model

    class DNN(nn.Module):
            
        @nn.compact
        def __call__(self, x):
            
            # x = nn.Dense(features=256)(x)
            # x = nn.softplus(x) # TODO: check tanh vs softplus
            x = nn.Dense(features=128)(x)
            x = nn.softplus(x) # TODO: check tanh vs softplus
            x = nn.Dense(features=64)(x)
            x = nn.softplus(x) # TODO: check tanh vs softplus

            x = nn.Dense(features=10)(x)
            x = nn.softmax(x)
            
            return x
        

    def model(x, y):
        
        module = DNN()
        
        net = random_flax_module(
            "DNN", 
            module, 
            # prior = {
            # "Dense_0.bias": dist.Normal(0, 10), # 50, 10, 100
            # "Dense_0.kernel": dist.Normal(0, 10), 
            # "Dense_1.bias": dist.Normal(0, 10), 
            # "Dense_1.kernel": dist.Normal(0, 10),
            # "Dense_2.bias": dist.Normal(0, 10), 
            # "Dense_2.kernel": dist.Normal(0, 10),
            # # "Dense_3.bias": dist.Normal(0, 10), 
            # # "Dense_3.kernel": dist.Normal(0, 10),
            # # "Dense_3.bias": dist.Normal(0, 10), 
            # # "Dense_3.kernel": dist.Normal(0, 10),
            # },
            # prior = {
            #     "Dense_0.bias": dist.Cauchy(), # 50, 10, 100
            #     "Dense_0.kernel": dist.Cauchy(), 
            #     "Dense_1.bias": dist.Cauchy(), 
            #     "Dense_1.kernel": dist.Cauchy(),
            #     "Dense_2.bias": dist.Cauchy(), 
            #     "Dense_2.kernel": dist.Cauchy(),
            # },
            # prior = dist.Cauchy(),
            prior = dist.StudentT(df=4.0, scale=0.1),
            input_shape=(3072, )
        
        )
                
        numpyro.sample("y_pred", dist.Multinomial(total_count=1, probs=net(x)), obs=y)
        # numpyro.sample("y_pred", dist.MultinomialLogits(logits=net(x)), obs=y)

    # Initialize parameters 

    model2 = DNN()
    batch = train_x[0]  # (N, H, W, C) format
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

    # Initialize MCMC

    # kernel = NUTS(model, init_strategy=init_to_value(values=init_new))

    kernel = NUTS(model, init_strategy=init_to_feasible(), 
                  target_accept_prob=0.80, max_tree_depth=10)
    mcmc = MCMC(  
        kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=1,
        progress_bar=True,
        # jit_model_args=True,
    )

    # Run MCMC

    mcmc.run(rng_key, train_x, y_train, 
             extra_fields = ("z", "i", 
                             "num_steps", 
                             "accept_prob", 
                             "adapt_state.step_size"))

    # for i in range(NUM_SAMPLES):
    #    mcmc.run(random.PRNGKey(i), temp_ds['image'], y_train)
    #    batches = [mcmc.get_samples()]
    #    mcmc._warmup_state = mcmc._last_state

    ### Prediction Utilities

    # TODO: convert the train_preds to probabilities over class, averaged by uncertainties?

    # Train accuracy calculation

    train_preds = Predictive(model, mcmc.get_samples())(rng_key_predict, train_x, y=None)["y_pred"]
    # print("Train Preds: \n\n")
    # print(train_preds)
    train_preds_ave = jnp.mean(train_preds, axis=0)
    # print("Train preds average: \n\n")
    # print(train_preds_ave)
    train_preds_index = jnp.argmax(train_preds_ave, axis=1)
    # print("Train preds index: \n\n")
    # print(train_preds_index)
    # print("Actual: \n\n")
    # print(temp_ds["label"])
    train_accuracy = (temp_ds["label"] == train_preds_index).mean()*100
    print("Train accuracy: ", train_accuracy)

    # Test accuracy calculation

    test_preds = Predictive(model, mcmc.get_samples())(rng_key_predict, test_x, y=None)["y_pred"]
    test_preds_ave = jnp.mean(test_preds, axis=0)
    test_preds_index = jnp.argmax(test_preds_ave, axis=1)
    test_accuracy = (test_ds["label"] == test_preds_index).mean()*100
    print("Test accuracy: ", test_accuracy)

    return mcmc, train_accuracy, test_accuracy

if __name__ == "__main__":
    
    # Parse arguments

    parser = argparse.ArgumentParser(description="FCN Deep Bayesian Neural Networks for CIFAR-10")
    parser.add_argument("--train_index", type=int, default=50000)

    parser.add_argument("--num_warmup", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--gpu", type=bool, default=True)
    args = parser.parse_args()

    # Create folder to save results

    output_path = make_output_folder()
    logging.basicConfig(filename=Path(output_path, 'results.log'), level=logging.INFO)
    logging.info('Deep Bayesian Net - Fully Connected')
    
    # Run main function
    
    mcmc, train_acc, test_acc = run_dense_bnn(args.train_index, args.num_warmup, args.num_samples, args.gpu)
    logging.info("Train accuracy: {}".format(train_acc))
    logging.info("Test accuracy: {}".format(test_acc))

    # Save trace plots 

    logging.info("=========================")
    logging.info("Plotting extra fields \n\n")
    plot_extra_fields(mcmc, output_path)
    # print_extra_fields(mcmc, output_path)

    # TODO: Trace plots
    
    # R-hat plot

    logging.info("=========================")
    logging.info("Histogram of R_hat and n_eff \n\n")
    df = mcmc_summary_to_dataframe(mcmc)
    rhat_histogram(df, output_path)
    df.to_parquet(Path(output_path, 'params_summary.parquet'))

    # Write train and test accuracy to file

    logging.info("=========================")
    logging.info("Writing results to file \n\n")
    results = ['Training Accuracy: {}'.format(train_acc), 
               'Test Accuracy: {}'.format(test_acc)]
    
    with open(Path(output_path, 'results.txt'), 'w') as f:
        f.write('-------- Results ----------\n\n')
        f.write('\n'.join(results))

    # TODO: write inputs into a file as well to track all experiments
