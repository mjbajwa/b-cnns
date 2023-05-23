import argparse
import copy
import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import colorlog
import einops
import flax
# import haiku as hk
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import jax.tools.colab_tpu
import ml_collections
import numpy as np
import numpyro
import numpyro.distributions as dist
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from clu import parameter_overview
from flax import linen as nn
from flax.core import freeze, unfreeze
from jax import numpy as jnp
from jax import random
from numpyro.contrib.module import random_flax_module
from numpyro.infer import (MCMC, NUTS, Predictive, init_to_feasible,
                           init_to_median, init_to_uniform, init_to_value)

from utils.load_data import load_cifar10_dataset
from utils.misc import (make_output_folder, mcmc_summary_to_dataframe,
                        plot_extra_fields, plot_traces, print_extra_fields,
                        rhat_histogram)


class MlpBlock(nn.Module):
  mlp_dim: int

  @nn.compact
  def __call__(self, x):
    y = nn.Dense(self.mlp_dim)(x)
    # y = nn.gelu(y)
    y = nn.softplus(y)
    return nn.Dense(x.shape[-1])(y)


class MixerBlock(nn.Module):
  """Mixer block layer."""
  tokens_mlp_dim: int
  channels_mlp_dim: int

  @nn.compact
  def __call__(self, x):
    # y = nn.LayerNorm()(x)
    y = x
    y = jnp.swapaxes(y, 1, 2)
    y = MlpBlock(self.tokens_mlp_dim, name='token_mixing')(y)
    y = jnp.swapaxes(y, 1, 2)
    x = x + y
    # y = nn.LayerNorm()(x)
    return x + MlpBlock(self.channels_mlp_dim, name='channel_mixing')(y)


class MlpMixer(nn.Module):
  """Mixer architecture."""
  patches: Any
  num_classes: int
  num_blocks: int
  hidden_dim: int
  tokens_mlp_dim: int
  channels_mlp_dim: int
  model_name: Optional[str] = None

  @nn.compact
  def __call__(self, inputs):
    x = nn.Conv(self.hidden_dim, self.patches.size,
                strides=self.patches.size, name='stem')(inputs)
    x = einops.rearrange(x, 'n h w c -> n (h w) c')
    for _ in range(self.num_blocks):
      x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
    # x = nn.LayerNorm(name='pre_head_layer_norm')(x)
    x = jnp.mean(x, axis=1)
    if self.num_classes:
      x = nn.Dense(self.num_classes, kernel_init=nn.initializers.zeros,
                   name='head')(x)
      x = nn.softmax(x)
    return x
  
def mixer_model():
   return MlpMixer(patches=ml_collections.ConfigDict({'size': (5, 5)}), 
                   num_classes=10, 
                   num_blocks=3, 
                   hidden_dim=64, 
                   tokens_mlp_dim=64, 
                   channels_mlp_dim=128)

def run_conv_bnn(train_index=50000, num_warmup=100, num_samples=100, gpu=False):

    # Administrative stuff

    print(jax.default_backend())
    print(jax.device_count())

    # Disable tensorflow from using GPU

    # tf.enable_v2_behavior()

    if gpu:

        # physical_devices = tf.config.list_physical_devices('GPU')
        # tf.config.experimental.set_visible_devices([], 'GPU')

        # try:
        #     # Disable first GPU
        #     # tf.config.set_visible_devices(physical_devices[1:], 'TPU')
        #     # logical_devices = tf.config.list_logical_devices('TPU')
        #     # tf.config.experimental.set_visible_devices([], 'GPU')
        #     # Logical device was not created for first GPU
        #     # assert len(logical_devices) == len(physical_devices) - 1
        # except:
        #     pass

        # Enable JAX/NumPyro to use GPU

        # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.87"
        numpyro.set_platform("gpu")
    
    else:
        
        numpyro.set_platform("cpu")
        numpyro.set_host_device_count(15)

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

    # Define model

    def model(x, y):

        module = mixer_model()

        net = random_flax_module(
            "Resnet",
            module,
            prior = dist.StudentT(df=4.0, scale=0.1),
            input_shape=(1, 32, 32, 3)
        )

        numpyro.sample("y_pred", dist.Multinomial(total_count=1, probs=net(x)), obs=y)
        # y1 = jnp.argmax(y, axis=0)
        # numpyro.sample("y_pred", dist.Categorical(logits=net(x)), obs=y)

        # Initialize parameters

    # model2 = ResNet20()
    # batch = train_x[0:1, ]  # (N, H, W, C) format
    # print("Batch shape: ", batch.shape)
    # variables = model2.init(jax.random.PRNGKey(42), batch)
    # output = model2.apply(variables, batch)
    # print("Output shape: ", output.shape)
    # init = flax.core.unfreeze(variables)["params"]
    
    model2 = mixer_model()
    key = jax.random.PRNGKey(0)
    variables = model2.init(key, np.random.randn(1, 32, 32, 3))
    print(parameter_overview.get_parameter_overview(variables))
    del model2, variables

    # Create more reasonable initial values by sampling from the prior

    # prior_dist = dist.Normal(0, 10)
    # init_new = init.copy()
    # total_params = 0

    # for i, high in enumerate(init_new.keys()):
    #     for low in init_new[high].keys():
    #         print(init_new[high][low].shape)
    #         init_new[high][low] = prior_dist.sample(
    #             jax.random.PRNGKey(i), init_new[high][low].shape)

    #         # increment count of total_params
    #         layer_params = np.prod(
    #             np.array([j for j in init_new[high][low].shape]))
    #         total_params += layer_params

    # print("Total parameters: ", total_params)

    # Initialize MCMC

    # kernel = NUTS(model, init_strategy=init_to_value(values=init_new), target_accept_prob=0.70)
    kernel = NUTS(model, 
                  # init_strategy = init_to_median(), # init_to_value(values=variables), # init_to_uniform(), 
                  init_strategy = init_to_feasible(), # init_to_value(values=variables),
                  target_accept_prob=0.70,
                  max_tree_depth=10,
                  )
    
    mcmc = MCMC(
        kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=1,
        progress_bar=True, # TOGGLE this...
        chain_method="vectorized", # "vectorized"
        # jit_model_args=True,
    )

    # Run MCMC

    mcmc.run(rng_key, train_x, y_train)
    # extra_fields = ("z", "i", 
    #                "num_steps", 
    #                "accept_prob", 
    #                "adapt_state.step_size"))

    # batches = []

    # for i in range(NUM_SAMPLES):
    #     print(i)
    #     mcmc.run(random.PRNGKey(i), train_x, y_train)
    #     batches.append(mcmc.get_samples())
    #     mcmc.post_warmup_state = mcmc.last_state

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
    train_preds_index = jnp.argmax(train_preds_ave, axis=1)
    train_accuracy = (temp_ds["label"] == train_preds_index).mean()*100
    print("Train accuracy: ", train_accuracy)

    # Test accuracy calculation

    test_preds = Predictive(model, mcmc.get_samples())(
        jax.random.PRNGKey(2), test_x, y=None)["y_pred"]
    test_preds_ave = jnp.mean(test_preds, axis=0)
    test_preds_index = jnp.argmax(test_preds_ave, axis=1)
    test_accuracy = (test_ds["label"] == test_preds_index).mean()*100
    print("Test accuracy: ", test_accuracy)

    return mcmc, train_accuracy, test_accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ResNet for CIFAR-10")
    parser.add_argument("--train_index", type=int, default=25000)
    parser.add_argument("--num_warmup", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--gpu", type=bool, default=True)
    args = parser.parse_args()

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"
    # os.environ["XLA_GPU_STRICT_CONV_ALGORITHM_PICKER"] = "true"
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    tf.config.experimental.set_visible_devices([], "GPU")

    # Create folder to save results

    output_path = make_output_folder()
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s:%(name)s:%(message)s'))
    logger = colorlog.getLogger(str(Path(output_path, 'results.log')))
    logger.addHandler(handler)
 
    logger.info('Deep Bayesian Net - MLP mixer')
    
    # Run main function
    
    mcmc, train_acc, test_acc = run_conv_bnn(args.train_index, args.num_warmup, args.num_samples, False) # args.gpu)
    logger.info("Train accuracy: {}".format(train_acc))
    logger.info("Test accuracy: {}".format(test_acc))

    # Save trace plots 

    # logging.info("=========================")
    logger.info("Plotting extra fields \n\n")
    # plot_extra_fields(mcmc, output_path)
    # print_extra_fields(mcmc, output_path)

    # TODO: Trace plots
    
    # R-hat plot

    # logging.info("=========================")
    logger.info("Histogram of R_hat and n_eff \n\n")
    df = mcmc_summary_to_dataframe(mcmc)
    rhat_histogram(df, output_path)

    # Write train and test accuracy to file

    # logging.info("=========================")
    logger.info("Writing results to file \n\n")
    results = ['Training Accuracy: {}'.format(train_acc), 
               'Test Accuracy: {}'.format(test_acc)]
    
    with open(Path(output_path, 'results.txt'), 'w') as f:
        f.write('-------- Results ----------\n\n')
        f.write('\n'.join(results))

    # TODO: write inputs into a file as well to track all experiments
