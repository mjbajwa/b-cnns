import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import tqdm
from sklearn.preprocessing import LabelBinarizer

# Load CIFAR-10 datasets


def load_cifar10_dataset(train_index=50000, flatten=False):

    # Training dataset

    train_ds = tfds.as_numpy(tfds.load(
        'cifar10', split=tfds.Split.TRAIN, batch_size=-1))
    train_ds = {'image': train_ds['image'].astype(jnp.float32) / 255.,
                'label': train_ds['label'].astype(jnp.int32)}
    y_train_all = LabelBinarizer().fit_transform(train_ds['label'])

    # Test dataset

    test_ds = tfds.as_numpy(
        tfds.load('cifar10', split=tfds.Split.TEST, batch_size=-1))
    test_ds = {'image': test_ds['image'].astype(jnp.float32) / 255.,
               'label': test_ds['label'].astype(jnp.int32)}
    y_test = LabelBinarizer().fit_transform(test_ds['label'])

    # Filter to the first 1000 images for configuration

    temp_ds = {}
    temp_ds['image'] = train_ds['image'][0:train_index]
    temp_ds['label'] = train_ds['label'][0:train_index]
    y_train = y_train_all[0:train_index]

    if flatten:

        # Flatten for dense neural networks only

        train_x = np.zeros(
            shape=(temp_ds['image'].shape[0], np.prod(temp_ds['image'].shape[1:])))
        test_x = np.zeros(
            shape=(test_ds['image'].shape[0], np.prod(test_ds['image'].shape[1:])))

        for i, im in tqdm.tqdm(enumerate(temp_ds['image'])):
            train_x[i, :] = im.flatten()

        for i, im in tqdm.tqdm(enumerate(test_ds['image'])):
            test_x[i, :] = im.flatten()

    else:

        train_x = train_ds['image']
        test_x = test_ds['image']

    return train_x, test_x, y_train, y_test, temp_ds, test_ds
