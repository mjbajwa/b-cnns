import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import tqdm
from sklearn.preprocessing import LabelBinarizer

# Load CIFAR-10 datasets

np.random.seed(42)

def load_cifar10_dataset(train_index=50000, flatten=False):

    # Training dataset

    train_ds = tfds.as_numpy(tfds.load(
        'cifar10', split=tfds.Split.TRAIN, batch_size=-1))
    train_ds = {'image': train_ds['image'].astype(jnp.float32), # / 255.
                'label': train_ds['label'].astype(jnp.int32)}
    # mean_train = train_ds['image'].mean(axis=(0, 1, 2)) # normalizing over each channel
    # std_train = train_ds['image'].std(axis=(0, 1, 2)) # normalizing over each channel
    # mean_train = jnp.array([125.30691805, 122.95039414, 113.86538318])
    # std_train = jnp.array([62.99321928, 62.08870764, 66.70489964])
    mean_train = jnp.array([127.5, 127.5, 127.5])
    std_train = jnp.array([128, 128, 128])
    mean_train = mean_train[None, None, None, :] # broadcasting trick
    std_train = std_train[None, None, None, :]
    train_ds['image'] = (train_ds['image'] - mean_train) / std_train
    y_train_all = LabelBinarizer().fit_transform(train_ds['label'])

    # Test dataset

    test_ds = tfds.as_numpy(
        tfds.load('cifar10', split=tfds.Split.TEST, batch_size=-1))
    test_ds = {'image': test_ds['image'].astype(jnp.float32), # / 255.
               'label': test_ds['label'].astype(jnp.int32)}
    test_ds['image'] = (test_ds['image'] - mean_train) / std_train
    y_test = LabelBinarizer().fit_transform(test_ds['label'])

    # Randomly select train_index images if full batch not being used

    temp_ds = {}

    if train_index != 50000:
        
        ind = np.sort(np.random.choice(50000, train_index, replace=False))
        print("Training indices chosen: ", ind)
        temp_ds['image'] = train_ds['image'][ind]
        temp_ds['label'] = train_ds['label'][ind]
        y_train = y_train_all[ind]

    else:

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

        train_x = temp_ds['image']
        test_x = test_ds['image']

    return train_x, test_x, y_train, y_test, temp_ds, test_ds
