import tensorflow_datasets as tfds
from data_utils import random_jitter, normalize, get_LR_HR_pair


def get_oxford_iiit_pet_dataset(train_type='train', size=(224, 224, 3), downsampling_factor=4, batch_size=32):
    count = {
        'test': 3669,
        'train': 3680,
    }

    data = tfds.load('oxford_iiit_pet')
    dataset = data[train_type]

    if train_type == 'train':
        dataset = dataset.map(lambda x: random_jitter(x['image'], size), 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(normalize, 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif train_type == 'test':
        dataset = dataset.map(lambda x: tf.image.resize(x['image'], [size[0], size[1]]), 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(normalize,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.repeat().batch(batch_size).prefetch(16)
    dataset = dataset.map(lambda x: get_LR_HR_pair(x, downsampling_factor), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))
    return dataset, count[train_type]
