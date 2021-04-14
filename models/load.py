import tensorflow as tf

def parse_image(example_proto):
    image_feature_description = {
        'in_or_out': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'kernel': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    image_features = tf.io.parse_single_example(example_proto, image_feature_description)
    
    in_or_out = tf.cast(image_features['in_or_out'], tf.int64)
    depth = tf.cast(image_features['depth'], tf.int64)
    height = tf.cast(image_features['height'], tf.int64)
    width = tf.cast(image_features['width'], tf.int64)
    kernel = tf.cast(image_features['kernel'], tf.int64)
    image_raw = image_features['image_raw']
    image_raw = tf.io.decode_raw(image_raw,tf.float32)
    image_raw = tf.reshape(image_raw,[in_or_out,depth,height,width,kernel])
    
    return image_raw[0], image_raw[1]

def load_png(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)

    w = tf.shape(image)[1]

    w = w // 2
    target = image[:, w:, :]
    stf = image[:, :w, :]

    target = tf.cast(target, tf.float32)
    stf = tf.cast(stf, tf.float32)

    return stf, target
    

def load_tfrecord(train_dir, test_dir, BATCH_SIZE, BUFFER_SIZE):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = tf.data.Dataset.list_files(f'{train_dir}/*.tfrecords')
    train_dataset = tf.data.TFRecordDataset(train_dataset)
    train_dataset = train_dataset.map(parse_image, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(f'{test_dir}/*.tfrecords', shuffle=False)
    names = test_dataset
    test_dataset = tf.data.TFRecordDataset(test_dataset)
    test_dataset = test_dataset.map(parse_image, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    
    return train_dataset, test_dataset, names

def load_png(train_dir_2d, test_dir_2d, BATCH_SIZE, BUFFER_SIZE):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = tf.data.Dataset.list_files(f'{train_dir_2d}/*.png')
    train_dataset = train_dataset.map(load_png, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(f'{test_dir_2d}/*.png')
    names = test_dataset
    test_dataset = train_dataset.map(load_png, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = train_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset, names