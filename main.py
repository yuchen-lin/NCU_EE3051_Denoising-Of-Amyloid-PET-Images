import os
import time
import numpy as np
import SimpleITK as sitk
import datetime
import PIL.Image
import argparse
import sys

parser = argparse.ArgumentParser(description='Train or predict.')
parser.add_argument("--mode", help="=train or =predict")
parser.add_argument("--epoch", help="If mode=train => number of training epoch./nIf mode=predict => predict on specified epoch weight.")
args = parser.parse_args()
if not args.mode :
    print('Please specify epoch!')
    print('Check --info for more information.')
    sys.exit(1)
elif not args.epoch :
    print('Please specify mode!')
    print('Check --info for more information.')
    sys.exit(1)

import tensorflow as tf
from tensorflow.keras.utils import plot_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("tensorflow version = "+str(tf.__version__))
    print()

train_dir = './tfrecord_denoise/train'
test_dir = './tfrecord_denoise/test'

BATCH_SIZE = 1
d=128
h=128
w=128
sli=50

OUTPUT_CHANNELS = 1

learning_rate = 2e-4

save_img_folder_name = 'outputs_png'
save_img_folder_name_nii = 'outputs_nii'

train_num = len(os.listdir(train_dir))
test_num = len(os.listdir(test_dir))

BUFFER_SIZE = train_num

def prepareTrainTestFolder():
    try:
        os.mkdir(f'./training_checkpoints')
    except:
        print("Checkpoint folder has already created")
    print("Total train data = "+str(train_num))
    print("Total test data = "+str(test_num))
    print()

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

def Generator():
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inputs = tf.keras.layers.Input(shape=[d,h,w,OUTPUT_CHANNELS])
    skips = []
    
    x = inputs
    
    x = tf.keras.layers.Conv3D(filters=16,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=16,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    skips.append(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='same')(x)
    
    x = tf.keras.layers.Conv3D(filters=32,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=32,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=32,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    skips.append(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='same')(x)
    
    x = tf.keras.layers.Conv3D(filters=64,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=64,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=64,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    skips.append(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='same')(x)
    
    x = tf.keras.layers.Conv3D(filters=128,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=128,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=128,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)
    
    x = tf.keras.layers.Concatenate()([x, skips[2]])
    x = tf.keras.layers.Conv3D(filters=192,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=64,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=64,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)
    
    x = tf.keras.layers.Concatenate()([x, skips[1]])
    x = tf.keras.layers.Conv3D(filters=96,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=32,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=32,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)
    
    x = tf.keras.layers.Concatenate()([x, skips[0]])
    x = tf.keras.layers.Conv3D(filters=48,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=16,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=16,kernel_size=3,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv3D(filters=OUTPUT_CHANNELS,kernel_size=1,strides=1,padding='same',kernel_initializer=initializer)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64, to_file="model.png")

def generator_loss(gen_output, target):
    
    l1_loss = tf.reduce_mean(tf.abs(gen_output - target))

    return l1_loss

generator_optimizer = tf.keras.optimizers.Adam(learning_rate)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)

log_dir="logs/"
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape:
        gen_output = generator(input_image, training=True)
        l1_loss = generator_loss(gen_output, target)

    generator_gradients = gen_tape.gradient(l1_loss,generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('l1_loss', l1_loss, step=epoch)

def fit(train_ds, epochs):
    print('To view training loss, use $tensorboard --logdir=logs')
    print()
    for epoch in range(epochs):
        start = time.time()
            
        # Train
        print(f"porcessing epoch{epoch+1}...")
        print()
        count=0
        for n, (input_image, target) in train_ds.enumerate():
            count+=1
            train_step(input_image, target, epoch)
            if count%10==0 :
                print(f'[{count}/{train_num}]')
        
        checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch+1,time.time()-start))

def generate_images_predict_png(model, test_input, tar, id):
    print("predicting image"+str(id)+"...")
    prediction = model(test_input, training=True)
    display_list = [test_input[0], tar[0], prediction[0]]
    
    for sl in range(sli,sli+21,2):
        im1=tf.squeeze(display_list[0][sl]) * 0.5 + 0.5
        im2=tf.squeeze(display_list[1][sl]) * 0.5 + 0.5
        im3=tf.squeeze(display_list[2][sl]) * 0.5 + 0.5
    
        output = np.hstack((im1,im2,im3))
        output = output/np.max(output)
        output = output*255
        #print(f'shape = {output.shape}')
        #print(f'max = {np.max(output)}')
        #print(f'min = {np.min(output)}')
    
        outname = f'./{save_img_folder_name}/{id}_{sl}th_slice.png'
        PIL.Image.fromarray(output).convert('L').save(outname)
        print(f'{outname} saved\n')

def generate_images_predict_nii(model, test_input, tar, id):
    print("predicting image"+str(id)+"...")
    prediction = model(test_input, training=True)
    display_list = [test_input[0], tar[0], prediction[0]]
    

    im1=tf.squeeze(display_list[0]) * 0.5 + 0.5
    im2=tf.squeeze(display_list[1]) * 0.5 + 0.5
    im3=tf.squeeze(display_list[2]) * 0.5 + 0.5
        
    out1 = sitk.GetImageFromArray(im1)
    out2 = sitk.GetImageFromArray(im2)
    out3 = sitk.GetImageFromArray(im3)
        
    outname1 = f'./{save_img_folder_name_nii}/{id}_input.nii.gz'
    sitk.WriteImage(out1,outname1)
    print(f'{outname1} saved')
    outname2 = f'./{save_img_folder_name_nii}/{id}_target.nii.gz'
    sitk.WriteImage(out2,outname2)
    print(f'{outname2} saved')
    outname3 = f'./{save_img_folder_name_nii}/{id}_predict.nii.gz'
    sitk.WriteImage(out3,outname3)
    print(f'{outname3} saved')
    
    print()

def load_weights(UseEpoch):
    print("Loading epoch "+str(UseEpoch)+"...")
    checkpoint.restore('./training_checkpoints/ckpt-' + str(UseEpoch))
    print('using epoc ' + str(UseEpoch))

def create_img_folder(name):
    try:
        os.mkdir(f'./{name}/')
        print(f"created folder {name}")
    except:
        print(f"folder <{name}> already existed")

def main():
    if args.mode == 'train' :
        prepareTrainTestFolder()
        fit(train_dataset, int(args.epoch))

    elif args.mode == 'predict' :
        #generate images using trained weight
        create_img_folder(save_img_folder_name)
        create_img_folder(save_img_folder_name_nii)
        load_weights(args.epoch)
        for (inp, tar), name in zip(test_dataset,names):
            name_string = name.numpy().decode().split('test\\')[1].split('_acq')[0]
            generate_images_predict_png(generator, inp, tar, name_string)
            generate_images_predict_nii(generator, inp, tar, name_string)
    
    else:
        print()
        print('Unvalid mode... please check --help for more info.')
        sys.exit(1)

if __name__ == "__main__":
    main()
