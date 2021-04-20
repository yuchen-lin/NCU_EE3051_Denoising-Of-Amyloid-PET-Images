import tensorflow as tf
from tensorflow.keras.utils import plot_model
import os
import time
import numpy as np
import datetime
import argparse
import sys

from models.load import load_tfrecord, load_png
from models.unet_2d import model as md2
from models.unet_3d import model as md3
from models.gen_img import generate_images_2d, generate_images_3d, stack_png_to_nii

def prepareCheckpointsFolder(selectModel):
    try:
        os.mkdir('./training_checkpoints')
    except:
        pass
    
    try:
        os.mkdir(f'./training_checkpoints/{selectModel}')
    except:
        pass

def generator_loss(gen_output, target):
    l1_loss = tf.reduce_mean(tf.abs(gen_output - target))

    return l1_loss

@tf.function
def train_step(generator, generator_optimizer, summary_writer, input_image, target, epoch):
    with tf.GradientTape() as gen_tape:
        gen_output = generator(input_image, training=True)
        l1_loss = generator_loss(gen_output, target)

    generator_gradients = gen_tape.gradient(l1_loss,generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('l1_loss', l1_loss, step=epoch)

def fit(generator, generator_optimizer, summary_writer, train_ds, epochs, train_num, checkpoint, checkpoint_prefix, BATCH_SIZE):
    print('To view training loss, type "tensorboard --logdir=logs" in anaconda prompt (environment activated)')
    print()
    for epoch in range(epochs):
        start = time.time()
            
        # Train
        print(f"porcessing epoch{epoch+1}...")
        print()
        count=0
        for n, (input_image, target) in train_ds.enumerate():
            count+=1
            train_step(generator, generator_optimizer, summary_writer, input_image, target, epoch)
            if count%10==0 :
                print(f'[{count}/{int(train_num/BATCH_SIZE)}]')
        
        checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch+1,time.time()-start))

def load_weights(UseEpoch, checkpoint, model_name):
    print("Loading epoch "+str(UseEpoch)+"...")
    checkpoint.restore(f'./training_checkpoints/{model_name}/ckpt-' + str(UseEpoch))
    print('using epoc ' + str(UseEpoch))

def create_img_folder(name):
    try:
        os.mkdir(f'./{name}/')
        print(f"created folder {name}")
    except:
        print(f"folder <{name}> already existed")

def main():
    train_dir_3d = './preprocessed_images/train_3D'
    test_dir_3d = './preprocessed_images/test_3D'
    train_dir_2d = './preprocessed_images/train_2D'
    test_dir_2d = './preprocessed_images/test_2D'
    
    d=128
    h=128
    w=128

    OUTPUT_CHANNELS = 1
    learning_rate = 2e-4
    save_img_folder_name = 'results/outputs_png_3d_Unet'
    save_img_folder_name_nii = 'results/outputs_nii_3d_Unet'
    save_img_folder_name_2d = 'results/outputs_png_2d_Unet'
    save_img_folder_name_nii_2d = 'results/outputs_nii_2d_Unet'
    train_num_2d = len(os.listdir(train_dir_2d))
    test_num_2d = len(os.listdir(test_dir_2d))
    train_num_3d = len(os.listdir(train_dir_3d))
    test_num_3d = len(os.listdir(test_dir_3d))

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="train or predict")
    parser.add_argument("--model", help="2d or 3d")
    parser.add_argument("--epoch", help="If mode=train => number of training epoch./nIf mode=predict => predict on specified epoch's weight.")
    args = parser.parse_args()
    if not args.mode :
        print('Please specify epoch!')
        print('Check --info for more information.')
        sys.exit(1)
    elif not args.model :
        print('Please specify model!')
        print('Check --info for more information.')
        sys.exit(1)
    elif not args.epoch :
        print('Please specify mode!')
        print('Check --info for more information.')
        sys.exit(1)

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

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
    if args.mode == 'train':
        if args.model == '2d':
            BATCH_SIZE = 50
            train_dataset, test_dataset, names = load_png(train_dir_2d, test_dir_2d, BATCH_SIZE, train_num_2d)
            generator = md2(h, w, OUTPUT_CHANNELS)

            prepareCheckpointsFolder('2d')
            checkpoint_dir = f'./training_checkpoints/2d'
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)
            log_dir="logs/"
            summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

            #tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64, to_file="model_unet2D.png")
            fit(generator, generator_optimizer, summary_writer, train_dataset, int(args.epoch), train_num_2d, checkpoint, checkpoint_prefix, BATCH_SIZE)

        elif args.model == '3d':
            BATCH_SIZE = 1
            train_dataset, test_dataset, names = load_tfrecord(train_dir_3d, test_dir_3d, BATCH_SIZE, train_num_3d)
            generator = md3(d, h, w, OUTPUT_CHANNELS)

            prepareCheckpointsFolder('3d')
            checkpoint_dir = f'./training_checkpoints/3d'
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)
            log_dir="logs/"
            summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

            #tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64, to_file="model_unet3D.png")
            fit(generator, generator_optimizer, summary_writer, train_dataset, int(args.epoch), train_num_3d, checkpoint, checkpoint_prefix, BATCH_SIZE)

        else:
            print()
            print('Unvalid model... please check --help for more info.')
            sys.exit(1)

    elif args.mode == 'predict':
        create_img_folder('results')
        if args.model == '2d':
            BATCH_SIZE = 1
            print("Total train data = "+str(train_num_2d))
            print("Total test data = "+str(test_num_2d))
            train_dataset, test_dataset, names = load_png(train_dir_2d, test_dir_2d, BATCH_SIZE, train_num_2d)
            generator = md2(h, w, OUTPUT_CHANNELS)
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)
            create_img_folder(save_img_folder_name_2d)
            create_img_folder(save_img_folder_name_nii_2d)
            load_weights(args.epoch, checkpoint, '2d')
            for (inp, tar), name in zip(test_dataset,names):
                name_string = name.numpy().decode().split('test_2D\\')[1]
                generate_images_2d(generator, inp, tar, name_string, save_img_folder_name_2d)
            stack_png_to_nii(save_img_folder_name_2d, save_img_folder_name_nii_2d)

        elif args.model == '3d':
            BATCH_SIZE = 1
            print("Total train data = "+str(train_num_3d))
            print("Total test data = "+str(test_num_3d))
            train_dataset, test_dataset, names = load_tfrecord(train_dir_3d, test_dir_3d, BATCH_SIZE, train_num_3d)
            generator = md3(d, h, w, OUTPUT_CHANNELS)
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)
            create_img_folder(save_img_folder_name)
            create_img_folder(save_img_folder_name_nii)
            load_weights(args.epoch, checkpoint, '3d')
            for (inp, tar), name in zip(test_dataset,names):
                name_string = name.numpy().decode().split('test_3D\\')[1].split('_acq')[0]
                generate_images_3d(generator, inp, tar, name_string, save_img_folder_name, save_img_folder_name_nii)
    
    else:
        print()
        print('Unvalid mode... please check --help for more info.')
        sys.exit(1)

if __name__ == "__main__":
    main()