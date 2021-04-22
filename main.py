import tensorflow as tf
from tensorflow.keras.utils import plot_model
import os
import time
import numpy as np
import datetime
import argparse
import sys
import PIL.Image

from models.mkfolder import mfdr
from models.load import load_tfrecord, load_png
from models.unet_2d import model as md2
from models.unet_3d import model as md3
from models.gen_img import generate_images

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

def demo(mode, model, inp, tar, sli, ep):
    if mode == '3d':
        prediction = model(inp, training=True)
        pre = prediction[0,sli]
        training_demo_path = f'training_demo/3d'
    elif mode == '2d':
        pre = model(tf.expand_dims(inp[0,sli], axis=0), training=True)[0]
        training_demo_path = f'training_demo/2d'  

    inp = inp[0,sli]
    tar = tar[0,sli]
    output = np.hstack((inp, tar, pre))   #input/target/prediction
    output = output*0.5+0.5 # Rescale from [-1,1] to [0,1]
    output = output/np.max(output)  # Normalize
    output = output*255 # Rescale from [0,1] to [0,255]

    outname = f'./{training_demo_path}/{str(ep)}.png'
    PIL.Image.fromarray(output[:,:,0]).convert('L').save(outname)
    print(f'{outname} saved\n')

def fit(generator, generator_optimizer, summary_writer, train_ds, test_ds, epochs, train_num, checkpoint, checkpoint_prefix, BATCH_SIZE, mode):
    print('\nTo view training loss, type "tensorboard --logdir=logs" in anaconda prompt (environment activated)\n')

    for epoch in range(epochs):
        start = time.time()

        # Train
        print(f"porcessing epoch{epoch+1}...")
        print()
        count=0
        for n, (input_image, target) in train_ds.enumerate():
            count+=1
            if mode == '3d':
                train_step(generator, generator_optimizer, summary_writer, input_image, target, epoch)
            elif mode == '2d':
                for i in range(90):
                    train_step(generator, generator_optimizer, summary_writer, tf.expand_dims(input_image[0,i], axis=0), tf.expand_dims(target[0,i], axis=0), epoch)
            if count%10==0 :
                print(f'[{count}/{int(train_num/BATCH_SIZE)}]')
        
        for inp, tar in test_ds.take(1):
            demo(mode, generator, inp, tar, 60, epoch+1)

        checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch+1,time.time()-start))

def load_weights(UseEpoch, checkpoint, model_name):
    print("Loading epoch "+str(UseEpoch)+"...")
    checkpoint.restore(f'./training_checkpoints/{model_name}/ckpt-' + str(UseEpoch))
    print('using epoc ' + str(UseEpoch))

def main():
    train_dir = './preprocessed_images/train'
    test_dir = './preprocessed_images/test'

    train_num = len(os.listdir(train_dir))
    test_num = len(os.listdir(test_dir))
    
    d=128
    h=128
    w=128
    OUTPUT_CHANNELS = 1
    learning_rate = 2e-4
    BATCH_SIZE = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="train or predict")
    parser.add_argument("--model", help="2d or 3d")
    parser.add_argument("--epoch", help="If mode=train => number of training epoch./nIf mode=predict => predict on specified epoch's weight.")
    args = parser.parse_args()

    if not args.mode :
        print('Please specify mode!')
        print('Check --info for more information.')
        sys.exit(1)
    elif not args.model :
        print('Please specify model!')
        print('Check --info for more information.')
        sys.exit(1)
    elif not args.epoch :
        print('Please specify epoch!')
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
        train_dataset, test_dataset, names = load_tfrecord(train_dir, test_dir, BATCH_SIZE, train_num)
        mfdr(('./training_checkpoints', f'./training_checkpoints/{args.model}'))
        checkpoint_dir = f'./training_checkpoints/{args.model}'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        log_dir="logs/"
        summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        if args.model == '2d':
            mfdr(('./training_demo', './training_demo/2d'))
            generator = md2(h, w, OUTPUT_CHANNELS)
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)
            #tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64, to_file="model_unet2D.png")
            
            # Result before traing
            for inp, tar in test_dataset.take(1):
                demo('2d', generator, inp, tar, 60, 0)
            
            fit(generator, generator_optimizer, summary_writer, train_dataset, test_dataset, int(args.epoch), train_num, checkpoint, checkpoint_prefix, BATCH_SIZE, '2d')

        elif args.model == '3d':
            mfdr(('./training_demo', './training_demo/3d'))
            generator = md3(d, h, w, OUTPUT_CHANNELS)
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)
            #tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64, to_file="model_unet3D.png")
            
            # Result before traing
            for inp, tar in test_dataset.take(1):
                demo('3d', generator, inp, tar, 60, 0)
            
            fit(generator, generator_optimizer, summary_writer, train_dataset, test_dataset, int(args.epoch), train_num, checkpoint, checkpoint_prefix, BATCH_SIZE, '3d')

        else:
            print()
            print('Unvalid model... please check --help for more info.')
            sys.exit(1)

    elif args.mode == 'predict':
        mfdr(['results'])
        print("Total train data = "+str(train_num))
        print("Total test data = "+str(test_num))
        train_dataset, test_dataset, names = load_tfrecord(train_dir, test_dir, BATCH_SIZE, train_num)

        if args.model == '2d':
            generator = md2(h, w, OUTPUT_CHANNELS)
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)
            mfdr([f'./results/epoch_{str(args.epoch).zfill(3)}'])
            save_img_folder_name_png = f'results/epoch_{str(args.epoch).zfill(3)}/png_2d'
            save_img_folder_name_nii = f'results/epoch_{str(args.epoch).zfill(3)}/nii_2d'
            mfdr((save_img_folder_name_png, save_img_folder_name_nii))
            load_weights(args.epoch, checkpoint, '2d')

        elif args.model == '3d':
            generator = md3(d, h, w, OUTPUT_CHANNELS)
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)
            mfdr([f'./results/epoch_{str(args.epoch).zfill(3)}'])
            save_img_folder_name_png = f'results/epoch_{str(args.epoch).zfill(3)}/png_3d'
            save_img_folder_name_nii = f'results/epoch_{str(args.epoch).zfill(3)}/nii_3d'
            mfdr((save_img_folder_name_png, save_img_folder_name_nii))
            load_weights(args.epoch, checkpoint, '3d')
            
        for (inp, tar), name in zip(test_dataset,names):
            name_string = name.numpy().decode().split('test\\')[1].split('_acq')[0]
            generate_images(generator, inp, tar, name_string, save_img_folder_name_png, save_img_folder_name_nii, args.model)
    
    else:
        print()
        print('Unvalid mode... please check --help for more info.')
        sys.exit(1)

if __name__ == "__main__":
    main()