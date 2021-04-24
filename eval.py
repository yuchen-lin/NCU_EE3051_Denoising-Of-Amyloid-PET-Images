import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import os
import csv
import math
from skimage.metrics import structural_similarity as ssim

from models.load import load_tfrecord
from models.unet_2d import model as md2
from models.unet_3d import model as md3
from models.mkfolder import mfdr

def load_weights(UseEpoch, checkpoint, model_name):
    print("Loading epoch "+str(UseEpoch)+"...")
    checkpoint.restore(f'./training_checkpoints/{model_name}/ckpt-' + str(UseEpoch))
    print('using epoc ' + str(UseEpoch))
    print('model = '+model_name)

def calc_PSNR_RMSE(im, tar):
    threshold = 0.01
    dif = []
    for d in range(90):
        for h in range(128):
            for w in range(128):
                if(tar[d,h,w] > threshold):
                    dif.append(im[d,h,w] - tar[d,h,w])
                        
    dif = np.array(dif)
    mse = np.mean(dif**2)
    max_val = 1

    PSNR = 20*math.log10(max_val/math.sqrt(mse))
    RMSE = math.sqrt(mse)

    return PSNR, RMSE 

def calc_SSIM(im, tar):
    SSIM = []
    for d in range(90):
        SSIM.append(ssim(im[d], tar[d], win_size=11, data_range=1.0, K1=0.01, K2=0.03))

    SSIM = np.mean(SSIM)
    
    return SSIM

def main():
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

    mfdr(['results'])
    train_dir = './preprocessed_images/train'
    test_dir = './preprocessed_images/test'
    train_num = len(os.listdir(train_dir))
    test_num = len(os.listdir(test_dir))

    d=128
    h=128
    w=128
    OUTPUT_CHANNELS = 1
    learning_rate = 2e-4
    result_dir = './results'
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
    train_dataset, test_dataset, names = load_tfrecord(train_dir, test_dir, 1, train_num)

    for ckpt_name in os.listdir('training_checkpoints'):
        with open(f'{result_dir}/{ckpt_name}_eval.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['', 'input', '', '', '', '', '', 'prediction'])
            writer.writerow(['epoch', 'PSNR_mean', 'PSNR_std', 'SSIM_mean','SSIM_std', 'RMSE_mean', 'RMSE_std', 'PSNR_mean', 'PSNR_std', 'SSIM_mean', 'SSIM_std', 'RMSE_mean', 'RMSE_std'])

            if(ckpt_name == '2d'):
                generator = md2(h, w, OUTPUT_CHANNELS)
            elif(ckpt_name == '3d'):
                generator = md3(d, h, w, OUTPUT_CHANNELS)
            else:
                break
        
            checkpoint_dir = f'./training_checkpoints/{ckpt_name}'
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)

            finished_calc_inp = False

            for i in range(int((len(os.listdir(f'./training_checkpoints/{ckpt_name}'))-1)/2)):
                load_weights(i+1, checkpoint, ckpt_name)
                PSNR_inp = []
                PSNR_inp_std = []
                SSIM_inp = []
                SSIM_inp_std = []
                RMSE_inp = []
                RMSE_inp_std = []
                PSNR_pre = []
                PSNR_pre_std = []
                SSIM_pre = []
                SSIM_pre_std = []
                RMSE_pre = []
                RMSE_pre_std = []
                if(ckpt_name == '2d'):
                    for n, (inp, tar) in enumerate(test_dataset):
                        prediction = []
                        for sli in range(90):
                            prediction.append(generator(tf.expand_dims(inp[0,sli], axis=0), training=True)[0])

                        inp_norm = np.array(tf.squeeze(inp[0]) * 0.5 + 0.5)
                        tar_norm = np.array(tf.squeeze(tar[0]) * 0.5 + 0.5)
                        pre_norm = np.array(prediction)[:,:,:,0] * 0.5 + 0.5

                        if not finished_calc_inp:
                            inp_psnr, inp_rmse = calc_PSNR_RMSE(inp_norm, tar_norm)
                            inp_ssim = calc_SSIM(inp_norm, tar_norm)
                            PSNR_inp.append(inp_psnr)
                            SSIM_inp.append(inp_ssim)
                            RMSE_inp.append(inp_rmse)

                        pre_psnr, pre_rmse = calc_PSNR_RMSE(pre_norm, tar_norm)
                        pre_ssim = calc_SSIM(pre_norm, tar_norm)
                        PSNR_pre.append(pre_psnr)
                        SSIM_pre.append(pre_ssim)
                        RMSE_pre.append(pre_rmse)

                        if((n+1)%5 == 0):
                            print(f'[{n+1}/{test_num}]')

                elif(ckpt_name == '3d'):
                    for n, (inp, tar) in enumerate(test_dataset):
                        prediction = generator(inp, training=True)

                        inp_norm = np.array(tf.squeeze(inp[0]) * 0.5 + 0.5)
                        tar_norm = np.array(tf.squeeze(tar[0]) * 0.5 + 0.5)
                        pre_norm = np.array(tf.squeeze(prediction[0]) * 0.5 + 0.5)

                        if not finished_calc_inp:
                            inp_psnr, inp_rmse = calc_PSNR_RMSE(inp_norm, tar_norm)
                            inp_ssim = calc_SSIM(inp_norm, tar_norm)
                            PSNR_inp.append(inp_psnr)
                            SSIM_inp.append(inp_ssim)
                            RMSE_inp.append(inp_rmse)

                        pre_psnr, pre_rmse = calc_PSNR_RMSE(pre_norm, tar_norm)
                        pre_ssim = calc_SSIM(pre_norm, tar_norm)
                        PSNR_pre.append(pre_psnr)
                        SSIM_pre.append(pre_ssim)
                        RMSE_pre.append(pre_rmse)

                        if((n+1)%5 == 0):
                            print(f'[{n+1}/{test_num}]')

                if not finished_calc_inp:
                    write_psnr_inp = str(np.mean(np.array(PSNR_inp)))
                    write_psnr_inp_std = str(np.std(np.array(PSNR_inp)))
                    write_ssim_inp = str(np.mean(np.array(SSIM_inp)))
                    write_ssim_inp_std = str(np.std(np.array(SSIM_inp)))
                    write_rmse_inp = str(np.mean(np.array(RMSE_inp)))
                    write_rmse_inp_std = str(np.std(np.array(RMSE_inp)))
                    finished_calc_inp = True
                write_psnr_pre = str(np.mean(np.array(PSNR_pre)))
                write_psnr_pre_std = str(np.std(np.array(PSNR_pre)))
                write_ssim_pre = str(np.mean(np.array(SSIM_pre)))
                write_ssim_pre_std = str(np.std(np.array(SSIM_pre)))
                write_rmse_pre = str(np.mean(np.array(RMSE_pre)))
                write_rmse_pre_std = str(np.std(np.array(RMSE_pre)))

                writer.writerow([str(i+1), write_psnr_inp, write_psnr_inp_std, write_ssim_inp, write_ssim_inp_std, write_rmse_inp, write_rmse_inp_std, write_psnr_pre, write_psnr_pre_std, write_ssim_pre, write_ssim_pre_std, write_rmse_pre, write_rmse_pre_std])
                print(f'Finished epoch {i+1}:')
                print(f'inp_psnr = {write_psnr_inp}')
                print(f'inp_psnr_std = {write_psnr_inp_std}\n')
                print(f'pre_psnr = {write_psnr_pre}')
                print(f'pre_psnr_std = {write_psnr_pre_std}\n')
                print(f'inp_ssim = {write_ssim_inp}')
                print(f'inp_ssim_std = {write_ssim_inp_std}\n')
                print(f'pre_ssim = {write_ssim_pre}')
                print(f'pre_ssim_std = {write_ssim_pre_std}\n')
                print(f'inp_rmse = {write_rmse_inp}')
                print(f'inp_rmse_std = {write_rmse_inp_std}\n')
                print(f'pre_rmse = {write_rmse_pre}')
                print(f'pre_rmse_std = {write_rmse_pre_std}\n')

if __name__ == "__main__":
    main()