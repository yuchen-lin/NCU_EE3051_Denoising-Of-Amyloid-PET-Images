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
    
    for d in range(0,tar.shape[0]):
        for h in range(0,tar.shape[1]):
            for w in range(0,tar.shape[2]):
                if(tar[d,h,w] > threshold):
                    dif.append(im[d,h,w] - tar[d,h,w])
                        
    dif = np.array(dif)
    mse = np.mean(dif**2)
    max_val = 1

    PSNR = 20*math.log10(max_val/math.sqrt(mse))
    RMSE = math.sqrt(mse)

    return PSNR, RMSE 

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

    mfdr('results')
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
            writer.writerow(['', 'input', '', '', 'prediction'])
            writer.writerow(['epoch', 'PSNR', 'SSIM', 'RMSE', 'PSNR', 'SSIM', 'RMSE'])

            if(ckpt_name == '2d'):
                generator = md2(h, w, OUTPUT_CHANNELS)
            elif(ckpt_name == '3d'):
                generator = md3(d, h, w, OUTPUT_CHANNELS)
            else:
                break
        
            checkpoint_dir = f'./training_checkpoints/{ckpt_name}'
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)

            # I'm gonna deal with input-target first. Not gonna recalculate it for 100 times. NO

            finished_calc_inp = False

            for i in range(int((len(os.listdir(f'./training_checkpoints/{ckpt_name}'))-1)/2)):
                load_weights(i+1, checkpoint, ckpt_name)
                PSNR_inp = SSIM_inp = RMSE_inp = PSNR_pre = SSIM_pre = RMSE_pre = []
                if(ckpt_name == '2d'):
                    for n, (inp, tar) in enumerate(test_dataset):
                        prediction = []
                        for sli in range(90):
                            prediction.append(generator(tf.expand_dims(inp[0,sli], axis=0), training=True))

                        inp_norm = np.array(tf.squeeze(inp[0]) * 0.5 + 0.5)
                        tar_norm = np.array(tf.squeeze(tar[0]) * 0.5 + 0.5)
                        pre_norm = np.array(prediction[:,:,:,0] * 0.5 + 0.5)

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

                        if((n+1)%10 == 0):
                            print(f'[{n+1}/{test_num_2d}]')

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

                        if((n+1)%10 == 0):
                            print(f'[{n+1}/{test_num_3d}]')

                if not finished_calc_inp:
                    write_psnr_inp = str(np.mean(np.array(PSNR_inp)))
                    write_ssim_inp = str(np.mean(np.array(SSIM_inp)))
                    write_rmse_inp = str(np.mean(np.array(RMSE_inp)))
                    finished_calc_inp = True

                writer.writerow([str(i+1), write_psnr_inp, write_ssim_inp, write_rmse_inp, str(np.mean(np.array(PSNR_pre))), str(np.mean(np.array(SSIM_pre))), str(np.mean(np.array(RMSE_pre)))])
                print(f'Finished epoch {i+1}\n')

if __name__ == "__main__":
    main()