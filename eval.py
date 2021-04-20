import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import os
import csv
import math
from skimage.metrics import structural_similarity as ssim

from models.load import load_tfrecord, load_png
from models.unet_2d import model as md2
from models.unet_3d import model as md3

def mkFolder(path):
    try:
        os.mkdir(path)
    except:
        print(f'{path} already exist.')

def load_weights(UseEpoch, checkpoint, model_name):
    print("Loading epoch "+str(UseEpoch)+"...")
    checkpoint.restore(f'./training_checkpoints/{model_name}/ckpt-' + str(UseEpoch))
    print('using epoc ' + str(UseEpoch))
    print('model = '+model_name)

def calc_PSNR_RMSE(inp, tar, pre, finished_calc_inp):
    threshold = 0.01
    dif_inp = []
    dif_pre = []
    for h in range(0,128):
        for w in range(0,128):
            if(tar[h,w] > threshold):
                if not finished_calc_inp:
                    dif_inp.append(tar[h,w] - inp[h,w])
                dif_pre.append(tar[h,w] - pre[h,w])
                        
    if not finished_calc_inp:
        dif_inp = np.array(dif_inp)
    dif_pre = np.array(dif_pre)
    
    if not finished_calc_inp:
        mse_inp = np.mean(dif_inp**2)
    mse_pre = np.mean(dif_pre**2)
    
    max_val = 1

    if finished_calc_inp:
        return 0, 20*math.log10(max_val/math.sqrt(mse_pre)), 0, math.sqrt(mse_pre)

    return 20*math.log10(max_val/math.sqrt(mse_inp)), 20*math.log10(max_val/math.sqrt(mse_pre)), math.sqrt(mse_inp), math.sqrt(mse_pre)

def calc_SSIM(inp, tar, pre, finished_calc_inp):
    if not finished_calc_inp:
        return ssim(inp, tar), ssim(pre, tar)
    else:
        return 0, ssim(pre, tar)
    

def calc_PSNR_RMSE_3D(inp, tar, pre, finished_calc_inp):
    threshold = 0.01
    dif_inp = []
    dif_pre = []
    for d in range(0,90):
        for h in range(0,128):
            for w in range(0,128):
                if(tar[d,h,w] > threshold):
                    if not finished_calc_inp:
                        dif_inp.append(tar[d,h,w] - inp[d,h,w])
                    dif_pre.append(tar[d,h,w] - pre[d,h,w])

    if not finished_calc_inp:  
        dif_inp = np.array(dif_inp)
    dif_pre = np.array(dif_pre)
    
    if not finished_calc_inp:
        mse_inp = np.mean(dif_inp**2)
    mse_pre = np.mean(dif_pre**2)
    
    max_val = 1

    if finished_calc_inp:
        return 0, 20*math.log10(max_val/math.sqrt(mse_pre)), 0, math.sqrt(mse_pre)

    return 20*math.log10(max_val/math.sqrt(mse_inp)), 20*math.log10(max_val/math.sqrt(mse_pre)), math.sqrt(mse_inp), math.sqrt(mse_pre)

def calc_SSIM_3D(inp, tar, pre, finished_calc_inp):
    save_inp = []
    save_pre = []
    for d in range(0,90):
        if not finished_calc_inp:
            save_inp.append(ssim(inp[d], tar[d]))
        save_pre.append(ssim(pre[d], tar[d]))
    
    if not finished_calc_inp:
        return np.mean(np.array(save_inp)), np.mean(np.array(save_pre))
    else:
        return 0, np.mean(np.array(save_pre))

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

    mkFolder('results')
    train_dir_3d = './preprocessed_images/train_3D'
    test_dir_3d = './preprocessed_images/test_3D'
    train_dir_2d = './preprocessed_images/train_2d'
    test_dir_2d = './preprocessed_images/test_2d'
    train_num_2d = len(os.listdir(train_dir_2d))
    test_num_2d = len(os.listdir(test_dir_2d))
    train_num_3d = len(os.listdir(train_dir_3d))
    test_num_3d = len(os.listdir(test_dir_3d))

    d=128
    h=128
    w=128
    OUTPUT_CHANNELS = 1
    learning_rate = 2e-4
    result_dir = './results'
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate)

    for ckpt_name in os.listdir('training_checkpoints'):
        with open(f'{result_dir}/{ckpt_name}_eval.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['', 'input', '', '', 'prediction'])
            writer.writerow(['epoch', 'PSNR', 'SSIM', 'RMSE', 'PSNR', 'SSIM', 'RMSE'])

            if(ckpt_name == '2d'):
                train_dataset, test_dataset, names = load_png(train_dir_2d, test_dir_2d, 1, train_num_2d)
                generator = md2(h, w, OUTPUT_CHANNELS)
            elif(ckpt_name == '3d'):
                train_dataset, test_dataset, names = load_tfrecord(train_dir_3d, test_dir_3d, 1, train_num_3d)
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
                SSIM_inp = []
                RMSE_inp = []
                PSNR_pre = []
                SSIM_pre = []
                RMSE_pre = []
            
                if(ckpt_name == '2d'):
                    for n, (inp, tar) in enumerate(test_dataset):
                        prediction = generator(inp, training=True)
                        display_list = [tf.squeeze(inp[0]), tf.squeeze(tar[0]), tf.squeeze(prediction[0])]

                        inp_norm=np.array(display_list[0] * 0.5 + 0.5)
                        tar_norm=np.array(display_list[1] * 0.5 + 0.5)
                        pre_norm=np.array(display_list[2] * 0.5 + 0.5)

                        inp_psnr, pre_psnr, inp_rmse, pre_rmse = calc_PSNR_RMSE(inp_norm, tar_norm, pre_norm, finished_calc_inp)
                        inp_ssim, pre_ssim = calc_SSIM(inp_norm, tar_norm, pre_norm, finished_calc_inp)

                        if not finished_calc_inp:
                            PSNR_inp.append(inp_psnr)
                            SSIM_inp.append(inp_ssim)
                            RMSE_inp.append(inp_rmse)

                        PSNR_pre.append(pre_psnr)
                        SSIM_pre.append(pre_ssim)
                        RMSE_pre.append(pre_rmse)

                        if((n+1)%10 == 0):
                            print(f'[{n+1}/{test_num_2d}]')

                elif(ckpt_name == '3d'):
                    for n, (inp, tar) in enumerate(test_dataset):
                        prediction = generator(inp, training=True)
                        display_list = [tf.squeeze(inp[0]), tf.squeeze(tar[0]), tf.squeeze(prediction[0])]

                        inp_norm=np.array(display_list[0] * 0.5 + 0.5)
                        tar_norm=np.array(display_list[1] * 0.5 + 0.5)
                        pre_norm=np.array(display_list[2] * 0.5 + 0.5)

                        inp_psnr, pre_psnr, inp_rmse, pre_rmse = calc_PSNR_RMSE_3D(inp_norm, tar_norm, pre_norm, finished_calc_inp)
                        inp_ssim, pre_ssim = calc_SSIM_3D(inp_norm, tar_norm, pre_norm, finished_calc_inp)

                        if not finished_calc_inp:
                            PSNR_inp.append(inp_psnr)
                            SSIM_inp.append(inp_ssim)
                            RMSE_inp.append(inp_rmse)

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