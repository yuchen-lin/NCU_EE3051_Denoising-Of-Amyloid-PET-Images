import SimpleITK as sitk
import numpy as np
import PIL.Image
import tensorflow as tf
import os

def generate_images_3d(model, test_input, tar, name, save_img_folder_name, save_img_folder_name_nii):
    print("predicting image"+str(name)+"...")
    prediction = model(test_input, training=True)
    display_list = [tf.squeeze(test_input[0]), tf.squeeze(tar[0]), tf.squeeze(prediction[0])]
    
    im1=tf.display_list[0] * 0.5 + 0.5
    im2=tf.display_list[1] * 0.5 + 0.5
    im3=tf.display_list[2] * 0.5 + 0.5

    out1 = sitk.GetImageFromArray(im1)
    out2 = sitk.GetImageFromArray(im2)
    out3 = sitk.GetImageFromArray(im3)
        
    outname1 = f'./{save_img_folder_name_nii}/{name}_input.nii.gz'
    sitk.WriteImage(out1,outname1)
    print(f'{outname1} saved')
    outname2 = f'./{save_img_folder_name_nii}/{name}_target.nii.gz'
    sitk.WriteImage(out2,outname2)
    print(f'{outname2} saved')
    outname3 = f'./{save_img_folder_name_nii}/{name}_predict.nii.gz'
    sitk.WriteImage(out3,outname3)
    print(f'{outname3} saved')
    print()

    for sl in range(128):
        im1_sl=im1[sl]
        im2_sl=im2[sl]
        im3_sl=im3[sl]
    
        output = np.hstack((im1_sl,im2_sl,im3_sl))   #input/target/prediction
        output = output/np.max(output)
        output = output*255
    
        outname = f'./{save_img_folder_name}/{str(name).zfill(3)}.png'
        PIL.Image.fromarray(output).convert('L').save(outname)
        print(f'{outname} saved\n')

def generate_images_2d(model, test_input, tar, name, save_img_folder_name):
    print("predicting image"+str(name)+"...")
    prediction = model(test_input, training=True)
    display_list = [tf.squeeze(test_input[0]), tf.squeeze(tar[0]), tf.squeeze(prediction[0])]

    im1=tf.display_list[0] * 0.5 + 0.5
    im2=tf.display_list[1] * 0.5 + 0.5
    im3=tf.display_list[2] * 0.5 + 0.5

    output = np.hstack((im1,im2,im3))   #input/target/prediction
    output = output/np.max(output)
    output = output*255

    outname = f'./{save_img_folder_name}/{name}'
    PIL.Image.fromarray(output).convert('L').save(outname)
    print(f'{outname} saved\n')

def stack_png_to_nii(load_folder, save_folder):
    output = np.zeros(shape=(128, 384))
    for n, img in enumerate(os.listdir(load_folder)):
        img_arr = np.array(PIL.Image.open(img))[:,:,0]
        if(n%126 == 0): #start stacking
            i=0
            output = np.stack((output, img_arr), axis=0)
        elif(i < 127):
            i+=1
            output = np.stack((output, img_arr), axis=0)
        else:
            name = img.split('.png')[0]
            outname1 = f'./{save_folder}/{name}_input.nii.gz'
            sitk.WriteImage(output[:,:,:128],outname1)
            print(f'{outname1} saved')
            outname2 = f'./{save_folder}/{name}_target.nii.gz'
            sitk.WriteImage(output[:,:,128:256],outname2)
            print(f'{outname2} saved')
            outname3 = f'./{save_folder}/{name}_predict.nii.gz'
            sitk.WriteImage(output[:,:,256:],outname3)
            print(f'{outname3} saved')
            print()
            output = np.zeros(shape=(128, 384))