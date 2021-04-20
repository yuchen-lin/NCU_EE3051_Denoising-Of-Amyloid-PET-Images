import SimpleITK as sitk
import numpy as np
import PIL.Image
import tensorflow as tf
import os

def generate_images_3d(model, test_input, tar, name, save_img_folder_name, save_img_folder_name_nii):
    print("predicting image"+str(name)+"...")

    if mode == '3d':
        prediction = model(test_input, training=True)
    elif mode = '2d':
        prediction = []
        for i in range(90):
            prediction.append(model(test_input[i]))
    
    display_list = [tf.squeeze(test_input[0]), tf.squeeze(tar[0]), tf.squeeze(prediction[0])]
    
    im1=display_list[0][0:90] * 0.5 + 0.5
    im2=display_list[1][0:90] * 0.5 + 0.5
    im3=display_list[2][0:90] * 0.5 + 0.5

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

    for n, sl in enumerate(range(90)):
        im1_sl=im1[sl]
        im2_sl=im2[sl]
        im3_sl=im3[sl]

        output = np.hstack((im1_sl,im2_sl,im3_sl))   #input/target/prediction
        output = output*255
    
        outname = f'./{save_img_folder_name}/{name}_{str(sl).zfill(3)}.png'
        PIL.Image.fromarray(output).convert('L').save(outname)
        print(f'{outname} saved\n')