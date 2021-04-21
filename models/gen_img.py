import SimpleITK as sitk
import numpy as np
import PIL.Image
import tensorflow as tf
import os

def generate_images(model, inp, tar, name, save_img_folder_name, save_img_folder_name_nii, mode):
    print("predicting image"+str(name)+"...")

    if mode == '3d':
        pre = model(test_input, training=True)[0,:,:,:,0]
    elif mode == '2d':
        pre = []
        for i in range(90):
            pre.append(model(tf.expand_dims(inp[0,i], axis=0), training=True))
        pre = np.array(pre)[0,:,:,:,0]
    
    pre = pre * 0.5 + 0.5
    inp = inp[0,0:90,:,:,0] * 0.5 + 0.5
    tar = tar[0,0:90,:,:,0] * 0.5 + 0.5

    # Save to nii
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

    # Save to png
    for n, sl in enumerate(range(90)):
        im1_sl = inp[sl]
        im2_sl = tar[sl]
        im3_sl = pre[sl]

        output = np.hstack((im1_sl,im2_sl,im3_sl))   #input/target/prediction
        output = output*255 # Rescale from [0,1] to [0,255]
    
        outname = f'./{save_img_folder_name}/{name}_{str(sl).zfill(3)}.png'
        PIL.Image.fromarray(output).convert('L').save(outname)
        print(f'{outname} saved\n')