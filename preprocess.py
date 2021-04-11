import numpy as np
import os
import tensorflow as tf
import SimpleITK as sitk
import argparse
from matplotlib import pyplot as plt

test_path = './tfrecord_denoise/test'
train_path = './tfrecord_denoise/train'
test2D_path = './tfrecord_denoise/test2D'
train2D_path = './tfrecord_denoise/train2D'
try:
    os.mkdir("./tfrecord_denoise")
    os.mkdir(test_path)
    os.mkdir(train_path)
    os.mkdir(test2D_path)
    os.mkdir(train2D_path)
except:
    pass

test_name = ['sub-OAS30024_ses-d0084', # start of positive
             'sub-OAS30026_ses-d129',
             'sub-OAS30026_ses-d0696',
             'sub-OAS30039_ses-d0103',
             'sub-OAS30042_ses-d0067',
             'sub-OAS30075_ses-d148',
             'sub-OAS30075_ses-d0442',
             'sub-OAS30075_ses-d0967',
             'sub-OAS30080_ses-d1318',
             'sub-OAS30098_ses-d0036',
             'sub-OAS30114_ses-d0086',
             'sub-OAS30127_ses-d110',
             'sub-OAS30127_ses-d0837',
             'sub-OAS30135_ses-d2367',
             'sub-OAS30135_ses-d2931', # end of positive
             'sub-OAS30003_ses-d3731', # start of negative
             'sub-OAS30004_ses-d3457',
             'sub-OAS30005_ses-d2384',
             'sub-OAS30006_ses-d2342',
             'sub-OAS30007_ses-d1636',
             'sub-OAS30007_ses-d2722',
             'sub-OAS30008_ses-d1327',
             'sub-OAS30010_ses-d0068',
             'sub-OAS30013_ses-d0102',
             'sub-OAS30025_ses-d2298',
             'sub-OAS30028_ses-d1260',
             'sub-OAS30028_ses-d1847',
             'sub-OAS30044_ses-d61',
             'sub-OAS30044_ses-d1319',
             'sub-OAS30046_ses-d1968'] #end of negative

t = 26  #time frames
d = 128 #depth
h = 256 #height
w = 256 #width

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def normalize(i):
    norm = (i/(np.max(i)/2))-1
    norm[norm>1] = 1.
    norm[norm<-1] = -1.
    return norm

def create_tfrecords(image_arr):
    image_arr_shape = image_arr.shape

    feature = {
        'in_or_out': _int64_feature(image_arr_shape[0]),
        'depth': _int64_feature(image_arr_shape[1]),
        'height': _int64_feature(image_arr_shape[2]),
        'width': _int64_feature(image_arr_shape[3]),
        'kernel': _int64_feature(image_arr_shape[4]),
        'image_raw': _bytes_feature(image_arr.astype(np.float32).tobytes()),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def read(image_path):
    itk_img = sitk.ReadImage(image_path)
    img = sitk.GetArrayFromImage(itk_img)
    return img

def aver(img, frames):
    ave = np.sum(img, axis=0)/frames
    return ave

def select_time(img):
    img_in = img[22]
    img_in = np.append(img_in,np.zeros(shape=(1,h,w)),axis=0)
    img_out = aver(img[22:26],4)
    img_out = np.append(img_out,np.zeros(shape=(1,h,w)),axis=0)
    
    return img_in, img_out

def select_time2D(img):
    img_in = img[22]
    img_out = aver(img[22:26],4)
    return img_in, img_out


def show_info(i,n,num,total):
    print("------------------------------------------------")
    print(f'[{num}/{total}]')
    print(n)
    print(f'shape = {i.shape}')
    print(f'max = {np.max(i)}')
    print(f'min = {np.min(i)}')
    print(f'mean = {np.mean(i)}')

def find_boundary(im_in, im_out):
    top = 0
    bot = h-1
    left = 0
    right = w-1
    threshold = 0.001
    center_depth = 60
    shift = 7
    
    for i in range(1,128):
        if np.mean(im_out[center_depth,top]) > threshold:
            break
        else:
            top+=1
    for i in range(1,128):
        if np.mean(im_out[center_depth,bot]) > threshold:
            break
        else:
            bot-=1
    for i in range(1,128):
        if np.mean(im_out[center_depth,:,left]) > threshold:
            break
        else:
            left+=1
    for i in range(1,128):
        if np.mean(im_out[center_depth,:,right]) > threshold:
            break
        else:
            right-=1
    
    mid1 = int((top+bot)/2+shift)
    mid2 = int((left+right)/2)
    d = 64
    
    im_in = im_in[:,mid1-d:mid1+d,mid2-d:mid2+d]
    im_in = np.rot90(im_in, k=2, axes=(1, 2))
    im_out = im_out[:,mid1-d:mid1+d,mid2-d:mid2+d]
    im_out = np.rot90(im_out, k=2, axes=(1, 2))
    
    return im_in, im_out

def search_folder(path):
    if (path.find(".nii.gz") == -1):
        folder_within_path = "".join(os.listdir(path))
        return search_folder(os.path.join(path,folder_within_path))
    else:
        return path, path.split('\\')[-1]

#Start transforming data from .nii to .tfrecord
def start_preprocessing(in_dir):
    num=0
    total=len(os.listdir(in_dir))
    for name in os.listdir(in_dir):
        path, img_name = search_folder(os.path.join(in_dir,name))
        img = read(path)
        if img.shape != np.zeros(shape=(t,d-1,h,w)).shape :
            print("------------------------------------------------")
            print(f'{img_name} = {img.shape}\ndiscarded...')
            continue
        img_in, img_out = select_time(img)
        
        img = np.append(normalize(img_in),normalize(img_out),axis=0)    #STF,GT分別NORMALIZE
        img_in = img[0:d]
        img_out = img[d:2*d]
        img_in, img_out = find_boundary(img_in, img_out)
        img_final = np.append(np.expand_dims(img_in,axis=0),np.expand_dims(img_out,axis=0),axis=0)
        img_final = np.expand_dims(img_final,axis=4)
        img_final2D = np.append(img_in,img_out,axis=2) #合併 左STF右GT
        num+=1
        show_info(img_final,img_name,num,total)
        #finished transforming from [STF1,STF2,STF3,...,STF26] to [STF23,AVG[STF23,STF24,STF25,STF26]] and normalized
        
        #Start saving to .tfrecord
        record_path = os.path.join(train_path,img_name)
        for tname in test_name:
            if(tname == img_name.split('_acq')[0]):
                record_path = os.path.join(test_path,img_name)
                test_name.remove(tname)
        
        with tf.io.TFRecordWriter(record_path+'.tfrecords') as writer:
            example = create_tfrecords(img_final)
            writer.write(example.SerializeToString())
        #儲存成2D slice
        slice = 0
        while slice <=127:
            plt.imsave(record_path+slice.zfill(3)+'.png', img_final2D[slice], cmap='gray')   #slice=第幾層
            slice+=1   
            
            
def main():
    parser = argparse.ArgumentParser(description='Preprocess OASIS3 AV45 .nii.gz, output will be .tfrecord format.')
    parser.add_argument("--data-dir", help="Folder path, all .nii.gz downloaded from OASIS3 saved in one folder.")
    parser.add_argument("--folder-struc", help="a=> .nii.gz barried within layers of folders; b=> ")
    args = parser.parse_args()
    start_preprocessing(args.data_dir)
if __name__ == "__main__":
    main()
