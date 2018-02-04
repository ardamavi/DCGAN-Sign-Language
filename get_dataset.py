# Arda Mavi
import os
import numpy as np
from os import listdir
from scipy.misc import imread, imresize


# Settings:
img_size = 64
channel_size = 1 # 1: Grayscale, 3: RGB


def get_img(data_path):
    # Getting image array from path:
    img = imread(data_path, flatten= True if channel_size == 1 else False)
    img = imresize(img, (img_size, img_size, channel_size))
    return img

def get_dataset(dataset_path='Data/Train_Data'):
    # Getting all data from data path:
    try:
        X = np.load('Data/npy_dataset/X.npy')
    except:
        labels = listdir(dataset_path) # Geting labels
        X = []
        for label in labels:
            datas_path = dataset_path+'/'+label
            for data in listdir(datas_path):
                img = get_img(datas_path+'/'+data)
                X.append(img)
        # Create dateset:
        X = 1-np.array(X).astype('float32')/255.
        X = X.reshape(X.shape[0], img_size, img_size, channel_size)
        if not os.path.exists('Data/npy_dataset/'):
            os.makedirs('Data/npy_dataset/')
        np.save('Data/npy_dataset/X.npy', X)
    return X

if __name__ == '__main__':
    get_dataset()
