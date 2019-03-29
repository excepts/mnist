# import input_data
# mnist = input_data.read_data_sets("/Users/excepts/Downloads/", one_hot=True)

import gzip
import numpy as np
from struct import unpack


def read_image(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img


def normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img


def read_label(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab


def one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab


def load_mnist(train_image_path, train_label_path, test_image_path, test_label_path,
               normalize=True, one_hot=True):
    """读入MNIST数据集
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot :
        one_hot为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    Returns
    ----------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    image = {
        'train': read_image(train_image_path),
        'test': read_image(test_image_path)
    }

    label = {
        'train': read_label(train_label_path),
        'test': read_label(test_label_path)
    }

    if normalize:
        for key in ('train', 'test'):
            image[key] = normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = one_hot_label(label[key])

    return (image['train'], label['train']), (image['test'], label['test'])


if __name__ == '__main__':
    load_mnist('/Users/excepts/Downloads/train-images-idx3-ubyte.gz',
               '/Users/excepts/Downloads/train-labels-idx1-ubyte.gz',
               '/Users/excepts/Downloads/t10k-images-idx3-ubyte.gz',
               '/Users/excepts/Downloads/t10k-labels-idx1-ubyte.gz')
