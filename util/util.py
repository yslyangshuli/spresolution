'''
一些公用的方法
'''
import tensorflow as tf
import cv2
import numpy as np

#图片归一化
def ImgUniform(img):
    return img / 255

def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)

'''
将图像从路径读取，并转换成张量
'''
def _image_process(imgpath1, imgpath2):
    img1_content = tf.read_file(imgpath1)
    img2_content = tf.read_file(imgpath2)
    img1 = tf.image.decode_image(img1_content)
    img2 = tf.image.decode_image(img2_content)
    return img1, img2

'''
计算图像的PSNR
返回一个PSNR张量
'''
def PSNR(imgpath1, imgpath2):
    img1, img2 = _image_process(imgpath1, imgpath2)
    return tf.image.psnr(img1, img2)


'''
计算图像的SSIM
返回一个SSIM张量
'''
def SSIM(imgpath1, imgpath2):
    img1, img2 = _image_process(imgpath1, imgpath2)
    return tf.image.ssim(img1, img2)

'''
保存批量训练的图像
'''
def Save(ten, step, outpath):
    ten = im2uint8(ten)
    write_image_name = outpath + str(step) + ".png"
    shape = ten.shape
    row = 4#每一行4个
    list2 = []
    for i in range(shape[0] // row):
        list1 = []
        for j in range(i*row, (i+1)*row):
            list1.append(ten[j, :, :, :])
        list2.append(np.concatenate(list1, axis=1))
    tmp = np.concatenate(list2, axis=0)
    cv2.imwrite(write_image_name, tmp)

def SaveSingle(ten, step, outpath):
    ten = im2uint8(ten)
    write_image_name = outpath + str(step) + ".png"
    cv2.imwrite(write_image_name, ten[0])