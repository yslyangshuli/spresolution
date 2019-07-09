'''
数据处理的类
'''
import tensorflow as tf
import random

'''
这个产生的数据是img/255,相应地，图像测试也应该img/255
'''
class InputProducer(object):

    def __init__(self, datalistpath, trainpath, epoch=300, crop_size=256, chns=3, batch_size=10 ):
        self.crop_size = crop_size
        self.chns = chns
        self.data_list = open(datalistpath, 'rt').read().splitlines()
        self.data_list = list(map(lambda x: x.split(' '), self.data_list))
        random.shuffle(self.data_list)
        self.batch_size = batch_size
        self.trainpath = trainpath
        self.epoch = epoch
        self.data_size = (len(self.data_list)) // self.batch_size
        self.max_steps = int(self.epoch * self.data_size)

    def GetMaxStep(self):
        return self.max_steps

    def _readdata(self):
        img_a = tf.image.decode_image(tf.read_file(tf.string_join([self.trainpath, self.data_queue[0]])),
                                      channels=3)
        img_b = tf.image.decode_image(tf.read_file(tf.string_join([self.trainpath, self.data_queue[1]])),
                                      channels=3)
        img_a, img_b = self._preprocessing([img_a, img_b])  # 图片归一化
        return img_a, img_b

    def _preprocessing(self, imgs):
        imgs = [tf.cast(img, tf.float32) / 255.0 for img in imgs]
        #tf.unstack：矩阵分解
        img_crop = tf.unstack(tf.random_crop(tf.stack(imgs, axis=0), [2, self.crop_size, self.crop_size, self.chns]),
                              axis=0)
        return img_crop

    def GetDataProducer(self):
        with tf.variable_scope('input'):
            list_all = tf.convert_to_tensor(self.data_list, dtype=tf.string)
            gt_list = list_all[:, 0]#第一列是清晰图像
            blur_list = list_all[:, 1] #第二列是模糊图像
            self.data_queue = tf.train.slice_input_producer([blur_list, gt_list], capacity=20)
            image_blur, image_gt = self._readdata()
            batch_blur, batch_gt = tf.train.batch([image_blur, image_gt], batch_size=self.batch_size, num_threads=8, capacity=20)
        return batch_blur, batch_gt


