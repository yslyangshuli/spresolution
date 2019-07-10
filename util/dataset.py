'''
数据处理的类
'''
import tensorflow as tf
import random
import util.util as util

'''
这个产生的数据是img/255,相应地，图像测试也应该img/255
'''
class InputProducer(object):

    def __init__(self,
                 datalistpath,
                 trainpath,
                 epoch=300,
                 crop_size=192,
                 chns=3,
                 batch_size=10,
                 factor=4):
        self.crop_size = crop_size
        self.chns = chns
        self.data_list = open(datalistpath, 'rt').read().splitlines()
        # self.data_list = list(map(lambda x: x.split(' '), self.data_list))
        random.shuffle(self.data_list)
        self.batch_size = batch_size
        self.trainpath = trainpath
        self.epoch = epoch
        self.factor = factor
        self.data_size = (len(self.data_list)) // self.batch_size
        self.max_steps = int(self.epoch * self.data_size)

    def GetMaxStep(self):
        return self.max_steps

    def _readdata(self):
        img_a = tf.image.decode_image(tf.read_file(tf.string_join([self.trainpath, self.data_queue[0]])),
                                      channels=3)
        # img_b = tf.image.decode_image(tf.read_file(tf.string_join([self.trainpath, self.data_queue[1]])),
        #                               channels=3)
        img_a = self._preprocessing(img_a)  # 图片归一化
        img_b = util.ScaleImage(img_a, scale=1/self.factor)
        return img_a, img_b

    def _preprocessing(self, imgs):
        imgs = [tf.cast(img, tf.float32) / 255.0 for img in imgs]
        img_crop = tf.random_crop(imgs, [1, self.crop_size, self.crop_size, self.chns])
        return img_crop

    def GetDataProducer(self):
        with tf.variable_scope('input'):
            list_all = tf.convert_to_tensor(self.data_list, dtype=tf.string)
            gt_list = list_all[:]   # 清晰图像
            self.data_queue = tf.train.slice_input_producer([gt_list], capacity=20)
            img_a, img_b = self._readdata()  # 左边HR图像 右边LR图像
            batch_a, batch_b = tf.train.batch([img_a, img_b], batch_size=self.batch_size, num_threads=8, capacity=20)
        return batch_a, batch_b


