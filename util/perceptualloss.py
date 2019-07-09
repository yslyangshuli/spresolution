'''
感知损失函数
'''
import tensorflow as tf
import network.vgg as vgg
import tensorflow.contrib.slim as slim

class PerceptualLoss(object):

    def __init__(self):
        pass

    def _rgbmeanfun(self, rgb):
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        #将RGB转成BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3,
                                    value=rgb)
        rgbmean = tf.concat(axis=3, values=[red - _R_MEAN, green - _G_MEAN, blue - _B_MEAN])
        return rgbmean

    def Loss(self, image1, image2, is_default=True, *point_name):
        '''
        输入归一化之后的图像
        :param image1: 第一幅图像
        :param image2: 第二幅图像
        :param is_default: 是否要指定卷积层的结果
        :param point_name: 如果is_default为false的话，那么这里就需要值
        :return: 返回感知损失
        '''
        image1 = (image1 + 1)*127.5
        image2 = (image2 + 1)*127.5
        dbatch = tf.concat([image1, image2], axis=0)
        rgbmean = self._rgbmeanfun(dbatch)
        _, end_points = vgg.vgg_19(rgbmean, num_classes=1000,
                                   is_training=False, spatial_squeeze=False)
        content_loss = 0
        if is_default:
            conv = end_points['vgg_19/conv5/conv5_4']
            #tf.split(value, num_or_size_splits, axis, num, name)
            #这里是在axis=0这个维度上平均分成两份，所以不用担心batch的大小
            fmap = tf.split(conv, 2)#这里的2代表将conv分成两份，默认在axis=0这个维度上划分
            content_loss = tf.losses.mean_squared_error(fmap[0], fmap[1])
        else:
            for name in point_name:
                conv = end_points[name]
                fmap = tf.split(conv, 2)
                content_loss += tf.losses.mean_squared_error(fmap[0], fmap[1])
        return content_loss


    def LoadMoel(self, checkpointpath, sess, vggname='vgg_19'):
        '''
        :param checkpointpath: vgg模型的文件，一直到cpkt文件
        :param sess: tf.Session()函数里面的sess
        :param vggname:要使用的vgg模型是vgg19还是vgg16
        '''
        init_fn = slim.assign_from_checkpoint_fn(checkpointpath, slim.get_model_variables(vggname))
        init_fn(sess)

