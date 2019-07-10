import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from .base import BaseTrain


class Attention(BaseTrain):

    def __init__(self, p=1, t=2, r=1,
                 datalistpath='./',
                 trainpath='./',
                 logout='./',
                 trainout='./',
                 checkpoint='./'):
        super(Attention, self).__init__(datalistpath, trainpath, checkpoint, logout)
        self.max_scale = 3                      # 上采样最大尺度
        self.each_scale_atten_num = [2, 3, 4]   # 各个尺度中，attention模块的个数。最低的尺度应该层数越小，因为模糊核的尺寸最小
        self.trunk_number = 2                   # 主干分支的残差数
        self.num_mask_layers = 4
        self.max_num_feature = 312
        self.num_init_feature = 160
        self.scale = 1/2
        self.trainout = trainout
        self.logout = logout
        self.checkpoint = checkpoint
        if not os.path.exists(trainout):
            os.makedirs(trainout)
        self.batch_size = 1

    # 预处理的残差网络
    def PreResNet(self, input, dim, scope='rb'):
        with tf.variable_scope(scope):
            net = slim.conv2d(input, dim, [1, 1], normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv1')
            net = slim.conv2d(net, dim, [3, 3], normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv2')
            net = slim.conv2d(net, dim * 4, [1, 1], normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv3')
            b = slim.conv2d(input, dim * 4, [1, 1], normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv1_1')
            return net + b

    # 残差网络
    def ResNet(self, x, scope='rb'):
        with tf.variable_scope(scope):
            c = x.get_shape().as_list()[-1]
            net = slim.conv2d(x, c, [1, 1], normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv1')
            net = slim.conv2d(net, c, [3, 3], normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv2')
            net = slim.conv2d(net, c, [1, 1], normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv3')
            return net + x

    def DownUp_1(self, x, scope='downup'):
        with tf.variable_scope(scope):

            # 下采样
            # c = x.get_shape().as_list()[-1]
            pool1 = slim.max_pool2d(x, ksize=[2, 2], padding='SAME')
            # conv1 = slim.conv2d(x, c, [], 2, normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv1')
            # res11_1 = self.ResNet(conv1, scope='res11_1')
            res11_1 = self.ResNet(pool1, scope='res11_1')
            skip_connection_1 = self.ResNet(res11_1, scope='skip_connection_1')

            pool2 = slim.max_pool2d(res11_1, ksize=[2, 2], padding='SAME')
            res11_2 = self.ResNet(pool2, scope='res11_2')
            skip_connection_2 = self.ResNet(res11_2, scope='skip_connection_2')
            # conv2 = slim.conv2d(res11_1, c, [], 2, normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv2')
            # res11_2 = self.ResNet(conv2, scope='res11_2')

            pool3 = slim.max_pool2d(res11_2, ksize=[2, 2], padding='SAME')
            res11_3 = self.ResNet(pool3, scope='res11_3')
            skip_connection_3 = self.ResNet(res11_3, scope='skip_connection_3')
            # conv3 = slim.conv2d(res11_2, c, [], 2, normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv3')
            # res11_3 = self.ResNet(conv2, scope='res11_3')

            pool4 = slim.max_pool2d(res11_3, ksize=[2, 2], padding='SAME')
            res11_4 = self.ResNet(pool4, scope='res11_4')

            # 上采样
            res11_4 = self.ResNet(res11_4, scope='res11_4')
            interp1_4 = self._PixelShufflerUpsample(res11_4, scope='interp1_4')+res11_3
            up_sample1_3 = interp1_4 + skip_connection_3

            res11_33 = self.ResNet(up_sample1_3, scope='res11_33')
            interp1_3 = self._PixelShufflerUpsample(res11_33, scope='inter1_3') + res11_2
            up_sample1_2 = interp1_3 + skip_connection_2

            res11_22 = self.ResNet(up_sample1_2, scope='res11_22')
            interp1_2 = self._PixelShufflerUpsample(res11_22, scope='inter1_2') + res11_1
            up_sample1_1 = interp1_2 + skip_connection_1

            res11_11 = self.ResNet(up_sample1_1, scope='res11_11')
            return res11_11

    def DownUp_2(self, x, scope='downup'):
        with tf.variable_scope(scope):
            # 下采样
            # c = x.get_shape().as_list()[-1]
            pool1 = slim.max_pool2d(x, ksize=[2, 2], padding='SAME')
            # conv1 = slim.conv2d(x, c, [], 2, normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv1')
            # res11_1 = self.ResNet(conv1, scope='res11_1')
            res11_1 = self.ResNet(pool1, scope='res11_1')
            skip_connection_1 = self.ResNet(res11_1, scope='skip_connection_1')

            pool2 = slim.max_pool2d(res11_1, ksize=[2, 2], padding='SAME')
            res11_2 = self.ResNet(pool2, scope='res11_2')
            skip_connection_2 = self.ResNet(res11_2, scope='skip_connection_2')
            # conv2 = slim.conv2d(res11_1, c, [], 2, normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv2')
            # res11_2 = self.ResNet(conv2, scope='res11_2')

            pool3 = slim.max_pool2d(res11_2, ksize=[2, 2], padding='SAME')
            res11_3 = self.ResNet(pool3, scope='res11_3')
            # conv3 = slim.conv2d(res11_2, c, [], 2, normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv3')
            # res11_3 = self.ResNet(conv2, scope='res11_3')

            # 上采样
            res11_3 = self.ResNet(res11_3, scope='res11_3')
            interp1_3 = self._PixelShufflerUpsample(res11_3, scope='inter1_3') + res11_2
            up_sample1_2 = interp1_3 + skip_connection_2

            res11_22 = self.ResNet(up_sample1_2, scope='res11_22')
            interp1_2 = self._PixelShufflerUpsample(res11_22, scope='inter1_2') + res11_1
            up_sample1_1 = interp1_2 + skip_connection_1

            res11_11 = self.ResNet(up_sample1_1, scope='res11_11')

            return res11_11
    def DownUp_3(self, x, scope='downup'):
        with tf.variable_scope(scope):
            # 下采样
            # c = x.get_shape().as_list()[-1]
            pool1 = slim.max_pool2d(x, ksize=[2, 2], padding='SAME')
            # conv1 = slim.conv2d(x, c, [], 2, normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv1')
            # res11_1 = self.ResNet(conv1, scope='res11_1')
            res11_1 = self.ResNet(pool1, scope='res11_1')
            skip_connection_1 = self.ResNet(res11_1, scope='skip_connection_1')

            pool2 = slim.max_pool2d(res11_1, ksize=[2, 2], padding='SAME')
            res11_2 = self.ResNet(pool2, scope='res11_2')
            # conv2 = slim.conv2d(res11_1, c, [], 2, normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv2')
            # res11_2 = self.ResNet(conv2, scope='res11_2')


            # 上采样
            res11_2 = self.ResNet(res11_2, scope='res11_2')
            interp1_2 = self._PixelShufflerUpsample(res11_2, scope='inter1_2') + res11_1
            up_sample1_1 = interp1_2 + skip_connection_1

            res11_11 = self.ResNet(up_sample1_1, scope='res11_11')
            return res11_11

    def DownUp_4(self, x, scope='downup'):
        with tf.variable_scope(scope):
            # 下采样
            # c = x.get_shape().as_list()[-1]
            pool1 = slim.max_pool2d(x, ksize=[2, 2], padding='SAME')
            # conv1 = slim.conv2d(x, c, [], 2, normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv1')
            # res11_1 = self.ResNet(conv1, scope='res11_1')
            res11_1 = self.ResNet(pool1, scope='res11_1')
            res11_1 = self.ResNet(res11_1, scope='res11_1')
            return res11_1

    def AttenModule_1(self, x, scope='attenmodule'):
        with tf.variable_scope(scope):
            resnet_result = x
            for i in range(self.trunk_number):
                resnet_result = self.ResNet(resnet_result, scope='rb_{}'.format(i))

            b2 = self.DownUp_1(x)
            b2 = self._PixelShufflerUpsample(b2, scope='interp1_1') + resnet_result
            multiply = b2 * resnet_result
            add_result = multiply + resnet_result
            output = self.ResNet(add_result)
            # 一层compression，保证与输入通道数一致，方便后面相加
            # output = slim.conv2d(output, self.num_init_feature, [1, 1], normalizer_fn=None, activation_fn=tf.nn.relu,
            #                 scope='compression')
            return output

    def AttenModule_2(self, x, scope='attenmodule'):
        with tf.variable_scope(scope):
            resnet_result = x
            for i in range(self.trunk_number):
                resnet_result = self.ResNet(resnet_result, scope='rb_{}'.format(i))

            b2 = self.DownUp_2(x)
            b2 = self._PixelShufflerUpsample(b2, scope='interp1_1') + resnet_result
            multiply = b2 * resnet_result
            add_result = multiply + resnet_result
            output = self.ResNet(add_result)
            # 一层compression，保证与输入通道数一致，方便后面相加
            # output = slim.conv2d(output, self.num_init_feature, [1, 1], normalizer_fn=None, activation_fn=tf.nn.relu,
            #                 scope='compression')
            return output

    def AttenModule_3(self, x, scope='attenmodule'):
        with tf.variable_scope(scope):
            resnet_result = x
            for i in range(self.trunk_number):
                resnet_result = self.ResNet(resnet_result, scope='rb_{}'.format(i))

            b2 = self.DownUp_3(x)
            b2 = self._PixelShufflerUpsample(b2, scope='interp1_1') + resnet_result
            multiply = b2 * resnet_result
            add_result = multiply + resnet_result
            output = self.ResNet(add_result)
            # 一层compression，保证与输入通道数一致，方便后面相加
            # output = slim.conv2d(output, self.num_init_feature, [1, 1], normalizer_fn=None, activation_fn=tf.nn.relu,
            #                 scope='compression')
            return output

    def AttenModule_4(self, x, scope='attenmodule'):
        with tf.variable_scope(scope):
            resnet_result = x
            for i in range(self.trunk_number):
                resnet_result = self.ResNet(resnet_result, scope='rb_{}'.format(i))

            b2 = self.DownUp_4(x)
            b2 = self._PixelShufflerUpsample(b2, scope='interp1_1') + resnet_result
            multiply = b2 * resnet_result
            add_result = multiply + resnet_result
            output = self.ResNet(add_result)
            # 一层compression，保证与输入通道数一致，方便后面相加
            # output = slim.conv2d(output, self.num_init_feature, [1, 1], normalizer_fn=None, activation_fn=tf.nn.relu,
            #                 scope='compression')
            return output

    '''
    按尺度缩放, scale的值为缩放倍数
    '''
    def _ScaleImage(self, x, scale):
        # 这里不加as_list()函数，后面的操作无法进行
        [n, h, w, c] = x.get_shape().as_list()
        newh = int(round(h * scale))
        neww = int(round(w * scale))
        newx = tf.image.resize_images(x, [newh, neww], method=0)
        return newx

    # 上采样层之前的卷积
    def _ConvBeforeUp(self, x, scope='convbeforeup'):
        with tf.variable_scope(scope):
            input_feature = x.get_shape()[-1]
            net = x
            if input_feature > self.max_num_feature:
                net = slim.conv2d(net, self.max_num_feature, [1, 1], normalizer_fn=None, activation_fn=tf.nn.relu, scope='max_compression')
                net = slim.conv2d(net, input_feature, [3, 3], normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv3_3')
            return net
    #上采样层
    def _PixelShufflerUpsample(self, x, scale_factor=2, scope='pixelshuttlferupsample'):
        with tf.variable_scope(scope):
            channel = x.get_shape()[-1] * (scale_factor ** 2)
            net = slim.conv2d(x, channel, [1, 1], normalizer_fn=None, activation_fn=tf.nn.relu, scope='modify_channels')
            net = tf.depth_to_space(net, block_size=scale_factor)
            return net

    def Generator(self, x, scope='Generator', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            self.outputs = []
            for i in range(self.max_scale):
                temp = x  # 用来与最后一层相加
                net = slim.conv2d(x, self.num_init_feature, [3, 3], scope='init_conv{}'.format(i),
                                  normalizer_fn=None, activation_fn=tf.nn.relu)  # 预处理之前的feature
                initnet = net
                atten_num = self.each_scale_atten_num[i]
                # 每一个尺度里面的attention
                for j in range(atten_num):
                    pre_res = self.PreResNet(net, self.num_init_feature, scope='preresnet_{}'.format(j))
                    net = self.AttenModule(pre_res, scope='attenrese_{}_{}'.format(i, j))
                # attention之后是一个conv(3,3)
                net = self._ConvBeforeUp(net, scope='convbeforeup_{}'.format(i))
                net = net + initnet  # 上采样层之前的跳跃连接
                net = self._PixelShufflerUpsample(net, scope='pixelshufflerupsample_{}'.format(i))
                # 最后一个尺度重建成一幅比输入图像大两倍的图像
                dimg = slim.conv2d(net, 3, [3, 3], normalizer_fn=None, activation_fn=None, scope='reconstruct_{}'.format(i))
                # dimg是拉普拉斯金字塔，加上x上采样的图像就可以得到原来的图像
                finalimg = dimg + self._ScaleImage(temp, 2)

                self.outputs.append(finalimg)

    def GetOutput(self, x, gtimg):
        self.Generator(x)
        # 计算多尺度损失函数
        img_in = self.outputs
        self.loss_total = 0
        for i in range(self.max_scale):
            [n, h, w, c] = img_in[i].get_shape()
            gt_i = tf.image.resize_images(gtimg, [h, w], method=0)
            loss = tf.reduce_mean((gt_i - img_in[i]) ** 2)
            self.loss_total += loss
            # 记录损失
            tf.summary.scalar('loss_{}'.format(i), loss)
        return self.outputs, self.loss_total


