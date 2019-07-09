import tensorflow as tf
import tensorflow.contrib.slim as slim
import os


class Attention(BaseTrain):

    def __init__(self,
                 datalistpath='./',
                 trainpath='./',
                 logout='./',
                 trainout='./',
                 checkpoint='./'):
        super(Attention, self).__init__(datalistpath, trainpath, checkpoint, logout)
        self.max_scale = 3                      #
        self.each_scale_atten_num = [3, 6, 9]   # 最低的尺度应该层数越小，因为模糊核的尺寸最小
        self.trunk_number = 6                   #
        self.num_mask_layers = 5                #
        self.trainout = trainout
        self.logout = logout
        self.checkpoint = checkpoint
        if not os.path.exists(trainout):
            os.makedirs(trainout)
        self.batch_size = 1
        # 预处理

    # 残差网络
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

    def DownUp(self, x, scope='downup'):
        with tf.variable_scope(scope):
            return x

    # 56层残差的深度卷积
    def AttenModule(self, x, scope='AttenModule'):
        with tf.variable_scope(scope):
            resnet_result = x
            for i in range(self.trunk_number):
                resnet_result = self.ResNet(resnet_result, scope='rb_{}'.format(i))
            # 第二个分支
            b2 = self.DownUp(x)
            multiply = b2 * resnet_result
            add_result = multiply + resnet_result
            output = self.ResNet(add_result)
            return output

    # bottom-up top-down权重选择,先下采样，再上采样
    def max_pool(self, x, scope='max_pool'):
        with tf.variable_scope(scope):
            return slim.max_pool2d(x, ksize=[2, 2], padding='SAME')

    def AttenMask(self, x, dim, num_mask_layers, scope='attenmask'):
        # bottom-up top-down
        with tf.variable_scope(scope):
            net = x
            # for i in range(num_mask_layers):
            #     net = self.Mask(net, 'mask_{}'.format(i))
            mconv1 = slim.conv2d(net, dim, [1, 1], stride=1, scope='mconv1')
            mconv2 = slim.conv2d(mconv1, dim, [1, 1], stride=1, scope='mconv2', normalizer_fn=True)

            return mconv2

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
            dimg = x
            dimg = self._ScaleImage(dimg, self.scale ** self.max_scale)#self.scale ** self.max_scale == 1/8
            for i in range(self.max_scale):
                net = x
                net = self._ScaleImage(net, self.scale ** (self.max_scale - i))
                temp = net#用来与最后一层相加
                #stop gradient，不让第二个尺度的更新影响第一个尺度的更新
                dimg = tf.stop_gradient(dimg)
                net = tf.concat([net, dimg], axis=3)
                net = slim.conv2d(net, self.num_init_feature, [7, 7], scope='init_conv{}'.format(i),
                                  normalizer_fn=None, activation_fn=tf.nn.relu)
                initnet = net #DCU之前的feature
                dcu_num = self.each_scale_atten_num[i]
                # 每一个尺度里面的DCU
                for j in range(dcu_num):
                    net = self.AttenModule(net, scope='denserese_{}_{}'.format(i, j))
                #dcu之后是一个conv(3,3)
                net = self._ConvBeforeUp(net, scope='convbeforeup_{}'.format(i))
                net = net + initnet #上采样层之前的跳跃连接
                net = self._PixelShufflerUpsample(net, scope='pixelshufflerupsample_{}'.format(i))
                #最后一个尺度重建成一幅比输入图像大两倍的图像
                dimg = slim.conv2d(net, 3, [3, 3], normalizer_fn=None, activation_fn=None, scope='reconstruct_{}'.format(i))
                #dimg是拉普拉斯金字塔，加上x上采样的图像就可以得到原来的图像
                finalimg = dimg + self._ScaleImage(temp, 2)
                self.outputs.append(finalimg)

