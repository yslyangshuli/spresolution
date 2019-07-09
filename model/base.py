'''
基类，训练模型的基类
'''
import time
from util.dataset import InputProducer
from util.util import *
import os

class BaseTrain(object):

    def __init__(self, datalistpath, trainpath, checkpoint, logout):
        self.datalistpath = datalistpath
        self.trainpath = trainpath
        self.checkpoint = checkpoint
        self.logout = logout
        self.batch_size = 8


    '''
    左边是生成器的输出，右边是计算得到的损失函数
    '''
    def GetOutput(self, x, gtimg):
        loss = 0
        return x, loss


    def BuildModel(self, datalistpath, trainpath):
        input = InputProducer(datalistpath, trainpath, 2000, batch_size=self.batch_size)
        self.max_steps = input.GetMaxStep()
        print('一共需要迭代{}次'.format(self.max_steps))
        blurimg, gtimg = input.GetDataProducer()
        # Generator(blurimg)
        outputs, loss = self.GetOutput(blurimg, gtimg)
        self.outputs = outputs
        self.loss_total = loss
        tf.summary.scalar('loss_total', self.loss_total)
        self.vars = tf.trainable_variables()

    def SaveInformation(self, duration, step, sess, saver, sum_writer, merge_all):
        pass

    def Train(self):
        global_step = tf.Variable(dtype=tf.int32, initial_value=0, trainable=False)
        self.BuildModel(self.datalistpath, self.trainpath)
        #衰减学习率
        self.lr = tf.train.polynomial_decay(1e-4, global_step, self.max_steps, end_learning_rate=0.0,
                                            power=0.3)
        tf.summary.scalar('learning_rate', self.lr)
        trainop = tf.train.AdamOptimizer(self.lr).minimize(self.loss_total, var_list=self.vars, global_step=global_step)
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = sess
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables())
        restore = tf.train.Saver(var_list=tf.global_variables())
        #开启数据队列
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # training summary
        sum_all = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.logout, sess.graph, flush_secs=30)
        cpt = tf.train.latest_checkpoint(self.checkpoint)
        # 恢复模型
        if cpt is not None:
            restore.restore(sess, cpt)
        for step in range(sess.run(global_step), self.max_steps):
            start_time = time.time()
            #开始训练
            sess.run(trainop)
            duration = time.time() - start_time
            self.SaveInformation(duration, step, sess, self.saver, summary_writer, sum_all)
