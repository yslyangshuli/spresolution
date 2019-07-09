'''
主函数
注意：图像在训练的时候，一定不能放大，只能缩小
测试的时候，可以设置一个大小比实际大小更大的shape，然后补0，复原之后crop这个图像就行了
'''
from mutilscale.prodeblur import DenseDeblur

if __name__ == '__main__':
    # gan = MultiScaleGAN(720, 3, 720, 720)
    # train = True
    # if train:
    #     gan.BuildModel()
    #     gan.Train()
    # else:
    #     gan.Test()
    datalistpath = 'F:/data/blurred_sharp/datalist.txt'
    trainpath = 'F:/data/blurred_sharp/train/'
    trainout = 'D:/exprimental/logs/out/'
    logout = 'D:/exprimental/logs/log/'
    checkpoint = './snapshot/'
    denseblur = DenseDeblur(datalistpath=datalistpath, trainpath=trainpath, trainout=trainout, logout=logout,
                            checkpoint=checkpoint)
    denseblur.Train()

