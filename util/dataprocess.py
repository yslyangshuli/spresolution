'''
数据处理,生成文件名列表，以供tensorflow的队列机制使用
'''
import os
import shutil


blurpath = 'F:/data/blurred_sharp/blurred'
gtpath = 'F:/data/blurred_sharp/sharp'

file = 'F:/data/blurred_sharp/datalist.txt'
outpath = 'F:/data/blurred_sharp/train'

blurlist = os.listdir(blurpath)
gtlist = os.listdir(gtpath)
print(blurlist)
print(gtlist)
if not os.path.exists(outpath):
    os.mkdir(outpath)

with open(file, 'w') as f:
    for i in range(len(blurlist)):
        b = blurpath + '/' + blurlist[i]
        g = gtpath + '/' + gtlist[i]
        o1 = '{}/blur_{}.png'.format(outpath, i)
        o2 = '{}/gt_{}.png'.format(outpath, i)
        shutil.copy(b, o1)
        shutil.copy(g, o2)
        f.write('gt_{}.png'.format(i))
        f.write(' ')
        f.write('blur_{}.png'.format(i))
        f.write('\n')




