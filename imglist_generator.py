import os
'''
path="./data/images_classic/cinic/valid"
save_path="./data/benchmark_imglist/cifar10/val_cinic10.txt"
prefix="cinic/valid/"
category=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
with open(save_path,'a') as f:
    for name in category:
        label=category.index(name)
        sub_path=path+'/'+name
        files=os.listdir(sub_path)
        for file in files:
            line=prefix+name+'/'+file+' '+str(label)+'\n'
            f.write(line)
    f.close()
'''

path = './data/images_classic/cifar100c'
save_path = './data/benchmark_imglist/cifar100/test_cifar100c.txt'
prefix = 'cifar100c/'
files = os.listdir(path)
with open(save_path, 'a') as f:
    for file in files:
        splits = file.split('_')
        label = (splits[1].split('.'))[0]
        line = prefix + file + ' ' + label + '\n'
        f.write(line)
    f.close()
'''
path="./data/images_largescale/imagenet_v2"
save_path="./data/benchmark_imglist/imagenet/test_imagenetv2.txt"
prefix="imagenet_v2/"
with open(save_path,'a') as f:
    for i in range(0,1000):
        label=str(i)
        sub_path=path+'/'+label
        files=os.listdir(sub_path)
        for file in files:
            line=prefix+label+'/'+file+' '+label+'\n'
            f.write(line)
    f.close()
'''
