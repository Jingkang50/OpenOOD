import os.path

src_path = './data/imglist/objects/train_cifar10.txt'
save_path = './data/imglist/objects'
normal_class = 3
save_name = os.path.join(save_path, 'train_cifar10_ood' + '.txt')
with open(src_path, 'r') as imgfile:
    imglist = imgfile.readlines()
for index in range(len(imglist)):
    line = imglist[index].strip('\n')
    tokens = line.split(' ', 1)
    image_name, extra_str = tokens[0], tokens[1]
    target = int(extra_str)
    if target != normal_class:
        with open(save_name, 'a') as f:
            f.write(imglist[index])
            f.close()
    else:
        continue
