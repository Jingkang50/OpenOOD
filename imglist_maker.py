import os
import os.path

src_path = './data/images/Imagenet_resize/Imagenet_resize'
save_path = './data/imglist/objects'
save_name = os.path.join(save_path, 'test_Imagenet_resize' + '.txt')
file_list = os.listdir(src_path)
files_name = []
for i in file_list:
    files_name.append('Imagenet_resize/Imagenet_resize/{} -1\n'.format(i))
for j in files_name:
    with open(save_name, 'a') as f:
        f.write(j)
f.close()
