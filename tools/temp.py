import os

path = 'data/imglist/mvtecList/'
names = os.listdir(path)


def read_content(file_pth):
    with open(file_pth) as file:
        lines = file.readlines()
    return lines


for name in names:
    if name.endswith('.txt'):
        lines = read_content(path + name)
        changed_file = open(path + name, 'w')
        os.remove(name)
        for line in lines:
            changed_file.write(line.replace('./mvtec/', 'mvtec/'))
