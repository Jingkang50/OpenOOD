import argparse
import os


def read_content(file_pth):
    with open(file_pth) as file:
        lines = file.readlines()
    return lines


def change_content_byline(file_pth, old_str, new_str):
    imglist_root = './data/imglist'
    file_pth = os.path.join(imglist_root, file_pth)
    contents = read_content(file_pth)
    changed_file = open(file_pth, 'w')

    for line in contents:
        changed_file.write(line.replace(old_str, new_str))


def split_imglist(pth):
    file_names = os.listdir(pth)
    for file_name in file_names:
        if file_name.endswith('.txt'):
            file = read_content(file_name)
            new_file = open(file_name[:-4] + '_ood.txt', 'w')
            for line in file:
                if '/good/' not in line:
                    new_file.write(line.replace(' 0', ' 1'))
            os.remove(file_name)


def change_label(file_name='./test_bottle_ood.txt',
                 new_file_name='./new_test_bottle_ood.txt'):

    file = read_content(file_name)
    new_file = open(new_file_name, 'w')
    for line in file:
        new_file.write(line[:-2] + '1\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_pth', help='file path')
    parser.add_argument('--old_str', help='string to be replaced')
    parser.add_argument('--new_str', help='new string')
    parser.add_argument('--change_byline',
                        action='store_true',
                        help='modify contents line by line')
    args = parser.parse_args()

    if args.change_byline:
        change_content_byline(args.file_pth, args.old_str, args.new_str)
