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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_pth', help='string to be replaced')
    parser.add_argument('--old_str', help='string to be replaced')
    parser.add_argument('--new_str', help='new string')
    parser.add_argument('--change_byline',
                        action='store_true',
                        help='modify contents line by line')
    args = parser.parse_args()

    if args.change_byline:
        change_content_byline(args.file_pth, args.old_str, args.new_str)
