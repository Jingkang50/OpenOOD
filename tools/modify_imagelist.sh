#!/bin/bash
# sh tools/modify_imagelist.sh

PYTHONPATH='.':$PYTHONPATH \
python tools/modify_imagelist.py \
--file_pth 'digits/test_texture.txt' \
--old_str ' -1' \
--new_str ' 0' \
--change_byline
