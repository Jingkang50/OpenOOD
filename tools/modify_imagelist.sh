#!/bin/bash
# sh tools/modify_imagelist.sh

PYTHONPATH='.':$PYTHONPATH \
python tools/modify_imagelist.py \
--file_pth 'test_tin_ood.txt' \
--old_str 'images/' \
--new_str '' \
--change_byline
