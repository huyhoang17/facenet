import argparse
import os
import sys


def rename_image_folder(args):
    '''
    Rename images in specific folder to correct form

        Ex: Phan_Hoang_0001.png
            Phan_Hoang_0002.png
            Phan_Hoang_0003.png
            ...
    '''
    path_exp = os.path.expanduser(args.path_dir)

    file_names = os.listdir(path_exp)

    for i, file_name in enumerate(file_names, start=1):
        _, suffix = file_name.rsplit(".")
        new_name = args.label.upper() + "_" + \
            str(i).rjust(int(args.n), args.default_char) + "." + suffix
        os.rename(
            os.path.join(path_exp, file_name),
            os.path.join(path_exp, new_name)
        )


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path_dir', type=str,
    )
    parser.add_argument(
        '--label', type=str,
    )
    parser.add_argument(
        '--n', type=str, default="4",
    )
    parser.add_argument(
        '--default_char', type=str, default="0"
    )

    return parser.parse_args(argv)


if __name__ == '__main__':
    rename_image_folder(parse_arguments(sys.argv[1:]))

'''
COMMAND
-------

python3 util/naming_image_folder.py \
--path_dir /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan \
--label Manh_Tuan
'''
