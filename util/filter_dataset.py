import os
import shutil
import random


# PATH_DIR = "/home/phanhoang/workspace/projects/fork/facenet/datasets/lfw_160"
PATH_DIR_RAW = "/home/phanhoang/workspace/dataset/face_dataset_raw"
PATH_DIR_ALIGNED = "/home/phanhoang/workspace/dataset/face_dataset_aligned_2"


MIN_IMAGES = 3
MAX_IMAGES = 20


def filter_images():
    total = 0
    total_image = 0  # images/label
    total_dir = 0  # labels
    if os.path.exists(PATH_DIR_ALIGNED):
        # dir_path, dir_name, filenames
        for root, dirs, files in os.walk(PATH_DIR_ALIGNED):
            total += 1
            if MAX_IMAGES >= len(files) >= MIN_IMAGES:
                total_image += len(files)
                total_dir += 1
            elif len(files) > MAX_IMAGES:
                n_image_removed = len(files) - MAX_IMAGES
                # root::path image directories
                images = [os.path.join(root, image)
                          for image in os.listdir(root)]
                path_image_removed = random.sample(images, n_image_removed)
                for path_image in path_image_removed:
                    os.remove(path_image)
                    print("Removed %s" % path_image)
                del images, path_image_removed
            else:
                print("pass")
    else:
        # FileNotFoundError
        print("Directory does not existed")

    # print("Total directory removed: %s" % total - total_dir)
    # print("Total labels: %s" % total_dir)
    # print("Total images: %s" % total_image)
    # print("Images / label: %s" % total_image / total_dir)


if __name__ == "__main__":
    filter_images()
