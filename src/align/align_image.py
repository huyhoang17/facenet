"""Performs face alignment and stores face thumbnails in the output directory.
"""
# from scipy import misc
import sys
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
from PIL import Image


def main(args):
    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=args.gpu_memory_fraction
        )
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False)
        )
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img = np.array(Image.open(args.path_image))  # ndarray

    nrof_successfully_aligned = 0

    if img.ndim == 2:
        img = facenet.to_rgb(img)
    img = img[:, :, 0:3]
    print('=' * 30)
    print(img.ndim, img.shape)

    bounding_boxes, _ = align.detect_face.detect_face(
        img, minsize, pnet, rnet, onet, threshold, factor
    )
    print(bounding_boxes)
    nrof_faces = bounding_boxes.shape[0]
    print(nrof_faces)

    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces > 1:
            if args.detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (
                    det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack(
                    [(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                     (det[:, 1] + det[:, 3]) / 2 - img_center[0]]
                )
                offset_dist_squared = np.sum(
                    np.power(offsets, 2.0), 0)
                # some extra weight on the centering
                index = np.argmax(
                    bounding_box_size - offset_dist_squared * 2.0)
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))

        print("det_arr", det_arr)
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            left = bb[0] = np.maximum(det[0] - args.margin / 2, 0)
            top = bb[1] = np.maximum(det[1] - args.margin / 2, 0)
            right = bb[2] = np.minimum(
                det[2] + args.margin / 2, img_size[1])
            bottom = bb[3] = np.minimum(
                det[3] + args.margin / 2, img_size[0])

            nrof_successfully_aligned += 1

            print(left, top, right, bottom)
            left, top, right, bottom = \
                int(left), int(top), int(right), int(bottom)
            face_image = img[top:bottom, left:right, :]

            pil_image = Image.fromarray(face_image)
            # pil_image.save()
            pil_image.show()
        # main_image = Image.fromarray(img)
        # main_image.show()
        # import time
        # time.sleep(5)
        # main_image.close()
    else:
        print('Unable to align')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path_image', type=str,
        help='Directory with unaligned images.'
    )
    parser.add_argument(
        '--input_dir', type=str,
        help='Directory with unaligned images.'
    )
    parser.add_argument(
        '--output_dir', type=str, default='.',
        help='Directory with aligned face thumbnails.'
    )
    parser.add_argument(
        '--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160
    )
    parser.add_argument(
        '--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=32  # noqa
    )
    parser.add_argument(
        '--random_order',
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true'  # noqa
    )
    parser.add_argument(
        '--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.25  # noqa
    )
    parser.add_argument(
        '--detect_multiple_faces', type=bool,
        help='Detect and align multiple faces per image.', default=False
    )
    parser.add_argument(
        '--use_webcam', type=bool,
        help='Use webcam to detect face realtime', default=False
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
