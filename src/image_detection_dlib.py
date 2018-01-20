"""Performs face alignment and stores face thumbnails in the output directory.
"""
import os
import sys
import argparse
import tensorflow as tf

import cv2
import dlib
from scipy import misc
from sklearn.externals import joblib
from PIL import Image

from decorators import timer_format
import facenet

CONST_DIST = 0.90
# CONST_DIST = 0.65


def css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), \
        min(css[2], image_shape[0]), max(css[3], 0)


@timer_format()
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
            facenet.load_model(args.model_trained)

            # Get input and output tensors
            images_placeholder = \
                tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = \
                tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = \
                tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # Embedding size: (None, 128)

            _, class_names, _ = joblib.load(
                args.model_filename
            )
            # Classify images
            model = joblib.load(args.classifier_filename)

    # Load all the models we need: a detector to find the faces
    # a shape predictor to find face landmarks so we can precisely localize the face  # noqa
    detector = dlib.get_frontal_face_detector()

    while True:
        path_image = input(">>> Path to image: ")
        if path_image == "q":
            break
        if not os.path.isfile(path_image):
            print("Path file incorrect!")
            continue
        # img = misc.imread(args.path_image)
        img = misc.imread(path_image)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        detected_faces = detector(img, 1)
        if len(detected_faces) == 0:
            print("No faces found in '{}'".format(args.path_image))
            # exit()

        for i, face_rect in enumerate(detected_faces, start=1):
            left, top, right, bottom = \
                int(face_rect.left()), int(face_rect.top()), \
                int(face_rect.right()), int(face_rect.bottom())

            img_croped = misc.imresize(
                img[top:bottom, left:right, :],
                (args.image_size, args.image_size),
                interp='bilinear'
            )

            face_image_4d = facenet.load_test_web_data(
                img_croped,
                False, False, args.image_size
            )

            # RUN
            feed_dict = {
                images_placeholder: face_image_4d,  # 4d ndarray
                phase_train_placeholder: False
            }
            emb_array = sess.run(
                embeddings, feed_dict=feed_dict
            )
            # Draw a box around the face
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left, bottom - 35),
                          (right, bottom), (0, 0, 255))
            # print("\t>>> Embed shape: ", emb_array.shape)
            distances, indexes = model.kneighbors(
                emb_array.reshape(1, -1), return_distance=True
            )

            # PREDICTION
            predictions = model.predict(emb_array)
            checked = any(d <= CONST_DIST for d in distances[0])
            font = cv2.FONT_HERSHEY_DUPLEX
            # print("\t>>> Distance: ", distances[0])
            if checked:
                cv2.putText(
                    img,
                    "{} ".format(i) + class_names[predictions[0]],
                    (left + 6, bottom - 6), font,
                    0.5, (255, 255, 255), 1
                )
            else:
                cv2.putText(
                    img, "{} ".format(i) + "---",
                    (left + 6, bottom - 6), font,
                    0.5, (255, 255, 255), 1
                )
        main_image = Image.fromarray(img)
        main_image.show()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_trained', type=str,
        help='Link to model trained'
    )
    parser.add_argument(
        '--path_image', type=str, default='',
        help='Unaligned images.'
    )
    parser.add_argument(
        '--path_dir', type=str, default='',
        help='Directory with unaligned images.'
    )
    parser.add_argument(
        '--classifier_filename', type=str,
        help='Link to classifier filename trained'
    )
    parser.add_argument(
        '--model_filename', type=str,
        help='Link to model filename trained'
    )
    parser.add_argument(
        '--predictor_path', type=str,
        help='Link to dlib model'
    )
    parser.add_argument(
        '--output_dir', type=str,
        help='Link to directory to save image detection'
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
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
