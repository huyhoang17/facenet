"""Performs face alignment and stores face thumbnails in the output directory.
"""
import argparse
import sys

import cv2
import dlib
import tensorflow as tf
from scipy import misc
from sklearn.externals import joblib

from decorators import timer_format
import facenet


CONST_DIST = 1.2
FRAME_INTERVAL = 3
N_CHANNELS = 3


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
            # (None, 128)

            labels, class_names, embed_arrays = joblib.load(
                args.model_filename
            )
            # Classify images
            model = joblib.load(args.classifier_filename)

    detector = dlib.get_frontal_face_detector()
    # https://gist.github.com/ageitgey/ae340db3e493530d5e1f9c15292e5c74
    # face_pose_predictor = dlib.shape_predictor(args.predictor_path)
    # sp = dlib.shape_predictor(args.predictor_path)

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    c = 0
    while True:
        ret, frame = video_capture.read()

        if c % FRAME_INTERVAL == 0:
            img = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            if img.ndim == 2:
                img = facenet.to_rgb(img)
            img = img[:, :, 0:3]

            detected_faces = detector(img, 0)
            if len(detected_faces) == 0:
                print("No faces found")
                # cv2.imshow('Video', frame)
                # continue
            else:
                for i, face_rect in enumerate(detected_faces, start=1):
                    left, top, right, bottom = \
                        int(face_rect.left()) * 4, int(face_rect.top()) * 4, \
                        int(face_rect.right()) * 4, int(face_rect.bottom()) * 4

                    img_croped = misc.imresize(
                        frame[top:bottom, left:right, :],
                        (args.image_size, args.image_size),
                        interp='bilinear'
                    )
                    face_image_4d = facenet.load_test_web_data(
                        img_croped,
                        False, False, args.image_size
                    )
                    feed_dict = {
                        images_placeholder: face_image_4d,  # ndarray
                        phase_train_placeholder: False
                    }
                    emb_array = sess.run(
                        embeddings, feed_dict=feed_dict
                    )

                    # Draw a box around the face
                    cv2.rectangle(
                        frame,
                        (left, top),
                        (right, bottom),
                        (0, 0, 255), 2
                    )

                    # Draw a label with a name below the face
                    cv2.rectangle(
                        frame,
                        (left, bottom - 35),
                        (right, bottom),
                        (0, 0, 255), cv2.FILLED
                    )

                    # print("\t>>> Embed shape: ", emb_array.shape)
                    distances, indexes = model.kneighbors(
                        emb_array.reshape(1, -1), return_distance=True
                    )

                    # PREDICTION
                    predictions = model.predict(emb_array)
                    print("\t>>> Index (non threshold): ",
                          class_names[predictions[0]]
                          )
                    print("\t>>> Predictions (nonthreshold): ", predictions[0])

                    # https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py
                    checked = any(d < CONST_DIST for d in distances[0])
                    print("\t>>> Distance: ", distances[0])
                    font = cv2.FONT_HERSHEY_DUPLEX
                    if checked:
                        cv2.putText(
                            frame,
                            "{} ".format(i) + class_names[predictions[0]],
                            (left + 6, bottom - 6), font,
                            1.0, (255, 255, 255), 1
                        )
                        print("\t>>> Label: %s" % class_names[predictions[0]])
                    else:
                        cv2.putText(
                            frame, "{} ".format(i) + "---",
                            (left + 6, bottom - 6), font,
                            1.0, (255, 255, 255), 1
                        )
                        print("\t>>> Label: %s" % "Unknown")
                del face_image_4d, emb_array, detected_faces

            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        c += 1

    video_capture.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_trained', type=str,
        help='Link to model trained'
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
