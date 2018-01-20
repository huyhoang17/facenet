"""Performs face alignment and stores face thumbnails in the output directory.
"""
import argparse
from scipy import misc
import sys

import cv2
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib

import align.detect_face
import facenet


CONST_DIST = 1.2
FRAME_INTERVAL = 3


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
            facenet.load_model(args.model_trained)

            # Get input and output tensors
            images_placeholder = \
                tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = \
                tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = \
                tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # (?, 128)
            print(">>> Embedding size: ", embeddings.get_shape())

            labels, class_names, embed_arrays = joblib.load(
                args.model_filename
            )
            # Classify images
            model = joblib.load(args.classifier_filename)

            # =================================================================
            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor

            # Get a reference to webcam #0 (the default one)
            video_capture = cv2.VideoCapture(0)
            while True:
                ret, img = video_capture.read()
                nrof_successfully_aligned = 0

                if img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = img[:, :, 0:3]
                # print("Ndim: %d" % img.ndim)
                # print("Shape: %d" % img.shape)

                bounding_boxes, _ = align.detect_face.detect_face(
                    img, minsize, pnet, rnet, onet, threshold, factor
                )
                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    det_arr = []
                    img_size = np.asarray(img.shape)[0:2]
                    if nrof_faces > 1:
                        if args.detect_multiple_faces:
                            for i in range(nrof_faces):
                                det_arr.append(np.squeeze(det[i]))
                        else:
                            bounding_box_size = (det[:, 2] - det[:, 0]) * \
                                                (det[:, 3] - det[:, 1])
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

                    for i, det in enumerate(det_arr):
                        det = np.squeeze(det)
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0] - args.margin / 2, 0)
                        bb[1] = np.maximum(det[1] - args.margin / 2, 0)
                        bb[2] = np.minimum(
                            det[2] + args.margin / 2, img_size[1])
                        bb[3] = np.minimum(
                            det[3] + args.margin / 2, img_size[0])

                        # CROP IMAGE
                        print(">>> Resize image")
                        img_croped = misc.imresize(
                            img[bb[1]:bb[3], bb[0]:bb[2], :],
                            (args.image_size, args.image_size),
                            interp='bilinear'
                        )
                        img_croped = facenet.load_test_web_data(
                            img_croped,
                            False, False, args.image_size
                        )
                        # RUN
                        print(">>> Feed dict")
                        feed_dict = {
                            images_placeholder: img_croped,  # ndarray
                            phase_train_placeholder: False
                        }
                        emb_array = sess.run(
                            embeddings, feed_dict=feed_dict
                        )
                        # print(emb_array)

                        nrof_successfully_aligned += 1

                        # left, top, right, bottom
                        print(bb[0], bb[1], bb[2], bb[3])
                        cv2.rectangle(img, (bb[0], bb[1]),
                                      (bb[2], bb[3]), (0, 0, 255), 2)
                        # Draw a label with a name below the face
                        cv2.rectangle(
                            img, (bb[0], bb[3] - 35), (bb[2], bb[1]),
                            (0, 0, 255)
                        )

                        print("\t>>> Embed shape: ", emb_array.shape)
                        distances, indexes = model.kneighbors(
                            emb_array.reshape(1, -1), return_distance=True
                        )
                        print(emb_array.tolist())

                        # PREDICTION
                        predictions = model.predict(emb_array)
                        print("\t>>> Index (non threshold): ",
                              class_names[predictions[0]])
                        print("\t>>> Predictions (non threshold): ",
                              predictions[0]
                              )

                        checked = any(d < CONST_DIST for d in distances[0])
                        print("\t>>> Distance: ", distances[0])
                        font = cv2.FONT_HERSHEY_DUPLEX
                        if checked:
                            # max_dist = np.argmin(distances[0])
                            # class_name = \
                            #     class_names[labels[indexes[0][max_dist]]]
                            # cv2.putText(
                            #     img, "{} ".format(i) + class_name,
                            #     (bb[0] + 6, bb[3] - 6), font,
                            #     1.0, (255, 255, 255), 1
                            # )
                            cv2.putText(
                                img,
                                "{} ".format(i) + class_names[predictions[0]],
                                (bb[0] + 6, bb[3] - 6), font,
                                1.0, (255, 255, 255), 1
                            )
                            print("\t>>> Label: %s" %
                                  class_names[predictions[0]])
                        else:
                            cv2.putText(
                                img, "{} ".format(i) + "---",
                                (bb[0] + 6, bb[3] - 6), font,
                                1.0, (255, 255, 255), 1
                            )
                            print("\t>>> Label: %s" % "Unknown")

                        # emb_array = np.array(emb_array).reshape(1, -1)
                        # print(">>> Prediction")

                        # predictions = model.predict(emb_array)
                        # print("Predictions: ", predictions[0])
                        # font = cv2.FONT_HERSHEY_DUPLEX
                        # cv2.putText(
                        #     img, class_names[predictions[0]],
                        #     (bb[0] + 6, bb[3] - 6), font,
                        #     1.0, (255, 255, 255), 1
                        # )
                        # print("Label: %s" % class_names[predictions[0]])

                        del img_croped, emb_array
                    del det, det_arr
                else:
                    print('Unable to align')

                cv2.imshow('Video', img)
                print("Show frame")

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

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
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.5  # noqa
    )
    parser.add_argument(
        '--detect_multiple_faces', type=bool,
        help='Detect and align multiple faces per image.', default=False
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
