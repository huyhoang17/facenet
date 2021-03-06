"""Performs face alignment and stores face thumbnails in the output directory.
"""
# from scipy import misc
import sys
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import cv2


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

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, img = video_capture.read()
        # ndarray
        # img = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        nrof_successfully_aligned = 0

        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:, :, 0:3]
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

            print(">>> det_arr", det_arr)
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - args.margin / 2, 0)
                bb[1] = np.maximum(det[1] - args.margin / 2, 0)
                bb[2] = np.minimum(
                    det[2] + args.margin / 2, img_size[1])
                bb[3] = np.minimum(
                    det[3] + args.margin / 2, img_size[0])

                nrof_successfully_aligned += 1

                # left, top, right, bottom
                print("left, top, right, bottom")
                print(bb[0], bb[1], bb[2], bb[3])
                cv2.rectangle(img, (bb[0], bb[1]),
                              (bb[2], bb[3]), (0, 0, 255), 2)
                # Draw a label with a name below the face
                cv2.rectangle(
                    img, (bb[0], bb[3] - 35), (bb[2], bb[1]),
                    (0, 0, 255)
                )
                # font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(
                #     img, name, (bb[0] + 6, bb[3] - 6), font,
                #     1.0, (255, 255, 255), 1
                # )
        else:
            print('Unable to align')

        cv2.imshow('Video', img)
        print("=" * 30)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160
    )
    parser.add_argument(
        '--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=32  # noqa
    )
    parser.add_argument(
        '--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.25  # noqa
    )
    parser.add_argument(
        '--detect_multiple_faces', type=bool,
        help='Detect and align multiple faces per image.', default=False
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
