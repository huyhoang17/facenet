import argparse
import os
import sys

import numpy as np
from scipy import misc
from sklearn.externals import joblib
import tensorflow as tf

import align.detect_face
import facenet
from facenet import ImageClass


def get_label_and_image_class_objects(path_dir, new_label,
                                      last_index_label):
    '''
    Return
    ------
        dataset: list of ImageClass's object::[name, image_paths]
    '''
    # create image folder of new person
    path_exp = os.path.expanduser(path_dir)
    # path_new_label = os.path.join(path_exp, new_label)
    # if not os.path.exists(path_new_label):
    #     pass

    file_names = os.listdir(path_exp)
    file_names.sort()

    image_paths = [os.path.join(path_exp, file_name)
                   for file_name in file_names]

    label = [last_index_label + 1] * len(image_paths)

    # label, list of image paths
    return ImageClass(new_label, image_paths), label


def load_and_align_data(image_paths, image_size, margin,
                        gpu_memory_fraction):

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in range(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(
            img, minsize, pnet, rnet, onet, threshold, factor
        )
        if len(bounding_boxes) == 0:
            print("Can not align face in image: \n\t{}".format(image_paths[i]))
            continue

        det = np.squeeze(bounding_boxes[0, 0:4])

        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(
            cropped, (image_size, image_size), interp='bilinear'
        )
        # uncomment to save align image
        # misc.imsave(image_paths[i], aligned)
        # path, name = image_paths[i].rsplit("/")
        # suffix = name.rsplit(".")[-1]
        # file_name = str(i) + suffix
        # aligned.save(os.path.join(path, file_name))

        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack([i for i in img_list if i is not None])
    return images


def main(args):
    labels, class_names, emb_arrays = joblib.load(args.model_filename)
    print(len(set(labels)), len(class_names), len(emb_arrays))

    # Generate embedding's image
    emb_array = None

    image_obj, label = get_label_and_image_class_objects(
        args.input_dir, args.new_label, labels[-1]
    )
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=args.gpu_memory_fraction
        )
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False)
        )
        with tf.Session() as sess:
            images = load_and_align_data(
                image_obj.image_paths, args.image_size,
                args.margin, args.gpu_memory_fraction
            )

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = \
                tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = \
                tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = \
                tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            for image in images:
                image = image.reshape(-1, args.image_size, args.image_size, 3)
                feed_dict = {
                    images_placeholder: image,
                    phase_train_placeholder: False
                }

                emb_array = sess.run(
                    embeddings, feed_dict=feed_dict
                )
                print(">>> emb", emb_array.shape)
                print(">>> emb", emb_array)
                emb_arrays = np.concatenate((emb_arrays, emb_array), axis=0)

    # =========================================================================
    # Training model
    label = [labels[-1] + 1] * len(images)
    labels = labels + label
    class_names.append(args.new_label)
    # emb_arrays = np.concatenate((emb_arrays, emb_array), axis=0)
    assert (len(labels) == len(emb_arrays))
    assert (len(set(labels)) == len(class_names))
    print(len(set(labels)), len(class_names), len(emb_arrays))
    joblib.dump(
        (labels, class_names, emb_arrays),
        args.model_filename
    )

    # SVM
    from sklearn.svm import SVC  # noqa
    print('>>> Training SVM classifier')
    model = SVC(kernel='linear', probability=True)
    model.fit(emb_arrays, labels)

    joblib.dump(model, "models/backup/svm_dump.pkl")
    print('Saved svm classifier model')

    # KNN
    from sklearn.neighbors import KNeighborsClassifier  # noqa
    print('>>> Training KNN model')
    model = KNeighborsClassifier()
    model.fit(emb_arrays, labels)

    joblib.dump(model, args.classifier_filename)
    print('Saved knn model to file "%s"' %
          args.classifier_filename)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_dir', type=str,
        help='Directory with unaligned images.'
    )
    parser.add_argument(
        '--new_label', type=str,
        help='Name person'
    )
    parser.add_argument(
        '--model', type=str,
        help='Link to model trained'
    )
    parser.add_argument(
        '--mode', type=str, choices=['SVM', 'KNN'],
        help='Choose classifier', default='KNN'
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
