"""An example of how to use your own dataset to train a classifier that
 recognizes people.
"""
import argparse
import os
import sys
import math
import pickle

import tensorflow as tf
import numpy as np
from sklearn.externals import joblib

from decorators import timer_format
import facenet


@timer_format()
def main(args):

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=args.gpu_memory_fraction
        )
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False)
        )
        with tf.Session() as sess:

            np.random.seed(seed=args.seed)

            # if args.use_split_dataset:
            #     dataset_tmp = facenet.get_dataset(args.data_dir)
            #     train_set, test_set = split_dataset(
            #         dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
            #     if (args.mode == 'TRAIN'):
            #         dataset = train_set
            #     elif (args.mode == 'CLASSIFY'):
            #         dataset = test_set
            # else:
            # [name, image_paths]

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = \
                tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = \
                tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = \
                tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # (?, 128)
            print(">>> Embedding size: ", embeddings.get_shape())
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            dataset = facenet.get_dataset(args.data_dir)
            classifier_filename_path = os.path.expanduser(
                args.classifier_filename
            )
            # Create a list of class names
            class_names = [cls.name.replace('_', ' ') for cls in dataset]

            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths) > 0,
                       'There must be at least one image for each class in the dataset'  # noqa
                       )

            # [image_paths_flat, labels_flat]
            paths, labels = facenet.get_image_paths_and_labels(dataset)

            with open('models/path_images.txt', 'w') as f:
                for path in paths:
                    f.write(path + "\n")

            print('Number of classes: %d' % len(dataset))
            del dataset
            print('Number of images: %d' % len(paths))

            print('Calculating features for all images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(
                math.ceil(1.0 * nrof_images / args.batch_size)
            )
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(
                    paths_batch, False, False, args.image_size
                )
                feed_dict = {
                    images_placeholder: images,
                    phase_train_placeholder: False
                }
                emb_array[start_index:end_index, :] = sess.run(
                    embeddings, feed_dict=feed_dict
                )

            print('Total embed array: ', len(emb_array))
            print('Embed array shape: ', emb_array.shape)
            print('Total images: ', len(paths))
            joblib.dump(
                (labels, class_names, emb_array), 'models/all_model.pkl'
            )

            print('Total embed array: ', len(emb_array))

            # override
            classifier_filename_path = "models/all_svm_dump.pkl"
            # if args.mode == 'SVM':
            from sklearn.svm import SVC  # noqa
            print('Training SVM classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)

            joblib.dump(model, classifier_filename_path)
            print('Saved svm classifier model to file "%s"' %
                  classifier_filename_path)
            # override
            classifier_filename_path = "models/all_knn_dump.pkl"
            # elif args.model == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier  # noqa
            print('Training KNN model')
            model = KNeighborsClassifier()
            model.fit(emb_array, labels)

            joblib.dump(model, classifier_filename_path)
            print('Saved knn model to file "%s"' %
                  classifier_filename_path)

            if args.mode == 'CLASSIFY':
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_path, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' %
                      classifier_filename_path)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(
                    len(best_class_indices)), best_class_indices]

                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (
                        i, class_names[best_class_indices[i]],
                        best_class_probabilities[i])
                    )

                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)


def split_dataset(dataset, min_nrof_images_per_class,
                  nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(
                cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(
                cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     'mode_image', type=str, choices=['ALL', 'ONE'],
    #     help='Embedding evaluate image or all images in dataset', default='ONE'  # noqa
    # )
    parser.add_argument(
        'mode', type=str, choices=['SVM', 'KNN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification model should be used for classification', default='CLASSIFY'  # noqa
    )
    parser.add_argument(
        'data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.'
    )
    parser.add_argument(
        'model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file'  # noqa  # noqa
    )
    parser.add_argument(
        '--classifier_filename',
        help='Classifier model file name as a pickle (.pkl) file. For training this is the output and for classification this is an input.'  # noqa
    )
    parser.add_argument(
        '--use_split_dataset',
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true'  # noqa
    )
    parser.add_argument(
        '--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.'  # noqa
    )
    parser.add_argument(
        '--batch_size', type=int,
        help='Number of images to process in a batch.', default=10
    )
    parser.add_argument(
        '--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160
    )
    parser.add_argument(
        '--seed', type=int,
        help='Random seed.', default=42
    )
    parser.add_argument(
        '--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20  # noqa
    )
    # for split dataset
    parser.add_argument(
        '--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10  # noqa
    )
    parser.add_argument(
        '--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.25  # noqa
    )

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
