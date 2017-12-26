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
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import facenet


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
            # embedding_size = embeddings.get_shape()[1]

            test_image = facenet.load_test_data(
                args.path_image, False, False, args.image_size
            )
            feed_dict = {
                images_placeholder: test_image,  # ndarray
                phase_train_placeholder: False
            }
            emb_array = sess.run(
                embeddings, feed_dict=feed_dict
            )
            # emb_array = sess.run(
            #     [embeddings], feed_dict=feed_dict
            # )[0]

            print(emb_array)
            labels, class_names, emb_arrays = joblib.load(
                args.model_filename
            )
            # Classify images
            model = joblib.load(args.classifier_filename)
            # predictions = model.predict_proba(emb_array)
            emb_array = np.array(emb_array).reshape(1, -1)
            print(emb_array)
            if args.mode == 'SVM':
                predictions = model.predict_proba(emb_array)
                joblib.dump(predictions, 'models/predictions_svm.pkl')
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(
                    len(best_class_indices)), best_class_indices]

                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (
                        i, class_names[best_class_indices[i]],
                        best_class_probabilities[i])
                    )
                max_ind = np.argmax(best_class_indices)
                print('%d  %s: %f' % (
                    max_ind, class_names[max_ind],
                    best_class_probabilities[max_ind])
                )
            elif args.mode == 'KNN':
                print("emb_array".upper(), emb_array.shape)
                predictions = model.predict(emb_array)
                joblib.dump(predictions, 'models/predictions_knn.pkl')
                print(predictions)
                max_ind = predictions[0]
                print('%d  %s' % (max_ind, class_names[max_ind]))

            # accuracy = np.mean(np.equal(best_class_indices, labels))
            # print('Accuracy: %.3f' % accuracy)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'mode', type=str, choices=['SVM', 'KNN'],
        help='Indicates if a new classifier should be trained or a classification model should be used for classification', default='SVM'  # noqa
    )
    parser.add_argument(
        'path_image', type=str,
        help='Path to test image'
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
        '--model_filename',
        help='Classifier model file name as a pickle (.pkl) file. For training this is the output and for classification this is an input.'  # noqa
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
        '--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.5  # noqa
    )

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
