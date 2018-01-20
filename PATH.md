export PYTHONPATH=~/workspace/projects/fork/facenet/src

for N in {1..4}; do python3 src/align/align_dataset_mtcnn.py ./datasets/raw ./datasets/lfw_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 & done

ALIGN
---
python3 src/align/align_dataset_mtcnn.py ./datasets/raw ./datasets/lfw_160_2 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.75

python3 src/align/align_dataset_mtcnn.py ./src/align/image_test ./src/align/image_align --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.75

COMPARE IMAGES (VECTOR EMBEDDING)
---
python3 src/compare.py \
./models/20170512-110547 \
/home/phanhoang/workspace/projects/fork/facenet/datasets/lfw_160/Aaron_Peirsol/Aaron_Peirsol_0002.png \
/home/phanhoang/workspace/projects/fork/facenet/datasets/lfw_160/Aaron_Peirsol/Aaron_Peirsol_0004.png \
/home/phanhoang/Pictures/Phan_Hoang.JPG \
/home/phanhoang/Pictures/phh.jpg

VALIDATE
---
python3 src/validate_on_lfw.py ./datasets/lfw_160 ./models/20170512-110547

WEBCAM
---
python3 align_realtime.py \
--use_webcam True

TAG IMAGE
---
python3 src/align/align_image.py \
--path_image /home/phanhoang/workspace/projects/fork/facenet/datasets/raw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg

python3 src/align/align_image.py \
--detect_multiple_faces True \
--path_image /home/phanhoang/Pictures/btn-tsmt-2015.jpg

python3 src/align/align_image.py \
--detect_multiple_faces True \
--path_image /home/phanhoang/Pictures/Phan_Hoang.JPG

python3 src/align/image_detection.py \
--detect_multiple_faces True \
--path_image /home/phanhoang/Pictures/Phan_Hoang.JPG \
--classifier_filename models/knn_dump_updated.pkl \
--model_filename models/model_updated.pkl

python3 src/image_detection.py \
--model_trained models/20170512-110547 \
--detect_multiple_faces True \
--path_image /home/phanhoang/Pictures/btn-tsmt-2015.jpg \
--classifier_filename models/knn_dump_updated.pkl \
--model_filename models/model_updated.pkl \
--gpu_memory_fraction 0.75

######################################################

EMBED / TRAIN / CLASSIFIER
---
python3 src/classifier.py EMBED \
./datasets/lfw_160 \
./models/20170512-110547

python3 src/classifier.py SVM \
./datasets/lfw_160 \
./models/20170512-110547 \
--classifier_filename ./src/models/svm_dump.pkl \
--min_nrof_images_per_class 1

python3 src/classifier.py KNN \
./datasets/lfw_160 \
./models/20170512-110547 \
--classifier_filename ./src/models/knn_dump.pkl \
--min_nrof_images_per_class 1

python3 src/classifier.py CLASSIFY \
./datasets/lfw_160 \
./models/20170512-110547 \
--classifier_filename ./src/align/classifier.pkl \
--min_nrof_images_per_class 1

python3 src/classifier.py SVM \
/home/phanhoang/workspace/dataset/face_dataset_aligned_2 \
./models/20170512-110547 \
--classifier_filename ./src/models/svm_dump.pkl \

python3 src/classifier.py KNN \
/home/phanhoang/workspace/dataset/face_dataset_aligned_2 \
./models/20170512-110547 \
--classifier_filename ./src/models/all_knn_dump.pkl \
--gpu_memory_fraction 0.75

EVALUATE
---
python3 src/train_classifier.py KNN \
/home/phanhoang/workspace/projects/fork/facenet/datasets/raw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg \
./models/20170512-110547 \
--classifier_filename ./models/knn_dump.pkl\
--model_filename ./models/model.pkl

python3 src/train_classifier.py KNN \
/home/phanhoang/workspace/dataset/face_dataset_aligned_2/George_W_Bush/George_W_Bush_0052.png \
./models/20170512-110547 \
--classifier_filename models/knn_dump_2.pkl\
--model_filename models/model_2.pkl

######################################################

test image
python3 src/classifier.py CLASSIFY \
./src/align/image_test \
./models/20170512-110547 \
./src/align/classifier.pkl \
--min_nrof_images_per_class 1

add person:
    python3 src/add_person_recogize.py \
    --mode KNN \
    --input_dir /home/phanhoang/workspace/dataset/iCOMM/Ngọc_Bùi \
    --new_label Ngoc_Bui \
    --model models/20170512-110547 \
    --classifier_filename models/backup/knn_dump_updated \
    --model_filename models/backup/model_updated.pkl \
    --gpu_memory_fraction 0.75

image detection:
    python3 src/image_detection.py \
--model_trained models/20170512-110547 \
--detect_multiple_faces True \
--path_dir /home/phanhoang/workspace/dataset/iCOMM/Ngoc_Bui \
--classifier_filename models/backup/knn_dump_updated.pkl \
--model_filename models/backup/model_updated.pkl \
--output_dir /home/phanhoang/workspace/dataset/output_detection \
--gpu_memory_fraction 0.75