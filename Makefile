export:
	export PYTHONPATH=~/workspace/projects/fork/facenet/src

align_dataset:
	python3 src/align/align_dataset_mtcnn.py ./datasets/raw ./datasets/lfw_160_2 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.75

align_per_dataset:
	python3 src/align/align_dataset_mtcnn_fix.py \
	/home/phanhoang/workspace/dataset/face_dataset_raw \
	/home/phanhoang/workspace/dataset/face_dataset_aligned \
	--image_size 160 \
	--margin 32 \
	--random_order \
	--min_images_per_label 4 \
	--max_images_per_label 10 \
	--gpu_memory_fraction 0.25

align_per_dataset_test:
	python3 src/align/align_dataset_mtcnn_fix.py \
	/home/phanhoang/Desktop/test_image \
	/home/phanhoang/Desktop/test_align \
	--image_size 160 \
	--margin 32 \
	--random_order \
	--min_images_per_label 2 \
	--max_images_per_label 10 \
	--gpu_memory_fraction 0.25

web:
	python3 src/align/align_realtime.py \
--use_webcam True

webcam:
	python3 src/realtime_detection.py \
	--model_trained models/20170512-110547 \
	--classifier_filename models/backup/knn_dump_updated.pkl \
	--model_filename models/backup/model_updated.pkl \
	--gpu_memory_fraction 0.5

webcam_dlib:
	python3 src/realtime_detection_dlib.py \
	--model_trained models/20170512-110547 \
	--classifier_filename models/backup/knn_dump_updated.pkl \
	--predictor_path models/shape_predictor_5_face_landmarks.dat \
	--model_filename models/backup/model_updated.pkl \
	--gpu_memory_fraction 0.5

image_dir:
	python3 src/image_detection.py \
--model_trained models/20170512-110547 \
--detect_multiple_faces True \
--path_dir /home/phanhoang/workspace/dataset/test_detection_2 \
--classifier_filename models/knn_dump_updated.pkl \
--model_filename models/model_updated.pkl \
--output_dir /home/phanhoang/workspace/dataset/output_detection \
--gpu_memory_fraction 0.75

image_path:
	python3 src/image_detection.py \
--model_trained models/20170512-110547 \
--detect_multiple_faces True \
--path_image /home/phanhoang/Pictures/btn-tsmt-2015.jpg \
--classifier_filename models/backup/knn_dump_updated.pkl \
--model_filename models/backup/model_updated.pkl \
--gpu_memory_fraction 0.3

image_dlib:
	python3 src/image_detection_dlib.py \
--model_trained models/20170512-110547 \
--predictor_path models/shape_predictor_5_face_landmarks.dat \
--path_image /home/phanhoang/Pictures/btn-tsmt-2015.jpg \
--classifier_filename models/backup/knn_dump_updated.pkl \
--model_filename models/backup/model_updated.pkl \
--gpu_memory_fraction 0.7

align_image:
	python3 src/align/align_image.py \
--path_image /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0002.jpg

add_person:
	python3 src/add_person_recogize.py \
	--mode KNN \
	--input_dir /home/phanhoang/workspace/dataset/iCOMM/Tien_Dinh \
	--new_label Tien_Dinh \
	--model models/20170512-110547 \
	--classifier_filename models/backup/knn_dump_updated.pkl \
	--model_filename models/backup/model_updated.pkl \
	--gpu_memory_fraction 0.5


embed:
	python3 src/classifier.py EMBED \
./datasets/lfw_160 \
./models/20170512-110547

svm:
	python3 src/classifier.py SVM \
./datasets/lfw_160 \
./models/20170512-110547 \
--classifier_filename ./models/svm_dump.pkl \
--min_nrof_images_per_class 1 \
--gpu_memory_fraction 0.5

knn:
	python3 src/classifier.py KNN \
./datasets/lfw_160 \
./models/20170512-110547 \
--classifier_filename ./models/knn_dump.pkl \
--min_nrof_images_per_class 1

evaluate_svm:
	python3 src/train_classifier.py SVM \
/home/phanhoang/workspace/projects/fork/facenet/datasets/lfw_160/Aaron_Eckhart/Aaron_Eckhart_0001.png \
./models/20170512-110547 \
--classifier_filename ./models/svm_dump.pkl \
--model_filename ./models/model.pkl

evaluate_knn:
	python3 src/train_classifier.py KNN \
/home/phanhoang/workspace/projects/fork/facenet/datasets/lfw_160/Andre_Agassi/Andre_Agassi_0032.png \
./models/20170512-110547 \
--classifier_filename ./models/knn_dump.pkl \
--model_filename ./models/model.pkl

evaluate_knn_test_image_2:
	python3 src/train_classifier.py KNN \
/home/phanhoang/workspace/dataset/face_dataset_aligned_2/Angelina_Jolie/Angelina_Jolie_0012.png \
./models/20170512-110547 \
--classifier_filename models/knn_dump_2.pkl \
--model_filename models/model_2.pkl

compare_test:
	python3 src/compare.py \
./models/20170512-110547 \
/home/phanhoang/workspace/projects/fork/facenet/datasets/lfw_160/Abdullah_al-Attiyah/Abdullah_al-Attiyah_0002.png \
/home/phanhoang/workspace/projects/fork/facenet/datasets/lfw_160/AJ_Cook/AJ_Cook_0001.png

	python3 src/compare.py \
./models/20170512-110547 \
/home/phanhoang/workspace/projects/fork/facenet/datasets/lfw_160/Aaron_Peirsol/Aaron_Peirsol_0002.png \
/home/phanhoang/workspace/projects/fork/facenet/datasets/lfw_160/Aaron_Peirsol/Aaron_Peirsol_0004.png \
/home/phanhoang/Pictures/Phan_Hoang.JPG \
/home/phanhoang/Pictures/phh.jpg
