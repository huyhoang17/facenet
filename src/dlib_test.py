import sys

import cv2
import dlib


predictor_path = sys.argv[1]  # models/shape_predictor_5_face_landmarks.dat
face_file_path = sys.argv[2]  # path to image

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

# Load the image using OpenCV
bgr_img = cv2.imread(face_file_path)
if bgr_img is None:
    print("Sorry, we could not load '{}' as an image".format(face_file_path))
    exit()

# Convert to RGB since dlib uses RGB images
img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
dets = detector(img, 2)
num_faces = len(dets)
if num_faces == 0:
    print("Sorry, there were no faces found in '{}'".format(face_file_path))
    exit()


def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


face_locations = [
    _trim_css_to_bounds(
        (detection.top(), detection.right(),
         detection.bottom(), detection.left()),
        img.shape
    ) for detection in dets
]

# Find the 5 face landmarks we need to do the alignment.
faces = dlib.full_object_detections()
for detection in dets:
    faces.append(sp(img, detection))
# Get the aligned face images
# image size: (160, 160, 3), padding: 32px
images = dlib.get_face_chips(img, faces, size=160, padding=0.2)
for (top, right, bottom, left), image in zip(face_locations, images):
    # cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw a box around the face
    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(img, (left, bottom - 35),
                  (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, "ahihi", (left + 6, bottom - 6),
                font, 1.0, (255, 255, 255), 1)

cv2.imshow('image', img)
cv2.waitKey(0)

# It is also possible to get a single chip
# image = dlib.get_face_chip(img, faces[0])
# cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# cv2.imshow('image', cv_bgr_img)
# cv2.waitKey(0)

cv2.destroyAllWindows()

# run command
# python3 src/dlib_test.py models/shape_predictor_5_face_landmarks.dat /home/phanhoang/Pictures/phh.jpg
