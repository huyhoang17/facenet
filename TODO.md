Paper: Facenet Google

1. Face detection + Face alignment ==> MTCNN: Multi Task Cascade Convolution Network

2. Face Recognition ==> Inception ResNet V1 + [SVM, KNN, ..]

Directory Structure
---
    Root: data folder
    ├── Tyra_Banks
    │ ├── Tyra_Banks_0001.jpg
    │ └── Tyra_Banks_0002.jpg
    ├── Tyron_Garner
    │ ├── Tyron_Garner_0001.jpg
    │ └── Tyron_Garner_0002.jpg

Process
---

Training Data 
-> Aligned Data (detect, crop, align) with MTCNN
-> Aligned Image Folder
--> Generate a Embedding Vector per Image
--> Train classifier model with [SVM, KNN, ..] algorithms -> Save
--> New testing image -> generate embedding -> evaluate model -> predict

TODO
---
Connect to db
Add person to recognize
Update model file, class_names, labels, embed_array


Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0002.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0004.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0005.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0006.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0008.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0009.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0010.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0011.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0013.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0014.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0015.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0016.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0018.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0019.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0022.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0024.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0025.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Manh_Tuan/MANH_TUAN_0026.jpg

Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Tien_Dinh/TIEN_DINH_0001.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Tien_Dinh/TIEN_DINH_0002.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Tien_Dinh/TIEN_DINH_0003.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Tien_Dinh/TIEN_DINH_0004.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Tien_Dinh/TIEN_DINH_0006.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Tien_Dinh/TIEN_DINH_0007.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Tien_Dinh/TIEN_DINH_0009.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Tien_Dinh/TIEN_DINH_0019.jpg
Can not align face in image: 
    /home/phanhoang/workspace/dataset/iCOMM/Tien_Dinh/TIEN_DINH_0024.jpg

