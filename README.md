# CNN_FaceMask_Detection
NOTE : Sample data file and the Model files are heavy in size and can be accessed from here
https://drive.google.com/drive/folders/17yKvB7S8fuvy3NCYe0wEmNERMa78IBWm?usp=sharing

Find the complete video presentation here : https://www.loom.com/share/26c70846c8334def9a669d2397b69204

..............


Steps to use model in your local machine

1. create a new environment using the requirements.txt file using "python -m pip install -r requirements.txt"
2. Open terminal and move to the directory you have downloaded "Face Mask Detection"
3. Run python main.py


main.py will create a webserver
it will call the camera.py script
camera.py script will initiate  the camera on your machine and start capturing live images
camera.py will send live images to model.py
model.py will load the model and send predictions
based on prediction camera.py will classify and highlight face as "Mask" or "No-Mask"


Details of each file,

1. File "Mask_NoMaks_CNN.ipynb" is the notebook where you can find all the code for model training and evaluation.
2. File "Webscrap_download from istock.ipynb", is code to web-scrap all the images from istock, shutterstock.
3. File "detect face to create dataset.ipynb", is code to clean web scrapped images, find images with humans, crop around headshot and resize to 128X128.
4. File "main.py" is to create webserver and uses index.html inside "template" folder.
5. File "camera.py" is to initiate live camera feed, capture frames, format frame for model, send to model for prediction and use prediction to highlight/classify face in live video feed.
6. File "haarcascade_frontalface_default.xml" is OpenCV model to detect faces in Camera.py
7. File "requirements.txt" and "requirement.txt" are environment requirement files created using PIP and Conda respectively.
8. Folder "best trained models" all the files to load trained model for Mask and No_Masks detection.


NOTE ; DATA FOLDER HAS ONLY FEWER IMAGES AS COMPARE TO ALL IMAGES USED FOR TRAINING AND EVALUATION!!