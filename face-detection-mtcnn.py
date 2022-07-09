import cv2
from mtcnn.mtcnn import MTCNN

image_path = r'C:\Users\soumy\Documents\Python Notebooks\OpenCV\object-detection-opencv\Human Face Datection\org_img.jpg'
imaging_org = cv2.imread(image_path)
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
imaging = rescaleFrame(imaging_org, 2.0)

# Create the Detector (default weights) and Detect Faces
detector = MTCNN()
faces = detector.detect_faces(imaging)

# Function to Bound Detected Objects with Rectangles
def draw_faces(imaging, faces):
    if len(faces)!=0:
        for face in faces:  
            a, b, width, height = face['box'][0:4]
            cv2.rectangle(imaging, (a, b), (a + height, b + width), (0, 275, 0), 3)

# Highlight Faces and Display
draw_faces(imaging, faces)
cv2.imshow('Detected', imaging)
# cv2.imwrite('mtcnn_img.jpg', imaging)
cv2.waitKey(0)