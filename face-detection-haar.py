import cv2

# Reading Image and Rescaling
image_path = r'C:\Users\soumy\Documents\Python Notebooks\OpenCV\object-detection-opencv\Human Face Datection\org_img.jpg'
imaging_org = cv2.imread(image_path)
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
imaging = rescaleFrame(imaging_org, 2.0)

# Haar Cascade XML file for Frontal Face Detection Model
haar_xml_path = r'C:\Users\soumy\Documents\Python Notebooks\OpenCV\object-detection-opencv\haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(haar_xml_path)  

# Detecting Faces using Model
faces = detector.detectMultiScale(imaging, minSize = (30, 30))  

# Bounding Detected Objects with Rectangles
if len(faces)!=0:
    for (a, b, width, height) in faces:  
        cv2.rectangle(imaging, (a, b), # Highlighting detected object with rectangle  
                      (a + height, b + width),   
                      (0, 275, 0), 3)

# Displaying
cv2.imshow('Detected', imaging)
# cv2.imwrite('haar_img.jpg', imaging)
cv2.waitKey(0)