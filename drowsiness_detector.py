import cv2 as cv
import dlib
from scipy.spatial import distance

def ear_aspect_ratio(eye):
	a = distance.euclidean(eye[1], eye[5])
	b = distance.euclidean(eye[2], eye[4])
	c = distance.euclidean(eye[0], eye[3])
	ratio = (a+b)/(2*c)
	return ratio

cap = cv.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("C:\Code From VScode\Python\OpenCV\Shape predictor Face landmark data file\shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)

    for face in faces:
		
        face_landmarks = dlib_facelandmark(gray, face)
		
        leftEye = []
        rightEye = []

        for n in range(36,42):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	leftEye.append((x,y))
        	next_point = n+1

        	if n == 41:
        		next_point = 36

        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	rightEye.append((x,y))
        	next_point = n+1

        	if n == 47:
        		next_point = 42

        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv.line(frame,(x,y),(x2,y2),(0,255,0),1)

        left_ear = ear_aspect_ratio(leftEye)
        right_ear = ear_aspect_ratio(rightEye)

        ear = (left_ear+right_ear)/2
        ear = round(ear,2)
        if ear<0.259:
        	cv.putText(frame,"Half Sleepy",(20,80),cv.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
        	cv.putText(frame,"Wake Up!!",(20,450),cv.FONT_HERSHEY_SIMPLEX,2,(255,0,255),4)
        	print("Drowsy")
        print(ear)

    cv.imshow("Drowsiness Detector", frame)

    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()