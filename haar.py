import cv2


def faces(image):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	k=0

	img = cv2.imread(image)
	gray = cv2.imread(image,0)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		k=1
   	 
	if(k==1):
   	 	cv2.imwrite('img.jpg',roi_color)

