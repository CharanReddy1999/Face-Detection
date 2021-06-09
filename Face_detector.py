import cv2

#loading pre trained data on face frontals from opencv of haar cascade algorithm
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#lets choose an image to work on
#img=cv2.imread('RC_group.jpg')
webcam=cv2.VideoCapture(0)


while True:
    #webcam.read() gives two out parameters: boolean whether frame is successful T/F,the frame
    successful_frame_read,frame=webcam.read()


    #converting to grayscale
    grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    #detect faces
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)
    #face cordinate gives [[x coordinate from left,y cordinate from top,length to be extended from x,length to be extended from top]]


    #draw rectangles around the faces
    for [x,y,l,w] in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)



    #printing image on the screen ; parameters are tab name to be printed,img to be shown
    cv2.imshow('Face Detector',frame)
    key=cv2.waitKey(1)

    #q(ASCCI 113) Q(ASCCI 81)
    #quit if key is Q or q
    if key==113 or key==81:
        break

    #after quitting get rid of webcam
webcam.release()




print("code completed")