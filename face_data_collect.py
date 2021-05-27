import numpy as np
import cv2

#initialize camera
cap=cv2.VideoCapture(0)

#detect faces
face_detect=cv2.CascadeClassifier("Desktop\haarcascade_frontalface_alt.xml")

name=input('ENTER YOUR NAME?\n')
skip=1
face_data=[]
path="E:\\Face detect\\"
while True:
    ret,frame=cap.read()

    if ret==False:
        continue
    
    faces=face_detect.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3])
    for (x,y,w,h) in faces[-1:]:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

      
        #extract the face
        offset=10
        frame_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        frame_section=cv2.resize(frame_section,(100,100))

        
        if skip%10==0:
            face_data.append(frame_section)
            print(len(face_data))
        skip+=1

    cv2.imshow('frame ',frame)
    if len(face_data)!=0:
        cv2.imshow('frame section',face_data[-1])
    key_pressed=cv2.waitKey(1)&0xFF
    if key_pressed==ord('q'):
        break

#converting to numpy array.
print(face_data)

face_data=np.asarray(face_data)
face_data=face_data.reshape(face_data.shape[0],-1)
print(face_data)
print(face_data.shape)

np.save(path+name+'.npy',face_data)

print("data succesfully saved")
cap.release()
cv2.destroyAllWindows()