import numpy as np
import cv2
import os

#KNN

def dis(x1,x2):
    return np.sqrt(sum((x1-x2)**2))
def knn(trainset1,test,k=5):
    values=[]
    size=trainset1.shape[0]
    for i in range(size):
        train=trainset1[i]
        d=dis(test,train[:-1])
        values.append((d,train[-1]))
    values=sorted(values)
    values=values[:k]
    print(values)
    values=np.array(values)
    values=np.unique(values[:,1],return_counts=True)
    ind=np.argmax(values[1])
    return values[0][ind]


# DATA PREPARATION
face_data=[]
label=[]
class_id=0
names={}
for file in os.listdir("E:\\Face detect\\"):
    if file.endswith('.npy'):
        names[class_id]=file[:-4]
        data_item=np.load("E:\\Face detect\\"+file)
        face_data.append(data_item)
    
        #create labels

        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        label.append(target)
face_dataset=np.concatenate(face_data,axis=0)
face_label=np.concatenate(label,axis=0).reshape((-1,1))
print(face_dataset.shape,face_label.shape)
print(face_dataset)

trainset=np.concatenate((face_dataset,face_label),axis=1)
print(face_dataset.shape,trainset.shape)

#testing

cap=cv2.VideoCapture(0)
face_detect=cv2.CascadeClassifier("Desktop\haarcascade_frontalface_alt.xml")
while True:
    ret,frame=cap.read()
    if ret==False:
        continue

    faces=face_detect.detectMultiScale(frame,1.3,5)
    for face in faces:
        x,y,w,h=face
        
        #region of interest

        offset=10
        frame_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        frame_section=cv2.resize(frame_section,(100,100))
        out=int(knn(trainset,frame_section.flatten()))

        #display name and rectangle
        cv2.putText(frame,names[out],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        cv2.imshow("frame",frame)

    key_pressed=cv2.waitKey(1)&0xFF
    if key_pressed==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()