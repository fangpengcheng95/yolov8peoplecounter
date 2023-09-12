import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone
import numpy as np
from datetime import datetime

model=YOLO('yolov8s.pt')

class PersonInArea:
    def __init__(self, trackId, entryTime, departureTime):
        self.id = trackId
        self.entryTime = entryTime
        self.departureTime = departureTime

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        
#四个顶点坐标
pts = np.array([[300,200], [400, 120], [570, 106], [650, 300]], np.int32)
#顶点个数：4，矩阵变成4*1*2
pts = pts.reshape((-1, 1, 2))
#print(pts)

# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('vidp.mp4')

#视频属性
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #获取原视频的宽
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #获取原视频的搞
fps = int(cap.get(cv2.CAP_PROP_FPS)) #帧率
#fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) #视频的编码
fourcc = cv2.VideoWriter_fourcc(*'mp4v') #视频的编码

#视频对象的输出
out = cv2.VideoWriter('vidp_out.mp4', fourcc, 20.0, (1020, 500))

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
persondown={}
tracker=Tracker()
counter1=[]

personup={}
counter2=[]
cy1=194
cy2=220
offset=6
personInAreas=dict()
while True:    
    ret,frame = cap.read()
    if not ret:
        break
#    frame = stream.read()

    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    cv2.polylines(frame, [pts], isClosed=True, color=(0,0,255), thickness=2)

    results=model.predict(frame)
    #print(results)
    a=results[0].boxes.data
    #print(a)
    px=pd.DataFrame(a.cpu()).astype("float")
    #print(px)
    list=[]
   
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])
       
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        # 判断点是否在框定区域内
        color = ()
        if cv2.pointPolygonTest(pts, (cx, cy), False) < 0:
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            color = (0,255,0)
            if id in personInAreas and "departureTime" not in personInAreas[id]:
                # 离开区域的时间
                personInAreas[id]["departureTime"] = datetime.now().strftime("%H:%M:%S")
                #写入文件
                with open('result.txt', 'a') as f:
                    f.write(str(personInAreas[id]))
        else:
            cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
            color = (0, 0, 255)
            if id not in personInAreas:
                # 进入区域的时间
                dic = {"id": id, "entryTime": datetime.now().strftime("%H:%M:%S")}
                personInAreas[id] = dic
        cv2.rectangle(frame, (x3,y3), (x4,y4), color=color, thickness=2)
        id_text = id_text = '{}'.format(int(id))
        cv2.putText(frame, id_text, (x3, y3), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),thickness=2) 
    out.write(frame)
        #out.write(frame)
    #cv2.line(frame,(3,194),(1018,194),(0,255,0),2)
    #cv2.line(frame,(5,220),(1019,220),(0,255,255),2)
   

    #cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
