import os
import cv2
import face_recognition
import pickle

def findEncodings(imgList):
    encodeList = []
    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            encodeList.append(encodings[0])
        else:
            print("⚠️ No face found in one of the images, skipping...")
    return encodeList

folderpath = "C:\\Users\\Ayon\\SIH\\face_recognation\\images"
pathList = os.listdir(folderpath)

imgList = []
studentsId = []

for path in pathList:
    img = cv2.imread(os.path.join(folderpath, path))
    if img is not None:
        imgList.append(img)
        studentsId.append(os.path.splitext(path)[0])
    else:
        print(f"⚠️ Could not read image: {path}")

print("Encoding Started.......")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds=[encodeListKnown,studentsId]
print("Encoding complete......")

file=open("EncodeFile.p","wb")
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("File saved")
