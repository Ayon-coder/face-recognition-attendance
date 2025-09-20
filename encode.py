
import os
import cv2
import face_recognition
import pickle

def findEncodings(imgList, imgNames):
    encodeList = []
    validIds = []

    for img, name in zip(imgList, imgNames):
        # Resize image if too small
        h, w = img.shape[:2]
        if h < 200 or w < 200:
            img = cv2.resize(img, (0, 0), fx=2, fy=2)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(imgRGB)

        if len(faces) == 0:
            print(f"⚠️ No face found in {name}, skipping...")
            continue

        encodings = face_recognition.face_encodings(imgRGB, faces)
        if len(encodings) == 0:
            print(f"⚠️ Could not encode face in {name}, skipping...")
            continue

        encodeList.append(encodings[0])
        validIds.append(name)
        print(f"✅ Encoded face for: {name}")

    return encodeList, validIds

# Path to your images folder
folderpath = "C:\\Users\\Ayon\\SIH\\face_recognation\\images"
pathList = os.listdir(folderpath)

imgList = []
studentsId = []

for path in pathList:
    imgPath = os.path.join(folderpath, path)
    img = cv2.imread(imgPath)
    if img is not None:
        imgList.append(img)
        studentsId.append(os.path.splitext(path)[0])
    else:
        print(f"⚠️ Could not read image: {path}")

print("🔄 Encoding Started...")
encodeListKnown, validIds = findEncodings(imgList, studentsId)
print("✅ Encoding complete.")

# Save the encodings and IDs
file = open("EncodeFile.p", "wb")
pickle.dump([encodeListKnown, validIds], file)
file.close()
print("💾 Encodings saved to EncodeFile.p")
