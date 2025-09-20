import cv2
import pickle
import face_recognition
import numpy as np
import cvzone
import time
import csv
import os
from datetime import datetime

# 📌 Load mobile camera (DroidCam)
cmp = cv2.VideoCapture(1)  # 1 = external/mobile cam
cmp.set(3, 640)
cmp.set(4, 480)

# 📌 Load known face encodings
print("Loading Encode file...")
with open("EncodeFile.p", "rb") as file:
    encodeListKnown, studentsId = pickle.load(file)
print("Encode file loaded")

# 📌 Setup CSV file for attendance
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
filename = current_date + '.csv'

file_exists = os.path.isfile(filename)
f = open(filename, 'a+', newline='')
lnwriter = csv.writer(f)

if not file_exists:
    lnwriter.writerow(["ID", "Time"])  # write header only once

# 📌 Load last marked data (persistent across runs)
if os.path.exists("last_marked.p"):
    with open("last_marked.p", "rb") as f2:
        last_marked = pickle.load(f2)
else:
    last_marked = {}

# 📌 Function to normalize ID (remove "_1", "_2", etc.)
def normalize_id(student_id):
    return student_id.split("_")[0]

# 📌 Face recognition loop
while True:
    success, img = cmp.read()
    if not success:
        print("⚠️ Failed to grab frame")
        break

    # Resize for faster processing
    imgSmall = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    # Detect faces and encodings
    faceCurrFrame = face_recognition.face_locations(imgSmall)
    encodeCurrFrame = face_recognition.face_encodings(imgSmall, faceCurrFrame)

    for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDist)

        # Scale back to original frame
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        bbox = (x1, y1, x2-x1, y2-y1)

        # ✅ Use stricter threshold to avoid false matches
        if matches[matchIndex] and faceDist[matchIndex] < 0.45:
            student_id = studentsId[matchIndex]
            base_id = normalize_id(student_id)
            current_time = datetime.now().strftime("%H:%M:%S")

            # Check last marked time (30-second rule)
            if base_id not in last_marked or (time.time() - last_marked[base_id]) > 30:
                lnwriter.writerow([base_id, current_time])
                f.flush()
                last_marked[base_id] = time.time()

                # Save updated last_marked
                with open("last_marked.p", "wb") as f2:
                    pickle.dump(last_marked, f2)

                print(f"*** ATTENDANCE MARKED for: {base_id} ***")
                cvzone.cornerRect(img, bbox, rt=0, colorC=(0, 255, 0))
                cv2.putText(img, f"{base_id} Marked", (x1, y1-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                print(f"⏳ Already marked for {base_id}, wait 30s")
                cvzone.cornerRect(img, bbox, rt=0, colorC=(0, 255, 255))
                cv2.putText(img, f"{base_id} Already Marked", (x1, y1-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        else:
            # ❌ Unknown face
            cvzone.cornerRect(img, bbox, rt=0, colorC=(0, 0, 255))
            cv2.putText(img, "Unknown", (x1, y1-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Attendance System", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Shutting down...")
cmp.release()
cv2.destroyAllWindows()
f.close()
