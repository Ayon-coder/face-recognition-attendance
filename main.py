import cv2
import pickle
import face_recognition
import numpy as np
import cvzone
import time
import csv
from datetime import datetime

cmp = cv2.VideoCapture(0)
cmp.set(3, 640)
cmp.set(4, 480)

# Load encoding file
print("Loading Encode file...")
with open("EncodeFile.p", "rb") as file:
    encodeListKnown, studentsId = pickle.load(file)
print("Encode file loaded")

# Setup CSV file for attendance
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)
lnwriter.writerow(["Name", "Time"])  # header

# Copy of student IDs (for marking attendance once)
students = studentsId.copy()

detected = False   # flag if a known face detected
start_time = None  # to track exit time

while True:
    success, img = cmp.read()
    if not success:
        print("⚠️ Failed to grab frame")
        break

    # Resize for faster processing
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    # Detect faces
    faceCurrFrame = face_recognition.face_locations(imgSmall)
    encodeCurrFrame = face_recognition.face_encodings(imgSmall, faceCurrFrame)

    for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDist)

        # Scale back to original frame
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        bbox = (x1, y1, x2 - x1, y2 - y1)

        if matches[matchIndex]:
            # ✅ Known face
            name = studentsId[matchIndex]
            img = cvzone.cornerRect(img, bbox, rt=0, colorC=(0, 255, 0))
            cv2.putText(img, "Successful", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            print("✅ Known Face Detected:", name)

            # Attendance marking
            if name in students:
                students.remove(name)
                current_time = datetime.now().strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])
                f.flush()
                print(f"*** ATTENDANCE MARKED for: {name} ***")

            if not detected:
                detected = True
                start_time = time.time()

        else:
            # ❌ Unknown face
            img = cvzone.cornerRect(img, bbox, rt=0, colorC=(0, 0, 255))
            cv2.putText(img, "Unknown", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Attendance System", img)

    # Auto close after 1 second if a known face was detected
    if detected and (time.time() - start_time > 1):
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Shutting down...")
cmp.release()
cv2.destroyAllWindows()
f.close()
