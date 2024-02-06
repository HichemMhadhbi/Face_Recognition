from datetime import datetime

import cv2
import numpy as np
import face_recognition
import os #manipulation de fichiers et de répertoires

path = "imagesattendance"
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)
for image in mylist:
    curlImg = cv2.imread(f'{path}/{image}') #Charge l’image depuis le fichieq
    images.append(curlImg)
    classNames.append(os.path.splitext(image)[0]) #extrait le nom de fichier sans son extension
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            encodeList.append(face_encodings[0])
        else:
            print("Aucun visage détecté dans l'image.")
    return encodeList

def markAttendance(name):
    with open('Attendances.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(', ')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            nowStr = now.strftime("%H : %M: %S")
            f.writelines(f'\n{name}, {nowStr}')


encodeListKnown = findEncodings(images)
##print("Longeur: ",len(encodeListKnown))

# Initialisation et préparation
cap = cv2.VideoCapture(0)  # Initialisation de la capture vidéo à partir de la webcam

while True:
    success, img = cap.read()  # Lecture d'une image depuis la webcam
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)  # Redimensionnement de l'image capturée
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Conversion de l'image en format RGB

    # Détection des visages dans l'image
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Comparaison des encodages avec ceux de la liste connue
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dessin d'un rectangle autour du visage
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)  # Rectangle rempli pour le nom
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # Ajout du nom

            # Enregistrement de la présence dans le fichier CSV
            markAttendance(name)

    cv2.imshow("WebCam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
