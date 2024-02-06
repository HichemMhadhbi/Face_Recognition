import cv2  # OpenCV, une bibliothèque pour le traitement d'images
import numpy as np  # calculs numériques
import face_recognition  # reconnaissance faciale

imgIlon = face_recognition.load_image_file("imagesbasic/elon1.jpeg")
# Conversion de l'image en RGB (car face_recognition utilise ce format)
imgIlon = cv2.cvtColor(imgIlon, cv2.COLOR_BGR2RGB)

imgtest = face_recognition.load_image_file("imagesbasic/elon2.jpeg")
imgtest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)

# Détection des emplacements des visages dans l'image d'Elon Musk
faceLoc = face_recognition.face_locations(imgIlon)

# Encodage des visages détectés dans l'image d'Elon Musk
encodeIlon = face_recognition.face_encodings(imgIlon)[0]

# Impression des emplacements des visages détectés
print("top, right, bottom, left de l'image original: ",faceLoc)

# Modification de la ligne pour afficher un rectangle autour du premier visage détecté
##cv2.rectangle(imgIlon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
if faceLoc:
    top, right, bottom, left = faceLoc[0]  # top (coin supérieur), right (coin droit), bottom (coin inférieur) et left (coin gauche)
    cv2.rectangle(imgIlon, (left, top), (right, bottom), (255, 0, 255), 2) # (left, top): coins supérieur gauche (right, bottom): coins inférieur droit


faceLoctest = face_recognition.face_locations(imgtest)
encodetest = face_recognition.face_encodings(imgtest)[0]
print("top, right, bottom, left de l'image test: ", faceLoctest)
if faceLoctest:
    top, right, bottom, left = faceLoctest[0]  # top (coin supérieur), right (coin droit), bottom (coin inférieur) et left (coin gauche)
    cv2.rectangle(imgtest, (left, top), (right, bottom), (255, 0, 255), 2) # (left, top): coins supérieur gauche (right, bottom): coins inférieur droit

results = face_recognition.compare_faces([encodeIlon], encodetest)
print("Resultats de comparison: ",results)

faceDist = face_recognition.face_distance([encodeIlon], encodetest)
print("Distance entre ces deux images: ",faceDist)

cv2.putText(imgtest, f'{results} {round(faceDist[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Ilon Mask", imgIlon)
cv2.imshow("test", imgtest)
cv2.waitKey(0)

