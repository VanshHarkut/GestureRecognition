import cv2
import mediapipe as mp
import time
import os
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
wCam = 640
hCam = 480
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "Finger_Images"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

# Load cheers emoji image
cheers_image = cv2.imread('Finger_Images/cheers.jpeg')  # Replace 'path_to_cheers_image.png' with the actual path to your cheers emoji image

# Load rock symbol image
rock_image = cv2.imread('Finger_Images/rock_symbol.jpg')  # Replace 'path_to_rock_symbol_image.png' with the actual path to your rock symbol image

# Load love symbol image
love_image = cv2.imread('Finger_Images/love.jpg')  # Replace 'path_to_love_symbol_image.png' with the actual path to your love symbol image

# Load forward symbol image
forward_image = cv2.imread('Finger_Images/forward.jpeg')  # Replace 'path_to_forward_symbol_image.png' with the actual path to your forward symbol image

# Load backward symbol image
backward_image = cv2.imread('Finger_Images/backward.png')  # Replace 'path_to_backward_symbol_image.png' with the actual path to your backward symbol image

print(len(overlayList))
cTime = 0
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)  # Hand/Finger is open
        else:
            fingers.append(0)  # Finger is closed

        # 4 fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)  # Hand/Finger is open
            else:
                fingers.append(0)  # Finger is closed

        # Cheers gesture: Thumb and pinky finger, and index, middle, ring fingers closed
        if fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
            # Display Cheers image and text
            h, w, c = cheers_image.shape
            img[0:h, 0:w] = cheers_image
            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Cheers!", (45, 375), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

        # Rock symbol: Index and pinky finger lifted, and other fingers closed
        elif fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
            # Display Rock image and text
            h, w, c = rock_image.shape
            img[0:h, 0:w] = rock_image
            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Rock!", (45, 375), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

        # Love gesture: Index, thumb, and pinky finger lifted, and other fingers closed
        elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
            # Display Love image and text
            h, w, c = love_image.shape
            img[0:h, 0:w] = love_image
            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Love!", (45, 375), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

        # Forward gesture: Thumb and index finger lifted, and other fingers closed
        elif fingers[0] == 1 and fingers[1] == 1 and all(fingers[i] == 0 for i in range(2, 5)):
            # Display Forward image and text
            h, w, c = forward_image.shape
            img[0:h, 0:w] = forward_image
            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Forward!", (45, 375), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

        # Backward gesture: Thumb lifted, and other fingers closed
        elif fingers[0] == 1 and all(fingers[i] == 0 for i in range(1, 5)):
            # Display Backward image and text
            h, w, c = backward_image.shape
            img[0:h, 0:w] = backward_image
            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Backward!", (45, 375), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

        else:
            # Display finger count image and text
            totalFingers = fingers.count(1)
            h, w, c = overlayList[0].shape
            img[0:h, 0:w] = overlayList[totalFingers - 1]
            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("My_Fingers", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
