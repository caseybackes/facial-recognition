import cv2, pickle

face_cascade_frontalface_alt2 = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

labels = {"person_name":1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}



cap = cv2.VideoCapture(0)
while True:
    # capture frame by frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces1 = face_cascade_frontalface_alt2.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=1)



    for (x, y, w, h) in faces1:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # recognize?
        id_, conf = recognizer.predict(roi_gray)

        if conf >=45 and conf <= 85:
            #print id_
            print labels[id_], " w/conf: ", conf
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = str(labels[id_])
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = 'my-image.png'
        cv2.imwrite(img_item, roi_gray)

        # make a box around the recognized face
        color = (255, 0, 0)  # blue green red
        stroke = 2  # how thick are the lines in the box
        end_cord_x = x + w
        end_cord_y = y + h
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # recognize other features (of a face)?
        subitems = smile_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew, eh) in subitems:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)


    # display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xf == ord('q'):
        break



# when all done, release the capture
cap.release()
cv2.destroyAllWindows()
