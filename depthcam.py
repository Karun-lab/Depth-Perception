import cv2
import torch
import matplotlib.pyplot as plt

#download these models
midas= torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

#Input transformation pipeline 
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transforms = transforms.small_transform

#hook into opencv
cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    #transform input for midas
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transforms(img).to('cpu')

    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode ='bicubic',
            align_corners=False

        ).squeeze()

        output= prediction.cpu().numpy()
        print(output)



    plt.imshow(output)
    cv2.imshow('CV2Frame', frame)
    plt.pause(0.00001)

    if cv2.waitKey(10) & 0xff == ord('q'):
        cap.release()
        cv2.destroyAllWindow()
plt.show()