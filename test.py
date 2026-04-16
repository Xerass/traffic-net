#needed for video input
import cv2

#streamlines the yt feed -> input pipeline
from vidgear.gears import CamGear

#for the model
from ultralytics import YOLO

model_pth = "runs/trafficnet_cloud/weights/best.pt"

model = YOLO(model_pth)

yt_url = "https://www.youtube.com/watch?v=FWvIPfxK5Jo"

#cam gear handles yt-dlp extraction and background threading automatically
#stream_mode = True makes it suitable for live broadcasts
#we can also pass res constraints

options = {"STREAM_RESOLUTION": "720p"}
stream = CamGear(source = yt_url, stream_mode = True, logging = True, **options).start()


#very very simple test
while True:
    #read from most recent frames
    frame = stream.read()

    if frame is None:
            break

    # run inference
    results = model(frame, stream=True)
    
    for r in results:
        annotated_frame = r.plot()
        cv2.imshow("Threaded YouTube Pipeline", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
stream.stop()
cv2.destroyAllWindows()
