import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter
model = YOLO('yolov8n.pt')

cam = cv2.VideoCapture('people.mp4')

# Define region points as a polygon with 5 points
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360), (20, 400)]
w, h, fps = (int(cam.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True,
                 line_thickness=2)
while(True):
    #Capturar camara
    success, frame = cam.read()

    #Mostrar la ventana con la camara
    #detect = model(source=frame, show=True, conf=0.4, save=True)

    tracks = model.track(frame, persist=True, show=False)
    frame = counter.start_counting(frame, tracks)
    video_writer.write(frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#despues del loop release the cap object
cam.release()

#destruir ventanas
cv2.destroyWindow()