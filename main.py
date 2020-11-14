import cv2 as cv
import datetime
import numpy as np

# This is used to compare the previous frame with the current frame

white_pixel_value = 255

# We allow 30 white pixel to be the margin/threshold (to not detect any small movement)
difference_threshold = white_pixel_value * 30

# varThreshold is a parameter used for sensitivity
background_subtractor = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50,
                                                          detectShadows=False)
capture = cv.VideoCapture(0)

# This is used to output frames into a video
codec = cv.VideoWriter_fourcc(*'XVID')
video_writer = cv.VideoWriter('output.avi', codec, 20.0, (640, 480))

width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv.CAP_PROP_FPS)

current_frame = None
previous_frame = None

if not capture.isOpened:
    print('Unable to open: ')
    exit(0)
while True:
    previous_frame = current_frame
    _, current_frame = capture.read()

    if previous_frame is None:
        continue

    # Gray conversion and noise reduction (smoothening)
    current_frame_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
    current_frame_gray = cv.GaussianBlur(current_frame_gray, (25, 25), 0)

    background_substractor_mask = background_subtractor.apply(current_frame_gray, fgmask=None, learningRate=-1)

    cv.imshow('Current and previous frames difference', background_substractor_mask)

    copy = current_frame.copy()

    difference = np.sum(background_substractor_mask)
    if difference < difference_threshold:

        # Draws the red diagonal lines
        cv.line(copy, pt1=(0, 0), pt2=(width, height), color=(0, 0, 255), thickness=8)
        cv.line(copy, pt1=(0, height), pt2=(width, 0), color=(0, 0, 255), thickness=8)

    else:
        video_writer.write(current_frame)

    # Shows the current date and time
    cv.putText(copy, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
               (10, copy.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    cv.imshow("Live Feed", copy)

    # Defines q as the exit button
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

# Releases the video capture object
capture.release()
# Closes all the windows currently opened.
cv.destroyAllWindows()
