from imutils.video import VideoStream
from cv2 import cv2
import numpy as np
import argparse
import datetime
import imutils
import time

WHITE_PIXEL_VALUE = 255

class BlankSeqRemoval:
    def __init__(self,out_vid_name='',mov_detected_pixels_threshold=30, kernel_size=7, history = 10):
        self.video_capture = cv2.VideoCapture(0)
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=50,
            detectShadows=True
        )
        self.knn = cv2.createBackgroundSubtractorKNN(history=10)
        self.current_frame = None 
        self.previous_frame = None # initially there is no previous frame
        self.kernel_size = kernel_size
        self.cap_frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(out_vid_name+'.avi', codec, 20.0, (self.cap_frame_width, self.cap_frame_height))
        self.mov_detected_pixels_threshold = mov_detected_pixels_threshold*WHITE_PIXEL_VALUE
        

    def mark_as_removed(self,frame):
        # Draws the red diagonal lines
        cv2.line(frame, pt1=(0, 0), pt2=(frame.shape[1], frame.shape[0]), color=(0, 0, 255), thickness=8)
        cv2.line(frame, pt1=(0, frame.shape[0]), pt2=(frame.shape[1], 0), color=(0, 0, 255), thickness=8)

    def mod(self):
        while(True):            
            # Capture the video frame by frame
            _ , self.current_frame = self.video_capture.read()         
            
	        # resize the frame, convert it to grayscale, and blur it
            self.current_frame = imutils.resize(self.current_frame, width=500)
            current_frame_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            current_frame_gray = cv2.GaussianBlur(current_frame_gray, (self.kernel_size, self.kernel_size), 0)
            
            # Do not initiate the processing if there's no previous frame so initiate the previous frame and then skip the first loop
            if (self.previous_frame is None):
                self.previous_frame = current_frame_gray
                continue   
            self.previous_frame = current_frame_gray
            
            # compute the absolute difference between the current frame and previous frame
            frameDelta = cv2.absdiff(self.previous_frame, current_frame_gray)
            thresh = cv2.threshold(frameDelta, thresh=25, maxval=255, type=cv2.THRESH_BINARY)[1]
            # dilate the thresholded image to fill in holes, then find contours on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)

            white_pixels_count = np.sum(thresh)
            if white_pixels_count < self.mov_detected_pixels_threshold:
                self.mark_as_removed(self.current_frame)
            else:
                self.video_writer.write(self.current_frame)

            cv2.imshow("Live Feed", self.current_frame)
            cv2.imshow("Thresh", thresh)
            cv2.imshow("Frame Delta", frameDelta)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

    def mog(self):
        while(True):
            self.previous_frame = self.current_frame
            # Capture the video frame by frame
            _ , self.current_frame = self.video_capture.read()
            # Do not initiate the processing if there's no previous frame so skip the first loop
            if (self.previous_frame is None):
                continue             
        
            # Apply gray conversion and noise reduction (smoothening) for better and faster processing
            current_frame_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            current_frame_gray = cv2.GaussianBlur(current_frame_gray, (self.kernel_size, self.kernel_size), 0)
            cv2.imshow('gray',current_frame_gray)
            
            foreground_mask = self.mog2.apply(current_frame_gray, fgmask=None, learningRate=-1) 
            # fmask is the output foreground mask as an 8-bit binary image.
            # Next video frame. Floating point frame will be used without scaling and should be in range [0,255].
            # learningRate	The value between 0 and 1 that indicates how fast the background model is learnt.
            # Negative parameter value makes the algorithm to use some automatically chosen learning rate. 
            # 0 means that the background model is not updated at all, 1 means that the background model is completely reinitialized from the last frame.
            cv2.imshow('Foreground Mask', foreground_mask)

            white_pixels_count = np.sum(foreground_mask)
            if white_pixels_count < self.mov_detected_pixels_threshold:
                self.mark_as_removed(self.current_frame)
            else:
                self.video_writer.write(self.current_frame)

            cv2.imshow("Live Feed", self.current_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break                

    def stop_capture(self):
        self.video_capture.release()
        cv2.destroyAllWindows() 

bsr = BlankSeqRemoval(out_vid_name='out',kernel_size=21,history=100,mov_detected_pixels_threshold=1000)
bsr.mod()
bsr.stop_capture()

