from cv2 import cv2
import numpy as np

# # define a video capture object
# video_stream = cv2.VideoCapture(0)
# bgs = cv2.createBackgroundSubtractorMOG2()
# knn = cv2.createBackgroundSubtractorKNN(history=10)
# while(True):

# 	# Capture the video frame by frame
#     _,frame = video_stream.read()
#     fmsk = knn.apply(frame)
# 	# Display the resulting frame q
#     cv2.imshow('frame', frame)
#     cv2.imshow('mask',fmsk)
# 	# the 'q' button is set as the 
# 	# quitting button you may use any 
# 	# desired button of your choice 
#     if cv2.waitKey(1) & 0xFF == ord('q'): 
#         break

# # After the loop release the cap object 
# video_stream.release() 
# # Destroy all the windows 
# cv2.destroyAllWindows() 

class BlankSeqRemoval:
    def __init__(self,out_vid_name=''):
        self.video_capture = cv2.VideoCapture(0)
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=50
        )
        self.knn = cv2.createBackgroundSubtractorKNN(history=10)
        self.current_frame = None 
        self.previous_frame = None # initially there is no previous frame
        self.cap_frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter('moving.avi', codec, 20.0, (self.cap_frame_width, self.cap_frame_height))

    def mark_as_removed(self,frame):
        # Draws the red diagonal lines
        cv2.line(frame, pt1=(0, 0), pt2=(self.cap_frame_width, self.cap_frame_height), color=(0, 0, 255), thickness=8)
        cv2.line(frame, pt1=(0, self.cap_frame_height), pt2=(self.cap_frame_width, 0), color=(0, 0, 255), thickness=8)
        return frame

    def mod(self):
        while(True):
            self.previous_frame = self.current_frame
            # Capture the video frame by frame
            _ , self.current_frame = self.video_capture.read()
            # Do not initiate the processing if there's no previous frame so skip the first loop
            if (self.previous_frame is None):
                continue            
        
            # Apply gray conversion and noise reduction (smoothening) for better and faster processing
            current_frame_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            current_frame_gray = cv2.GaussianBlur(current_frame_gray, (7, 7), 0)
            cv2.imshow('gray',current_frame_gray)
            
            foreground_mask = self.mog2.apply(current_frame_gray, fgmask=None, learningRate=-1) 
            # fmask is the output foreground mask as an 8-bit binary image.
            # Next video frame. Floating point frame will be used without scaling and should be in range [0,255].
            # learningRate	The value between 0 and 1 that indicates how fast the background model is learnt.
            # Negative parameter value makes the algorithm to use some automatically chosen learning rate. 
            # 0 means that the background model is not updated at all, 1 means that the background model is completely reinitialized from the last frame.
            cv2.imshow('Foreground Mask', foreground_mask)

            # copy = current_frame.copy()

            # difference = np.sum(background_substractor_mask)
            # if difference < difference_threshold:

            #     # Draws the red diagonal lines
            #     cv.line(copy, pt1=(0, 0), pt2=(width, height), color=(0, 0, 255), thickness=8)
            #     cv.line(copy, pt1=(0, height), pt2=(width, 0), color=(0, 0, 255), thickness=8)

            # else:
            #     video_writer.write(current_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

    def stop_capture(self):
        self.video_capture.release()
        cv2.destroyAllWindows() 

bsr = BlankSeqRemoval()
bsr.mod()
bsr.stop_capture()

