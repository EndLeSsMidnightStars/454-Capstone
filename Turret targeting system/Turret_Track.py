import cv2
from mss import mss

import time
import serial
import math
import numpy as np


### User Parameters ###

MOTOR_X_REVERSED = False

MAX_STEPS_X = 3

MAX_STEPS_Y = 2

STEPS_PER_REVOLUTION = 32
DEGREES_PER_STEP = 360 / STEPS_PER_REVOLUTION

#######################

def live_video(camera_port=0):
    """
    Opens a window with live video.
    :param camera:
    :return:
    """
    video_capture = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        # Display the resulting frame
        cv2.imshow('Video', frame)
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

def find_motion(uno, uno2, current_x_steps, current_y_steps, callback, camera_port=0, show_video=False):
    bounding_box = {'top': 0, 'left': 0, 'width': 500, 'height': 420}
    sct = mss()
    # camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
  #   time.sleep(0.25)
    # initialize the first frame in the video stream
    # firstFrame = None
    # tempFrame = None
    # count = 0
    # frame_counter = 0
    # loop over the frames of the video
    while True:
        # c = None
        # # grab the current frame and initialize the occupied/unoccupied
        # # text
        # (grabbed, frame) = camera.read()
        # # # if the frame could not be grabbed, then we have reached the end
        # # # of the video
        # if not grabbed:
        #     break
        # # resize the frame, convert it to grayscale, and blur it
        # #frame = imutils.resize(frame, width=500)
        # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # # if the first frame is None, initialize it
        # # Periodically reset firstFrame
        # if frame_counter % 150 == 0:  # Reset every 150 frames
        #     firstFrame = None
        #     print("Resetting background frame...")

        # if firstFrame is None:
        #     print("Waiting for video to adjust...")
        #     if tempFrame is None:
        #         tempFrame = gray
        #         continue
        #     else:
        #         delta = cv2.absdiff(tempFrame, gray)
        #         tempFrame = gray
        #         tst = cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY)[1]
        #         tst = cv2.dilate(tst, None, iterations=3)
        #         if count > 30:
        #             print("Done.\n Waiting for motion.")
        #             if not cv2.countNonZero(tst) > 0:
        #                 firstFrame = gray
        #             else:
        #                 continue
        #         else:
        #             count += 1
        #             continue
        # compute the absolute difference between the current frame and
        # first frame
        # ret, frame1 = camera.read()
        # ret, frame2 = camera.read()
        try: 
            sct_img = sct.grab(bounding_box)
            frame1 = np.array(sct_img)
            ret = True
        except Exception as e:
            print(f"Error capturing frame: {e}")
            ret = False
            frame1 = None

        try: 
            sct_img = sct.grab(bounding_box)
            frame2 = np.array(sct_img)
            ret = True
        except Exception as e:
            print(f"Error capturing frame: {e}")
            ret = False
            frame2= None

        frameDelta = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(frameDelta, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        dialated = cv2.dilate(thresh, None, iterations=3)
        c = get_best_contour(dialated.copy(), 10000)
        if c is not None:
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            x, y, w, h = cv2.boundingRect(c)
            # print(c)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            callback(c, current_x_steps, current_y_steps, frame1, uno, uno2)
        # show the frame and record if the user presses a key
        if show_video:
            cv2.imshow("Security Feed", frame1)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key is pressed, break from the lop
            if key == ord("q"):
                break
    # cleanup the camera and close any open windows
    # camera.release()
    cv2.destroyAllWindows()

def get_best_contour(imgmask, threshold):
    
    contours, hierarchy = cv2.findContours(imgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    best_area = threshold
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > best_area and area < 200000:
            best_area = area
            best_cnt = cnt
    return best_cnt

def spin(uno):
    uno.write(b'cw')

def calibrate(uno, uno2):
    """
    Waits for input to calibrate the turret's axis
    :return:
    """
  #   print("Please calibrate the tilt of the gun so that it is level. Commands: (w) moves up, " \
        #   "(s) moves down. Press (enter) to finish.\n")
  #   self.__calibrate_y_axis()
    print("Please calibrate the yaw of the gun so that it aligns with the camera. Commands: (a) moves left, " \
          "(d) moves right. Press (enter) to finish.\n")
    __calibrate_x_axis(uno)
    __calibrate_y_axis(uno2)
    print("Calibration finished.")

def __calibrate_x_axis(uno):
    """
    Waits for input to calibrate the x axis
    :return:
    """
    command = input()
    if command == "a":
        uno.write(b'c')
        print("Left")
    elif command == "d":
        uno.write(b'x')
        print("Right")

def __calibrate_y_axis(uno2):
    """
    Waits for input to calibrate the y axis.
    :return:
    """
    command = input()
    if command == "w":
        uno2.write(b'u')
        print("Up")
    elif command == "s":
        uno2.write(b'd')
        print("Down")

def motion_detection(uno, uno2, current_x_steps, current_y_steps, show_video=True):
    """
    Uses the camera to move the turret. OpenCV ust be configured to use this.
    :return:
    """
    find_motion(uno, uno2, current_x_steps, current_y_steps, __move_axis, show_video=show_video)

def __move_axis(contour, current_x_steps, current_y_steps, frame, uno, uno2):
    v_h, v_w = frame.shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    # find height
    target_steps_x = int((2*MAX_STEPS_X * (x + w / 2) / v_w) - MAX_STEPS_X)
    target_steps_y = int((2*MAX_STEPS_Y*(y+h/2) / v_h) - MAX_STEPS_Y)
    print ("x: %s" % str(target_steps_x))
    print("current x: %s" % str(current_x_steps))
    # move x
    if (target_steps_x - current_x_steps) > 0:
        for _ in range(target_steps_x - current_x_steps):
            current_x_steps += 1
            uno.write(b'x')
            print("ccw")
            time.sleep(0.5)
        # time.sleep(0.5)
    elif (target_steps_x - current_x_steps) < 0:
        for _ in range(current_x_steps - target_steps_x):
            current_x_steps -= 1
            uno.write(b'c')
            print("cw")
            time.sleep(0.5)
        # time.sleep(0.5)
    # move y
    if (target_steps_y - current_y_steps) > 0:
        for _ in range(target_steps_y - current_y_steps):
            current_y_steps += 1
            uno2.write(b'd')
            print("down")
            time.sleep(0.5)
        # time.sleep(0.5)
    elif (target_steps_y - current_y_steps) < 0:
        for _ in range(current_y_steps - target_steps_y):
            current_y_steps -= 1
            uno2.write(b'u')
            print("up")
            time.sleep(0.5)
        # time.sleep(0.5)

def calculate_steps(angle_difference):
    """Calculate the number of steps needed for a given angle difference."""
    steps = round(angle_difference / DEGREES_PER_STEP)
    return steps

def move_to_xyz(x, y, z, uno, uno2):
    global current_azimuth_angle, current_elevation_angle
    
    # Calculate the target azimuth and elevation angles
    target_azimuth_angle = math.degrees(math.atan2(y, x))
    distance_xy = math.sqrt(x**2 + y**2)
    target_elevation_angle = math.degrees(math.atan2(z, distance_xy))
    
    # Determine the angle differences
    azimuth_diff = target_azimuth_angle - current_azimuth_angle
    elevation_diff = target_elevation_angle - current_elevation_angle
    
    # Calculate the steps needed to reach the target angles
    azimuth_steps = calculate_steps(azimuth_diff)
    print(azimuth_steps)
    elevation_steps = round(elevation_diff / 30) # change depending on how much dc motor moves

    # Rotate horizontally (azimuth adjustment)
    if azimuth_steps != 0:
        direction = b'c' if azimuth_steps > 0 else b'x'
        for _ in range(abs(azimuth_steps)):
            uno.write(direction)
            time.sleep(0.2)
        print("Clockwise" if azimuth_steps > 0 else "Counter Clockwise")
        current_azimuth_angle += azimuth_steps * DEGREES_PER_STEP

    # Rotate vertically (elevation adjustment)
    if elevation_steps != 0:
        direction = b'u' if elevation_steps > 0 else b'd'
        for _ in range(abs(elevation_steps)):
            uno2.write(direction)
            time.sleep(0.2)
        print("Up" if elevation_steps > 0 else "Down")
        current_elevation_angle += elevation_steps * DEGREES_PER_STEP

    # Update current angles to the new target angles
    current_azimuth_angle = target_azimuth_angle
    current_elevation_angle = target_elevation_angle
    print(f"Moved to target at ({x}, {y}, {z}) with azimuth: {current_azimuth_angle}, elevation: {current_elevation_angle}")


uno = serial.Serial('COM6', 500000)
uno2 = serial.Serial('COM9', 500000)
current_x_steps = 0
current_y_steps = 0
current_azimuth_angle = 0  # Horizontal angle in degrees (0 is along positive x-axis)
current_elevation_angle = 0  # Vertical angle in degrees (0 is parallel to xy plane)

xyz_coord = [] # change to be input from audio code

try:       
    calibrate(uno, uno2)
    uno.write(b'o')
    time.sleep(1)
    move_to_xyz(xyz_coord[0], xyz_coord[1], xyz_coord[2], uno, uno2)
    motion_detection(uno, uno2, current_x_steps, current_y_steps, show_video=True)
except KeyboardInterrupt:
    print("\nProgram interrupted by user.")
finally:
    uno.close()
    uno2.close()