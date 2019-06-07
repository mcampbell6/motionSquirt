import time
from picamera.array import PiRGBArray
from picamera import PiCamera
from datetime import datetime
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
import cv2 as cv
import numpy as np

import Adafruit_PCA9685

class MotionDetectorAdaptative():
    def onChange(self, val):  # callback when the user change the detection threshold
        self.threshold = val

    def __init__(self, threshold=1, show_windwos=False):
    	#Servo config
    	self.pwm = Adafruit_PCA9685.PCA9685()
    	self.pwm.set_pwm_freq(60)
    	self.pos_x = 300
    	self.pos_y = 300

    	self.point_y_max = 480
    	self.point_x_max = 640

    	self.servo_x_min = 100
    	self.servo_x_max = 500
    	self.servo_y_min = 475
    	self.servo_y_max = 575

    	#5v Relay for water solenoid config
    	self.relay_gpio = 17
    	GPIO.setup(self.relay_gpio, GPIO.OUT)
    	GPIO.output(self.relay_gpio, GPIO.LOW)
    	self.is_squirting = False
        self.show = show_windwos  # Either or not show the 2 windows
        self.frame = None

    	#picamera
    	self.camera = PiCamera()
    	self.camera.resolution = (self.point_x_max, self.point_y_max)
    	self.camera.framerate = 32
    	self.camera.vflip = True
    	time.sleep(2)
    	self.camera.saturation = 50
    	self.camera.brightness = 60
    	self.capture = PiRGBArray(self.camera, size=(self.point_x_max, self.point_y_max))

    	#background subtraction tool
    	self.hist = 5000
    	self.thresh = 16
    	self.shadows = False
    	self.fgbg = cv.createBackgroundSubtractorMOG2(history=self.hist, varThreshold=self.thresh, detectShadows=self.shadows)
    	self.grmask = None

        self.gray_frame = np.zeros((self.point_y_max, self.point_x_max), dtype=np.uint8)
        self.average_frame = np.zeros((self.point_y_max, self.point_x_max, 3), np.float32)
        self.absdiff_frame = None
        self.previous_frame = None

        self.surface = self.point_x_max * self.point_y_max
        self.currentsurface = 0
        self.currentcontours = None
        self.threshold = threshold
        if show_windwos:
            cv.namedWindow("Image")
            cv.createTrackbar("Detection treshold: ", "Image", self.threshold, 100, self.onChange)

    	self.set_axis(0)
    	self.set_axis(1)


    def run(self):
        for frame in self.camera.capture_continuous(self.capture, format="bgr", use_video_port=True):
            currentframe = frame.array
            self.process_image(currentframe)
            moved_result = self.something_has_moved()
            if len(moved_result) > 0:
                cv.circle(currentframe, (moved_result[0], moved_result[1]), 5, (0, 0, 255), -1)

                self.pos_x = abs(moved_result[0] - self.point_x_max) #invert movement of servo
                self.pos_y = abs(moved_result[1] - self.point_y_max) #invert movement of servo

                self.set_axis(0)
                self.set_axis(1)
                time.sleep(0.1)
                self.squirt(True)
            else:
                self.squirt(False)
                print("Nothing moving...")

            cv.drawContours(currentframe, self.currentcontours, -1, (0, 255, 0), 1)

            if self.show:
                cv.imshow("Image", currentframe)

            c = cv.waitKey(1) % 0x100
            if c == 27 or c == 10:  # Break if user enters 'Esc'.
                break

            self.capture.truncate(0)


    def process_image(self, curframe):
        curframe = cv.blur(curframe, (5, 5))
        #subtract background from image mask
        self.fgmask = self.fgbg.apply(curframe)

        #create a green mask to remove all green from motion detection (trees and plants and grass and what not)
        lower = np.array([38, 90, 40])
        upper = np.array([102, 255, 255])
        hsv = cv.cvtColor(curframe, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower, upper)
        kernelOpen = np.ones((5,5))
        kernelClose = np.ones((20,20))
        maskOpen = cv.morphologyEx(mask, cv.MORPH_OPEN, kernelOpen)
        maskClose = cv.morphologyEx(maskOpen,cv.MORPH_CLOSE,kernelClose)

        #invert mask so green is removed
        self.grmask =  255 - maskClose

        #combine masks
        self.finmask = self.grmask & self.fgmask


    def something_has_moved(self):
        # Find contours
        _, contours, _ = cv.findContours(self.finmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.currentcontours = contours  # Save contours

        totX = 0
        totY = 0
        totAvgX = 0
        totAvgY = 0
        count = 0
        for contour in contours:
            self.currentsurface += cv.contourArea(contour)
            maxC = max(contours, key=cv.contourArea)
            M = cv.moments(maxC)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                totX += cX
                totY += cY
                count += 1
        if count > 0:
            totAvgX = totX/count
            totAvgY = totY/count

        avg = (self.currentsurface * 100) / self.surface  # Calculate the average of contour area on the total size
        self.currentsurface = 0  # Put back the current surface to 0
        print('avg threshold ' + str(avg))

        if avg > 8 and avg < 50:
            #Stop background subtraction from tweaking out
        	self.fgbg = cv.createBackgroundSubtractorMOG2(history=self.hist, varThreshold=self.thresh, detectShadows=self.shadows)
        	return []
        if avg > 50:
        	return []
        if avg > self.threshold:
            return [totAvgX, totAvgY]
        else:
            return []


	#Servo functions
    def set_axis(self, channel):
		if channel is 0: #0 is X-Axis
			normX = int(((self.pos_x) * (self.servo_x_max-self.servo_x_min))/(640)) + self.servo_x_min
			print("moving X-Axis to point " + str(normX))
			self.pwm.set_pwm(channel, 0, normX)
		if channel is 1: #1 is Y-Axis
			normY = int(((self.pos_y) * (self.servo_y_max-self.servo_y_min))/(475)) + self.servo_y_min
			self.pwm.set_pwm(channel, 0, normY)
			print("moving Y-Axis to point " + str(normY))


    def squirt(self, should_squirt):
    	if should_squirt:
    		if self.is_squirting is False:
    			GPIO.output(self.relay_gpio, GPIO.HIGH)
    	else:
    		if self.is_squirting:
    			GPIO.output(self.relay_gpio, GPIO.LOW)

if __name__ == "__main__":
    detect = MotionDetectorAdaptative()
    detect.run()
