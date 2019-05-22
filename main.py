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

    def __init__(self, threshold=1, doRecord=False, showWindows=True):
	#Servo config
	self.pwm = Adafruit_PCA9685.PCA9685()
	self.pwm.set_pwm_freq(60)
	self.posX = 300
	self.posY = 300

	self.pointYmax = 480
	self.pointXmax = 640

	self.servoXmin = 100
	self.servoXmax = 500
	self.servoYmin = 475
	self.servoYmax = 575

	#5v Relay for water solenoid config
	self.relay_gpio = 17
	GPIO.setup(self.relay_gpio, GPIO.OUT)
	GPIO.output(self.relay_gpio, GPIO.LOW)
	self.is_squirting = False
        self.writer = None
        self.font = None
        self.doRecord = doRecord  # Either or not record the moving object
        self.show = showWindows  # Either or not show the 2 windows
        self.frame = None

	#picamera
	self.camera = PiCamera()
	self.camera.resolution = (self.pointXmax, self.pointYmax)
	self.camera.framerate = 32
	self.camera.vflip = True
	time.sleep(2)
	self.camera.saturation = 50
	self.camera.brightness = 60
	self.capture = PiRGBArray(self.camera, size=(self.pointXmax, self.pointYmax))

	#background subtraction tool
	self.hist = 5000
	self.thresh = 16
	self.shadows = False
	self.fgbg = cv.createBackgroundSubtractorMOG2(history=self.hist, varThreshold=self.thresh, detectShadows=self.shadows)
	self.grmask = None
	time.sleep(2)
        if doRecord:
            self.initRecorder()

        self.gray_frame = np.zeros((self.pointYmax, self.pointXmax), dtype=np.uint8)
        self.average_frame = np.zeros((self.pointYmax, self.pointXmax, 3), np.float32)
        self.absdiff_frame = None
        self.previous_frame = None

        self.surface = self.pointXmax * self.pointYmax
        self.currentsurface = 0
        self.currentcontours = None
        self.threshold = threshold
        self.isRecording = False
        self.trigger_time = 0  # Hold timestamp of the last detection
	self.location = 0
        if showWindows:
            cv.namedWindow("Image")
            cv.createTrackbar("Detection treshold: ", "Image", self.threshold, 100, self.onChange)

	self.setAxis(0)
	self.setAxis(1)

    def initRecorder(self):  # Create the recorder
        codec = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')  # ('W', 'M', 'V', '2')
        self.writer = cv.VideoWriter(filename=datetime.now().strftime("%b-%d_%H_%M_%S") + ".avi", fourcc=codec, fps=5,
                                     frameSize=(self.pointXmax, self.pointYmax), isColor=True)


        self.font = cv.FONT_HERSHEY_SIMPLEX

    def run(self):
        started = time.time()
        for frame in self.camera.capture_continuous(self.capture, format="bgr", use_video_port=True):
            currentframe = frame.array
            instant = time.time()  # Get timestamp o the frame

	    self.processImage2(currentframe) # Process the image

            if not self.isRecording:
		movedRes = self.somethingHasMoved()
                if len(movedRes) > 0:
                    self.trigger_time = instant  # Update the trigger_time
                    if instant > started + 10:  # Wait 5 second after the webcam start for luminosity adjusting etc..
                        #print("Something is moving! Placing a dot here: {}".format(movedRes))
			cv.circle(currentframe, (movedRes[0], movedRes[1]), 5, (0, 0, 255), -1)
                        if self.doRecord:  # set isRecording=True only if we record a video
                            self.isRecording = True

			self.posX = abs(movedRes[0] - self.pointXmax) #invert movement of servo
			self.posY = abs(movedRes[1] - self.pointYmax) #invert movement of servo

			self.setAxis(0)
			self.setAxis(1)
			time.sleep(0.1)
			self.squirt(True)
		else:
			self.squirt(False)
			print("Nothing moving...")

                cv.drawContours(currentframe, self.currentcontours, -1, (0, 255, 0), 1)
            else:
                if instant >= self.trigger_time + 10:  # Record during 10 seconds
                    print("Stop recording")
                    self.isRecording = False
                else:
                    cv.putText(currentframe, datetime.now().strftime("%b %d, %H:%M:%S"), (25, 30), self.font, 1, 1, 2,
                               8, 0)  # Put date on the frame
                    self.writer.write(currentframe)  # Write the frame

            if self.show:
                cv.imshow("Image", currentframe)

            c = cv.waitKey(1) % 0x100
            if c == 27 or c == 10:  # Break if user enters 'Esc'.
                break
	    self.capture.truncate(0)


    def processImage2(self, curframe):
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


    def somethingHasMoved(self):
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
		self.fgbg = cv.createBackgroundSubtractorMOG2(history=self.hist, varThreshold=self.thresh, detectShadows=self.shadows)
		return []
	if avg > 50:
		return []
        if avg > self.threshold:
            return [totAvgX, totAvgY]
        else:
            return []


	#Servo functions
    def setAxis(self, channel):
		if channel is 0: #0 is X-Axis
			normX = int(((self.posX) * (self.servoXmax-self.servoXmin))/(640)) + self.servoXmin
			print("moving X-Axis to point " + str(normX))
			self.pwm.set_pwm(channel, 0, normX)
		if channel is 1: #1 is Y-Axis
			normY = int(((self.posY) * (self.servoYmax-self.servoYmin))/(475)) + self.servoYmin
			self.pwm.set_pwm(channel, 0, normY)
			print("moving Y-Axis to point " + str(normY))
    def normalizePulse(pulse):
		print("normalize")

    def squirt(self, should_squirt):
	if should_squirt:
		if self.is_squirting is False:
			GPIO.output(self.relay_gpio, GPIO.HIGH)
	else:
		if self.is_squirting:
			GPIO.output(self.relay_gpio, GPIO.LOW)

if __name__ == "__main__":
    detect = MotionDetectorAdaptative(doRecord=False)
    detect.run()

