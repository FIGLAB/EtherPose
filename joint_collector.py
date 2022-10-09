import sys
import os
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.opengl as gl
import numpy as np
import pyqtgraph as pg
from matplotlib import cm
import matplotlib.pyplot as plt
import time
import threading

import cv2
import mediapipe as mp

class JointCollector():
    def __init__(self):
        self.thread = None
        self.hand_data = np.zeros((2, 21, 3))
        self.cursor = np.zeros((2, 21, 3))
        self.init_camera()
        self.init_mediapipe()

    def init_camera(self):
        self.cap = cv2.VideoCapture(0)
        # self.cap = cv2.VideoCapture("./blazepose_hand_data/IMG_0140.MOV")

    def init_mediapipe(self):  
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

    def init_save_video(self):
        w = round(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = self.cap.get(cv2.CAP_PROP_FPS)
        fps = 60
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.out = cv2.VideoWriter("./blazepose_hand_data/output.mp4", fourcc, fps, (w, h))

    def save_video(self, frame):
        self.out.write(frame)

    # def run_visualizer(self):
    #     app = QtGui.QApplication([])
    #     w = gl.GLViewWidget()
    #     w.opts['distance'] = 1
    #     w.show()
    #     #w.orbit(45,90)
    #     w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')

    #     pos = np.empty((21, 3))
    #     size = np.empty((21,))
    #     color = np.empty((21, 4))
    #     pos[:] = (0.0, 0.0, 0.0)
    #     size[:] = 0.01
    #     color[:] = (1.0, 0.0, 0.0, 0.75)
            
    #     sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
    #     sp1.translate(0,0,0)
    #     w.addItem(sp1)

    #     ax = gl.GLAxisItem(QtGui.QVector3D(100, 100, 100))
    #     w.addItem(ax)

    #     self.sp1 = sp1 

    #     timer = QtCore.QTimer()
    #     timer.timeout.connect(self.update_visualizer)
    #     timer.start()

    #     import sys
    #     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #         QtGui.QApplication.instance().exec_()

    def update_visualizer(self):
        self.sp1.setData(pos=self.hand_data[0])

    def close(self):
        self.cap.release()
        self.out.release()

    def run(self):
        self.stopThread = False
        self.thread = threading.Thread(target=self.capture_data, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread is None:
            print("No currently running thread.")
        else:
            self.stopThread = True

    def isThereHand(self, hand):
        return np.sum(np.abs(hand))!=0.0

    def get_hands(self):
        right_hand = self.hand_data[0]
        left_hand = self.hand_data[1]
        is_right_hand = self.isThereHand(right_hand) and (np.abs(right_hand[0,0]) < 200)
        is_left_hand = self.isThereHand(left_hand) and (np.abs(left_hand[0,0]) < 200)
        return right_hand, left_hand, is_right_hand, is_left_hand

    def get_cursor(self):
        self.cursor[0] = self.to_palm_coordinate(self.hand_data[0],21)
        cursor = np.array(self.cursor[:,4,:], copy=True)
        cursor[:,0] = -cursor[:,0]
        # cursor = self.to_1d_slider(self.cursor[0],21)
        return cursor

    def get_axis(self):
        return self.axis

    def normalize(self, x):
        res = x / np.linalg.norm(x)
        return res

    def to_1d_slider(self, hand_data, num):
        p4 = hand_data[4]
        p5 = hand_data[5]
        p8 = hand_data[8]
        p67 = (hand_data[6]-hand_data[7])/2
        p1819 = (hand_data[18]-hand_data[19])/2
        hslide = np.linalg.norm(p4-p8)/np.linalg.norm(p8-p5)
        vslide = np.linalg.norm(p4-p67)/np.linalg.norm(p1819-p67)
        print(hslide,vslide)
        return np.array([[hslide,vslide]])

    def to_wrist_coordinate(self, hand_data, num):
        v05 = hand_data[5]-hand_data[0]
        v017 = hand_data[17]-hand_data[0]
        normal = self.normalize(np.cross(v017,v05))
        x_axis = self.normalize(v05)
        y_axis = self.normalize(normal)
        z_axis = self.normalize(np.cross(x_axis,y_axis))
        for i in range(num):
            v = np.copy(hand_data[i])
            hand_data[i, 0] = np.dot(v,x_axis)
            hand_data[i, 1] = np.dot(v,y_axis)
            hand_data[i, 2] = np.dot(v,z_axis)
        self.axis = np.array([x_axis,y_axis,z_axis])
        return hand_data

    def to_palm_coordinate(self, hand_data, num):
        v517 = hand_data[17]-hand_data[5]
        v912 = hand_data[12]-hand_data[9]
        x_axis = self.normalize(v517)
        y_axis = self.normalize(v912)
        z_axis = self.normalize(np.cross(x_axis,y_axis))
        for i in range(num):
            v = np.copy(hand_data[i])
            hand_data[i, 0] = np.dot(v,x_axis)
            hand_data[i, 1] = np.dot(v,y_axis)
            hand_data[i, 2] = np.dot(v,z_axis)
        self.axis = np.array([x_axis,y_axis,z_axis])
        return hand_data

    def to_global_coordinate(self, hand_data, num, x_axis, y_axis, z_axis):
        mat_inv = np.linalg.inv(self.axis)
        for i in range(num):
            v = np.copy(hand_data[i])
            hand_data[i, 0] = np.dot(v,mat_inv[0,:])
            hand_data[i, 1] = np.dot(v,mat_inv[1,:])
            hand_data[i, 2] = np.dot(v,mat_inv[2,:])
        return hand_data

    def get_image(self):
        height = int(self.image.shape[0]/2)
        width = int(self.image.shape[1]/2)
        image = cv2.resize(self.image,(width,height))
        cv2.imshow('MediaPipe Hands', image) # cannot use this line in MacOS, because MacOS does not allow UI control in the sub-thread
        # cv2.setWindowProperty('MediaPipe Hands', cv2.WND_PROP_TOPMOST, 0)
        mkey = cv2.waitKey(1)
        if mkey & 0xFF == ord('q'):
            return None
        return self.image

    def capture_data(self):
        with self.mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened() and not self.stopThread:
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                self.hand_data = np.zeros((2, 21, 3))
                if results.multi_hand_world_landmarks:
                    for h in range(len(results.multi_hand_world_landmarks)):
                        # Get the 3D position data from the landmarks
                        hand_landmarks = results.multi_hand_world_landmarks[h]
                        for i in range(len(hand_landmarks.landmark)):
                            marks = hand_landmarks.landmark[i]
                            self.hand_data[h, i] = np.array([marks.x, marks.z, marks.y])
                        self.hand_data[h] = self.to_wrist_coordinate(self.hand_data[h],len(hand_landmarks.landmark))
                        # self.hand_data[h] = self.to_global_coordinate(self.hand_data[h],len(hand_landmarks.landmark),self.axis[0],self.axis[1],self.axis[2])
                        #self.hand_data[h] -= self.hand_data[h, 0]
                        self.hand_data[:, :, 2] *= -1
                        # self.hand_data *= 500
                        # self.cursor = self.get_cursor()
                        # Performing drawing of the landmarks on the images
                        self.mp_drawing.draw_landmarks(
                            image,
                            results.multi_hand_landmarks[h],
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                        self.image = image
                        # Flip the image horizontally for a selfie-view display.
                        # cv2.imshow('MediaPipe Hands', image) # cannot use this line in MacOS, because MacOS does not allow UI control in the sub-thread
                        # #cv2.flip(image, 1))
                        # if cv2.waitKey(5) & 0xFF == 27:
                        #     break
                else:
                    self.image = image


if __name__ == "__main__":
    import time
    jc = JointCollector()
    jc.run()
    # jc.run_visualizer()
    # jc.init_save_video()
    while True:
        right_hand, left_hand, is_right_hand, is_left_hand = jc.get_hands()
        if is_right_hand:
            pass
        time.sleep(0.1)
        # img = jc.get_image()
        # if img is None:
        #     break
        # jc.save_video(img)
    #     jc.stop()
    #     time.sleep(2)
    #     jc.run()
    plt.show()
    jc.close()

    
