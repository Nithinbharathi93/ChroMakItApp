from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
import cv2
import numpy as np
from kivy.graphics.texture import Texture

class VideoProcessor(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        
        self.capture = cv2.VideoCapture(0)  # Access the camera (change the parameter if using a different camera index)

        if not self.capture.isOpened():
            print("Error accessing the camera")
            return

        self.image = Image()
        layout.add_widget(self.image)

        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update at 30 fps

        return layout

    def process_frame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        frame[mask > 0] = (255, 255, 255)
        mask_inv = cv2.bitwise_not(mask)
        frame[mask_inv > 0] = (0, 0, 0)
        return frame

    def update(self, dt):
        ret, frame = self.capture.read()

        if ret:
            processed_frame = self.process_frame(frame)

            buf1 = cv2.flip(processed_frame, 0)
            buf2 = buf1.tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf2, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    VideoProcessor().run()












#Guys finally the code works...