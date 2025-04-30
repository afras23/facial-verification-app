# faceid.py
# This file builds the Kivy app and performs the facial verification logic

# --- Import Kivy dependencies ---

# Core Kivy classes for app structure and UI components
from kivy.app import App # Base App class to inherit from
from kivy.uix.boxlayout import BoxLayout # Organizes widgets vertically
from kivy.uix.image import Image # Displays the real-time webcam feed
from kivy.uix.button import Button # Button widget for triggering verification
from kivy.uix.label import Label # Text labels for status updates

# Additional Kivy modules
from kivy.clock import Clock #this allows us to make continuous updates to our app and get realtime feed from our kivy app - allows the loop to keep running in a way where we can get realtime updates
from kivy.graphics.texture import Texture # Allows converting webcam frames into Kivy textures 
from kivy.logger import Logger # Utility for logging internal information for debugging

# --- Import Other Dependencies ---

import cv2 # OpenCV for capturing webcam feed and image preprocessing
import tensorflow as tf # TensorFlow for loading and running the Siamese model
from layers import L1Dist # Custom distance layer used by our model 
import os # Filepath management 
import numpy as np # Numerical operations (especially array and tensor manipulation)

# --- Build the Kivy App Class ---

class CamApp(App):

    #Kivy App for real-time facial verification using a Siamese neural network.

    def build(self): 

        #Initialize the UI, load the model, and start the webcam capture. Called automatically when app.run() is executed.

         # 1. Setup the UI components: webcam display, verify button, verification status label
        self.web_cam = Image(size_hint = (1, 0.8)) # Webcam feed (80% vertical space)
        self.button = Button(text = 'Verify', on_press = self.verify, size_hint = (1, 0.1)) # Verify button
        self.verification_label = Label(text = 'Verification Uninitiated', size_hint = (1, 0.1)) # Initial status text
                                                                                                    # The app will have 3 statuses for the verification label: Verification Unititiated (the intitial state), Verified (person matches), Unverified (person not matched)  
        
        # 2. Organize the layout vertically
        layout = BoxLayout(orientation = 'vertical') 
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # 3. Load the pre-trained Siamese model
        self.model = tf.keras.models.load_model('siamesemodelv2.keras', custom_objects = {'L1Dist':L1Dist}) 
        
        # 4. Initialize the webcam
        self.capture = cv2.VideoCapture(0) # 0 selects the default camera
        Clock.schedule_interval(self.update, 1.0/33.0) # Refresh webcam feed ~33 times/second to try to mimic what we see with the human eye

        return layout
    
       

    def resize_image(self, frame):
    
    # Resize the incoming webcam frame while maintaining aspect ratio. Crops the frame to 250x250 centered.

        if not self.ret or frame is None or frame.size == 0:
            print("Error: Failed to capture frame")
            return None
        
        h, w, _ = frame.shape  # Get frame dimensions
        
        # First resize the image while preserving aspect ratio so that the smaller dimension is 250
        if h > w:
            # Width is smaller, so make width = 250 and scale height proportionally
            scale = 250 / w
            new_h = int(h * scale)
            resized = cv2.resize(frame, (250, new_h), interpolation=cv2.INTER_AREA)
            
            # Now crop the center portion to get a 250x250 square
            start_y = (new_h - 250) // 2
            cropped = resized[start_y:start_y+250, 0:250]
        else:
            # Height is smaller or equal, so make height = 250 and scale width proportionally
            scale = 250 / h
            new_w = int(w * scale)
            resized = cv2.resize(frame, (new_w, 250), interpolation=cv2.INTER_AREA)
            
            # Now crop the center portion to get a 250x250 square
            start_x = (new_w - 250) // 2
            cropped = resized[0:250, start_x:start_x+250]
        
        return cropped

    def update(self, *args):

        # Capture frames continuously from the webcam and display them in the app.
        # Runs at ~33fps via Kivy's Clock scheduler.
        ret, frame = self.capture.read()
        if not ret:
            Logger.error("Webcam capture failed.")
            return
        
        # Convert OpenCV image array to a texture for Kivy rendering
        buf = cv2.flip(frame, 0).tostring() # Flip frame vertically and create byte buffer for natural webcam orientation
        
         # Convert OpenCV frame to Kivy Texture
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
        img_texture.blit_buffer(buf, colorfmt = 'bgr', bufferfmt = 'ubyte')

        # Assign the texture to the webcam display
        self.web_cam.texture = img_texture

    
    def preprocess(self, file_path):
        
       # Preprocess an image file:
       # - Load file
       # - Decode (JPEG preferred, fallback to general decoder)
       # - Resize to (100,100)
       # - Normalize pixel values to [0,1]

        try:
            byte_img = tf.io.read_file(file_path) # Read the image file as raw bytes from the given file path 

            try:
                img = tf.io.decode_jpeg(byte_img)

                img = tf.image.resize(img, (100,100))

                img = img / 255.0 # Normalization

                return img
    
            except:
                # Try fallback to decode_image which handles multiple formats
                try:
                    img = tf.io.decode_image(byte_img, channels=3)
                    img = tf.image.resize(img, (100, 100))
                    img = img / 255.0
                    return img
                    
                except:
                    print(f"Error: Could not decode image at {file_path}")
                    return tf.zeros((100, 100, 3), dtype=tf.float32)

        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return tf.zeros((100, 100, 3), dtype=tf.float32)
        
    def verify(self, *args):

       # Verification process:
       # - Capture current frame
       # - Compare against stored verification images
       # - Aggregate predictions
       # - Update app with 'Verified' / 'Unverified'
       # - Log the metrics

        # --- Threshold Settings ---
        detection_threshold = 0.5 # Single comparison minimum threshold
        verification_threshold = 0.8 # Overall verification threshold (how many comparisons must be positive)

        # --- Capture and Save the Input Image ---
        save_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = cv2.resize(frame, (250, 250))
        cv2.imwrite(save_path, frame)

        # --- Initialize Results ---
        results = []

        # --- Loop Through Stored Verification Images ---
        for image in os.listdir(os.path.join('application_data', 'verification_images')): 
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            # Predict similarity using Siamese model
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis = 1)))
            results.append(result)
            # Using model.predict to make a prediction on a single sample.
            # np.expand_dims adds an extra dimension to match the model's expected input shape.

        # --- Aggregate and Interpret Results ---

        detection = np.sum(np.array(results) > detection_threshold) # Number of predictions above detection threshold
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) # Proportion of positives

        verified = verification > verification_threshold # Final verification result

        # --- Update UI ---
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        # --- Log Metrics ---
        Logger.info(results)
        Logger.info(detection) 
        Logger.info(verification)
        Logger.info(verified)


        return results, verified 

# Run the App
if __name__ == '__main__': 
    CamApp().run() 