SETTING UP
Step 1: Set up the app folder — DONE

Step 2: Download Kivy — DONE

Step 3: Set up the validation folder — DONE
Reminder:
When setting up verification in the Jupyter Notebook, we looped over several (around 50) images in the verification_images folder to perform verification for more accurate predictions.
Now, copy the application_data folder into the app directory.

Step 4: Copy/set up dependencies and the custom layer — DONE

Step 5: Bring over the .h5 model — DONE
Copy the siamesemodel.keras file from the root folder to the app folder.
This file contains the entire model (weights, architecture, etc.), so we need it to reload the model within the app.

BUILDING OUR TEMPLATE KIVY APP
Step 6: Import Kivy dependencies — DONE

Step 7: Build the layout — DONE

Step 8: Build the update function to convert the webcam image to texture and render it — DONE
This refreshes the webcam feed in the app.

Step 9: Build the preprocessing function — DONE
(Migrate the existing preprocessing function from the Jupyter Notebook.)

Step 10: Bring over the verification function — DONE

Step 11: Update the verification function to handle new paths and save the current frame — DONE

Step 12: Update the verification function to display "Verified" text — DONE

Step 13: Link the verification function to a button in the Kivy app — DONE
Note:

Fine-tuning was needed to get the verification function working correctly.

Testing and tuning are critical steps when building ML models.

Initially, the model verified incorrectly after just the first iteration.

In real-life projects, you would go back and fine-tune your model.

To improve the model further, we could add more example images (e.g., holding up the phone) and apply data augmentation before retraining.

Step 14: Set up logging — DONE

IMPROVEMENTS MADE TO THE MODEL
Data Augmentation:
Augmented the positive and anchor datasets to significantly increase the amount of training data (from ~300 images to ~3000).
(Data augmentation was added after creating the pos, neg, and anc directories.)

Improved Performance Monitoring:
Added precision and recall metrics to better monitor model performance.

