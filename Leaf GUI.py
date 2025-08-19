import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the trained model
with open("model_new.json", "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_new.h5")

# Categories for classification
CATEGORIES = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", 
              "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust", "Corn_(maize)___healthy", 
              "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___healthy", 
              "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight", 
              "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold", 
              "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites_two-spotted_spider_mite", "Tomato___Target_Spot", 
              "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]

# Format category names
formatted_categories = {}
for category in CATEGORIES:
    plant, disease = category.split("___") if "___" in category else (category, "healthy")
    disease = disease.replace("_", " ").replace("(", "").replace(")", "").replace("  ", " ")
    formatted_categories[category] = (plant, disease)

# Function to preprocess image
def prepare(filepath):
    IMG_SIZE = 128
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    return new_array

# Function to browse and display image
def browse_image():
    global img_path
    img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if img_path:
        img = Image.open(img_path)
        img.thumbnail((250, 250))  # Maintain aspect ratio
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        result_label.config(text="")  # Clear previous result

# Function to classify image
def classify_image():
    if not img_path:
        result_label.config(text="Please select an image first!", fg="red", font=("Arial", 14, "bold"))
        return
    
    img_data = prepare(img_path)
    prediction = loaded_model.predict(img_data)
    predicted_class = np.argmax(prediction)
    
    # Get formatted plant name and disease
    plant, disease = formatted_categories[CATEGORIES[predicted_class]]
    result_text = f"Plant Name: {plant}, Disease: {disease}"
    
    result_label.config(text=result_text, fg="blue", font=("Arial", 16, "bold"))

# Create GUI
root = tk.Tk()
root.title("Leaf Disease Detection")
root.geometry("700x550")
root.configure(bg="white")

# Title Label
title_label = Label(root, text="Leaf Disease Detection", font=("Arial", 18, "bold"), bg="white", fg="green")
title_label.pack(pady=10)

# Image Display Area
frame = tk.Frame(root, width=250, height=250, bg="gray")
frame.pack(pady=10)
frame.pack_propagate(False)  # Prevents frame from resizing

image_label = Label(frame, bg="gray")
image_label.pack(fill="both", expand=True)

# Browse Button
browse_button = Button(root, text="Browse Image", font=("Arial", 14), command=browse_image, bg="lightblue")
browse_button.pack(pady=5)

# Classify Button
classify_button = Button(root, text="Classify Leaf Disease", font=("Arial", 14), command=classify_image, bg="orange")
classify_button.pack(pady=5)

# Result Label
result_label = Label(root, text="", font=("Arial", 16), bg="white")
result_label.pack(pady=10)

# Project Credit
credit_label = Label(root, text='Project by "Mohammed Abdul Aziz M"', font=("Arial", 10, "italic"), bg="white", fg="gray")
credit_label.pack(side="right", padx=10, pady=10)

root.mainloop()
