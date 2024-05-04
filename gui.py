from tkinter import Tk, Canvas, Button, PhotoImage, filedialog
from PIL import Image, ImageTk
from pathlib import Path
import numpy as np
import tensorflow as tf
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tensorflow.keras.applications.mobilenet import preprocess_input # type: ignore

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\AadiJ\Projects\Pictionary\Pictionary\frame0")

global img_path

XLModel = tf.keras.models.load_model('models\mobilenet_doodle_recognition_8_cat_apr4_32x32_XL.h5')
categories = ['alarm clock', 'airplane', 'apple', 'banana', 'beach', 'bicycle', 'bridge', 'Eiffel Tower']
colors =     ["salmon",    "dodgerblue", "green",  "yellow", "tan",   "magenta", "slategrey", "teal"]
LModel = tf.keras.models.load_model('models\mobilenet_doodle_recognition_8_cat_apr4_32x32_large.h5')
MModel = tf.keras.models.load_model('models\mobilenet_doodle_recognition_8_cat_apr4_32x32_med.h5')
SModel = tf.keras.models.load_model('models\mobilenet_doodle_recognition_8_cat_apr4_32x32_small.h5')
XL_pred = None
L_pred = None
M_pred = None
S_pred = None

spacer = 200



def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def preprocess_image(img):
        #img = cv2.resize(img, (224, 224))
        #img = np.expand_dims(img, axis=0)

        #if len(img.shape) == 2:  # Grayscale image (single channel)
        #    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #change array to two dimenstions
        img = np.reshape(img, (28, 28)) 
        # Resize the image to the target size
        #img = cv2.resize(img, (224, 224))
        img = cv2.resize(img, (32, 32))
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)


        # Handle batch dimension (depending on your data structure)
        #if len(img.shape) == 3:  # Single image, add batch dimension
        #    img = np.expand_dims(img, axis=0)  # Add batch dimension for a single image

        
        img = preprocess_input(img)
        return img

def update_image_2(image_path):
    global img_path
    # Open the image file
    image = Image.open(image_path)

    # Resize the image to 350x350 with no antialiasing
    image = image.resize((350, 350), Image.NEAREST)

    # Convert image to PhotoImage
    photo_image = ImageTk.PhotoImage(image)

    # Update image_2
    canvas.itemconfig(image_2, image=photo_image)
    # Keep a reference to the PhotoImage object to prevent it from being garbage collected
    canvas.image = photo_image
    img_path = image_path

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        update_image_2(file_path)

def filter_data(sizes, categories, colors, threshold):
    filtered_sizes = [size for size in sizes if size >= threshold]
    filtered_categories = [category for size, category in zip(sizes, categories) if size >= threshold]
    filtered_colors = [color for size, color in zip(sizes, colors) if size >= threshold]
    return filtered_sizes, filtered_categories, filtered_colors

def detect():
    global p1, p2, p3, p4, final_pred_text, array_text
    # Your detect function code here
    print("Detect function called on " + img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    #img = cv2.bitwise_not(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the image to single channel if necessary
    #if len(img.shape) == 3:    
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


    # Step 2: Reverse the preprocessing steps
    # Remove extra dimensions
    #img = np.squeeze(img)
    img_array = np.array(img)
    #plt.imshow(img_array.squeeze())
    #img_array = np.reshape(img_array, (784)) 

    # Resize the image to its original size
    #img1 = cv2.resize(img_array, (32, 32))#, interpolation=cv2.INTER_AREA)
    #img1 = cv2.bitwise_not(img1)
    # Step 3: Convert the image to a NumPy array
    #img_array = np.array(img)
    #print(img_array)
    print(img_array.shape)
    img_array = np.expand_dims(img_array, axis=0)
    #img_array = np.array(preprocess_input(img_array))
    #img_array= np.expand_dims(img_array, axis=3)
    #img_array = np.repeat(img_array, 3, axis=3)
    img_array1 = preprocess_input(img_array)
    
    XL_pred = categories[np.argmax(XLModel.predict(img_array1))]
    L_pred = categories[np.argmax(LModel.predict(img_array1))]
    M_pred = categories[np.argmax(MModel.predict(img_array1))]
    S_pred = categories[np.argmax(SModel.predict(img_array1))]
    print("XL Model Prediction:", XL_pred)
    print("L Model Prediction:", L_pred)
    print("M Model Prediction:", M_pred)
    print("S Model Prediction:", S_pred)
    
    canvas.itemconfig(p4, text=XL_pred)
    canvas.itemconfig(p3, text=L_pred)
    canvas.itemconfig(p2, text=M_pred)
    canvas.itemconfig(p1, text=S_pred)
    canvas.itemconfig(array_text, text=img_array[0][0:32][0:32][0:32][5:20][0].transpose())
    
    pred_counts = {
        XL_pred: 0,
        L_pred: 0,
        M_pred: 0,
        S_pred: 0
    }
    
    for pred in [XL_pred, L_pred, M_pred, S_pred]:
        pred_counts[pred] += 1
    
    # Find the majority prediction
    majority_pred = max(pred_counts, key=pred_counts.get)
    
    # Check if majority prediction has more than 3/4 votes
    if pred_counts[majority_pred] >= 3:
        final_pred = majority_pred
    elif pred_counts[majority_pred] == 2:
        # Find the second most frequent prediction
        second_pred = max(pred_counts, key=pred_counts.get)
        second_pred = second_pred if second_pred != majority_pred else None
        
        final_pred = f"{majority_pred} / {second_pred}" if second_pred else majority_pred
    else:
        final_pred = XL_pred  # If all predictions are different, use XL_pred
    
    print("Final Prediction:", final_pred)
    
    canvas.itemconfig(final_pred_text, text=final_pred)
    
    s_pred_percentage = SModel.predict(img_array1)[0]
    m_pred_percentage = MModel.predict(img_array1)[0]
    l_pred_percentage = LModel.predict(img_array1)[0]
    xl_pred_percentage = XLModel.predict(img_array1)[0]
    
    # Extract percentages and store them in s_graph_vals
    s_graph_vals = [round(percentage*100, 1) for category, percentage in zip(categories, s_pred_percentage)]
    print("S Model Prediction Percentages:", s_graph_vals)
    m_graph_vals = [round(percentage*100, 1) for category, percentage in zip(categories, m_pred_percentage)]
    print("M Model Prediction Percentages:", m_graph_vals)
    l_graph_vals = [round(percentage*100, 1) for category, percentage in zip(categories, l_pred_percentage)]
    print("L Model Prediction Percentages:", l_graph_vals)
    xl_graph_vals = [round(percentage*100, 1) for category, percentage in zip(categories, xl_pred_percentage)]
    print("XL Model Prediction Percentages:", xl_graph_vals)
    
    filtered_sizes1, filtered_categories1, filtered_colors1 = filter_data(s_graph_vals, labels, colors, 1)
    ax_1.clear()  # Clear previous plot
    pie_1 = ax_1.pie(filtered_sizes1, labels=filtered_categories1, autopct='%1.1f%%', colors=filtered_colors1)  # Plot updated pie chart
    plt.setp(pie_1[1] + pie_1[2], color='white')  # Set text color to white
    plt.setp(pie_1[1] + pie_1[2], color='white', fontsize=8)
    
    canvas1.draw()
    
    filtered_sizes2, filtered_categories2, filtered_colors2 = filter_data(m_graph_vals, labels, colors, 1)
    ax_2.clear()  # Clear previous plot
    pie_2 = ax_2.pie(filtered_sizes2, labels=filtered_categories2, autopct='%1.1f%%', colors=filtered_colors2)  # Plot updated pie chart
    plt.setp(pie_2[1] + pie_2[2], color='white')  # Set text color to white
    plt.setp(pie_2[1] + pie_2[2], color='white', fontsize=8)
    
    canvas2.draw()
    
    filtered_sizes3, filtered_categories3, filtered_colors3 = filter_data(l_graph_vals, labels, colors, 1)
    ax_3.clear()  # Clear previous plot
    pie_3 = ax_3.pie(filtered_sizes3, labels=filtered_categories3, autopct='%1.1f%%', colors=filtered_colors3)  # Plot updated pie chart
    plt.setp(pie_3[1] + pie_3[2], color='white')  # Set text color to white
    plt.setp(pie_3[1] + pie_3[2], color='white', fontsize=8)
    
    canvas3.draw()
    
    filtered_sizes4, filtered_categories4, filtered_colors4 = filter_data(xl_graph_vals, labels, colors, 1)
    ax_4.clear()  # Clear previous plot
    pie_4 = ax_4.pie(filtered_sizes4, labels=filtered_categories4, autopct='%1.1f%%', colors=filtered_colors4)  # Plot updated pie chart
    plt.setp(pie_4[1] + pie_4[2], color='white')  # Set text color to white
    plt.setp(pie_4[1] + pie_4[2], color='white', fontsize=8)
    
    canvas4.draw()
    
    window.mainloop()
    
    
    
    
window = Tk()
img_path = "frame0/image_2.png"
window.geometry("1920x1440")
window.configure(bg="#262626")

sizes = [12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5]
labels = ['clock', 'airplane', 'apple', 'banana', 'beach', 'bicycle', 'bridge', 'tower']

fig_1 = Figure(figsize = (3.35, 3.3), facecolor="#262626")
ax_1 = fig_1.add_subplot()
pie_1 = ax_1.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.setp(pie_1[1] + pie_1[2], color='white', fontsize=8)

fig_2 = Figure(figsize = (3.35, 3.3), facecolor="#262626")
ax_2 = fig_2.add_subplot()
pie_2 = ax_2.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.setp(pie_2[1] + pie_2[2], color='white', fontsize=8)

fig_3 = Figure(figsize = (3.35, 3.3), facecolor="#262626")
ax_3 = fig_3.add_subplot()
pie_3 = ax_3.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.setp(pie_3[1] + pie_3[2], color='white', fontsize=8)

fig_4 = Figure(figsize = (3.35, 3.3), facecolor="#262626")
ax_4 = fig_4.add_subplot()
pie_4 = ax_4.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.setp(pie_4[1] + pie_4[2], color='white', fontsize=8)

# Set text color to white for both labels and percentages
for text in pie_1[1] + pie_1[2]:
    text.set_color('white')
    
for text in pie_2[1] + pie_2[2]:
    text.set_color('white')
    
for text in pie_3[1] + pie_3[2]:
    text.set_color('white')
    
for text in pie_4[1] + pie_4[2]:
    text.set_color('white')



canvas = Canvas(
    window,
    bg="#262626",
    height=1440,
    width=1920,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=0)


canvas.create_text(
    766.0,
    26.0,
    anchor="nw",
    text="ART INSPECTOR",
    fill="#FFFFFF",
    font=("InknutAntiqua SemiBold", 48 * -1)
)

#
categories_text = "Apple, Airplane, Banana, Beach, Bridge, Bicycle, Alarm Clock, Eiffel Tower"

canvas.create_text(
    960.0,
    1320.0,
    anchor="center",
    text=categories_text,
    fill="#FFFFFF",
    font=("InknutAntiqua SemiBold", 48 * -1)
)


#

canvas.create_text(
    836.0,
    291.0,
    anchor="nw",
    text="You drew:",
    fill="#FFFFFF",
    font=("Inder Regular", 48 * -1)
)

final_pred_text = canvas.create_text(
    960.0,
    394.0,
    anchor="center",
    text="Waiting...",
    fill="#FFFFFF",
    font=("Inder Regular", 48 * -1)
)

canvas.create_text(
    200.0,
    980.0 + spacer,
    anchor="nw",
    text="S (1K)",
    fill="#FFFFFF",
    font=("Inder Regular", 48 * -1)
)

p1 = canvas.create_text(
    200.0,
    555.0 + spacer,
    anchor="nw",
    text="P1",
    fill="#FFFFFF",
    font=("Inder Regular", 28 * -1)
)

p2 = canvas.create_text(
    590.0,
    555.0 + spacer,
    anchor="nw",
    text="P2",
    fill="#FFFFFF",
    font=("Inder Regular", 28 * -1)
)

p3 = canvas.create_text(
    980.0,
    555.0 + spacer,
    anchor="nw",
    text="P3",
    fill="#FFFFFF",
    font=("Inder Regular", 28 * -1)
)

p4 = canvas.create_text(
    1370.0,
    555.0 + spacer,
    anchor="nw",
    text="P4",
    fill="#FFFFFF",
    font=("Inder Regular", 28 * -1)
)

canvas.create_text(
    590.0,
    980.0 + spacer,
    anchor="nw",
    text="M (10K)",
    fill="#FFFFFF",
    font=("Inder Regular", 48 * -1)
)

canvas.create_text(
    980.0,
    980.0 + spacer,
    anchor="nw",
    text="L (50K)",
    fill="#FFFFFF",
    font=("Inder Regular", 48 * -1)
)

canvas.create_text(
    1370.0,
    980.0 + spacer,
    anchor="nw",
    text="XL (120K)",
    fill="#FFFFFF",
    font=("Inder Regular", 48 * -1)
)

button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
button_1 = Button(
    window,
    image=button_image_1,
    bg="#262626",
    borderwidth=0,
    highlightthickness=0,
    command=select_image,
    relief="flat"
)
button_1.place(x=28.0, y=28.0, width=50.0, height=50.0)

image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(1867.0, 53.0, image=image_image_1)

image_image_2 = PhotoImage(file=relative_to_assets("image_3.png"))
image_2 = canvas.create_image(325.0, 275.0, image=image_image_2)

image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(375.0, 805.0 + spacer, image=image_image_3)

image_image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(765.0, 805.0 + spacer, image=image_image_4)

image_image_5 = PhotoImage(file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(1155.0, 805.0 + spacer, image=image_image_5)

image_image_6 = PhotoImage(file=relative_to_assets("image_6.png"))
image_6 = canvas.create_image(1545.0, 805.0 + spacer, image=image_image_6)

array_text = canvas.create_text(
    1300.0,
    275.0,
    anchor="nw",
    text="",
    fill="#FFFFFF",
    font=("Inter", 15 * -1)
)

button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))
button_2 = Button(
    window,
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=detect,
    relief="flat"
)
button_2.place(x=807.0, y=162.0, width=305.0, height=97.0)

canvas1 = FigureCanvasTkAgg(figure=fig_1)
canvas1.get_tk_widget().place(x=206.5, y=636.2 + spacer) 
canvas1.draw()

canvas2 = FigureCanvasTkAgg(figure=fig_2)
canvas2.get_tk_widget().place(x=592.5, y=636.2 + spacer) 
canvas2.draw()

canvas3 = FigureCanvasTkAgg(figure=fig_3)
canvas3.get_tk_widget().place(x=984.5, y=636.2 + spacer) 
canvas3.draw()

canvas4 = FigureCanvasTkAgg(figure=fig_4)
canvas4.get_tk_widget().place(x=1377.5, y=636.2 + spacer) 
canvas4.draw()

window.resizable(False, False)
window.mainloop()
