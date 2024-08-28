from PIL import Image, ImageDraw
import os
import datetime
from tkinter import messagebox

# Ensure the 'visualizations' directory exists
os.makedirs("visualizations", exist_ok=True)

def generate_image_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"visualizations/canvas_image_{timestamp}.jpg"

def get_color(canvas, item, attribute, default="black"):
    color = canvas.itemcget(item, attribute)
    if color == "" or color.startswith("system"):
        return default
    return color

def save_canvas(canvas, total_width):
    """
    Save the given Tkinter canvas as an image file (JPEG format).

    The image will be saved in the 'visualizations' directory with a timestamped filename.
    """
    # Generate the filename
    filename = generate_image_filename()

    # Get the canvas dimensions
    canvas.update()
    # width = canvas.winfo_width()
    width = total_width
    height = canvas.winfo_height()

    # Create a new PIL image with a white background
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Draw the canvas content onto the PIL image
    for item in canvas.find_all():
        coords = canvas.coords(item)
        item_type = canvas.type(item)
        if item_type == "line":
            draw.line(coords, fill=get_color(canvas, item, "fill"))
        elif item_type == "rectangle":
            draw.rectangle(coords, outline=get_color(canvas, item, "outline"), fill=get_color(canvas, item, "fill"))
        elif item_type == "oval":
            draw.ellipse(coords, outline=get_color(canvas, item, "outline"), fill=get_color(canvas, item, "fill"))
        elif item_type == "text":
            draw.text(coords[:2], canvas.itemcget(item, "text"), fill=get_color(canvas, item, "fill"))

    # Save the image as a JPEG
    image.save(filename, "JPEG")
    
    print(f"Canvas saved as {filename} in the 'visualizations/' directory.")

    # Show a messagebox indicating success
    messagebox.showinfo("Save Successful", f"Canvas saved as {filename} in the 'visualizations/' directory.")