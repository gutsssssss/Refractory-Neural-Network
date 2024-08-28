import csv
import os
import glob
import datetime
import tkinter as tk
from tkinter import filedialog

# Ensure the 'configurations' directory exists
os.makedirs("configurations", exist_ok=True)

def generate_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"configurations/neural_net_config_{timestamp}.csv"

def save_inputs(entries, vars):
    filename = generate_filename()
    with open(filename, "w", newline='') as file:
        writer = csv.writer(file)
        # Save entries (text inputs)
        for key, entry in entries.items():
            writer.writerow([key, entry.get()])
        # Save vars (radio button selections, etc.)
        for key, var in vars.items():
            writer.writerow([key, var.get()])
    print(f"Inputs saved to {filename}!")

def load_inputs(entries, vars):
    # Open a file dialog to allow the user to select the CSV file
    filename = filedialog.askopenfilename(title="Select file", initialdir="configurations", filetypes=[("CSV files", "*.csv")])
    
    if not filename:
        print("No file selected.")
        return

    try:
        with open(filename, "r") as file:
            reader = csv.reader(file)
            data = {rows[0]: rows[1] for rows in reader}
        # Load values back into Entry widgets
        for key, value in data.items():
            if key in entries:
                entries[key].delete(0, tk.END)
                entries[key].insert(0, value)
            elif key in vars:
                vars[key].set(value)
        print(f"Inputs loaded from {filename}!")
    except FileNotFoundError:
        print("File not found!")
        
def load_most_recent_inputs(entries, vars):
    # Get a list of all CSV files matching the pattern in the configurations directory
    list_of_files = glob.glob("configurations/neural_net_config_*.csv")
    if not list_of_files:
        print("No files found.")
        return

    # Find the most recent file
    latest_file = max(list_of_files, key=os.path.getctime)

    try:
        with open(latest_file, "r") as file:
            reader = csv.reader(file)
            data = {rows[0]: rows[1] for rows in reader}

        # Load values back into Entry widgets
        for key, value in data.items():
            if key in entries:
                entries[key].delete(0, tk.END)
                entries[key].insert(0, value)
            elif key in vars:
                vars[key].set(value)
        print(f"Inputs loaded from {latest_file}!")
    except FileNotFoundError:
        print("File not found!")