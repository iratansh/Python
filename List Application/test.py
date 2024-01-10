"""
This is a to-do list application that saves data using databases obtained by the sqlite3 library
Author: Ishaan Ratanshi
"""

import sqlite3 
import tkinter as tk
from tkinter import ttk
import time 

# Setup Window
window = tk.Tk()
window.title("To-Do List Application")
window.geometry("500x500")

# Icon Photo
window.iconbitmap('list_icon.png')

# Variable that keeps track of list item number
item_number = 1

# Create the grid layout
for i in range(2):
    window.grid_columnconfigure(i, weight=1)
for i in range(3):
    window.grid_rowconfigure(i, weight=1)

# Section 1: Displaying all the items in the list
display_frame = tk.Frame(window, relief="solid", bd=1)
display_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")

# Section 2: Adding and removing items in the list
add_remove_frame = tk.Frame(window, relief="solid", bd=1)
add_remove_frame.grid(row=0, column=1, sticky="nsew")

# Section 3: Buttons
buttons_frame = tk.Frame(window, relief="solid", bd=1)
buttons_frame.grid(row=1, column=1, rowspan=2, sticky="nsew")

# Function to add text to the to-do list
def add_list_item():
    global item_number
    displayed_text = user_input_add_text.get()
    if displayed_text:
        to_do_item = tk.Label(display_frame, text=f"{item_number}. {displayed_text}")
        to_do_item.pack(anchor="w")  # Align items to the left
        item_number += 1

# Function to remove text from the to-do list
def remove_list_item():
    global item_number
    try:
        line_number = int(user_input_remove_text.get())
        if line_number > 0:
            item_number -= 1
            widgets = display_frame.winfo_children()
            if line_number <= len(widgets):
                widgets[line_number - 1].destroy()
                for index, widget in enumerate(widgets[line_number:], start=line_number):
                    curr_text = widget.cget("text")
                    updated_number = int(curr_text.split(".")[0]) - 1
                    widget.configure(text=f"{updated_number}. {curr_text.split('.')[1]}")
            else:
                print("Line number exceeds number of items.")
    except ValueError:
        print("Please Enter a Valid Line Number. ")

# Store list in the database
def store_list_in_database():
    try:
        conn = sqlite3.connect("list.db")
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS list(number INTEGER PRIMARY KEY AUTOINCREMENT, list_item TEXT)""")

        # Clear existing items in the database table
        c.execute("DELETE FROM list")

        # Get all widgets in the display_frame
        widgets = display_frame.winfo_children()

        # Extract and store text from each widget in the database
        for widget in widgets:
            list_item = widget.cget("text").split(". ")[1]  # Extract text from Label widget
            c.execute("INSERT INTO list (list_item) VALUES (?)", (list_item,))
        
        conn.commit()
        conn.close()
        print("Items saved to the database.")
    except Exception as e:
        print(f"Error occurred: {e}")

# Load data from database
def retrieve_data_from_database():
    try:
        conn = sqlite3.connect("list.db")
        c = conn.cursor()

        # Retrieve data from the database
        c.execute("SELECT * FROM list")
        rows = c.fetchall()

        # Clear existing items in the display_frame
        for widget in display_frame.winfo_children():
            widget.destroy()

        # Display retrieved items in the to-do list frame
        for index, row in enumerate(rows, start=1):
            # Extract item from the database row
            list_item = row[1]  # Assuming the list_item is in the second column

            # Create Label widget to display the item in the to-do list frame
            to_do_item = tk.Label(display_frame, text=f"{index}. {list_item}")
            to_do_item.pack(anchor="w")  # Align items to the left
        conn.close()
        print("Data loaded to the to-do list.")
    except Exception as e:
        print(f"Error occurred: {e}")

# Quit Window
def quit_window():
    global pop_up_window
    pop_up_window = tk.Tk()
    pop_up_window.overrideredirect(1)
    pop_up_window.geometry("200x70+650+400")
    confirmation = tk.Label(pop_up_window, text="Exit?")
    confirmation.pack()
    yes_button = tk.Button(pop_up_window, text="Yes", cursor="hand2", bg="light blue", fg="red", width=7, command=exit).pack(pady=5, side="left")
    no_button = tk.Button(pop_up_window, text="No", cursor="hand2", bg="light blue", fg="red", width=7, command=no_selected).pack(pady=5, side="right")
    
# If user selects no on the quit window 
def no_selected():
    global pop_up_window
    time.sleep(1)
    pop_up_window.destroy()

# Exit out of App
def exit():
    global pop_up_window
    time.sleep(1)
    pop_up_window.destroy()
    window.destroy()

# Add things to the to-do list
add_text_label = tk.Label(add_remove_frame, text="Add To Listt:")
add_text_label.pack()
user_input_add_text = tk.Entry(add_remove_frame, width=40)
user_input_add_text.focus_set()
user_input_add_text.pack()

# Remove things from the to-do list
remove_text_label = tk.Label(add_remove_frame, text="Remove From List:")
remove_text_label.pack()
user_input_remove_text = tk.Entry(add_remove_frame, width=40)
user_input_remove_text.pack()

# Buttons to interact with the to-do list
ttk.Button(buttons_frame, text="Add", cursor="hand2", width=20, command=add_list_item).pack(pady=10)
ttk.Button(buttons_frame, text="Remove", cursor="hand2", width=20, command=remove_list_item).pack(pady=10)
ttk.Button(buttons_frame, text="Save", cursor="hand2", width=20, command=store_list_in_database).pack(pady=10)
ttk.Button(buttons_frame, text="Exit", cursor="hand2", width=20, command=quit_window).pack(pady=10)

# Load Saved Data
retrieve_data_from_database()
# Window Loop
window.mainloop()