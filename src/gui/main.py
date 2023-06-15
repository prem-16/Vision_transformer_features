from tkinter import *
from PIL import Image

from PIL import ImageTk

from src.gui.helpers import get_image_list
from src.models.model_wrapper_list import MODEL_DICT

master = Tk()

# Dimensions of the window
width = 1200
height = 600
master.geometry("{}x{}".format(width, height))

image_directory = "images/test_images/"
image_files, image_dirs = get_image_list(image_directory)

left_image_id = 0
right_image_id = 0
current_model_id = 0

# Preset colours
WHITE = "#ffffff"
VERY_VERY_LIGHT_GREY = "#f8f8f8"
VERY_LIGHT_GREY = "#f0f0f0"
LIGHT_GREY = "#d0d0d0"

# Style settings
background_colour = WHITE
outer_left_frame_colour = background_colour
outer_right_frame_colour = VERY_LIGHT_GREY
inner_frame_colour = VERY_LIGHT_GREY
list_box_colour = VERY_VERY_LIGHT_GREY

# Set the background colour of the window
master.configure(bg=background_colour)

# Set the title of the window
master.title("Vision Transformer Features")

# Split the window horizontally into two frames
outer_frame_ratio = 0.3
outer_x_pad = 10
outer_y_pad = 10

left_frame_width = (width * outer_frame_ratio) - outer_x_pad
right_frame_width = (width * (1 - outer_frame_ratio)) - (outer_x_pad * 2)

left_frame_height = height - (outer_y_pad * 2)
right_frame_height = height - (outer_y_pad * 2)

# --- SETUP OUTER FRAMES --- #
left_frame = Frame(
    master,
    width=left_frame_width, height=left_frame_height,
    bg=outer_left_frame_colour
)
left_frame.grid(row=0, column=0, padx=(0, outer_x_pad), pady=outer_y_pad)
left_frame.grid_propagate(False)

right_frame = Frame(
    master,
    width=right_frame_width, height=right_frame_height,
    bg=outer_right_frame_colour
)
right_frame.grid(row=0, column=1, padx=0, pady=outer_y_pad)
right_frame.grid_propagate(False)

# --- SETUP LEFT FRAME --- #
inner_x_pad = outer_x_pad
inner_y_pad = outer_y_pad
left_frame_split_ratio = 0.3

# --- --- SETUP LIST FRAME --- #
list_frame_width = left_frame_width
list_frame_height = left_frame_height * left_frame_split_ratio
list_frame = Frame(
    left_frame,
    width=list_frame_width,
    height=list_frame_height,
    bg=background_colour
)
list_frame.grid(row=0, column=0, padx=0, pady=0)
list_frame.propagate(False)
# list_frame.grid_propagate(False)

# Split the list frame into two list boxes (one for each image).
list_box_width = left_frame_width - (inner_x_pad * 2)
list_box_height = list_frame_height - (inner_y_pad * 2)

left_list_box_outer = Frame(
    list_frame,
    width=list_box_width,
    height=list_box_height,
    bg=inner_frame_colour,
    padx=inner_x_pad,
    pady=inner_y_pad
)
left_list_box_outer.pack(expand=True, side=LEFT, fill=BOTH, padx=(0, inner_x_pad))
left_list_box = Listbox(left_list_box_outer, bg=list_box_colour)
left_list_box.pack(expand=True, fill=BOTH)
left_list_box.config(highlightbackground=LIGHT_GREY, highlightthickness=0)
# Add items to list
for i, file in enumerate(image_files):
    left_list_box.insert(END, f"{file}")
# Set default left list box selection
# left_list_box.select_set(0)


right_list_box_outer = Frame(
    list_frame,
    width=list_box_width,
    height=list_box_height,
    bg=inner_frame_colour,
    padx=inner_x_pad,
    pady=inner_y_pad
)
right_list_box_outer.pack(expand=True, side=RIGHT, fill=BOTH, padx=(0, inner_x_pad))
right_list_box = Listbox(right_list_box_outer, bg=list_box_colour)
right_list_box.pack(expand=True, fill=BOTH)
right_list_box.config(highlightbackground=LIGHT_GREY, highlightthickness=0)
# Add items to list
for i, file in enumerate(image_files):
    right_list_box.insert(END, f"{file}")
# Set default right list box selection
# right_list_box.select_set(0)


# --- --- SETUP SETTINGS FRAME --- #
settings_frame_width = left_frame_width
settings_frame_height = left_frame_height * (1 - left_frame_split_ratio)
settings_frame = Frame(
    left_frame,
    width=left_frame_width,
    height=settings_frame_height,
    bg=inner_frame_colour
)
settings_frame.grid(row=1, column=0, padx=inner_x_pad, pady=inner_y_pad)
settings_frame.propagate(False)
settings_inner_frame = Frame(
    settings_frame,
    width=settings_frame_width,
    height=settings_frame_height,
    bg=inner_frame_colour
)
settings_inner_frame.pack(expand=False, fill=X, padx=inner_x_pad, pady=inner_y_pad)

# Option to select model
model_label = Label(settings_inner_frame, text="Model:", bg=inner_frame_colour)
# Pack and align left center
model_label.pack()
# Dropdown menu to select model
model_options = list(MODEL_DICT.keys())
model_variable = StringVar(settings_inner_frame)
model_variable.set(model_options[current_model_id])
model_dropdown = OptionMenu(settings_inner_frame, model_variable, *model_options)
model_dropdown.config(bg=inner_frame_colour, highlightbackground=LIGHT_GREY, highlightthickness=0)
model_dropdown.pack()

# Create vertical space between
space_label = Label(settings_inner_frame, text="", bg=inner_frame_colour)
space_label.pack(pady=(10, 0))

# Model settings
model_settings_label = Label(settings_inner_frame, text="Model settings:", bg=inner_frame_colour)
model_settings_label.pack()

model_settings_frame = Frame(
    settings_inner_frame,
    width=settings_frame_width,
    height=settings_frame_height,
    bg=inner_frame_colour
)
model_settings_frame.pack(expand=False, fill=X, padx=inner_x_pad, pady=inner_y_pad)


def populate_model_settings(current_model_id):
    global model_settings_frame
    # Clear the frame
    for widget in model_settings_frame.winfo_children():
        widget.destroy()
    # Get the current model
    settings = MODEL_DICT[list(MODEL_DICT.keys())[current_model_id]].SETTINGS
    if settings is not None:
        # Populate the frame with the settings
        for setting_name, setting_dict in settings.items():
            # Create a label for the setting
            setting_label = Label(model_settings_frame, text=f"{str(setting_name).capitalize()}:", bg=inner_frame_colour)
            setting_label.pack(side=LEFT)

            if setting_dict["type"] == "slider":
                setting_content = Scale(
                    model_settings_frame,
                    from_=setting_dict["min"],
                    to=setting_dict["max"],
                    orient=HORIZONTAL,
                    bg=inner_frame_colour,
                    highlightbackground=LIGHT_GREY,
                    highlightthickness=0
                )
                setting_content.set(setting_dict["default"])
                setting_content.pack(side=RIGHT)
            else:
                # Create label
                setting_content_label = Label(model_settings_frame, text=f"Invalid type", bg=inner_frame_colour)
                setting_content_label.pack(side=RIGHT)
    else:
        # Create a label saying empty
        empty_label = Label(model_settings_frame, text="Empty", bg=inner_frame_colour)
        empty_label.pack()

populate_model_settings(current_model_id)

# On changing the model, update the settings
model_variable.trace(
    "w", lambda name, index, mode, sv=model_variable: populate_model_settings(model_options.index(sv.get()))
)


# --- SETUP RIGHT FRAME --- #
image_frame_width = (right_frame_width / 2) - inner_x_pad
image_frame_height = right_frame_height - (inner_y_pad * 2)

# --- --- SETUP LEFT IMAGE FRAME --- #

left_image_frame = Frame(
    right_frame,
    width=image_frame_width,
    height=image_frame_height,
    bg=inner_frame_colour
)
left_image_frame.pack(expand=True, side=LEFT, fill=BOTH, padx=inner_x_pad, pady=inner_y_pad)
left_image_frame.propagate(False)

left_image_container = None
left_image = None
left_image_canvas = None


# On clicking on the left_image_canvas (the image), draw a point on the image
def left_image_canvas_click(event):
    # Get the x and y coordinates of the click
    x = event.x
    y = event.y
    # Draw a point on the image
    left_image_canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red")


def set_left_image(image_id):
    # I don't like this, but it has to be like this.
    global left_image_container
    global left_image
    global left_image_canvas

    left_image_container = Frame(
        left_image_frame,
        width=image_frame_width,
        height=image_frame_height,
        bg=inner_frame_colour
    )
    left_image_container.pack(expand=True, fill=BOTH)

    # Create image and add to left_image_container. Set image width to image_frame_width
    pil_image = Image.open(image_dirs[image_id])
    # Resize image to fit image_frame_width but respect height ratio
    pil_image = pil_image.resize((int(image_frame_width), int(image_frame_width * pil_image.height / pil_image.width)))

    left_image = ImageTk.PhotoImage(pil_image)
    # left_image_label = Label(left_image_container, image=left_image)
    # left_image_label.pack(expand=True, fill=BOTH)

    # Create canvas with image as background
    left_image_canvas = Canvas(left_image_container, width=pil_image.width, height=pil_image.height)
    left_image_canvas.pack(expand=True, fill=BOTH)
    # left_image_label.create_image(0, 0, image=left_image, anchor=NW)
    # CENTER THE IMAGE
    left_image_canvas.create_image(image_frame_width / 2, image_frame_height / 2, image=left_image, anchor=CENTER)

    left_image_canvas.bind("<Button-1>", left_image_canvas_click)


# On left list box selection change, update left image
def left_list_box_selection_change(event):
    # Make sure the user is select and not deselecting
    if len(left_list_box.curselection()) == 0:
        return
    global left_image_id
    global left_image_container
    # Empty the left image container
    left_image_container.destroy()
    left_image_id = left_list_box.curselection()[0]
    set_left_image(left_image_id)


left_list_box.bind("<<ListboxSelect>>", left_list_box_selection_change)

set_left_image(left_image_id)


# --- --- SETUP RIGHT IMAGE FRAME --- #
right_image_frame = Frame(
    right_frame,
    width=image_frame_width,
    height=image_frame_height,
    bg=inner_frame_colour
)
right_image_frame.pack(expand=True, side=RIGHT, fill=BOTH, padx=(0, inner_x_pad), pady=inner_y_pad)

right_image_container = None
right_image = None
right_image_label = None


def set_right_image(image_id):
    # I don't like this, but it has to be like this.
    global right_image_container
    global right_image
    global right_image_label

    right_image_container = Frame(
        right_image_frame,
        width=image_frame_width,
        height=image_frame_height,
        bg=inner_frame_colour
    )
    right_image_container.pack(expand=True, fill=BOTH)

    # Create image and add to right_image_container. Set image width to image_frame_width
    pil_image = Image.open(image_dirs[image_id])
    # Resize image to fit image_frame_width but respect height ratio
    pil_image = pil_image.resize((int(image_frame_width), int(image_frame_width * pil_image.height / pil_image.width)))

    right_image = ImageTk.PhotoImage(pil_image)
    right_image_label = Label(right_image_container, image=right_image)
    right_image_label.pack(expand=True, fill=BOTH)


set_right_image(right_image_id)


# On right list box selection change, update right image
def right_list_box_selection_change(event):
    # Make sure the user is select and not deselecting
    if len(right_list_box.curselection()) == 0:
        return
    global right_image_id
    global right_image_container
    # Empty the right image container
    right_image_container.destroy()
    right_image_id = right_list_box.curselection()[0]
    set_right_image(right_image_id)


right_list_box.bind("<<ListboxSelect>>", right_list_box_selection_change)


mainloop()
