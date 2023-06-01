import os
import cv2
import argparse
import numpy as np
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import messagebox
from tools.interact_tools import (SamControler, mask_color, mask_alpha, contour_color, contour_width, point_color_ne,
                                  point_color_ps, point_alpha, point_radius, contour_color, contour_width)
from tools.painter import mask_painter, point_painter


def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--mask_dir', type=str)
    parser.add_argument('--sam_checkpoint', type=str, default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument('--xmem_checkpoint', type=str, default="checkpoints/XMem-s012.pth")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    args = parser.parse_args()
    return args


args = parse_augment()
sam_checkpoint = args.sam_checkpoint
xmem_checkpoint = args.xmem_checkpoint


class SegAnything:
    def __init__(self, sam_checkpoint, model_type=args.sam_model_type, device=args.device):
        self.sam_checkpoint = sam_checkpoint
        self.samcontroler = SamControler(self.sam_checkpoint, model_type, device)

    def seg_anything(self, image: np.ndarray, points: np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image


class Application:
    def __init__(self, master=None):
        self.master = master
        self.master.title("SAM Annotate")
        self.mask_dir = None
        self.image_loaded = False

        self.frame = tk.Frame(self.master)
        self.frame.pack(fill='both', expand=True)

        self.create_button_frame()
        self.create_canvas_frame()

        self.master.drop_target_register(DND_FILES)
        self.master.dnd_bind('<<Drop>>', self.drop)

        self.points = np.empty((0, 2))
        self.labels = np.empty(0)

        self.model = SegAnything(sam_checkpoint)

    def create_button_frame(self):
        self.button_frame = tk.Frame(self.frame)
        self.button_frame.pack(side='top', fill='x')

        self.folder_button = tk.Button(self.button_frame, text="Choose mask folder", command=self.choose_folder)
        self.folder_button.pack(side='left')

        self.folder_path_label = tk.Label(self.button_frame, text="")
        self.folder_path_label.pack(side='left')

        # Add a new button to clear all points and labels and reset the overlay image
        self.clear_button = tk.Button(self.button_frame, text="Clear all", command=self.clear_all)
        self.clear_button.pack(side='right')

    def create_canvas_frame(self):
        self.canvas_frame = tk.Frame(self.frame)
        self.canvas_frame.pack(side='top', fill='both', expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, width=800, height=600)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.create_text(400, 300, text="Drag in the image you want SAM to annotate", fill="black")

        self.overlay_image = self.canvas.create_image(0, 0, anchor='nw', tags="overlay")

    def drop(self, event):
        if self.mask_dir is None or self.mask_dir == "":
            messagebox.showwarning("Warning", "Please choose a mask folder first!")
            return
        filepath = event.data
        # Get the filename without extension
        self.filename, _ = os.path.splitext(os.path.basename(filepath))
        mask_file = os.path.join(self.mask_dir, self.filename + '.npy')
        if os.path.isfile(filepath):
            self.original_image = cv2.imread(filepath)
            self.model.samcontroler.sam_controler.reset_image()
            image = Image.open(filepath)
            photo = ImageTk.PhotoImage(image)
            self.canvas.config(width=image.width, height=image.height)
            self.master.geometry(f"{image.width}x{image.height}")
            self.canvas.create_image(0, 0, anchor='nw', image=photo)
            self.canvas.image = photo  # Keep a reference to the image
            self.image_loaded = True
            self.canvas.bind("<Button-1>", self.left_click_event)
            self.canvas.bind("<Button-3>", self.right_click_event)
            # Reset points and labels
            self.points = np.empty((0, 2))
            self.labels = np.empty(0)

            # Check for an existing mask file
            if os.path.exists(mask_file):
                mask = np.load(mask_file)
                painted_image = self.paint(mask)
                painted_image_tk = ImageTk.PhotoImage(painted_image)
                self.canvas.itemconfig(self.overlay_image, image=painted_image_tk)
                self.canvas.image = painted_image_tk

    def clear_all(self):
        self.model.samcontroler.sam_controler.reset_image()
        # Clear all points and labels
        self.points = np.empty((0, 2))
        self.labels = np.empty(0)
        # Reset overlay image to original image
        if self.image_loaded:
            original_image_pil = Image.fromarray(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
            original_image_tk = ImageTk.PhotoImage(original_image_pil)
            self.canvas.itemconfig(self.overlay_image, image=original_image_tk)
            self.canvas.image = original_image_tk  # Keep a reference to the image
        # Delete existing mask file, if any
        mask_file = os.path.join(self.mask_dir, self.filename + '.npy')
        if os.path.exists(mask_file):
            os.remove(mask_file)

    def left_click_event(self, event):
        if not self.image_loaded:
            return
        x = event.x
        y = event.y
        self.points = np.append(self.points, [[x, y]], axis=0)
        self.labels = np.append(self.labels, [1])
        self.update_mask()

    def right_click_event(self, event):
        if not self.image_loaded:
            return
        x = event.x
        y = event.y
        self.points = np.append(self.points, [[x, y]], axis=0)
        self.labels = np.append(self.labels, [0])
        self.update_mask()

    def choose_folder(self):
        dir_ = filedialog.askdirectory()
        if dir_:
            self.mask_dir = dir_
            self.folder_path_label.config(text=self.mask_dir)

    def paint(self, mask, mask_color=3):
        painted_image = mask_painter(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB), mask.astype('uint8'), mask_color, mask_alpha, contour_color, contour_width)
        painted_image = point_painter(painted_image, np.squeeze(self.points[np.argwhere(self.labels > 0)], axis=1),
                                      point_color_ne, point_alpha, point_radius, contour_color, contour_width)
        painted_image = point_painter(painted_image, np.squeeze(self.points[np.argwhere(self.labels < 1)], axis=1),
                                      point_color_ps, point_alpha, point_radius, contour_color, contour_width)
        painted_image = Image.fromarray(painted_image)
        # return PIL image
        return painted_image

    def update_mask(self):
        mask, logit, painted_image = self.model.seg_anything(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB), self.points.astype(np.int32), self.labels.astype(np.int32))
        overlay_tk = ImageTk.PhotoImage(painted_image)
        self.canvas.itemconfig(self.overlay_image, image=overlay_tk)
        self.canvas.image = overlay_tk  # Keep a reference to the image
        np.save(os.path.join(self.mask_dir, self.filename + '.npy'), mask)


root = TkinterDnD.Tk()
app = Application(master=root)
root.mainloop()
