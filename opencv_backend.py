import os
import cv2
from tqdm import tqdm
import argparse
import numpy as np
from tools.interact_tools import SamControler
from tracker.base_tracker import BaseTracker


def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--mask_dir', type=str)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    args = parser.parse_args()
    return args


args = parse_augment()
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
xmem_checkpoint = "checkpoints/XMem-s012.pth"


class TrackingAnything:
    def __init__(self, sam_checkpoint, xmem_checkpoint, args):
        self.args = args
        self.sam_checkpoint = sam_checkpoint
        self.xmem_checkpoint = xmem_checkpoint
        self.samcontroler = SamControler(self.sam_checkpoint, args.sam_model_type, args.device)
        self.xmem = BaseTracker(self.xmem_checkpoint, device=args.device)

    def first_frame_click(self, image: np.ndarray, points: np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image

    def generator(self, images: list, template_mask: np.ndarray):

        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images)), desc="Tracking image"):
            if i == 0:
                mask, logit, painted_image = self.xmem.track(images[i], template_mask)
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)

            else:
                mask, logit, painted_image = self.xmem.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images


# initialize sam, xmem models
model = TrackingAnything(sam_checkpoint, xmem_checkpoint, args)


class VideoPlayer:
    def __init__(self, video_path, mask_dir):
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        self.mask_dir = mask_dir
        self.cap = cv2.VideoCapture(video_path)

        # Get total frames
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get fps
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.wait_time = int(1000 / fps)  # each frame's display time

        self.end_frame = self.total_frames - 1
        self.paused = False

        self.points = np.empty((0,2), int)
        self.labels = np.empty((0,), int)
        self.current_frame_number = 0

    def end_callback(self, pos):
        self.end_frame = pos

    def display(self, mask_path, display_frame):
        mask = np.load(mask_path)
        colored_mask = np.zeros_like(display_frame)
        colored_mask[mask == 1] = [0, 165, 255]  # set the mask to be orange
        display_frame = cv2.addWeighted(display_frame, 0.7, colored_mask, 0.3, 0)
        return display_frame

    def collect_frames_and_apply_function(self):
        frames = []
        for i in range(self.current_frame_number, self.end_frame + 1):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)

        # Display the processing message
        _, display_frame = self.cap.read()
        display_frame = self.frame.copy()
        text = 'Processing...'
        cv2.putText(display_frame, text, (display_frame.shape[1] // 2 - 60, display_frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video', display_frame)
        cv2.waitKey(1)  # update the window

        # Apply your custom function here
        masks, logits, painted_images = model.generator(frames, self.template_mask)

        # Save masks
        for i, mask in enumerate(masks):
            frame_number = self.current_frame_number + i
            filename = f"{frame_number:05}.npy"  # Format the filename to have 6 digits
            np.save(os.path.join(self.mask_dir, filename), mask)

        # clear GPU memory
        model.xmem.clear_memory()

        # Set the current frame to the start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        # Update the position slider to the start frame
        cv2.setTrackbarPos('Position', 'Video', self.current_frame_number)
        # Display the start frame
        ret, display_frame = self.cap.read()

        display_frame = self.frame.copy()
        mask_path = os.path.join(self.mask_dir, f"{self.current_frame_number:05}.npy")
        if os.path.exists(mask_path):
            display_frame = self.display(mask_path, display_frame)

        if ret:
            cv2.imshow('Video', display_frame)
        self.paused = True

    def click_event(self, event, x, y, flags, param):
        if self.paused:
            display_frame = self.frame.copy()
            if event == cv2.EVENT_LBUTTONDOWN:
                new_point = np.asarray([(x, y)])  # It's a 2D array
                self.points = np.concatenate((self.points, new_point))
                new_label = np.asarray([1])  # Note it's a 1D array
                self.labels = np.concatenate((self.labels, new_label))
                # self.points.append((x, y))
                # self.labels.append(1)
                cv2.circle(self.frame, (x, y), 3, (0, 255, 0), -1)
            elif event == cv2.EVENT_RBUTTONDOWN:
                new_point = np.asarray([(x, y)])  # It's a 2D array
                self.points = np.concatenate((self.points, new_point))
                new_label = np.asarray([0])  # Note it's a 1D array
                self.labels = np.concatenate((self.labels, new_label))
                cv2.circle(self.frame, (x, y), 3, (0, 0, 255), -1)
            if self.labels.shape[0] != 0:
                mask, logit, painted_image = model.first_frame_click(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), self.points, self.labels)
                filename = f"{self.current_frame_number:05}.npy"  # Format the filename to have 6 digits
                np.save(os.path.join(self.mask_dir, filename), mask)
                # painted_image = cv2.cvtColor(np.array(painted_image), cv2.COLOR_RGB2BGR)
                self.template_mask = mask
                # self.frame = painted_image

                mask_path = os.path.join(self.mask_dir, f"{self.current_frame_number:05}.npy")
                if os.path.exists(mask_path):
                    display_frame = self.display(mask_path, display_frame)
            cv2.imshow('Video', display_frame)
            print(self.points, self.labels)

    def clear_sam(self):
        self.points = np.empty((0, 2), int)
        self.labels = np.empty((0,), int)
        model.samcontroler.sam_controler.reset_image()

    def play(self):
        cv2.namedWindow('Video')
        cv2.setMouseCallback('Video', self.click_event)

        def position_callback(position):
            self.current_frame_number = position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            if self.paused:
                self.clear_sam()
                _, self.frame = self.cap.read()
                display_frame = self.frame.copy()
                mask_path = os.path.join(self.mask_dir, f"{self.current_frame_number:05}.npy")
                if os.path.exists(mask_path):
                    display_frame = self.display(mask_path, display_frame)
                cv2.imshow('Video', display_frame)

        cv2.createTrackbar('Position', 'Video', 0, self.total_frames - 1, position_callback)
        cv2.createTrackbar('End', 'Video', self.total_frames - 1, self.total_frames - 1, self.end_callback)

        while True:
            if not self.paused:
                self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                ret, self.frame = self.cap.read()

                # If video has ended, loop back to start
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                display_frame = self.frame.copy()
                mask_path = os.path.join(self.mask_dir, f"{self.current_frame_number:05}.npy")

                if os.path.exists(mask_path):
                    display_frame = self.display(mask_path, display_frame)

                # Show the frame
                cv2.imshow('Video', display_frame)

                # Update slider position
                cv2.setTrackbarPos('Position', 'Video', int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))

            key = cv2.waitKey(self.wait_time) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.paused = not self.paused
                self.clear_sam()
            elif key == ord('g'):  # Trigger frames collection when 'g' is pressed
                self.collect_frames_and_apply_function()
        self.cap.release()
        cv2.destroyAllWindows()


# python opencv_backend.py --video_path H:\data\holographic_body\single_video\video_2.mp4 --mask_dir H:\data\holographic_body\single_video\mask_2
video_path = args.video_path
mask_dir = args.mask_dir
player = VideoPlayer(video_path, mask_dir)
player.play()
