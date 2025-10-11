import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import json
from pathlib import Path
from ultralytics import YOLO


class YOLOLabelTool:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Labeling Tool")
        self.root.geometry("1200x800")

        # Data storage
        self.image_list = []
        self.current_image_idx = None  # Track current image index
        self.current_image_path = None
        self.current_image = None
        self.current_photo = None
        self.image_dir = None
        self.output_dir = None
        self.scale_factor = 1.0
        self.display_width = 0
        self.display_height = 0

        # Annotation data
        self.annotations = []  # List of boxes with keypoints
        self.current_box = None  # Box being drawn
        self.drag_start = None
        self.dragging_handle = None  # Which handle is being dragged
        self.selected_box_idx = None

        # Keypoint settings
        self.num_keypoint_classes = 0  # Default for COCO pose (can be changed)

        # Key counter for unique naming
        self.key_counter = self.load_key_counter()

        # YOLO model
        self.model = None
        self.model_path = None
        self.model_names = {}  # Store class names from model
        self.inference_params = {
            'conf': 0.5,
            'iou': 0.4,
            'show': False,
            'save': False
        }

        self.setup_ui()

    def load_key_counter(self):
        """Load the last used key counter from file"""
        counter_file = Path("yolo_gui/key_counter.json")
        if counter_file.exists():
            with open(counter_file, 'r') as f:
                data = json.load(f)
                return data.get('counter', 0)
        return 0

    def save_key_counter(self):
        """Save the key counter to file"""
        counter_file = Path("yolo_gui/key_counter.json")
        counter_file.parent.mkdir(exist_ok=True)
        with open(counter_file, 'w') as f:
            json.dump({'counter': self.key_counter}, f)

    def get_class_color(self, class_id):
        """Get color for a given class ID"""
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta',
                  'orange', 'purple', 'pink', 'lime', 'navy', 'teal']
        return colors[class_id % len(colors)]

    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left side - Controls
        left_frame = tk.Frame(main_frame, width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        left_frame.pack_propagate(False)

        # Middle - Canvas for image
        middle_frame = tk.Frame(main_frame)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(middle_frame, bg='gray', cursor='cross')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Motion>', self.on_mouse_move)

        # Right side - Image list
        right_frame = tk.Frame(main_frame, width=200)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        right_frame.pack_propagate(False)

        list_frame = tk.LabelFrame(right_frame, text="Images", padx=5, pady=5)
        list_frame.pack(fill=tk.BOTH, expand=True)

        list_scroll = tk.Scrollbar(list_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_listbox = tk.Listbox(list_frame, yscrollcommand=list_scroll.set)
        self.image_listbox.pack(fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.image_listbox.yview)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        # Mode selection
        mode_frame = tk.LabelFrame(left_frame, text="Annotation Mode", padx=5, pady=5)
        mode_frame.pack(fill=tk.X, pady=5)

        self.mode_var = tk.StringVar(value="box")
        mode_dropdown = ttk.Combobox(mode_frame, textvariable=self.mode_var,
                                      values=["box", "keypoint"], state="readonly")
        mode_dropdown.pack(fill=tk.X)
        mode_dropdown.bind('<<ComboboxSelected>>', self.on_mode_change)

        # Class/Label input
        label_frame = tk.LabelFrame(left_frame, text="Class Settings", padx=5, pady=5)
        label_frame.pack(fill=tk.X, pady=5)

        tk.Label(label_frame, text="Box/Keypoint Class:").pack(anchor=tk.W)
        self.class_var = tk.StringVar(value="0")
        class_entry = tk.Entry(label_frame, textvariable=self.class_var)
        class_entry.pack(fill=tk.X, pady=(0, 5))

        tk.Label(label_frame, text="Num Keypoint Classes:").pack(anchor=tk.W)
        self.num_kp_classes_var = tk.StringVar(value="0")
        num_kp_entry = tk.Entry(label_frame, textvariable=self.num_kp_classes_var)
        num_kp_entry.pack(fill=tk.X)

        # Directory controls
        dir_frame = tk.LabelFrame(left_frame, text="Directory", padx=5, pady=5)
        dir_frame.pack(fill=tk.X, pady=5)

        btn_load_dir = tk.Button(dir_frame, text="Load Image Directory",
                                 command=self.load_directory)
        btn_load_dir.pack(fill=tk.X, pady=2)

        btn_set_output = tk.Button(dir_frame, text="Set Output Directory",
                                   command=self.set_output_directory)
        btn_set_output.pack(fill=tk.X, pady=2)

        # YOLO Model controls
        model_frame = tk.LabelFrame(left_frame, text="YOLO Model", padx=5, pady=5)
        model_frame.pack(fill=tk.X, pady=5)

        btn_load_model = tk.Button(model_frame, text="Load Model",
                                   command=self.load_model, bg='lightblue')
        btn_load_model.pack(fill=tk.X, pady=2)

        btn_inference = tk.Button(model_frame, text="Run Inference",
                                 command=self.run_inference, bg='lightgreen')
        btn_inference.pack(fill=tk.X, pady=2)

        btn_inference_settings = tk.Button(model_frame, text="Inference Settings",
                                          command=self.open_inference_settings, bg='lightyellow')
        btn_inference_settings.pack(fill=tk.X, pady=2)

        # Action buttons
        action_frame = tk.Frame(left_frame)
        action_frame.pack(fill=tk.X, pady=5)

        btn_clear = tk.Button(action_frame, text="Clear Annotations",
                             command=self.clear_annotations, bg='orange')
        btn_clear.pack(fill=tk.X, pady=2)

        btn_delete = tk.Button(action_frame, text="Delete Selected Box",
                              command=self.delete_selected, bg='yellow')
        btn_delete.pack(fill=tk.X, pady=2)

        btn_clear_kps = tk.Button(action_frame, text="Clear Box Keypoints",
                                 command=self.clear_box_keypoints, bg='lightyellow')
        btn_clear_kps.pack(fill=tk.X, pady=2)

        btn_save = tk.Button(action_frame, text="Save & Next",
                            command=self.save_and_next, bg='lightgreen')
        btn_save.pack(fill=tk.X, pady=2)

        # Status bar
        self.status_var = tk.StringVar(value="Ready. Load a directory to start.")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                            relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_directory(self):
        """Load images from selected directory"""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if not directory:
            return

        self.image_dir = Path(directory)
        self.image_list = []

        # Load all image files
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_list.extend(list(self.image_dir.glob(ext)))
            self.image_list.extend(list(self.image_dir.glob(ext.upper())))

        self.image_list.sort()

        # Update listbox
        self.image_listbox.delete(0, tk.END)
        for img_path in self.image_list:
            self.image_listbox.insert(tk.END, img_path.name)

        self.status_var.set(f"Loaded {len(self.image_list)} images from {directory}")

    def set_output_directory(self):
        """Set output directory for labeled data"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = Path(directory)
            self.output_dir.mkdir(exist_ok=True)
            self.status_var.set(f"Output directory: {directory}")

    def on_image_select(self, event):
        """Handle image selection from list"""
        selection = self.image_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        self.current_image_idx = idx
        self.current_image_path = self.image_list[idx]
        self.load_image()

    def load_image(self):
        """Load and display selected image"""
        if not self.current_image_path:
            return

        # Load image
        self.current_image = Image.open(self.current_image_path)

        # Clear annotations
        self.annotations = []
        self.current_box = None
        self.selected_box_idx = None

        # Display image
        self.display_image()
        self.status_var.set(f"Loaded: {self.current_image_path.name}")

    def display_image(self):
        """Display image on canvas with current annotations"""
        if not self.current_image:
            return

        # Calculate scaling to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, self.display_image)
            return

        img_width, img_height = self.current_image.size

        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.scale_factor = min(scale_x, scale_y, 1.0)

        self.display_width = int(img_width * self.scale_factor)
        self.display_height = int(img_height * self.scale_factor)

        # Resize image
        display_img = self.current_image.resize((self.display_width, self.display_height),
                                                Image.Resampling.LANCZOS)

        # Draw annotations on image
        display_img = display_img.copy()
        draw = ImageDraw.Draw(display_img)

        for i, ann in enumerate(self.annotations):
            if ann['type'] == 'box':
                x1, y1, x2, y2 = ann['coords']
                x1_s = int(x1 * self.scale_factor)
                y1_s = int(y1 * self.scale_factor)
                x2_s = int(x2 * self.scale_factor)
                y2_s = int(y2 * self.scale_factor)

                color = self.get_class_color(ann['class'])
                width = 3 if i == self.selected_box_idx else 2
                draw.rectangle([x1_s, y1_s, x2_s, y2_s], outline=color, width=width)

                # Draw label text above box (top left)
                # Use class name from model if available, otherwise show "Class X"
                if ann['class'] in self.model_names:
                    label_text = self.model_names[ann['class']]
                else:
                    label_text = f"Class {ann['class']}"
                text_bbox = draw.textbbox((0, 0), label_text)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_y = y1_s - text_height - 5
                if text_y < 0:
                    text_y = y1_s + 2
                draw.rectangle([x1_s, text_y, x1_s + text_width + 4, text_y + text_height + 4],
                             fill=color)
                draw.text((x1_s + 2, text_y + 2), label_text, fill='white')

                # Draw keypoints
                keypoints = ann.get('keypoints', [])
                for kp in keypoints:
                    kp_x, kp_y = kp['coords']
                    kp_x_s = int(kp_x * self.scale_factor)
                    kp_y_s = int(kp_y * self.scale_factor)
                    kp_color = self.get_class_color(kp['class'])
                    radius = 4

                    if kp['visible']:
                        # Visible keypoint - solid fill
                        draw.ellipse([kp_x_s-radius, kp_y_s-radius, kp_x_s+radius, kp_y_s+radius],
                                   fill=kp_color, outline='white', width=2)
                        # Draw keypoint class label
                        kp_label = f"{kp['class']}"
                        draw.text((kp_x_s + 6, kp_y_s - 6), kp_label, fill=kp_color)
                    else:
                        # Invisible keypoint - outlined with X
                        draw.ellipse([kp_x_s-radius, kp_y_s-radius, kp_x_s+radius, kp_y_s+radius],
                                   outline='gray', width=2)
                        # Draw X over invisible keypoint
                        draw.line([kp_x_s-radius, kp_y_s-radius, kp_x_s+radius, kp_y_s+radius], fill='gray', width=2)
                        draw.line([kp_x_s-radius, kp_y_s+radius, kp_x_s+radius, kp_y_s-radius], fill='gray', width=2)
                        # Draw keypoint class label in gray
                        kp_label = f"{kp['class']}"
                        draw.text((kp_x_s + 6, kp_y_s - 6), kp_label, fill='gray')

                # Draw handles for selected box
                if i == self.selected_box_idx:
                    handle_size = 6
                    # Corners
                    for x, y in [(x1_s, y1_s), (x2_s, y1_s), (x1_s, y2_s), (x2_s, y2_s)]:
                        draw.rectangle([x-handle_size, y-handle_size, x+handle_size, y+handle_size],
                                     fill='blue', outline='white')
                    # Edges
                    for x, y in [(x1_s, (y1_s+y2_s)//2), (x2_s, (y1_s+y2_s)//2),
                               ((x1_s+x2_s)//2, y1_s), ((x1_s+x2_s)//2, y2_s)]:
                        draw.rectangle([x-handle_size, y-handle_size, x+handle_size, y+handle_size],
                                     fill='yellow', outline='white')

        # Draw current box being drawn
        if self.current_box:
            # Convert canvas coordinates to display image coordinates
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            x_offset = (canvas_width - self.display_width) // 2
            y_offset = (canvas_height - self.display_height) // 2

            x1, y1, x2, y2 = self.current_box
            x1_img = x1 - x_offset
            y1_img = y1 - y_offset
            x2_img = x2 - x_offset
            y2_img = y2 - y_offset
            # Normalize coordinates to ensure x1 <= x2 and y1 <= y2
            draw.rectangle([min(x1_img, x2_img), min(y1_img, y2_img),
                          max(x1_img, x2_img), max(y1_img, y2_img)], outline='blue', width=2)

        # Convert to PhotoImage and display
        self.current_photo = ImageTk.PhotoImage(display_img)

        x_offset = (canvas_width - self.display_width) // 2
        y_offset = (canvas_height - self.display_height) // 2

        self.canvas.delete('all')
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.current_photo)

    def get_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image coordinates"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        x_offset = (canvas_width - self.display_width) // 2
        y_offset = (canvas_height - self.display_height) // 2

        img_x = (canvas_x - x_offset) / self.scale_factor
        img_y = (canvas_y - y_offset) / self.scale_factor

        return img_x, img_y

    def on_mouse_down(self, event):
        """Handle mouse button press"""
        if not self.current_image:
            return

        img_x, img_y = self.get_image_coords(event.x, event.y)

        # Check if within image bounds
        if img_x < 0 or img_y < 0 or img_x >= self.current_image.width or img_y >= self.current_image.height:
            return

        mode = self.mode_var.get()

        if mode == 'box':
            # Check if clicking on a handle of selected box
            if self.selected_box_idx is not None:
                handle = self.get_handle_at_position(img_x, img_y, self.selected_box_idx)
                if handle:
                    self.dragging_handle = handle
                    self.drag_start = (img_x, img_y)
                    return

            # Check if clicking on an existing box
            for i, ann in enumerate(self.annotations):
                if ann['type'] == 'box':
                    x1, y1, x2, y2 = ann['coords']
                    if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                        self.selected_box_idx = i
                        self.display_image()
                        return

            # Start drawing new box
            self.selected_box_idx = None
            self.drag_start = (event.x, event.y)
            self.current_box = [event.x, event.y, event.x, event.y]

        elif mode == 'keypoint':
            # Check if clicking on an existing box to select it
            clicked_box = False
            for i, ann in enumerate(self.annotations):
                if ann['type'] == 'box':
                    x1, y1, x2, y2 = ann['coords']
                    if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                        # Check if clicking on a keypoint first
                        keypoints = ann.get('keypoints', [])
                        clicked_kp = False
                        for idx, kp in enumerate(keypoints):
                            kp_x, kp_y = kp['coords']
                            distance = ((img_x - kp_x)**2 + (img_y - kp_y)**2)**0.5
                            if distance < 10 / self.scale_factor:  # Within 10 pixels
                                # Toggle keypoint visibility
                                keypoints[idx]['visible'] = 0 if keypoints[idx]['visible'] else 1
                                self.selected_box_idx = i
                                self.display_image()
                                self.status_var.set(f"Toggled keypoint {keypoints[idx]['class']} visibility")
                                clicked_kp = True
                                break

                        if not clicked_kp:
                            # Not clicking on keypoint, check if we should select box or add new keypoint
                            if self.selected_box_idx == i:
                                # Already selected, add new keypoint
                                try:
                                    class_id = int(self.class_var.get())
                                    if 'keypoints' not in ann:
                                        ann['keypoints'] = []
                                    ann['keypoints'].append({
                                        'class': class_id,
                                        'coords': (img_x, img_y),
                                        'visible': 1
                                    })
                                    self.display_image()
                                    self.status_var.set(f"Added keypoint class {class_id} at ({int(img_x)}, {int(img_y)})")
                                except ValueError:
                                    messagebox.showerror("Error", "Invalid class ID")
                            else:
                                # Select this box
                                self.selected_box_idx = i
                                self.display_image()
                                self.status_var.set(f"Selected box {i} (class {ann['class']}). Click inside to add keypoints.")
                        clicked_box = True
                        break

            if not clicked_box:
                # Didn't click on any box, start drawing new box
                self.selected_box_idx = None
                self.drag_start = (event.x, event.y)
                self.current_box = [event.x, event.y, event.x, event.y]
                self.status_var.set("Drawing new box...")

    def get_handle_at_position(self, img_x, img_y, box_idx):
        """Check if position is on a handle of the box"""
        if box_idx >= len(self.annotations):
            return None

        ann = self.annotations[box_idx]
        if ann['type'] != 'box':
            return None

        x1, y1, x2, y2 = ann['coords']
        threshold = 10 / self.scale_factor  # pixels threshold

        # Check corners
        if abs(img_x - x1) < threshold and abs(img_y - y1) < threshold:
            return 'tl'  # top-left
        if abs(img_x - x2) < threshold and abs(img_y - y1) < threshold:
            return 'tr'  # top-right
        if abs(img_x - x1) < threshold and abs(img_y - y2) < threshold:
            return 'bl'  # bottom-left
        if abs(img_x - x2) < threshold and abs(img_y - y2) < threshold:
            return 'br'  # bottom-right

        # Check edges
        if abs(img_x - x1) < threshold and y1 <= img_y <= y2:
            return 'l'  # left
        if abs(img_x - x2) < threshold and y1 <= img_y <= y2:
            return 'r'  # right
        if abs(img_y - y1) < threshold and x1 <= img_x <= x2:
            return 't'  # top
        if abs(img_y - y2) < threshold and x1 <= img_x <= x2:
            return 'b'  # bottom

        return None

    def on_mouse_drag(self, event):
        """Handle mouse drag"""
        if not self.current_image:
            return

        mode = self.mode_var.get()

        if mode == 'box':
            if self.dragging_handle and self.selected_box_idx is not None:
                # Adjust existing box
                img_x, img_y = self.get_image_coords(event.x, event.y)
                ann = self.annotations[self.selected_box_idx]
                x1, y1, x2, y2 = ann['coords']

                handle = self.dragging_handle
                if handle == 'tl':
                    x1, y1 = img_x, img_y
                elif handle == 'tr':
                    x2, y1 = img_x, img_y
                elif handle == 'bl':
                    x1, y2 = img_x, img_y
                elif handle == 'br':
                    x2, y2 = img_x, img_y
                elif handle == 'l':
                    x1 = img_x
                elif handle == 'r':
                    x2 = img_x
                elif handle == 't':
                    y1 = img_y
                elif handle == 'b':
                    y2 = img_y

                # Ensure x1 < x2 and y1 < y2
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1

                ann['coords'] = [x1, y1, x2, y2]
                self.display_image()

            elif self.current_box:
                # Update current box being drawn
                self.current_box[2] = event.x
                self.current_box[3] = event.y
                self.display_image()

        elif mode == 'keypoint':
            if self.current_box:
                # Update current box being drawn
                self.current_box[2] = event.x
                self.current_box[3] = event.y
                self.display_image()

    def on_mouse_up(self, event):
        """Handle mouse button release"""
        if not self.current_image:
            return

        mode = self.mode_var.get()

        if mode == 'box':
            if self.dragging_handle:
                self.dragging_handle = None
                self.drag_start = None

            elif self.current_box:
                # Finalize box
                x1_c, y1_c, x2_c, y2_c = self.current_box

                # Convert to image coordinates
                x1, y1 = self.get_image_coords(x1_c, y1_c)
                x2, y2 = self.get_image_coords(x2_c, y2_c)

                # Ensure x1 < x2 and y1 < y2
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1

                # Check minimum size
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    try:
                        class_id = int(self.class_var.get())
                        self.annotations.append({
                            'type': 'box',
                            'class': class_id,
                            'coords': [x1, y1, x2, y2],
                            'keypoints': []
                        })
                        self.status_var.set(f"Added box: class {class_id}")
                    except ValueError:
                        messagebox.showerror("Error", "Invalid class ID")

                self.current_box = None
                self.display_image()

        elif mode == 'keypoint':
            if self.current_box:
                # Finalize box
                x1_c, y1_c, x2_c, y2_c = self.current_box

                # Convert to image coordinates
                x1, y1 = self.get_image_coords(x1_c, y1_c)
                x2, y2 = self.get_image_coords(x2_c, y2_c)

                # Ensure x1 < x2 and y1 < y2
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1

                # Check minimum size
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    try:
                        class_id = int(self.class_var.get())
                        self.annotations.append({
                            'type': 'box',
                            'class': class_id,
                            'coords': [x1, y1, x2, y2],
                            'keypoints': []
                        })
                        # Auto-select the newly created box
                        self.selected_box_idx = len(self.annotations) - 1
                        self.status_var.set(f"Added box: class {class_id}. Click inside to add keypoints.")
                    except ValueError:
                        messagebox.showerror("Error", "Invalid class ID")

                self.current_box = None
                self.display_image()

    def on_mouse_move(self, event):
        """Handle mouse movement for cursor changes"""
        if not self.current_image or self.mode_var.get() != 'box':
            return

        if self.selected_box_idx is not None and not self.dragging_handle and not self.current_box:
            img_x, img_y = self.get_image_coords(event.x, event.y)
            handle = self.get_handle_at_position(img_x, img_y, self.selected_box_idx)

            if handle in ['tl', 'br']:
                self.canvas.config(cursor='top_left_corner')
            elif handle in ['tr', 'bl']:
                self.canvas.config(cursor='top_right_corner')
            elif handle in ['l', 'r']:
                self.canvas.config(cursor='sb_h_double_arrow')
            elif handle in ['t', 'b']:
                self.canvas.config(cursor='sb_v_double_arrow')
            else:
                self.canvas.config(cursor='cross')
        else:
            self.canvas.config(cursor='cross')

    def on_mode_change(self, event):
        """Handle annotation mode change"""
        mode = self.mode_var.get()
        if mode == 'box':
            self.status_var.set("Box mode: Drag to draw boxes. Click on box to select and resize.")
        elif mode == 'keypoint':
            self.status_var.set("Keypoint mode: Drag to draw box, click box to select, click inside to add keypoints. Click keypoint to toggle visibility.")
        self.selected_box_idx = None
        self.display_image()

    def clear_annotations(self):
        """Clear all annotations"""
        if messagebox.askyesno("Confirm", "Clear all annotations?"):
            self.annotations = []
            self.selected_box_idx = None
            self.display_image()
            self.status_var.set("Annotations cleared")

    def delete_selected(self):
        """Delete selected annotation"""
        if self.selected_box_idx is not None and self.selected_box_idx < len(self.annotations):
            self.annotations.pop(self.selected_box_idx)
            self.selected_box_idx = None
            self.display_image()
            self.status_var.set("Deleted selected annotation")
        else:
            self.status_var.set("No annotation selected")

    def clear_box_keypoints(self):
        """Clear all keypoints from selected box"""
        if self.selected_box_idx is not None and self.selected_box_idx < len(self.annotations):
            ann = self.annotations[self.selected_box_idx]
            if ann['type'] == 'box':
                ann['keypoints'] = []
                self.display_image()
                self.status_var.set("Cleared keypoints from selected box")
            else:
                self.status_var.set("Selected annotation is not a box")
        else:
            self.status_var.set("No box selected")

    def load_model(self):
        """Load YOLO model from file"""
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("Model files", "*.pt *.pth"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.model = YOLO(file_path)
                self.model_path = file_path
                # Store class names from model
                if hasattr(self.model, 'names'):
                    self.model_names = self.model.names
                self.status_var.set(f"Model loaded: {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.model = None
                self.model_path = None
                self.model_names = {}

    def run_inference(self):
        """Run YOLO inference on current image"""
        if not self.current_image_path:
            messagebox.showerror("Error", "No image loaded")
            return

        if not self.model:
            messagebox.showerror("Error", "No model loaded. Please load a model first.")
            return

        try:
            # Run inference
            results = self.model(
                str(self.current_image_path),
                conf=self.inference_params['conf'],
                iou=self.inference_params['iou'],
                save=self.inference_params['save'],
                verbose=False
            )[0]

            # Store class names from model if not already stored
            if hasattr(self.model, 'names') and not self.model_names:
                self.model_names = self.model.names

            # Get detected boxes
            boxes = results.boxes

            if len(boxes) == 0:
                self.status_var.set("No objects detected")
                return

            # Check if model has keypoints (pose model)
            has_keypoints = hasattr(results, 'keypoints') and results.keypoints is not None

            # Add detected boxes to annotations
            img_width, img_height = self.current_image.size

            for idx, box in enumerate(boxes):
                # Get box coordinates in xyxy format (absolute coordinates)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())

                # Create box annotation
                box_ann = {
                    'type': 'box',
                    'class': class_id,
                    'coords': [float(x1), float(y1), float(x2), float(y2)],
                    'keypoints': []
                }

                # Add keypoints if available
                if has_keypoints:
                    kp_data = results.keypoints.data[idx].cpu().numpy()  # Shape: (num_keypoints, 3) - x, y, conf
                    for kp_class, kp_point in enumerate(kp_data):
                        kp_x, kp_y, kp_conf = kp_point
                        # Consider keypoint visible if confidence > threshold
                        visible = 1 if kp_conf > 0.5 else 0
                        if visible:
                            box_ann['keypoints'].append({
                                'class': kp_class,
                                'coords': (float(kp_x), float(kp_y)),
                                'visible': visible
                            })

                # Add to annotations
                self.annotations.append(box_ann)

            self.display_image()
            if has_keypoints:
                self.status_var.set(f"Detected {len(boxes)} objects with keypoints")
            else:
                self.status_var.set(f"Detected {len(boxes)} objects")

        except Exception as e:
            messagebox.showerror("Error", f"Inference failed: {str(e)}")

    def open_inference_settings(self):
        """Open window to configure inference parameters"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Inference Settings")
        settings_window.geometry("300x230")
        settings_window.transient(self.root)
        settings_window.grab_set()

        # Confidence threshold
        conf_frame = tk.Frame(settings_window, padx=10, pady=5)
        conf_frame.pack(fill=tk.X)

        tk.Label(conf_frame, text="Confidence Threshold:").pack(side=tk.LEFT)
        conf_var = tk.StringVar(value=str(self.inference_params['conf']))
        conf_entry = tk.Entry(conf_frame, textvariable=conf_var, width=10)
        conf_entry.pack(side=tk.RIGHT)

        # IOU threshold
        iou_frame = tk.Frame(settings_window, padx=10, pady=5)
        iou_frame.pack(fill=tk.X)

        tk.Label(iou_frame, text="IOU Threshold:").pack(side=tk.LEFT)
        iou_var = tk.StringVar(value=str(self.inference_params['iou']))
        iou_entry = tk.Entry(iou_frame, textvariable=iou_var, width=10)
        iou_entry.pack(side=tk.RIGHT)

        # Show parameter
        show_frame = tk.Frame(settings_window, padx=10, pady=5)
        show_frame.pack(fill=tk.X)

        tk.Label(show_frame, text="Show Results:").pack(side=tk.LEFT)
        show_var = tk.BooleanVar(value=self.inference_params['show'])
        show_check = tk.Checkbutton(show_frame, variable=show_var)
        show_check.pack(side=tk.RIGHT)

        # Save parameter
        save_frame = tk.Frame(settings_window, padx=10, pady=5)
        save_frame.pack(fill=tk.X)

        tk.Label(save_frame, text="Save Results:").pack(side=tk.LEFT)
        save_var = tk.BooleanVar(value=self.inference_params['save'])
        save_check = tk.Checkbutton(save_frame, variable=save_var)
        save_check.pack(side=tk.RIGHT)

        # Buttons
        button_frame = tk.Frame(settings_window, padx=10, pady=10)
        button_frame.pack(fill=tk.X)

        def save_settings():
            try:
                conf = float(conf_var.get())
                iou = float(iou_var.get())
                if not (0 <= conf <= 1):
                    messagebox.showerror("Error", "Confidence must be between 0 and 1")
                    return
                if not (0 <= iou <= 1):
                    messagebox.showerror("Error", "IOU must be between 0 and 1")
                    return

                self.inference_params['conf'] = conf
                self.inference_params['iou'] = iou
                self.inference_params['show'] = show_var.get()
                self.inference_params['save'] = save_var.get()
                self.status_var.set("Inference settings updated")
                settings_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Invalid confidence or IOU value")

        def cancel():
            settings_window.destroy()

        tk.Button(button_frame, text="Save", command=save_settings, bg='lightgreen', width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=cancel, bg='lightcoral', width=10).pack(side=tk.RIGHT, padx=5)

    def save_and_next(self):
        """Save current annotations and move to next image"""
        if not self.current_image_path or not self.output_dir:
            messagebox.showerror("Error", "No image loaded or output directory not set")
            return

        if not self.annotations:
            if not messagebox.askyesno("Warning", "No annotations. Save anyway?"):
                return

        # Create subdirectories
        images_dir = self.output_dir / "images"
        labels_dir = self.output_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        # Generate unique key
        self.key_counter += 1
        keynum = self.key_counter

        # Get original filename without extension
        original_name = self.current_image_path.stem

        # Create new filenames
        img_filename = f"{original_name}_{keynum}{self.current_image_path.suffix}"
        label_filename = f"{original_name}_{keynum}.txt"

        # Save image
        img_output_path = images_dir / img_filename
        self.current_image.save(img_output_path)

        # Save label in YOLO format
        label_output_path = labels_dir / label_filename
        img_width, img_height = self.current_image.size

        # Get number of keypoint classes
        try:
            num_kp_classes = int(self.num_kp_classes_var.get())
        except ValueError:
            num_kp_classes = 0  # Default

        with open(label_output_path, 'w') as f:
            for ann in self.annotations:
                if ann['type'] == 'box':
                    # YOLOv11-Pose format: class x_center y_center width height kp1_x kp1_y kp1_v ...
                    x1, y1, x2, y2 = ann['coords']
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    line = f"{ann['class']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

                    # Create keypoint array indexed by class
                    kp_array = {}
                    for kp in ann.get('keypoints', []):
                        kp_array[kp['class']] = kp

                    # Output keypoints in order of class (0 to num_kp_classes-1)
                    for kp_class in range(num_kp_classes):
                        if kp_class in kp_array:
                            kp = kp_array[kp_class]
                            x_norm = kp['coords'][0] / img_width
                            y_norm = kp['coords'][1] / img_height
                            visible = kp['visible']
                            line += f" {x_norm:.6f} {y_norm:.6f} {visible}"
                        else:
                            # Keypoint not present, write as 0 0 0
                            line += " 0.000000 0.000000 0"

                    f.write(line + "\n")

        # Save counter
        self.save_key_counter()

        self.status_var.set(f"Saved: {img_filename} and {label_filename}")

        # Move to next image
        if self.current_image_idx is not None:
            next_idx = self.current_image_idx + 1
            if next_idx < len(self.image_list):
                self.image_listbox.selection_clear(0, tk.END)
                self.image_listbox.selection_set(next_idx)
                self.image_listbox.see(next_idx)
                self.current_image_idx = next_idx
                self.current_image_path = self.image_list[next_idx]
                self.load_image()
            else:
                messagebox.showinfo("Done", "All images labeled!")


def main():
    root = tk.Tk()
    app = YOLOLabelTool(root)
    root.mainloop()


if __name__ == '__main__':
    main()
