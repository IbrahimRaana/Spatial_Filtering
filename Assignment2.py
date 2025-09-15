import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tkinter import Tk, Frame, Label, filedialog, StringVar, IntVar, messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage
from skimage.util import random_noise

def apply_mean_filter(img, ksize=3):
    return cv2.blur(img, (ksize, ksize))

def apply_median_filter(img, ksize=3):
    return cv2.medianBlur(img, ksize)

def apply_mode_filter(img, ksize=3):
    return ndimage.generic_filter(img, lambda x: np.bincount(x.astype(int)).argmax(), size=ksize)

def apply_laplacian_filter(img):
    lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    lap_abs = cv2.convertScaleAbs(lap)
    return cv2.subtract(img, lap_abs)

def apply_sobel_filter(img):
    sobelx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)
    return cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

class SpatialFilteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🩻 Spatial Filtering Dashboard (Corrected Logic)")
        self.root.geometry("1200x800")
        self.root.configure(bg="#eaf2f8")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=8, background="#2980b9", foreground="white")
        style.map("TButton", background=[("active", "#3498db")])
        style.configure("TLabel", background="#eaf2f8", font=("Segoe UI", 11))
        style.configure("Header.TLabel", background="#17202a", foreground="white", font=("Segoe UI", 22, "bold"))
        self.original_img = None
        self.processing_base_img = None
        self.current_name = ""
        self.kernel_size_var = IntVar(value=3)
        self.results = {}
        self.filter_kernels = {
            "Laplacian Sharpening": "Kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]\n(Subtracted from Original)",
            "Sobel Edge Detection": "Sobel X: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]\nSobel Y: [[-1,-2,-1], [0,0,0], [1,2,1]]",
            "Mean Filter": lambda k: f"Kernel: 1/{k*k} * ones({k},{k})",
            "Median Filter": "Non-linear: Takes median value in KxK neighborhood.",
            "Mode Filter": "Non-linear: Takes most frequent value in KxK neighborhood."
        }
        header = Frame(root, bg="#17202a"); header.pack(fill="x")
        ttk.Label(header, text="Computer Vision: Spatial Filtering Analysis", style="Header.TLabel", padding=15).pack()

        controls = Frame(root, bg="#eaf2f8", pady=10); controls.pack(fill="x")

        col1_frame = Frame(controls, bg="#eaf2f8"); col1_frame.grid(row=0, column=0, padx=10, sticky="ns")
        ttk.Button(col1_frame, text="1. Upload Image", command=self.upload_image).pack(fill="x", pady=5)
        ttk.Button(col1_frame, text="Reset to Original", command=self.reset_image).pack(fill="x", pady=5)
        noise_frame = ttk.LabelFrame(col1_frame, text="2. (Optional) Add Noise", padding=10); noise_frame.pack(fill="x", pady=10)
        self.noise_var = StringVar(value="None")
        ttk.Radiobutton(noise_frame, text="Gaussian Noise", variable=self.noise_var, value="gaussian").pack(anchor="w")
        ttk.Radiobutton(noise_frame, text="Salt & Pepper Noise", variable=self.noise_var, value="s&p").pack(anchor="w")
        ttk.Button(noise_frame, text="Add Noise to Base Image", command=self.add_noise).pack(pady=5)

        col2_frame = Frame(controls, bg="#eaf2f8"); col2_frame.grid(row=0, column=1, padx=10, sticky="ns")
        filter_frame = ttk.LabelFrame(col2_frame, text="3. Select Filtering Technique", padding=10); filter_frame.pack(fill="x", pady=10)
        self.method_var = StringVar(value="Mean Filter")
        self.method_menu = ttk.Combobox(filter_frame, textvariable=self.method_var, state="readonly", values=["Mean Filter", "Median Filter", "Mode Filter", "Laplacian Sharpening", "Sobel Edge Detection"], font=("Segoe UI", 12), width=25); self.method_menu.pack(pady=5)
        self.method_menu.bind("<<ComboboxSelected>>", self.toggle_controls)

        col3_frame = Frame(controls, bg="#eaf2f8"); col3_frame.grid(row=0, column=2, padx=10, sticky="ns")
        params_frame = ttk.LabelFrame(col3_frame, text="4. Adjust Parameters", padding=10); params_frame.pack(fill="x", pady=10)
        self.kernel_frame = Frame(params_frame, bg="#eaf2f8")
        ttk.Label(self.kernel_frame, text="Kernel Size:", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, padx=5)
        self.kernel_label = ttk.Label(self.kernel_frame, text=f"{self.kernel_size_var.get()}x{self.kernel_size_var.get()}"); self.kernel_label.grid(row=0, column=2, padx=5)
        self.kernel_slider = ttk.Scale(self.kernel_frame, from_=3, to=15, variable=self.kernel_size_var, orient="horizontal", length=150, command=self.update_kernel_label); self.kernel_slider.grid(row=0, column=1, padx=5); self.kernel_slider.set(3)
        self.kernel_frame.pack()
        self.smooth_before_sharp_var = IntVar(value=0)
        self.smooth_before_sharp_check = ttk.Checkbutton(params_frame, text="Smooth before Sharpening?", variable=self.smooth_before_sharp_var); self.smooth_before_sharp_check.pack(pady=10)

        col4_frame = Frame(controls, bg="#eaf2f8"); col4_frame.grid(row=0, column=3, padx=20, sticky="ns")
        action_frame = ttk.LabelFrame(col4_frame, text="5. Actions", padding=10); action_frame.pack(fill="both", expand=True)
        ttk.Button(action_frame, text="Apply Filter", command=self.apply_method).pack(pady=5, fill="x", ipady=5)
        ttk.Button(action_frame, text="Save Analysis Report", command=self.save_report).pack(pady=5, fill="x", ipady=5)

        self.display = Frame(root, bg="white", relief="sunken", bd=2); self.display.pack(fill="both", expand=True, padx=15, pady=10)
        self.status = Label(root, text="Ready. Please upload an image to begin.", bd=1, relief="sunken", anchor="w", font=("Segoe UI", 10)); self.status.pack(side="bottom", fill="x")
        self.toggle_controls()

    def set_status(self, msg): self.status.config(text=msg)

    def toggle_controls(self, event=None):
        method = self.method_var.get()
        if method in ["Mean Filter", "Median Filter", "Mode Filter"]: self.kernel_frame.pack()
        else: self.kernel_frame.pack_forget()

    def update_kernel_label(self, val):
        ksize = round(float(val)); ksize += 1 if ksize % 2 == 0 else 0
        self.kernel_size_var.set(ksize); self.kernel_label.config(text=f"{ksize}x{ksize}")

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.bmp")])
        if not path: return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: messagebox.showerror("Error", "Could not read the image file."); return
        self.original_img = img
        self.current_name = Path(path).stem
        self.reset_image() 
        self.set_status(f"Loaded: {Path(path).name}. Base image is now the original.")

    def reset_image(self):
        if self.original_img is None: messagebox.showwarning("Warning", "No image loaded."); return
        self.processing_base_img = self.original_img.copy()
        self.results = {"Original": self.original_img}
        self.show_result("Original Image", self.original_img, self.processing_base_img)
        self.set_status("Image reset. All filters will now be applied to the original image.")

    def add_noise(self):
        if self.original_img is None: messagebox.showwarning("Warning", "Upload an image first."); return
        noise_type = self.noise_var.get()
        if noise_type == "gaussian":
            noisy_img = random_noise(self.original_img, mode='gaussian', var=0.01)
            title = "Base: Gaussian Noise"
        elif noise_type == "s&p":
            noisy_img = random_noise(self.original_img, mode='s&p', amount=0.05)
            title = "Base: Salt & Pepper Noise"
        else: messagebox.showwarning("Warning", "Select a noise type."); return
        self.processing_base_img = (noisy_img * 255).astype(np.uint8)
        self.results[title] = self.processing_base_img
        self.set_status(f"Applied {noise_type} noise.")
        self.show_result(title, self.original_img, self.processing_base_img)

    def apply_method(self):
        if self.processing_base_img is None: messagebox.showwarning("Warning", "Upload an image first."); return
        method = self.method_var.get()
        ksize = self.kernel_size_var.get()
        img_in = self.processing_base_img.copy()
        status_prefix = ""
        if self.smooth_before_sharp_var.get() and method in ["Laplacian Sharpening", "Sobel Edge Detection"]:
            img_in = apply_median_filter(img_in, ksize=5)
            status_prefix = "Smoothed then "
        if method == "Mean Filter": enhanced = apply_mean_filter(img_in, ksize)
        elif method == "Median Filter": enhanced = apply_median_filter(img_in, ksize)
        elif method == "Mode Filter": enhanced = apply_mode_filter(img_in, ksize)
        elif method == "Laplacian Sharpening": enhanced = apply_laplacian_filter(img_in)
        elif method == "Sobel Edge Detection": enhanced = apply_sobel_filter(img_in)
        else: return
        title = f"{status_prefix}{method}"
        self.results[title] = enhanced
        self.set_status(f"Applied: {title}")
        self.show_result(title, self.original_img, enhanced)

    def show_result(self, title, left_image, right_image):
        for w in self.display.winfo_children(): w.destroy()
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(title, fontsize=16, weight="bold", color="#17202a")
        axs[0].imshow(left_image, cmap="gray"); axs[0].set_title("Pristine Original"); axs[0].axis("off")
        axs[1].imshow(right_image, cmap="gray"); axs[1].set_title("Current Result"); axs[1].axis("off")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        canvas = FigureCanvasTkAgg(fig, master=self.display); canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def save_report(self):
        if len(self.results) <= 1: messagebox.showwarning("Warning", "No results to save."); return
        outdir = "results"; os.makedirs(outdir, exist_ok=True)
        pdf_path = os.path.join(outdir, f"{self.current_name}_Filtering_Report.pdf")
        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(8.5, 11)); fig.clf()
            fig.text(0.5, 0.8, "Spatial Filtering Analysis", ha="center", size=20, weight="bold")
            fig.text(0.5, 0.7, f"Image: {self.current_name}", ha="center", size=14)
            pdf.savefig(fig); plt.close(fig)
            for name, proc_img in self.results.items():
                if name == "Original": continue
                fig, axs = plt.subplots(1, 2, figsize=(8.5, 4.5))
                fig.suptitle(f"Analysis: {name}", fontsize=16, weight="bold")
                axs[0].imshow(self.original_img, cmap="gray"); axs[0].set_title("Original"); axs[0].axis("off")
                axs[1].imshow(proc_img, cmap="gray"); axs[1].set_title("Processed"); axs[1].axis("off")
                clean_name = name.replace("Smoothed then ", "")
                if clean_name in self.filter_kernels:
                    kernel_text = self.filter_kernels[clean_name]
                    if callable(kernel_text): kernel_text = kernel_text(self.kernel_size_var.get())
                    fig.text(0.5, 0.1, f"Filter Expression:\n{kernel_text}", ha="center", va="bottom", fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="#fdf5e6", alpha=0.7))
                plt.tight_layout(rect=[0, 0, 1, 0.90])
                pdf.savefig(fig); plt.close(fig)
        messagebox.showinfo("Success", f"PDF report saved to:\n{pdf_path}")
        self.set_status(f"Report saved: {pdf_path}")

if __name__ == "__main__":
    root = Tk()
    app = SpatialFilteringApp(root)
    root.mainloop()
