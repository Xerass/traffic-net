"""
train.py, YOLOv8 Training Script for UA-DETRAC Vehicle Detection

PURPOSE
    Train a YOLOv8 model on the prepared UA-DETRAC dataset. Has the options to work locally and on the cloud (utilize the full dataset)
"""

from ultralytics import YOLO
from pathlib import Path
import torch
import os
import sys
import shutil
import time

try:
    import runpod
except ImportError:
    runpod = None


#script directory
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_YAML = SCRIPT_DIR / "data" / "data.yaml"

# LOCAL  = True: train on small portion of data (0.05%), sole purpose is to test
# LOCAL  = False : trains on the full data

LOCAL_MODE = False

# RunPod Auto-Shutdown (ensure 'pip install runpod' is run and RUNPOD_API_KEY env var is set)
AUTO_STOP_POD = True 
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")

#for local mode we use nano, in cloud we will use large model ~roughly 10x the params
if LOCAL_MODE:
    MODEL_VARIANT = "yolov8n.pt"  # nano for fast local iteration
else:
    MODEL_VARIANT = "yolov8s.pt"  # medium for cloud, adjust as needed


#training params
if LOCAL_MODE:
    # -Local settings, highly minimized
    EPOCHS       = 2        
    BATCH_SIZE   = 16      
    IMG_SIZE     = 640     
    FRACTION     = 0.005   
    WORKERS      = 4        
    PATIENCE     = 0        
else:
    # -Cloud settings, full dataset, more workers, more batches and epochs, more patience
    EPOCHS       = 100     
    BATCH_SIZE   = 64      # Increased to 64: the RTX 5090 has 32GB VRAM, can easily handle this!
    IMG_SIZE     = 640     
    FRACTION     = 1.0      
    WORKERS      = 8        
    PATIENCE     = 20       

#output and project settings
PROJECT_NAME = "runs"                       # output directory name
RUN_NAME     = "trafficnet_local" if LOCAL_MODE else "trafficnet_cloud"


def check_environment():
    #prints system info for double checking
    print("TrafficNet — YOLOv8 Training Script")
    print(f"Mode:          {'LOCAL (sample)' if LOCAL_MODE else 'CLOUD (full dataset)'}")
    print(f"Model:         {MODEL_VARIANT}")
    print(f"Epochs:        {EPOCHS}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Image size:    {IMG_SIZE}")
    print(f"Data fraction: {FRACTION * 100:.0f}%")
    print(f"Patience:      {PATIENCE}")
    print(f"Workers:       {WORKERS}")
    print()

    # PyTorch / CUDA info
    print(f"  PyTorch:       {torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU:           {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"CUDA:          {torch.version.cuda}")
    else:
        print("GPUNot available")
    print()

    # dataset verification
    if not DATA_YAML.exists():
        sys.exit(f"ERROR: data.yaml not found at {DATA_YAML}\n"
                 f"Run data-prep.py first to prepare the dataset.")

    # count images to show what we're working with
    train_dir = SCRIPT_DIR / "data" / "images" / "train"
    val_dir   = SCRIPT_DIR / "data" / "images" / "val"

    train_count = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
    val_count   = len(list(val_dir.glob("*.jpg")))   if val_dir.exists() else 0

    effective_train = int(train_count * FRACTION)
    print(f"Dataset:       {DATA_YAML}")
    print(f"Train images:  {train_count:,} total → {effective_train:,} used ({FRACTION*100:.0f}%)")
    print(f"Val images:    {val_count:,}")


#actual train of yolov8
def train():
    #check for env
    check_environment()

    #prep pretrained model for finetuning
    print(f"(1/3) Loading {MODEL_VARIANT}...")
    model = YOLO(MODEL_VARIANT)

    # Start training
    print(f"(2/3) Starting training...")
    print(f"Output → {SCRIPT_DIR / PROJECT_NAME / RUN_NAME}")
    print()

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        fraction=FRACTION,         # subsample the training data
        patience=PATIENCE,         # early stopping
        workers=WORKERS,
        project=str(SCRIPT_DIR / PROJECT_NAME),
        name=RUN_NAME,
        exist_ok=True,             # overwrite previous run with same name

        #Augmentations, adjusted for the traffic task
        hsv_h=0.015,               # hue shift, lighting variation cams often have varying lighting
        hsv_s=0.7,                 # saturation, impacts weather/time-of-day
        hsv_v=0.4,                 # brightness,  shadows/highlights
        degrees=0.0,               # no rotation (i doubt we'd see an upside down car, better save compute)
        translate=0.1,             # slight position shifts
        scale=0.5,                 # zoom variation (near/far vehicles simulation)
        fliplr=0.5,               # horizontal flip (roads are often symmetric so we need to account for it)
        flipud=0.0,               # no vertical flip (again, cars dont go upside down normally)
        mosaic=1.0,                # mosaic augmentation
        mixup=0.0,                 # no mixup for detection tasks

        #Logging
        verbose=True,
        plots=True,                # save training plots (loss, mAP, etc.)
    )

    #post-training info
    print("[3/3] Training complete!")

    #best model location
    best_model = SCRIPT_DIR / PROJECT_NAME / RUN_NAME / "weights" / "best.pt"
    last_model = SCRIPT_DIR / PROJECT_NAME / RUN_NAME / "weights" / "last.pt"
    print(f"Best model:  {best_model}")
    print(f"Last model:  {last_model}")
    print(f"Results:     {SCRIPT_DIR / PROJECT_NAME / RUN_NAME}")
    print()

    #run validation on best model
    print("Running final validation on best model...")
    best = YOLO(str(best_model))
    metrics = best.val(data=str(DATA_YAML))

    print("FINAL METRICS (best.pt on validation set)")

    #YOLO metrics are given as mean Average Precision
    #the number following refers to the intersection over union (how well the bounding box overlapped with truth bounding box)
    #mAP50 is mAP when 50% of the box is the only threshold (very forgiving, mainly just looks to see if it can detect, not how accurate)
    #mAP50-95 is mAP when 50% to 95% of the box is the threshold (very strict, looks for both detection and accuracy of label and box)
    print(f"mAP50:       {metrics.box.map50:.4f}")
    print(f"mAP50-95:    {metrics.box.map:.4f}")

    # per-class AP if available
    if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
        class_names = ["car", "bus", "van", "others"]
        print(f"\n  Per-class AP50:")
        for i, name in enumerate(class_names):
            if i < len(metrics.box.maps):
                print(f"    {name:<8s}  {metrics.box.maps[i]:.4f}")
    print("[Finished Training]")
    print(f'from ultralytics import YOLO')
    print(f'model = YOLO("{best_model}")')
    print(f'results = model.predict("your_image.jpg")')

    # RunPod Auto-Stop Logic
    if not LOCAL_MODE and AUTO_STOP_POD and os.environ.get("RUNPOD_POD_ID"):
        if runpod is None:
            print("\n[WARN] 'runpod' python package not installed. Cannot auto-stop pod.")
            print("Please run 'pip install runpod' next time.")
        elif not RUNPOD_API_KEY:
            print("\n[WARN] RUNPOD_API_KEY not set in environment variables. Cannot auto-stop pod.")
        else:
            pod_id = os.environ.get("RUNPOD_POD_ID")
            print(f"\n[INFO] Auto-stopping RunPod Pod {pod_id} in 60 seconds to save costs...")
            print("[INFO] (Your 'runs' directory will be safely preserved on the volume!)")
            runpod.api_key = RUNPOD_API_KEY
            # Wait a bit to ensure all buffers (like Weights & Biases if used) have flushed
            time.sleep(60) 
            try:
                runpod.stop_pod(pod_id)
                print("[INFO] Shutdown command sent successfully.")
            except Exception as e:
                print(f"[ERR] Failed to stop pod: {e}")


if __name__ == "__main__":
    train()
