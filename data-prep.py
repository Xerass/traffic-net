"""
data-prep.py — UA-DETRAC Dataset Download & XML-to-YOLO Conversion

PURPOSE
- UA-DETRAC is not actually formatted in YOLO format, so we need some means to convert it to YOLO FORMAT

UA-DETRAC and YOLO represent object locations in fundamentally different ways:

    ┌──────────────────────────────────────────────────────────────────────┐
    │  UA-DETRAC (XML)                   │  YOLO (.txt)                   │
    │────────────────────────────────────│────────────────────────────────│
    │  One XML per VIDEO SEQUENCE        │  One .txt per IMAGE            │
    │  Bounding box: left, top, w, h     │  Bounding box: cx, cy, w, h   │
    │  Coordinates in PIXELS (absolute)  │  Coordinates NORMALIZED [0,1] │
    │  Origin: top-left corner of box    │  Origin: CENTER of box         │
    │  Vehicle type as string attribute  │  Vehicle type as integer ID    │
    └──────────────────────────────────────────────────────────────────────┘


    ### UA-DETRAC stores annotations like this (one file = one video sequence):

        <sequence name="MVI_20011">
          <frame num="1">
            <target_list>
              <target id="1">
                <box left="592.75" top="378.8" width="160.05" height="162.2"/>
                <attribute vehicle_type="car" truncation_ratio="0.1" .../>
              </target>
            </target_list>
          </frame>
          ...many more frames...
        </sequence>

    stores it a perframe basis, so we can easuly just find frame markers, seprate by targets then get the boxes inside

    ### YOLO expects one .txt file per image, where each line is:

        class_id  x_center  y_center  width  height

    all four coordinate values must be normalized to [0, 1] relative to the
    
    image dimensions (to be calculated like this):
        x_center = (left + width/2)  / image_width
        y_center = (top  + height/2) / image_height
        w_norm   = width  / image_width
        h_norm   = height / image_height

    this normalization makes the labels resolution-independent the same
    label works whether the image is 960x540 or resized to 640x640 (which yolo squishes them to).


#LAYOUT (WE ALSO GET TRAIN AND TEST SO WE DO EVERYTHING AT ONCE)
-------------------------
YOLO trainers (Ultralytics, etc.) expect a specific folder structure where
image and label directories mirror each other by filename:

    TrafficNet/data/
    ├── images/
    │   └── train/
    │       ├── MVI_20011_img00001.jpg    flat, prefixed with sequence name
    │       ├── MVI_20011_img00002.jpg
    │       └── ...
    ├── labels/
    │   └── train/
    │       ├── MVI_20011_img00001.txt    same basename, .txt extension
    │       ├── MVI_20011_img00002.txt
    │       └── ...
    └── data.yaml                         dataset config for YOLO

    Images are COPIED (not symlinked) for cloud portability (we plan to do full training on the cloud) the entire data/ folder can be zipped and uploaded without broken references.

    Filenames are prefixed with the sequence name (e.g. MVI_20011_) because every sequence contains img00001.jpg, img00002.jpg, etc. Without the prefix, files from different sequences would collide in the flat directory.


CLASS MAPPING:

UA-DETRAC annotates four vehicle types as strings. YOLO needs integer IDs:

    0 = car
    1 = bus
    2 = van
    3 = others   (trucks, tankers, etc.)
"""

import xml.etree.ElementTree as ET  # stdlib XML parser no extra dependencies
import os
import sys
import shutil
import random
from pathlib import Path

# PIL is used ONLY to read image dimensions for normalization
# we need the actual width/height of each image to convert pixel coordinates
# to the [0,1] range that YOLO requires
from PIL import Image

#CONFIG

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "data"

# UA-DETRAC vehicle types to YOLO class IDs
# This dict is the single source of truth for the mapping
CLASS_MAP = {
    "car":    0,
    "bus":    1,
    "van":    2,
    "others": 3,
}

# default image dimensions if PIL can't read the file (UA-DETRAC standard)
# This is a safety net, not the primary path we always try PIL first
DEFAULT_IMG_W = 960
DEFAULT_IMG_H = 540

# fraction of train sequences to hold out for validation
# UA-DETRAC only ships train (60 seqs) and test (40 seqs) annotations,
# so we carve val out of the train set to get 3 distinct splits
VAL_SPLIT_RATIO = 0.2



#cant do much if we dont download lmao
def download_dataset() -> Path:
    import kagglehub

    print("[1/4] Downloading UA-DETRAC dataset from Kaggle...")
    path = kagglehub.dataset_download("bratjay/ua-detrac-orig")
    print(f"      Dataset cached at: {path}")
    return Path(path)

#just walk the dataset and find image_dirs and annot_dirs and confirm that they do contain what is expected
def discover_layout(dataset_root: Path) -> dict:
    print("[2/4] Discovering dataset layout...")

    annotation_files = sorted(dataset_root.rglob("*.xml"))
    image_files = sorted(dataset_root.rglob("*.jpg"))

    if not annotation_files:
        sys.exit("ERROR: No XML annotation files found under " + str(dataset_root))
    if not image_files:
        sys.exit("ERROR: No JPG image files found under " + str(dataset_root))

    # === MAP BUILDING ==
    # build the map of sequence_name:directory containing that sequence's images
    # we identify sequences by their parent folder name (MVI_XXXXX)
    
    image_dirs: dict[str, Path] = {}
    for img_path in image_files:
        seq_name = img_path.parent.name          # e.g. "MVI_20011"
        if seq_name.startswith("MVI_"):
            image_dirs[seq_name] = img_path.parent

    # build the 2nd map sequence_name:XML annotation file.
    annotation_map: dict[str, Path] = {}
    for xml_path in annotation_files:
        # XML filename is formatted in the data as MVI_XXXXX.xml
        seq_name = xml_path.stem                  # e.g. "MVI_20011"
        annotation_map[seq_name] = xml_path

    # only keep sequences that have BOTH images and annotations
    valid_sequences = sorted(set(image_dirs.keys()) & set(annotation_map.keys()))

    print(f"      Found {len(annotation_files)} XML files, "
          f"{len(image_dirs)} image directories")
    print(f"      {len(valid_sequences)} sequences have both images & annotations")

    if not valid_sequences:
        sys.exit("ERROR: No sequences found with matching images and annotations.")

    return {
        "sequences": valid_sequences,
        "image_dirs": image_dirs,
        "annotation_map": annotation_map,
    }



#parsing time
def get_image_dimensions(image_path: Path) -> tuple[int, int]:

    #return width,height of image
    #PIL to read the images themselves
    #fallback on default sizes if not found

    try:
        with Image.open(image_path) as img:
            return img.size 
    except Exception:
        return (DEFAULT_IMG_W, DEFAULT_IMG_H)


# we find the image corresponding to the xml
# read dims
# extract specific bounding boxes from xml
# convert box left,top,w,h pixels to our normalized cx,cy,w,h
#create the .txt label file for yolo

#just returns a verification dict
def parse_and_convert(xml_path: Path,
                      image_dir: Path,
                      seq_name: str,
                      output_images_dir: Path,
                      output_labels_dir: Path) -> dict:


    # xml.etree.ElementTree is part of Python's stdlib, very useful
    tree = ET.parse(xml_path)
    root = tree.getroot()

    stats = {"images": 0, "labels": 0, "boxes": 0, "class_counts": {c: 0 for c in CLASS_MAP}}
    skipped_types = set()

    #iter over the frames
    for frame in root.iter("frame"):
        frame_num = int(frame.get("num"))

        # UA-DETRAC names frames as img00001.jpg, img00002.jpg ...
        img_filename = f"img{frame_num:05d}.jpg"
        src_img_path = image_dir / img_filename

        if not src_img_path.exists():
            continue  # skip frames with missing images (very rare but better safe than sorry)

        #get dims
        img_w, img_h = get_image_dimensions(src_img_path)

        # always prefix the filename with the sequence name to avoid collisions
        # without this, MVI_20011/img00001.jpg and MVI_20012/img00001.jpg would overwrite each other in the flat output directory
        out_basename = f"{seq_name}_{img_filename}"
        out_img_path = output_images_dir / out_basename
        out_lbl_path = output_labels_dir / out_basename.replace(".jpg", ".txt")

        #extract targets
        yolo_lines = []

        target_list = frame.find("target_list")
        if target_list is None:
            # Frame exists but has no vehicles , which is very valid since we dont have objects on the screen at all times
            out_lbl_path.write_text("")
            shutil.copy2(src_img_path, out_img_path)
            #just increment the image and label count and skip, but still copy them
            stats["images"] += 1
            stats["labels"] += 1
            continue
        

        # if the frame DOES contain objects we need to convert that
        for target in target_list.iter("target"):
            #if no bounding boxes found / left skip
            box = target.find("box")
            if box is None:
                continue

            # UA-DETRAC format: top-left corner + size
            left   = float(box.get("left"))
            top    = float(box.get("top"))
            width  = float(box.get("width"))
            height = float(box.get("height"))

            # vehichle type to clas ID conversion
            attr = target.find("attribute")
            if attr is not None:
                vtype = attr.get("vehicle_type", "others").lower()
            else:
                vtype = "others"

            if vtype not in CLASS_MAP:
                skipped_types.add(vtype)
                vtype = "others"  # fallback for unexpected types

            class_id = CLASS_MAP[vtype]

            #conversion to a center based normalized valus
            x_center = (left + width  / 2.0) / img_w
            y_center = (top  + height / 2.0) / img_h
            w_norm   = width  / img_w
            h_norm   = height / img_h

            # Clamp to [0, 1] some annotations may slightly exceed image bounds
            # due to truncation (vehicle partially off-screen). YOLO expects
            # all values within [0, 1], so we clip rather than discard.
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm   = max(0.0, min(1.0, w_norm))
            h_norm   = max(0.0, min(1.0, h_norm))

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} "
                              f"{w_norm:.6f} {h_norm:.6f}")

            stats["boxes"] += 1
            stats["class_counts"][vtype] += 1

        # Copy image first, then write label, if the copy fails we dont leave orphaned label files.
        shutil.copy2(src_img_path, out_img_path)
        out_lbl_path.write_text("\n".join(yolo_lines) + "\n" if yolo_lines else "")

        stats["images"] += 1
        stats["labels"] += 1

    if skipped_types:
        print(f"[WARN] Unknown vehicle types mapped to 'others': {skipped_types}")

    return stats



# generate the config file for yolo
# data.yaml supports train, val, and test fields
# train and val are required, test is optional (used by `yolo val` / `yolo predict`)
def generate_data_yaml(output_dir: Path, has_test: bool = False):
    #very simple to make, yolo has several things you can fill in the config file but all we need for this is just
    #path (root path of dataset)
    #train,val,test (dirs of train,val,test), nc (number of classes), names (class names)

    test_line = "\ntest: images/test" if has_test else ""

    yaml_content = f"""\

# data.yaml — UA-DETRAC dataset configuration for YOLO
# auto-generated by data-prep.py

path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val{test_line}

nc: {len(CLASS_MAP)}

names:
  0: car
  1: bus
  2: van
  3: others
"""
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"data.yaml written to {yaml_path}")


# sanity checks on a single split
def verify_split(output_dir: Path, split_name: str):
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name

    if not images_dir.exists():
        return

    image_files = sorted(images_dir.glob("*.jpg"))
    label_files = sorted(labels_dir.glob("*.txt"))

    print(f"\n  [{split_name}]")
    print(f"    Images: {len(image_files)}")
    print(f"    Labels: {len(label_files)}")

    if len(image_files) != len(label_files):
        print("    [WARN] Image/label count mismatch!")

    # spot check random label files
    sample_size = min(10, len(label_files))
    if sample_size == 0:
        return
    samples = random.sample(label_files, sample_size)
    errors = 0
    
    #ensure things like label path names are correct, class ids are correct, and bounding box values are correct
    for lbl_path in samples:
        lines = lbl_path.read_text().strip().split("\n")
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[ERR] {lbl_path.name}: expected 5 columns, got {len(parts)}")
                errors += 1
                continue

            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            if cls_id not in CLASS_MAP.values():
                print(f"[ERR] {lbl_path.name}: unknown class ID {cls_id}")
                errors += 1

            for val in coords:
                if val < 0.0 or val > 1.0:
                    print(f"[ERR] {lbl_path.name}: value {val} out of [0,1]")
                    errors += 1

    if errors == 0:
        print(f"    Spot-checked {sample_size} label files -- all OK")
    else:
        print(f"    Spot-checked {sample_size} label files -- {errors} errors found!")

    #get class distributions for the splits
    full_counts = {v: 0 for v in CLASS_MAP.values()}
    total_boxes = 0
    for lbl_path in label_files:
        for line in lbl_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            cls_id = int(line.split()[0])
            if cls_id in full_counts:
                full_counts[cls_id] += 1
                total_boxes += 1

    #just to verify how skewed classes may be (may prompt a reshuffle if so)
    print(f"Class distribution:")
    for name, cid in CLASS_MAP.items():
        count = full_counts[cid]
        pct = (count / total_boxes * 100) if total_boxes > 0 else 0
        print(f"{cid} ({name:>6s}): {count:>8,d}  ({pct:5.1f}%)")
    print(f"      {'':>9s} Total: {total_boxes:>8,d} boxes")



def verify_output(output_dir: Path):
    print("\n[VERIFY] Running sanity checks...")
    for split in ["train", "val", "test"]:
        verify_split(output_dir, split)


#main tasks
def process_split(split_name: str, layout: dict):
    #convert a specified split into YOLO format
    sequences = layout["sequences"]
    if not sequences:
        print(f"\n  [{split_name}] No sequences found, skipping.")
        return None

    out_img_dir = OUTPUT_DIR / "images" / split_name
    out_lbl_dir = OUTPUT_DIR / "labels" / split_name

    # skip if this split was already processed (has images)
    if out_img_dir.exists() and any(out_img_dir.glob("*.jpg")):
        existing_count = len(list(out_img_dir.glob("*.jpg")))
        print(f"\n  [{split_name}] Already processed ({existing_count} images found), skipping.")
        return None

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Converting {len(sequences)} sequences -> YOLO ({split_name})...")

    total_stats = {"images": 0, "labels": 0, "boxes": 0,
                   "class_counts": {c: 0 for c in CLASS_MAP}}

    for i, seq_name in enumerate(sequences, 1):
        xml_path  = layout["annotation_map"][seq_name]
        image_dir = layout["image_dirs"][seq_name]

        print(f"      [{i}/{len(sequences)}] {seq_name}...", end=" ", flush=True)

        stats = parse_and_convert(
            xml_path=xml_path,
            image_dir=image_dir,
            seq_name=seq_name,
            output_images_dir=out_img_dir,
            output_labels_dir=out_lbl_dir,
        )

        print(f"{stats['images']} images, {stats['boxes']} boxes")

        total_stats["images"] += stats["images"]
        total_stats["labels"] += stats["labels"]
        total_stats["boxes"]  += stats["boxes"]
        for c in CLASS_MAP:
            total_stats["class_counts"][c] += stats["class_counts"][c]

    print(f"\n      {split_name} totals: {total_stats['images']} images, "
          f"{total_stats['boxes']} boxes")
    return total_stats


def main():

    dataset_root = download_dataset()
    layout = discover_layout(dataset_root)
    all_sequences = layout["sequences"]

    # Split train sequences into train + val 
    # UA-DETRAC only provides train annotations (60 seqs) and test annotations (40 seqs)
    # we split at the SEQUENCE level, not the image level, so that frames from the same video don't leak between train and val (they'd be nearly identical frames otherwise, which defeats the purpose of validation)
    random.seed(42)  # reproducible split
    shuffled = all_sequences.copy()
    random.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * VAL_SPLIT_RATIO))
    val_sequences   = shuffled[:val_count]
    train_sequences = shuffled[val_count:]

    print(f"\n[2/4] Splitting {len(all_sequences)} sequences -> "
          f"{len(train_sequences)} train, {len(val_sequences)} val")

    # build per-split layouts that reuse the same image_dirs / annotation_map
    def make_layout(seqs):
        return {
            "sequences": seqs,
            "image_dirs": layout["image_dirs"],
            "annotation_map": layout["annotation_map"],
        }

    # convert every split
    print("\n[3/4] Converting to YOLO format...")
    process_split("train", make_layout(train_sequences))
    process_split("val",   make_layout(val_sequences))
    # test sequences will be picked up if test annotation XMLs exist
    # in the downloaded dataset (separate from train annotations)

    # generate the yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    if yaml_path.exists():
        print("\n[4/4] data.yaml already exists, skipping.")
    else:
        print("\n[4/4] Generating data.yaml...")
        has_test = (OUTPUT_DIR / "images" / "test").exists()
        generate_data_yaml(OUTPUT_DIR, has_test=has_test)

    # verify
    verify_output(OUTPUT_DIR)

    print("\nDone! Dataset ready at:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
