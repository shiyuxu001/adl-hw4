import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    # raise NotImplementedError("Not implemented")
    with open(info_path) as f:
        info = json.load(f)

    detections = info.get("detections", [])
    if view_index >= len(detections):
        return []

    frame_detections = detections[view_index]

    def lookup_kart_name(track_id: int) -> str:
        candidates = [
            info.get("kart_names"),
            info.get("karts"),
            info.get("kart_info"),
            info.get("players"),
            info.get("racers"),
        ]

        for container in candidates:
            if container is None:
                continue

            if isinstance(container, dict):
                if str(track_id) in container:
                    v = container[str(track_id)]
                    if isinstance(v, dict):
                        return v.get("name", f"kart {track_id}")
                    return str(v)
                if track_id in container:
                    v = container[track_id]
                    if isinstance(v, dict):
                        return v.get("name", f"kart {track_id}")
                    return str(v)

            if isinstance(container, list):
                for item in container:
                    if isinstance(item, dict):
                        item_id = item.get("id", item.get("track_id", item.get("instance_id")))
                        if item_id == track_id:
                            return item.get("name", item.get("kart_name", f"kart {track_id}"))
                    else:
                        # fallback: assume list is ordered by track_id
                        if 0 <= track_id < len(container):
                            return str(container[track_id])

        return f"kart {track_id}"

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    kart_objects = []
    image_center = (img_width / 2.0, img_height / 2.0)

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y

        box_w = x2_scaled - x1_scaled
        box_h = y2_scaled - y1_scaled

        if box_w < min_box_size or box_h < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        center_x = (x1_scaled + x2_scaled) / 2.0
        center_y = (y1_scaled + y2_scaled) / 2.0

        kart_objects.append(
            {
                "instance_id": track_id,
                "kart_name": lookup_kart_name(track_id),
                "center": (center_x, center_y),
                "bbox": (x1_scaled, y1_scaled, x2_scaled, y2_scaled),
                "distance_to_image_center": (center_x - image_center[0]) ** 2 + (center_y - image_center[1]) ** 2,
                "is_center_kart": False,
            }
        )

    if not kart_objects:
        return []

    ego_idx = min(range(len(kart_objects)), key=lambda i: kart_objects[i]["distance_to_image_center"])
    kart_objects[ego_idx]["is_center_kart"] = True

    return kart_objects    


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    # raise NotImplementedError("Not implemented")
    with open(info_path) as f:
        info = json.load(f)

    for key in ["track_name", "track", "course", "map"]:
        if key in info:
            value = info[key]
            if isinstance(value, dict):
                for subkey in ["name", "track_name"]:
                    if subkey in value:
                        return str(value[subkey])
            return str(value)

    metadata = info.get("metadata", {})
    for key in ["track_name", "track", "course", "map"]:
        if key in metadata:
            return str(metadata[key])

    return "unknown"


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """

    # raise NotImplementedError("Not implemented")

    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not kart_objects:
        return []

    ego_kart = next(k for k in kart_objects if k["is_center_kart"])
    ego_x, ego_y = ego_kart["center"]
    track_name = extract_track_info(info_path)

    qa_pairs = []

    def relative_lr(other_x: float) -> str:
        return "left" if other_x < ego_x else "right"

    def relative_fb(other_y: float) -> str:
        return "in front of" if other_y < ego_y else "behind"

    def relative_combo(other_x: float, other_y: float) -> str:
        lr = "left" if other_x < ego_x else "right"
        fb = "front" if other_y < ego_y else "behind"
        return f"{fb} {lr}"

    # 1. Ego car question
    # What kart is the ego car?
    qa_pairs.append(
        {
            "question": "What kart is the ego car?",
            "answer": ego_kart["kart_name"],
        }
    )

    # 2. Total karts question
    # How many karts are there in the scenario?
    qa_pairs.append(
        {
            "question": "How many karts are there in the scenario?",
            "answer": str(len(kart_objects)),
        }
    )

    # 3. Track information questions
    # What track is this?
    qa_pairs.append(
        {
            "question": "What track is this?",
            "answer": track_name,
        }
    )

    others = [k for k in kart_objects if not k["is_center_kart"]]

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?
    for kart in others:
        x, y = kart["center"]
        name = kart["kart_name"]

        qa_pairs.append(
            {
                "question": f"Is {name} to the left or right of the ego car?",
                "answer": relative_lr(x),
            }
        )
        qa_pairs.append(
            {
                "question": f"Is {name} in front of or behind the ego car?",
                "answer": relative_fb(y),
            }
        )
        qa_pairs.append(
            {
                "question": f"Where is {name} relative to the ego car?",
                "answer": relative_combo(x, y),
            }
        )

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?
    left_count = sum(k["center"][0] < ego_x for k in others)
    right_count = sum(k["center"][0] >= ego_x for k in others)
    front_count = sum(k["center"][1] < ego_y for k in others)
    behind_count = sum(k["center"][1] >= ego_y for k in others)

    qa_pairs.extend(
        [
            {
                "question": "How many karts are to the left of the ego car?",
                "answer": str(left_count),
            },
            {
                "question": "How many karts are to the right of the ego car?",
                "answer": str(right_count),
            },
            {
                "question": "How many karts are in front of the ego car?",
                "answer": str(front_count),
            },
            {
                "question": "How many karts are behind the ego car?",
                "answer": str(behind_count),
            },
        ]
    )

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)

'''
write QA pairs to `..._qa_pairs.json` file in `data/train/` for training
'''

def write_qa_pairs(split: str = "train"):
    """
    Generate *_qa_pairs.json files for every *_info.json file in data/<split>.
    """
    data_dir = Path(__file__).parent.parent / "data" / split

    info_files = sorted(data_dir.glob("*_info.json"))
    if not info_files:
        print(f"No *_info.json files found in {data_dir}")
        return

    total_written = 0

    for info_path in info_files:
        base_name = info_path.stem.replace("_info", "")
        image_files = sorted(info_path.parent.glob(f"{base_name}_*_im.jpg"))

        all_qa_pairs = []

        for image_file in image_files:
            parts = image_file.stem.split("_")
            if len(parts) < 3:
                continue

            view_index = int(parts[1])
            qa_pairs = generate_qa_pairs(str(info_path), view_index)

            for qa in qa_pairs:
                all_qa_pairs.append(
                    {
                        "image_file": f"{split}/{image_file.name}",
                        "question": qa["question"],
                        "answer": qa["answer"],
                    }
                )

        out_file = info_path.parent / f"{base_name}_qa_pairs.json"
        with open(out_file, "w") as f:
            json.dump(all_qa_pairs, f, indent=2)

        total_written += len(all_qa_pairs)
        print(f"Wrote {out_file} with {len(all_qa_pairs)} QA pairs")

    print(f"Done. Total QA pairs written: {total_written}")



"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs, "write": generate_qa_pairs})


if __name__ == "__main__":
    main()
