from pathlib import Path
import json

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    # raise NotImplementedError("Not implemented")
    kart_objects = extract_frame_info  # keep imported name used elsewhere
    del kart_objects  # silence linters if needed

    from .generate_qa import extract_kart_objects, extract_track_info

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not karts:
        return []

    ego_kart = next(k for k in karts if k["is_center_kart"])
    ego_x, ego_y = ego_kart["center"]
    track_name = extract_track_info(info_path)

    captions = []
    captions.append(f"{ego_kart['kart_name']} is the ego car.")
    captions.append(f"There are {len(karts)} karts in the scenario.")
    captions.append(f"The track is {track_name}.")

    for kart in karts:
        if kart["is_center_kart"]:
            continue

        x, y = kart["center"]
        lr = "left" if x < ego_x else "right"
        fb = "in front of" if y < ego_y else "behind"
        captions.append(f"{kart['kart_name']} is {fb} and to the {lr} of the ego car.")

    return captions

def write_captions(split: str = "train"):
    """
    Generate *_captions.json files for every *_info.json file in data/<split>.
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

        all_captions = []

        for image_file in image_files:
            parts = image_file.stem.split("_")
            if len(parts) < 3:
                continue

            view_index = int(parts[1])
            captions = generate_caption(str(info_path), view_index)

            for caption in captions:
                all_captions.append(
                    {
                        "image_file": f"{split}/{image_file.name}",
                        "caption": caption,
                    }
                )

        out_file = info_path.parent / f"{base_name}_captions.json"
        with open(out_file, "w") as f:
            json.dump(all_captions, f, indent=2)

        total_written += len(all_captions)
        print(f"Wrote {out_file} with {len(all_captions)} captions")

    print(f"Done. Total captions written: {total_written}")


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "write": write_captions})


if __name__ == "__main__":
    main()
