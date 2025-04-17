import random
import json
import os
from PIL import Image

canonical_map = {
    "dreadnought": "dreadnought", "dreadnought R": "dreadnought",
    "endurance": "endurance", "endurance R": "endurance",
    "prince_of_wales": "prince_of_wales", "prince_of_wales R": "prince_of_wales",
    "queen_elizabeth": "queen_elizabeth", "queen_elizabeth R": "queen_elizabeth",
    "victory": "victory", "victory R": "victory"
}

def is_overlapping(new_box, existing_boxes):
    return any(
        new_box[0] < box[2] and new_box[2] > box[0] and
        new_box[1] < box[3] and new_box[3] > box[1]
        for box in existing_boxes
    )

def overlay_images(background_path, overlay_names, overlay_sizes, output_path, canvas_size=(3850, 3850)):
    background = Image.open(background_path).convert("RGBA").resize(canvas_size)
    combined = background.copy()
    placed_boxes = []

    num_overlays = random.randint(1, 4)

    selected = []
    used_canonicals = set()

    #try placements until we get a non-conflicting selection
    shuffled = random.sample(overlay_names, len(overlay_names))
    for name in shuffled:
        canonical = canonical_map[name]
        if canonical not in used_canonicals:
            selected.append(name)
            used_canonicals.add(canonical)
            if len(selected) == num_overlays:
                break

    actual_labels = []

    for name in selected:
        try:
            overlay = Image.open(f"ship_images/{name}.png").convert("RGBA")
        except FileNotFoundError:
            print(f"[WARNING] Missing overlay image: {name}.png — skipping.")
            continue

        size = overlay_sizes[name]
        overlay = overlay.resize(size, resample=Image.LANCZOS)

        for _ in range(100):
            max_x = canvas_size[0] - size[0]
            max_y = canvas_size[1] - size[1]
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            new_box = (x, y, x + size[0], y + size[1])

            if not is_overlapping(new_box, placed_boxes):
                combined.paste(overlay, (x, y), overlay)
                placed_boxes.append(new_box)
                actual_labels.append(canonical_map[name])
                break
        else:
            print(f"[INFO] Could not place {name} without overlap after 100 attempts.")

    combined.save(output_path, format="PNG")
    return actual_labels

def generate_dataset(output_dir, num_images, overlay_names, overlay_sizes, background_path):
    os.makedirs(output_dir, exist_ok=True)
    labels = {}

    for i in range(num_images):
        output_file = os.path.join(output_dir, f"combined_{i}.png")
        used_labels = overlay_images(
            background_path=background_path,
            overlay_names=overlay_names,
            overlay_sizes=overlay_sizes,
            output_path=output_file
        )
        labels[f"combined_{i}.png"] = used_labels

        if i % 100 == 0:
            print(f"[{output_dir}] Generated {i} images...")

    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    print(f"[{output_dir}] Metadata saved.")


# Config
overlay_names = [
    "dreadnought", "endurance", "prince_of_wales", "queen_elizabeth", "victory",
    "dreadnought R", "endurance R", "prince_of_wales R", "queen_elizabeth R", "victory R"
]

overlay_sizes = {
    "dreadnought": (252, 1054), "endurance": (123, 624), "prince_of_wales": (511, 1836),
    "queen_elizabeth": (202, 1286), "victory": (108, 454),
    "dreadnought R": (1054, 252), "endurance R": (624, 123),
    "prince_of_wales R": (1836, 511), "queen_elizabeth R": (1286, 202), "victory R": (454, 108)
}

background_image = "square_grid.png"

# Run
generate_dataset("training_library", 2000, overlay_names, overlay_sizes, background_image)
generate_dataset("testing_library", 2000, overlay_names, overlay_sizes, background_image)
