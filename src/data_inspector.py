"""
This script examines the training images inside the train folder,
and the annotations for each image.

Creates a folder with the bounding boxes, keys, and labels displayed on each image.
Searches for differences in image names and JSON names.
Lists important infos about the images and the metadata in a TXT file.
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
import cv2


def analyze_and_report(img_dir: Path, ann_dir: Path, report_path: Path):
    """
    Compares image names with JSON names. 
    Counts images and annotated traffic signs. 
    Creates a report about relevant data.
    """
    image_names = {p.stem for p in img_dir.glob("*.jpg")}
    json_names = {p.stem for p in ann_dir.glob("*.json")}
    missing_json = image_names - json_names
    missing_images = json_names - image_names
    common = image_names & json_names

    class_counter = Counter()
    property_counter = Counter()
    property_groups = defaultdict(list)

    total_objects = 0

    for name in common:
        json_path = ann_dir / f"{name}.json"

        with open(json_path, "r") as f:
            data: dict = json.load(f)

        objects: list[dict] = data.get("objects", [])
        total_objects += len(objects)

        for obj in objects:
            label = obj.get("label", "unknown")
            key = obj.get("key", "no_key")
            props: dict = obj.get("properties", {})

            class_counter[label] += 1

            # property stat + grouping
            for prop, value in props.items():
                if value:
                    property_counter[prop] += 1
                    property_groups[prop].append((name, key))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Report ===\n\n")

        f.write(f"Képek száma: {len(image_names)}\n")
        f.write(f"Annotációk száma: {len(json_names)}\n")
        f.write(f"Párosított fájlok: {len(common)}\n\n")
        f.write(f"Hiányzó képek: {len(missing_images)}\n")
        f.write(f"Hiányzó annotációk: {len(missing_json)}\n\n")
        f.write(f"Összes objektum: {total_objects}\n\n")

        f.write("=== OSZTÁLYELOSZLÁS ===\n")
        for cls, count in class_counter.most_common():
            f.write(f"{cls}: {count}\n")

        f.write("\n=== PROPERTY STAT ===\n")
        for prop, count in property_counter.most_common():
            f.write(f"{prop}: {count}\n")

        f.write("\n=== PROPERTY GROUPS ===\n")
        for prop, items in property_groups.items():
            f.write(f"\n{prop} ({len(items)}):\n")
            for name, key in items:
                f.write(f"\t{name}.json | {key}\n")

    print(f"Report elkészült: {report_path}")


def visualize_dataset(
    img_dir: Path,
    ann_dir: Path,
    output_dir: Path,
):
    """
    Creates the *visualized* folder and puts the annotations 
    on the images. 
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_names = {p.stem for p in img_dir.glob("*.jpg")}
    json_names = {p.stem for p in ann_dir.glob("*.json")}
    common = image_names & json_names

    print(f"Vizualizálás indul... ({len(common)} kép)")

    for name in common:
        img_path = img_dir / f"{name}.jpg"
        json_path = ann_dir / f"{name}.json"
        img = cv2.imread(str(img_path))
        with open(json_path, "r") as f:
            data: dict = json.load(f)

        for obj in data.get("objects", []):
            label = obj.get("label", "unknown")
            key = obj.get("key", "no_key")
            bbox = obj.get("bbox", {})
            xmin = int(bbox.get("xmin", 0))
            ymin = int(bbox.get("ymin", 0))
            xmax = int(bbox.get("xmax", 0))
            ymax = int(bbox.get("ymax", 0))

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            text = f"{label} | {key[:6]}"
            cv2.putText(
                img,
                text,
                (xmin, max(ymin - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        out_path = output_dir / f"{name}.jpg"
        cv2.imwrite(str(out_path), img)
    print("Vizualizáció kész.")


def main():
    """
    Main entry point for data inspection.
    """
    img_dir = Path("train")
    ann_dir = Path("train/annotations")
    output_dir = Path("visualized")
    report_path = Path("report.txt")
    analyze_and_report(img_dir, ann_dir, report_path)
    visualize_dataset(img_dir, ann_dir, output_dir)


if __name__ == "__main__":
    main()
