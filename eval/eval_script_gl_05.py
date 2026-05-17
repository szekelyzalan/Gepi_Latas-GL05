import argparse
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Protocol, Sequence, cast



def load_sample_annotations(dataset_dir: Path, key: str) -> "list[BBox]":
    """
    Load the sample annotations from the given dataset.

    The structure of the dataset is as follows:

        + *<key>.jpg
        + annotations
          + *<key>.json

    Here, ``<key>.jpg`` and ``<key.json>`` denote the samples and the corresponding annotations. An annotation json file has the following structure:

        {
            ...,
            "objects": [
                {
                    ...,
                    "label": "warning--railroad-crossing-with-barriers--g1",
                    "bbox": {
                        "xmin": 1195.5,
                        "ymin": 655.875,
                        "xmax": 1253.0,
                        "ymax": 703.125
                    },
                    ...
                }
            ]
        }

    Parameters
    ----------
    dataset_dir
        The directory of the dataset.
    key
        The key of the sample.

    Returns
    -------
    v
        The ground truth bounding boxes of the given sample.

    Raises
    ------
    ...
    """
    annot_json_path = dataset_dir / "annotations" / f"{key}.json"
    loaded_json = json.loads(annot_json_path.read_text())
    objects_dict = loaded_json["objects"]

    obj_elems: list[BBox] = []
    for obj_elem_dict in objects_dict:
        original_label = obj_elem_dict["label"]
        if original_label not in cat_resolver_dict:
            continue
        obj_elems.append(
            BBox(
                xmin=obj_elem_dict["bbox"]["xmin"],
                xmax=obj_elem_dict["bbox"]["xmax"],
                ymin=obj_elem_dict["bbox"]["ymin"],
                ymax=obj_elem_dict["bbox"]["ymax"],
                category=cat_resolver_dict[obj_elem_dict["label"]],
            )
        )

    return obj_elems


def load_student_generated_annotations(
    student_preds_dir: Path, key: str
) -> "list[BBox]":
    """
    Load the student-generated bounding box data. The student-generated bounding box data has a file format similar to the original annotations. More precisely, this function assumes that ``student_preds_dir`` points to a directory that contains json files. Similarly to the annotations, the stem of the json file names is the key in the dataset, while the extension is json.
    
    The annotations preserve the order of the bounding boxes in the original file.

    Here, ``<key>.jpg`` and ``<key.json>`` denote the samples and the corresponding annotations. An annotation json file has the following structure:

        {
            ...,
            "objects": [
                {
                    ...,
                    "reduced_label": "Warning",
                    "confidence": 1.0,
                    "bbox": {
                        "xmin": 1195.5,
                        "ymin": 655.875,
                        "xmax": 1253.0,
                        "ymax": 703.125
                    },
                    ...
                }
            ]
        }

    Parameters
    ----------
    dataset_dir
        The directory of the dataset.
    key
        The key of the sample.

    Returns
    -------
    v
        The ground truth bounding boxes of the given sample.

    Raises
    ------
    ...
    """
    annot_json_path = student_preds_dir / f"{key}.json"
    loaded_json = json.loads(annot_json_path.read_text())
    objects_dict = loaded_json["objects"]

    obj_elems: list[BBox] = []
    for obj_elem_dict in objects_dict:
        category = str(obj_elem_dict["reduced_label"])
        if category not in cat_resolver_dict.values():
            raise RuntimeError(f"Unknown category \"{category}\"")
        
        obj_elems.append(
            BBox(
                xmin=float(obj_elem_dict["bbox"]["xmin"]),
                xmax=float(obj_elem_dict["bbox"]["xmax"]),
                ymin=float(obj_elem_dict["bbox"]["ymin"]),
                ymax=float(obj_elem_dict["bbox"]["ymax"]),
                category=category,
                confidence=float(obj_elem_dict["confidence"]),
            )
        )

    return obj_elems


def get_dataset_keys(dataset_dir: Path) -> set[str]:
    """
    Get the keys of a dataset at the given path.

    Parameters
    ----------
    dataset_dir
        The directory of the dataset.

    Returns
    -------
    v
        The keys in the dataset at the given directory.
    """
    img_paths = dataset_dir.glob("*.jpg")
    keys = {p.stem for p in img_paths}
    return keys


@dataclass(frozen=True)
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    category: str
    confidence: float | None = None

    def get_confidence(self) -> float:
        if self.confidence is None:
            raise RuntimeError("The confidence is None.")
        
        return self.confidence


    def __post_init__(self) -> None:
        if self.xmin > self.xmax:
            raise ValueError(f"xmin ({self.xmin}) must not be greater than xmax ({self.xmax})")
        if self.ymin > self.ymax:
            raise ValueError(f"ymin ({self.ymin}) must not be greater than ymax ({self.ymax})")
        
        if self.confidence is not None:
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError(f"confidence ({self.confidence}) must be in [0, 1]")

    def iou_with(self, b: "BBox") -> float:
        inter_xmin = max(self.xmin, b.xmin)
        inter_ymin = max(self.ymin, b.ymin)
        inter_xmax = min(self.xmax, b.xmax)
        inter_ymax = min(self.ymax, b.ymax)
        inter_area = max(0.0, inter_xmax - inter_xmin) * max(0.0, inter_ymax - inter_ymin)
        area_a = (self.xmax - self.xmin) * (self.ymax - self.ymin)
        area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin)
        union_area = area_a + area_b - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

@dataclass(frozen=True)
class PRCurve:
    """
    The precision-recall curve for a given category.

    Parameters
    ----------
    category
        The category for which teh precision-recall curve is calculated.
    precisions_sorted
        The precisions sorted by the recalls. Samller indices belong to smaller recalls.
    recalls_sorted
        The sorted recalls. Samller indices belong to smaller recalls.
    
    Raises
    ------
    ValueError
        If the number of precisions and recalls are different or the precisions are not decreasing or the recalls are not increasing.

        If the number of precisions and recalls is not 0 and:

        - The first recall is not almost 0
        - The last recall is not almost 1
    """
    category: str
    precisions_sorted: "tuple[float, ...]"
    recalls_sorted: "tuple[float, ...]"

    def __post_init__(self):
        if len(self.precisions_sorted) != len(self.recalls_sorted):
            raise ValueError(f"The length of the precisions ({len(self.precisions_sorted)}) and the length of the recalls ({len(self.recalls_sorted)}) should be equal.")
        
        if len(self.precisions_sorted) == 0:
            return
        
        if len(self.precisions_sorted) == 1:
            raise ValueError("The number of precision-recall pairs should be either zero or at least 2. Current value: 1")
        
        for i in range(len(self.precisions_sorted)-1):
            p0 = self.precisions_sorted[i]
            p1 = self.precisions_sorted[i+1]
            r0 = self.recalls_sorted[i]
            r1 = self.recalls_sorted[i+1]

            if not (0<=p0<=1):
                raise ValueError(f"A precision value is out of the [0, 1] range. Value: {p0}")

            if r0 > r1:
                raise ValueError("The recalls should be increasing.")
            
            if p0 < p1:
                raise ValueError("The precisions should be decreasing.")
        
        if abs(self.recalls_sorted[0]) > 1e-9:
            raise ValueError("The first recall value should be almost 0.")
        
        if abs(self.recalls_sorted[-1]-1) > 1e-9:
            raise ValueError("The last recall value should be almost 1.")

    @staticmethod
    def from_predictions(*, gt_by_key: dict[str, list[BBox]], pred_by_key: dict[str, list[BBox]], for_category: str, iou_threshold: float) -> "PRCurve":
        """
        Calculate the precision-recall curve from predictions.

        Parameters
        ----------
        gt_by_key
            A dictionary, where the keys are the keys in the dataset, while the values are the ground truth bounding boxes in the dataset.
        pred_by_key
            A dictionary, where the keys are the keys in the dataset, while the values are the corresponding predicted bounding boxes.
        for_category
            The category for which the precision-recall curve is calculated.
        iou_threshold
            The threshold for the IoU above which a predicted bounding box is considered a match.
        
        Returns
        -------
        v
            The calculated prediction-recall curve.
        
        
        Raises
        ------
        CategoryMissingError
            If no ground truth sample was found with the given category.
        ValueError
            If the prediction and ground truth keys are different.
        """
        if not (0 <= iou_threshold <= 1):
            raise ValueError(f"The IoU threshold should be in the [0, 1] range. Value: {iou_threshold}")

        cum_precisions, cum_recalls = PRCurve._calculate_precisions_sorted_by_recalls_without_envelopes(
            for_category=for_category,
            gt_by_key=gt_by_key,
            pred_by_key=pred_by_key,
            iou_threshold=iou_threshold
        )

        cum_precisions = PRCurve._apply_envelopes(cum_precisions)
        
        return PRCurve(
            category=for_category,
            precisions_sorted=tuple(cum_precisions),
            recalls_sorted=tuple(cum_recalls)
        )
    
    @staticmethod
    def _calculate_precisions_sorted_by_recalls_without_envelopes(gt_by_key: dict[str, list[BBox]], pred_by_key: dict[str, list[BBox]], for_category: str, iou_threshold: float) -> tuple[list[float], list[float]]:
        """
        Calculate the cumulative precisions and cumulative recalls that make up the precision-recall curve from predictions.

        Note that this function *does not* apply the envelope technique, it only gives back the raw precision-recall data.

        Parameters
        ----------
        gt_by_key
            A dictionary, where the keys are the keys in the dataset, while the values are the ground truth bounding boxes in the dataset.
        pred_by_key
            A dictionary, where the keys are the keys in the dataset, while the values are the corresponding predicted bounding boxes.
        for_category
            The category for which the precision-recall curve is calculated.
        
        Returns
        -------
        cum_precisions
            The calculated precision values.
        cum_recalls
            The calculated recall values.
        
        Raises
        ------
        CategoryMissingError
            If no ground truth sample was found with the given category.
        ValueError
            If the prediction and ground truth keys are different.
        """
        if gt_by_key.keys() != pred_by_key.keys():
            raise ValueError("The prediction and ground truth keys are different.")

        keys = gt_by_key.keys()

        # filter the ground truth bounding boxes for the given category
        gt_by_image: dict[str, list[BBox]] = {
            key: [b for b in gt_by_key[key] if b.category == for_category]
            for key in keys
        }

        # 1. filter the predicted bounding boxes for the given category
        # 2. flatten the key-value structure to (key, value) tuples to make
        #    subsequent iteration easier
        all_preds: list[tuple[str, BBox]] = [
            (key, b)
            for key in keys
            for b in pred_by_key[key]
            if b.category == for_category
        ]
        n_gt_total = sum(len(boxes) for boxes in gt_by_image.values())

        if n_gt_total == 0:
            raise CategoryMissingError(f"No sample was found with the given category. Category: {for_category}")

        if len(all_preds) == 0:
            # Missed all GT objects; curve is flat at 0.
            return [], []

        # Global ranking by confidence descending
        all_preds.sort(key=lambda t: t[1].get_confidence(), reverse=True)

        # Per-image matched-GT sets prevent cross-image box leaking
        matched_gt_by_image: dict[str, set[int]] = {key: set() for key in keys}
        gt_obj_hit_count = 0
        precisions_sorted: list[float] = []
        recalls_sorted: list[float] = []

        for k, (img_key, pred) in enumerate(all_preds, 1):
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_by_image[img_key]):
                if gt_idx in matched_gt_by_image[img_key]:
                    continue
                iou = pred.iou_with(gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if (best_iou >= iou_threshold) and (best_gt_idx >= 0):
                gt_obj_hit_count += 1
                matched_gt_by_image[img_key].add(best_gt_idx)
            recalls_sorted.append(gt_obj_hit_count / n_gt_total)
            precisions_sorted.append(gt_obj_hit_count / k)
        
        if recalls_sorted[0] >= 1e-9:
            precisions_sorted = [precisions_sorted[0]] + precisions_sorted
            recalls_sorted = [0]+recalls_sorted
        
        if recalls_sorted[-1] < 1-(1e-9):
            precisions_sorted = precisions_sorted + [0, 0]
            recalls_sorted = recalls_sorted + [recalls_sorted[-1], 1]

        return precisions_sorted, recalls_sorted

    @staticmethod
    def _apply_envelopes(precisions_sorted_by_recalls: Sequence[float]) -> list[float]:
        """
        Apply the technique of evelopes to eliminate the wobbles from the precision-recall curve.

        Parameters
        ----------
        precisions_sorted_by_recalls
            The precisions without the envelopes.
        
        Returns
        -------
        v
            The precisions with the envelopes.
        """
        if len(precisions_sorted_by_recalls) == 0:
            return []

        new_precisions_sorted = list(precisions_sorted_by_recalls)
        # Envelope: replace each precision with the maximum seen from here to end
        for i in range(len(precisions_sorted_by_recalls) - 2, -1, -1):
            new_precisions_sorted[i] = max(new_precisions_sorted[i], new_precisions_sorted[i + 1])
        
        return new_precisions_sorted

    def get_ap(self) -> float:
        """
        Get the AP (average precision) metric for this precision-recall curve.
        """

        acc = 0
        for i in range(1, len(self.recalls_sorted)):
            if self.recalls_sorted[i] != self.recalls_sorted[i - 1]:
                acc += (self.recalls_sorted[i] - self.recalls_sorted[i - 1]) * self.precisions_sorted[i]
        
        return acc

class CategoryMissingError(Exception):
    """
    Raised if a category was not found in the given dataset.
    """
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def generate_cat_resolver_dict():
    return {
        "regulatory--yield--g1": "yield",
        "regulatory--no-entry--g1": "no_entry",
        "regulatory--stop--g1": "stop",
        "information--pedestrians-crossing--g1": "pedestrian_crossing",
        "warning--pedestrians-crossing--g1": "pedestrian_crossing",
        "warning--pedestrians-crossing--g5": "pedestrian_crossing",
        "regulatory--maximum-speed-limit-70--g1": "speed_limit",
        "regulatory--maximum-speed-limit-40--g1": "speed_limit",
        "regulatory--maximum-speed-limit-50--g1": "speed_limit",
        "regulatory--maximum-speed-limit-60--g1": "speed_limit",
        "regulatory--maximum-speed-limit-30--g1": "speed_limit",
        "regulatory--maximum-speed-limit-80--g1": "speed_limit",
        "regulatory--maximum-speed-limit-90--g1": "speed_limit",
        "regulatory--maximum-speed-limit-110--g1": "speed_limit",
        "regulatory--maximum-speed-limit-led-100--g1": "speed_limit",
        "regulatory--maximum-speed-limit-100--g1": "speed_limit",
        "regulatory--maximum-speed-limit-10--g1": "speed_limit",
        "regulatory--maximum-speed-limit-20--g1": "speed_limit",
        "regulatory--maximum-speed-limit-led-60--g1": "speed_limit",
        "regulatory--maximum-speed-limit-led-80--g1": "speed_limit",
        "regulatory--maximum-speed-limit-15--g1": "speed_limit",
        "regulatory--maximum-speed-limit-120--g1": "speed_limit"
    }


cat_resolver_dict = generate_cat_resolver_dict()


class _ShowArgs(Protocol):
    """Typed view of the ``show`` subcommand's parsed arguments."""

    dataset_dir: Path
    student_preds_dir: "Path | None"
    key: "str | None"
    width: "int | None"


class _EvaluateArgs(Protocol):
    """Typed view of the ``evaluate`` subcommand's parsed arguments."""

    dataset_dir: Path
    student_preds_dir: Path
    detailed: bool


def draw_bboxes(
    image,
    bboxes: "list[BBox]",
    color: "tuple[int, int, int]",
    label_prefix: str,
    scale: float
) -> None:
    """
    Draw bounding boxes on the given image in-place.

    Parameters
    ----------
    image
        The OpenCV image array to draw on.
    bboxes
        The bounding boxes to draw.
    color
        The BGR color used for the rectangle and label text.
    label_prefix
        A short string prepended to the category name in the label, e.g. ``"GT"`` or ``"Pred"``.
    scale
        The scale of the image compared to the original image.
    """
    import cv2

    for bbox in bboxes:
        pt1 = (int(bbox.xmin*scale), int(bbox.ymin*scale))
        pt2 = (int(bbox.xmax*scale), int(bbox.ymax*scale))
        cv2.rectangle(image, pt1, pt2, color, 2)
        label = f"{label_prefix}: {bbox.category}"
        text_y = max(15, pt1[1] - 5)
        cv2.putText(
            image, label, (pt1[0], text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )


def existing_dir(value: str) -> Path:
    """
    Validate that a string represents an existing directory path.

    Parameters
    ----------
    value
        The raw string provided on the command line.

    Returns
    -------
    v
        The validated :class:`Path` object.

    Raises
    ------
    argparse.ArgumentTypeError
        If the path does not exist or is not a directory.
    """
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {value}")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Path is not a directory: {value}")
    return path

def dataset_key(value: str) -> str:
    key_pattern = r"^[a-zA-Z0-9\-_]+$"
    if not re.match(key_pattern, value):
        raise argparse.ArgumentTypeError("The dataset key does not have the correct format. It should match the regular expression: "+key_pattern)
    
    return value

def _cmd_show(args: _ShowArgs) -> None:
    """
    Handle the ``show`` subcommand.

    Loads one sample by key (or picks one at random) and opens an OpenCV
    window.  Ground truth boxes are drawn in green when ``--dataset-dir`` is
    provided; prediction boxes are drawn in red when ``--student-preds-dir``
    is provided.  When only one directory is given the other set of
    annotations is simply omitted.  If ``--dataset-dir`` is absent the image
    is replaced by a black canvas sized to 1920 × 1080.

    Parameters
    ----------
    args
        The parsed command-line arguments.

    Raises
    ------
    SystemExit
        If neither directory is provided, no samples are found, the requested
        key is absent, or the image file cannot be read.
    """
    import cv2

    dataset_dir = args.dataset_dir
    exit_if_dataset_empty(dataset_dir)

    student_preds_dir = args.student_preds_dir
    all_keys = get_dataset_keys(args.dataset_dir)

    if args.key is not None:
        if args.key not in all_keys:
            print(f"Error: Key '{args.key}' not found in dataset.", file=sys.stderr)
            sys.exit(1)
        key_to_show = args.key
    else:
        all_keys = get_dataset_keys(args.dataset_dir)
        key_to_show = random.choice(sorted(all_keys))

    img_path = dataset_dir / f"{key_to_show}.jpg"
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Failed to load image: {img_path}", file=sys.stderr)
        sys.exit(1)

    if args.width is not None:
        scale = args.width/image.shape[1]
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    else:
        scale = 1
        

    gt_bboxes = load_sample_annotations(dataset_dir=dataset_dir, key=key_to_show)
    draw_bboxes(image=image, bboxes=gt_bboxes, color=(0, 200, 0), label_prefix="GT", scale=scale)

    if student_preds_dir is not None:
        student_bboxes = load_student_generated_annotations(
            student_preds_dir=student_preds_dir, key=key_to_show
        )
        draw_bboxes(image=image, bboxes=student_bboxes, color=(0, 0, 200), label_prefix="Pred", scale=scale)


    cv2.imshow(f"Sample: {key_to_show}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def exit_if_eval_conf_probably_invalid(dataset_dir: Path, student_preds_dir: Path) -> None:
    dataset_keys = get_dataset_keys(dataset_dir)
    if len(dataset_keys) == 0:
        print(f"The dataset at path {dataset_dir} does not contain any jpeg file. Exiting.")
        sys.exit(1)
    student_pred_keys = [
        p.stem for p in student_preds_dir.glob("*.json")
    ]

    if len(student_pred_keys) == 0:
        print(f"The predictions directory at path {student_preds_dir} does not contain any json file. Exiting.")
        sys.exit(1)


def exit_if_dataset_empty(dataset_dir: Path) -> None:
    all_keys = get_dataset_keys(dataset_dir)
    if len(all_keys) == 0:
        print(f"The dataset at path {dataset_dir} does not contain any jpeg file. Exiting.")
        sys.exit(1)

def get_categories(gt_by_key: dict[str, list[BBox]]) -> set[str]:
    result: set[str] = set()
    for li in gt_by_key.values():
        for elem in li:
            result.add(elem.category)
    
    return result

def _cmd_evaluate(args: _EvaluateArgs) -> None:
    """
    Handle the ``evaluate`` subcommand.

    Computes the dataset-level mAP@50 and prints the result.  With
    ``--detailed``, per-category AP values are printed before the overall score.

    Parameters
    ----------
    args
        The parsed command-line arguments.
    """
    exit_if_eval_conf_probably_invalid(
        dataset_dir=args.dataset_dir,
        student_preds_dir=args.student_preds_dir
    )
    keys = get_dataset_keys(dataset_dir=args.dataset_dir)
    gt_by_key = {
        key: load_sample_annotations(dataset_dir=args.dataset_dir, key=key)
        for key in keys
    }
    pred_by_key = {
        key: load_student_generated_annotations(
            student_preds_dir=args.student_preds_dir, key=key
        )
        for key in keys
    }
    all_categories = get_categories(gt_by_key)
    pr_curves = [
        PRCurve.from_predictions(
            gt_by_key=gt_by_key,
            pred_by_key=pred_by_key,
            for_category=category,
            iou_threshold=0.5
        ) for category in all_categories
    ]
    cat_aps = {
        curve: curve.get_ap() for curve in pr_curves
    }

    if args.detailed:
        for pr_curve in sorted(pr_curves, key=lambda c: c.category):
            print(f"{pr_curve.category}: {cat_aps[pr_curve]:.4f}")

    map50 = statistics.mean(cat_aps.values()) if len(cat_aps) > 0 else 1.0
    print(f"\nOverall mAP@50: {map50:.4f}")


class _PRCurveArgs(Protocol):
    """Typed view of the ``pr-curve`` subcommand's parsed arguments."""

    dataset_dir: Path
    student_preds_dir: Path


def _cmd_pr_curve(args: _PRCurveArgs) -> None:
    """
    Handle the ``pr-curve`` subcommand.

    Computes the dataset-level precision-recall curve with envelope interpolation
    for every contributing category and either displays the figure interactively
    or saves it to a file.

    Parameters
    ----------
    args
        The parsed command-line arguments.
    """
    import matplotlib.pyplot as plt

    exit_if_eval_conf_probably_invalid(
        dataset_dir=args.dataset_dir,
        student_preds_dir=args.student_preds_dir,
    )
    keys = get_dataset_keys(dataset_dir=args.dataset_dir)
    gt_by_key = {
        key: load_sample_annotations(dataset_dir=args.dataset_dir, key=key)
        for key in keys
    }
    all_categories = get_categories(gt_by_key)
    pred_by_key = {
        key: load_student_generated_annotations(
            student_preds_dir=args.student_preds_dir, key=key
        )
        for key in keys
    }
    curves = [
        PRCurve.from_predictions(
            gt_by_key=gt_by_key,
            pred_by_key=pred_by_key,
            for_category=category,
            iou_threshold=0.5
        ) for category in all_categories
    ]

    for curve in curves:
        plt.plot(curve.recalls_sorted, curve.precisions_sorted, label=curve.category)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.title("Precision-Recall curve for each category")
    plt.legend()
    plt.show(block=True)
    plt.close()

def positive_int(v: str) -> int:
    if re.match(r"^[0-9]+$", v):
        int_v = int(v)
        if int_v == 0:
            raise argparse.ArgumentTypeError("The specified value is 0.")
        return int_v
    else:
        raise argparse.ArgumentTypeError(f"The value \"{v}\" is not a valid integer.")

def main() -> None:
    """
    Entry point for the evaluation CLI.

    Subcommands
    -----------
    show
        Visualise ground truth and student predicted bounding boxes for one
        sample using OpenCV.
    evaluate
        Compute the mAP@50 metric for every sample and print the results.
    pr-curve
        Plot the dataset-level precision-recall curve for every category.
    """
    parser = argparse.ArgumentParser(
        description="Object detection evaluation tool for student predictions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser(
        "show",
        help="Visualise ground truth and predicted bounding boxes for one sample.",
    )
    show_parser.add_argument(
        "--dataset-dir",
        type=existing_dir,
        default=None,
        metavar="DIR",
        required=True,
        help="The directory of the original dataset. GT annotations are shown only when provided.",
    )
    show_parser.add_argument(
        "--student-preds-dir",
        type=existing_dir,
        default=None,
        metavar="DIR",
        help="The directory containing the student predictions. Prediction annotations are shown only when provided.",
    )
    show_parser.add_argument(
        "--key",
        type=dataset_key,
        default=None,
        metavar="KEY",
        help="The sample key to display. A random key is selected if omitted.",
    )
    show_parser.add_argument(
        "--width",
        type=positive_int,
        help="The width of the displayed image. If not set, then the width of the displayed image will match the width of the original image.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Calculate mAP@50 for all samples.",
    )
    evaluate_parser.add_argument(
        "--dataset-dir",
        type=existing_dir,
        required=True,
        metavar="DIR",
        help="The directory of the original dataset.",
    )
    evaluate_parser.add_argument(
        "--student-preds-dir",
        type=existing_dir,
        required=True,
        metavar="DIR",
        help="The directory containing the student predictions.",
    )
    evaluate_parser.add_argument(
        "--detailed",
        action="store_true",
        help="If true, then the mAP@50 will be printed for each category individually.",
    )

    pr_curve_parser = subparsers.add_parser(
        "pr-curve",
        help="Plot the dataset-level precision-recall curve for every category.",
    )
    pr_curve_parser.add_argument(
        "--dataset-dir",
        type=existing_dir,
        required=True,
        metavar="DIR",
        help="The directory of the original dataset.",
    )
    pr_curve_parser.add_argument(
        "--student-preds-dir",
        type=existing_dir,
        required=True,
        metavar="DIR",
        help="The directory containing the student predictions.",
    )

    args = parser.parse_args()

    if args.command == "show":
        _cmd_show(cast(_ShowArgs, args))
    elif args.command == "evaluate":
        _cmd_evaluate(cast(_EvaluateArgs, args))
    elif args.command == "pr-curve":
        _cmd_pr_curve(cast(_PRCurveArgs, args))


if __name__ == "__main__":
    main()
