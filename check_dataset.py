import argparse
import hashlib
import os
import shutil

import cv2
import mediapipe as mp


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_cleaned")
LANDMARKER_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
CLASSES = ("real", "spoof")
SPLITS = ("train", "test")
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MAX_DETECTION_SIZE = 1280


def resize_for_detection(image):
    height, width = image.shape[:2]
    largest_side = max(height, width)
    if largest_side <= MAX_DETECTION_SIZE:
        return image

    scale = MAX_DETECTION_SIZE / largest_side
    return cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(rebuild=False):
    if rebuild and os.path.exists(OUTPUT_DIR):
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists; move it before rebuilding so it "
            "is not overwritten."
        )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=5,
    )

    summary = {}
    unreadable_images = []
    images_without_faces = []
    images_with_multiple_faces = []
    missing_folders = []
    image_hashes = {}
    accepted_hashes = {}
    cleaned_counts = {}
    if rebuild:
        os.makedirs(OUTPUT_DIR)

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for split in SPLITS:
            for class_name in CLASSES:
                folder = os.path.join(DATASET_DIR, split, class_name)
                key = f"{split}/{class_name}"
                counts = {
                    "images": 0,
                    "readable": 0,
                    "unreadable": 0,
                    "with_face": 0,
                    "no_face": 0,
                    "multiple_faces": 0,
                }
                summary[key] = counts
                cleaned_counts[key] = 0

                if not os.path.isdir(folder):
                    missing_folders.append(folder)
                    continue

                filenames = sorted(
                    filename
                    for filename in os.listdir(folder)
                    if filename.lower().endswith(VALID_EXTENSIONS)
                )
                counts["images"] = len(filenames)

                for filename in filenames:
                    image_path = os.path.join(folder, filename)
                    digest = sha256_file(image_path)
                    image_hashes.setdefault(digest, []).append(image_path)
                    image_bgr = cv2.imread(image_path)
                    if image_bgr is None:
                        counts["unreadable"] += 1
                        unreadable_images.append(image_path)
                        continue

                    counts["readable"] += 1
                    image_bgr = resize_for_detection(image_bgr)
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    result = detector.detect(
                        mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=image_rgb,
                        )
                    )

                    face_count = len(result.face_landmarks)
                    if face_count == 0:
                        counts["no_face"] += 1
                        images_without_faces.append(image_path)
                    else:
                        counts["with_face"] += 1
                    if face_count > 1:
                        counts["multiple_faces"] += 1
                        images_with_multiple_faces.append(
                            (image_path, face_count)
                        )

                    if rebuild and face_count > 0:
                        if digest in accepted_hashes:
                            continue
                        destination = os.path.join(
                            OUTPUT_DIR, split, class_name, filename
                        )
                        os.makedirs(os.path.dirname(destination), exist_ok=True)
                        shutil.copy2(image_path, destination)
                        accepted_hashes[digest] = destination
                        cleaned_counts[key] += 1

    cross_split_duplicates = [
        paths
        for paths in image_hashes.values()
        if len({os.path.relpath(path, DATASET_DIR).split(os.sep)[0] for path in paths}) > 1
    ]
    conflicting_label_duplicates = [
        paths
        for paths in image_hashes.values()
        if len({os.path.relpath(path, DATASET_DIR).split(os.sep)[1] for path in paths}) > 1
    ]

    print("Source dataset audit (original images are never modified)")
    print(
        f"{'split/class':<16} {'images':>7} {'readable':>9} "
        f"{'unreadable':>11} {'with face':>10} {'no face':>8} {'multi-face':>11}"
    )
    for key, counts in summary.items():
        print(
            f"{key:<16} {counts['images']:>7} {counts['readable']:>9} "
            f"{counts['unreadable']:>11} {counts['with_face']:>10} "
            f"{counts['no_face']:>8} {counts['multiple_faces']:>11}"
        )

    for heading, paths in (
        ("Missing dataset folders", missing_folders),
        ("Unreadable images", unreadable_images),
        ("Images with no detected face", images_without_faces),
    ):
        if paths:
            print(f"\n{heading}:")
            for path in paths:
                print(os.path.relpath(path, BASE_DIR))

    if images_with_multiple_faces:
        print("\nImages with multiple detected faces:")
        for path, face_count in images_with_multiple_faces:
            print(f"{os.path.relpath(path, BASE_DIR)} ({face_count})")

    if cross_split_duplicates:
        duplicate_file_count = sum(len(paths) for paths in cross_split_duplicates)
        print(
            f"\nExact duplicate images across train/test: "
            f"{len(cross_split_duplicates)} groups, {duplicate_file_count} files"
        )
        for paths in cross_split_duplicates[:10]:
            print("  " + " <-> ".join(
                os.path.relpath(path, DATASET_DIR) for path in paths
            ))
        if len(cross_split_duplicates) > 10:
            print(f"  ... and {len(cross_split_duplicates) - 10} more groups")

    if conflicting_label_duplicates:
        print("\nExact duplicates assigned to different classes:")
        for paths in conflicting_label_duplicates:
            print("  " + " <-> ".join(
                os.path.relpath(path, DATASET_DIR) for path in paths
            ))

    if rebuild:
        print(f"\nCleaned dataset copied to: {OUTPUT_DIR}")
        print("Kept images with a detected face; exact duplicates kept once.")
        for key, count in cleaned_counts.items():
            print(f"  {key}: {count}")

    print(
        "\nNote: face detection checks image suitability, not whether its "
        "real/spoof label is correct. Train/test duplicates make test "
        "accuracy unreliable."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit the face-spoof dataset without changing the source."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Create a non-destructive cleaned copy in dataset_cleaned.",
    )
    main(rebuild=parser.parse_args().rebuild)
