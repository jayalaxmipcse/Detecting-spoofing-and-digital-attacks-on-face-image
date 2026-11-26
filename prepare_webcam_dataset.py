import hashlib
import os
import random
import shutil
import tempfile

import cv2

from face_preprocessing import crop_face_bgr


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "webcam_samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_webcam")
CLASSES = ("real", "spoof")
SPLITS = ("train", "validation", "test")
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
SEED = 42


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_samples():
    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(
            f"No webcam sample folder found at {SOURCE_DIR}. "
            "Capture labeled samples first."
        )

    participant_samples = {}
    for participant in sorted(os.listdir(SOURCE_DIR)):
        participant_dir = os.path.join(SOURCE_DIR, participant)
        if not os.path.isdir(participant_dir):
            continue

        samples = {label: [] for label in CLASSES}
        for session in sorted(os.listdir(participant_dir)):
            session_dir = os.path.join(participant_dir, session)
            if not os.path.isdir(session_dir):
                continue
            for label in CLASSES:
                label_dir = os.path.join(session_dir, label)
                if not os.path.isdir(label_dir):
                    continue
                samples[label].extend(
                    (os.path.join(label_dir, name), session)
                    for name in sorted(os.listdir(label_dir))
                    if name.lower().endswith(VALID_EXTENSIONS)
                )

        if samples["real"] and samples["spoof"]:
            participant_samples[participant] = samples
        elif any(samples.values()):
            raise ValueError(
                f"Participant {participant} must have both real and spoof "
                "captures before preparation."
            )

    if len(participant_samples) < 3:
        raise ValueError(
            "At least 3 participants with both labels are required to make "
            "separate train, validation, and test splits."
        )
    return participant_samples


def split_participants(participants):
    shuffled = sorted(participants)
    random.Random(SEED).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * 0.15))
    validation_count = max(1, round(len(shuffled) * 0.15))
    if test_count + validation_count >= len(shuffled):
        raise ValueError("Not enough participants to create all three splits.")

    return {
        "test": shuffled[:test_count],
        "validation": shuffled[test_count : test_count + validation_count],
        "train": shuffled[test_count + validation_count :],
    }


def main():
    if os.path.exists(OUTPUT_DIR):
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists; move it before rebuilding so it "
            "is not overwritten."
        )

    participants = collect_samples()
    participant_splits = split_participants(participants)
    rng = random.Random(SEED)
    seen_hashes = {}
    skipped_no_face = 0
    prepared = {split: {label: [] for label in CLASSES} for split in SPLITS}

    with tempfile.TemporaryDirectory(prefix="webcam_faces_") as temporary_dir:
        for split in SPLITS:
            people = participant_splits[split]
            for participant in people:
                for label in CLASSES:
                    for path, session in participants[participant][label]:
                        digest = file_hash(path)
                        previous = seen_hashes.get(digest)
                        if previous:
                            if previous[0] != label:
                                raise ValueError(
                                    "An identical capture has conflicting "
                                    f"labels: {previous[1]} and {path}"
                                )
                            continue
                        seen_hashes[digest] = (label, path)

                        image = cv2.imread(path)
                        if image is None:
                            raise ValueError(f"Could not read captured image: {path}")
                        try:
                            face = crop_face_bgr(image)
                        except ValueError:
                            skipped_no_face += 1
                            continue

                        filename = (
                            f"{participant}_{session}_{os.path.basename(path)}"
                        )
                        temporary_path = os.path.join(
                            temporary_dir, f"{len(seen_hashes)}.jpg"
                        )
                        if not cv2.imwrite(
                            temporary_path,
                            face,
                            [cv2.IMWRITE_JPEG_QUALITY, 95],
                        ):
                            raise OSError(
                                f"Could not write prepared face: {temporary_path}"
                            )
                        prepared[split][label].append(
                            (temporary_path, filename)
                        )

        for split in ("validation", "test"):
            balanced_count = min(
                len(prepared[split]["real"]),
                len(prepared[split]["spoof"]),
            )
            if balanced_count == 0:
                raise ValueError(
                    f"{split} has no usable samples for one or both labels "
                    "after face detection."
                )
            prepared[split]["real"] = rng.sample(
                prepared[split]["real"], balanced_count
            )
            prepared[split]["spoof"] = rng.sample(
                prepared[split]["spoof"], balanced_count
            )

        for split in SPLITS:
            for label in CLASSES:
                if not prepared[split][label]:
                    raise ValueError(
                        f"{split}/{label} has no usable face samples."
                    )

        os.makedirs(OUTPUT_DIR)
        for split in SPLITS:
            for label in CLASSES:
                destination_dir = os.path.join(OUTPUT_DIR, split, label)
                os.makedirs(destination_dir)
                for source_path, filename in prepared[split][label]:
                    shutil.copy2(
                        source_path,
                        os.path.join(destination_dir, filename),
                    )

    print(f"Prepared webcam dataset: {OUTPUT_DIR}")
    print("Participant groups are isolated; validation and test are balanced.")
    for split in SPLITS:
        print(
            f"{split}: real={len(prepared[split]['real'])}, "
            f"spoof={len(prepared[split]['spoof'])}, "
            f"participants={len(participant_splits[split])}"
        )
    print(f"Skipped captures with no detected face: {skipped_no_face}")


if __name__ == "__main__":
    main()
