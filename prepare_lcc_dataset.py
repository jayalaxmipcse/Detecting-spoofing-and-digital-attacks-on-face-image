import hashlib
import os
import random
import re
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "LCC_FASD_development")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_lcc")
CLASSES = ("real", "spoof")
SPLITS = ("train", "validation", "test")
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
SEED = 42


def image_identity(filename):
    match = re.search(r"_id([^_]+)_", filename)
    if not match:
        raise ValueError(f"Cannot extract identity from filename: {filename}")
    return match.group(1)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if os.path.exists(OUTPUT_DIR):
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists; move it before rebuilding so it "
            "is not overwritten."
        )

    identity_counts = {}
    source_files = {}
    for class_name in CLASSES:
        class_dir = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing source class folder: {class_dir}")

        files = sorted(
            os.path.join(class_dir, filename)
            for filename in os.listdir(class_dir)
            if filename.lower().endswith(VALID_EXTENSIONS)
        )
        source_files[class_name] = files
        identity_counts[class_name] = {
            image_identity(os.path.basename(path))
            for path in files
        }

    shared_ids = sorted(
        identity_counts["real"] & identity_counts["spoof"],
        key=int,
    )
    real_only_ids = sorted(
        identity_counts["real"] - identity_counts["spoof"],
        key=int,
    )
    if len(shared_ids) < 5 or len(real_only_ids) < 4:
        raise ValueError(
            "Not enough identities to create separate train, validation, "
            "and test splits."
        )

    rng = random.Random(SEED)
    rng.shuffle(shared_ids)
    rng.shuffle(real_only_ids)

    identity_split = {}
    for split, identities in (
        ("test", shared_ids[:2] + real_only_ids[:2]),
        ("validation", shared_ids[2:4] + real_only_ids[2:4]),
        ("train", shared_ids[4:] + real_only_ids[4:]),
    ):
        for identity in identities:
            identity_split[identity] = split

    candidates = {split: {name: [] for name in CLASSES} for split in SPLITS}
    for class_name, files in source_files.items():
        for path in files:
            split = identity_split[image_identity(os.path.basename(path))]
            candidates[split][class_name].append((path, file_hash(path)))

    seen_hashes = {}
    selected = {split: {name: [] for name in CLASSES} for split in SPLITS}
    for split in SPLITS:
        for class_name in CLASSES:
            for path, digest in candidates[split][class_name]:
                previous = seen_hashes.get(digest)
                if previous:
                    if previous[1] != class_name:
                        raise ValueError(
                            "An identical image appears under conflicting "
                            f"labels: {previous[0]} and {path}"
                        )
                    continue
                seen_hashes[digest] = (path, class_name)
                selected[split][class_name].append(path)

    for split in ("validation", "test"):
        real_count = len(selected[split]["real"])
        spoof_count = len(selected[split]["spoof"])
        if spoof_count < real_count:
            raise ValueError(
                f"{split} has fewer spoof than real samples; cannot balance "
                "without duplicating images."
            )
        selected[split]["spoof"] = rng.sample(
            selected[split]["spoof"], real_count
        )

    os.makedirs(OUTPUT_DIR)
    for split in SPLITS:
        for class_name in CLASSES:
            destination_dir = os.path.join(OUTPUT_DIR, split, class_name)
            os.makedirs(destination_dir)
            for source_path in selected[split][class_name]:
                shutil.copy2(
                    source_path,
                    os.path.join(destination_dir, os.path.basename(source_path)),
                )

    print(f"Created LCC-only dataset: {OUTPUT_DIR}")
    print("Identity groups are kept separate across splits; originals are untouched.")
    for split in SPLITS:
        counts = {
            name: len(selected[split][name])
            for name in CLASSES
        }
        print(
            f"{split}: real={counts['real']}, spoof={counts['spoof']}"
        )


if __name__ == "__main__":
    main()
