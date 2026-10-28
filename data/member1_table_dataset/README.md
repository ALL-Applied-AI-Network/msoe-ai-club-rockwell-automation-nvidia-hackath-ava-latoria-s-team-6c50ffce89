# Member 1 table-camera dataset

This is the shared Isaac Sim dataset for the AI/ML teammate working on stack-quality classification. It contains cropped table-camera observations and the labels needed to train and evaluate a model without logging into Brev.

## What is included

- 2,520 PNG images total
- 1,080 original Isaac Sim images
- 1,440 saved augmented copies
- 2,160 training images, 180 validation images, and 180 test images
- Labels: 1,260 `GOOD`, 1,234 `SHIFTED`, and 26 `ROTATED`
- `manifest.json`, which connects every image path to its label, split, source, and simulation details

The image paths in `manifest.json` are relative to this folder. Use the provided `train`, `val`, and `test` split values instead of making a new random split.

## Get the images after cloning

The PNG files use Git LFS so the normal repository stays manageable.

1. Install Git LFS once from <https://git-lfs.com/>.
2. Open a terminal in the repository.
3. Run `git lfs install`.
4. Run `git lfs pull`.
5. Check that images appear under `data/member1_table_dataset/raw/` and `data/member1_table_dataset/augmented/`.

## Suggested first steps for Member 1

1. Load `manifest.json`.
2. Train with rows whose `split` is `train`.
3. Tune with `val`, then report final results once on `test`.
4. Start with a small pretrained image classifier such as MobileNetV3 or ResNet18.
5. Report per-class precision, recall, and a confusion matrix. Overall accuracy alone can hide poor `ROTATED` performance because that class currently has only 26 images.

Keep raw and augmented versions of the same sample in the split already listed in the manifest. Do not move augmented versions into validation or test.
