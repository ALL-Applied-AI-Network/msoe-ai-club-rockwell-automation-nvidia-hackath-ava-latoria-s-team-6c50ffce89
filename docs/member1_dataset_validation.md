# Member 1 Dataset Validation Report

## Dataset Overview

- Total images: 2,520
- Original Isaac Sim images: 1,080
- Augmented images: 1,440
- Training images: 2,160
- Validation images: 180
- Test images: 180

## Class Distribution

| Label | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| GOOD | 1,080 | 90 | 90 | 1,260 |
| SHIFTED | 1,056 | 89 | 89 | 1,234 |
| ROTATED | 24 | 1 | 1 | 26 |

## Validation Results

The following checks were completed using PowerShell and the dataset manifest:

- Missing image paths: 0
- Invalid label values: 0
- Images explicitly flagged with invalid observation quality: 0
- Duplicate image paths: 0
- Original sample IDs appearing across multiple splits: 0
- Invalid PNG signatures: 0

One original image from each class was opened successfully in VS Code for preliminary visual inspection.

## Findings and Limitations

The dataset passed initial metadata, path, and PNG signature checks.

The ROTATED class is significantly underrepresented, with only 10 original Isaac Sim captures and 16 augmented images. Validation and testing each contain only one ROTATED image, limiting the reliability of class-specific performance measurements.

The validation checks do not guarantee that all images fully decode or that all labels are visually correct.

## Recommended Next Steps

1. Discuss the ROTATED class imbalance with Member 2.
2. Coordinate additional original ROTATED image collection in Isaac Sim if needed.
3. Preserve the existing train, validation, and test separation.
4. Perform broader visual inspection and full image decoding checks before final dataset handoff.

## Integration Note

The final grasp verifier uses a binary safety decision: `GOOD` versus failure.
`SHIFTED` and `ROTATED` are both treated as failure cases. The independent
holdout and matched Isaac acceptance campaign remain the final readiness gates.
