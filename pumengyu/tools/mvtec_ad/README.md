# MVTec AD tools

This folder mirrors the lightweight utilities from `/home/PuMengYu/MVTec_AD`.
They are kept inside the nnUNet repo because the same inspection/indexing/
visualization pattern is useful for quick external dataset checks and future
industrial anomaly-detection demos.

These scripts do not touch nnU-Net training state, plans, splits, checkpoints,
or `Dataset003_Liver`.

## What is useful here

- `inspect_mvtec.py`: checks whether an extracted MVTec AD tree has all expected
  categories and counts train/test/mask images.
- `make_mvtec_index.py`: builds a flat CSV index with category, split,
  defect type, image path, anomaly label, and mask path.
- `visualize_mvtec_sample.py`: exports a quick image/mask overlay sheet for one
  category.

## Current default data location

```text
/home/PuMengYu/MVTec_AD/data/raw
```

The scripts can also point to another extracted MVTec AD root with `--root`.

## Commands

```bash
python -m pumengyu.tools.mvtec_ad.inspect_mvtec \
  /home/PuMengYu/MVTec_AD/data/raw

python -m pumengyu.tools.mvtec_ad.make_mvtec_index \
  --root /home/PuMengYu/MVTec_AD/data/raw \
  --output /home/PuMengYu/MVTec_AD/data/processed/mvtec_index.csv

python -m pumengyu.tools.mvtec_ad.visualize_mvtec_sample \
  --root /home/PuMengYu/MVTec_AD/data/raw \
  --category transistor \
  --output /home/PuMengYu/MVTec_AD/outputs/transistor_sample_sheet.png
```

## Not directly reusable for nnU-Net segmentation

MVTec AD is 2D RGB/texture anomaly detection with image-level labels and
optional pixel masks. It is not a direct nnU-Net raw dataset converter. To train
nnU-Net on it, a separate converter would be needed to create `imagesTr`,
`labelsTr`, `imagesTs`, and `dataset.json` with a 2D segmentation setup.
