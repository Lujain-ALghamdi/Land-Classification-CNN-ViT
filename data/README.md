# Dataset

The dataset is **not stored in this repository.** It is third-party course
material and is excluded via `.gitignore`. This page explains what it is, where
to get it, and where to put it.

## What it is

`images-dataSAT` is a binary land-use image dataset used throughout the IBM AI
Engineering capstone:

| Property | Value |
| --- | --- |
| Task | Binary classification: agricultural vs. non-agricultural land |
| Classes | `class_0_non_agri`, `class_1_agri` |
| Images | 6,000 total &mdash; 3,000 per class (balanced) |
| Format | RGB JPEG |
| Native size | 64 &times; 64 px |
| Archive | `images-dataSAT.tar` (&approx; 19 MB) |

The image tiles are derived from **Copernicus Sentinel-2** Level-2A products
(European Space Agency / European Union Copernicus programme). Copernicus
Sentinel data are free and open; when you use them, attribute
"Contains modified Copernicus Sentinel data".

The curated, cropped and labelled `images-dataSAT.tar` bundle is distributed by
**IBM Skills Network** for the course. It is redistributed here neither in
archive nor extracted form. Consult the course terms before redistributing it
yourself.

## How to obtain it

The course labs download the archive from IBM Cloud Object Storage:

```
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar
```

Download and extract it into this `data/` directory, for example:

```bash
cd data
curl -L -o images-dataSAT.tar \
  "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar"
tar -xf images-dataSAT.tar
```

Any equivalent balanced two-class image folder works too &mdash; point the code
at it with `--data-dir` or by editing `src/config.py`.

## Required layout

The code (`src/data.py`) expects exactly this structure:

```
data/
└── images_dataSAT/
    ├── class_0_non_agri/
    │   ├── *.jpg
    │   └── ...
    └── class_1_agri/
        ├── *.jpg
        └── ...
```

`src.data.resolve_dataset_dir()` verifies this and raises a clear error if
either class folder is missing.
