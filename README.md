# Cardiac MRI: Automated Left Ventricle Segmentation and Ejection Fraction Estimation

A deep learning pipeline for automated Left Ventricle (LV) segmentation from short-axis cardiac MRI, with downstream computation of Ejection Fraction using Simpson's method of discs.

Trained and validated on the M&Ms (Multi-Centre, Multi-Vendor and Multi-Disease) cardiac MRI dataset.

---

## Table of Contents

- [Background](#background)
- [Pipeline Overview](#pipeline-overview)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup and Usage](#setup-and-usage)
- [Clinical Validation](#clinical-validation)
- [Limitations](#limitations)
- [References](#references)

---

## Background

Ejection Fraction (EF) quantifies the proportion of blood ejected by the Left Ventricle per cardiac cycle. It is among the most clinically actionable measurements in cardiology, used in the diagnosis of heart failure, dilated cardiomyopathy, and pre-surgical risk stratification.

Manual EF measurement from MRI requires a trained clinician to delineate the LV cavity boundary on every short-axis slice at both End-Diastole and End-Systole , a process that is time-intensive, operator-dependent, and difficult to scale.

This project automates that measurement end-to-end.

**Ejection Fraction formula:**

$$EF = \frac{EDV - ESV}{EDV} \times 100$$

Where EDV is End-Diastolic Volume (maximum LV volume) and ESV is End-Systolic Volume (minimum LV volume).

**Clinical interpretation:**

| EF (%) | Classification |
|--------|---------------|
| >= 55 | Normal |
| 45 – 54 | Mildly Reduced |
| 30 – 44 | Moderately Reduced |
| < 30 | Severely Reduced |

---

## Pipeline Overview

```
Short-axis NIfTI (.nii.gz)
         |
         v
Extract ED/ES frames (only labeled frames retained)
Save each slice as .npy for efficient loading
         |
         v
Attention U-Net (2D, slice-by-slice)
Input:  (1, 256, 256) — single MRI slice
Output: (4, 256, 256) — per-pixel class probabilities
         |
         v
Argmax -> Segmentation mask  (0=BG, 1=LV, 2=Myo, 3=RV)
         |
         v  [label 1 only from here]
LV pixel count per slice x pixel spacing -> LV area (cm2)
LV area x slice thickness -> volume contribution (mL)
Sum across all slices -> LV volume per frame
         |
         v
Max volume = EDV   |   Min volume = ESV
         |
         v
EF = (EDV - ESV) / EDV x 100
```

---

## Dataset

**M&Ms (Multi-Centre, Multi-Vendor and Multi-Disease Cardiac Image Segmentation Challenge (2020))**

The M&Ms dataset was selected because it reflects the heterogeneity of real clinical environments: images were acquired across 6 clinical centres using scanners from 4 different vendors (Siemens, Philips, Canon, GE), covering both healthy subjects and patients with common structural heart diseases.

| Property | Detail |
|----------|--------|
| Modality | Cardiac MRI, short-axis cine |
| File format | NIfTI (.nii.gz) |
| Volume shape | (H, W, Slices, Frames) |
| Segmentation labels | 0 Background, 1 LV cavity, 2 Myocardium, 3 RV |
| Annotated frames | End-Diastole and End-Systole only |
| Scanner vendors | Siemens, Philips, Canon, GE |
| Pathologies covered | Dilated Cardiomyopathy, Hypertrophic Cardiomyopathy, Abnormal RV, Normal |

Dataset access: https://www.ub.edu/mnms/

**Preprocessing decisions:**

Non-labeled frames (all frames other than ED and ES) carry all-zero segmentation masks. Training on these would cause the model to learn to predict background everywhere. Only ED/ES frames are retained for training.

Each 2D slice is extracted and saved as a `.npy` file prior to training. This eliminates repeated NIfTI decompression during training, which is the dominant I/O bottleneck when working with `.nii.gz` files on Colab.

---

## Model

**Architecture: Attention U-Net (2D)**

```
Input (1 x 256 x 256)
        |
   Encoder path:   32 -> 64 -> 128 -> 256 -> 320 channels
        |               skip connections with attention gates
   Decoder path:  320 -> 256 -> 128 -> 64 -> 32 channels
        |
Output (4 x 256 x 256)  ->  argmax  ->  label map
```

Attention gates suppress activations in background regions and amplify responses near cardiac structures. This is particularly useful in short-axis cardiac MRI where the heart occupies a small fraction of the total image area.

The model is trained in 2D (one slice at a time) rather than 3D for two reasons. First, the M&Ms dataset has variable and sometimes large inter-slice gaps, which makes volumetric convolution physically inconsistent. Second, 2D convolutions reduce compute by roughly 16x compared to 3D at equivalent depth, enabling larger batch sizes and faster iteration on a single GPU.

**Training configuration:**

| Setting | Value |
|---------|-------|
| Input size | 256 x 256 |
| Batch size | 16 |
| Optimizer | AdamW, lr=3e-4, weight_decay=1e-4 |
| LR schedule | CosineAnnealingWarmRestarts (T0=10) |
| Loss | 0.5 x DiceLoss + 0.5 x Weighted CrossEntropyLoss |
| Precision | Mixed (float16 via torch.amp) |
| Early stopping | Patience = 10 epochs |
| Max epochs | 60 |

**Loss function note:**

LV cavity pixels account for roughly 3-5% of all pixels in a short-axis slice. Standard DiceLoss alone tends to converge on LV but ignore Myocardium and RV, which are even smaller. The weighted CrossEntropyLoss component assigns inverse-frequency weights per class, penalising missed predictions on rare structures proportionally to how rarely they appear.

---

## Results

**LV segmentation (validation set):**

| Structure | Dice Score | Target |
|-----------|-----------|--------|
| LV Cavity | 0.9219 | > 0.90 |

EF computation depends only on LV cavity segmentation (label 1). The model meets the clinical target threshold.

**EF estimation across all patients:**

| Metric | Result | Reference range |
|--------|--------|----------------|
| Mean EF | 56.6% | 55 – 75% |
| Mean EDV | 173.7 mL | 100 – 210 mL |
| Mean ESV | 76.4 mL | 30 – 110 mL |
| Outliers | 1 patient | < 5% of cohort |

All three population means fall within published reference ranges for a mixed healthy/pathological cohort. The single outlier (EDV 479.78 mL, EF 45.41%) is consistent with Dilated Cardiomyopathy and is considered a genuine pathological case rather than a model error.

---

## Project Structure

```
cardiac-mri-ef-calculation/
|
|-- Cardiac_MRI.ipynb            Main notebook, all cells in sequence
|-- best_attention_unet.pth      Saved model checkpoint
|
|-- results/
|   |-- ef_results.csv               Per-patient EDV, ESV, EF values
|   |-- ef_results_with_flags.csv    With physiological outlier flags
|   |-- ef_results.png               EF distribution plot
|   |-- clinical_validation.png      Six-panel validation dashboard
|   |-- segmentation_quality.png     GT vs prediction visual check
|
|-- README.md
```

---

## Setup and Usage

**Requirements**

```
torch >= 2.0
monai >= 1.3
nibabel >= 5.0
numpy
pandas
matplotlib
scikit-learn
scipy
```

```bash
pip install monai nibabel scikit-learn scipy matplotlib pandas
```

**Running the notebook**

Open `Cardiac_MRI.ipynb` in Google Colab. Set the runtime to GPU (Runtime > Change runtime type > T4 GPU).

Place `Labeled.zip` (the M&Ms labeled split) in the root of your Google Drive before running.

| Cell | Purpose | Re-run on session restart |
|------|---------|--------------------------|
| 0 | Install dependencies, all imports | Yes |
| 1 | Mount Drive, extract zip | Only if /content/Labeled missing |
| 2 | Extract ED/ES slices to .npy | Only if /content/slices missing |
| 3 | Dataset class, DataLoaders | Yes |
| 4 | Model, loss, optimizer | Yes |
| 5 | Training loop | No — restore from Drive |
| 6 | Load checkpoint from Drive | Yes |
| 7 | Inference function | Yes |
| 8 | Volume and EF functions | Yes |
| 9 | Run inference on all patients | No — results saved to Drive |
| 10–17 | Validation, plots, reports | No — outputs saved to Drive |

The training checkpoint is saved to Google Drive after each improvement so it survives Colab session resets.

---

## Clinical Validation

Results were assessed against established physiological reference ranges:

| Check | Acceptable range | Violations |
|-------|-----------------|------------|
| EF | 5 – 90% | 0 |
| EDV | 30 – 400 mL | 1 (DCM case) |
| ESV | 10 – 350 mL | 0 |
| Stroke volume (EDV - ESV) | > 10 mL | 0 |

**Validation methodology:**

Bland-Altman analysis was performed on stroke volume (EDV minus ESV) to assess measurement consistency across the cohort. Per-patient clinical classification was assigned using ACC/AHA EF thresholds. Segmentation predictions were visually reviewed against ground truth for a random sample of patients.

**Note on the flagged case:**

Patient M4P7Q6 returned EDV 479.78 mL and EF 45.41%. This falls outside the 30–400 mL EDV range used for automated flagging. Given the EF of 45.41% (Mildly Reduced) and the markedly elevated EDV, this presentation is consistent with Dilated Cardiomyopathy, in which ventricular dilation is a primary feature. Visual inspection of the segmentation confirmed anatomically reasonable LV delineation. The case was retained and flagged for clinician review, which is the correct handling for edge-case volumes in a clinical pipeline.

---

## Limitations

This pipeline was developed and validated on one dataset (M&Ms). Generalisation to scanner models, acquisition protocols, or patient populations not represented in M&Ms has not been evaluated.

The 2D slice-by-slice approach does not enforce volumetric consistency between adjacent slices. An inter-slice artifact or segmentation error on a single slice propagates directly into the volume estimate without any geometric smoothing.

Ground-truth EF values (measured by expert clinicians) are not available in the M&Ms public release. The EF results reported here are model outputs compared to physiological reference ranges, not to paired expert measurements. A formal correlation study against cardiologist-derived EF would be needed before clinical deployment.

---

## References

Campello V.M. et al. (2021). Multi-Centre, Multi-Vendor and Multi-Disease Cardiac Image Segmentation Challenge. IEEE Transactions on Medical Imaging.

Oktay O. et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. arXiv:1804.03999.

Lang R.M. et al. (2015). Recommendations for Cardiac Chamber Quantification by Echocardiography in Adults. Journal of the American Society of Echocardiography.

MONAI Consortium (2020). MONAI: Medical Open Network for AI. https://monai.io

---

## Author

**DUA QAISAR**
BS Bioinformatics,NUST
---

*This project is for academic and research purposes. It is not validated for clinical diagnostic use.*
