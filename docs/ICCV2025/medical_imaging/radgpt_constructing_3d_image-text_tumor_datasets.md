---
title: >-
  [Paper Note] RadGPT: Constructing 3D Image-Text Tumor Datasets
description: >-
  [ICCV 2025][Medical Imaging][CT report generation] This paper proposes RadGPT — an anatomy-aware vision-language AI pipeline that converts radiologist-revised tumor segmentation masks into structured reports via deterministic algorithms, then adapts them into narrative-style reports using an LLM. This pipeline is used to construct AbdomenAtlas 3.0, the first large-scale public abdominal CT image-text tumor dataset (9,262 CT scans with per-voxel annotations and reports). The w…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "CT report generation"
  - "tumor dataset"
  - "segmentation-assisted reporting"
  - "abdominal CT"
  - "structured report"
date: 2026-05-08
content_hash: 2e0f4dd225e9e5a2
---

# RadGPT: Constructing 3D Image-Text Tumor Datasets

**Conference**: ICCV 2025  
**arXiv**: [2501.04678](https://arxiv.org/abs/2501.04678)  
**Code**: [https://github.com/MrGiovanni/RadGPT](https://github.com/MrGiovanni/RadGPT)  
**Area**: Medical Imaging / Report Generation  
**Keywords**: CT report generation, tumor dataset, segmentation-assisted reporting, abdominal CT, structured report

## TL;DR
This paper proposes RadGPT — an anatomy-aware vision-language AI pipeline that converts radiologist-revised tumor segmentation masks into structured reports via deterministic algorithms, then adapts them into narrative-style reports using an LLM. This pipeline is used to construct AbdomenAtlas 3.0, the first large-scale public abdominal CT image-text tumor dataset (9,262 CT scans with per-voxel annotations and reports). The work demonstrates that segmentation assistance significantly improves tumor detection rates in AI-generated reports.

## Background & Motivation

**Background**: The United States performs over 85 million CT scans annually, with a 6% annual growth rate far exceeding the 0.7% annual growth rate of radiologists. There is enormous demand for AI-assisted report generation, yet training data remains scarce — no public abdominal CT dataset currently provides both radiology reports and voxel-level annotations simultaneously.

**Limitations of Prior Work**: (a) Public CT datasets either provide segmentation masks without reports, or text annotations without voxel-level labels; (b) existing abdominal CT report generation models (M3D, Merlin) are trained on limited data and perform extremely poorly on tumor detection, particularly near-completely missing small tumors ≤2 cm; (c) conventional report generation metrics (BLEU, ROUGE) are easily confounded by writing style differences and fail to measure diagnostic accuracy.

**Key Challenge**: Tumors may occupy as little as 0.0001% of the total CT volume, making localization of such minute lesions extremely difficult for purely visual VLMs. Segmentation models excel at voxel-level localization but cannot produce structured text reports.

**Goal**: (a) Construct the first large-scale CT–mask–report triplet dataset; (b) design a deterministic, interpretable pipeline from segmentation to report; (c) demonstrate that segmentation assistance substantially improves tumor detection in generated reports.

**Key Insight**: Rather than having a VLM generate reports directly from images (which tends to miss small tumors), the pipeline first localizes tumors with a segmentation model, then applies deterministic rules to extract attributes and populate templates to produce structured reports, and finally uses an LLM to convert them into narrative style. This "segmentation → attribute extraction → template filling → style adaptation" pipeline guarantees full consistency and interpretability between reports and segmentation masks.

**Core Idea**: Translating segmentation masks into radiology reports via deterministic algorithms ensures both interpretability and accuracy while overcoming the tendency of VLMs to miss minute tumors.

## Method

### Overall Architecture
RadGPT is a three-stage pipeline: Stage I — Segmentation (DiffTumor + nnU-Net segments 26 anatomical structures → radiologist revision); Stage II — Structured report generation (deterministic algorithms extract attributes from segmentations → template filling); Stage III — Style adaptation (LLM converts structured reports into the narrative style of the target institution). Human revision is incorporated when constructing AbdomenAtlas 3.0; it is skipped in fully automatic inference mode.

### Key Designs

1. **Organ Sub-Segmentation for Tumor Localization (Stage II-a)**:

    - Function: Divides the liver into 8 Couinaud segments and the pancreas into head/body/tail to enable precise tumor location description in structured reports.
    - Mechanism: Liver sub-segmentation — the liver region HU is offset by 200 as input, and nnU-Net is fine-tuned on 131 LiTS cases; pancreas sub-segmentation — the superior mesenteric artery (SMA) is used as an anatomical landmark, with a deterministic algorithm dividing the pancreas into head/body/tail (no public annotations exist; this dataset is the first to provide them).
    - Design Motivation: Tumor location is critical for surgical planning (e.g., resectability); radiology reports must specify the sub-segment containing the tumor.

2. **Radiologist-Style Tumor Measurement (Stage II-b)**:

    - Function: Extracts WHO-standard longest tumor diameter $D$, its perpendicular diameter $d$, volume, and HU attenuation from segmentation masks.
    - Mechanism: A deterministic algorithm implements the WHO measurement standard — finding the longest diameter and its in-plane perpendicular across any axial plane. Structured reports also include organ volume (for diagnosing organomegaly) and mean HU (for diagnosing hepatic steatosis <40 HU and fatty pancreas <0.7 pancreas/spleen ratio).
    - Design Motivation: Standardized measurements ensure inter-report comparability; deterministic rules guarantee full consistency between reports and masks.

3. **Pancreatic Cancer Staging (Stage II-c)**:

    - Function: Automatically performs PDAC T-staging (T1–T4) from tumor and vascular segmentations.
    - Mechanism: Five key vessels (SMA, CHA, CA, SMV, PV) are segmented, and the tumor-vessel contact angle is measured. A contact angle >180° indicates unresectable tumor (T4 stage). Deterministic algorithms faithfully implement radiological guidelines.
    - Design Motivation: PDAC carries extremely high mortality, and staging determines the surgical approach. This dataset is the first to provide public PDAC T-stage labels.

4. **LLM Style Adaptation (Stage III)**:

    - Function: Converts the populated structured report into the narrative style of the target institution.
    - Mechanism: In-context learning is employed, preferentially selecting example reports with similar diagnoses as few-shot demonstrations. The LLM is instructed to preserve medical information and perform self-consistency checking. An enhanced mode also supports merging structured reports with existing human-authored reports.
    - Design Motivation: Report styles vary considerably across institutions; LLM adaptation broadens the dataset's applicability.

### Loss & Training
RadGPT itself requires no end-to-end training. The segmentation models used (DiffTumor for tumor segmentation and nnU-Net for organ/vessel segmentation) are trained separately on different datasets. Report quality is assessed using newly proposed diagnostic metrics: an LLM extracts labels (tumor present/absent) from both AI-generated and human reports, and sensitivity and specificity are then computed. Zero-shot label extraction accuracy by the LLM was validated by radiologists at 96%.

## Key Experimental Results

### Main Results
Internal validation on the AbdomenAtlas 3.0 test split and external validation on UCSF, reporting tumor detection sensitivity and specificity:

| Model | Pancreas Sen. (≤2cm) | Pancreas Sen. (>2cm) | Pancreas Spec. | Kidney Sen. (≤2cm) | Liver Sen. (≤2cm) |
|------|--------|--------|--------|--------|--------|
| CT-CHAT | 66.7 | 51.9 | 61.2 | 31.1 | 5.7 |
| CT2Rep | 0.0 | 0.0 | 92.5 | 36.5 | 35.8 |
| M3D | 0.0 | 7.4 | 97.2 | 8.1 | 9.4 |
| Merlin | 33.3 | 51.9 | 71.8 | 28.4 | 30.2 |
| RadFM | 0.0 | 0.0 | 99.9 | 3.7 | 3.3 |
| **RadGPT** | **66.7** | **81.5** | **93.2** | **54.8** | **39.6** |

### Ablation Study / Dataset Statistics

| Metric | Value |
|------|------|
| Total CT scans | 9,262 |
| CTs containing tumors | 3,955 |
| Newly annotated tumors | 3,011 (expanding source dataset annotations by 4.2×) |
| Total report tokens | 1,843,262 |
| Small tumors (≤2 cm) | 7,003 |
| Liver tumors | 5,582 |
| Kidney tumors | 4,424 |
| Pancreatic tumors | 368 |

### Key Findings
- **Segmentation assistance dramatically improves tumor detection**: RadGPT achieves 81.5% sensitivity for pancreatic tumors >2 cm, far exceeding all purely vision-language models (best: 51.9%). The gap is even larger for small tumors ≤2 cm.
- **Pure VLMs nearly fail on small tumor detection**: CT2Rep, M3D, and RadFM achieve near-zero sensitivity on small pancreatic and liver tumors, demonstrating severe inadequacy of VLMs when lesions occupy a negligible fraction of the image volume.
- **External validation performance is maintained**: On the UCSF external dataset, RadGPT continues to substantially outperform all comparison methods.
- **Conventional text metrics are unreliable**: Merlin achieves BLEU/ROUGE scores comparable to RadGPT, yet its diagnostic sensitivity is substantially lower, further confirming the necessity of diagnostic metric-based evaluation.

## Highlights & Insights
- **Interpretability of the deterministic pipeline**: The key innovation lies in Stage II being implemented entirely with deterministic algorithms (no ML), guaranteeing 100% consistency between structured reports and segmentation masks. Radiologists need only review the segmentation to trust the report, without separately auditing the text.
- **Comprehensive dataset value**: AbdomenAtlas 3.0 simultaneously provides CT scans, voxel-level annotations, and reports as triplets, filling a critical gap in abdominal CT datasets and enabling a new paradigm of segmentation-assisted report generation.
- **Diagnostically oriented evaluation metrics**: The proposed scheme of using an LLM to extract diagnostic labels from reports and then computing sensitivity/specificity is more clinically relevant than BLEU/ROUGE.
- **Automatic pancreatic cancer staging**: The first fully automatic pipeline from segmentation masks to T-stage is demonstrated, with important clinical implications for the highly lethal PDAC.

## Limitations & Future Work
- **Dependence on segmentation quality**: In fully automatic mode, segmentation errors propagate directly into reports; false positives and false negatives for small tumors are particularly impactful.
- **Only three organ tumor types covered**: The current system handles only liver, kidney, and pancreatic tumors; other common tumor sites such as lung and colon are not yet addressed.
- **Limited stylistic diversity in narrative reports**: LLM style adaptation is based on example reports from a single institution; cross-institutional generalizability remains to be validated.
- **Future directions**: Injecting segmentation information as visual prompts into VLMs for end-to-end utilization of positional information could be explored; uncertainty estimation could also be incorporated — flagging low-confidence segmentations with cautionary notes in the report.

## Related Work & Insights
- **vs. CT2Rep [ECCV'24]**: CT2Rep is a purely visual approach with limited sensitivity to subtle findings near tumors, resulting in very low tumor detection sensitivity.
- **vs. Merlin [MICCAI'24]**: Merlin performs well on BLEU metrics but shows a large gap in diagnostic sensitivity, demonstrating that text similarity ≠ diagnostic accuracy.
- **vs. traditional 2D X-ray report generation**: Lesions in X-ray occupy approximately 5–10% of the image, whereas CT tumors may occupy as little as 0.0001%; directly transferring 2D methods is not feasible.

## Rating
- Novelty: ⭐⭐⭐⭐ The deterministic "segmentation → report" pipeline is novel and practical, though the approach is engineering-oriented.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six model comparisons, internal and external validation, and new evaluation metrics validated by radiologists.
- Writing Quality: ⭐⭐⭐⭐ Content is detailed and dataset description is clear, though the paper is lengthy.
- Value: ⭐⭐⭐⭐⭐ The dataset contribution is substantial; code and data are publicly released, representing a landmark contribution to the community.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance](multiverseg_scalable_interactive_segmentation_of_biomedical_imaging_datasets_wit.md)
- [\[CVPR 2026\] VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation](../../CVPR2026/medical_imaging/voxtell_free-text_promptable_universal_3d_medical_image_segmentation.md)
- [\[ICCV 2025\] Scaling Tumor Segmentation: Best Lessons from Real and Synthetic Data](scaling_tumor_segmentation_best_lessons_from_real_and_synthetic_data.md)
- [\[ICCV 2025\] UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation](ukbob_one_billion_mri_labeled_masks_for_generalizable_3d_medical_image_segmentat.md)
- [\[ICCV 2025\] M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast](m-net_mri_brain_tumor_sequential_segmentation_network_via_mesh-cast.md)

</div>

<!-- RELATED:END -->
