---
title: >-
  [Paper Note] PetFace: A Large-Scale Dataset and Benchmark for Animal Identification
description: >-
  [ECCV 2024][LLM Evaluation][Animal Identification] A large-scale animal face recognition dataset, PetFace, is constructed, containing 13 animal families, 319 breeds, and 257,484 individuals (over 1 million images). Two benchmark tests are established: seen individual re-identification (Re-ID) and unseen individual verification, providing infrastructure for non-invasive automatic animal identification.
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "Animal Identification"
  - "Face Recognition"
  - "Large-Scale Dataset"
  - "Re-identification"
  - "Benchmark"
date: 2026-05-08
content_hash: e30012cae170e86e
---

# PetFace: A Large-Scale Dataset and Benchmark for Animal Identification

**Conference**: ECCV 2024  
**arXiv**: [2407.13555](https://arxiv.org/abs/2407.13555)  
**Code**: Yes ([https://dahlian00.github.io/PetFacePage/](https://dahlian00.github.io/PetFacePage/))  
**Area**: LLM Evaluation  
**Keywords**: Animal Identification, Face Recognition, Large-Scale Dataset, Re-identification, Benchmark

## TL;DR

A large-scale animal face recognition dataset, PetFace, is constructed, containing 13 animal families, 319 breeds, and 257,484 individuals (over 1 million images). Two benchmark tests are established: seen individual re-identification (Re-ID) and unseen individual verification, providing infrastructure for non-invasive automatic animal identification.

## Background & Motivation

Individual animal identification is crucial in scenarios such as behavior monitoring, habitat surveys, lost pet retrieval, and health check-ups. Traditional methods (ear tags, tattoos, toe-clipping) are invasive, causing stress and pain in animals, and their use should be minimized. Although newer tools like digital IDs reduce invasiveness, they require equipping animals individually, which is costly and still causes stress.

Human face recognition is highly mature, thanks to large-scale datasets and benchmarks (e.g., MS-Celeb, VGGFace2). However, the development of animal face recognition is limited by the **severe scarcity of datasets**:

| Dataset | Species | Individuals | Images |
|--------|------|--------|--------|
| CTai | Chimpanzee | 78 | 5,078 |
| DogFaceNet | Dog | 1,393 | 8,363 |
| MacaqueFaces | Monkey | 34 | 6,280 |
| **PetFace (Ours)** | **13 Families** | **257,484** | **1,012,934** |

The number of individuals in PetFace is over 110 times larger than that of the previous largest animal face dataset (DogFaceNet), spanning 13 animal families and 319 breeds, filling the data gap in the field of animal face recognition.

## Method

### Overall Architecture

PetFace is not a model methodology paper, but rather a **dataset and benchmark paper**. Its core contributions include:

1. **Dataset Construction**: Efficiently collecting large-scale, high-quality animal face images from the web.
2. **Two Evaluation Protocols**: Seen individual re-identification (Re-ID) and unseen individual verification.
3. **Benchmark Experiments**: Establishing baselines across various loss functions and pre-trained models.

### Key Designs

**1. Data Collection Strategy**

Images are acquired from two types of web sources:
- **Pet store websites**: Providing high-quality, multi-angle images with detailed individual information (color, gender, breed).
- **Pet adoption websites**: Providing images in diverse backgrounds, uploaded by pet owners.

Only one adoption website per region was selected to avoid duplication. Chimpanzee data was additionally sourced from collaborative research institutions. The initial collection yielded 1,443,737 images of 325,420 individuals.

**2. Face Detection and Alignment**

The AnyFace model is used to detect facial keypoints. Due to the large variations in facial structures across different types of species, independent reference points and alignment methods are defined for each species. A frontal reference image is first selected, and the average keypoint positions of all images after alignment with the reference are calculated as the alignment target.

**3. Data Filtering**

A two-stage filtering workflow:
- **Automated stage**: Removing images where multiple faces are detected.
- **Manual stage**: Verified individually by the authors (taking about 100 person-hours) to remove non-animal images and poorly aligned images. Ultimately, 70% of the initial images were retained.

**4. Fine-grained Annotation**

| Annotation Type | Coverage | Description |
|----------|--------|------|
| Gender | 94% (240,861 individuals) | Extracted from websites |
| Breed | 8 species (319 breeds) | Cat, Dog, Guinea pig, etc. |
| Color/Pattern | 11 species | Two-tier hierarchical annotation |

### Loss & Training

Benchmark experiments use a ResNet-50 backbone, comparing four loss functions:

1. **Softmax**: Basic classification loss.
2. **Center Loss**: Minimizes intra-class variation to make features of the same individual more compact.
3. **Triplet Loss**: Ensures that the distance of positive pairs is smaller than that of negative pairs.
4. **ArcFace Loss**: Minimizes distance in the angular space with an added margin penalty to enhance feature discriminability.

## Key Experimental Results

### Main Results

**Re-identification Results (Top-1 Accuracy %) — By Species**

| Method | Cat | Dog | Chimp | Chinchilla | Guinea | Hamster | Hedgehog | Average |
|------|-----|-----|-------|------------|--------|---------|----------|------|
| Softmax | 30.46 | 59.14 | 41.70 | 58.13 | 60.07 | 38.27 | 27.81 | 41.88 |
| Center | 0.00 | 0.00 | 5.38 | 29.76 | 31.77 | 9.46 | 13.76 | 9.81 |
| ArcFace | 54.29 | 77.86 | 43.27 | 67.34 | 67.90 | 47.37 | 30.90 | 51.23 |
| Joint ArcFace | **70.30** | 68.75 | 34.30 | **69.86** | 68.66 | **54.33** | **44.38** | **53.80** |

**Verification Results (AUC %)**

| Method | Cat | Dog | Chimp | Chinchilla | Guinea pig | Average |
|------|-----|-----|-------|------------|------------|------|
| Softmax | 97.97 | 98.98 | 85.22 | 84.44 | 95.30 | 90.38 |
| Triplet | 96.94 | 97.97 | 77.10 | 76.12 | 83.37 | 83.48 |
| ArcFace | 97.71 | **99.45** | 83.76 | **87.70** | **96.03** | **91.30** |

### Ablation Study

Comparison with models trained on other datasets:

| Pre-training Data | Architecture | Cat Verification AUC | Dog Verification AUC | Average AUC |
|------------|------|-----------|-----------|---------|
| ImageNet | ResNet-50 | 73.71 | 73.04 | - |
| CLIP | ResNet-50 | 74.98 | 87.22 | - |
| MegaDescriptor | SwinT-B | 88.52 | 97.44 | - |
| **PetFace (ArcFace)** | ResNet-50 | **97.71** | **99.45** | **91.30** |

### Key Findings

1. **ArcFace is the most suitable loss function for animal face recognition**: Consistently leading in both re-identification and verification tasks.
2. **Center Loss fails significantly**: Totally failing to learn on categories with massive numbers of individuals like Cat and Dog (0% accuracy), indicating that pure intra-class compactness constraint is insufficient.
3. **Double-edged sword of joint training**: Joint training across species increases performance on Cat from 54.29% to 70.30%, but decreases it on Chimp from 43.27% to 34.30%, showing that joint training on imbalanced data still needs improvement.
4. **Models trained on PetFace significantly outperform other datasets**: Even using a simple ResNet-50, it outperforms SwinTransformer trained on MegaDescriptor (a joint collection of 33 datasets).
5. **Cross-species generalization is promising**: Demonstrating a degree of generalization even on unseen animal families.

## Highlights & Insights

- **Innovative data procurement strategy**: Ingeniously leveraging pet stores and adoption websites as data sources, where each page naturally corresponds to an individual ID, avoiding expensive field filming.
- **Obvious advantage in scale**: Spanning from 1,393 to 257,484 individuals, establishing a solid foundation for unseen individual verification.
- **Value of fine-grained metadata**: Annotations of breed, color, and gender can be used to construct more challenging fine-grained evaluations (e.g., distinguishing different individuals within the same breed).
- **Revealing unique animal ID challenges**: Unlike human faces, animal facial structures vary dramatically across species, making a unified model highly challenging.

## Limitations & Future Work

1. **Bias toward pets in data sources**: Primarily domestic animals; wild animals (such as zebras, whales, etc.) are not covered.
2. **Uneven image quality**: Images sourced from the web exhibit significant variations in quality, illumination, and background.
3. **Imbalance in image count per individual**: Some individuals have only 2-3 images, which limits the training efficacy.
4. **Suboptimal joint training strategies**: Current simple cross-species joint training is unstable under severe class imbalances.
5. **Lack of 3D information**: Utilizing only 2D facial images, omitting 3D facial shape information.

## Related Work & Insights

- **Difference from WildlifeDatasets**: WildlifeDatasets aggregates 33 existing small datasets, but each remains independent and limited in individual count. PetFace, by contrast, is a freshly collected, unified, large-scale dataset.
- **Direct transfer of face recognition techniques**: Human face recognition methods like ArcFace remain effective on animal faces, confirming cross-domain generalizability of these approaches.
- **Insights**: To deploy animal ID systems at scale, the challenge of unified cross-species modeling must be addressed—potentially requiring species-aware hierarchical recognition architectures.

## Rating

| Dimension | Score (1-5) | Evaluation |
|------|-----------|------|
| Novelty | 4 | Leap-forward improvement in dataset scale and coverage |
| Technical Depth | 3 | Modeling mainly applies existing methods; innovation lies in dataset construction |
| Experimental Thoroughness | 4 | Comprehensive comparison across multiple loss functions, pre-training schemes, and cross-species evaluation |
| Writing Quality | 4 | Clear structure with detailed description of the dataset construction process |
| Value | 4.5 | Significant infrastructural value for the animal identification community |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Sync from the Sea: Retrieving Alignable Videos from Large-Scale Datasets](sync_from_the_sea_retrieving_alignable_videos_from_large-scale_datasets.md)
- [\[ACL 2025\] HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning](../../ACL2025/llm_evaluation/hellaswag-pro_a_large-scale_bilingual_benchmark_for_evaluating_the_robustness_of.md)
- [\[ICML 2026\] Authority, Truth, and Citation Bias: A Large-Scale Multi-Domain Benchmark for Studying Epistemic Susceptibility in Large Language Models](../../ICML2026/llm_evaluation/authority_truth_and_citation_bias_a_large-scale_multi-domain_benchmark_for_study.md)
- [\[ICLR 2026\] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](../../ICLR2026/llm_evaluation/anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)
- [\[ICLR 2026\] Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective](../../ICLR2026/llm_evaluation/can_vision_language_models_assess_graphic_design_aesthetics_a_benchmark_evaluati.md)

</div>

<!-- RELATED:END -->
