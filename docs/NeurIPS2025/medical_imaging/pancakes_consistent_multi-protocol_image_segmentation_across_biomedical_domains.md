---
title: >-
  [Paper Note] Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains
description: >-
  [NeurIPS 2025][Medical Imaging][Multi-protocol segmentation] This paper proposes the Pancakes framework, which, given a collection of biomedical images from an unseen domain, automatically generates label maps for multiple plausible segmentation protocols, ensuring **semantic consistency** across images within the same protocol—i.e., the same label refers to the same anatomical structure across all images.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Multi-protocol segmentation
  - semantic consistency
  - foundation models
  - biomedical imaging
  - unsupervised segmentation
date: 2026-05-08
content_hash: 73bef1204f592af8
---

# Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains

**Conference**: NeurIPS 2025
**arXiv**: [2512.13534](https://arxiv.org/abs/2512.13534)
**Code**: Unavailable
**Area**: Medical Imaging / Image Segmentation
**Keywords**: Multi-protocol segmentation, semantic consistency, foundation models, biomedical imaging, unsupervised segmentation

## TL;DR

This paper proposes the Pancakes framework, which, given a collection of biomedical images from an unseen domain, automatically generates label maps for multiple plausible segmentation protocols, ensuring **semantic consistency** across images within the same protocol—i.e., the same label refers to the same anatomical structure across all images.

## Background & Motivation

Biomedical images can be segmented in multiple ways—according to different protocols based on tissue type, vascular territory, anatomical region, pathology, and more. Existing automated segmentation methods suffer from two fundamental problems:

**Single-protocol binding**: Traditional models (e.g., nnU-Net) support only the single protocol specified at training time; switching protocols requires re-annotation and retraining.

**Burden of interactive methods**: While methods such as SAM and ScribblePrompt allow new protocols to be specified, they require per-image user interactions (clicks, scribbles, exemplars), which is extremely laborious when processing large image collections.

**Cross-image inconsistency**: Existing foundation models (SAM, UnSAM, etc.), despite supporting fully automatic segmentation of individual images, produce labels that are **semantically inconsistent** across images—the same label ID may refer to entirely different structures in different images, and the same structure may receive different label IDs across images, making population-level analysis infeasible.

Pancakes introduces a novel capability: given a set of images from a new domain, it automatically discovers and generates multiple plausible segmentation protocols, each maintaining semantic consistency within the image collection. Users need not specify a protocol in advance; instead, they select from the model-discovered candidate protocols whichever best suits their needs.

## Method

### Overall Architecture

Given an image $x$, Pancakes produces distribution parameters $\phi$, from which label maps for multiple protocols can be sampled. The pipeline consists of three steps:
1. A U-Net $f_{\theta_f}(x) = \phi$ estimates per-pixel distribution parameters.
2. Based on random variable $r_m = (M, K)$, $M$ protocols are sampled, each with $K$ labels.
3. A shallow convolutional network $h_{\theta_h}(\phi \| v_m) = \hat{y}_m$ generates concrete label maps from the distribution parameters.

### Key Designs

1. **Protocol sampling mechanism**: Positional-encoding-like vectors distinguish different protocols and labels. For protocol $m$ and label $k$, the vector $v_{m,k} = u_m \| u_k$ is constructed, where $u_t^{2j} = \sin(z_{t,2j} + \pi/2)$ and $z_{t,j} = \frac{t\pi}{T} 2^{2j\pi/J}$. The period $T$ is determined by the sampled $M$ and $K$. This design ensures encodings for different protocols and labels remain distinguishable while supporting flexible numbers of protocols and labels. Crucially, $\phi$ is computed only once (in the main network forward pass), after which multiple label maps for different $r_m$ can be generated efficiently.

2. **Set consistency loss**: During training, a set of images $\{x_s\}$ and corresponding binary labels $\{y_s\}$ are sampled; label maps for $M$ protocols are generated. The loss encourages cross-image consistency by averaging Dice scores within the set:

$$d_{m,k}(\{\hat{y}_{s,m,k}\}, \{y_s\}) = \mathbb{E}_s[\mathcal{L}_{\text{Dice}}(\hat{y}_{s,m,k}, y_s)]$$

$$\mathcal{L}_{\text{seg}} = \min_{m,k} d_{m,k}(\{\hat{y}_{s,m,k}\}, \{y_s^t\})$$

The $\min$ operation (rather than expectation) encourages diversity—at least one protocol–label combination must match the ground truth, while other protocols are free to discover alternative segmentations. Averaging across the set ensures that label $k$ under protocol $m$ refers to the same region across all images.

3. **Synthetic data augmentation**: Using Anatomix and TotalSegmentator, 120K training pairs are synthesized. Binary labels are sampled from segmentation maps; different affine/elastic transforms are applied to the same label map to simulate images within a set, and intensity values are assigned to generate synthetic images. This increases training diversity and improves generalization to new domains.

### Loss & Training

- Input dimensions $B \times S \times C \times H \times W$ are flattened to $(B \times S) \times C \times H \times W$ for compatibility with 2D convolutions.
- During training: $K \in [5, 40]$, $M \in [5, 15]$, $S \in [2, 5]$ are sampled randomly.
- Two categories of data augmentation: in-task (independently increases set diversity) and task (applied consistently across the set, increasing protocol diversity).
- AdamW optimizer with learning rate 0.0001.
- Cross-domain training data sourced from Megamedical (comprising dozens of biomedical datasets).
- At inference, $K$ and $M$ are user-specified; SoftMax ensures non-overlapping labels.

## Key Experimental Results

### Main Results: Set Dice Comparison on 7 Held-Out Datasets

Pancakes outperforms all baselines on all 7 held-out datasets, typically by 20+ Dice points.

| Method | Parameters | S=1 Inference Time | S=3 Inference Time |
|--------|-----------|-------------------|-------------------|
| Pancakes | **0.22M** | **0.10s** | **0.12s** |
| SAM | 641M | 3.13s | 2.94s |
| ScribblePrompt | 93.7M | 1.99s | 1.85s |
| MedSAM | 93.7M | 2.12s | 1.94s |
| UnSAM | 23M | 0.54s | 0.45s |

### Accuracy vs. Consistency Analysis

| Setting | Pancakes | Baselines |
|---------|----------|-----------|
| S=1 (accuracy only) | Comparable to SAM, superior to others | SAM best |
| S=3 (accuracy + consistency) | **No performance degradation** | **All baselines degrade substantially** |
| S=5 (larger set) | Remains stable | Further deterioration |

### Ablation Study: Effect of Synthetic Data (M=16)

| Training Data | S=1 | S=2 | S=3 | S=5 |
|--------------|-----|-----|-----|-----|
| Real + Synthetic | **73.2** | **67.3** | **67.4** | **68.4** |
| Real only | 71.1 | 65.8 | 65.7 | 67.4 |
| Synthetic only | 56.3 | 45.8 | 44.3 | 42.7 |

### Effect of M and K

| Configuration | Key Findings |
|--------------|-------------|
| Increasing M | Performance improves consistently; more protocols → higher probability of covering ground truth |
| Increasing K | More complex effect; optimal around K=20; larger K yields finer structural segmentation |
| Interactive initialization | Pancakes initialization + ScribblePrompt: click count halved (3–4 vs. 5–8) |

### Key Findings

- **Consistency is the core advantage**: Pancakes is the only method whose performance does not degrade as set size increases, since all baselines lack cross-image semantic consistency guarantees.
- Joint training on synthetic and real data achieves the best results ($p < 0.05$).
- The protocol space is approximately smooth—protocols with nearby embeddings produce similar segmentations.
- Only 0.22M parameters; ~30× faster than SAM (641M).

## Highlights & Insights

- **Novel problem formulation**: Introduces "multi-protocol consistent segmentation" as a new task, filling a critical gap in existing foundation model capabilities.
- **Elegance of min-Dice loss**: By taking the $\min$ over the best protocol–label combination rather than averaging, the method encourages diversity while avoiding regression to the mean.
- **Exceptional efficiency**: A fully convolutional architecture with 0.22M parameters matches SAM (641M parameters) in accuracy while being 30× faster at inference.
- **Two practical use cases**: (1) Rapid segmentation under a new protocol—select the best-matching candidate protocol; (2) exploratory population analysis—discover candidate segmentation regions associated with clinical outcomes.
- **Complementary to interactive segmentation**: Can serve as initialization for methods such as ScribblePrompt, reducing the number of required user interactions.

## Limitations & Future Work

- The discovered protocols represent "model-identified plausible segmentations" and do not necessarily correspond to established clinical standard protocols.
- Training relies on the Megamedical dataset; generalization to highly out-of-distribution domains remains to be verified.
- The method operates in 2D and has not been extended to 3D volumetric segmentation.
- Semantic interpretation of protocols requires domain expert involvement—label indices carry no intrinsic meaning.
- Potential societal biases have not been evaluated.

## Related Work & Insights

- **UniverSeg / Tyche**: Context-based segmentation models that require exemplars to specify protocols.
- **SAM / SAM2**: General-purpose interactive segmentation, but lacks cross-image consistency.
- **SynthSeg**: Segmentation trained on synthetic data.
- **Anatomix**: Synthetic biomedical data generation.
- Insights: Consistency is an underappreciated yet critical property for population-level analysis; positional encoding techniques can be leveraged to parameterize discrete choice spaces.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Entirely new problem formulation; multi-protocol consistent segmentation is unprecedented.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Seven held-out datasets, ablation studies, and efficiency analysis provide comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem motivation is articulated with exceptional clarity; figures and tables are highly informative.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a core limitation of foundation models in biomedical applications; high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Orochi: Versatile Biomedical Image Processor](orochi_versatile_biomedical_image_processor.md)
- [\[NeurIPS 2025\] LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation](lomix_learnable_weighted_multi-scale_logits_mixing_for_medical_image_segmentatio.md)
- [\[NeurIPS 2025\] Unpaired Image-to-Image Translation for Segmentation and Signal Unmixing](unpaired_image-to-image_translation_for_segmentation_and_signal_unmixing.md)
- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](../../CVPR2026/medical_imaging/biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[ICCV 2025\] MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance](../../ICCV2025/medical_imaging/multiverseg_scalable_interactive_segmentation_of_biomedical_imaging_datasets_wit.md)

</div>

<!-- RELATED:END -->
