---
title: >-
  [Paper Note] Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs
description: >-
  [CVPR 2026][Multimodal VLM][OOD Detection] This paper identifies that existing VLM-based OOD detection methods select negative texts using intra-modal distances (text-to-text or image-to-image)…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "OOD Detection"
  - "CLIP"
  - "Cross-modal Distance"
  - "Negative Text Selection"
  - "Zero-shot"
date: 2026-05-08
content_hash: 72beb88d6f0a6e19
---

# Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs

**Conference**: CVPR 2026
**arXiv**: [2603.02618](https://arxiv.org/abs/2603.02618)
**Code**: [https://github.com/ZhikangXu0112/InterNeg](https://github.com/ZhikangXu0112/InterNeg)
**Area**: Multimodal VLM
**Keywords**: OOD Detection, CLIP, Cross-modal Distance, Negative Text Selection, Zero-shot

## TL;DR
This paper identifies that existing VLM-based OOD detection methods select negative texts using intra-modal distances (text-to-text or image-to-image), which are inconsistent with the cross-modal distances optimized by CLIP. The proposed InterNeg framework systematically leverages cross-modal distances from both textual and visual perspectives, achieving a 3.47% FPR95 reduction on ImageNet.

## Background & Motivation

**Background**: VLMs such as CLIP have demonstrated strong performance in OOD detection. The NegLabel method mines negative texts from WordNet by selecting candidates with large intra-modal distances from ID label texts to approximate OOD labels, inspiring a series of follow-up works (CLIPScope, CSP, AdaNeg).

**Limitations of Prior Work**: NegLabel and its successors rely on intra-modal distances (text-space distances from ID labels) when selecting negative texts, and use image-to-image distances when generating visual proxies. However, CLIP is optimized for cross-modal distances (image-text pairs)—a large intra-modal distance does not guarantee a large cross-modal distance.

**Key Challenge**: The inconsistency between intra-modal and cross-modal distances leads to two types of ID misclassification: (1) Max-OOD dominance—a negative text has a smaller cross-modal distance to an ID image than the corresponding ID label; (2) Sum-OOD dominance—the aggregated scores of multiple negative texts exceed the ID score.

**Goal**: To ensure that the entire OOD detection pipeline uses cross-modal distances consistent with CLIP's optimization objective.

**Key Insight**: Define an ID cross-modal reference distance $d_i^{base} = 1 - \cos(\mathbf{e}_i, \mathbf{p}_i)$, and use this reference to filter and generate negative texts.

**Core Idea**: Unify the use of cross-modal distances from both a textual perspective (cross-modal-guided negative text selection) and a visual perspective (inverting high-confidence OOD images into additional negative embeddings in text space).

## Method

### Overall Architecture
The InterNeg framework consists of two components: (1) the textual perspective—replacing intra-modal distances with cross-modal distances for negative text selection; (2) the visual perspective—dynamically identifying high-confidence OOD images at inference time and projecting them into text space to generate additional negative embeddings. The entire process requires no training on ID data or any auxiliary dataset.

### Key Designs

1. **Cross-modal-guided Negative Text Selection**:

    - **Function**: Ensures that selected negative texts have cross-modal distances to all ID classes greater than the ID reference distance.
    - **Mechanism**: For each class, $N$ ID images are randomly sampled and encoded as visual proxies $\mathbf{p}_i$. The ID cross-modal reference distance is computed as $d_i^{base} = 1 - \cos(\mathbf{e}_i, \mathbf{p}_i)$. For each candidate text $y$, the cross-modal distance $d_i(\mathbf{e}^y)$ to each class's visual proxy is computed; only texts satisfying $\forall i, d_i(\mathbf{e}^y) > d_i^{base}$ are retained.
    - Candidates are ranked by deviation score $D(\mathbf{e}^y) = \sum_{i=1}^C d_i(\mathbf{e}^y) - d_i^{base}$ and the top-$M$ are selected.
    - **Design Motivation**: Guarantees that selected negative texts are genuinely distant from the ID distribution in CLIP's cross-modal space.

2. **Visual Inversion for Additional Negative Embeddings**:

    - **Function**: Dynamically discovers OOD samples from test data at inference time and converts them into additional negative texts.
    - **Mechanism**: When a test sample's OOD score satisfies $S(\mathbf{x}) \leq \beta$, it is classified as a high-confidence OOD sample. Its image embedding is projected into text space via modal inversion to generate an additional negative text embedding $\mathbf{e}_v^-$.
    - **Cross-modal-guided dynamic filtering**: Additional embeddings must also satisfy $\forall i, d_i(\mathbf{e}_v^-) > d_i^{base}$ to filter out noisy samples.
    - A fixed-size buffer pool of size $K$ retains the top-$K$ embeddings ranked by deviation score.

3. **OOD Scoring Function**:

    - The final OOD score is computed by integrating ID labels, filtered negative texts, and additional negative embeddings.
    - The formulation follows a structure similar to NegLabel, but with higher-quality negative texts and a dynamically growing negative set.

## Key Experimental Results

### Main Results (ImageNet-1K, CLIP ViT-B/16)

| OOD Dataset | Metric | NegLabel | AdaNeg | InterNeg | Gain |
|-------------|--------|----------|--------|----------|------|
| Avg. over 4 datasets | AUROC↑ | Baseline | 2nd best | **Best** | +0.77% |
| Avg. over 4 datasets | FPR95↓ | Baseline | 2nd best | **Best** | -3.47% |

### Near-OOD Challenging Benchmark

| Benchmark | AUROC Gain | FPR95 Reduction |
|-----------|-----------|-----------------|
| Near-OOD | +5.50% | -2.09% |

### Ablation Study
- Cross-modal negative text selection: significantly reduces both Max-OOD and Sum-OOD misclassification rates.
- Additional negative embeddings: particularly beneficial in the Near-OOD setting.
- Dynamic filtering effectively suppresses interference from noisy OOD images.
- Results are robust to hyperparameters including negative text count $M$ and buffer pool size $K$.

## Highlights & Insights
- First work to explicitly identify the intra-modal/cross-modal distance inconsistency in VLM-based OOD detection—a novel and important perspective.
- The method is training-free and requires no auxiliary data, yet achieves substantial improvements (FPR95 −3.47%).
- Cross-modal distances are unified across both textual and visual perspectives, yielding a theoretically consistent framework.
- The largest gains are observed in the most challenging Near-OOD scenario (AUROC +5.50%).
- The Max-OOD and Sum-OOD misclassification taxonomy provides a clear diagnostic lens for analyzing failure modes.

### Ablation Study
- Cross-modal vs. intra-modal negative selection: Max-OOD misclassification rate drops by approximately 40%, with comparable reduction in Sum-OOD misclassification.
- Contribution of additional negative embeddings: largest gain in Near-OOD setting (AUROC +3.2%), where samples are harder to distinguish.
- Dynamic filtering: removing the filter degrades performance, confirming that noisy OOD samples introduce non-negligible interference.
- Hyperparameter robustness: insensitive to negative text count $M$ (500–2000) and buffer pool size $K$ (50–500).
- Consistent improvements across CLIP backbones (ViT-B/16 and ViT-L/14).

## Limitations & Future Work
- A small number of ID training samples are required to compute visual proxies ($N$ images per class), making the method not strictly zero-shot.
- The quality of high-confidence OOD image inversion depends on the distributional composition of the test data.
- The implementation details of modal inversion (e.g., mapping network architecture) warrant further optimization.
- In streaming test scenarios, the buffer pool update strategy requires further investigation.
- The cross-modal distance consistency principle could be extended to other VLM downstream tasks (e.g., retrieval, classification).
- Computational overhead for reference distance estimation grows with very large ID class sets (e.g., ImageNet-21K).

### Additional Experimental Notes
- InterNeg achieves state-of-the-art AUROC on iNaturalist, demonstrating effectiveness on fine-grained categories.
- Without any training, InterNeg achieves comparable or superior performance to training-based methods such as LoCoOp.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](../../ICCV2025/multimodal_vlm/negrefine_refining_negative_label-based_zero-shot_ood_detection.md)
- [\[CVPR 2026\] Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models](activation_matters_test-time_activated_negative_labels_for_ood_detection_with_vi.md)
- [\[AAAI 2026\] Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models](../../AAAI2026/multimodal_vlm/cross-modal_proxy_evolving_for_ood_detection_with_vision-lan.md)
- [\[CVPR 2026\] Rethinking VLMs for Image Forgery Detection and Localization](rethinking_vlms_for_image_forgery_detection_and_localization.md)
- [\[CVPR 2026\] MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)

</div>

<!-- RELATED:END -->
