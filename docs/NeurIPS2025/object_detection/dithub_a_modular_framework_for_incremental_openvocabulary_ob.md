---
title: >-
  [Paper Note] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection
description: >-
  [NeurIPS 2025][Object Detection][open-vocabulary detection] DitHub reformulates the incremental adaptation problem in open-vocabulary object detection as a "version control" problem — training independent LoRA expert modules per category and managing an ever-growing module library via three primitives: branch, fetch, and merge. On ODinW-13 with full data, the method achieves 62.19 mAP, surpassing ZiRa by 4.21 points, while maintaining 47.01 zero-shot COCO performance.
tags:
  - NeurIPS 2025
  - Object Detection
  - open-vocabulary detection
  - incremental learning
  - modular deep learning
  - LoRA
  - model merging
  - version control
date: 2026-05-08
content_hash: 35a6107786258f44
---

# DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection

**Conference**: NeurIPS 2025
**arXiv**: [2503.09271](https://arxiv.org/abs/2503.09271)
**Code**: [https://aimagelab.github.io/DitHub/](https://aimagelab.github.io/DitHub/)
**Area**: Object Detection
**Keywords**: open-vocabulary detection, incremental learning, modular deep learning, LoRA, model merging, version control

## TL;DR

DitHub reformulates the incremental adaptation problem in open-vocabulary object detection as a "version control" problem — training independent LoRA expert modules per category and managing an ever-growing module library via three primitives: branch, fetch, and merge. On ODinW-13 with full data, the method achieves 62.19 mAP, surpassing ZiRa by 4.21 points, while maintaining 47.01 zero-shot COCO performance.

## Background & Motivation

**State of the Field**: Open-vocabulary object detectors (e.g., Grounding DINO) generalize to arbitrary categories via text prompts, yet efficient incremental learning strategies remain necessary when handling rare categories or continuously adapting across diverse specialized domains. Methods such as ZiRa have begun exploring Incremental Vision-Language Object Detection (IVLOD), achieving competitive results on ODinW-13.

**Limitations of Prior Work**: Existing incremental adaptation methods adopt a "monolithic" strategy — all new knowledge is compressed into a single weight set. This leads to: (1) difficulty selectively updating knowledge for specific categories without affecting others; (2) knowledge of rare categories being diluted within unified weights; (3) inability to gracefully update corresponding knowledge when the same category appears across different domains (e.g., RGB vs. thermal imaging).

**Root Cause**: Incremental detection requires simultaneously achieving "category specialization" and "cross-category/cross-domain composition," yet monolithic weight architectures inherently couple all category knowledge together, making selective updating and composition extremely difficult.

**Paper Goals**: To achieve incremental adaptation to new domains/categories in open-vocabulary detection while supporting: (1) selective updating of previously learned categories; (2) flexible composition of cross-domain modules; (3) effective preservation of zero-shot capability.

**Starting Point**: Drawing from Modular Deep Learning and borrowing the concept of version control systems (Git), each category's knowledge is encapsulated as an independent LoRA module, and detection knowledge is managed in the same manner as code branches.

**Core Idea**: An independent LoRA A matrix is maintained per category as an "expert branch," while the B matrix is shared to ensure memory efficiency. A warmup→branch→fetch→merge pipeline enables scalable incremental detection.

## Method

### Overall Architecture

DitHub is built upon pretrained Grounding DINO, freezing the backbone and inserting LoRA modules only into the encoder for adaptation. The core idea is to decompose the LoRA low-rank factorization $\Delta W = BA$ into two distinct roles: the A matrix encodes category-specific knowledge (one per category), while the B matrix encodes general knowledge (globally shared). The entire framework operates through three Git-style primitives:

- **Branch**: Upon arrival of a new task, a warmup phase (category-agnostic) is first performed, followed by branching an independent A matrix for each category in the task.
- **Fetch**: If a category has appeared in a previous task, the corresponding category expert module is retrieved from the module library.
- **Merge**: The retrieved old module is combined with the current warmup module via weighted averaging, serving as the initialization for a new round of specialization.

At inference, a specific category expert module can be activated independently for fine-grained detection, or multiple category modules can be averaged and merged for multi-category detection in a single forward pass.

### Key Designs

1. **Warmup-then-Specialization Two-Stage Training**
    - Function: Decouples LoRA training into a category-agnostic warmup phase and a category-specific specialization phase.
    - Mechanism: At the start of each new task, a shared warmup matrix $A_{wu}$ is first trained to provide a robust common initialization for all categories. The model then branches into $|C_t|$ independent experts, each updated via a random strategy only on images containing its corresponding category.
    - Design Motivation: A common initialization point yields better linear mode connectivity, which is a prerequisite for the composability of subsequent modules. Additionally, warmup provides a solid starting point for rare categories, avoiding insufficient training due to data scarcity.

2. **Asymmetric Design with Per-Category A Matrix and Shared B Matrix**
    - Function: Halves memory overhead while maintaining category specialization.
    - Mechanism: The A matrix ($r \times k$) is stored independently per category, encoding category-specific detection knowledge; the B matrix ($d \times r$) is globally shared, encoding general detection capabilities. At the end of each task, B is updated via weighted merging: $B_t = (1-\lambda_B) B_{t-1} + \lambda_B B^{opt}$.
    - Design Motivation: Memory for independent $(A, B)$ pairs grows linearly with the number of categories; sharing B directly halves this cost. Experiments confirm that at rank=2, DitHub matches ZiRa in memory while exceeding it by +2.28 mAP, with rank=1 (A degenerating to a vector) remaining competitive.

3. **Fetch-Merge Mechanism for Category Reappearance**
    - Function: Handles scenarios where the same category reappears across different tasks or domains.
    - Mechanism: If category $c$ already has a stored module $A_c^{old}$, a new initialization is generated via weighted merging: $A_c^{cur} = (1-\lambda_A) A_c^{old} + \lambda_A A_{wu}$, where $\lambda_A$ is set to a low value (0.1–0.3) to prioritize preserving existing knowledge.
    - Design Motivation: Re-training from warmup would discard previously accumulated category knowledge, while directly using the old module fails to incorporate domain information from the new task. Weighted merging balances historical knowledge retention with new-domain adaptation, proving particularly effective on ODinW-O (+4.75 mAP).

### Loss & Training

- Grounding DINO's standard detection losses are employed: Focal Loss (classification) + L1 Loss + GIoU Loss (localization).
- Equal epochs are allocated to the warmup and specialization phases.
- Random training strategy during the specialization phase: for each image, a randomly selected present category is chosen and only its corresponding A matrix is updated.
- $\lambda_A$: set to 0.3 for ODinW-13 and 0.1 for ODinW-O (more overlapping categories require greater retention of historical knowledge).
- $\lambda_B$: fixed at 0.7 (biased toward the latest task's $B^{opt}$, as it implicitly subsumes knowledge from earlier tasks).
- Only the encoder of Grounding DINO is adapted; LoRA rank defaults to $r=16$.

## Key Experimental Results

### Main Results

**ODinW-13 Full-Shot**:

| Method | ZCOCO | Avg mAP | vs. ZiRa |
|--------|-------|---------|----------|
| Grounding DINO (zero-shot) | 47.41 | 46.80 | — |
| TFA | 30.97 | 47.93 | -10.05 |
| AT | 42.30 | 51.14 | -6.84 |
| OW-DETR | 31.22 | 55.58 | -2.40 |
| CL-DETR | 32.15 | 57.26 | -0.72 |
| iDETR | 37.32 | 58.71 | +0.73 |
| ZiRa | 46.26 | 57.98 | — |
| **DitHub** | **47.01** | **62.19** | **+4.21** |

**ODinW-13 Few-Shot**:

| Method | 1-shot Avg | 5-shot Avg | 10-shot Avg |
|--------|-----------|-----------|------------|
| ZiRa | 48.56 | 51.77 | 53.20 |
| DitHub | 49.19 | **52.85** | **54.43** |
| Gain | +0.63 | +1.08 | +1.23 |

**ODinW-O (Category Reappearance)**:

| Method | ZCOCO | Avg mAP |
|--------|-------|---------|
| Grounding DINO | 47.41 | 53.15 |
| ZiRa | 44.43 | 57.63 |
| **DitHub** | **46.51** | **62.38 (+4.75)** |

### Ablation Study

**Component Ablation (ODinW-13 Full-Shot)**:

| Configuration | Avg mAP | ZCOCO | Notes |
|--------------|---------|-------|-------|
| Base (no warmup, no merging) | ~56 | ~48 | Complete catastrophic forgetting |
| +Warmup | ~59 | ~47.5 | Warmup yields significant improvement |
| +Warmup +B merge | ~59.5 | ~47.3 | B merging provides marginal gain |
| DitHub (+A merge) | 62.19 | 47.01 | A merging is the most critical component |

**Specialization vs. Non-Specialization (EnE)**:

| Method | ZCOCO | Avg mAP |
|--------|-------|---------|
| EnE (random assignment, no category specialization) | 46.86 | 60.96 |
| DitHub (category specialization) | 47.01 | 62.19 (+1.23) |

**LoRA Rank Ablation**:

| Rank | Memory (MB) | Avg mAP |
|------|------------|---------|
| r=1 | ~9 | 57.04 |
| r=2 | ~18 | 60.26 |
| ZiRa | ~18 | 57.98 |
| r=8 | ~74 | 61.93 |
| r=16 | ~147 | 62.19 |

### Key Findings

- DitHub achieves the best performance on 9/13 tasks under the full-shot setting, surpassing ZiRa by an average of +4.21 mAP, while better preserving zero-shot capability (+0.75).
- As tasks accumulate, DitHub's performance degradation rate is lower than ZiRa's; the anti-forgetting advantage of the modular design amplifies with the number of tasks.
- The advantage is even larger in the category reappearance scenario (ODinW-O, +4.75 mAP), validating the effectiveness of the fetch-merge mechanism.
- DitHub at rank=2 matches ZiRa in memory while outperforming it by +2.28 mAP, demonstrating an excellent performance–memory trade-off.
- Category specialization (vs. EnE) yields a +1.23 mAP gain, confirming the necessity of per-category branch training over random assignment.
- DitHub supports training-free unlearning: subtracting a category's A module removes the model's detection capability for that category.

## Highlights & Insights

- **The Git version control analogy is remarkably precise**: the three primitives — branch/fetch/merge — map perfectly to the "create/retrieve/fuse" operations in incremental detection, yielding a conceptually clear and easily extensible framework.
- **The decoupled warmup→specialization design is highly instructive**: a common initialization point guarantees composability across modules (linear mode connectivity), establishing a bridge between modular detection and model merging theory.
- **The asymmetric role assignment of A and B matrices is a key engineering innovation**: A encodes category-specific knowledge (requiring independence) while B encodes general knowledge (amenable to sharing), naturally aligning LoRA's mathematical structure with modular design requirements.
- This work represents the first systematic study of LoRA module composability in object detection, filling the research gap between NLP (LoRAHub) and visual detection.
- The training-free unlearning capability (via module weight subtraction) carries practical compliance value, as privacy regulations may require selective removal of specific category detection capabilities.

## Limitations & Future Work

- **Validated only on Grounding DINO**: other open-vocabulary detectors such as YOLO-World and OWLv2 are not evaluated; the generalizability of the framework remains to be verified.
- **Module selection at inference requires manual specification**: an automated routing mechanism is absent, preventing the model from automatically determining which expert modules to activate based on input.
- **Multi-category merging strategy is relatively simple**: only simple averaging is currently used to merge multiple category modules; more sophisticated adaptive weighting or attention-based routing is unexplored.
- **ODinW scale is limited**: the largest setting covers only 13/35 sub-datasets; the scalability of module library management in large-scale scenarios (hundreds of categories/domains) is unknown.
- **Hyperparameters $\lambda$ require dataset-specific tuning**: the optimal values of $\lambda_A$ differ between ODinW-13 and ODinW-O (0.3 vs. 0.1), and no adaptive mechanism is provided.

## Related Work & Insights

- **vs. ZiRa** (NeurIPS 2024): ZiRa employs reparameterizable side branches for monolithic adaptation; DitHub uses modular LoRA to achieve category-level granularity management, yielding superior performance under equivalent memory constraints.
- **vs. LoRAHub** (NLP): LoRAHub validates LoRA composability in LLMs; DitHub is the first to introduce this idea to object detection, finding that detection tasks exhibit different merging characteristics due to non-uniform category distributions.
- **vs. CL-DETR/iDETR**: traditional incremental detection methods prevent forgetting through architectural modifications or regularization; DitHub fundamentally avoids inter-category interference through modular design.
- **Insights**: The modular adaptation paradigm can be directly transferred to segmentation (e.g., training per-domain SAM LoRA modules for subsequent merging), personalized generation (concept customization in text-to-image models), and similar settings. The framework is naturally compatible with MoE architectures, and future work may consider training a router to automatically select modules.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The analogy from Git version control to detection module management is novel and practical, though LoRA composition has precedent in NLP.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Dual benchmarks (ODinW-13 + ODinW-O), full coverage of full/1/5/10-shot settings, comprehensive component ablations, rank ablations, $\lambda$ sensitivity analysis, EnE comparisons, and unlearning demonstrations.
- **Writing Quality**: ⭐⭐⭐⭐ — The Git analogy makes the framework intuitively accessible and the algorithmic pseudocode is clear; however, the dense notation requires some initial familiarization.
- **Value**: ⭐⭐⭐⭐ — Modular incremental detection addresses a genuine need in real-world deployment; the framework design is broadly generalizable, and the +4.21 mAP improvement is substantive.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](../../ICCV2025/object_detection/dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)
- [\[CVPR 2026\] ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](../../CVPR2026/object_detection/abra_teleporting_finetuned_knowledge_across_domain.md)
- [\[CVPR 2026\] NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection](../../CVPR2026/object_detection/noovd_novel_category_discovery_and_embedding_for_open-vocabulary_object_detectio.md)
- [\[CVPR 2026\] Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection](../../CVPR2026/object_detection/parameter-efficient_semantic_augmentation_for_enhancing_open-vocabulary_object_d.md)
- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](../../AAAI2026/object_detection/vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)

<!-- RELATED:END -->
