---
title: >-
  [Paper Note] Unlocking Positive Transfer in Incrementally Learning Surgical Instruments: A Self-reflection Hierarchical Prompt Framework
description: >-
  [CVPR 2026][Medical Imaging][Class-incremental segmentation] This paper restructures per-instrument prompt parameters from isolated, independent prompts into a tree-structured hierarchy that progressively decomposes shared knowledge across layers. This design enables new instruments to inherit prior knowledge for rapid learning, while allowing new knowledge to gently revise existing representations, thereby simultaneously improving performance on new, regular, and old classes in surgical instrument class-incremental segmentation.
tags:
  - CVPR 2026
  - Medical Imaging
  - Class-incremental segmentation
  - surgical instruments
  - hierarchical prompts
  - forward transfer
  - backward transfer
date: 2026-05-08
content_hash: e6ea18de1f222b73
---

# Unlocking Positive Transfer in Incrementally Learning Surgical Instruments: A Self-reflection Hierarchical Prompt Framework

**Conference**: CVPR 2026
**arXiv**: [2604.02877](https://arxiv.org/abs/2604.02877)
**Code**: No public code available
**Area**: Medical Imaging / Surgical Video Analysis
**Keywords**: Class-incremental segmentation, surgical instruments, hierarchical prompts, forward transfer, backward transfer

## TL;DR

This paper restructures per-instrument prompt parameters from isolated, independent prompts into a tree-structured hierarchy that progressively decomposes shared knowledge across layers. This design enables new instruments to inherit prior knowledge for rapid learning, while allowing new knowledge to gently revise existing representations, thereby simultaneously improving performance on new, regular, and old classes in surgical instrument class-incremental segmentation.

## Background & Motivation

Surgical video segmentation differs fundamentally from general semantic segmentation: instrument categories expand continuously, and data typically arrives in batches. Many surgical instruments appear in only a subset of procedures, while others are present in nearly every operation. Real-world systems therefore more closely resemble a pipeline of continuous capability expansion than a static, one-time training setup with all categories available upfront.

The dominant objective of existing class-incremental segmentation methods is anti-forgetting. While important, this framing carries an overly conservative implicit assumption: old-class knowledge needs only to be preserved, not revisited. The issue is that surgical instruments are highly structured objects — different instruments frequently share shaft morphology, jaw component counts, and overall contour. If each class prompt is trained and frozen independently, the model cannot explicitly reuse knowledge common to all instruments or shared across instruments with the same component structure.

The authors decompose the problem into two directions. First, **forward knowledge transfer**: can knowledge from old instruments help new instruments converge faster? Second, **backward knowledge transfer**: can features learned from new instruments reorganize existing representations, rather than simply leaving old knowledge untouched? Most continual learning work focuses on controlling forgetting but does not systematically exploit either form of positive transfer.

The paper's core observation is direct and compelling. For a new instrument, what truly needs to be learned from scratch is usually only the discriminative specifics of that instrument, not the entire visual vocabulary from edges to parts to contours. For old instruments, the arrival of a new instrument redefines what counts as "distinctive" — a handle shape once considered unique may prove common across multiple instruments, warranting reclassification into a shared layer rather than retention as an overly exclusive memory.

The paper therefore targets not merely reduced forgetting, but a process of continuously reorganizing the knowledge structure as categories expand. The authors propose a Hierarchical Prompt Parsing Tree (HPPT) and a Self-Reflection Refinement Strategy (SRS) to enable knowledge both to be inherited forward and to flow backward.

## Method

### Overall Architecture

The overall framework is built on a frozen large-model backbone, with two backends demonstrated: a conventional DeepLabv3+ and the stronger SAM. Rather than retraining the entire backbone, the authors incrementally append instrument-aware prompts and per-class segmentation heads on top of the pretrained model.

In each incremental episode, the current data contains three types of instruments:

- **Regular classes**: instruments appearing in both historical and current episodes.
- **Old classes**: instruments learned in previous episodes that do not appear in the current one.
- **New classes**: instruments appearing for the first time in the current episode.

The authors' strategy is as follows:

- The backbone encoder body remains frozen; adapters and decoders are trained only in the initial episode.
- Each instrument class corresponds to a set of hierarchical prompts rather than a flat prompt.
- When a new class arrives, instead of learning a full prompt from scratch, only the most class-specific layer is newly created and optimized.
- After new-class learning, a self-reflection pass is triggered to refine existing tree nodes.

This transforms incremental learning from "adding an independent parameter block for each new class" into "attaching a new leaf to a knowledge tree and permitting localized restructuring of the entire tree."

### Key Designs

1. **Hierarchical Prompt Parsing Tree (HPPT)**:

    - **Function**: Decomposes each instrument class's prompt knowledge into a shared layer, a component-shared layer, and an instrument-specific layer, making explicit which knowledge is inheritable.
    - **Mechanism**: Each class prompt consists of three parts. The root node is an instrument-shared partition $P_{IS}$, encoding visual priors common to all surgical instruments; intermediate nodes are organized by component count as $P_{n\text{-}p}$, so that instruments sharing a given number of components share a set of mid-level semantics; leaf nodes $P_{ID}^c$ capture the unique appearance of each class. When a new class with $k$ components arrives, a new leaf is attached under the corresponding $k$-component intermediate node, inheriting the root and intermediate nodes.
    - **Design Motivation**: This design explicitly separates reusable knowledge from knowledge that must be learned anew. Compared to training an entire prompt independently, the learning burden is substantially reduced, and the structure more naturally reflects the hierarchical similarity among fine-grained instrument categories.

2. **Forward Transfer-Driven New-Class Learning**:

    - **Function**: Reduces the learning burden for new instrument prompts, directing the model to focus on discriminative differences rather than redundantly relearning shared properties.
    - **Mechanism**: At $t=1$, the model jointly learns $P_{IS}$ for all initial classes, $P_{n\text{-}p}$ organized by component count, and per-class $P_{ID}$. When a new class $c$ arrives at $t \geq 2$, the already-trained shared components are frozen; only $P_{ID}^c$ and the corresponding segmentation head are newly created and optimized. In cross-attention, the three prompt layers are inserted at different decoder depths, enabling semantic refinement from coarse to fine.
    - **Design Motivation**: If shared knowledge already suffices to represent "this is an instrument" and "it comprises a certain number of structural segments," new classes need only to supplement terminal distinctive details — distal tip morphology, edge shape, handle specifics — significantly reducing the sample complexity of new-class learning.

3. **Self-Reflection Refinement Strategy (SRS)**:

    - **Function**: Allows knowledge learned from new classes to flow back to old classes and shared nodes, improving old knowledge representations without inducing catastrophic forgetting.
    - **Mechanism**: The authors convert the current tree into a directed weighted graph rooted at the new leaf. Edges propagate outward from the new node, with weights decaying exponentially with distance as $\gamma^d$, indicating that nodes closer to new knowledge should be updated more, while distant old knowledge should remain largely stable. A directed graph network performs propagation, with a teleport term ensuring the transition matrix is well-posed. After each new class is added, the system inspects which old features are no longer distinctively exclusive and should be merged into shared layers, and which shared representations should be corrected by new evidence.
    - **Design Motivation**: This step is the paper's primary source of backward transfer. Rather than crudely fine-tuning all historical prompts, update magnitudes are governed by tree distance — large updates for near neighbors, small updates for distant ones — thereby enabling knowledge boundary reorganization while suppressing forgetting.

### Loss & Training

The training objective is fundamentally per-class segmentation cross-entropy, but the set of unlocked parameters varies by phase.

- **Initial episode**: Adapters, decoder, all class segmentation heads, and all three prompt partitions are trained.
- **Subsequent episodes**: Old-class segmentation heads are frozen; for new classes, only $P_{ID}^c$ and the corresponding head are trained.
- **Self-reflection phase**: The main network is frozen; the directed graph propagation module updates tree node representations.

The authors validate on both DeepLabv3+ and SAM, demonstrating that the method is not specific to foundation models but represents a more general strategy for prompt organization and updating.

## Key Experimental Results

### Main Results

Experiments are conducted on two sequential data streams: (1) EndoVis 2017→2018 for nephrectomy instrument segmentation, and (2) CholecSeg8k→M2CAI-Seg for cholecystectomy instrument segmentation. IoU at the final episode and cross-episode BWT/FWT are reported.

| Stream / Method | Old Classes | Regular Classes | New Classes | All | BWT | FWT |
|---|---:|---:|---:|---:|---:|---:|
| Nephrectomy Seq-F | 43.95 | 64.50 | 35.09 | 35.09 | -0.88 | 5.62 |
| Nephrectomy Ours (DeepLabv3+) | 54.81 | 63.98 | 40.92 | 40.92 | 6.22 | 6.96 |
| Nephrectomy Ours (SAM) | 73.48 | 65.67 | 59.44 | 59.44 | 1.55 | 1.22 |
| Nephrectomy Joint-T Upper Bound | 79.79 | 71.45 | 63.16 | 63.16 | — | — |

| Stream / Method | Old Classes | Regular Classes | New Classes | All | BWT | FWT |
|---|---:|---:|---:|---:|---:|---:|
| Cholecystectomy Seq-F | 39.07 | 53.67 | 46.82 | 46.82 | -12.52 | -1.00 |
| Cholecystectomy CAT-SD | 55.14 | 62.39 | 51.74 | 51.74 | -0.13 | 0.92 |
| Cholecystectomy Ours (DeepLabv3+) | 58.08 | 66.18 | 56.12 | 56.12 | 3.24 | 5.70 |
| Cholecystectomy Ours (SAM) | 74.90 | 80.85 | 62.58 | 62.58 | 2.80 | 0.69 |

The most notable aspect of these tables is not any single number but the trend: the proposed method improves not only new-class performance but also old and regular classes simultaneously, indicating that gains stem from both forward and backward transfer rather than a bias toward new classes.

### Ablation Study

The method is decomposed into two key components: `HPPT` (hierarchical prompt tree) and `SRS` (self-reflection strategy). Ablation results confirm that both modules contribute meaningfully.

| Backbone / Configuration | Old Classes | Regular Classes | New Classes | All |
|---|---:|---:|---:|---:|
| SAM baseline | 21.42 | 69.11 | 22.24 | 48.15 |
| SAM + HPPT | 65.29 | 69.23 | 22.16 | 57.89 |
| SAM + HPPT + SRS | 68.52 | 70.67 | 22.30 | 59.44 |
| DeepLabv3+ baseline | 3.93 | 45.99 | 5.37 | 27.62 |
| DeepLabv3+ + HPPT + SRS | 22.46 | 59.79 | 12.19 | 40.92 |

### Key Findings

- HPPT alone yields substantial gains, particularly for old and regular classes, demonstrating that hierarchical inheritance alone is sufficient to structuralize knowledge reuse.
- Adding SRS on top of HPPT yields further improvement, confirming that backward transfer is not superficial: reorganizing the existing tree after new-class learning provides genuine value.
- The SAM variant consistently and substantially outperforms DeepLabv3+, yet both benefit from the same framework, indicating a high degree of backbone independence.
- Compared to Seq-F and various continual baselines, the proposed method frequently flips BWT from negative to positive, suggesting that the improvement is not merely "forgetting less" but constitutes genuine backward improvement.

## Highlights & Insights

- The paper's most significant contribution is transforming prompts from "class-exclusive parameter slots" into "interpretable knowledge structures." Once the structure is explicitly modeled, both forward and backward transfer have concrete attachment points rather than remaining abstract aspirations.
- By decomposing knowledge into three layers — instrument-shared, component-shared, and class-specific — the authors embed the inductive bias of highly structured medical objects into continual segmentation, making this design more problem-appropriate than the memory replay strategies common in general continual learning.
- The directed weighted propagation in SRS is well-conceived. Methods that claim backward transfer often resort to broad fine-tuning of historical parameters; this paper constrains updates by tree-distance decay, achieving a principled balance between "enabling updates" and "avoiding interference."
- Validation on both a CNN-based solution and a foundation model positions the contribution as a knowledge organization layer rather than a backbone-specific trick.

## Limitations & Future Work

- The current tree structure relies on manually defined mid-level semantics such as component count. This is effective for instruments but may not generalize to more complex or continuously varying visual concepts; automatic discovery of intermediate nodes warrants future investigation.
- SRS uses fixed-decay graph propagation, with update magnitudes still governed by a hand-designed distance function. If the true relationships between categories deviate from tree distance, suboptimal updates may result.
- Experiments cover only two surgical data streams with a relatively small number of episodes. Whether the hierarchical tree becomes excessively deep or sparse as the number of categories scales to tens or more remains to be validated.
- The method implicitly requires prior knowledge of which component-count category a new instrument belongs to. A fully automated system would need to predict such structural attributes jointly.
- A promising future direction is extending this hierarchical prompt framework to vision-language surgical assistants, enabling detection, segmentation, and question answering to share a single cross-task knowledge tree rather than operating on single-task segmentation alone.

## Related Work & Insights

- **vs. independent prompt bank methods**: Prior work stores each class prompt independently — maximally stable but minimally reusable. This paper organizes prompts into a hierarchical structure, substantially improving reusability.
- **vs. typical continual segmentation methods**: SI, LwF, MiB, and PLOP primarily address distillation, regularization, and old-knowledge preservation. This paper additionally answers how old knowledge aids new classes and how new classes benefit old knowledge.
- **vs. foundation model adaptation methods**: Much prior work treats SAM merely as a strong backbone for freeze-and-tune adaptation. This paper genuinely explores sustainable category expansion on top of SAM, which is particularly valuable in medical settings.
- A broader insight is that incremental learning need not be organized around replay buffers. For category systems with clear structural regularity, decomposing knowledge into layers and defining transferable paths may be more effective than pure memory replay.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Simultaneously incorporating forward and backward transfer into class-incremental surgical segmentation, instantiated via a hierarchical prompt tree, constitutes a complete and well-motivated contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Dual data streams, dual backbones, upper bounds, and multiple baselines provide solid evidence; longer episode sequences would further strengthen the case.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly stated; method diagrams and training procedures are well-organized, and the correspondence between formulations and structural components is easy to follow.
- **Value**: ⭐⭐⭐⭐⭐ Highly relevant to real-world clinical continual updating scenarios; particularly suitable as a foundational approach for continuous adaptation of medical foundation models.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis](focus-to-perceive_representation_learning_a_cognition-inspired_hierarchical_fram.md)
- [\[CVPR 2026\] Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)
- [\[CVPR 2026\] Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning](forecasting_epileptic_seizures_from_contactless_ca.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[CVPR 2026\] Benchmarking Endoscopic Surgical Image Restoration and Beyond](benchmarking_endoscopic_surgical_image_restoration_and_beyond.md)

<!-- RELATED:END -->
