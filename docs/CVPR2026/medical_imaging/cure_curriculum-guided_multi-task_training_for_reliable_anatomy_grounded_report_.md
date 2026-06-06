---
title: >-
  [Paper Note] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation
description: >-
  [CVPR 2026][Medical Imaging][Curriculum Learning] This paper proposes CURE — an error-aware curriculum learning framework for multi-task training that dynamically adjusts sampling distributions to emphasize hard samples…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Curriculum Learning"
  - "Visual Grounding"
  - "Radiology Report Generation"
  - "Multi-task Learning"
  - "Hallucination Mitigation"
date: 2026-05-08
content_hash: 8e9166d59fe2003c
---

# CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation

**Conference**: CVPR 2026
**arXiv**: [2601.15408](https://arxiv.org/abs/2601.15408)  
**Code**: [Available](https://github.com/PabloMessina/CURE)  
**Area**: Medical Imaging
**Keywords**: Curriculum Learning, Visual Grounding, Radiology Report Generation, Multi-task Learning, Hallucination Mitigation

## TL;DR

This paper proposes CURE — an error-aware curriculum learning framework for multi-task training that dynamically adjusts sampling distributions to emphasize hard samples, improving visual grounding accuracy by +0.37 IoU and reducing hallucination rate by 18.6% without introducing additional data.

## Background & Motivation

**Background**: Medical vision-language models (VLMs) such as MAIRA-2 and MedGemma can automatically generate radiology reports from medical images and have achieved strong performance on multiple benchmarks.

**Limitations of Prior Work**: Existing models lack reliable visual grounding capability — described lesions cannot be accurately mapped to their corresponding image regions, leading to frequent hallucinations (e.g., as shown in Figure 1, MAIRA-2 falsely reports fractures in normal clavicle regions).

**Key Challenge**: Conventional phrase grounding training data is naturally skewed toward abnormal findings, causing models to over-associate normal anatomical regions with abnormal labels and yield high false-positive rates. Additionally, severe scale imbalances across datasets (Chest ImaGenome: 12.9M vs. MS-CXR: only 815 samples) cause small datasets to be overwhelmed under standard proportional sampling.

**Goal**: To simultaneously improve visual grounding accuracy and factual consistency of generated reports without introducing additional private data.

**Key Insight**: Drawing inspiration from Curriculum Learning — rather than fixing sampling ratios, CURE dynamically adjusts sampling weights based on the model's current performance, directing more attention toward data sources and anatomical categories that are not yet well learned.

**Core Idea**: A two-level error-aware curriculum learning approach — dynamically reweighting sampling probabilities at both inter-dataset and intra-dataset (category-level) granularities based on model evaluation errors, combined with anatomy-level fine-grained task decomposition (AGRG) that generates multiple training instances from a single image, enabling efficient utilization of existing public data.

## Method

### Overall Architecture

CURE uses MedGemma-4B-IT as the backbone and applies LoRA (rank=16, 4-bit) fine-tuning in two stages:

1. **Pre-training Stage** (3,000 steps): Warm up grounding capability on Chest ImaGenome.
2. **Multi-task Curriculum Stage** (6,000 steps): Jointly train three task types — PG, GRG, and AGRG — across three public datasets, with error-aware sampling weight updates every 3,000 steps.

All training instances are unified into `(image, instruction, response)` triplet format.

### Key Designs

#### 1. Error-Aware Curriculum Learning

- **Function**: Periodically evaluates model performance on each data source and category during training, and dynamically adjusts sampling probabilities based on observed errors.
- **Mechanism**:
    - **Inter-Dataset Level**: A composite score is computed for each data source $D_i$ as $s_i = \alpha \cdot \text{IoU}_i + (1-\alpha) \cdot \text{CXRFEScore}_i$, with error $e_i = 1 - s_i$ and next-round sampling probability $p_i = e_i / \sum_j e_j$ — data sources with worse performance are sampled more frequently.
    - **Intra-Dataset Level**: MS-CXR is reweighted across 8 phrase categories; Chest ImaGenome is reweighted across 29–38 anatomical regions; PadChest-GR remains uniformly sampled due to its multi-label nature.
- **Design Motivation**: Standard proportional sampling allows Chest ImaGenome (12.9M samples) to dominate training while small datasets like MS-CXR (815 samples) are nearly ignored. Similarly, high-frequency anatomical regions tend to overfit while clinically important but low-frequency regions are under-learned.

#### 2. Anatomy-Level Fine-Grained Task Decomposition (AGRG)

- **Function**: Decomposes Chest ImaGenome scene graphs into three sub-tasks.
- **Mechanism**:
    - **Locate**: Given an anatomical location name → output bounding box `[cx, cy, w, h]` (36 locations).
    - **Describe**: Given an anatomical location name → output textual description (38 locations).
    - **Locate & Describe**: Output both localization and description simultaneously (29 locations).
    - The three sub-tasks are sampled uniformly to maintain task balance; a single image can yield 9–36 training instances.
- **Design Motivation**: Explicitly decoupling spatial localization from textual description allows the model to learn each skill independently before combining them. Fine-grained decomposition also expands ~237K images into 12.9M training instances, greatly improving data utilization efficiency.

#### 3. Unified Instruction Format and Data Augmentation

- **Function**: Unifies PG (Phrase Grounding), GRG (Grounded Report Generation), and AGRG into an instruction-following format.
- **Mechanism**:
    - PG: `"Ground the phrase: {phrase}"` → `"phrase: [cx,cy,w,h]..."`
    - GRG: `"Generate a grounded report"` → full report with bounding box coordinates
    - PadChest-GR additionally generates label-box pairs from sentence-box pairs, nearly doubling PG training data.
- **Design Motivation**: Unifying heterogeneous supervision signals (bounding boxes, phrases, descriptions, labels) under a single template enables parameter sharing and reduces task conflicts.

### Loss & Training

- Standard autoregressive language modeling loss (next-token prediction).
- Optimizer: AdamW, learning rate $2 \times 10^{-4}$, linear schedule with 0.03 warmup ratio.
- Effective batch size = 25 (per-device 5 × gradient accumulation 5).
- Gradient clipping with max_norm = 0.3.
- Data augmentation: spatial transformations + CLAHE (Contrast Limited Adaptive Histogram Equalization).
- Optimizer states are independently initialized for the pre-training and multi-task stages; only model weights are carried over.

## Key Experimental Results

### Main Results

**Table 1: Phrase Grounding IoU**

| Model | MS-CXR Mi.↑ | MS-CXR Ma.↑ | PadChest Mi.↑ | PadChest Ma.↑ | VinDr (Zero-shot) Mi.↑ | VinDr Ma.↑ |
|-------|-------------|-------------|---------------|---------------|------------------------|------------|
| MAIRA-2 | 0.496 | 0.452 | 0.280 | 0.287 | 0.162 | 0.115 |
| **CURE** | **0.554** | **0.495** | **0.453** | **0.438** | **0.244** | **0.205** |

CURE outperforms MAIRA-2 across all datasets and metrics, with particularly notable gains on PadChest-GR (+0.173 / +0.151).

**Table 2: Anatomy Grounded Report Generation (AGRG) — Chest ImaGenome**

| Model | IoU↑ | F1-Mi↑ | F1-Ma↑ | Cos.↑ | CXRFEScore↑ |
|-------|------|--------|--------|-------|-------------|
| MAIRA-2 | 0.226 | 0.272 | 0.100 | 0.557 | 0.360 |
| MedGemma-4B-IT | — | 0.344 | 0.294 | 0.631 | 0.477 |
| **CURE** | **0.596** | **0.474** | 0.273 | **0.649** | **0.548** |

IoU improves from 0.226 → 0.596 (+0.37), more than doubling grounding accuracy; CXRFEScore improves by +0.188.

**Table 3: MIMIC-CXR Report Generation**

| Model | F1-Ma↑ | F1-Mi↑ | Cos.↑ | CXRFEScore↑ | RadF1↑ |
|-------|--------|--------|-------|-------------|--------|
| CXRMate-RRG24 | 0.414 | 0.589 | 0.764 | 0.656 | **0.255** |
| MAIRA-2 (w/ grounding) | 0.304 | 0.490 | 0.751 | 0.603 | 0.120 |
| CURE (AGRG+GRG) | **0.415** | 0.562 | **0.792** | 0.655 | 0.176 |

CURE achieves the best CheXbert Cosine Similarity (0.792) and F1-Ma (0.415), with mixed comparisons against the competition winner CXRMate-RRG24.

### Ablation Study

**Table 4: Stepwise Ablation (excerpted from Table 7)**

| Configuration | AGRG IoU↑ | AGRG CXRS↑ | GRG IoU↑ | MS-CXR PG↑ | PadChest PG↑ | VinDr PG↑ |
|---------------|-----------|-----------|----------|------------|-------------|----------|
| v1: Base | 0.393 | 0.565 | 0.171 | 0.389 | 0.356 | 0.192 |
| v2: +Aug | 0.378 | 0.552 | 0.185 | 0.406 | 0.365 | 0.205 |
| v5: +Aug+CL(3k) | 0.419 | 0.552 | 0.180 | 0.432 | 0.393 | 0.205 |
| v8: +Aug+CIG(3k)+CL(3k) | 0.469 | 0.546 | 0.206 | 0.497 | 0.422 | 0.225 |
| **v9 (CURE)**: +HPS | **0.596** | **0.548** | **0.265** | **0.554** | **0.453** | **0.244** |

- Curriculum reweighting every 3,000 steps yields the best results (compared to 1.5k and 2k frequencies).
- Extending CIG pre-training from 1k to 3k steps consistently improves grounding.
- Hyperparameter search (HPS) provides the final leap, improving AGRG IoU from 0.469 → 0.596.

### Key Findings

1. **Significant hallucination reduction**: CURE achieves an average abnormal hallucination rate of 8.78% vs. 26.50% for MAIRA-2 (−67%); the clavicle region is particularly striking — MAIRA-2 hallucination rate is 59–63%, while CURE's is only 1%.
2. **Contradiction rate halved**: Under NLI evaluation, CURE's contradiction rate is 17.44% vs. 33.22% for MAIRA-2; entailment rate is 39.50% vs. 15.94%.
3. **Zero-shot generalization**: On VinDr-CXR (unseen during training), CURE achieves PG IoU of 0.244 vs. 0.162 for MAIRA-2.
4. **No private data required**: CURE, trained exclusively on public data, surpasses MAIRA-2 in IoU despite MAIRA-2 using 190K private reports.

## Highlights & Insights

- **Two-level curriculum learning is the key innovation**: Simultaneously addresses inter-dataset and intra-category imbalance without introducing additional network modules.
- **Fine-grained task decomposition dramatically improves data efficiency**: ~237K images → 12.9M training instances, with only ~1.74% of them used per training run.
- **AGRG replaces the traditional finding-generation objective**: Exposing the model to both normal and abnormal descriptions fundamentally mitigates false-positive bias.
- **Grounding capability acquired from scratch**: MedGemma-4B-IT originally lacks visual grounding; after CURE training, its grounding performance surpasses the purpose-built MAIRA-2.
- **Low training cost**: LoRA rank=16 / 4-bit quantization, completed in 9,000 steps without requiring large-scale compute resources.

## Limitations & Future Work

1. **Slightly weaker report-level text quality**: On PadChest-GR GRG, text metrics remain below MAIRA-2 trained on private data (F1-Mi: 0.507 vs. 0.592).
2. **Validated only on chest X-rays**: All experiments are limited to the CXR domain; transferability to CT, MRI, pathology, and other modalities remains unexplored.
3. **Cardiac silhouette hallucination increases**: CURE's hallucination rate in the cardiac silhouette region (25.67%) is higher than MAIRA-2's (2.00%), suggesting that curriculum learning may over-correct for certain categories.
4. **Low RadGraph F1**: CURE's RadF1 (0.176) is substantially lower than CXRMate-RRG24 (0.255), which specifically optimizes this metric via RL with RadF1 reward.
5. **Curriculum evaluation relies on Gemini for NLI**: Hallucination assessment depends on an external LLM, which may introduce evaluation bias.

## Related Work & Insights

- **MAIRA-2**: The primary baseline, also performing multi-task grounding and report generation, but relying on the private USMix dataset and lacking curriculum learning.
- **MedGemma-4B-IT**: The backbone of CURE; its original lack of visual grounding demonstrates the generalizability of the CURE framework.
- **CXRMate-RRG24**: The competition winner trained with RL + RadGraph F1 reward, leading on RadF1 but lacking grounding capability.
- **Self-Paced Curriculum Learning (SPCL)**: CURE's curriculum strategy can be viewed as an extension of SPCL to medical multi-task scenarios, replacing loss-based difficulty estimation with a dual-metric (IoU + CXRFEScore) measure.
- **Broader Implication**: The error-aware sampling strategy is generalizable to any multi-dataset, multi-task training setting (e.g., multi-sensor fusion in autonomous driving, multilingual NLP).

## Rating

⭐⭐⭐⭐ The method is elegant and efficient, with thorough and rigorous experiments. CURE achieves significant improvements in grounding accuracy and reliability without increasing data volume or model complexity. Limitations include room for improvement in text generation quality and validation restricted to the CXR domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CAME-Grad: The Double Dilemma in Multi-Task Radiology Report Generation — A Gradient Dynamics Analysis and Solution](../../ICML2026/medical_imaging/the_double_dilemma_in_multi-task_radiology_report_generation_a_gradient_dynamics.md)
- [\[ACL 2026\] CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning](../../ACL2026/medical_imaging/cure-med_curriculum-informed_reinforcement_learning_for_multilingual_medical_rea.md)
- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)
- [\[AAAI 2026\] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation](../../AAAI2026/medical_imaging/priorrg_prior-guided_contrastive_pre-training_and_coarse-to-fine_decoding_for_ch.md)
- [\[AAAI 2026\] GuideGen: A Text-Guided Framework for Paired Full-Torso Anatomy and CT Volume Generation](../../AAAI2026/medical_imaging/guidegen_a_text-guided_framework_for_paired_full-torso_anatomy_and_ct_volume_gen.md)

</div>

<!-- RELATED:END -->
