---
title: >-
  [Paper Note] No More Sibling Rivalry: Debiasing Human-Object Interaction Detection
description: >-
  [ICCV 2025][Social Computing][Human-Object Interaction Detection] This paper identifies and systematically analyzes the "Toxic Siblings Bias" in HOI detection—highly similar HOI triplets that mutually interfere and compe…
tags:
  - "ICCV 2025"
  - "Social Computing"
  - "Human-Object Interaction Detection"
  - "Toxic Siblings Bias"
  - "Contrastive-Calibration Learning"
  - "Merge-then-Split Strategy"
  - "Debiasing"
date: 2026-05-08
content_hash: 2b1b0ee32ba48bc7
---

# No More Sibling Rivalry: Debiasing Human-Object Interaction Detection

**Conference**: ICCV 2025
**arXiv**: [2509.00760](https://arxiv.org/abs/2509.00760)
**Code**: None
**Area**: Social Computing
**Keywords**: Human-Object Interaction Detection, Toxic Siblings Bias, Contrastive-Calibration Learning, Merge-then-Split Strategy, Debiasing

## TL;DR

This paper identifies and systematically analyzes the "Toxic Siblings Bias" in HOI detection—highly similar HOI triplets that mutually interfere and compete at both the input and output levels. Two debiasing learning objectives are proposed: Contrastive-then-Calibration (C2C) and Merge-then-Split (M2S), achieving +9.18% mAP over the baseline and +3.59% over the previous state-of-the-art on HICO-DET.

## Background & Motivation

Human-Object Interaction (HOI) detection aims to recognize interactions between humans and objects in images, formalized as ⟨human, action, object⟩ triplets. DETR-based methods have achieved notable progress, yet the authors identify a critical problem: the **Toxic Siblings Bias**.

**What is the Toxic Siblings Bias?**

The bias manifests at two levels:

**Input-level bias (within the same image)**: Multiple HOI triplets sharing the same object or action frequently co-occur within a single image. For example, a person's interaction with a stool may be misclassified as "sitting" rather than "standing" because all other people in the image are seated. Such similar and mutually interfering features make it difficult for the interaction decoder to learn discriminative representations.

**Output-level bias (across images)**: Even across different images, certain categories exhibit strong semantic similarity—e.g., "look at bird," "feed bird," "hold bird," and "release bird"—leading to highly similar classification heads that cause confusion and competition.

**How severe is this bias?**

Statistical analysis reveals three key findings:
- The learning curves of sibling categories exhibit **mutual suppression**: as one improves, another degrades.
- At the input level, the presence of sibling HOIs in an image increases the probability of complete prediction failure (mAP=0) by an average of **27.59%**.
- At the output level, every 0.1 increase in cosine similarity of classification head initializations corresponds to a drop of **5.51 AP**—the more siblings, the harder the learning.

The paper's goal is to **mitigate the bias solely by modifying the learning mechanism**, without relying on additional pretraining data or heavy external models such as LLMs or diffusion models.

## Method

### Overall Architecture

Building upon the single-stage two-branch HOI detector GEN-VLKT (visual encoder → instance decoder → interaction decoder), the main architecture is preserved, and two auxiliary learning objectives are introduced only within the interaction decoder.

### Key Designs

1. **Contrastive-then-Calibration (C2C) — Addressing Input-Level Bias**

   **Core Idea**: Train the interaction decoder to resist sibling bias by prioritizing spatial cues to focus on the correct interaction region.

   **Positive/Negative Sample Definition**: For HOI triplet $t_i^{hoi}$, the positive set $\mathcal{P}_i$ contains triplets in the same image sharing identical interaction-object labels, while the negative set $\mathcal{N}_i$ contains triplets sharing either the interaction or object label but not both.

   **Triplet Feature Representation**: Semantic features are concatenated with human/object spatial features:
   $\hat{v}_i^{tri} = \text{cat}(\hat{q}_i^a, \hat{f}_i^h, \hat{f}_i^o)$

   **Contrastive Loss**: Enlarges distance to negative samples and reduces distance to positive samples:
   $\mathcal{L}_{con} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\sum_{v_j^{tri}\in\mathcal{P}_i}\exp(\hat{v}_i^{tri}\odot v_j^{tri}/\tau)}{\sum_{v_k^{tri}\in\mathcal{E}_i}\exp(\hat{v}_i^{tri}\odot v_k^{tri}/\tau)}$

   **Calibration Loss**: The semantic features of positive samples are **replaced** with incorrect semantic features randomly sampled from the negative set, while correct spatial features (strong spatial prior) are retained; the interaction decoder is required to **correct** them back to the proper triplet:
   $\underline{v}_i^{tri} = \text{cat}(F_{text}(s^o, a), f_i^h, f_i^o), \quad (s^o, a) = \text{sample}(\mathcal{N}_i)$

   Correction quality is evaluated via an L1 loss. Attention masks ensure the auxiliary queries do not interfere with the original queries.

2. **Merge-then-Split (M2S) — Addressing Output-Level Bias**

   **Design Motivation**: Directly requiring the model to distinguish a large number of highly similar HOI categories is susceptible to sibling bias. A more effective approach is to **first learn commonalities, then learn distinctions**.

   **Merge Phase**: CLIP text features are used to cluster interaction categories and object categories separately into $M_1$ super-interaction classes and $M_2$ super-object classes, forming $M = M_1 \times M_2$ super-categories. A super-category classification head is added to the interaction decoder and trained with cross-entropy loss:
   $\mathcal{L}_{merge} = \text{CrossEntropy}(\{m_i\}_{i=1}^N, \{\hat{m}_i\}_{i=1}^N)$

   This allows the model to first learn to distinguish different "groups," leveraging the shared features among siblings.

   **Split Phase**: For each query, the $k_1$ most similar categories are identified; the $k_2$ most frequent category features across all queries are then concatenated with the image features. A fine-grained discriminative loss requires the model to correctly classify among these most confusable categories:
   $\mathcal{L}_{split} = -\frac{1}{N_{split}}\sum_{i=1}^{N_{split}}\log\frac{\exp(q_i^a \odot t_i / \tau)}{\sum_{j=1}^{k_2}\exp(q_i^a \odot t_j / \tau)}$

### Loss & Training

The total loss is a weighted sum of five terms:
$$\mathcal{L}_{all} = \lambda_1\mathcal{L}_{detector} + \lambda_2\mathcal{L}_{con} + \lambda_3\mathcal{L}_{cal} + \lambda_4\mathcal{L}_{merge} + \lambda_5\mathcal{L}_{split}$$

where $\lambda_1=1, \lambda_2=1, \lambda_3=0.5, \lambda_4=1, \lambda_5=0.5$. The AdamW optimizer is used for 90 epochs with an initial learning rate of $10^{-4}$, decayed by a factor of 10 at epoch 60. Optimal hyperparameters: $k_1=2, k_2=10, M=25$ ($M_1 \times M_2 = 5 \times 5$).

## Key Experimental Results

### Main Results

On the HICO-DET dataset (R50 + CLIP):

| Method | VLM | Full | Rare | Non-Rare |
|--------|-----|------|------|----------|
| GEN-VLKT (baseline) | CLIP | 33.75 | 29.25 | 35.10 |
| HOICLIP | CLIP | 34.69 | 31.12 | 35.74 |
| ADA-CM | CLIP | 38.40 | 37.52 | 38.66 |
| BCOM | CLIP | 39.34 | 39.90 | 39.17 |
| **Ours** | **CLIP** | **42.93** | **42.41** | **43.11** |

On the V-COCO dataset: AP^S1=69.8, AP^S2=72.1, both state-of-the-art.

Under the R50 + BLIP2 configuration: Full=43.98, surpassing SICHOI (41.79) which uses LLM+SAM.

### Ablation Study

| Configuration | Full | Rare | Non-Rare | Notes |
|---------------|------|------|----------|-------|
| Baseline (GEN-VLKT-s) | 33.75 | 29.25 | 35.10 | Baseline |
| + HOR mask | 35.37 | 30.70 | 36.51 | Attention mask |
| + Contrastive Loss | 36.82 | 32.29 | 37.92 | Contrastive learning |
| + Calibration Loss | 38.03 | 34.05 | 39.80 | Calibration loss +1.21% |
| + Merge Loss | 39.88 | 36.12 | 40.20 | Merge objective +1.85% |
| + Split Loss | **42.93** | **42.41** | **43.11** | Split objective +3.05% |

Hyperparameter ablations confirm $k_1=2, k_2=10, M_1 \times M_2 = 5 \times 5$ as the optimal configuration.

### Key Findings

- Rare categories exhibit the largest gain (29.25→42.41, +13.16%), validating that the M2S strategy effectively leverages head-category data to benefit tail categories.
- The R50 backbone outperforms CyCleHOI using R101, demonstrating high parameter efficiency.
- The method remains competitive in zero-shot settings (RF=35.48, comparable to supervised methods), indicating the generalizability of the debiasing strategy.
- The approach surpasses methods relying on heavy external models such as Stable Diffusion, LLMs, and SAM, without using any of them.

## Highlights & Insights

- **Insightful problem formulation**: The Toxic Siblings Bias is systematically quantified through learning curves and statistical analysis rather than intuition alone.
- **Elegant input-level debiasing via "contrastive-then-calibrate"**: Training the decoder's correction capability using incorrect semantics paired with correct spatial features is both concise and effective.
- **Output-level "merge-then-split" aligns with human cognition**: Learning coarse categories before fine-grained distinctions cleverly exploits shared features among siblings.
- **Lightweight design**: Only learning objectives are modified; the inference architecture and computational overhead remain unchanged.

## Limitations & Future Work

- The selection of clustering hyperparameters $M_1, M_2$ relies on manual tuning; automatic determination of the optimal cluster count would improve generalizability.
- Validation is limited to static image HOI; temporal sibling bias in video HOI detection warrants further exploration.
- The negative sampling strategy in C2C is relatively simple (random sampling); incorporating harder negative mining may yield further improvements.

## Related Work & Insights

- The concept of "sibling bias" is generalizable to other fine-grained recognition tasks, such as fine-grained object classification and relation detection.
- The hierarchical learning strategy of merge-then-split is conceptually aligned with curriculum learning and is applicable to long-tail distribution problems.
- The two-stage contrastive-and-calibration strategy offers a general-purpose approach for handling semantic confusion at the input level.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Insightful problem identification; both debiasing objectives are novel in design
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple datasets, backbones, VLMs, zero-shot settings, and detailed ablations
- Writing Quality: ⭐⭐⭐⭐ Clear analysis with rich visualizations
- Value: ⭐⭐⭐⭐⭐ Substantial margin over SOTA on HICO-DET; broadly transferable methodology

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction](../../ICLR2026/social_computing/human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction.md)
- [\[ICLR 2026\] Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation](../../ICLR2026/social_computing/adaptive_debiasing_tsallis_entropy_for_test-time_adaptation.md)
- [\[ICCV 2025\] Gradient Extrapolation for Debiased Representation Learning](gradient_extrapolation_for_debiased_representation_learning.md)
- [\[NeurIPS 2025\] Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector](../../NeurIPS2025/social_computing/any_large_language_model_can_be_a_reliable_judge_debiasing_w.md)
- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](learning_visual_proxy_for_compositional_zero-shot_learning.md)

</div>

<!-- RELATED:END -->
