---
title: >-
  [Paper Note] CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment
description: >-
  [CVPR2026][Multimodal VLM][prompt tuning] This paper proposes CAPT, a confusion-aware prompt tuning framework that explicitly models systematic misalignment patterns in VLMs via a Semantic Confusion Miner (SEM) and a Sample Confusion Miner (SAM). A Multi-Granularity Discrepancy Expert (MGDE) further integrates confusion information across different granularities. CAPT achieves a state-of-the-art HM of 83.90% across 11 benchmarks.
tags:
  - CVPR2026
  - Multimodal VLM
  - prompt tuning
  - vision-language alignment
  - confusion-awareness
  - CLIP
  - few-shot
  - fine-grained classification
date: 2026-05-08
content_hash: a0510a9aab369b26
---

# CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment

**Conference**: CVPR2026
**arXiv**: [2603.02557](https://arxiv.org/abs/2603.02557)
**Code**: [github.com/greatest-gourmet/CAPT](https://github.com/greatest-gourmet/CAPT)
**Area**: Multimodal VLM
**Keywords**: prompt tuning, vision-language alignment, confusion-awareness, CLIP, few-shot, fine-grained classification

## TL;DR
This paper proposes CAPT, a confusion-aware prompt tuning framework that explicitly models systematic misalignment patterns in VLMs via a Semantic Confusion Miner (SEM) and a Sample Confusion Miner (SAM). A Multi-Granularity Discrepancy Expert (MGDE) further integrates confusion information across different granularities. CAPT achieves a state-of-the-art HM of 83.90% across 11 benchmarks.

## Background & Motivation

### Starting Point

**Goal**: **Background**: 1. Vision-language models such as CLIP exhibit systematic misalignment: confusion between specific class pairs is not random but occurs as persistent, fixed patterns.
2. For example, in the OxfordPets dataset, *terrier* is misclassified as *bulldog* 30 times yet is almost never confused with other classes.
3. Existing prompt tuning methods (MaPLe, PromptSRC) optimize global image-text feature alignment while ignoring such fixed confusion patterns.
4. Model misalignment arises from ambiguous semantic boundaries and locally similar representations among visually and semantically similar classes.
5. Models should be made to learn from their own misalignment — by explicitly modeling confusion relationships and correcting them.
6. Prior methods do not mine discriminative fine-grained cues from confused samples.

## Method

### Overall Architecture
Building upon CLIP prompt tuning, CAPT introduces: (1) a Confusion Bank that records misclassified samples; (2) SEM to mine semantic-level confusion patterns; (3) SAM to mine sample-level confusion cues; and (4) MGDE to fuse multi-granularity information.

### Key Designs

**Confusion Bank**: Records, for each sample, the class to which it is misclassified, thereby forming an inter-class confusion index. Using pseudo-GT (the class with the highest model confidence) instead of annotated GT better reflects the model's actual confusion tendencies.

**SEM (Semantic Confusion Miner)**:
- Computes a confusion score using confusion statistics $n_i$ and the current sample confidence $C_i$: $S_i = (1 + \frac{n_i}{\sum n_i}) C_i$
- Selects top-$k$ pairs to construct semantic confusion pairs
- Employs an LLM (in a CoT-style manner) to generate *commonality* and *difference* prompts for each confusion pair

**SAM (Sample Confusion Miner)**:
- Retrieves a confused sample set $U \in \mathbb{R}^{c \times l}$ from the Confusion Bank
- Selects the representative confused sample most similar to the current instance: $I_c^* = \arg\max \cos(E_I(I), E_I(U_j^i))$
- Diff-Manner Adapter: fuses ViT global attention with 2D depthwise separable convolution for local detail
  $$[X] \leftarrow [X] + \alpha \cdot DWConv2D(\hat{[X]})$$

**MGDE (Multi-Granularity Discrepancy Expert)**:
- MoE architecture comprising semantic experts (initialized from textual difference/commonality prompts) and sample experts (initialized from CLIP FFN weights)
- K-means clustering compresses prompt tokens by removing low-discriminability tokens
- A routing network adaptively determines the output weight of each expert

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{ori} + \mathcal{L}_{confuse}$$
- $\mathcal{L}_{ori}$: standard cross-entropy alignment loss
- $\mathcal{L}_{confuse}$: InfoNCE-style contrastive loss applied to confused sample features and prompts

## Key Experimental Results

### Main Results: Base-to-New Generalization (16-shot)

| Method | Base | Novel | HM |
|--------|------|-------|----|
| CoOp (IJCV'22) | 82.69 | 63.22 | 71.66 |
| MaPLe (CVPR'23) | 82.28 | 75.14 | 78.55 |
| PromptKD (CVPR'24) | 86.96 | 80.73 | 83.73 |
| TAC (CVPR'25) | 85.42 | 77.60 | 81.24 |
| 2SFS (CVPR'25) | 85.55 | 75.48 | 80.20 |
| **CAPT (Ours)** | **87.41** | **80.90** | **83.90** |

### Confused Sample Correction Rate

| Metric | Value |
|--------|-------|
| Confusion pair correction rate | 50.72% |
| Base class accuracy | 87.41% |
| Novel class accuracy | 80.90% |

### Key Findings
- CAPT surpasses all prior methods on HM with notable improvements on both Base and Novel classes.
- 50.72% of confused sample pairs are successfully corrected, validating the effectiveness of confusion-aware learning.
- SEM and SAM contribute complementary signals — the semantic level captures global confusion patterns while the sample level captures fine-grained discriminative cues.

## Highlights & Insights
- Distinctive perspective: improvement signals are mined from the model's own misalignment, turning a "bug" into a "feature."
- The Confusion Bank is simple yet effective, offering a reusable tool for subsequent research.
- The Diff-Manner Adapter's strategy of fusing ViT global representations with CNN local features has broad applicability.

## Limitations & Future Work
- The quality of pseudo-GT depends on the model's initial predictive capability; weaker models may yield excessively noisy Confusion Banks.
- The quality and consistency of LLM-generated semantic prompts are not sufficiently analyzed.
- Increased model complexity (SEM + SAM + MGDE) may lead to substantially higher training costs compared to simpler prompt tuning approaches.

## Related Work & Insights
- Fundamental distinction from PromptSRC/MaPLe: CAPT focuses not on achieving better alignment, but on learning from misalignment.
- The Confusion Bank concept is transferable to hard negative mining in contrastive learning.
- Insight: systematic error patterns of models are themselves valuable signals worth exploiting across a broader range of tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (the confusion-aware perspective is highly original; learning from misalignment is a fresh paradigm)
- Experimental Thoroughness: ⭐⭐⭐⭐ (11 datasets with comprehensive ablations and visualizations)
- Writing Quality: ⭐⭐⭐⭐ (architecture diagrams are clear; method description is systematic)
- Value: ⭐⭐⭐⭐ (opens a new direction for prompt tuning; the confusion correction rate metric is thought-provoking)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Towards Calibrating Prompt Tuning of Vision-Language Models](towards_calibrating_prompt_tuning_of_vision-language_models.md)
- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[ICLR 2026\] A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models](../../ICLR2026/multimodal_vlm/a-tpt_angular_diversity_calibration_properties_for_test-time_prompt_tuning_of_vi.md)
- [\[ICLR 2026\] Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts](../../ICLR2026/multimodal_vlm/revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)
- [\[CVPR 2026\] Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](noise-aware_few-shot_learning_through_bi-directional_multi-view_prompt_alignment.md)

<!-- RELATED:END -->
