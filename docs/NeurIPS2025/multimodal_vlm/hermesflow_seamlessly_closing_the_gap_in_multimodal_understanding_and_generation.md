---
title: >-
  [Paper Note] HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation
description: >-
  [NeurIPS 2025][Multimodal VLM][Pair-DPO] This work is the first to identify the systematic phenomenon that understanding capability consistently surpasses generation capability in unified multimodal large language models. It proposes the HermesFlow framework, which constructs paired understanding-generation preference data from homologous inputs, and employs Pair-DPO with iterative self-play optimization to simultaneously improve both capabilities and narrow the gap between them—without relying on any external high-quality data.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Pair-DPO
  - multimodal alignment
  - understanding-generation gap
  - self-play optimization
  - homologous preference data
date: 2026-05-08
content_hash: 0af63aa0157b42a9
---

# HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation

**Conference**: NeurIPS 2025
**arXiv**: [2502.12148](https://arxiv.org/abs/2502.12148)
**Code**: [https://github.com/Gen-Verse/HermesFlow](https://github.com/Gen-Verse/HermesFlow)
**Area**: Multimodal VLM / Unified Understanding and Generation / Preference Alignment
**Keywords**: Pair-DPO, multimodal alignment, understanding-generation gap, self-play optimization, homologous preference data

## TL;DR
This work is the first to identify the systematic phenomenon that understanding capability consistently surpasses generation capability in unified multimodal large language models. It proposes the HermesFlow framework, which constructs paired understanding-generation preference data from homologous inputs, and employs Pair-DPO with iterative self-play optimization to simultaneously improve both capabilities and narrow the gap between them—without relying on any external high-quality data.

## Background & Motivation

**Background**: Unified MLLMs such as Show-o, Transfusion, and Emu3 have demonstrated impressive performance by handling both multimodal understanding and image generation within a single Transformer architecture.

**Limitations of Prior Work**: Existing studies such as Liquid and MetaMorph investigate the synergy between understanding and generation from a data perspective, yet overlook a critical phenomenon—**understanding capability systematically outperforms generation capability**, with a significant gap between the two.

**Key Challenge**: Simply increasing training data for either understanding or generation does not proportionally improve both; the capability imbalance persists after pre-training. Existing DPO methods either optimize understanding alone (e.g., CSR) or generation alone (e.g., Emu3), and cannot jointly improve both.

**Goal**: How can both understanding and generation capabilities be simultaneously improved and their gap reduced, without relying on external high-quality data?

**Key Insight**: The paper leverages an intrinsic property of MLLMs—that understanding is stronger than generation—using the model's superior understanding ability to evaluate generation quality, and constructing paired preference data for both modalities from homologous input data.

**Core Idea**: Pair-DPO combined with iterative self-play optimization; paired preference data are constructed from homologous inputs to jointly optimize understanding and generation.

## Method

### Overall Architecture
**Input**: A set of homologous (image $x$, caption/prompt $y$) data pairs. The pipeline proceeds in three steps: (1) construct homologous preference data; (2) jointly optimize via Pair-DPO; (3) iterate via self-play. **Output**: An MLLM with simultaneously improved understanding and generation capabilities and a reduced gap between them.

### Key Designs

1. **Understanding Preference Data Construction**:

    - **Function**: Generate $n$ distinct captions per image and select winning/losing samples via BERT similarity scoring.
    - **Mechanism**: Given image $x$, the MLLM generates $n$ captions; the BERT similarity $s(y_k, y)$ between each caption and the ground-truth prompt $y$ is computed. The highest-scoring caption serves as $y_w$ and the lowest as $y_l$.
    - **Design Motivation**: The captioning task comprehensively reflects the MLLM's ability to capture visual features (including object attributes, spatial relations, and fine-grained details), while BERT similarity provides an automated quality metric.

2. **Generation Preference Data Construction**:

    - **Function**: Randomly generate $n$ images per prompt and select the best/worst via self-VQA judgment.
    - **Mechanism**: TIFA is used to generate $q$ VQA pairs $\{(Q_i, A_i)\}$ for prompt $y$; for each generated image, the VQA accuracy $Acc(x_j) = \frac{1}{q}\sum_{i=1}^{q}\mathbb{I}(R_{j,i}=A_i)$ is computed. The image with the highest accuracy (above 0.6) becomes $x_w$, and the lowest-scoring image becomes $x_l$.
    - **Design Motivation**: This design cleverly exploits the "understanding-over-generation" asymmetry—the model's stronger understanding capability is used to critique its own generation, realizing a self-critic mechanism without external evaluators.

3. **Pair-DPO Joint Optimization**:

    - **Function**: Pair the understanding and generation preference data for joint optimization.
    - **Mechanism**: The loss function is $\mathcal{L}_{\text{Pair-DPO}}(\theta) = -\mathbb{E}[\log\sigma(\Delta_{Und}\cdot\Delta_{Gen})]$, where $\Delta_{Und}$ and $\Delta_{Gen}$ are the preference margins for understanding and generation, respectively. The key innovation lies in **multiplying** the two modality margins before applying the sigmoid, so that the gradient is maximized only when both objectives are simultaneously satisfied.
    - **Design Motivation**: Unlike separate DPO training, the multiplicative coupling ensures that optimization directions are coordinated within the same semantic space, preventing trade-offs between the two modalities.

4. **Iterative Self-Play Optimization**:

    - **Function**: Multi-round optimization in which each round uses the updated model to regenerate candidates, dynamically refreshing the preference pairs.
    - **Mechanism**: In round $i$, if the best new caption surpasses the previous winning sample (i.e., $s(y_{\max}^i, y) > s(y_w^{i-1}, y)$), it replaces the winning sample and the old winner becomes the losing sample; otherwise, the new best replaces the old losing sample, providing a smoother learning gradient.
    - **Design Motivation**: This adaptive "raise the bar" or "lower the difficulty" strategy prevents the model from stagnating during training.

### Loss & Training
- Optimizer: AdamW; learning rate: 2e-5; cosine schedule
- $\beta = 0.2$; batch size: 4; training steps: 3000
- Base model: Show-o (1.3B); homologous data: 5,000 pairs from JourneyDB
- Hardware: 8×A100 GPUs

## Key Experimental Results

### Main Results — Understanding

| Model | Params | POPE↑ | MME↑ | Flickr30k↑ | VQAv2↑ | GQA↑ | MMMU↑ |
|-------|--------|-------|------|------------|--------|------|-------|
| SEED-X | 17B | 84.2 | 1435.7 | 52.3 | - | 47.9 | 35.6 |
| Chameleon | 34B | - | - | 74.7 | 66.0 | - | - |
| Show-o | 1.3B | 80.0 | 1232.9 | 67.6 | 74.7 | 61.0 | 27.4 |
| **HermesFlow** | **1.3B** | **81.4** | **1249.7** | **69.2** | **75.3** | **61.7** | **28.3** |

### Main Results — Generation (GenEval)

| Method | Params | Single Obj. | Two Obj. | Counting | Colors | Position | Overall |
|--------|--------|-------------|----------|----------|--------|----------|---------|
| SD 2.1 | 865M | 0.97 | 0.50 | 0.46 | 0.80 | 0.07 | 0.49 |
| Show-o | 1.3B | 0.95 | 0.52 | 0.49 | 0.82 | 0.11 | 0.53 |
| Janus | 1.3B | 0.97 | 0.68 | 0.30 | 0.84 | 0.46 | 0.61 |
| **HermesFlow** | **1.3B** | **0.98** | **0.84** | **0.66** | **0.82** | **0.32** | **0.69** |

### Ablation Study — Pair-DPO vs. DPO & Number of Iterations

| Method | POPE↑ | MME↑ | MMMU↑ | GenEval Overall↑ | DPG-Bench↑ |
|--------|-------|------|-------|------------------|------------|
| Show-o (baseline) | 80.0 | 1232.9 | 27.4 | 0.53 | 67.48 |
| DPO (understanding only) | 80.8 | 1242.2 | 27.8 | 0.58 | 67.88 |
| DPO (generation only) | 80.5 | 1239.3 | 27.5 | 0.70 | 70.03 |
| Pair-DPO Iter.1 | 81.1 | 1246.7 | 28.0 | 0.68 | 70.19 |
| Pair-DPO Iter.3 | **81.4** | **1249.7** | **28.3** | **0.69** | **70.22** |

### Understanding-Generation Gap Quantification

| Method | Understanding Score↑ | Generation Score↑ | Gap↓ |
|--------|---------------------|-------------------|------|
| VILA-U (7B) | 0.646 | 0.477 | 0.169 |
| Janus (1.3B) | 0.599 | 0.417 | 0.182 |
| Show-o (1.3B) | 0.520 | 0.433 | 0.087 |
| **HermesFlow (1.3B)** | **0.533** | **0.497** | **0.036** |

### Key Findings
- A single round of Pair-DPO substantially improves both understanding and generation simultaneously, outperforming separate understanding DPO and generation DPO.
- The first iteration contributes the most gains; beyond two iterations, generation quality largely converges while understanding continues to improve marginally.
- The understanding-generation gap is reduced from 0.087 (Show-o) to 0.036 (a 59% reduction).
- Generation preference data construction is more sensitive to the sampling size $n$; small $n$ introduces significant noise and noticeably degrades generation quality.

## Highlights & Insights
- The **self-critic mechanism** is particularly elegant: it exploits the asymmetry between understanding and generation to let the model use its stronger understanding ability to evaluate its own generated outputs, requiring no external evaluator.
- The **multiplicative coupling in Pair-DPO**: by formulating the loss via the product $\Delta_{Und} \cdot \Delta_{Gen}$, understanding and generation are jointly updated within the same optimization step rather than independently.
- **Adaptive standard adjustment in self-play**: when the model improves, the bar is raised (new best replaces old winning sample); otherwise, the difficulty is reduced (new best replaces old losing sample), ensuring continuous and effective learning.
- **Transferability**: The Pair-DPO framework is in principle applicable to any unified understanding-generation MLLM (e.g., Janus, VILA-U).

## Limitations & Future Work
- Validation is limited to Show-o (1.3B); the framework has not been extended to larger models or additional backbones.
- The homologous dataset consists of only 5,000 pairs; the effect of scaling up remains unclear.
- Generation evaluation relies on the quality of TIFA-generated VQA pairs; inaccurate VQA annotations introduce noise into the preference data.
- Beyond two self-play iterations, marginal returns diminish and generation quality plateaus, potentially requiring more sophisticated curriculum learning strategies.
- BERT similarity as the understanding preference signal may lack precision; replacing it with a stronger semantic evaluation model warrants exploration.

## Related Work & Insights
- **vs. DPO (understanding/generation separately)**: Separate DPO for understanding and generation is individually effective but lacks coordination; Pair-DPO unifies both through multiplicative coupling and achieves superior results.
- **vs. MetaMorph/Liquid**: These works study understanding-generation synergy at the data level but lack an explicit optimization framework for closing the gap.
- **vs. Emu3 (generation DPO)**: Emu3 requires human ranking to construct preference data; HermesFlow is fully automated and jointly optimizes both understanding and generation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to quantify the understanding-generation gap and propose a paired DPO optimization framework; the concept is clear, though the core contribution lies in a principled combination of existing ideas.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive validation across multiple understanding and generation benchmarks with clear ablations; limited to a single backbone.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is well articulated, the pipeline diagram is intuitive, and mathematical formulations are complete.
- **Value**: ⭐⭐⭐⭐ As a general post-training alignment framework for unified MLLMs, it has strong transferability and potential applicability to next-generation models.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Closing the Modality Gap Aligns Group-Wise Semantics](../../ICLR2026/multimodal_vlm/closing_the_modality_gap_aligns_group-wise_semantics.md)
- [\[NeurIPS 2025\] UniTok: A Unified Tokenizer for Visual Generation and Understanding](unitok_a_unified_tokenizer_for_visual_generation_and_understanding.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](../../ICCV2025/multimodal_vlm/metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[ICCV 2025\] Harmonizing Visual Representations for Unified Multimodal Understanding and Generation](../../ICCV2025/multimodal_vlm/harmonizing_visual_representations_for_unified_multimodal_un.md)
- [\[NeurIPS 2025\] Systematic Reward Gap Optimization for Mitigating VLM Hallucinations](systematic_reward_gap_optimization_for_mitigating_vlm_hallucinations.md)

<!-- RELATED:END -->
