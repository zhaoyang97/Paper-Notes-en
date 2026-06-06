---
title: >-
  [Paper Note] RESTORE: Improving Visual Token Reduction via Distortion Correction for Enhanced MLLM Inference Efficiency
description: >-
  [ICML 2026][Multimodal VLM][Visual Token Reduction] RESTORE highlights two overlooked issues in existing Visual Token Reduction (VTR): "positional distortion" and "attention attenuation." By adding a distance-aware inver…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Visual Token Reduction"
  - "MLLM Acceleration"
  - "RoPE Attention Calibration"
  - "Anchor Token Selection"
  - "LLaVA"
date: 2026-05-08
content_hash: e49bbf19f434d08b
---

# RESTORE: Improving Visual Token Reduction via Distortion Correction for Enhanced MLLM Inference Efficiency

**Conference**: ICML 2026  
**arXiv**: [2606.01711](https://arxiv.org/abs/2606.01711)  
**Code**: https://cvlab.yonsei.ac.kr/projects/RESTORE (Project Homepage)  
**Area**: Multimodal VLM / LLM Efficiency  
**Keywords**: Visual Token Reduction, MLLM Acceleration, RoPE Attention Calibration, Anchor Token Selection, LLaVA  

## TL;DR
RESTORE highlights two overlooked issues in existing Visual Token Reduction (VTR): "positional distortion" and "attention attenuation." By adding a distance-aware inverse compensation term to RoPE attenuation and improving token merging with an anchor selection strategy that balances representativeness and distinctiveness, LLaVA-1.5-7B achieves performance close to the full-token baseline even with only 64 tokens (approx. 11% retention).

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs, e.g., LLaVA, Qwen2.5-VL) encode visual patches into hundreds or thousands of visual tokens, which are concatenated with text tokens for LLM processing. Given the $O(N^2)$ complexity of self-attention, these visual tokens represent a primary computational and memory bottleneck. Visual Token Reduction (VTR) has emerged as a solution, primarily following two paths: pruning (FastV, SparseVLM, HoloV), which retains high-attention tokens and discards the rest; and merging (ToMe, PruMerge, VisionZip), which aggregates similar tokens into representative anchors.

**Limitations of Prior Work**: The authors identify two types of distortion that have long been ignored. First is **positional distortion**—after sequence reduction, existing methods either re-index retained tokens into continuous positions ("reindex" approach) or maintain original indices ("retain" approach). The former disrupts the true spatial distance between visual and text tokens, while the latter suffers from the long-range decay of RoPE, which severely suppresses distant tokens. Second is **attention attenuation**—softmax normalization redistributes the probability mass of pruned tokens. Since text tokens are closer to each other and naturally have larger logits, the overall attention proportion of visual tokens after merging/pruning is significantly lower than the full-sequence baseline. This forces the model to "rely less on images and guess more from text," leading to hallucinations and weakened visual grounding.

**Key Challenge**: Reindexing sacrifices spatial authenticity for attention volume, while retaining indices preserves spatial authenticity but loses attention volume. Neither path simultaneously achieves "positional semantic alignment" and "attention distribution alignment."

**Goal**: Without modifying LLM weights or adding significant inference overhead: (1) allow reduced visual sequences to retain original positional indices to maintain spatial relationships; (2) pull the total attention proportion of visual tokens back to full-sequence levels; and (3) select more representative anchors during token merging to reduce detail loss from feature averaging.

**Key Insight**: Since the long-range decay function of RoPE $\mathcal{D}(|m-n|)=\frac{2}{d_h}\sum_{j=1}^{d_h/2}\cos(|m-n|\theta_j)$ can be analytically derived, one can construct an **inverse distance-increasing** compensation term $c-\mathcal{D}(|m-n|)$ to analytically restore the attention suppressed by RoPE. For anchor selection, inspired by density peak clustering, anchors should be both "centers of their neighborhoods" and "sufficiently distant from each other."

**Core Idea**: Use "distance-aware softmax calibration" to correct positional/attention distortion and a "representative × distinctive" dual-metric for anchor selection, forming a plug-and-play universal enhancement module for VTR.

## Method

### Overall Architecture
RESTORE is a **universal VTR enhancer** for standard MLLMs (represented by LLaVA-1.5). It does not modify the visual encoder or the LLM; instead, it only alters two components: the softmax formula for attention calculation within the LLM and the anchor selection logic during the token merging phase.

Mechanism: Input image → visual encoder output $\mathbf{X}_{\text{vis}}\in\mathbb{R}^{N_{\text{vis}}\times d}$ ($N_{\text{vis}}{=}576$) → VTR phase (compatible with any pruning/merging/hybrid method) output $\hat{\mathbf{X}}_{\text{vis}}\in\mathbb{R}^{n_{\text{vis}}\times d}$ ($n_{\text{vis}}\in\{64, 128, 192\}$) → retained tokens use **original positional indices** from the full sequence → LLM uses **calibrated** softmax in each attention layer → output text response. If the underlying VTR involves merging (e.g., VisionZip), RESTORE's anchor selection strategy replaces the original sampling method.

### Key Designs

1. **Distance-aware Attention Calibration**:
    - **Function**: Restores the total visual token attention proportion to full-sequence levels while retaining original positional indices.
    - **Mechanism**: Adds an analytical calibration term that increases monotonically with relative distance to the softmax logits, directly counteracting RoPE's long-range decay $\mathcal{D}(|m-n|)$. The calibrated attention is expressed as:
      $$\hat{A}_{m,n}=\frac{\exp(z_{m,n}+\log s_n(c-\mathcal{D}(|m-n|)))}{\sum_{i}\exp(z_{m,i}+\log s_i(c-\mathcal{D}(|m-i|)))}$$
      where $z_{m,n}$ is the original logit, $s_n$ is the number of original tokens merged into the $n$-th token, and $c$ is a constant ensuring non-negative compensation. $\log s_n$ follows the ToMe approach of scaling merged tokens, while $(c-\mathcal{D})$ is RESTORE's addition, responsible for boosting attention for distant visual tokens suppressed by distance decay.
    - **Design Motivation**: ToMe's $\log s_n$ was designed for vision-only tasks. In MLLMs, text tokens are closer and have larger logits; $\log s_n$ alone cannot recover visual attention redistributed by softmax. RESTORE isolates the "positional distance" factor from $z_{m,n}$ and compensates for it, achieving both the attention volume of reindexing and the authentic spatial relationships of retaining original indices. The paper also extends this to M-RoPE (multimodal RoPE used in Qwen2.5-VL), making it applicable to 1D and multi-dimensional spatial encodings.

2. **Distinctive Anchor Token Selection**:
    - **Function**: Selects anchors during token merging that are both representative of their neighborhood and non-redundant, reducing detail loss from averaging.
    - **Mechanism**: Inspired by density peak clustering. Pairwise correlation matrix $\mathbf{C}=\mathbf{X}_{\text{vis}}\mathbf{X}_{\text{vis}}^T/\|\mathbf{X}_{\text{vis}}\|^2$ is calculated on normalized visual features. **Representativeness** is defined as the sum of a token's correlation with all other visual tokens $\mathcal{R}_i=\sum_j \mathbf{C}_{ij}$. **Distinctiveness** is defined as $1-\max_j \hat{\mathbf{C}}_{ij}$, where $\hat{\mathbf{C}}$ is the correlation matrix masked by $\mathbf{M}_{ij}=\mathbb{I}(\mathcal{R}_j>\mathcal{R}_i)$ (masking non-"stronger" rivals). This highlights the maximum similarity to more "central" tokens—a smaller value indicates the token is not covered by any stronger rival and is thus more unique. The final anchor set $\mathcal{A}$ is selected as the Top-K tokens with the highest product of representativeness and distinctiveness. Remaining tokens are merged with their most similar anchor.
    - **Design Motivation**: PruMerge selects high-attention anchors and VisionZip uses uniform sampling; neither guarantees similarity between anchors and merged tokens, often resulting in blurred features. Selecting multiple highly correlated anchors also wastes budget. The "representative × distinctive" approach solves both issues using a pre-calculated correlation matrix without significant extra overhead.

3. **Unified Plug-and-Play Interface for Index Retention**:
    - **Function**: Enforces "original positional index" as a constraint and provides a unified interface for any VTR backbone (FastV, SparseVLM, ToMe, etc.).
    - **Mechanism**: Without modifying the core "which tokens to pick" logic of various VTR methods, it only replaces the softmax and anchor selection logic. VTR methods output $\hat{\mathbf{X}}_{\text{vis}}$ and size vectors $\{s_n\}$ as usual, and RESTORE consumes these outputs.
    - **Design Motivation**: Since distortion is a paradigm-level issue for VTR, the fix should also be at the paradigm level. Results show that RESTORE improves average scores by 1–4 points across five different VTR backbones.

### Loss & Training
RESTORE is a **purely inference-time module**. it introduces no trainable parameters and requires no retraining of the LLM or visual encoder. The calibration constant $c$ is fixed, and long-range decay $\mathcal{D}$ is analytically derived from RoPE frequency parameters $\theta_j$. Anchor selection only depends on the feature correlation matrix calculated once during the visual encoder forward pass.

## Key Experimental Results

### Main Results
Evaluated on LLaVA-1.5-7B across 8 benchmarks (GQA, MMB, MME, POPE, SQA$^{\text{IMG}}$, VQA$^{\text{V2}}$, VQA$^{\text{Text}}$, SEED). The "Relative average percentage of full-token score" is used as the metric. Representative results for 192 tokens (33.3% retention):

| Method | Type | Avg Score | GQA | MME | POPE | VQA$^{\text{V2}}$ |
|------|------|---------|-----|-----|------|-------------------|
| LLaVA-1.5-7B (576 tokens) | Baseline | 100.0% | 61.9 | 1862 | 85.9 | 78.5 |
| FastV | Text-aware | 96.0% | 57.1 | 1821 | 75.8 | 74.7 |
| SparseVLM | Text-aware | 98.1% | 59.5 | 1782 | 85.4 | 77.0 |
| VisionZip | Hybrid | 96.8% | 59.2 | 1749 | 85.2 | 77.2 |
| **VisionZip + RESTORE** | Hybrid | **98.0%** | **60.6** | **1782** | **86.6** | 77.0 |
| DivPrune | Text-agnostic | 96.9% | 58.9 | 1723 | 86.5 | 76.1 |
| **DivPrune + RESTORE** | Text-agnostic | **98.7%** | **60.9** | **1813** | **86.6** | **77.4** |
| HoloV | Text-agnostic | 96.5% | 58.6 | 1779 | 85.0 | 76.0 |
| **HoloV + RESTORE** | Text-agnostic | **98.8%** | **61.0** | **1793** | **86.6** | **77.6** |

At 128 tokens (22.2%), FastV drops to 91.8% and PDrop collapses to 84.5%, while text-agnostic backbones with RESTORE remain stable at 95%+. At 64 tokens (11.1%), RESTORE pushes multiple VTR backbones toward near-full-token performance.

### Ablation Study
| Configuration | Avg Score (192 tokens) | Description |
|------|---------------------|------|
| HoloV (Baseline) | 96.5% | Pruning only, reindexed positions |
| + Retain original indices | Slight decrease | Loss of attention volume (confirms retain pain point) |
| + Distance calibration ($c-\mathcal{D}$) | 98.4% | Restores attention volume; main gain source |
| + Distinctive anchors | +0.3~0.5% | Additional gain for merging methods like VisionZip |
| **Full RESTORE** | **98.8%** | Calibration + anchor selection combined |

### Key Findings
- **Calibration is the primary contributor**: Compounding the restoration of attention volume ($c-\mathcal{D}$) with the retention of original indices closes the performance gap across all VTR backbones. POPE (hallucination evaluation) shows particularly significant gains, consistent with the recovery of visual grounding.
- **Agnostic to VTR backbones**: RESTORE consistently provides a 1.5–2.3 point average gain across text-aware (SparseVLM), text-agnostic (HoloV, DivPrune), and hybrid (VisionZip) methods.
- **Criticality scales with reduction**: The benefit of RESTORE increases as more tokens are reduced, as visual attention attenuation becomes more severe at lower token counts.
- **Effective on M-RoPE**: The calibration term extends analytically to multimodal RoPE used in Qwen2.5-VL, with consistent empirical gains.

## Highlights & Insights
- **Inverting the analytical property of RoPE**: instead of treating RoPE as a black box, its decay function $\mathcal{D}$ is used as an inverse compensation signal. The cost is negligible (a few lines of cosine sums), providing a model-prior-correction strategy applicable to any relative position encoding scenario.
- **Decomposition of VTR distortion**: By splitting the problem into "positional distortion" and "attention attenuation," the paper proves that neither reindexing nor simple retaining is correct, shifting the field's focus beyond just "token selection strategies."
- **Efficient Anchor Selection**: "Representative × Distinctive" metric reuses the correlation matrix, bringing density peak clustering to token merging naturally without extra overhead, solving the "anchor non-centrality" problem.
- **Zero-training deployment**: Plug-and-play for LLaVA/Qwen2.5-VL; highly friendly for industrial deployment.

## Limitations & Future Work
- **Dependency on base VTR**: RESTORE enhances but does not replace VTR. If the base method prunes critical tokens (e.g., FastV at 128 tokens), calibration cannot recover the lost information.
- **Fixed Calibration Constant $c$**: While selection principles are provided, the optimal $c$ may vary by resolution or token count, lacking an adaptive mechanism.
- **Purely Visual Anchors**: Anchor selection does not currently incorporate text or task awareness, potentially missing patches that are text-relevant but visually non-salient.
- **Limited Generative Evaluation**: Benchmarks are mostly discriminative/QA-based; visual grounding during long-text generation requires further study.

## Related Work & Insights
- **vs FastV / SparseVLM**: They rely on cross-modal attention for selection but require full attention in early layers and ignore distortion post-reduction. RESTORE improves their performance by fixing the attention distribution.
- **vs ToMe / VisionZip**: ToMe's $\log s_n$ fails in MLLMs due to text interference; RESTORE fills this gap with $(c-\mathcal{D})$. VisionZip's uniform sampling is replaced by RESTORE's more effective anchor strategy.
- **vs DivPrune**: DivPrune emphasizes diversity during pruning (keeping tokens far from each other). RESTORE's distinctiveness metric is similar in motivation but applied to merging (selecting anchors that are neighborhood centers but distant from each other).
- **vs HoloV**: HoloV optimizes "where to keep tokens," while RESTORE optimizes "the attention quality of kept tokens." They are orthogonal and achieve state-of-the-art results when combined.

## Rating
- Novelty: ⭐⭐⭐⭐ First to formally address VTR attention/positional distortion with an analytical solution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive testing across 8 benchmarks, 3 ratios, and 5+ backbones.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and compact derivations.
- Value: ⭐⭐⭐⭐⭐ Zero-training plugin with universal gains; a powerful tool for MLLM acceleration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration](../../AAAI2026/multimodal_vlm/filter_correlate_compress_training-free_token_reduction_for_.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)
- [\[CVPR 2026\] DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference](../../CVPR2026/multimodal_vlm/duet-vlm_dual_stage_unified_efficient_token_reduction_for_vlm_training_and_infer.md)
- [\[ICLR 2026\] Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping](../../ICLR2026/multimodal_vlm/constructive_distortion_improving_mllms_with_attention-guided_image_warping.md)
- [\[ACL 2026\] From Inheritance to Saturation: Disentangling the Evolution of Visual Redundancy for Architecture-Aware MLLM Inference Acceleration](../../ACL2026/multimodal_vlm/from_inheritance_to_saturation_disentangling_the_evolution_of_visual_redundancy_.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2026\] Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token](../../CVPR2026/multimodal_vlm/rethinking_mllm_itself_as_a_segmenter_with_a_single_segmentation_token.md)
- [\[AAAI 2026\] Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration](../../AAAI2026/multimodal_vlm/filter_correlate_compress_training-free_token_reduction_for_.md)
- [\[ICML 2026\] WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation](weathersyn_an_instruction_tuning_mllm_for_weather_forecasting_report_generation.md)
- [\[ICML 2026\] On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression](on_the_adversarial_robustness_of_large_vision-language_models_under_visual_token.md)
- [\[ICML 2026\] ECG-R1: Protocol-Guided and Modality-Agnostic MLLM for Reliable ECG Interpretation](ecg-r1_protocol-guided_and_modality-agnostic_mllm_for_reliable_ecg_interpretatio.md)

</div>

<!-- RELATED:END -->
