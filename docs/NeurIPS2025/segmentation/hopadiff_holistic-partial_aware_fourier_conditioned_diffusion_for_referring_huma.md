---
title: >-
  [Paper Note] HopaDIFF: Holistic-Partial Aware Fourier Conditioned Diffusion for Referring Human Action Segmentation in Multi-Person Scenarios
description: >-
  [NeurIPS 2025][Segmentation][Referring Human Action Segmentation] This paper introduces the Referring Human Action Segmentation (RHAS) task—localizing a specific individual in multi-person videos via textual descriptions…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Referring Human Action Segmentation"
  - "Multi-Person Scenarios"
  - "Diffusion Models"
  - "xLSTM"
  - "Fourier Conditioning"
  - "RHAS"
date: 2026-05-08
content_hash: 0531897bc79eb466
---

# HopaDIFF: Holistic-Partial Aware Fourier Conditioned Diffusion for Referring Human Action Segmentation in Multi-Person Scenarios

**Conference**: NeurIPS 2025
**arXiv**: [2506.09650](https://arxiv.org/abs/2506.09650)
**Code**: [https://github.com/KPeng9510/HopaDIFF](https://github.com/KPeng9510/HopaDIFF)
**Area**: Temporal Action Segmentation / Multi-Person Video Understanding
**Keywords**: Referring Human Action Segmentation, Multi-Person Scenarios, Diffusion Models, xLSTM, Fourier Conditioning, RHAS

## TL;DR
This paper introduces the Referring Human Action Segmentation (RHAS) task—localizing a specific individual in multi-person videos via textual descriptions and performing frame-level action segmentation. The authors construct the RHAS133 dataset comprising 133 movies, 137 action categories, and 33 hours of video, and propose HopaDIFF, a holistic-partial aware Fourier-conditioned diffusion framework that substantially outperforms existing baselines across multiple evaluation settings.

## Background & Motivation

**Background**: Action segmentation aims to temporally partition untrimmed videos into action segments and classify them. Existing methods (FACT, ActDiff, ASQuery) primarily target **single-person scenarios**, where action sequences follow predefined protocols (e.g., preparing a salad according to a fixed recipe or assembling objects from instructions), and thus fail to capture the randomness and complexity of real-world multi-person scenarios.

**Limitations of Prior Work**:
- No multi-person action segmentation dataset exists—existing datasets (50Salads, Assembly101, Breakfast) are all single-person, procedure-following, and first-person.
- No text-guided mechanism is available to specify which individual to segment.
- Existing action segmentation methods lack target-aware partial feature reasoning and fine-grained generative control.

**Core Idea**: Define the new RHAS task + construct the RHAS133 dataset + propose the dual-branch diffusion model HopaDIFF, which enhances controllability via holistic-partial feature interaction and frequency-domain conditioning.

## RHAS133 Dataset

### Dataset Statistics

| Property | RHAS133 | 50Salads | Assembly101 | Breakfast |
|----------|---------|----------|-------------|-----------|
| Multi-Person | ✓ | ✗ | ✗ | ✗ |
| Text-Guided | ✓ | ✗ | ✗ | ✗ |
| Multi-Label | ✓ | ✗ | ✗ | ✗ |
| Free-Form Procedure | ✓ | ✗ | ✗ | ✗ |
| Viewpoint | Third-Person | First-Person | First-Person | Third-Person |
| Duration | 33h | 4h | 513h | 77h |
| Action Categories | 137 | 50 | 101 | 10 |
| Individuals | 542 | 25 | 53 | 52 |

- 133 movies, 542 annotated individuals, cross-validated by 6 domain experts.
- Action labels follow the AVA protocol, extended from 80 to 137 fine-grained categories.
- Textual descriptions refer to the target individual's appearance without revealing action content.

## Method

### Overall Architecture
HopaDIFF employs a dual-branch diffusion architecture: the **holistic branch** captures full-video context, while the **partial branch** extracts fine-grained features from cropped video clips of the target person detected by GroundingDINO. Both branches share a VLM feature extractor (BLIP-2/CLIP) but use independent encoders.

### 1. Holistic-Partial Conditioned Diffusion
Dual-branch feature extraction:

$$\mathbf{z}_h, \mathbf{z}_p = \mathbf{E}_h(\mathbf{F}_\phi(\mathbf{v}_h, \mathbf{r})), \quad \mathbf{E}_p(\mathbf{F}_\phi(\text{G-Dino}(\mathbf{v}_h, \mathbf{r}), \mathbf{r}))$$

where $\mathbf{v}_h$ denotes the original video, $\mathbf{r}$ the textual reference, $\mathbf{F}_\phi$ the VLM feature extractor, and G-Dino performs target person detection and cropping.

### 2. HP-xLSTM: Cross-Input Gated Attention xLSTM
Building on the mLSTM variant, bidirectional cross-attention (BCA) is introduced at the input gate to enable holistic-partial feature interaction:

$$\hat{\mathbf{z}}^h, \hat{\mathbf{z}}^p = \text{HP-xLSTM}(\mathbf{z}^h, \mathbf{z}^p)$$

The mLSTM matrix memory update: $\boldsymbol{C}_t^m = f_t^m \boldsymbol{C}_{t-1}^m + i_t^m \boldsymbol{v}_t^m \boldsymbol{k}_t^{m\top}$

The key innovation lies in the input gate: $\tilde{i}_t^h, \tilde{i}_t^p = \text{BCA}(\boldsymbol{w}_i^{h\top} \boldsymbol{z}_t^h + b_i^h, \; \boldsymbol{w}_i^{p\top} \boldsymbol{z}_t^h + b_i^h)$

Through bidirectional cross-attention, the input gate of the holistic branch is modulated by partial features and vice versa, enabling bidirectional information exchange between the two branches.

### 3. Fourier Frequency-Domain Conditioning
Discrete Fourier Transform (DFT) is applied to the HP-xLSTM-enhanced features to extract frequency-domain information as additional conditioning:

$$\hat{\mathbf{z}}_f^h, \hat{\mathbf{z}}_f^p = \text{DFT}(\hat{\mathbf{z}}^h), \text{DFT}(\hat{\mathbf{z}}^p)$$

The decoder receives both spatiotemporal and frequency-domain features:

$$\mathbf{s}^h, \mathbf{s}^p = \mathbf{D}_h(\mathbf{y}_t, \hat{\mathbf{z}}_f^h, \mathbf{z}_f^h), \quad \mathbf{D}_p(\mathbf{y}_t, \hat{\mathbf{z}}_f^p, \mathbf{z}_f^p)$$

Frequency-domain conditioning enhances fine-grained controllability—low frequencies encode overall action rhythm while high frequencies capture action boundary transitions.

### 4. Loss & Training
- Training: Binary cross-entropy loss + temporal boundary loss (aligning the denoised sequence with GT action boundaries).
- Inference: Iterative denoising from Gaussian noise; final predictions are averaged across the two branches.

## Key Experimental Results

### BLIP-2 Features + Random Split

| Method | ACC↑ | EDIT↑ | F1@10↑ | F1@25↑ | F1@50↑ |
|--------|------|-------|--------|--------|--------|
| FACT | 26.08 | 0.27 | 52.91 | 50.77 | 47.06 |
| ActDiff | 41.85 | 7.20 | 70.56 | 68.34 | 63.29 |
| LTContent | 34.23 | 0.31 | 64.70 | 63.09 | 58.50 |
| RefAtomNet | 38.01 | 0.13 | 34.01 | 31.93 | 27.62 |
| **HopaDIFF** | **62.58** | **7.75** | **87.96** | **85.50** | **79.39** |

HopaDIFF achieves a 50% relative improvement in ACC (62.58 vs. 41.85) and a 25% improvement in F1@50 (79.39 vs. 63.29), substantially outperforming all baselines.

### BLIP-2 Features + Cross-Movie Evaluation (Stricter Generalization Test)

| Method | ACC↑ | EDIT↑ | F1@10↑ | F1@25↑ | F1@50↑ |
|--------|------|-------|--------|--------|--------|
| ActDiff | 2.36 | 15.09 | 22.44 | 22.28 | 21.80 |
| LTContent | 52.52 | 0.37 | 49.35 | 47.24 | 42.55 |
| **HopaDIFF** | **59.63** | **19.37** | **90.91** | **90.33** | **89.26** |

Under the cross-movie setting, the advantage is even more pronounced—F1@50 improves from 42.55 to 89.26, demonstrating the strong generalization of the dual-branch architecture. ActDiff degrades severely (ACC drops to 2.36%), while HopaDIFF remains robust.

## Highlights & Insights
1. **New Task + New Dataset**: RHAS is defined for the first time, and RHAS133 is the first multi-person text-guided action segmentation dataset.
2. **HP-xLSTM**: Bidirectional cross-attention introduced at the xLSTM input gate elegantly enables holistic-partial information exchange.
3. **Fourier Conditioning**: Frequency-domain conditioning supplements spatiotemporal features, enhancing the diffusion model's fine-grained control over action boundaries.
4. **Cross-Movie Generalization**: F1@50 reaches 89.26%, far exceeding competing methods, validating the robustness of the dual-branch architecture.

## Limitations & Future Work
1. The framework relies on GroundingDINO for target person detection, which may fail under severe occlusion or when faces are not visible.
2. RHAS133 covers only 133 movies; while broad in scope, the dataset size remains limited.
3. The two-stage architecture (VLM feature pre-extraction + diffusion segmentation) precludes end-to-end training.
4. Inference requires simultaneous execution of GroundingDINO, the VLM, and the dual-branch diffusion model, resulting in considerable computational overhead.

## Related Work & Insights
- The RHAS task can be extended to sports analytics (tracking the action sequences of specific athletes) and movie content understanding.
- The cross-input gating design of HP-xLSTM is applicable to other long-sequence modeling tasks requiring multi-source information fusion.
- The frequency-domain conditioning strategy can be adopted in video generation to maintain temporal consistency.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ New task + new dataset + innovative technical approach; comprehensive contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple evaluation settings and two feature extractors; ablation study could be strengthened.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework presentation, though the dense notation warrants careful cross-referencing.
- **Value**: ⭐⭐⭐⭐⭐ Pioneers the RHAS direction; the dataset holds long-term research value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance](haodiff_human-aware_one-step_diffusion_via_dual-prompt_guidance.md)
- [\[ICCV 2025\] UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](../../ICCV2025/segmentation/uniglyph_unified_segmentation-conditioned_diffusion_for_precise_visual_text_synt.md)
- [\[NeurIPS 2025\] FAST: Foreground-aware Diffusion with Accelerated Sampling Trajectory for Segmentation-oriented Anomaly Synthesis](fast_foreground-aware_diffusion_with_accelerated_sampling_trajectory_for_segment.md)
- [\[NeurIPS 2025\] Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)
- [\[NeurIPS 2025\] Diffusion-Driven Two-Stage Active Learning for Low-Budget Semantic Segmentation](diffusion-driven_two-stage_active_learning_for_low-budget_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
