---
title: >-
  [Paper Note] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model
description: >-
  [Image Generation] EmotiCrafter is proposed as the first emotional image generation method based on a continuous Valence-Arousal (V-A) model. It integrates V-A values into text features via an emotional embedding mapping…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 1f7b371baf5256f6
---

# EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model

## Paper Information

- **Conference**: ICCV 2025
- **arXiv**: 2501.05710
- **Code**: [https://github.com/idvxlab/EmotiCrafter](https://github.com/idvxlab/EmotiCrafter)
- **Area**: Image Generation / Affective Computing
- **Keywords**: emotional image generation, valence-arousal, SDXL, continuous emotion, cross-attention injection

## TL;DR

EmotiCrafter is proposed as the first emotional image generation method based on a continuous Valence-Arousal (V-A) model. It integrates V-A values into text features via an emotional embedding mapping network, which is then injected into Stable Diffusion XL to achieve precise dual control over content and emotion. The generated images significantly outperform existing methods in terms of emotional continuity and controllability.

## Background & Motivation

Emotion plays a critical role in information conveyance; however, generating images with precise emotional content remains an open problem. Limitations of prior work:

**EmoGen** (pioneering work in emotional image generation): supports only discrete emotion labels (happy/sad), cannot control specific image content, and fails to capture subtle emotional nuances.

**Discrete vs. Continuous Emotion**: Psychologists have not reached consensus on emotion category boundaries, and discrete labels offer limited expressive range.

**Emotion editing methods** (IET): rely on specific visual elements (color, texture), resulting in insufficient depth of emotional expression.

EmotiCrafter introduces the **Continuous Emotional Image Content Generation (C-EICG)** task: given a free-form text prompt and continuous V-A values, generate an image that simultaneously satisfies both the content description and the emotional expression. The V-A model represents emotions in a two-dimensional Cartesian space: Valence (pleasantness, −3 to 3) and Arousal (activation, −3 to 3), naturally capturing subtle emotional transitions such as from "boredom" to "fatigue."

## Method

### Overall Architecture

EmotiCrafter = V-A Encoder + Emotion Injection Transformer (EIT) + SDXL Generator

Input: text prompt → encoder $\mathcal{E}$ → neutral feature $f_n$ → emotional embedding network $\mathcal{M}$ → emotional feature $\hat{f}_e$ → injected into SDXL → emotional image

$$\hat{f}_e = \mathcal{M}(f_n | (v, a))$$

### Key Designs 1: V-A Encoder

Two independent MLPs process the Valence and Arousal values separately:
- $e_v = \text{MLP}_V(v)$: V-feature
- $e_a = \text{MLP}_A(a)$: A-feature

### Key Designs 2: Emotion Injection Transformer (EIT)

Based on a modified GPT-2 architecture, comprising 12 Emotion Injection Blocks (EIBs). The processing pipeline within each EIB:

1. **Input Projection**: Projects neutral features into the Transformer space.
$$h_0 = P_{\text{in}}(f_n) + \text{PE}$$

2. **Emotion Injection** (per EIB):
$$h'_i = \text{self-attn}(\text{LN}(h_{i-1})) + h_{i-1}$$
$$h^{(v)}_i = \text{cross-attn}(\text{LN}(h'_i), e_v) + h'_i$$
$$h^{(v,a)}_i = \text{cross-attn}(\text{LN}(h^{(v)}_i), e_a) + h^{(v,a)}_i$$
$$h_i = \text{FFN}(\text{LN}(h^{(v,a)}_i)) + h^{(v,a)}_i$$

3. **Residual Prediction**: The network predicts the **residual** between the neutral and emotional features rather than directly predicting the emotional feature:
$$\hat{f}_r = P_{\text{out}}(\text{LN}(h_{12}))$$
$$\hat{f}_e = \hat{f}_r + f_n$$

Notably, the causal mask from the original GPT-2 is removed, as autoregressive generation is not required.

### Key Designs 3: Loss Function for Enhanced Emotional Expression

$$\mathcal{L} = \frac{1}{n} \mathbb{E}\left(\frac{1}{d(v,a)} \|\hat{f}_e - f^t_e\|^2\right)$$

**Scaled Residual Learning**: The target residual is amplified to enhance emotional variation:
$$f^t_e = f_n + \alpha(f_e - f_n)$$

where $\alpha = 1.5$ (determined via ablation), making emotional changes in generated images more pronounced.

**V-A Density Weighting**: Kernel density estimation (KDE) is used to compute the distribution density $d(v,a)$ of training samples in the V-A space, assigning higher weights to sparse regions to mitigate data imbalance.

### Training Data Construction

- Based on 39,843 images with human-annotated V-A values (from the OASIS, EMOTIC, and FindingEmo datasets).
- GPT-4 is used to generate paired neutral and emotional prompts for each image (sharing core semantics but differing in emotional expression).
- All LLM-generated prompts are validated via crowdsourcing, with disagreements resolved by majority voting.
- Training: AdamW optimizer, 2× A800 GPUs, 200 epochs, batch size 768, approximately 7–8 hours.

## Key Experimental Results

### Main Results: Quantitative Comparison

| Method | A-Error ↓ | V-Error ↓ | CLIPScore ↑ | CLIP-IQA ↑ |
|------|-----------|-----------|-------------|------------|
| Cross Attention | 1.923 | 2.080 | 26.266 | 0.949 |
| Time Embedding | 1.941 | 2.031 | **26.566** | 0.786 |
| Textual Inversion | 1.958 | 1.923 | 22.346 | 0.370 |
| GPT-4+SDXL | 1.860 | 1.517 | 25.907 | 0.906 |
| **Ours** | **1.828** | **1.510** | 23.067 | 0.881 |

- Achieves the best emotional accuracy (V/A-Error).
- Continuity comparison (LPIPS-Continuous): Ours 0.220 vs. GPT-4+SDXL 0.361, indicating smoother emotional transitions.

### Ablation Study

| Configuration | A-Error ↓ | V-Error ↓ | CLIPScore ↑ |
|------|-----------|-----------|-------------|
| α=1.0 | ~1.95 | ~1.65 | ~24.5 |
| **α=1.5 (Ours)** | **1.828** | **1.510** | **23.067** |
| α=2.0 | ~1.75 | ~1.40 | ~21.5 |
| w/o d(v,a) | 1.829 | 1.546 | 21.977 |

**Key Findings**:
- $\alpha$ controls the content–emotion trade-off: larger $\alpha$ improves emotional accuracy but increases semantic drift.
- $\alpha=1.5$ achieves the optimal balance and outperforms linear regression prediction in practice.
- Density weighting $d(v,a)$ improves CLIPScore by approximately 1 point and reduces V-Error by 0.04.

### User Study

| Metric | Ours | GPT-4+SDXL |
|------|------|-----------|
| A-Ranking Consistency ↑ | **0.759** | 0.165 |
| V-Ranking Consistency ↑ | **0.887** | 0.584 |
| A-Error ↓ | **1.327** | 2.029 |
| V-Error ↓ | **0.692** | 1.229 |
| Emotion Consistency ↑ | **4.215** | 3.525 |
| Emotion Smoothness ↑ | **4.240** | 3.195 |

All metrics are significantly superior to the baseline as confirmed by Wilcoxon signed-rank tests (p<0.05).

## Highlights & Insights

1. **New Task Definition (C-EICG)**: The first work to introduce continuous emotion control into image generation, which is more consistent with psychological modeling than discrete labels.
2. **Residual Learning + Scaling Factor**: Rather than directly predicting emotional features, the network predicts the neutral-to-emotional residual and amplifies it, elegantly enhancing emotional expressiveness.
3. **Emotion–Content Decoupling**: V-A values can override the semantic emotion implied by the prompt (e.g., "children's playground" paired with sad V-A values).
4. **Empty-Prompt Generation**: Providing only V-A values without any text prompt still yields emotionally consistent images, validating the effectiveness of the emotional embeddings.
5. **Fine-Grained Control**: A V-A increment of 0.2 produces perceptible changes in the generated images.

## Limitations & Future Work

- Arousal control is more challenging than Valence control, consistent with findings in the affective computing literature where annotator agreement on Arousal is lower.
- Even when prompts do not mention humans, the model frequently generates scenes involving human activities, due to insufficient non-human scenes in the training data.
- Emotional adjustment slightly shifts semantic content (reflected in CLIPScore degradation), suggesting the need for a semantic preservation loss term.
- The method is only validated on SDXL; its effectiveness on newer architectures such as DiT/VAR has not been investigated.

## Related Work & Insights

- Compared to EmoGen: C-EICG represents a significant advancement over EICG, transitioning from discrete to continuous emotion and from no content control to free-form text conditioning.
- The emotional embedding network design is generalizable to other continuous conditional control tasks (e.g., image style intensity, lighting parameters).
- The cross-attention injection strategy from IP-Adapter performs poorly in the Cross Attention baseline, suggesting that emotion is a higher-level semantic concept that requires feature-level fusion rather than injection at the UNet level.

## Rating

⭐⭐⭐⭐ — The first definition of the C-EICG task is pioneering, the emotional embedding network is well-designed, and the user study is thorough. However, practical applicability is constrained by the inherent difficulty of Arousal control and semantic drift issues. The work offers valuable insights for the affective computing and creative generation communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[ICCV 2025\] FlowTok: Flowing Seamlessly Across Text and Image Tokens](flowtok_flowing_seamlessly_across_text_and_image_tokens.md)
- [\[ICCV 2025\] What Makes for Text to 360-degree Panorama Generation with Stable Diffusion?](what_makes_for_text_to_360-degree_panorama_generation_with_stable_diffusion.md)

</div>

<!-- RELATED:END -->
