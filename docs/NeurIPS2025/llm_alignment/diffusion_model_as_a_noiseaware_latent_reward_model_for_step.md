---
title: >-
  [Paper Note] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization
description: >-
  [NeurIPS 2025][LLM Alignment][diffusion model] This paper proposes the Latent Reward Model (LRM) and Latent Preference Optimization (LPO)…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "diffusion model"
  - "preference optimization"
  - "reward model"
  - "latent space"
  - "step-level"
  - "noise-aware"
  - "DPO"
date: 2026-05-08
content_hash: f65870fa03b9e34f
---

# Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2502.01051](https://arxiv.org/abs/2502.01051)
**Code**: [https://github.com/Kwai-Kolors/LPO](https://github.com/Kwai-Kolors/LPO)
**Area**: Image Generation / Preference Optimization
**Keywords**: diffusion model, preference optimization, reward model, latent space, step-level, noise-aware, DPO
**Institution**: Institute of Automation, Chinese Academy of Sciences + Kuaishou Technology

## TL;DR
This paper proposes the Latent Reward Model (LRM) and Latent Preference Optimization (LPO), which repurpose the pretrained diffusion model itself as a noise-aware latent-space reward model to perform step-level preference optimization directly in the noisy latent space. Compared to Diffusion-DPO, LPO achieves a 10–28× training speedup; compared to SPO, it achieves a 2.5–3.5× speedup.

## Background & Motivation

### Three Key Limitations of Prior Work
Existing step-level preference optimization methods (e.g., SPO) employ VLMs (such as CLIP) as pixel-space reward models (PRMs), which suffer from three critical problems:

1. **Complex transformation**: At each timestep $t$, an additional diffusion forward pass ($x_t \rightarrow \hat{x}_{0,t}$) followed by VAE decoding ($\hat{x}_{0,t} \rightarrow I_t$) is required to obtain a pixel image for the VLM, resulting in a sampling time 6× longer than that of LRM.
2. **Incompatibility with high noise levels**: At large timesteps (high noise), the predicted pixel images are severely blurred, leading to a significant distribution shift from the VLM's training data (clean images), causing PRM predictions to be unreliable at high noise levels.
3. **Timestep insensitivity**: PRMs do not take the timestep as input and thus cannot capture the varying influence of different denoising stages on image evaluation.

### Core Insight
**A pretrained diffusion model inherently satisfies all requirements for step-level reward modeling**:
- It possesses text–image alignment capability (from large-scale text–image pretraining).
- It can directly process noisy latent images $x_t$ without additional decoding.
- It is compatible with high noise levels (pretraining covers all noise levels).
- It is naturally sensitive to the denoising timestep.

## Method

### 1. Latent Reward Model (LRM) Architecture

LRM reuses the U-Net (or DiT) and text encoder components of the diffusion model:

- **Text features**: The text encoder extracts prompt features $f_p$; the EOS token feature $f_{\text{eos}}$ is passed through a text projection layer to obtain the final text feature $T \in \mathbb{R}^{1 \times n_d}$.
- **Visual features**: The noisy latent image $x_t$ is passed through the U-Net; spatial average pooling is applied to obtain multi-scale down-block features $V_{\text{down}}$ and mid-block features $V_{\text{mid}}$.
- **Visual Feature Enhancement (VFE)**: Inspired by Classifier-Free Guidance, an unconditional (text-free) mid-block feature $V_{\text{mid\_uncond}}$ is additionally extracted to enhance the text relevance of visual features: $V_{\text{enh}} = V_{\text{mid}} + (g_s - 1) \cdot (V_{\text{mid}} - V_{\text{mid\_uncond}})$, with $g_s = 7.5$.
- **Preference score**: $V_{\text{enh}}$ and $V_{\text{down}}$ are concatenated and projected to yield the visual feature $V$; the final score is $S(p, x_t) = \tau \cdot \ell_2(V) \cdot \ell_2(T)$ (CLIP-style dot product).

Effect of VFE: Larger $g_s$ strengthens text-alignment relevance (higher CLIP-Corr) while moderately reducing aesthetic relevance (Aes-Corr); $g_s = 7.5$ achieves the best balance.

### 2. Multi-Preference Consistent Filtering (MPCF)

**Problem**: In the training data, approximately half of the winning images are aesthetically inferior to the losing images, and about 40% score lower on CLIP/VQA metrics. Preference rankings may reverse after adding noise.

**Solution**: The Pick-a-Pic v1 dataset is filtered along three dimensions — aesthetic score $S_A$, CLIP score $S_C$, and VQA score $S_V$:
- Strategy 1 (strictest): $G_A \geq 0, G_C \geq 0, G_V \geq 0$ → 101K pairs, but LRM overfits to aesthetics.
- **Strategy 2 (adopted)**: $G_A \geq -0.5, G_C \geq 0, G_V \geq 0$ → 169K pairs, best balance between aesthetics and alignment.
- Strategy 3 (most lenient): $G_A \geq -1, G_C \geq 0, G_V \geq 0$ → 202K pairs, LRM neglects aesthetics.

### 3. Latent Preference Optimization (LPO)

**Sampling**: At each timestep $t$, $K=4$ samples $x_t^i$ are drawn from the same $x_{t+1}$. LRM directly predicts preference scores $S_t^i$ in the noisy latent space; the highest-scoring sample is designated $x_t^w$ and the lowest-scoring $x_t^l$ (requiring that the SoftMax-normalized score difference exceeds threshold $th_t$).

**Training objective**: The same step-level DPO loss as SPO (Eq. 6), but performed entirely in the noisy latent space without $\hat{x}_{0,t}$ prediction or VAE decoding.

**Timestep coverage $t \in [0, 950]$**: Because the SPM is unreliable at high noise levels, SPO is limited to $t \in [0, 750]$. As a noise-aware model, LRM covers the full denoising process. Ablation studies show that the high-noise range $t \in [750, 950]$ is critical for preference optimization.

**Dynamic threshold**: $\sigma_t$ decreases as $t$ decreases; a fixed threshold performs poorly. A linear mapping is used: $th_t \in [th_{\min}, th_{\max}] = [0.35, 0.5]$ (SD1.5) / $[0.45, 0.6]$ (SDXL), with lower thresholds applied at smaller timesteps.

**Homogeneous / heterogeneous optimization**: LRM and the model being optimized (DMO) may share the same architecture (homogeneous) or differ (heterogeneous); the only constraint is a shared VAE encoder. Experiments demonstrate that an LRM trained on SD1.5 can effectively fine-tune SD2.1 (same VAE), but fails to fine-tune SDXL (different VAE).

## Key Experimental Results

### Main Results (SD1.5 / SDXL)

| Metric | SD1.5 Base | SPO | LPO | SDXL Base | SPO | LPO |
|--------|-----------|-----|-----|-----------|-----|-----|
| PickScore | 20.56 | 21.22 | **21.69** | 21.65 | 22.70 | **22.86** |
| ImageReward | 0.008 | 0.168 | **0.659** | 0.478 | 0.995 | **1.217** |
| Aesthetic | 5.468 | 5.927 | **5.945** | 5.920 | 6.343 | **6.360** |
| GenEval(20s) | 42.56 | 40.46 | **48.39** | 49.40 | 50.52 | **59.27** |

On SDXL, LPO even slightly surpasses InterComp, which uses an internal high-quality dataset.

### T2I-CompBench++ (Fine-Grained Text–Image Alignment)
LPO comprehensively outperforms SPO and Diffusion-DPO across all dimensions, including color, shape, texture, spatial relationships, and counting.

### Training Efficiency

| Method | SD1.5 Total Training | SDXL Total Training |
|--------|---------------------|---------------------|
| Diffusion-DPO | 240 A100h | 2560 A100h |
| SPO | 80 A100h | 234 A100h |
| **LPO** | **23 A100h** | **92 A100h** |

Per-step sampling: LRM 0.039s vs. SPM 0.243s (6.2× speedup), by eliminating $\hat{x}_{0,t}$ prediction and VAE decoding.

### Ablation Study
- **Timestep range**: The full range $[0, 950]$ is optimal; using only $[750, 950]$ (high-noise segment) already achieves near-full performance, confirming the importance of high-noise step-level optimization.
- **MPCF strategy**: LPO without MPCF still outperforms SPO, demonstrating the inherent advantage of LRM; MPCF provides additional improvement.
- **Dynamic threshold**: Outperforms all fixed-threshold configurations; $[0.35, 0.5]$ is optimal.

## Highlights & Insights
1. **Original insight**: "The diffusion model itself is the best step-level reward model" — transforming the diffusion model from an optimization target into a source of reward signals, and performing reward modeling in the noisy latent space for the first time.
2. **Substantial efficiency gains**: By eliminating round-trip computation through pixel space, the full SD1.5 optimization pipeline requires only 23 A100h.
3. **High-noise coverage**: LRM reliably predicts preferences at $t \in [750, 950]$, overcoming the high-noise limitation of PRMs.
4. **VFE module**: Borrowing the CFG idea to enhance the text relevance of visual features — simple yet effective.
5. **Heterogeneous optimization**: A lower-capacity LRM can fine-tune a higher-capacity model, provided they share the same VAE encoder.

## Limitations & Future Work
- The accuracy of LRM preference predictions is bounded by the representation quality of the diffusion model itself.
- In homogeneous optimization, sharing parameters between LRM and DMO may introduce bias.
- Heterogeneous optimization requires an identical VAE encoder, limiting cross-architecture generalization.
- Validation is limited to image generation; extension to video diffusion models remains unexplored.
- MPCF relies on external scorers (Aesthetic Score, CLIP Score), introducing additional computational cost.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to repurpose a diffusion model as a noise-aware reward model.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ SD1.5 / SDXL / SD3 + multi-dimensional evaluation + extensive ablations + heterogeneous optimization.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and intuitive diagrams.
- Value: ⭐⭐⭐⭐⭐ A practical solution with 10–28× speedup and open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)
- [\[NeurIPS 2025\] What Makes a Reward Model a Good Teacher? An Optimization Perspective](what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)
- [\[NeurIPS 2025\] Ask a Strong LLM Judge when Your Reward Model is Uncertain](ask_a_strong_llm_judge_when_your_reward_model_is_uncertain.md)
- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[NeurIPS 2025\] Preference Optimization by Estimating the Ratio of the Data Distribution](preference_optimization_by_estimating_the_ratio_of_the_data_distribution.md)

</div>

<!-- RELATED:END -->
