---
title: >-
  [Paper Note] Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling
description: >-
  [ICML2026][Multimodal VLM][Diffusion Reward Modeling] DiNa-LRM is proposed, rooting preference learning directly in the noisy latent space of diffusion models. By using a noise-calibrated Thurstone likelihood and inferen…
tags:
  - "ICML2026"
  - "Multimodal VLM"
  - "Diffusion Reward Modeling"
  - "Preference Alignment"
  - "Noise-Calibrated Thurstone"
  - "Latent Space Reward"
  - "Test-Time Ensemble"
date: 2026-05-08
content_hash: f9f2eeab6194a01c
---

# Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling

**Conference**: ICML2026  
**arXiv**: [2602.11146](https://arxiv.org/abs/2602.11146)  
**Code**: https://github.com/HKUST-C4G/diffusion-rm  
**Area**: Image Generation  
**Keywords**: Diffusion Reward Modeling, Preference Alignment, Noise-Calibrated Thurstone, Latent Space Reward, Test-Time Ensemble

## TL;DR
DiNa-LRM is proposed, rooting preference learning directly in the noisy latent space of diffusion models. By using a noise-calibrated Thurstone likelihood and inference-time multi-noise ensemble, it achieves preference prediction accuracy close to SOTA VLM reward models at significantly lower computational costs.

## Background & Motivation

**Background**: Preference alignment for Diffusion/Flow-Matching models (e.g., ReFL, DPO, GRPO) relies on reward models to provide supervisory signals. Current mainstream approaches use VLMs (e.g., Qwen2VL-7B) as reward backbones to score generated images in pixel space.

**Limitations of Prior Work**: VLM reward models suffer from two core issues. First, computational and memory costs are high, as reward evaluation is repeatedly invoked during alignment training. Second, there is a **latent-to-pixel mismatch** between latent space diffusion generators and pixel-space VLM rewards, necessitating additional VAE decoding steps and complicating reward-gradient-based alignment methods.

**Key Challenge**: While diffusion pre-training has been shown to learn rich discriminative representations (transferable to classification or adversarial discrimination), existing work has not fully exploited its potential as a general-purpose reward model—especially in scenarios requiring scoring of clean samples, similar to VLMs.

**Goal**: Construct a reward model that operates directly in the diffusion latent space, achieving: (1) preference prediction accuracy near VLM rewards, (2) memory and computation efficiency for alignment training, and (3) a robust, scalable scoring mechanism at inference time.

**Key Insight**: It is observed that diffusion models provide multiple "perspectives" of the same sample across different noise levels. Explicitly introducing noise uncertainty calibration in preference modeling allows these complementary perspectives to be leveraged for enhanced robustness.

**Core Idea**: Extend the Thurstone preference model from clean samples to diffusion noise states. Calibrate the preference likelihood using comparison uncertainty proportional to noise levels, and achieve test-time scaling through multi-noise ensembles during inference.

## Method

### Overall Architecture
The input consists of preference pairs $(\bm{x}_0^+, \bm{x}_0^-)$ with a text prompt $\bm{c}$ (in VAE latent space). Forward noise is added to obtain $(\bm{x}_t^+, \bm{x}_t^-)$. A pre-trained diffusion backbone (SD3.5-Medium) extracts multi-layer visual/textual features, which are modulated by timestep via FiLM and fed into a gated Q-Former reward head to output a scalar reward $r_\theta(\bm{x}_t, t, \bm{c})$. Training utilizes a noise-calibrated Thurstone likelihood + Fidelity Loss; inference supports single-noise evaluation or token-level multi-noise ensembles.

### Key Designs

1. **Noise-Calibrated Thurstone Preference Modeling**:

    - **Function**: Extends preference learning from clean samples to diffusion noise states, aligning the reward model's input distribution with diffusion pre-training.
    - **Mechanism**: The standard Thurstone model assumes perceived quality $u = r_\theta(\bm{x}_0, \bm{c}) + \eta$ where $\eta \sim \mathcal{N}(0, \sigma_u^2)$. This work sets the comparison uncertainty as a function of the noise level $\sigma_u^2(t) = k \cdot \sigma^2(t) + \sigma_u^2$, where $k=2$ and $\sigma_u=0.1$. The preference probability becomes $\mathbb{P}(\bm{x}_t^+ \succ \bm{x}_t^-) = \Phi\big(\frac{r_\theta(\bm{x}_t^+, t, \bm{c}) - r_\theta(\bm{x}_t^-, t, \bm{c})}{\sqrt{2\sigma_u^2(t)}}\big)$. High-noise regions automatically produce more conservative likelihoods, preventing uninformative gradients from destabilizing training.
    - **Design Motivation**: Diffusion backbones are pre-trained on noisy states. Learning directly on $\bm{x}_0$ causes distribution shift. Noise calibration allows the model to learn diverse, complementary features across noise levels, benefiting inference-time ensemble.

2. **Timestep-Aware Latent Reward Architecture**:

    - **Function**: Extracts multi-layer features from the pre-trained diffusion backbone and aggregates them into a scalar reward after timestep-conditioned adaptation.
    - **Mechanism**: Visual and textual token features are extracted from a selected set of layers $\mathcal{S}$. FiLM modulation (based on timestep embedding $t_{\text{emb}}$) is applied to each layer's features, which are then projected and concatenated into unified visual $\mathbf{V}_t$ and textual $\mathbf{T}_t$ sequences. A Q-Former, using $N_q$ learnable query tokens and value-gated cross-attention, aggregates these sequences. Finally, a scalar $r_\theta = \text{MLP}(\text{Pool}(\tilde{\mathbf{Q}}))$ is produced.
    - **Design Motivation**: FiLM modulation enables the reward head to explicitly perceive noise levels. The query-based architecture naturally handles variable-length inputs, providing a seamless interface for multi-noise ensembles.

3. **Inference-Time Multi-Noise Ensemble (Test-Time Scaling)**:

    - **Function**: Generates robust reward scores by aggregating features across multiple noise levels, acting as a diffusion-native test-time scaling knob.
    - **Mechanism**: A clean sample $\bm{x}_0$ is noised at $K$ different timesteps $\{t_k\}_{k=1}^K$. Features are extracted and adapted via FiLM for each timestep. Token features across all timesteps are concatenated into $\mathbf{V}_{\text{ensemble}} \in \mathbb{R}^{(K \times N_v) \times C}$ and scored simultaneously by the Q-Former head. Default settings use $t \in \{0.2, 0.5, 0.7\}$ to cover low/medium/high noise ranges.
    - **Design Motivation**: Different noise levels emphasize different representation aspects (low noise for details, high noise for global semantics). Token-level concatenation is more flexible than simple averaging, allowing the Q-Former to learn cross-timestep attention weights.

### Loss & Training
Optimized using Fidelity Loss $\mathcal{L}_{\text{fid}} = \mathbb{E}[1 - \sqrt{y\hat{p}_\theta + (1-y)(1-\hat{p}_\theta)}]$, with timesteps sampled uniformly from $\mathcal{U}(0,1)$. Trained on HPDv3 (~0.8M pairs) for 1 epoch using 8 GPUs, AdamW (lr=$5 \times 10^{-5}$), and EMA decay of 0.995. The backbone is fine-tuned using LoRA.

## Key Experimental Results

### Main Results

| Category | Model | Backbone | ImageReward | HPDv2 | HPDv3 | GenAI-Bench | Average |
|---------|------|------|-------------|-------|-------|-------------|------|
| CLIP-based | MPS | CLIP | 66.37 | 83.27 | 64.33 | 68.08 | 70.51 |
| VLM-based | HPSv3 | Qwen2VL-7B | 67.03 | **85.36** | **76.03** | **70.95** | **74.84** |
| VLM-based | UnifiedReward | LLaVA-OV-7B | 63.82 | 83.10 | 71.96 | 72.38 | 72.81 |
| Diffusion-based | LRM-SDXL | SDXL | 60.35 | 71.19 | 53.80 | 61.58 | 61.73 |
| Diffusion-based | **DiNa-LRM** | SD3.5-M-2B | 60.34 | 82.13 | 75.04 | 68.43 | 71.49 |
| Diffusion-based | **DiNa-LRM*** | SD3.5-M-2B | 61.75 | 84.31 | 74.86 | 68.98 | **72.48** |

Ours improves average accuracy by **+9.76%** over the previous diffusion reward baseline LRM-SDXL and approaches the strongest VLM reward HPSv3 (72.48 vs 74.84).

### Ablation Study

| Configuration | HPDv2 | HPDv3 | GenAI-Bench | Average |
|------|-------|-------|-------------|------|
| Uniform + Noise-Calibrated (Full Model) | 82.13 | 75.04 | 68.43 | 71.49 |
| Uniform + Fixed variance | 78.72 | 75.11 | 68.01 | 70.68 |
| Const $t=0$ + Fixed | 59.20 | 74.37 | 67.55 | 64.93 |
| Uniform + Noise-Calibrated + Ensemble | **84.31** | 74.86 | **68.98** | **72.48** |
| Freeze backbone (No LoRA) | — | 73.52 | 67.09 | 70.27 |

### Alignment Efficiency Analysis (ReFL on SD3.5-M, 1024×1024)

| Metric | HPSv3 (VLM) | DiNa-LRM | Savings |
|------|-------------|----------|------|
| Peak Memory | ~40 GB | ~19.4 GB | **51.4%** |
| Reward TFLOPS | ~8.5 | ~2.5 | **71.1%** |
| Optimization TFLOPS | ~14 | ~7.5 | **46.4%** |

### Key Findings
- **Noise-calibrated variance is a core contribution**: It improves HPDv2 performance from 78.72 to 82.13 (+3.4%), and up to 84.31 with ensemble (+6.2%), indicating that noise-aware uncertainty modeling enables the learning of complementary features across timesteps.
- Optimal inference noise levels are within $t \in [0.3, 0.7]$. Accuracy drops when samples are too clean ($t=0$) or too noisy ($t=0.8$).
- Distributed timestep sampling (Uniform/LogitNormal) significantly outperforms fixed timestep training, raising average accuracy from 64.93–68.75 to 70.58–71.49.
- In ReFL alignment, DiNa-LRM's proxy scores converge faster and correlate with the gold standard (PickScore) without significant reward hacking.

## Highlights & Insights
- **Feasibility of Diffusion Models as General Reward Backbones**: Diffusion pre-trained representations are capable of high-quality preference discrimination as well as generation. This supports a "one backbone, two purposes" paradigm, keeping the entire alignment pipeline within latent space.
- **Elegance of Noise-Calibrated Thurstone**: A simple linear relationship $\sigma_u^2(t) = k\sigma^2(t) + \sigma_u^2$ unifies diffusion noise scheduling with uncertainty modeling in preference learning.
- **Token-Level Ensemble Beats Score Averaging**: Aggregating features via Q-Former attention rather than averaging scalar scores is more effective. This design is transferable to other discriminative tasks requiring multi-perspective fusion.

## Limitations & Future Work
- Rewards are learned and evaluated in specific backbone latent spaces, limiting cross-backbone transferability (e.g., SD3.5 to FLUX requires retraining).
- Latent modeling may overlook certain pixel-level artifacts (e.g., texture distortion). Long-range optimization might lead to reward hacking, such as spurious object insertion or style drift.
- Performance on the ImageReward test set (~61%) remains lower than VLM methods (~67%), suggesting certain gaps in semantic understanding.
- Future directions: (1) training on stronger unified backbones for better generalization, (2) adding lightweight pixel-space regularization, (3) exploring generative or dense reward modeling.

## Related Work & Insights
- **CLIP-based RM** (ImageReward, PickScore, HPSv2): Efficient but limited by CLIP's representation upper bound.
- **VLM-based RM** (HPSv3, UnifiedReward): High accuracy but computationally expensive and operates in pixel space.
- **Diffusion Discriminative Representations** (DDPMClassifier, DiffAE): Prior work proving diffusion features can transfer to discriminative tasks like classification.
- **Concurrent Work LRM** (Zhang et al., 2025): Utilizes step-level rewards on intermediate noisy states for trajectory optimization, whereas Ours targets clean sample scoring for general preference alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling](mitigating_perceptual_judgment_bias_in_multimodal_llm-as-a-judge_via_perceptual_.md)
- [\[CVPR 2026\] VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery](../../CVPR2026/multimodal_vlm/vlm-guided_group_preference_alignment_for_diffusion-based_human_mesh_recovery.md)
- [\[ICLR 2026\] GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?](../../ICLR2026/multimodal_vlm/glyph-sr_can_we_achieve_both_high-quality_image_super-resolution_and_high-fideli.md)
- [\[ICML 2026\] Conditional Diffusion Sampling](conditional_diffusion_sampling.md)
- [\[ICCV 2025\] MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding](../../ICCV2025/multimodal_vlm/musevl_modeling_unified_vlm_through_semantic_discrete_encodi.md)

</div>

<!-- RELATED:END -->
