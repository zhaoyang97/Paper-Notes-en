---
title: >-
  [Paper Note] Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation
description: >-
  [NeurIPS 2025][Image Generation][Auto-regressive model acceleration] This paper proposes Distilled Decoding 2 (DD2), which reinterprets auto-regressive image models as conditional score models and designs a Conditional S…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Auto-regressive model acceleration"
  - "one-step generation"
  - "score distillation"
  - "conditional score distillation"
date: 2026-05-08
content_hash: 1ecb85ce39d04dad
---

# Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation

**Conference**: NeurIPS 2025
**arXiv**: [2510.21003](https://arxiv.org/abs/2510.21003)
**Code**: [GitHub](https://github.com/imagination-research/Distilled-Decoding-2)
**Area**: Image Generation / Model Acceleration
**Keywords**: Auto-regressive model acceleration, one-step generation, score distillation, conditional score distillation, image generation

## TL;DR

This paper proposes Distilled Decoding 2 (DD2), which reinterprets auto-regressive image models as conditional score models and designs a Conditional Score Distillation (CSD) loss to compress multi-step AR sampling into one-step generation. On ImageNet-256, DD2 achieves only a marginal FID degradation from 3.40 to 5.43 while obtaining 8.0× speedup (VAR) and 238× speedup (LlamaGen), closing 67% of the performance gap relative to DD1 and training 12.3× faster.

## Background & Motivation

**Background**: Image auto-regressive (AR) models such as VAR and LlamaGen have achieved state-of-the-art image generation quality, surpassing VAEs, GANs, and diffusion models. These models generate images by predicting discrete tokens sequentially while preserving favorable likelihood modeling properties.

**Limitations of Prior Work**: The fundamental bottleneck of AR models lies in their sequential sampling procedure — VAR requires 10 steps and LlamaGen requires 256 steps, each involving a full model forward pass, making generation far slower than one-step methods such as GANs. Set prediction approaches (e.g., MaskGIT-style parallel token prediction) completely lose inter-token correlations in the extreme one-step case. Speculative decoding achieves only limited compression of less than 3× on image AR models. DD1 is the first method to achieve one-step sampling but relies on a predefined noise-to-data mapping that is difficult to learn, resulting in notable FID degradation (VAR-d20: 3.40 → 9.55) and slower training.

**Key Challenge**: DD1 converts each AR sampling step into ODE solving under flow matching to construct a deterministic mapping, essentially exploiting score signals only for building the mapping. This predefined mapping (1) is difficult for the model to learn, and (2) limits flexibility — models such as GANs and VAEs that do not rely on explicit mappings are in fact more broadly applicable in downstream tasks.

**Goal**: Can a one-step generator be trained such that its output distribution matches a given AR model, without relying on any predefined mapping?

**Key Insight**: The paper reinterprets AR models as conditional score models — given all preceding tokens, the next-token probability vector output by an AR model enables analytic computation of the conditional score in the codebook embedding space. Drawing on score distillation from diffusion models, the conditional scores of the generator and the teacher AR model are aligned at every token position.

**Core Idea**: The next-token probability vector of an AR model is treated as the source of conditional scores. Through Conditional Score Distillation (CSD), the generator is aligned with the teacher's conditional distribution at all token positions simultaneously, enabling one-step AR image generation without any predefined mapping.

## Method

### Overall Architecture

Training proceeds in two stages: (1) **Initialization stage** — the classification head of the teacher AR model is replaced with an MLP head and fine-tuned into an AR-diffusion model using the GTS loss; (2) **CSD training stage** — the AR-diffusion model initializes both the generator and the guidance network, which are then alternately trained using the CSD loss and the FCS loss, respectively. At inference time, the generator produces the entire token sequence in a single forward pass.

### Key Designs

1. **Teacher AR Model as a Conditional Score Model**

   - **Function**: Converts discrete probability vectors into continuous score signals.
   - **Mechanism**: The sampling of token $q_i$ is viewed as a flow matching process from a weighted sum of Dirac functions to a Gaussian distribution. Given the teacher's output probability vector $p = (p_1, \ldots, p_V)$ and flow matching timestep $t$, the conditional score is computed analytically as:
     $$s(x_t, t, p) = -\frac{\sum_j p_j(x_t - (1-t)c_j)e^{-\frac{(x_t-(1-t)c_j)^2}{2t^2}}}{t^2 \sum_j p_j e^{-\frac{(x_t-(1-t)c_j)^2}{2t^2}}}$$
     Unlike DD1, DD2 does not use this score to construct an ODE mapping but employs it directly for distillation.
   - **Design Motivation**: AR models implicitly encode complete conditional score information at every token position. DD1 exploits only a portion of this information (for mapping construction), whereas DD2 utilizes it more fully.

2. **Conditional Score Distillation Loss (CSD Loss)**

   - **Function**: Trains a one-step generator whose output sequence joint distribution matches the teacher AR model.
   - **Mechanism**: At each token position $i$, the teacher's conditional score $s_\Phi(q_i^{t_i}, t_i | q_{<i})$ is aligned with the fake conditional score $s_{\text{fake}}(q_i^{t_i}, t_i | q_{<i})$ learned by the guidance network. The CSD loss sums score distillation losses across all positions:
     $$\mathcal{L}_{CSD} = \mathbb{E}\sum_{i=1}^n d\!\left(s_\Phi(q_i^{t_i}, t_i | sg(q_{<i})),\, s_{\text{fake}}(q_i^{t_i}, t_i | sg(q_{<i}))\right)$$
     where $sg(\cdot)$ denotes stop-gradient. The SiD loss form is adopted. A key theoretical guarantee (Proposition 1) establishes that minimizing the CSD loss implies that the generator's joint distribution matches the teacher's.
   - **Design Motivation**: The progressive alignment logic — first aligning the marginal distribution of $q_1$ (unconditional), then $q_2 | q_1$ conditioned on it, and so on — is more tractable to optimize than directly aligning the full joint distribution.

3. **AR-Diffusion Initialization Strategy**

   - **Function**: Provides a well-conditioned initialization for both the generator and the guidance network.
   - **Mechanism**: Directly initializing from the teacher AR model is infeasible since the teacher outputs discrete probabilities while the generator must output continuous values. The solution replaces the teacher's classification head with an MLP and fine-tunes it using the Ground Truth Score (GTS) loss:
     $$\mathcal{L}_{GTS} = \mathbb{E}\sum_{i=1}^n \|s_\psi(q_i^{t_i}, t_i | q_{<i}) - s_\Phi(q_i^{t_i}, t_i | q_{<i})\|^2$$
     i.e., directly regressing the analytic score of the teacher. The resulting model serves as the initialization for both the generator and the guidance network.
   - **Design Motivation**: Experiments demonstrate that without proper initialization, score distillation collapses entirely. Even randomly initializing only the last layer of the generator leads to severe performance degradation. The GTS loss converges faster and more stably than the standard AR-diffusion loss.

### Loss & Training

- **Generator training**: CSD loss (Eq. 4), using the SiD loss form.
- **Guidance network training**: FCS loss (Eq. 5), standard score matching loss.
- The two networks are trained alternately.
- The GTS initialization stage performs full fine-tuning; the subsequent CSD training stage is substantially shorter than that of DD1.

## Key Experimental Results

### Main Results

One-step generation quality on ImageNet-256:

| Method | Model | FID↓ | IS↑ | Steps | Speedup |
|--------|-------|------|-----|-------|---------|
| VAR-d20 (original) | 600M | 3.40 | 305.1 | 10 | 1× |
| DD1 | VAR-d20 | 9.55 | 197.2 | 1 | — |
| **DD2** | **VAR-d20** | **5.43** | **233.7** | **1** | **8.0×** |
| LlamaGen-L (original) | 343M | 4.11 | 283.5 | 256 | 1× |
| DD1 | LlamaGen-L | 11.35 | 193.6 | 1 | — |
| **DD2** | **LlamaGen-L** | **8.59** | **229.1** | **1** | **238×** |

DD2 achieves better one-step FID than DD1's two-step results across all configurations.

### Ablation Study

| Configuration | FID (LlamaGen) | FID (VAR-d24) | Note |
|---------------|---------------|---------------|------|
| Gui-Init ✓ Gen-Init ✓ | 14.77 | 11.53 | Full initialization (best) |
| Gui-Init ✓ Gen-Init ✗ | 16.08 | >200 (collapse) | Missing generator initialization |
| Gui-Init ✗ Gen-Init ✓ | 21.76 | >200 (collapse) | Missing guidance initialization |

Training efficiency comparison:

| Method | Model | GPU Hours (8×A800) | Training Speedup |
|--------|-------|-------------------|-----------------|
| DD1 | LlamaGen-L | 647.7 | 1× |
| DD2 | LlamaGen-L | 52.6 | **12.3×** |
| DD1 | VAR-d24 | 604.2 | 1× |
| DD2 | VAR-d24 | 96.1 | **6.3×** |

### Key Findings

- **DD2 substantially outperforms DD1**: On VAR, DD2 closes 67% of the performance gap between the teacher and one-step generation.
- **Initialization is critical**: Omitting either initialization component causes collapse (FID > 200 on VAR-d24); even randomly initializing only the last layer leads to severe degradation.
- **High training efficiency**: Up to 12.3× training speedup, as DD2 does not require pre-constructing all noise-to-data mappings as DD1 does.
- **Lower PPL**: DD2's Perceptual Path Length (7231.9 vs. 18437.6) is far lower than DD1's, indicating a smoother learned latent space.
- **Multi-step sampling as a bonus**: DD2 at 3 steps (FID 4.88) outperforms DD1 at 6 steps (FID 5.03), offering a flexible quality-speed trade-off.

## Highlights & Insights

- **The reinterpretation of AR models as score models is particularly elegant**: AR models originally output discrete next-token probabilities; this paper treats these as score signals in the continuous embedding space. This shift in perspective allows the entire theoretical apparatus and toolset of diffusion distillation to be directly transferred to AR acceleration — a cross-paradigm knowledge transfer well worth attention.
- **The core insight of "eliminating predefined mappings"**: DD1's mapping is a hard target (the model must precisely learn each noise-to-data correspondence), whereas DD2's CSD is a soft target (only distributional matching is required), making the latter substantially easier to optimize. By analogy: DD1 learns the unique answer to a fill-in-the-blank question, while DD2 learns the distribution over all valid answers.
- **General value of the initialization strategy**: The GTS loss — which substitutes the teacher's analytic score for Monte Carlo estimation — converges faster and more stably, a technique transferable to other score distillation settings.

## Limitations & Future Work

- **Residual performance gap**: The FID degradation from 3.40 to 5.43 on VAR-d20 (a gap of 2.03) may be unacceptable in quality-critical applications.
- **Validation limited to VQ-based AR models**: Compatibility with continuous-space AR models such as MAR is discussed theoretically but not verified experimentally.
- **Evaluation restricted to ImageNet-256**: The method has not been extended to more practical settings such as text-to-image generation.
- **Non-trivial absolute training cost**: Although 12× faster than DD1, training still requires 8× A800 GPUs.
- **Strong dependence on initialization is a double-edged sword**: Heavy reliance on GTS initialization implies that a low-quality teacher model could adversely affect DD2's performance.

## Related Work & Insights

- **vs. DD1**: DD1 constructs a deterministic mapping via flow matching and fits it; DD2 performs distributional matching via score distillation. DD2 comprehensively outperforms DD1 in generation quality (−67% performance gap), training speed (12.3×), and latent space smoothness (PPL halved).
- **vs. Diffusion Score Distillation (DMD/SiD)**: Although both perform score matching, the generation processes of AR models and diffusion models are fundamentally different — AR models perform sequential conditional generation, while diffusion models perform iterative denoising. DD2's contribution lies in extending score distillation from independent variables to sequential variables with AR dependencies.
- **vs. Set-of-token prediction (MaskGIT/VAR)**: Predicting all tokens in a single step discards inter-token correlations (as illustrated by the {(0,0),(1,1)} example). DD2 preserves these correlations by performing conditional score alignment at all positions via CSD.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Introduces score distillation into AR model acceleration; the reinterpretation of AR models as score models is highly creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers two AR model families (VAR + LlamaGen), multiple model sizes, and includes comprehensive ablations and comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation and theoretical derivations are clear, though Section 3 is notation-dense.
- **Value**: ⭐⭐⭐⭐⭐ — A milestone contribution to AR image generation acceleration with potential to reshape the speed comparison between AR and diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](../../ICCV2025/image_generation/sana-sprint_one-step_diffusion_with_continuous-time_consistency_distillation.md)
- [\[CVPR 2026\] WaDi: Weight Direction-aware Distillation for One-step Image Synthesis](../../CVPR2026/image_generation/wadi_weight_direction-aware_distillation_for_one-step_image_synthesis.md)
- [\[NeurIPS 2025\] Ψ-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models](psi-sampler_initial_particle_sampling_for_smc-based_inference-time_reward_alignm.md)
- [\[ICCV 2025\] LUSD: Localized Update Score Distillation for Text-Guided Image Editing](../../ICCV2025/image_generation/lusd_localized_update_score_distillation_for_text-guided_image_editing.md)
- [\[CVPR 2026\] DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](../../CVPR2026/image_generation/duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md)

</div>

<!-- RELATED:END -->
