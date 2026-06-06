---
title: >-
  [Paper Note] Bootstrap Your Generator: Unpaired Visual Editing with Flow Matching
description: >-
  [ICML2026][Image Generation][unpaired training] The authors propose Bootstrap Your Generator (ByG), a flow matching editing training framework that requires no paired data. By extracting editing direction priors from a f…
tags:
  - "ICML2026"
  - "Image Generation"
  - "unpaired training"
  - "flow matching"
  - "image editing"
  - "video editing"
  - "cycle consistency"
date: 2026-05-08
content_hash: 61d62b874e7b3300
---

# Bootstrap Your Generator: Unpaired Visual Editing with Flow Matching

**Conference**: ICML2026  
**arXiv**: [2606.03911](https://arxiv.org/abs/2606.03911)  
**Code**: https://research.nvidia.com/labs/par/byg/ (Project Page)  
**Area**: image_generation  
**Keywords**: unpaired training, flow matching, image editing, video editing, cycle consistency  

## TL;DR
The authors propose Bootstrap Your Generator (ByG), a flow matching editing training framework that requires no paired data. By extracting editing direction priors from a frozen base model, maintaining source structure via cycle consistency, and bridging the training-inference gap with gradient routing, ByG outperforms supervised baselines trained on millions of paired data points in both image and video editing tasks.

## Background & Motivation

**Background**: Current mainstream visual editing methods (e.g., FLUX-Kontext, Qwen-Image-Edit, Ditto) rely on supervised training with large-scale paired datasets (millions of source-target pairs). While effective, the cost of data acquisition is extremely high.

**Limitations of Prior Work**: Collecting paired data is nearly impossible for long-tail editing scenarios (e.g., converting cartoons to photorealism, changing liquid viscosity) and video editing, as ground-truth "before-after" pairs do not exist. Zero-shot methods (e.g., FlowEdit) do not require paired data but offer limited quality. NP-Edit avoids paired data but relies on external VLM feedback and struggles to scale to multi-step generative models and videos.

**Key Challenge**: Supervised methods require paired data which is hard to scale, while unsupervised methods lack effective training signals. For flow matching models, additional challenges exist: standard training requires adding noise to ground-truth outputs as inputs, which is unavailable without pairs. Furthermore, training occurs on intermediate noisy states, whereas constraints like cycle consistency require clean, fully denoised outputs.

**Goal**: Design a general framework that utilizes only the knowledge of pre-trained generative models to train flow matching editing models without any paired data or external models.

**Key Insight**: Visual editing has two goals—following the editing instruction and preserving the source content. The former is achieved by extracting editing direction signals from the velocity field differences of a frozen T2I base model; the latter is realized through cycle consistency constraints. A gradient routing mechanism (an STE variant) is introduced to resolve the gap between noise prediction during training and clean outputs during inference.

## Method

### Overall Architecture
The input consists of a source image $\mathbf{x}$, an editing instruction $c$, and source/target text descriptions $(p_{\text{src}}, p_{\text{tgt}})$, without a paired target image $\mathbf{y}$. The framework fine-tunes a pre-trained T2I model $\mathbf{G}_{\text{t2i}}$ into an editing model $\mathbf{G}_{\text{edit}}$. Training is driven by three complementary signals: (1) an EMA copy of the editing model generates pseudo-noisy targets as inputs (bootstrapping); (2) a frozen T2I model provides editing direction priors (instruction following); (3) a cycle constraint ensures the source image is reconstructed after a forward edit and backward pass (source preservation). The total loss is $\mathcal{L} = \mathcal{L}_{\text{cycle}} + \lambda_{\text{prior}}(\mathcal{L}_{\text{prior}}^{\text{fwd}} + \mathcal{L}_{\text{prior}}^{\text{rev}}) + \lambda_{\text{id}}\mathcal{L}_{\text{id}}$.

### Key Designs

1. **Directional Prior Loss**:
    - **Function**: Provides training signals for instruction following without paired supervision.
    - **Mechanism**: A frozen T2I model queries the same noise input using $p_{\text{src}}$ and $p_{\text{tgt}}$ to obtain velocities $\mathbf{v}_{\text{src}}$ and $\mathbf{v}_{\text{tgt}}$. Instead of matching $\mathbf{v}_{\text{tgt}}$ directly (which causes content drift), the editing model's velocity change direction is constrained to align with the T2I model's editing direction $\mathbf{v}_{\text{tgt}} - \mathbf{v}_{\text{src}}$ using a cosine loss: $\mathcal{L}_{\text{dir}} = 1 - \cos(\mathbf{v}_{\text{fwd}} - \mathbf{v}_{\text{src}},\; \mathbf{v}_{\text{tgt}} - \mathbf{v}_{\text{src}})$. An MSE term $\alpha\|\mathbf{v}_{\text{fwd}} - \mathbf{v}_{\text{tgt}}\|^2$ is added to prevent norm explosion.
    - **Design Motivation**: The directional loss only constrains the orientation rather than the magnitude, preventing the model from being pulled toward the T2I model's unconditional reconstruction. The difference form isolates text-induced semantic changes, allowing cycle consistency to handle shared structure preservation.

2. **Gradient Routing via STE**:
    - **Function**: Bridges the gap between single-step blurred predictions during training and multi-step clean outputs during inference.
    - **Mechanism**: In the reverse pass of the cycle, the forward pass uses a clean estimate $\tilde{\mathbf{y}}_0$ obtained via full sampling from an EMA model as a condition (matching the inference distribution). During backpropagation, gradients bypass $\tilde{\mathbf{y}}_0$ and flow through the single-step prediction $\hat{\mathbf{y}}$: $\hat{\mathbf{y}}^{\text{hyb}} = \text{sg}(\tilde{\mathbf{y}}_0) + (\hat{\mathbf{y}} - \text{sg}(\hat{\mathbf{y}}))$, where $\text{sg}$ denotes stop-gradient.
    - **Design Motivation**: Using single-step predictions directly as conditions leads to blurred inputs, causing the model to ignore conditional signals. The STE adaptation allows the model to see clean images while gradients still update the forward edit, eliminating train-test mismatch.

3. **Bootstrapped Noisy Inputs**:
    - **Function**: Solves the absence of valid noisy inputs for flow matching training when paired data is unavailable.
    - **Mechanism**: Maintains an EMA copy of the editing model. At each training step, the EMA model samples $n$ steps from pure noise $t=1$ to time step $t$ to generate a pseudo-noisy input $\tilde{\mathbf{y}}_t$ for the trainable model. As training progresses, the EMA generates progressively better inputs, forming a bootstrap loop.
    - **Design Motivation**: The EMA smoothes training fluctuations and breaks the chicken-and-egg cycle where a good model cannot be trained without good inputs.

### Loss & Training
The total loss includes four terms: cycle reconstruction loss (forward edit $\mathbf{x} \to \hat{\mathbf{y}}$, backward reconstruction of $\mathbf{x}$ using inverse instruction $\bar{c}$), bidirectional prior loss (directional alignment applied to both forward and backward passes), and identity loss (reconstructing $\mathbf{x}$ as-is when $\mathbf{x}$ is both the input and condition to prevent discarding conditional information). Training requires only unpaired images and their source/target captions. For video editing, all losses are applied directly to video latents without architectural changes.

## Key Experimental Results

### Main Results — User Study on Video Editing

| Editing Direction | ByG Win Rate | Ditto Win Rate | Votes |
|-------------------|--------------|----------------|-------|
| Cartoon → Photo | **80.5% ± 2.9%** | 19.5% | 119 |
| Photo → Cartoon | **70.0% ± 5.4%** | 30.0% | 119 |
| OOD 3D-CGI | **85.0%** | 15.0% | — |
| Overall | **75.3% ± 2.2%** | 24.7% | 238 |

Ours (ByG) was trained on ~330 unpaired videos, whereas Ditto was trained on millions of paired videos. Binomial test $p < 3 \times 10^{-15}$.

### Main Results — Quantitative Video Metrics

| Method | CLIP dir ↑ | DINO Sim. ↑ | Motion Fid. ↑ | Dyn. Deg. ↑ | Aesthetic ↑ | Temp. Flicker ↑ |
|--------|-----------|------------|--------------|------------|------------|----------------|
| **Ours (ByG)** | **0.104** | **0.718** | **0.715** | **0.597** | 0.574 | 0.967 |
| Ditto | 0.091 | 0.536 | 0.616 | 0.560 | **0.585** | **0.972** |

### Main Results — Long-tail Style Editing (6 Unseen Styles: GTA V / Minecraft / US Comics / Low-poly / Voxel / Lego)

| Method | Style→Photo Semantic ↑ | Overall ↑ | Photo→Style Semantic ↑ | Overall ↑ |
|--------|----------------------|-----------|----------------------|-----------|
| **Ours (ByG)** | **7.67** | **8.30** | **5.22** | **6.33** |
| Kontext (Sup.) | 6.87 | 7.85 | 3.97 | 6.00 |
| Qwen-Image-Edit (Sup.) | 6.86 | 7.75 | 4.87 | 5.17 |
| FlowEdit (Zero-shot) | 4.27 | 6.20 | 1.46 | 3.64 |

### Ablation Study

| Configuration | Edit Success ↑ | Source Pres. ↑ | Description |
|---------------|---------------|---------------|-------------|
| Full Model | 8.317 | 7.617 | Best balance between editing and preservation |
| w/o Gradient Routing | 8.917 | 7.183 | Aggressive editing but worse source preservation |
| w/o Cycle Loss | 8.983 | 7.233 | Similar to above; loss of preservation constraint |
| w/o Directional Loss | 8.400 | 7.233 | MSE only leads to stronger source drift |
| w/o Bootstrapping | 5.517 | 7.050 | Distribution mismatch; both metrics drop |
| w/o Regularization | 0.633 | 9.767 | Collapses to an identity map |

## Highlights & Insights
- **Core Innovation**: The first flow matching editing framework that operates without paired data or external models. The three components (bootstrapping, directional prior, gradient routing) complementarily address the three fundamental challenges of unsupervised training.
- **Gradient Routing** is the technical highlight: Adapting STE to a continuous denoising setting allows using clean images for forward conditioning while passing gradients through noisy predictions, effectively eliminating the train-test gap.
- **Strong Generalization**: Achieved an 85% win rate on 3D-CGI styles unseen during training. It outperformed supervised baselines across six completely untrained styles.
- **Extreme Data Efficiency**: Surpassed Ditto (trained on millions of pairs) using only ~330 unpaired videos for video editing.

## Limitations & Future Work
- Inherits the knowledge boundaries and biases of the base model—editing fails if the T2I model does not understand the target domain.
- **Object removal** performance is weak (GEdit-Bench 1.91 vs. Kontext 6.94): Target captions only "omit" the object, lacking an explicit removal signal.
- **Text editing** is also weak (2.10 vs. 5.44), stemming from limitations in caption supervision.
- Bootstrapping + EMA increases training computation costs due to the need for multi-step sampling to generate pseudo-inputs.

## Related Work & Insights
- **CycleGAN** (Zhu et al., 2017): Classic source of cycle consistency, but limited to two-domain translation rather than open-instruction editing.
- **NP-Edit** (Kumari et al., 2025): Unpaired but relies on external VLM feedback and single-step distillation, making it hard to scale to multi-step models and videos.
- **DDS** (Hertz et al., 2023): Compares editing directions in pixel space; ByG instead extracts direction via dual-prompt queries in a single state.
- **STE** (Bengio et al., 2013): ByG’s gradient routing is derived from STE but adapted from discrete quantization to continuous denoising.

## Rating
- Novelty: 9/10 — The first to combine bootstrapping, directional priors, and STE gradient routing for unpaired flow matching editing.
- Experimental Thoroughness: 9/10 — Covers image and video modalities, long-tail and general benchmarks, with comprehensive user studies, automatic metrics, and ablations.
- Writing Quality: 9/10 — Clear logical chain with tight problem-solution mapping.
- Value: 8/10 — Significant reduction in data requirements for editing models, though a gap remains in scenarios where paired data excels (e.g., object removal/text editing).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Shifting the Breaking Point of Flow Matching for Multi-Instance Editing](shifting_the_breaking_point_of_flow_matching_for_multi-instance_editing.md)
- [\[ICML 2026\] Principled RL for Flow Matching Emerges from the Chunk-level Policy Optimization](principled_rl_for_flow_matching_emerges_from_the_chunk-level_policy_optimization.md)
- [\[ICML 2026\] (HB-ARFM) History-Bootstrapped Flow Matching for Inverse Boiling Reconstruction](hb-arfm_history-bootstrapped_flow_matching_for_inverse_boiling_reconstruction.md)
- [\[ICML 2026\] A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)
- [\[ICML 2026\] Stable Velocity: A Variance Perspective on Flow Matching](stable_velocity_a_variance_perspective_on_flow_matching.md)

</div>

<!-- RELATED:END -->
