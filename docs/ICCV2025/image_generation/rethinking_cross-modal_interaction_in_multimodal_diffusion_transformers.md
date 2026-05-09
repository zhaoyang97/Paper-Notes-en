---
title: >-
  [Paper Note] Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers
description: >-
  [ICCV 2025][Image Generation][MM-DiT] This paper identifies two structural issues in MM-DiT architectures (FLUX, SD3.5): the token count asymmetry between visual and text modalities suppresses cross-modal attention, and attention weights are insensitive to timestep. To address these, the authors propose TACA (Temperature-Adjusted Cross-modal Attention), which rebalances multimodal interaction via temperature scaling and timestep-adaptive adjustment. Combined with LoRA fine-tuning, TACA achieves significant improvements in text-image alignment on T2I-CompBench (spatial relations +16.4%, shape +5.9%) with negligible additional computational overhead.
tags:
  - ICCV 2025
  - Image Generation
  - MM-DiT
  - Cross-Attention Suppression
  - Temperature Scaling
  - FLUX
  - SD3.5
  - Text-Image Alignment
date: 2026-05-08
content_hash: afcd9c51cd7267dc
---

# Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers

**Conference**: ICCV 2025
**arXiv**: [2506.07986](https://arxiv.org/abs/2506.07986)
**Code**: [https://github.com/Vchitect/TACA](https://github.com/Vchitect/TACA)
**Area**: Image Generation / Diffusion Transformer / Text-Image Alignment
**Keywords**: MM-DiT, Cross-Attention Suppression, Temperature Scaling, FLUX, SD3.5, Text-Image Alignment

## TL;DR
This paper identifies two structural issues in MM-DiT architectures (FLUX, SD3.5): the token count asymmetry between visual and text modalities suppresses cross-modal attention, and attention weights are insensitive to timestep. To address these, the authors propose TACA (Temperature-Adjusted Cross-modal Attention), which rebalances multimodal interaction via temperature scaling and timestep-adaptive adjustment. Combined with LoRA fine-tuning, TACA achieves significant improvements in text-image alignment on T2I-CompBench (spatial relations +16.4%, shape +5.9%) with negligible additional computational overhead.

## Background & Motivation

### State of the Field
Multimodal Diffusion Transformers (MM-DiT) concatenate text and visual tokens into a unified sequence and apply full self-attention, forming the core architecture of state-of-the-art text-to-image models such as Stable Diffusion 3/3.5 and FLUX. Nevertheless, even the most advanced model, FLUX.1 Dev, still exhibits severe text-image misalignment, including missing objects and incorrect attribute binding.

### Two Core Problems

#### Problem 1: Suppression of Cross-Modal Attention
In the unified softmax of MM-DiT, the number of visual tokens greatly exceeds that of text tokens (e.g., $N_{vis}/N_{txt} = 4096/512 = 8$ when FLUX generates 1024×1024 images), causing the attention probability from visual tokens to text tokens to be severely suppressed:

$$P_{\text{vis-txt}}^{(i,j)} = \frac{e^{s_{ij}^{vt}/\tau}}{\sum_{k=1}^{N_{txt}} e^{s_{ik}^{vt}/\tau} + \sum_{k=1}^{N_{vis}} e^{s_{ik}^{vv}/\tau}} \approx \frac{e^{s_{ij}^{vt}/\tau}}{\sum_{k=1}^{N_{vis}} e^{s_{ik}^{vv}/\tau}}$$

The denominator is dominated by the large number of visual-visual interaction terms, effectively drowning out the text guidance signal. This stands in sharp contrast to conventional cross-attention, where the denominator contains only text tokens.

#### Problem 2: Timestep-Insensitive QK Weights
During early denoising steps (large $t$), strong text guidance is needed to establish the global layout; in later steps (small $t$), the focus shifts to visual details. However, the projection matrices $W^Q$ and $W^K$ in MM-DiT are shared across all timesteps and cannot dynamically adjust cross-modal attention strength according to the denoising stage — the initial steps do not assign relatively greater weight to $s_{ik}^{vt}$ over $s_{ik}^{vv}$.

### Key Observation
Through visualization of the stepwise prediction of $x_0$ during denoising (Fig. 3), the authors find that the overall composition of the image is determined within the first few steps. If the initial layout is misaligned with the text, subsequent steps cannot correct it.

## Method

### Overall Architecture
TACA introduces minimal modifications to the attention mechanism of MM-DiT: (1) modality-specific temperature scaling to amplify cross-modal attention; (2) timestep-dependent piecewise adjustment; and (3) LoRA fine-tuning to eliminate artifacts introduced by amplification.

### Key Design 1: Modality-Specific Temperature Scaling
A temperature coefficient $\gamma > 1$ is introduced to amplify the logits of visual-text interactions:

$$P_{\text{vis-txt}}^{(i,j)} = \frac{e^{{\color{blue}\gamma} s_{ij}^{vt}/\tau}}{\sum_{k=1}^{N_{txt}} e^{{\color{blue}\gamma} s_{ik}^{vt}/\tau} + \sum_{k=1}^{N_{vis}} e^{s_{ik}^{vv}/\tau}}$$

$\gamma$ acts as a "signal amplifier," boosting the relative weight of cross-modal interactions within the softmax competition. Visualizations show that as $\gamma$ increases, visual features corresponding to text descriptions such as "brown backpack" and "glass mirror" become progressively more salient.

### Key Design 2: Timestep-Dependent Piecewise Adjustment

$$\gamma(t) = \begin{cases} \gamma_0 & t \geq t_{\text{thresh}} \\ 1 & t < t_{\text{thresh}} \end{cases}$$

where $t_{\text{thresh}} = 970$ (the first 10% of 1000 denoising steps). This aligns with denoising dynamics: temperature scaling is applied during the early stage (large $t$) when strong text guidance is needed to establish composition, and normal attention is restored during the later stage (small $t$) when the focus shifts to visual details.

### LoRA Training to Eliminate Artifacts
Amplifying the cross-modal attention logits alters the denoising output distribution and may introduce distorted boundaries or inconsistent textures. LoRA adapters applied to the attention layers are used to recover the true image distribution:

$$W' = W + \alpha \cdot BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$$

Training samples timesteps $t \geq t_{\text{thresh}} = 970$, focusing on the initial denoising stage where semantic information is most prominent. The training loss is:

$$\mathcal{L} = \mathbb{E}_{x_0, t \geq t_{\text{thresh}}} \left[\|v(x_t, t) - v_\theta(x_t, t, \mathcal{P}_{\text{txt}}, \gamma(t))\|_2^2\right]$$

### Implementation Highlights
- **Zero new parameters**: Temperature scaling requires only element-wise multiplication during attention computation.
- **Minimal implementation**: The core code modification is extremely compact.
- **Two hyperparameters**: $\gamma_0$ (base temperature) and $t_{\text{thresh}}$ (timestep threshold), determinable via simple ablation.

## Experiments

### Experimental Setup
- **Models**: FLUX.1-Dev, SD3.5-Medium
- **Training data**: 10K LAION image-text pairs (with LLaVA-enhanced captions)
- **Evaluation**: T2I-CompBench (attribute binding, spatial relations, complex prompts)
- **LoRA configuration**: $(r, \alpha) = (16, 16)$ and $(64, 64)$
- **Hyperparameters**: $\gamma_0 = 1.2$, $t_{\text{thresh}} = 970$

### Main Results: T2I-CompBench Alignment Evaluation

| Model | Color↑ | Shape↑ | Texture↑ | Spatial↑ | Non-Spatial↑ | Complex↑ |
|------|--------|--------|----------|----------|-------------|----------|
| FLUX.1-Dev | 0.7678 | 0.5064 | 0.6756 | 0.2066 | 0.3035 | 0.4359 |
| **+ TACA (r=64)** | **0.7843** | **0.5362** | **0.6872** | **0.2405** | 0.3041 | **0.4494** |
| SD3.5-Medium | 0.7890 | 0.5770 | 0.7328 | 0.2087 | 0.3104 | 0.4441 |
| **+ TACA (r=64)** | **0.8074** | **0.5938** | **0.7522** | **0.2678** | 0.3106 | **0.4470** |

On FLUX.1-Dev: spatial relations **+16.4%** (0.2066→0.2405), shape **+5.9%** (0.5064→0.5362).
On SD3.5-Medium: spatial relations **+28.3%** (0.2087→0.2678), shape **+2.9%**.

### Ablation Study: Effect of Temperature Coefficient $\gamma_0$

| $\gamma_0$ | Color↑ | Shape↑ | Spatial↑ |
|------------|--------|--------|----------|
| 1.0 (no scaling) | 0.7678 | 0.5064 | 0.2066 |
| 1.1 | ~0.775 | ~0.52 | ~0.22 |
| **1.2** | **0.7843** | **0.5362** | **0.2405** |
| 1.3 | ~0.78 | ~0.53 | ~0.24 |

$\gamma_0 = 1.2$ achieves the best balance between text-image alignment and image quality. Excessively large $\gamma$ introduces artifacts.

### Ablation Study: Role of Timestep Threshold and LoRA
- Applying temperature scaling only to the initial 10% of denoising steps captures the majority of the benefit.
- Applying TACA without LoRA introduces local artifacts.
- LoRA + TACA eliminates artifacts while preserving alignment improvements.

### Attention Map Visualization
Comparisons show that TACA substantially amplifies the attention from visual tokens to text tokens during the initial denoising steps, enabling image regions to better attend to their corresponding textual descriptions.

## Highlights & Insights
- **Precise problem diagnosis**: The two root causes of text-image misalignment in MM-DiT are accurately identified — cross-attention suppression due to token count asymmetry, and timestep insensitivity.
- **Minimal and efficient solution**: The core intervention is a single temperature coefficient $\gamma(t)$, requiring no new parameters and implementable in a few lines of code.
- **Cross-architecture effectiveness**: The method is effective on both FLUX and SD3.5, two distinct MM-DiT implementations.
- **Mechanistic insight**: The work reveals an inherent limitation of full self-attention (compared to conventional cross-attention) in multimodal settings.
- **Denoising dynamics analysis**: The observation that compositional layout is established in the initial steps provides important design guidance for future work.

## Limitations & Future Work
- LoRA fine-tuning still requires training data (10K image-text pairs) and GPU resources.
- The piecewise constant $\gamma(t)$ may not be the optimal timestep-adaptive strategy; a smooth continuous function could potentially perform better.
- Evaluation is limited to T2I-CompBench; human preference studies and generation quality metrics such as FID/IS are absent.
- Temperature scaling only addresses the magnitude of cross-modal attention and does not resolve the directionality of attention allocation.
- The quality of training captions (generated by LLaVA) may affect LoRA fine-tuning performance.

## Related Work & Insights
- **Diffusion Transformer architectures**: DiT (adaLN conditioning), CrossDiT/PixArt-α (cross-attention), MM-DiT (unified full attention)
- **Text-image alignment improvements**: CLIP-guided optimization, cross-attention control (Attend-and-Excite), layout planning modules, feedback-driven optimization
- **MM-DiT models**: FLUX.1, SD3/3.5, CogVideo, HunyuanVideo

## Rating
- Novelty: ⭐⭐⭐⭐ — The analysis of MM-DiT attention mechanisms is original and insightful.
- Technical Depth: ⭐⭐⭐⭐ — Mathematical derivations are clear and the root cause analysis is convincing.
- Experimental Thoroughness: ⭐⭐⭐ — Quantitative evaluation relies on a single benchmark (T2I-CompBench only).
- Value: ⭐⭐⭐⭐⭐ — Directly applicable to existing MM-DiT models at minimal cost.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing](exploring_multimodal_diffusion_transformers_for_enhanced_prompt-based_image_edit.md)
- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[ICCV 2025\] IRGPT: Understanding Real-world Infrared Image with Bi-cross-modal Curriculum on Large-scale Benchmark](irgpt_understanding_real-world_infrared_image_with_bi-cross-modal_curriculum_on_.md)
- [\[ICCV 2025\] End-to-End Multi-Modal Diffusion Mamba](end-to-end_multi-modal_diffusion_mamba.md)
- [\[CVPR 2026\] CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration](../../CVPR2026/image_generation/ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)

<!-- RELATED:END -->
