---
title: >-
  [Paper Note] Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity
description: >-
  [CVPR 2026][Image Generation][color fidelity] To address the problem of T2I models generating images that appear "too vivid to be real," this work proposes the Color Fidelity Dataset (CFD, 1.3M images), the Color Fidelity Metric (CFM, based on Qwen2-VL + softrank loss), and Color Fidelity Refinement (CFR, a training-free spatiotemporal adaptive guidance modulation scheme), forming an integrated evaluation-and-improvement framework.
tags:
  - CVPR 2026
  - Image Generation
  - color fidelity
  - text-to-image evaluation
  - guidance scale
  - realistic generation
  - evaluation bias
date: 2026-05-08
content_hash: d5a5f8cf0918475d
---

# Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity

**Conference**: CVPR 2026
**arXiv**: [2603.10990](https://arxiv.org/abs/2603.10990)
**Code**: [GitHub](https://github.com/ZhengyaoFang/CFM)
**Area**: Image Generation
**Keywords**: color fidelity, text-to-image evaluation, guidance scale, realistic generation, evaluation bias

## TL;DR

To address the problem of T2I models generating images that appear "too vivid to be real," this work proposes the Color Fidelity Dataset (CFD, 1.3M images), the Color Fidelity Metric (CFM, based on Qwen2-VL + softrank loss), and Color Fidelity Refinement (CFR, a training-free spatiotemporal adaptive guidance modulation scheme), forming an integrated evaluation-and-improvement framework.

## Background & Motivation

**Over-saturation in T2I models**: Statistical analysis shows that virtually all T2I models produce outputs with significantly higher saturation and contrast than real photographs when prompted to generate "photorealistic" images — a phenomenon termed "too vivid to be real."

**Amplification of evaluation bias**: Existing metrics (PickScore, ImageReward, HPSv3, MPS) and human raters both favor vivid, high-contrast images. Controlled experiments demonstrate that these metrics systematically assign higher scores to over-saturated images, creating a positive feedback loop.

**Absence of color fidelity evaluation**: Current metrics measure semantic alignment or aesthetic preference, with no dedicated dimension for assessing whether the color distribution of generated images resembles that of real photography.

**The double-edged nature of CFG**: A larger classifier-free guidance scale $s$ improves text alignment but worsens color distortion (over-saturation and high contrast). This controllable relationship provides a tool for constructing ordered color fidelity datasets.

## Method

### Overall Architecture

CFD (dataset) → Training → CFM (evaluation metric) → Attention guidance → CFR (refinement module): a unified three-component pipeline.

### Key Design 1: Color Fidelity Dataset (CFD)

| Step | Operation | Scale |
|---|---|---|
| Real image collection | COCO/Open Images + CLIPIQA + Qwen2.5-VL (72B) filtering | 189,490 images |
| Automatic captioning | Qwen2.5-VL generates text descriptions | — |
| CFG-controlled synthesis | 11 T2I models × 12 categories, progressively increasing CFG scale to generate 6 distortion levels | 7 images per group |
| Data split | 160K groups for training / 30K groups for testing | ~1.33M images |

- 11 T2I models covered: SDXL, SD3, SD3.5, PixArt-Σ, Kolors, CogView4, Hunyuan-DiT, Flux-dev, Qwen-Image, Playground-v2.5, SRPO
- 12 semantic categories covering people, natural scenes, urban environments, etc.
- Human-annotated subset CFD-Human: 6,690 images × 3 annotators, Spearman consistency > 0.85

### Key Design 2: Color Fidelity Metric (CFM)

**Architecture**: Based on Qwen2-VL, image and text are jointly encoded into a unified sequence:

$$\mathbf{F} = [\mathbf{f}_1^v, \ldots, \mathbf{f}_M^v, \mathbf{f}_1^t, \ldots, \mathbf{f}_N^t] \in \mathbb{R}^{(M+N) \times d}$$

An MLP head outputs token-level logits, and the scalar score $S_{\text{CFM}}$ is extracted from the `<|Reward|>` token.

**Training objective: Differentiable Softrank Loss**

For each group of $K$ images (1 real + $K-1$ levels of decreasing fidelity), pairwise probabilities and soft ranks are computed:

$$P_{ij} = \sigma\left(\frac{r_j - r_i}{\tau}\right), \quad \hat{R}_i = 1 + \sum_{j=0}^{K-1} P_{ij}$$

$$\mathcal{L} = \frac{1}{K} \sum_{i=0}^{K-1} (\hat{R}_i - R_i)^2$$

- Ground-truth ranks $R = [1, 2, \ldots, K]$, with the real image ranked first
- Compared to pairwise loss, softrank achieves +7.4% accuracy on SynPairs and +6.6 Spearman correlation

**Role of text conditioning**: Removing the text branch leads to −6.5% accuracy on SynPairs and −6.0 Spearman correlation, demonstrating that text provides semantic context for judging whether the colors in a given scene are natural.

### Key Design 3: Color Fidelity Refinement (CFR)

CFR leverages CFM attention maps to localize color-distorted regions and adaptively reduces the guidance scale in those regions during diffusion denoising.

**Attention extraction**:

$$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{F}^t (\mathbf{F}^v)^\top}{\kappa}\right) \in \mathbb{R}^{N \times M}$$

Averaging over the text token dimension yields a pixel-level attention map $\mathbf{a}'$.

**Spatiotemporal guidance modulation**:

$$s_t(u,v) = s_0 \left[1 - \lambda \alpha(t) \mathbf{a}'(u,v)\right]$$

- $\lambda \in [0,1]$: modulation strength; $\alpha(t) = 1 - t/T$: temporal decay factor
- Guidance is automatically reduced in high-attention regions (where color deviation is large), suppressing over-saturation
- Entirely training-free and plug-and-play, requiring no modification to model parameters

## Key Experimental Results

### CFM Color Fidelity Discrimination Accuracy

| Method | CFD-SynPairs | CFD-Real&Syn |
|---|---|---|
| MUSIQ | 53.4% | 21.5% |
| ImageReward | 44.3% | 42.7% |
| PickScore | 51.4% | 48.5% |
| HPSv3 | 57.5% | 58.3% |
| **CFM (Ours)** | **83.6%** | **80.1%** |

### Correlation with Human Judgment (CFD-Human)

| Metric | Spearman | Pearson | Kendall |
|---|---|---|---|
| ImageReward | 62.8 | 63.5 | 49.2 |
| HPSv3 | 74.4 | 76.0 | 62.8 |
| **CFM (Ours)** | **84.9** | **85.4** | **71.4** |

### CFR Color Improvement Results

| Model | Setting | FID↓ | CLIPScore↑ | ΔSat.↓ | CFM↑ |
|---|---|---|---|---|---|
| SD3.5 | Baseline | 13.3 | 28.2 | 0.15 | 4.9 |
| SD3.5 | CFR_HPSv3 | 13.2 | 28.1 | 0.11 | 5.6 |
| SD3.5 | **CFR_CFM** | **13.1** | **28.2** | **0.07** | **6.9** |
| PixArt-Σ | Baseline | 16.5 | 27.2 | 0.09 | 4.4 |
| PixArt-Σ | **CFR_CFM** | **16.4** | **27.5** | **0.02** | **6.4** |
| Hunyuan | Baseline | 22.1 | 27.5 | 0.14 | 0.8 |
| Hunyuan | **CFR_CFM** | **19.9** | **27.5** | **0.03** | **2.1** |

- Saturation deviation reduced by 0.08–0.11; CFM score improved by 1.3–2.0
- FID/CLIPScore remain largely unchanged — color improvement does not sacrifice quality or semantic alignment

### Ablation Study: Spatiotemporal Modulation in CFR

| Setting | FID | CLIPScore | ΔSat. | CFM |
|---|---|---|---|---|
| Baseline | 13.3 | 28.2 | 0.15 | 4.9 |
| Temporal only | 18.0 | 25.9 | 0.18 | -1.3 |
| Spatial only | 13.2 | 28.2 | 0.12 | 6.8 |
| **Full spatiotemporal** | **13.0** | **28.2** | **0.07** | **6.9** |

- Temporal modulation alone is detrimental (FID +4.7, CFM turns negative) — globally decaying CFG strength disrupts semantics
- Spatial modulation is the core contributor; temporal decay serves a stabilizing complementary role

### Ablation Study: CFM

| Variant | CFD-SynPairs | CFD-Real&Syn | Spearman | Kendall |
|---|---|---|---|---|
| Pairwise loss | 76.2% | 74.8% | 78.3 | 66.1 |
| Visual-only | 77.1% | 74.3% | 78.9 | 67.0 |
| **Ours (softrank + multimodal)** | **83.6%** | **80.1%** | **84.9** | **71.4** |

## Highlights & Insights

| Dimension | Assessment |
|---|---|
| Problem definition | ⭐⭐⭐⭐⭐ Precisely defines "color fidelity" and quantifies evaluation bias, filling a notable gap |
| Dataset quality | ⭐⭐⭐⭐⭐ 1.3M images × 11 models × 12 categories × human-validated subset |
| Evaluation metric | ⭐⭐⭐⭐ Softrank loss + multimodal encoding, high human correlation |
| Refinement method | ⭐⭐⭐⭐ Training-free and plug-and-play, strong practical utility |
| Experimental design | ⭐⭐⭐⭐ Multi-model evaluation + human annotation + comprehensive ablation |
| Limitations | ⭐⭐⭐ Assumes color distortion primarily stems from CFG; CFR cannot handle non-CFG models; color fidelity is only one dimension of "photorealism" |

## Limitations & Future Work

1. **Evaluation bias deserves attention**: Current "human preference alignment" may be steering T2I models away from photorealism — the systematic preference of preference-trained metrics for over-saturated images constitutes a subtle but persistent bias.
2. **Using CFG to construct datasets**: Exploiting the monotonic relationship between guidance scale and color distortion to build ordered annotations is a methodologically elegant way to avoid costly human labeling.
3. **Softrank > Pairwise**: Continuous modeling of ordinal relationships outperforms discrete contrastive learning, and is well-suited for perceptual quality tasks with natural rank structure.
4. **Spatially adaptive guidance**: Generalizing the global CFG scale to a pixel-wise adaptive guidance field $s_t(u,v)$ is a broadly applicable technique transferable to other guidance modulation scenarios.

## Related Work & Insights

| Dimension | PickScore / ImageReward / HPSv3 / MPS | IQA methods (MUSIQ / CLIPIQA) | **CFM (Ours)** |
|---|---|---|---|
| Objective | Semantic alignment / aesthetic preference | Real-image quality degradation | Color fidelity |
| Training data | Human preference comparison pairs | Real-image distortion annotations | CFG-controlled ordered color sequences |
| On over-saturated images | Systematic preference (assigns high scores) | Near random | Correctly penalizes |
| Human correlation | Spearman 62.8–74.4 | — | **84.9** |
| Usable for refinement | Weak (attention unrelated to color) | No | Strong (CFR directly leverages attention) |

**Additional insights**:

1. **Evaluation–improvement closed loop**: CFM not only outputs a score but its internal attention maps are directly reused by CFR as improvement signals — the design paradigm of "the evaluation metric itself as a refinement tool" is worth adopting more broadly.
2. **Using the generation process to construct annotations**: The monotonicity of CFG scale automatically yields ordered annotations, avoiding large-scale human labeling — equally applicable to other perceptual dimensions that are difficult to annotate (texture realism, lighting consistency, etc.).
3. **Pixel-level adaptive guidance modulation**: Generalizing global scalar CFG to a spatial field $s_t(u,v)$ is transferable to local style control and region-specific detail enhancement.
4. **Generality of softrank loss**: Applicable to any ordered perceptual annotation task (image quality ranking, similarity ranking), offering greater stability and efficiency than pairwise approaches.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Color fidelity is systematically defined and quantified for the first time; the CFG-based ordered dataset construction is methodologically clever
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 11-model benchmark + human annotation validation + CFR experiments on 3 models + comprehensive ablation
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, well-motivated problem statement, rich figures and tables
- **Value**: ⭐⭐⭐⭐ Fills a gap in T2I evaluation; training-free refinement solution has strong practical utility; limited in scope to the "color" dimension only

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Leveraging Multispectral Sensors for Color Correction in Mobile Cameras](leveraging_multispectral_sensors_for_color_correction_in_mobile_cameras.md)
- [\[CVPR 2026\] OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution](oars_process-aware_online_alignment_for_generative_real-world_image_super-resolu.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[CVPR 2026\] PROMO: Promptable Outfitting for Efficient High-Fidelity Virtual Try-On](promo_promptable_virtual_tryon_efficient.md)
- [\[CVPR 2026\] SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images](simlbr_learning_to_detect_fake_images_by_learning_to_detect_real_images.md)

</div>

<!-- RELATED:END -->
