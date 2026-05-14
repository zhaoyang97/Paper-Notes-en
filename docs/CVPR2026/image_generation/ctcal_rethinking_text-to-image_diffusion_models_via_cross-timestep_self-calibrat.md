---
title: >-
  [Paper Note] CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration
description: >-
  [CVPR 2026][Image Generation][Text-to-Image Generation] This paper proposes CTCal (Cross-Timestep Self-Calibration), which leverages reliable text-image alignments (cross-attention maps) formed at small timesteps (low no…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Text-to-Image Generation"
  - "Diffusion Models"
  - "Cross-Attention Alignment"
  - "Self-Calibration"
  - "Compositional Generation"
date: 2026-05-08
content_hash: 55edff0ffd1e6afb
---

# CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration

**Conference**: CVPR 2026
**arXiv**: [2603.20741](https://arxiv.org/abs/2603.20741)
**Code**: [https://github.com/xiefan-guo/ctcal](https://github.com/xiefan-guo/ctcal)
**Area**: Image Generation / Text-to-Image Diffusion Models
**Keywords**: Text-to-Image Generation, Diffusion Models, Cross-Attention Alignment, Self-Calibration, Compositional Generation

## TL;DR
This paper proposes CTCal (Cross-Timestep Self-Calibration), which leverages reliable text-image alignments (cross-attention maps) formed at small timesteps (low noise) to calibrate representation learning at large timesteps (high noise), providing explicit cross-timestep self-supervision for text-to-image generation. CTCal comprehensively outperforms existing methods on T2I-CompBench++ and GenEval.

## Background & Motivation

**Background**: Diffusion models dominate text-to-image generation, yet precise text-image alignment—especially for compositional generation with complex prompts—remains an open challenge.

**Limitations of Prior Work**: (1) Conventional diffusion losses provide only implicit supervision, making it difficult to capture fine-grained text-image correspondences; (2) inference-time optimization methods (e.g., Attend-and-Excite) exhibit poor generalization and do not scale; (3) at large timesteps, severe noise degrades cross-attention map quality, preventing correct alignment—a critical bottleneck for generation quality.

**Key Challenge**: At small timesteps, text-image alignment is reliable but "too easy"; at large timesteps, alignment is poor but "critical" (determining generation quality in the early stages of inference).

**Goal**: How to provide explicit supervision for establishing correct text-image correspondences at large timesteps in diffusion models?

**Key Insight**: **Key Observation**—for the same image-text-noise triplet, cross-attention maps extracted at different timesteps during training differ dramatically in quality: maps at small timesteps closely match the true image structure and semantics, while those at large timesteps are entirely degraded.

**Core Idea**: Use reliable attention maps from small timesteps as a "teacher" to calibrate attention maps at large timesteps (the "student"), enabling the model to supervise itself.

## Method

### Overall Architecture
Given an image, text, and noise, two timesteps are sampled such that $t_{\text{tea}} < t_{\text{stu}}$. Forward passes are performed separately to obtain cross-attention maps $\mathbf{A}_{\text{tea}}$ and $\mathbf{A}_{\text{stu}}$. The latter is calibrated using $\mathbf{A}_{\text{tea}}$ as the target, with only the network parameters corresponding to $t_{\text{stu}}$ being optimized.

### Key Designs

1. **Part-of-Speech-based Cross-Attention Map Selection**:

    - **Function**: Only attention maps of noun tokens are extracted for the CTCal loss; tokens without spatial semantics—such as articles ("the") and conjunctions ("and")—are ignored.
    - **Mechanism**: POS tagging is performed using Stanza; $\mathcal{Y}_{\text{noun}}$ denotes the set of nouns, and the loss is defined as $\mathcal{L}_{\text{CTCal}} = \frac{1}{N_{\text{noun}}} \sum_{\mathbf{y}_i \in \mathcal{Y}_{\text{noun}}} \mathcal{D}(\mathbf{A}_{\text{stu},\mathbf{y}_i}, \mathbf{A}_{\text{tea},\mathbf{y}_i})$
    - **Design Motivation**: Ablation studies show that applying constraints to all tokens actually degrades performance, since attention maps of non-noun tokens carry no meaningful spatial semantic information and introduce noise.

2. **Pixel-Semantic Space Joint Optimization**:

    - **Function**: Attention maps are aligned simultaneously in pixel space and semantic space. A lightweight autoencoder $(f_{\text{attn}}^{\text{enc}}, f_{\text{attn}}^{\text{dec}})$ is introduced, with a reconstruction auxiliary task to prevent mode collapse.
    - **Core Formula**:
    $$\mathcal{L}_{\text{CTCal}} = \lambda_1 \underbrace{\mathcal{D}(\mathbf{A}_{\text{stu}}, \mathbf{A}_{\text{tea}})}_{\text{Pixel}} + \lambda_2 \underbrace{\mathcal{D}(f^{\text{enc}}(\mathbf{A}_{\text{stu}}), f^{\text{enc}}(\mathbf{A}_{\text{tea}}))}_{\text{Semantic}} + \lambda_3 \underbrace{\mathcal{D}(f^{\text{dec}}(\mathbf{A}_{\text{tea}}), \mathbf{A}_{\text{tea}})}_{\text{Reconstruction}}$$
    - **Design Motivation**: Pixel-level alignment captures spatial location information; semantic-level alignment captures high-level semantic consistency; the reconstruction task prevents encoder degeneration.

3. **Subject Response Alignment Regularization**:

    - **Function**: Aligns the attention responses of all subjects (nouns) to that of the highest-responding subject:
    $$\mathcal{R}_{\text{subject}} = \frac{1}{N_{\text{noun}}} \sum \text{ReLU}(\mathcal{S}_{\text{attn}} - \max(\mathbf{A}_{\text{stu},\mathbf{y}_i}) - \tau)$$
    - **Design Motivation**: Prevents high-response subjects from suppressing low-response ones, which would cause the latter to fail to render correctly in the generated image (e.g., generating only a cat for the prompt "cat and dog").

4. **Timestep-aware Adaptive Weighting**:

    - **Function**: $\lambda_t = t_{\text{stu}} / T_{\text{train}}$; the CTCal weight is larger at large timesteps, while the diffusion loss dominates at small timesteps.
    - **Design Motivation**: At small timesteps, the diffusion loss alone is sufficient to establish alignment; explicit calibration from CTCal is only needed at large timesteps.

### Loss & Training
- $\mathcal{L} = \mathcal{L}_{\text{diffusion}} + \lambda_t \mathcal{L}_{\text{CTCal}}$
- LoRA is used to fine-tune the self-attention layers of the text encoder and the attention layers of the denoising network.
- For SD 2.1, $t_{\text{tea}}=0$ is set; for SD 3, $t_{\text{tea}}$ is selected according to the logit-normal sampling distribution.
- Dataset: High-quality text-image pairs are selected from generated data using a reward-driven approach.

## Key Experimental Results

### Main Results — T2I-CompBench++

| Method | Color↑ | Shape↑ | 2D-Spatial↑ | Numeracy↑ | Complex↑ |
|------|--------|--------|------------|----------|---------|
| SD 2.1 | 0.507 | 0.422 | 0.134 | 0.458 | 0.339 |
| SD 2.1 + AE | 0.640 | 0.452 | 0.146 | 0.477 | 0.340 |
| SD 2.1 + GORS | 0.643 | 0.486 | 0.178 | 0.486 | 0.337 |
| **SD 2.1 + CTCal** | **0.723** | **0.515** | **0.214** | **0.508** | **0.340** |
| SD 3 (2B) | 0.813 | 0.589 | 0.320 | 0.617 | 0.377 |
| **SD 3 + CTCal** | **0.844** | **0.597** | **0.348** | **0.629** | **0.381** |

### GenEval

| Method | Overall↑ | Two Object↑ | Counting↑ | Colors↑ | Position↑ |
|------|---------|-------------|----------|---------|-----------|
| SD 3 (2B) | 0.62 | 0.74 | 0.63 | 0.67 | 0.34 |
| **SD 3 + CTCal** | **0.69** | **0.85** | **0.70** | **0.79** | **0.38** |

### Ablation Study

| Configuration | Color↑ | 2D-Spatial↑ | Note |
|------|--------|------------|------|
| SD 2.1 + GORS baseline | 0.643 | 0.178 | — |
| + naive all-token constraint (a) | 0.629 (−2.2%) | 0.169 (−4.6%) | Performance drops! |
| + noun selection (b) | Significant gain | Significant gain | Noun selection is critical |
| + pixel + semantic (c) | Further gain | Further gain | Joint optimization is effective |
| + response alignment (d) | Further gain | Further gain | Subject balancing helps |
| + adaptive weighting (e) | **Best** | **Best** | Full CTCal |

### Key Findings
- Noun selection is the most critical design choice—indiscriminately aligning all tokens actually harms performance.
- CTCal is effective for both the cross-attention-based SD 2.1 and the MM-DiT-based SD 3, demonstrating generality.
- In user studies, CTCal achieves an overwhelming preference rate (SD 2.1: 76.67%, SD 3: 54.17%).

## Highlights & Insights
- **Training-stage perspective**: Unlike inference-time optimization methods, CTCal addresses text-image alignment at training time, yielding lasting improvements with no inference overhead.
- **Self-supervised paradigm**: The model uses its own small-timestep outputs to supervise large-timestep learning, requiring no additional labels or external teacher models.
- **Architecture-agnostic**: Applicable to both U-Net (SD 2.1) and Transformer (SD 3) architectures.

## Limitations & Future Work
- Training data construction relies on reward-driven sampling, incurring non-trivial data preparation costs.
- Only LoRA fine-tuning is explored; the effect of full fine-tuning remains uninvestigated.
- Noun selection depends on POS tagging tools; applicability to non-English prompts is unknown.

## Related Work & Insights
- Inference-time methods such as Attend-and-Excite motivated the focus on cross-attention, but CTCal addresses the problem more elegantly at the training stage.
- The reward-driven data selection strategy from GORS serves as the foundation for CTCal, which adds explicit alignment supervision on top of it.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The cross-timestep self-calibration idea is novel and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two benchmarks, SD 2.1 and SD 3, user studies, and complete ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from observation to design to validation is clear.
- Value: ⭐⭐⭐⭐⭐ Delivers substantial improvements in text-image alignment for text-to-image generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](solace_self_confidence_rewards_t2i.md)
- [\[CVPR 2026\] Guiding Diffusion Models with Semantically Degraded Conditions](guiding_diffusion_models_with_semantically_degraded_conditions.md)

</div>

<!-- RELATED:END -->
