---
title: >-
  [Paper Note] Copyright Infringement Detection in Text-to-Image Diffusion Models via Differential Privacy
description: >-
  [AAAI2026][Image Generation][copyright infringement detection] This paper formalizes copyright infringement from the perspective of Differential Privacy (DP), and proposes the D-Plus-Minus (DPM) framework. By fine-tuning diffusion models in two opposing directions—"learning" and "unlearning"—DPM measures conditional sensitivity differences to perform post-hoc detection of copyright infringement in text-to-image models.
tags:
  - AAAI2026
  - Image Generation
  - copyright infringement detection
  - differential privacy
  - diffusion models
  - machine unlearning
  - text-to-image generation
date: 2026-05-08
content_hash: a176ff205a746e01
---

# Copyright Infringement Detection in Text-to-Image Diffusion Models via Differential Privacy

**Conference**: AAAI2026
**arXiv**: [2509.23022](https://arxiv.org/abs/2509.23022)
**Code**: [Project Page](https://leo-xfm.github.io/pubs/dpm)
**Area**: Image Generation
**Keywords**: copyright infringement detection, differential privacy, diffusion models, machine unlearning, text-to-image generation

## TL;DR

This paper formalizes copyright infringement from the perspective of Differential Privacy (DP), and proposes the D-Plus-Minus (DPM) framework. By fine-tuning diffusion models in two opposing directions—"learning" and "unlearning"—DPM measures conditional sensitivity differences to perform post-hoc detection of copyright infringement in text-to-image models.

## Background & Motivation

- Large-scale visual generative models such as Stable Diffusion have been found to **memorize and reproduce** copyrighted content from training data, raising serious legal and ethical concerns.
- Existing detection methods suffer from notable limitations:
    - Model-level frameworks such as **CopyScope** can only quantify the overall degree of infringement and cannot localize specific infringing concepts or samples, making them insufficient for legal evidence.
    - **Prompt-based query** methods rely on crafting specific prompts to elicit infringing outputs, and are susceptible to model updates and sampling stochasticity, lacking robustness and theoretical interpretability.
    - **DIAGNOSIS** requires pre-processing protected datasets with "coating," necessitating access to the original training data.
- With the emergence of AI regulatory frameworks worldwide (e.g., the EU AI Act), there is an urgent need for a post-hoc copyright detection method that **requires neither original training data nor input prompts, while providing theoretical guarantees**.

## Core Problem

How to determine, under realistic white-box model access conditions where training data and corresponding prompts are unavailable, whether a specific image or visual concept has been memorized by a diffusion model (i.e., constitutes copyright infringement), while providing quantifiable and interpretable detection results?

## Method

### 1. Theoretical Foundation: Formalizing Copyright Infringement via Differential Privacy

The paper reinterprets copyright infringement as a **violation of conditional differential privacy**:

- **Conditional Publicity**: When the presence or absence of a copyrighted concept in the training data causes significant changes in model output ($\varepsilon > 200$), it indicates that the model heavily depends on that concept.
- **Copyright Infringement Definition**: If model $G$, trained on dataset $D$ containing copyrighted sample $x_c$, produces output distributions that differ significantly from those of a model trained on $D'$ (which excludes $x_c$) in response to prompts semantically related to $x_c$, infringement is determined.
- **Non-Infringement Definition**: If the output distribution is insensitive to whether $x$ is included in the training data (i.e., the distribution remains invariant), it is considered non-infringing.

### 2. Conditional Sensitivity Metric

Analogous to local sensitivity in differential privacy, the paper defines Conditional Sensitivity $CS(M, \hat{x}_i)$ to quantify the dependence of model query function $M$ on a specific training sample $\hat{x}_i$:

$$CS(M, \hat{x}_i) = \max_{D, D': D \triangle D' \leq \{\hat{x}_i\}} |M(D) - M(D')|$$

### 3. D-Plus-Minus (DPM) Detection Framework

The detection pipeline consists of the following steps:

**(a) Preprocessing: Concept Extraction and Image Collection**

- Extract the core concept from target image $\hat{x}_i$.
- Construct a semantic neighborhood set $U(\hat{x}_i)$ by collecting images semantically similar to that concept.
- Specify a generic prompt (format: `a photo of [V] [class]`).

**(b) Branch Training**

Two fine-tuning branches are trained simultaneously in opposing directions:

- **Learning branch $G_{D+}$**: Encourages the model to memorize the target concept ($I = +1$).
- **Unlearning branch $G_{D-}$**: Trains the model to forget the target concept ($I = -1$).

The training objective is the standard diffusion model loss multiplied by the branch indicator $I$.

**(c) Conditional Sensitivity Measurement**

A CLIP image encoder is used as the query function. Cosine similarity is computed to compare the outputs of fine-tuned models against the original model under the same prompt. Models at multiple training steps are sampled for measurement.

**(d) Statistical Analysis: Orthogonal Distribution Calibration**

Fine-tuning inevitably affects model outputs for unrelated content. DPM addresses this by generating orthogonal images (content unrelated to the target concept) to construct a reference distribution, which is used to normalize the conditional sensitivity:

$$\hat{CS}(M, \hat{x}_i, D^*) = \frac{CS(M, x_i, D^*)}{\overline{CS(M, X_{\text{orth}}, D^*)}}$$

**(e) Branch Merging and Final Scoring**

The contrastive sensitivity difference $\Delta\hat{CS}$ between the two branches is computed, then min-max normalized and mapped through a Sigmoid function to yield the final DPM score in $[0, 1]$. A higher score indicates a higher likelihood of copyright infringement.

### 4. Copyright Infringement Detection Dataset (CIDD)

The paper introduces the CIDD dataset, which includes:

- Three high-risk categories: **Human Face**, **Architecture**, and **Arts Painting**.
- A total of 429 concepts and 2,397 images.
- Binary infringement/non-infringement labels per concept, paired with 3–6 neighborhood images.
- A proposed four-level hierarchical taxonomy of copyright infringement: Technical Layer → Content Layer → Structural/Style Layer → Semantic Layer.

## Key Experimental Results

Detection results across four models (weighted average AUC / SoftAcc):

| Model | AUC | SoftAcc |
|-------|-----|---------|
| SD1.4 | 0.858 | 0.764 |
| SDXL-1.0 | 0.817 | 0.752 |
| SANA-0.6B | 0.840 | 0.757 |
| FLUX.1 | 0.812 | 0.725 |

- Weighted average AUC exceeds **80%** and SoftAcc exceeds **72%** across all models.
- The Architecture category achieves the best performance on most models (except SD1.4), reaching AUC of 0.9256 on SDXL and 0.9500 on FLUX.
- Ablation studies show: dual-branch merging outperforms single-branch; multi-timestep measurement improves and stabilizes detection performance; image degradation has minimal impact on detection.

## Highlights & Insights

1. **Theoretical Innovation**: This is the first work to introduce differential privacy theory into copyright infringement detection, formalizing infringement as a violation of conditional DP and providing a rigorous mathematical foundation.
2. **Dual-Branch Contrastive Design**: By fine-tuning in opposing learning/unlearning directions to simulate the "inclusion" and "exclusion" of training data, the framework elegantly circumvents the difficulty of lacking direct access to original training data.
3. **Statistical Calibration Mechanism**: Orthogonal prompt distributions are used to eliminate global parameter drift introduced by fine-tuning, enabling comparable sensitivity scores across different samples.
4. **Practical Detection Setting**: White-box model access without requiring original training data or corresponding prompts aligns closely with real-world scenarios.
5. **Hierarchical Infringement Taxonomy**: The four-level classification from technical to semantic layers provides a systematic framework for copyright infringement research.

## Limitations & Future Work

1. **High Computational Cost**: Each candidate sample requires two rounds of fine-tuning (learning + unlearning), limiting efficiency in large-scale detection scenarios.
2. **Restricted to Text-to-Image Diffusion Models**: The approach has not yet been extended to copyright detection for LLMs or LVLMs, which the paper acknowledges as future work.
3. **Dependence on CLIP Model Capability**: Detection performance is bounded by the generalization ability of the CLIP image encoder, resulting in notable performance variation across categories.
4. **Limited Dataset Scale**: CIDD covers only three categories and 429 concepts, which may be insufficient to represent all real-world copyright infringement scenarios.
5. **Requires White-Box Access**: The method cannot be directly applied to closed-source commercial models, limiting its practical deployment scope.
6. **Semantic-Layer Infringement Not Covered**: The highest level of the taxonomy—the Semantic Layer (e.g., plot, theme)—is not included in CIDD and remains the most challenging category to detect.

## Related Work & Insights

| Method | Requires Training Data | Requires Prompt | Theoretical Guarantee | Localizes Specific Concepts | Detection Granularity |
|--------|:---:|:---:|:---:|:---:|------|
| CopyScope | ✗ | ✓ | ✗ | ✗ | Model-level |
| DIAGNOSIS | ✓ (coated) | ✓ | Partial | ✓ | Dataset-level |
| Prompt Engineering Methods | ✗ | ✓ | ✗ | Partial | Sample-level |
| **DPM (Ours)** | **✗** | **✗** | **✓** | **✓** | **Concept-level** |

DPM is the only detection framework that requires neither original training data nor corresponding prompts, while providing theoretical guarantees grounded in differential privacy.

The intersection of **differential privacy and copyright protection** offers an inspiring perspective; migrating the quantification paradigm of privacy leakage to copyright infringement detection is an elegant analogy. The dual-branch learning/unlearning design is closely related to the **machine unlearning** literature, and more advanced unlearning algorithms could be incorporated to improve efficiency. The orthogonal prompt calibration strategy resembles a **control group design** in experimental settings and can be adapted for other detection tasks requiring the elimination of confounding factors. The approach shares methodological common ground with **membership inference attacks**, but DPM offers a stronger theoretical framework and a more practical detection pipeline.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Formalizing copyright infringement via differential privacy combined with dual-branch contrastive detection demonstrates originality in both theory and methodology.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Four models, three categories, and complete ablation studies; however, direct quantitative comparisons with existing methods are lacking.)
- Writing Quality: ⭐⭐⭐⭐ (Theoretical derivations are clear and the notation system is complete, though the correspondence between certain definitions and their practical implementations could be made more intuitive.)
- Value: ⭐⭐⭐⭐⭐ (AI copyright issues are highly important and timely; both the theoretical contributions and the dataset construction carry substantial practical value.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration](../../CVPR2026/image_generation/ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)
- [\[CVPR 2026\] BlackMirror: Black-Box Backdoor Detection for Text-to-Image Models via Instruction-Response Deviation](../../CVPR2026/image_generation/blackmirror_black-box_backdoor_detection_for_text-to-image_models_via_instructio.md)
- [\[CVPR 2026\] TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models](../../CVPR2026/image_generation/tina_text-free_inversion_attack_for_unlearned_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](../../ICCV2025/image_generation/compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [\[ICLR 2026\] Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](../../ICLR2026/image_generation/continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)

</div>

<!-- RELATED:END -->
