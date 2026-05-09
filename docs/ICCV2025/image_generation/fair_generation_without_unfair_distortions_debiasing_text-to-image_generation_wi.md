---
title: >-
  [Paper Note] Fair Generation without Unfair Distortions: Debiasing Text-to-Image Generation with Entanglement-Free Attention
description: >-
  [ICCV 2025][Image Generation][Debiasing] This paper proposes Entanglement-Free Attention (EFA), a training-free inference-time debiasing method that injects target attributes (e.g., gender, race) into person regions by modifying the cross-attention mechanism, while preserving non-target attributes (e.g., background, objects). EFA eliminates generative bias without introducing new unfair associations.
tags:
  - ICCV 2025
  - Image Generation
  - Debiasing
  - Attribute Disentanglement
  - Cross-Attention
  - Diffusion Models
  - Fairness
date: 2026-05-08
content_hash: 722e0bb4140563dd
---

# Fair Generation without Unfair Distortions: Debiasing Text-to-Image Generation with Entanglement-Free Attention

**Conference**: ICCV 2025
**arXiv**: [2506.13298](https://arxiv.org/abs/2506.13298)
**Code**: Unavailable
**Area**: Text-to-Image Generation / Fairness
**Keywords**: Debiasing, Attribute Disentanglement, Cross-Attention, Diffusion Models, Fairness

## TL;DR

This paper proposes Entanglement-Free Attention (EFA), a training-free inference-time debiasing method that injects target attributes (e.g., gender, race) into person regions by modifying the cross-attention mechanism, while preserving non-target attributes (e.g., background, objects). EFA eliminates generative bias without introducing new unfair associations.

## Background & Motivation

Diffusion models (e.g., Stable Diffusion) frequently exhibit social biases in text-to-image generation — for instance, generating predominantly female figures for "nurse" or white males for "CEO." While existing debiasing methods can adjust target attribute distributions, they suffer from a critical problem: **Attribute Entanglement**.

The paper illustrates this with a compelling example: when applying debiasing to diversify race in the prompt "face of successful man with his house," existing methods not only alter the subject's race but also change the material and architectural style of the house — assigning low-quality housing to Black subjects and high-end housing to white subjects, thereby **exacerbating socioeconomic stereotypes**.

Specific issues with existing methods:

**Fine-tuning methods** (UCE, Finetuning Diffusion): Fine-tuning on small datasets degrades generation quality and diversity and inevitably modifies model parameters.

**Inference-time guidance methods** (Interpret Diffusion, Concept Algebra): Although model parameters remain unchanged, unintended modifications to non-target attributes such as backgrounds are still unavoidable.

The paper's core insight is that bias removal should be **confined to semantically relevant person regions**, and non-target attributes (e.g., scenes, objects, backgrounds) should remain unaffected by debiasing.

## Method

### Overall Architecture

EFA is a lightweight attention augmentation module inserted into specific cross-attention layers of the UNet. For each target bias category $C$ (e.g., gender), a separate EFA module is trained for each target attribute $a_i$ (e.g., female, male). At inference time, an attribute is sampled uniformly at random and its corresponding EFA module is applied.

Training data construction: Images are generated using the original model with prompts that explicitly describe target attributes. Person segmentation masks are extracted via Grounded SAM2, forming an $(\text{image}, \text{mask})$ dataset.

### Key Designs

1. **Attention Predictor (AP)**: The core component of EFA is a lightweight AP module (3 convolutional layers + 2 SiLU activations) that receives intermediate features $\mathbf{z}_t$ and predicts additional attention values. These values are concatenated with the original cross-attention values before softmax, controlling the injection intensity of the target attribute value vector $V_{a_i}$:

$$\text{EFA}_C^{a_i}(\mathbf{z}_t) = \text{softmax}\left(\left[\frac{QK^\top}{\sqrt{d}}, \text{AP}_{a_i}(\mathbf{z}_t)\right]\right) [V, V_{a_i}]$$

where $V_{a_i} = \pi(p_{a_i}) W_v$ is the value vector of the target attribute text. The softmax competition mechanism automatically balances the weights between the original and attribute value vectors.

2. **Dual-Scenario Training Strategy**: EFA is trained under two scenarios to learn adaptive augmentation intensity:

    - **Scenario 1 (Target attribute already present)**: Input consists of a prompt $p_a$ containing the target attribute $a$ and the corresponding noisy image. EFA should make no modifications; L1 regularization drives the AP output toward zero:
    $\mathcal{L}_{reg}^{trg} = \|W \odot A_a(\mathbf{z}_t)\|_1, \quad W = \mathbf{1}$

    - **Scenario 2 (Counterfactual attribute)**: Input uses a counterfactual prompt $p_a^{cf}$ (e.g., male when the target is female); EFA should enhance the target attribute in the person region. A mask-constrained reconstruction loss is used:
    $\mathcal{L}_{recon} = \mathbb{E}[\|M \odot (\epsilon - \epsilon_\theta(\mathbf{x}_t, t, p_a^{cf}))\|_2^2]$

   Regularization is additionally applied to non-person regions to prevent the influence from spreading to the background:
    $\mathcal{L}_{reg}^{cf} = \|\mathbf{1} - \bar{M}) \odot A_a(\mathbf{z}_t^{cf})\|_1$

3. **Multi-Attribute AP with Shared Backbone**: All AP modules within the same bias category share convolutional backbone weights, differing only in output channels. This enables efficient joint training of EFA modules for multiple attributes.

4. **Mask-Free Inference**: Segmentation masks are used as supervision signals during training but are not required at inference time. The AP has learned to apply attribute enhancement to semantically relevant regions, enabling fully automatic debiased generation.

### Loss & Training

The total loss is:
$$\mathcal{L}_{tot} = \mathcal{L}_{recon} + \lambda_1 \mathcal{L}_{reg}^{trg} + \lambda_2 \mathcal{L}_{reg}^{cf}$$

- EFA is applied to the 16×16 input-resolution layers in the UNet upsampling module (low-resolution features capture high-level semantic information).
- Only AP module parameters are updated; all original diffusion model parameters are frozen.
- Experiments are conducted on Stable Diffusion v1.5.
- The approach supports extension to multiple bias categories (e.g., simultaneous debiasing of gender × race).

## Key Experimental Results

### Main Results

Evaluation on 36 WinoBias occupations under two prompt templates ($\mathcal{T}_{basic}$ / $\mathcal{T}_{complex}$):

**Gender Bias Removal**:

| Method | DR↓ (basic) | PSNR↑ (basic) | LPIPS↓ (basic) | DINO↑ (basic) |
|------|------------|---------------|----------------|---------------|
| Original SD | 0.71 | - | - | - |
| UCE | 0.34 | 21.04 | 0.1374 | 0.757 |
| Interpret Diff. | 0.26 | 17.18 | 0.2290 | 0.616 |
| Finetuning Diff. | 0.48 | 22.62 | 0.1166 | 0.814 |
| **EFA (Ours)** | **0.06** | **32.52** | **0.0411** | **0.916** |

**Race Bias Removal**:

| Method | DR↓ (basic) | PSNR↑ (basic) | LPIPS↓ (basic) | DINO↑ (basic) |
|------|------------|---------------|----------------|---------------|
| Original SD | 0.60 | - | - | - |
| UCE | 0.27 | 21.55 | 0.1261 | 0.787 |
| Interpret Diff. | 0.16 | 16.87 | 0.2416 | 0.584 |
| **EFA (Ours)** | **0.04** | **30.93** | **0.0353** | **0.938** |

Model Preservation (COCO-no-person dataset):

| Method | FID↓ | CLIP-T↑ |
|------|------|---------|
| Original SD | - | 26.17 |
| UCE | 11.65 | 25.17 |
| Interpret Diff. | 15.78 | 24.80 |
| Finetuning Diff. | 1.92 | 25.79 |
| **EFA (Ours)** | **0.23** | **26.03** |

### Ablation Study

| Configuration | Description |
|------|------|
| Layer selection | 16×16 resolution layers are optimal; higher-resolution layers compromise detail preservation |
| Mask usage | Training without masks degrades non-target attribute preservation |
| Counterfactual training | Removing the counterfactual scenario significantly reduces debiasing effectiveness |
| Regularization weights | $\lambda_1$, $\lambda_2$ balance bias removal and attribute preservation |
| Multi-bias extension | Simultaneous gender × race debiasing (8 categories) reduces DR from 0.56 to 0.03 |

### Key Findings

- EFA achieves substantially lower DR than all baselines (gender: 0.06 vs. second-best 0.26; race: 0.04 vs. second-best 0.16).
- Non-target attribute preservation is dramatically superior: PSNR 32.52 vs. second-best 22.62 (+10 dB), DINO 0.916 vs. second-best 0.814.
- Model preservation is strongest: FID of only 0.23 (non-person images are virtually unaffected), compared to 11.65 for UCE.
- Inference overhead is minimal: only a lightweight AP module is added to selected layers.

## Highlights & Insights

- **Precise problem formulation**: The paper is the first to explicitly identify the "attribute entanglement" problem, revealing that existing debiasing methods may introduce new biases.
- **Elegant design**: Appending additional attention values before softmax exploits the attention competition mechanism to naturally achieve attribute injection/suppression.
- **Dual-scenario training**: Enables the AP to learn adaptive behavior — augmenting when necessary and remaining inactive otherwise.
- **High practicality**: No masks required at inference, no model parameter modifications, supports multiple bias categories, and allows users to control attribute distributions via sampling probabilities.
- **Remarkable quantitative margins**: Non-target attribute preservation metrics decisively outperform all baseline methods.

## Limitations & Future Work

- Currently limited to person-centric biases (gender, race); applicability to object- or scene-level biases has not been validated.
- Attribute distribution evaluation relies on CLIP classifiers, which may themselves exhibit classification biases.
- The binary gender setup used in evaluation is a simplification that does not reflect gender diversity.
- Training effectiveness depends on person segmentation mask quality; heavily occluded scenes may be problematic.
- For undefined attribute biases (e.g., socioeconomic status), prior knowledge is still required to define the target attribute set.
- Validation is limited to SD v1.5; applicability to newer architectures (e.g., SDXL, SD3) remains to be verified.

## Related Work & Insights

- **UCE** [Gandikota et al., 2023]: Debiases by modifying cross-attention weights, but alters model parameters.
- **Interpret Diffusion** [Li & Parihar et al., 2024]: Debiases via h-space manipulation, but causes background changes.
- **Semantic Guidance** [Friedrich et al., 2023]: Requires pre-registered concepts, limiting flexibility.
- **Grounded SAM2** [Liu et al., 2024]: Provides person segmentation masks for training.
- Insight: Performing "addition" in the attention mechanism (concatenation rather than replacement) is safer than "subtraction" (erasing/modifying weights).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The identification of the attribute entanglement problem is highly insightful; EFA is elegantly designed and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three-dimensional evaluation (bias / attribute preservation / model preservation) across two prompt templates and multiple bias combinations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The motivation figure (Fig. 1) is immediately clear; problem articulation is exceptionally precise.
- **Value**: ⭐⭐⭐⭐⭐ A practically deployable debiasing solution that addresses the long-overlooked attribute entanglement problem.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering](larender_training-free_occlusion_control_in_image_generation_via_latent_renderin.md)
- [\[ICCV 2025\] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](dense2moe_restructuring_diffusion_transformer_to_moe_for_efficient_text-to-image.md)
- [\[ICCV 2025\] Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences](cycle_consistency_as_reward_learning_image-text_alignment_without_human_preferen.md)
- [\[ICCV 2025\] Addressing Text Embedding Leakage in Diffusion-Based Image Editing](addressing_text_embedding_leakage_in_diffusion-based_image_editing.md)
- [\[ICCV 2025\] Rethink Sparse Signals for Pose-guided Text-to-Image Generation](rethink_sparse_signals_for_pose-guided_text-to-image_generation.md)

<!-- RELATED:END -->
