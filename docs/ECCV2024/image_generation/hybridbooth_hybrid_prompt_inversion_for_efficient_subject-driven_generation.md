---
title: >-
  [Paper Note] HybridBooth: Hybrid Prompt Inversion for Efficient Subject-Driven Generation
description: >-
  [ECCV 2024][Image Generation][Subject-Driven Generation] This paper proposes HybridBooth, a two-stage hybrid prompt inversion framework. By first generating an initial word embedding using a regressor (Probe) and then performing residual fine-tuning (Refinement), it achieves efficient subject-driven personalized image generation in only 3-5 iteration steps.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Subject-Driven Generation"
  - "Prompt Inversion"
  - "Text Embedding"
  - "Diffusion Models"
  - "Personalized Generation"
date: 2026-05-08
content_hash: 34246e48554a40fe
---

# HybridBooth: Hybrid Prompt Inversion for Efficient Subject-Driven Generation

**Conference**: ECCV 2024  
**arXiv**: [2410.08192](https://arxiv.org/abs/2410.08192)  
**Code**: [Project Page](https://sites.google.com/view/hybridbooth)  
**Area**: Image Generation  
**Keywords**: Subject-Driven Generation, Prompt Inversion, Text Embedding, Diffusion Models, Personalized Generation

## TL;DR

This paper proposes HybridBooth, a two-stage hybrid prompt inversion framework. By first generating an initial word embedding using a regressor (Probe) and then performing residual fine-tuning (Refinement), it achieves efficient subject-driven personalized image generation in only 3-5 iteration steps.

## Background & Motivation

Subject-driven generation aims to generate images of a specific subject in new scenes or styles based on reference images. Existing methods fall into two main categories:

**Optimization-based methods** (Textual Inversion, DreamBooth, Custom Diffusion): Align subject features by iteratively optimizing text embeddings or model weights. Although highly precise, they suffer from heavy computational costs, typically requiring hundreds to thousands of iteration steps.

**Direct regression-based methods** (ELITE, FastComposer): Train an encoder to map images directly into the text embedding space, achieving zero-shot generation. However, they struggle with detail preservation and cross-domain generalization.

Both paradigms have trade-offs, and the key challenge lies in the fact that optimization methods are slow but precise, while regression methods are fast but coarse. The core insight of HybridBooth is that: **a powerful encoder reduces the cost of iterative optimization, while efficient iterative optimization alleviates the accuracy requirement for the encoder**. Therefore, integrating these two paradigms can simultaneously achieve high efficiency and high fidelity.

## Method

### Overall Architecture

HybridBooth consists of two stages:

**Stage 1: Word Embedding Probe**
- Trains a prompt regressor that takes the subject image as input and outputs an initial estimation of word embeddings.
- Pre-trained on large-scale datasets (such as the FFHQ 70k images).
- Provides a coarse-grained but robust initial embedding.

**Stage 2: Word Embedding Refinement**
- Performs fast fine-tuning on the regressor (only 3-5 steps) to adapt to a specific subject.
- Employs a residual refinement strategy to achieve rapid adaptation while preserving pre-trained priors.

### Key Designs

#### 1. Multi-grained Image Feature Merging

Existing methods rely solely on CLIP features, lacking pixel-level detail information. HybridBooth utilizes two complementary features simultaneously:

- **CLIP Feature** $\boldsymbol{f}_c$: Provides global semantic information (e.g., subject category).
- **DINOv2 Feature** $\boldsymbol{f}_d$: Provides detailed pixel-level information.

Merging process:

$$\boldsymbol{f} = \text{Linear}([\boldsymbol{f}_c, \text{MLP}(\boldsymbol{f}_d)])$$

Where the MLP structure is LayerNorm-Linear-GELU-Linear, which projects the DINOv2 feature to a dimension compatible with the CLIP feature. Alignment is then completed through concatenation and a linear layer. This simple module brings significant performance improvements.

#### 2. Multiple-word Regression

A single word embedding cannot fully describe all characteristics of a subject (e.g., "man" conveys category information but lacks details like hairstyle). Thus, the merged features are mapped to multiple word embeddings:

$$\boldsymbol{e} = \mathcal{R}(\boldsymbol{f})$$

Where $\boldsymbol{e} = \{\boldsymbol{e}_i\}_{i=1}^n$. An empirical setting of $n=5$ word embeddings is used to balance efficiency and expressiveness.

#### 3. Residual Refinement

The key innovation of the refinement stage is to avoid directly fine-tuning all parameters, adopting a residual form instead:

$$\boldsymbol{W}'_{\phi} = \boldsymbol{W}_{\phi} + \lambda \Delta \boldsymbol{W}_{\phi}$$

- $\boldsymbol{W}_{\phi}$: Parameters learned in the Probe stage (acting as anchors to preserve priors).
- $\Delta \boldsymbol{W}_{\phi}$: Learned residual parameters.  
- $\lambda$: A hyperparameter controlling the residual magnitude (set to $1\text{e}{-2}$).

This design offers several key advantages:
- **Inhibiting Overfitting**: $\boldsymbol{W}_{\phi}$ acts as an anchor to stabilize the direction of updates.
- **Robustness to Hyperparameters**: Good results are obtained even when $\lambda$ and iteration steps vary over a wide range.
- **Fine-tuning Critical Parameters Only**: Based on layer importance experiments, the KQV matrices of cross-attention layers in the regressor are selected for fine-tuning (importance score 56.3), rather than self-attention (43.9) or convolutional layers (12.4).

#### 4. Prompt Regressor Design

The architecture of PromptNet is adopted, which has a structure similar to LDM blocks and can be initialized with pre-trained weights from the LDM. This design naturally endows the regressor with powerful visual understanding capabilities.

### Loss & Training

**The training loss consists of two parts:**

$$\mathcal{L} = \mathcal{L}_{\epsilon} + \alpha_{\boldsymbol{M}} \mathcal{L}_{\boldsymbol{M}}$$

**1. Diffusion Denoising Loss** $\mathcal{L}_{\epsilon}$: Standard LDM noise prediction loss.

$$\mathcal{L}_{\epsilon} = \mathbb{E}_{z, \epsilon, c, t}\left[\|\epsilon - \mathcal{M}_{\theta}(z_t, c, t)\|_2^2\right]$$

**2. Mask Regularization Loss** $\mathcal{L}_{\boldsymbol{M}}$:

Experiments show that the cross-attention map of subject word embeddings tends to leak into irrelevant background regions. A segmentation mask $\boldsymbol{M}$ (generated by InSPyReNet) is introduced to restrict attention to the subject region:

$$\mathcal{L}_{\boldsymbol{M}} = \frac{1}{n}\sum_{i=1}^{n}\text{mean}(\boldsymbol{A}_{\boldsymbol{e}_i} \cdot (1 - \boldsymbol{M})) - \text{mean}(\boldsymbol{A}_{\boldsymbol{e}_i} \cdot \boldsymbol{M})$$

This loss minimizes attention outside the mask and maximizes attention inside the mask.

**Training Details:**
- Base model: Stable Diffusion v1.5
- Probe stage: AdamW, lr=2e-5, batch size=8, trained on a single A100 GPU for 40 hours.
- Refinement stage: AdamW, lr=2e-5, weight decay=1e-2, 5 steps only.
- Hyperparameters: $\alpha_{\boldsymbol{M}} = 1\text{e}{-3}$, $\lambda = 1\text{e}{-2}$.

## Key Experimental Results

### Main Results

Quantitative evaluation on CelebA-HQ and DreamBooth datasets:

| Method | Type | CLIP-T ↑ | CLIP-I ↑ | DINO-I ↑ | Iteration Steps ↓ |
|------|------|----------|----------|----------|-----------|
| Textual Inversion | Optimization | 0.164 | 0.612 | 0.236 | 5000 |
| DreamBooth | Optimization | 0.251 | 0.564 | 0.376 | 1000 |
| Custom Diffusion | Optimization | 0.237 | 0.675 | 0.398 | 200 |
| ELITE | Regression | 0.169 | 0.592 | 0.311 | 1 |
| FastComposer | Regression | 0.201 | 0.782 | 0.581 | 1 |
| **HybridBooth** | **Hybrid** | **0.246** | **0.865** | **0.644** | **5** |

Results on the DreamBooth dataset:

| Method | CLIP-T ↑ | CLIP-I ↑ | DINO-I ↑ |
|------|----------|----------|----------|
| Custom Diffusion | 0.245 | 0.801 | 0.695 |
| ELITE | 0.255 | 0.762 | 0.652 |
| **HybridBooth** | **0.261** | **0.865** | **0.755** |

### Ablation Study

| Variant | CLIP-T ↑ | CLIP-I ↑ | DINO-I ↑ |
|------|----------|----------|----------|
| HybridBooth (Full) | 0.246 | 0.865 | 0.644 |
| w/o Refinement | 0.177 | 0.842 | 0.568 |
| w/o Probe | 0.153 | 0.408 | 0.068 |
| w/o DINO Feature | 0.161 | 0.837 | 0.453 |
| w/o CLIP Feature | 0.182 | 0.734 | 0.510 |
| w/o Mask Regularization | 0.203 | 0.831 | 0.625 |

### Key Findings

1. **Both Probe and Refinement are Indispensable**: Removing the Probe (w/o Probe) causes DINO-I to plunge from 0.644 to 0.068, showing that residual optimization can hardly function without a well-initialized starting point.
2. **Significant Contribution of DINO Features**: Removing DINO features drops DINO-I from 0.644 to 0.453, verifying the importance of pixel-level features for subject fidelity.
3. **Effectiveness of Mask Regularization**: Removing it reduces CLIP-T from 0.246 to 0.203, as attention leakage worsens text alignment.
4. **Cross-species Generalization**: Encoders trained on human face data can transfer to other species with similar semantic structures, such as dogs and cats.
5. **DINO-I is more than 10% higher than FastComposer**, while requiring only 5 iteration steps.

## Highlights & Insights

1. **Philosophy of the Hybrid Paradigm**: A good initial estimation reduces the difficulty of refinement, while the capacity for refinement relaxes the precision requirements of the initial estimation—this synergy is the core idea of this method design.
2. **Elegance of Residual Refinement**: Preserving pre-trained weights as anchors prevents overfitting in single-image fine-tuning while maintaining model generalization capability.
3. **High Practical Value**: Seamlessly compatible with community models and control methods like ControlNet, as the method solely operates in the text embedding space without modifying the generative model itself.
4. **Only 5 iteration steps** are required to outperform optimization methods that take 200-5000 steps, yielding a 40-1000x speedup.

## Limitations & Future Work

1. **Inability to Perform Precise Semantic Editing**: Lacks sufficient control over fine-grained attributes such as adjusting facial expressions or age.
2. **Inherited Flaws of Stable Diffusion**: Generates poor quality for fine structures such as fingers.
3. **Dataset Limitations**: Primarily trained and evaluated on facial data; generalization to other domains requires more comprehensive validation.
4. **The authors suggest incorporating vision-language models** to enhance understanding and planning capabilities, as well as utilizing more 3D structural information.

## Related Work & Insights

- **Textual Inversion / DreamBooth**: Representatives of the optimization paradigm, offering a high theoretical ceiling but low efficiency.
- **ELITE / FastComposer**: Representatives of the regression paradigm, high-speed but lacking fidelity.
- **HyperDreamBooth**: Also attempts a hybrid strategy, but limits personalized expressiveness due to low-rank weight updates.
- **Custom Diffusion**: Shares a similar idea of selecting parameters to fine-tune based on layer importance analysis.
- **Insight**: The hybrid paradigm (coarse-to-fine) is a general strategy in embedding learning and can be generalized to other personalized generation tasks.

## Rating

- **Novelty**: ★★★★☆ — Clear hybrid paradigm concept, elegant design for residual refinement.
- **Experimental Thoroughness**: ★★★★☆ — Exhaustive ablation studies, covering both facial and non-facial evaluations.
- **Writing Quality**: ★★★★☆ — Clear structure and comprehensive comparison with baseline methods.
- **Value**: ★★★★★ — Requires only 5 iteration steps, compatible with community models, low barrier to deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)
- [\[CVPR 2026\] FlowFixer: Towards Detail-Preserving Subject-Driven Generation](../../CVPR2026/image_generation/flowfixer_towards_detail-preserving_subject-driven_generation.md)
- [\[ECCV 2024\] The Fabrication of Reality and Fantasy: Scene Generation with LLM-Assisted Prompt Interpretation](the_fabrication_of_reality_and_fantasy_scene_generation_with_llm-assisted_prompt.md)
- [\[ECCV 2024\] Soft Prompt Generation for Domain Generalization](soft_prompt_generation_for_domain_generalization.md)
- [\[ECCV 2024\] ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](ziplora_any_subject_in_any_style_by_effectively_merging_loras.md)

</div>

<!-- RELATED:END -->
