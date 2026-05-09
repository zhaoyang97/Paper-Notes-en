---
title: >-
  [Paper Note] PureCC: Pure Learning for Text-to-Image Concept Customization
description: >-
  [CVPR 2026][Image Generation][Concept Customization] PureCC introduces a decoupled learning objective that separates "target concept implicit guidance" from "original condition prediction," coupled with a dual-branch training pipeline comprising a frozen representation extractor and a trainable flow model, along with adaptive guidance scaling $\lambda^{\star}$ derived from projection error. This enables high-fidelity concept customization while minimizing disruption to the original model's behavior and capabilities.
tags:
  - CVPR 2026
  - Image Generation
  - Concept Customization
  - Diffusion Model Fine-tuning
  - Implicit Guidance
  - Model Preservation
  - Adaptive Scaling
date: 2026-05-08
content_hash: 4e030b7e3116d2b7
---

# PureCC: Pure Learning for Text-to-Image Concept Customization

**Conference**: CVPR 2026
**arXiv**: [2603.07561](https://arxiv.org/abs/2603.07561)
**Code**: [https://github.com/lzc-sg/PureCC](https://github.com/lzc-sg/PureCC)
**Area**: Image Generation
**Keywords**: Concept Customization, Diffusion Model Fine-tuning, Implicit Guidance, Model Preservation, Adaptive Scaling

## TL;DR
PureCC introduces a decoupled learning objective that separates "target concept implicit guidance" from "original condition prediction," coupled with a dual-branch training pipeline comprising a frozen representation extractor and a trainable flow model, along with adaptive guidance scaling $\lambda^{\star}$ derived from projection error. This enables high-fidelity concept customization while minimizing disruption to the original model's behavior and capabilities.

## Background & Motivation

**Background**: Concept Customization uses 3–5 reference images to teach T2I models personalized concepts (subjects, styles, etc.). Mainstream approaches fall into tuning-free methods (e.g., DreamO, UNO, which encode reference image features for injection) and tuning-based methods (e.g., DreamBooth full-parameter fine-tuning, LoRA low-rank fine-tuning).

**Limitations of Prior Work**: Existing methods focus on high fidelity and multi-concept customization while overlooking two critical issues:
   - **Original behavior corruption**: After learning [V] dog, non-target elements (background, style, lighting) are unintentionally altered, because redundant information in limited reference images cannot be disentangled from the target concept.
   - **Original capability degradation**: Text-following ability and image quality decline after fine-tuning; KL divergence visualization reveals significant distribution drift.

**Key Challenge**: Existing methods treat all language–visual knowledge in the customization set as the learning source. However, with only 3–5 reference images, the model cannot distinguish the target concept from redundant background information. Furthermore, the learning objective lacks explicit consideration of the original model, causing distribution drift during concept learning.

**Key Insight**: Inspired by the implicit guidance formulation of Classifier-Free Guidance (CFG)—which treats conditional generation as "unconditional prediction + implicit conditional guidance"—concept customization can analogously be viewed as "original conditional prediction + implicit target concept guidance." This decoupled formulation naturally supports learning new concepts while preserving the original model.

**Core Idea**: $v_t^{PureCC} = v_t^{original} + \lambda^{\star} \cdot v_t^{target}$, where the original prediction is provided by the trainable model (preserving original capabilities), the target guidance is provided by the frozen extractor (clean concept representation), and $\lambda^{\star}$ adaptively balances the two via projection error.

## Method

### Overall Architecture
Built upon SD 3.5-M (a flow-based generative model). Training proceeds in two stages: (1) training the representation extractor—fine-tuning a pretrained flow model on the customization set using LoRA with hierarchical tunable concept embeddings; (2) pure learning—the frozen extractor provides target concept guidance, while a separate trainable flow model provides the original prediction, jointly optimized with $\mathcal{L}_{PureCC}$.

### Key Designs

1. **Representation Extractor (Stage 1)**

    - **Function**: Enhances the model's understanding of the personalized concept and provides clean target concept representations.
    - **Mechanism**: Fine-tunes a pretrained flow model $v_t^{\theta_1}$ with LoRA and introduces **hierarchical tunable concept embeddings** $\{\mathbf{Y}_{tar}^l\}_{l=1}^L$—replacing the [V] token at each Transformer layer with a distinct learnable embedding, allowing different layers to capture different aspects of the target concept (texture, shape, etc.).
    - **Training Loss**: Standard CFM loss $\mathcal{L}_{CC}^{Rep}$.
    - **Design Motivation**: Hierarchical embeddings capture richer concept details than a single unified embedding.

2. **Decoupled Learning Objective**

    - **Function**: Decomposes the concept-customized velocity field into two independent components: "original" and "target."
    - **Core Formula**: $v_t^{PureCC} = v_t^{\theta_2}(x_t | y_{base}) + \lambda^{\star} \cdot [v_t^{\theta_1}(x_t | y_{tar}) - v_t^{\theta_1}(x_t | \emptyset)]$
    - $v_t^{original} = v_t^{\theta_2}(x_t | y_{base})$ uses the Base Text (excluding [V]) as condition, representing the original model's predictive capability.
    - $v_t^{target} = \mathbf{R}(y_{tar})$ is the difference between the frozen extractor's predictions under the Target Text and the null condition, representing the pure target concept guidance offset.
    - **Design Motivation**: Output conditioned on Base Text sufficiently represents original model performance; additive composition preserves original capabilities.

3. **Adaptive Guidance Scaling $\lambda^{\star}$**

    - **Function**: Dynamically balances concept fidelity and original model preservation.
    - **Mechanism**: $\lambda^{\star}$ is defined as the projection coefficient of the trainable model's learned concept representation onto the frozen model's concept guidance: $\lambda^{\star} = \frac{\langle \mathbf{R}(y_{complete}, y_{base}), \mathbf{R}(y_{tar}) \rangle}{\|\mathbf{R}(y_{tar})\|^2}$
    - **Intuition**: Early in training, when the trainable model has not yet learned the concept direction, $\lambda^{\star}$ is automatically reduced to avoid contaminating the original model; as alignment improves later in training, $\lambda^{\star}$ increases to reinforce concept learning.
    - Closed-form solution; no additional hyperparameter tuning required.

4. **Dual-Branch Training Pipeline (Stage 2)**

    - **Frozen branch**: Representation extractor $v_t^{\theta_1}$, providing $v_t^{target}$.
    - **Trainable branch**: A separate pretrained flow model $v_t^{\theta_2}$, trained with the joint loss $\mathcal{L}_{PCC} = \mathcal{L}_{CC} + \eta \cdot \mathcal{L}_{PureCC}$.
    - $\mathcal{L}_{PureCC}$ constrains the full prediction to align with the decoupled target; $\mathcal{L}_{CC}$ preserves the generative prior of the velocity field.

### Loss & Training
- Backbone: SD 3.5-M; LoRA rank=4; learning rate 1e-4.
- Customization set: 14 concepts from the DreamBooth dataset plus 16 self-collected concepts (instance and style).
- Evaluation benchmark: DreamBenchPCC (extended DreamBench + 12 style concepts).

## Key Experimental Results

### Main Results (DreamBenchPCC, Instance Concepts)

| Method | ΔCLIP-T↑ | ΔHPSv2.1↑ | Seg-Cons↑ | CLIP-I↑ | DINO↑ |
|--------|----------|-----------|-----------|---------|-------|
| DreamBooth | -4.81 | -2.17 | 18.38 | 0.63 | 0.62 |
| Mix-of-Show | -2.71 | -1.08 | 15.72 | 0.72 | 0.61 |
| CIFC | -1.93 | -1.62 | 13.23 | 0.78 | 0.65 |
| DreamO (free) | - | - | - | 0.71 | 0.67 |
| **PureCC** | **-0.31** | **+0.10** | **69.37** | **0.81** | **0.73** |

### Ablation Study

| Strategy | ΔCLIP-T↑ | ΔHPSv2.1↑ | Seg-Cons↑ | CLIP-I↑ | DINO↑ |
|----------|----------|-----------|-----------|---------|-------|
| $\mathcal{L}_{CC}$ (baseline) | -4.52 | -2.01 | 23.74 | 0.65 | 0.66 |
| Merged Training | -1.17 | -0.34 | - | - | - |
| **PureCC (full)** | **-0.31** | **+0.10** | **69.37** | **0.81** | **0.73** |

### Key Findings
- **Seg-Cons is the most prominent advantage**: PureCC achieves 69.37, far exceeding the second-best DreamBooth+EWC at 26.37, demonstrating superior original behavior preservation.
- **ΔCLIP-T is near zero** (-0.31 vs. DreamBooth's -4.81), indicating that text-following capability is almost entirely intact.
- **ΔHPSv2.1 is even positive** (+0.10), showing that image quality actually improves after customization.
- Concept fidelity simultaneously reaches state-of-the-art (CLIP-I 0.81, DINO 0.73), demonstrating that preservation does not come at the cost of fidelity.
- In multi-concept customization, semantic entanglement (e.g., color contamination between [V1] man and [V2] sunglasses) is effectively avoided.

## Highlights & Insights
- The **decoupled learning objective** is elegantly designed—naturally extending the CFG formulation to the training stage and reformulating concept customization as "original prediction + concept increment."
- The **closed-form adaptive $\lambda^{\star}$** is a refined design—the projection coefficient automatically reflects learning progress without manual hyperparameter tuning.
- **Hierarchical tunable concept embeddings** effectively augment standard Textual Inversion—using distinct embeddings per Transformer layer captures multi-scale features of the concept.
- This work is the first to systematically define and evaluate "behavior preservation" in concept customization (via the Seg-Cons metric), filling a gap in the evaluation landscape.

## Limitations & Future Work
- The dual-branch pipeline requires forward passes through two flow models, approximately doubling training cost compared to a single-branch approach.
- Hierarchical embeddings increase parameter count and training complexity; effectiveness with very few reference images (1–2 images) remains to be validated.
- Experiments are conducted primarily on SD 3.5-M; adaptability to other architectures (e.g., DiT-based FLUX) has not been explored.
- Adaptive $\lambda^{\star}$ depends on the quality of representation alignment between the two branches; insufficient extractor training may degrade scaling accuracy.
- Only static image generation is evaluated; temporal consistency in video customization scenarios is not discussed.

## Related Work & Insights
- **vs. DreamBooth**: Full-parameter fine-tuning in DreamBooth causes severe distribution drift (ΔCLIP-T -4.81); PureCC's decoupled objective and dual-branch design limit this to -0.31.
- **vs. CIFC**: CIFC constrains model preservation via cross-attention feature regularization; PureCC directly separates concept and original components in velocity field space, addressing the problem more fundamentally.
- **vs. DreamO/UNO** (tuning-free): These methods achieve lower concept fidelity (DINO 0.67/0.62) compared to PureCC (0.73) and cannot support multi-concept composition.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The decoupled learning objective derives naturally from CFG and extends it to the training stage; the closed-form adaptive scaling is both theoretically elegant and methodologically novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Quantitative evaluation introduces preservation metrics, multi-concept, and style-instance combination assessments, though experiments are limited to a single backbone model.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear and the pipeline diagram (Fig. 2) is intuitive, though some notation could be defined more concisely.
- **Value**: ⭐⭐⭐⭐⭐ The first work to systematically address original model preservation in concept customization; high practical value for real-world applications such as continuous multi-concept customization without capability degradation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CoLoGen: Progressive Learning of Concept-Localization Duality for Unified Image Generation](cologen_progressive_learning_of_concept-localization_duality_for_unified_image_g.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](dreamvideoomni_omnimotion_controlled_multisubject.md)
- [\[CVPR 2026\] LumiCtrl: Learning Illuminant Prompts for Lighting Control in Personalized Text-to-Image Models](lumictrl_learning_illuminant_prompts_for_lighting_control_in_personalized_text-t.md)

</div>

<!-- RELATED:END -->
