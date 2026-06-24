---
title: >-
  [Paper Note] LoRACLR: Contrastive Adaptation for Customization of Diffusion Models
description: >-
  [CVPR 2025][Image Generation][LoRA Merging] LoRACLR proposes a LoRA model merging method based on a contrastive learning objective. By learning a delta weight, it fuses multiple independently trained single-concept LoRA models into a unified model without retraining or accessing the original training data. This achieves high-fidelity multi-concept image generation, requiring only 5 minutes to merge 12 concepts.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "LoRA Merging"
  - "Contrastive Learning"
  - "Multi-Concept Generation"
  - "Model Fusion"
  - "Diffusion Model Customization"
date: 2026-05-08
content_hash: fd27144d98e80339
---

# LoRACLR: Contrastive Adaptation for Customization of Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2412.09622](https://arxiv.org/abs/2412.09622)  
**Code**: [https://loraclr.github.io](https://loraclr.github.io)  
**Area**: Diffusion Models / Personalized Image Generation  
**Keywords**: LoRA Merging, Contrastive Learning, Multi-Concept Generation, Model Fusion, Diffusion Model Customization

## TL;DR

LoRACLR proposes a LoRA model merging method based on a contrastive learning objective. By learning a delta weight, it fuses multiple independently trained single-concept LoRA models into a unified model without retraining or accessing the original training data. This achieves high-fidelity multi-concept image generation, requiring only 5 minutes to merge 12 concepts.

## Background & Motivation

**Background**: Text-to-image diffusion models (such as Stable Diffusion) combined with LoRA fine-tuning realize efficient personalized image generation. An extensive collection of pre-trained LoRA models for single concepts (characters, objects, styles) is already available in the community.

**Limitations of Prior Work**: When generating multiple personalized concepts simultaneously in a single image, existing methods face severe challenges. Weighted averaging (FedAvg) leads to feature interference; Custom Diffusion requires joint training on multiple concepts; Mix-of-Show requires a specific ED-LoRA format, making it incompatible with standard community LoRAs; ZipLoRA is limited to merging style and content; OMG relies on the accuracy of detection/segmentation models; Orthogonal Adaptation requires refinetuning each LoRA on the original data.

**Key Challenge**: Independently trained LoRA models encode different concepts in their respective weight spaces. Direct merging leads to concept bleeding (e.g., mixing the hair features of character A onto B) or concept omission. However, users desire to directly combine existing community LoRA models without retraining.

**Goal**: How to merge multiple independent LoRA models into a single unified model that can accurately generate all concepts, without retraining and without access to the original training data?

**Key Insight**: Leveraging contrastive learning to align the weight space of the merged model. The output of the same concept should align with that of the original LoRA (positive pairs), while outputs of different concepts should push away from each other (negative pairs), thereby naturally maintaining the independence of each concept.

**Core Idea**: By learning an additive delta weight, a contrastive loss is employed to make the merged model's feature outputs close to the original LoRA for the corresponding concept while pushing them away from other LoRAs, enabling training-free multi-concept LoRA merging.

## Method

### Overall Architecture

The input to LoRACLR consists of $N$ independently trained single-concept LoRA models $\{V_i\}_{i=1}^N$. In the first stage, each LoRA model is used to generate concept-specific input-output feature pairs $(X_i, Y_i)$ to establish positive and negative sample pairs. In the second stage, an all-zero delta weight $\Delta W$ is initialized and optimized by minimizing a contrastive loss combined with L2 regularization, enabling the merged model $W + \Delta W$ to accurately reconstruct all concepts. After optimization, the delta weight $\Delta W$ is added to all LoRA weights to obtain the final unified model. The entire process takes approximately 5 minutes on an NVIDIA A100 GPU for 12 concepts.

### Key Designs

1. **Contrastive Merging Objective**:

    - **Function**: Aligns multiple LoRA models in the weight space to maintain concept independence.
    - **Mechanism**: For each concept $i$, the positive sample distance is defined as $d_{p,i} = \|Y_i - \hat{Y}_i\|_2$ (original LoRA output vs. merged model output), and the negative sample distance as $d_{n,i} = \min_{j \neq i} \|Y_i - \hat{Y}_j\|_2$ (original output of concept $i$ vs. merged model outputs of other concepts). The contrastive loss is formulated as $\mathcal{L}_{contrastive} = \frac{1}{N} \sum_{i=1}^{N} (d_{p,i}^2 + \max(0, m - d_{n,i})^2)$, where $m$ is a margin parameter. Pulling positive pairs closer ensures identity preservation, while pushing negative pairs apart prevents concept bleeding.
    - **Design Motivation**: Simple weighted average merging (FedAvg) cannot handle interference between concepts, as different LoRAs might encode different concepts in similar weight directions. Contrastive learning is naturally suited to maintain individual characteristics and prevent confusion.

2. **Delta-Based Merging Strategy**:

    - **Function**: Merges LoRAs via incremental updates, protecting the integrity of each model's original weights.
    - **Mechanism**: Instead of directly modifying the weights of any LoRA model, an additive delta weight $\Delta W$ (initialized to zero) is learned. An L2 regularization $\mathcal{L}_{delta} = \lambda_{delta} \|\Delta W\|_2$ is imposed on $\Delta W$ to encourage sparsity and minimal adjustment. The total loss is defined as $\mathcal{L}_{total} = \mathcal{L}_{contrastive} + \mathcal{L}_{delta}$. $\Delta W$ is optimized using gradient descent.
    - **Design Motivation**: Directly modifying LoRA weights might destroy the original concept representations. The delta-based strategy confines the merging process to the minimal necessary adjustments, maintaining backward compatibility—the original LoRAs can still be used individually.

3. **Plug-and-Play Design Compatible with Community LoRAs**:

    - **Function**: Enables direct usage of pre-trained LoRA models from community platforms (e.g., Civitai).
    - **Mechanism**: The entire pipeline of LoRACLR requires neither access to the original training data nor specific LoRA variants (e.g., ED-LoRA). It only requires the weight files of existing LoRA models to generate feature pairs and perform merging optimization. It supports standard LoRA and ED-LoRA formats. Once merged, the model can generate images with arbitrary prompts without extra inference overhead.
    - **Design Motivation**: Prior methods (e.g., Mix-of-Show, Orthogonal Adaptation) either require specific LoRA formats or retraining on original datasets, which severely restricts their practicality. The design of LoRACLR directly targets practical deployment scenarios.

### Loss & Training

The total loss is formulated as $\mathcal{L}_{total} = \mathcal{L}_{contrastive} + \mathcal{L}_{delta}$. Stable Diffusion with the ChilloutMix checkpoint is used. The learning rate is set to 1e-4, margin $m = 0.5$, and $\lambda_{delta} = 0.001$. LoRAs are applied to $W_{in}$ and $W_{out}$ of the cross-attention layers. Running on an NVIDIA A100, merging 12 concepts takes about 5 minutes, and image generation takes approximately 10 seconds.

## Key Experimental Results

### Main Results

Merging results of 12 concept identities:

| Method | Text Align ↑ | Image Align ↑ (Δ) | Identity Align ↑ (Δ) |
|------|-------------|-------------------|---------------------|
| P+ | .643 | .683 (—) | .515 (—) |
| Custom Diffusion | .673 | .623 (-.025) | .408 (-.096) |
| DB-LoRA (FedAvg) | .682 | .531 (-.213) | .098 (-.585) |
| MoS (Grad Fusion) | .631 | .729 (-.016) | .717 (-.011) |
| Orthogonal Adapt. | .644 | .741 (-.007) | .745 (+.005) |
| **Ours** | .665 | **.776 (+.010)** | **.828 (+.029)** |

User study (50 participants, Identity Alignment score of 1-5):

| Method | Average Score |
|------|---------|
| **Ours** | **3.42** |
| Orthogonal Adapt. | 2.41 |
| Mix-of-Show | 2.21 |
| Prompt+ | 2.01 |

### Ablation Study

| Configuration | Key Observations |
|------|---------|
| margin = 0.25-0.5 | Optimal range, yielding the best identity preservation and visual consistency. |
| margin > 0.5 | Performance degradation; excessive separation leads to concept degradation. |
| λ_delta = 0.001 | Optimal, balancing merger performance and weight sparsity. |
| λ_delta > 0.01 | Restraints are too strong, leading to insufficient merging. |
| Number of concepts 2→12 | All metrics remain stable, demonstrating good scalability. |

### Key Findings

- Ours is the only method where both Image and Identity Alignment improve (rather than decrease) after merging: Image +.010, Identity +.029. Other methods generally degrade after merging.
- DB-LoRA (FedAvg) experiences a sharp drop in Identity Alignment from .683 to .098 (a decrease of .585) post-merging, showing that simple averaging completely fails when dealing with many concepts.
- As the number of concepts increases from 2 to 12, our metrics remain stable, whereas other methods significantly degrade at 6+ concepts (e.g., failing to preserve Messi's identity).
- The merging time is only 5 minutes compared to 120 minutes for Orthogonal Adaptation, representing a significant efficiency advantage.
- Integration of style LoRAs is supported, allowing for changing the artistic style while preserving concept identities.

## Highlights & Insights

- **Contrastive Learning for Weight Space Alignment**: Transferring contrastive learning from the feature space to the model weight merging process, where the definition of positive and negative pairs is elegant and natural—same-concept attraction and cross-concept repulsion. This idea can be transferred to other model merging scenarios (e.g., multi-task LoRA merging, model aggregation in federated learning).
- **One-time Post-training Merging**: Ours is a post-training method where the merged model can generate images containing arbitrary combinations via prompt composition without progressive, per-generation optimization. This makes it highly valuable in production environments—users can freely combine community LoRAs without retraining.
- **Robustness Against Increasing Concepts**: From 2 to 12 concepts, performance metrics show almost no degradation, which is highly rare among multi-concept methods. This indicates that the contrastive loss effectively prevents mutual interference caused by "concept crowding."

## Limitations & Future Work

- The performance of the method is limited by the quality of the underlying LoRA models—if the original single-concept LoRAs perform poorly, merging will not improve them.
- Verification is currently limited to Stable Diffusion 1.5; effectiveness on newer models like SDXL or Flux remains to be verified.
- The paper does not address scenarios where concepts exhibit fine-grained similarities (e.g., two people with highly similar appearances).
- There is a potential risk of malicious misuse, such as generating deepfakes.
- The margin parameter in the contrastive loss may require adaptive adjustment for different concept combinations.

## Related Work & Insights

- **vs Mix-of-Show**: MoS requires specialized ED-LoRA formats and original training data, limiting compatibility with community models. LoRACLR supports any standard LoRA structure and achieves higher Identity Alignment after merging (.828 vs .717).
- **vs Orthogonal Adaptation**: OA requires refinetuning each LoRA on the original data (~120 minutes) and suffers from feature shifting under high concept counts. LoRACLR takes only 5 minutes and operates without original data.
- **vs ZipLoRA**: ZipLoRA only supports binary merging of style and content, and cannot handle merging multiple content LoRAs. LoRACLR has no such limitation.
- **New Paradigm of Contrastive Learning + Model Merging**: This work proposes a new model merging paradigm—instead of handcrafted merging rules (weighted average, gradient fusion), learning-based methods (contrastive loss) are adopted to automatically identify the optimal merging point.

## Rating

- Novelty: ⭐⭐⭐⭐ Applying contrastive learning to LoRA merging is innovative, and the delta-based strategy is simple and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative, qualitative, user studies, and ablation analysis are comprehensively provided, though validation on newer models is currently missing.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, accompanied by abundant illustrations.
- Value: ⭐⭐⭐⭐⭐ Highly practical, directly addressing real needs of community users, merging 12 concepts in just 5 minutes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](../../ICCV2025/image_generation/contrastive_flow_matching.md)
- [\[CVPR 2025\] DreamRelation: Bridging Customization and Relation Generation](dreamrelation_bridging_customization_and_relation_generation.md)
- [\[NeurIPS 2025\] Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](../../NeurIPS2025/image_generation/towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)
- [\[CVPR 2025\] Everything to the Synthetic: Diffusion-driven Test-time Adaptation via Synthetic-Domain Alignment](everything_to_the_synthetic_diffusion-driven_test-time_adaptation_via_synthetic-.md)
- [\[ICML 2025\] IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models](../../ICML2025/image_generation/intlora_integral_low-rank_adaptation_of_quantized_diffusion_models.md)

</div>

<!-- RELATED:END -->
