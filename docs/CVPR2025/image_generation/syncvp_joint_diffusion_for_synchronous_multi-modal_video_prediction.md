---
title: >-
  [Paper Note] SyncVP: Joint Diffusion for Synchronous Multi-Modal Video Prediction
description: >-
  [CVPR 2025][Image Generation][Multi-Modal Video Prediction] Proposes the SyncVP multi-modal video prediction framework, which synchronously generates future RGB and depth frames using a dual-branch diffusion model coupled with highly efficient spatio-temporal cross-modal attention. By utilizing innovative shared noise and cross-modality guidance training strategies, it achieves SOTA performance on Cityscapes while supporting partial-modality inputs.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Multi-Modal Video Prediction"
  - "Joint Diffusion"
  - "Cross-Modal Attention"
  - "Noise Sharing"
  - "Depth Prediction"
date: 2026-05-08
content_hash: 9a975bd0f69fe2f2
---

# SyncVP: Joint Diffusion for Synchronous Multi-Modal Video Prediction

**Conference**: CVPR 2025  
**arXiv**: [2503.18933](https://arxiv.org/abs/2503.18933)  
**Code**: [https://SyncVp.github.io/](https://SyncVp.github.io/)  
**Area**: Image Generation  
**Keywords**: Multi-Modal Video Prediction, Joint Diffusion, Cross-Modal Attention, Noise Sharing, Depth Prediction

## TL;DR
Proposes the SyncVP multi-modal video prediction framework, which synchronously generates future RGB and depth frames using a dual-branch diffusion model coupled with highly efficient spatio-temporal cross-modal attention. By utilizing innovative shared noise and cross-modality guidance training strategies, it achieves SOTA performance on Cityscapes while supporting partial-modality inputs.

## Background & Motivation
1. **Background**: Video prediction is a critical task for decision-making systems such as autonomous driving and weather forecasting. Existing methods primarily predict single-modality (RGB) future frames, with diffusion models serving as the dominant paradigm in this field.
2. **Limitations of Prior Work**: Predicting RGB frames alone cannot fully capture the complexity of real-world physical dynamics. Multi-modal information (depth, semantics) can provide complementary cues, but existing multi-modal generative research (such as LDM3D, MM-Diffusion) is not tailored for video prediction, and simply concatenating modalities yields poor performance.
3. **Key Challenge**: Modalities like RGB and depth possess vastly different characteristics (RGB contains rich appearance details, whereas depth encodes geometric structures). A single network struggles to simultaneously learn the complex distribution of both modalities.
4. **Goal**: Build a scalable multi-modal video prediction framework that supports both dual-modality and partial-modality inputs.
5. **Key Insight**: Initialize the framework with two pre-trained modality-specific diffusion models and establish cross-modal information exchange via a lightweight cross-attention mechanism.
6. **Core Idea**: Dual-branch diffusion + factorized spatio-temporal cross-modal attention + shared noise + cross-modality guidance training.

## Method

### Overall Architecture
SyncVP consists of two branches (RGB and depth), with each branch utilizing a lightweight UNet based on the PVDM architecture. The two branches are connected at the deepest layer through the **Spatio-Temporal Cross-Modal Attention (STCA)** module. The training process operates in two stages: first, independently pre-training each modality-specific diffusion model; then, jointly fine-tuning them using shared noise. During inference, the framework generates future frames autoregressively in units of 8 frames.

### Key Designs

1. **Spatio-Temporal Cross-Modal Attention (STCA)**

    - **Function**: Efficiently exchanges spatio-temporal information between RGB and depth features.
    - **Mechanism**: The latent space of PVDM is factorized into spatial vectors $z^s$, height-temporal vectors $z^h$, and width-temporal vectors $z^w$. Instead of performing a computationally expensive full cross-attention over the entire latent space, STCA executes cross-modality attention on each of the three vector pairs separately, sharing a single attention matrix $A = Q_R Q_D^\top / \sqrt{d_k}$. RGB features extract depth information via $\text{Softmax}(A) \cdot V_D$, while depth features extract RGB information via $\text{Softmax}(A^\top) \cdot V_R$. At a resolution of 64×64, this requires only 37% of the computation compared to standard full cross-attention.
    - **Design Motivation**: Full-resolution cross-attention is computationally prohibitive for video sequences. Factoring the spatio-temporal dimensions leverages the structural characteristics of the PVDM latent space.

2. **Shared Noise**

    - **Function**: Ensures synchronization of the multi-modal diffusion processes and accelerates training convergence.
    - **Mechanism**: During the forward diffusion process, the exact same noise sample $\epsilon$ is applied to both modalities. By adding identical noise $\epsilon \sim \mathcal{N}(0, I)$ to the latents of both RGB and depth, the two denoising networks are forced to learn the same inverted "noise-to-data" mapping. The target noise term in the loss functions of both branches is identical: $\mathcal{L}_M = \|\epsilon - \epsilon_{\theta_R}(\cdot)\|_2^2 + \|\epsilon - \epsilon_{\theta_D}(\cdot)\|_2^2$.
    - **Design Motivation**: Since both modalities describe the same physical scene, sharing the noise coordinates forces the networks to learn consistent reverse trajectories. This empirically yields significant improvements in convergence speed and quality of conditional generation.

3. **Cross-Modality Guidance**

    - **Function**: Enables the model to predict complete multi-modal outputs even when only a subset of the modalities is provided as input.
    - **Mechanism**: Analogous to classifier-free guidance, during training, dual-modality conditioning is provided with a 50% probability, while either only RGB or only depth conditioning is provided with a 25% probability each (with the missing modality replaced by zeros). The network simultaneously learns three distributions: $p(r_x, d_x | r_c, d_c)$, $p(r_x, d_x | 0, d_c)$, and $p(r_x, d_x | r_c, 0)$.
    - **Design Motivation**: In real-world scenarios, some sensors may be unavailable (such as depth sensor failure in autonomous driving systems). The model must be robust enough to handle partial inputs.

### Loss & Training
- Independent pre-training for each modality: Standard DDPM loss.
- Joint fine-tuning: $\mathcal{L}_M = \mathbb{E}[\|\epsilon - \epsilon_{\theta_R}(\cdot)\|_2^2 + \|\epsilon - \epsilon_{\theta_D}(\cdot)\|_2^2]$
- DDPM 1000 steps for training, DDIM 100 steps for inference.
- Only 58M parameters per branch (11% of PVDM-L).

## Key Experimental Results

### Main Results (Cityscapes, 2→28)

| Method | FVD↓ | SSIM↑ | LPIPS↓ |
|------|------|-------|--------|
| ExtDM-K4 | 121.3 | 0.745 | 108 |
| STDiff | 107.31 | 0.658 | 136.26 |
| SyncVP (w/o depth) | 97.31 | 0.652 | 161.1 |
| **SyncVP** | **84** | 0.649 | 159.7 |

### Ablation Study

| Configuration | FVD↓ | Description |
|------|------|------|
| Full SyncVP | 84 | Complete model |
| Simple modality concatenation | ~130 | Naive solution fails |
| w/o Shared Noise | ~110 | Slow convergence, poor consistency |
| Standard CA instead of STCA | ~95 | Lower efficiency and slightly worse performance |
| w/o Cross-Modality Guidance | ~90 | Cannot handle partial inputs |

### Key Findings
- Shared noise is the most critical design: removing it degrades FVD by ~30%.
- Multi-modal information (depth) indeed improves the quality of RGB prediction (FVD decreases from 97 to 84).
- STCA is 50% more computationally efficient than standard cross-attention (CA) while delivering better performance.
- The generalizability of the framework to other modalities is verified on SYNTHIA (semantics) and ERA5-Land (climate data).
- Even with single-modality inputs, SyncVP outperforms single-modality SOTA methods.

## Highlights & Insights
- **Shared noise** is a simple yet profound insight: different modal representations of the same scene should be reconstructed starting from the same "chaotic state," which dramatically reduces the difficulty of learning.
- **Spatio-temporal factorized cross-modal attention** leverages the structural properties of the PVDM latent space, maintaining high performance while significantly reducing computational complexity.
- **Cross-modality guidance** cleverly adapts the concepts of classifier-free guidance for application in scenarios with missing modalities.

## Limitations & Future Work
- Only the combination of two modalities was verified; scalability to three or more modalities remains to be tested.
- Error accumulation may occur during autoregressive generation of long videos.
- Current verification is limited to low resolutions (64×64/128×128).
- Future work can explore integration with large-scale video diffusion models.

## Related Work & Insights
- **vs PVDM**: Single-modality video prediction baseline. SyncVP adds a cross-modal branch on top of it, outperforming it even without using depth information.
- **vs LDM3D**: Joint RGB+D generation in the image domain (via simple channel concatenation), which is not suitable for video prediction.
- **vs MM-Diffusion**: Joint audio-video generation using fully-connected cross-attention; SyncVP's STCA is significantly more efficient.

## Rating
- Novelty: ⭐⭐⭐⭐ First multi-modal video prediction diffusion framework, featuring an ingenious shared noise strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets + multiple modality types + exhaustive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear framework presentation and logically designed ablation scenarios.
- Value: ⭐⭐⭐⭐ Opens up a new direction for multi-modal video prediction with a highly generalizable and scalable framework design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction](pqpp_a_joint_benchmark_for_text-to-image_prompt_and_query_performance_prediction.md)
- [\[CVPR 2025\] DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation](diffsensei_bridging_multi-modal_llms_and_diffusion_models_for_customized_manga_g.md)
- [\[CVPR 2025\] Generative Modeling of Class Probability for Multi-Modal Representation Learning](generative_modeling_of_class_probability_for_multi_modal_representation_learning.md)
- [\[CVPR 2025\] OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows](omniflow_any-to-any_generation_with_multi-modal_rectified_flows.md)
- [\[CVPR 2025\] MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling](mmar_towards_lossless_multi-modal_auto-regressive_probabilistic_modeling.md)

</div>

<!-- RELATED:END -->
