---
title: >-
  [Paper Note] DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models
description: >-
  [AAAI 2026][Image Generation][Diffusion Models] This paper proposes DogFit, which internalizes Domain Guidance (DoG) into the fine-tuning loss of diffusion models, enabling the model to learn the guidance direction during training. At inference time, a controllable fidelity–diversity trade-off is achieved without double forward passes, surpassing the state-of-the-art guidance methods on 6 target domains with half the sampling TFLOPS.
tags:
  - AAAI 2026
  - Image Generation
  - Diffusion Models
  - Transfer Learning
  - Guidance Mechanism
  - Domain Adaptation
  - Efficient Inference
date: 2026-05-08
content_hash: b0159e1a3cbf89f5
---

# DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models

**Conference**: AAAI 2026
**arXiv**: [2508.05685](https://arxiv.org/abs/2508.05685)
**Code**: [GitHub](https://github.com/yaramohamadi/DogFit)
**Area**: Image Generation
**Keywords**: Diffusion Models, Transfer Learning, Guidance Mechanism, Domain Adaptation, Efficient Inference

## TL;DR
This paper proposes DogFit, which internalizes Domain Guidance (DoG) into the fine-tuning loss of diffusion models, enabling the model to learn the guidance direction during training. At inference time, a controllable fidelity–diversity trade-off is achieved without double forward passes, surpassing the state-of-the-art guidance methods on 6 target domains with half the sampling TFLOPS.

## Background & Motivation

**State of the Field**: Transferring diffusion models to small-scale target domains is prone to overfitting. Guidance methods such as CFG and DoG can improve generation quality, but require double forward passes at inference time (2× computational overhead). MG internalizes guidance during training but inherits the limitations of CFG in transfer learning settings.

**Limitations of Prior Work**:
- The unconditional noise estimator of CFG underfits on small target domains, resulting in inaccurate guidance directions.
- DoG employs the source model for marginal estimation, which is more effective, but doubles inference cost.
- MG incurs no inference overhead but hard-codes the guidance strength, precluding runtime control.

**Root Cause**: Can a guidance mechanism be designed that leverages the source model's strong marginal estimation, incurs no inference overhead, and still supports controllable guidance strength at inference time?

**Starting Point**: Inject the domain guidance offset from DoG into the training loss, using the source model (rather than the unconditional branch of the target model) to supply the guidance direction, while encoding the guidance strength $w$ as an additional model input.

**Core Idea**: Internalize domain guidance direction from the source model into the training loss + encode guidance strength as a model input for inference-time controllability + late-start/cut-off scheduling to improve training stability.

## Method

### Overall Architecture
A domain guidance offset is injected into the standard diffusion fine-tuning objective: $\epsilon' = \epsilon + (w-1) \cdot \text{sg}(\epsilon_\theta(x_t|c, \mathcal{D}^T) - \epsilon_{\theta_0}(x_t))$. The model learns to directly predict the guided noise direction, enabling single-pass inference.

### Key Designs

1. **Domain Guidance Offset Injection**:

    - Function: Transfers the DoG guidance signal from inference time into the training objective.
    - Mechanism: The training loss becomes $\mathcal{L} = \|\epsilon_\theta(x_t|c) - \epsilon'\|^2$, where $\epsilon'$ incorporates the guidance offset between the fine-tuned model and the source model.
    - Design Motivation: The source model, pre-trained on large-scale data, provides more reliable marginal estimation than the unconditional model trained on the target domain.

2. **Controllable Guidance Strength**:

    - Function: Enables dynamic adjustment of the fidelity–diversity trade-off at inference time.
    - Mechanism: $w$ is encoded as an additional conditional input to the model; during training, $w$ is sampled over a specified range.
    - Design Motivation: MG hard-codes $w$ and cannot be controlled at inference time. DogFit addresses this by conditioning on $w$, at the cost of only a lightweight embedding layer.

3. **Training Schedule Strategies**:

    - **Late-start**: Delays the injection of guidance until the model has learned a sufficiently stable representation of the target domain.
    - **Cut-off**: Applies guidance only in later denoising steps, where fine-grained domain features predominantly appear.
    - Design Motivation: Empirical findings show that applying guidance too early or throughout the entire training leads to instability.

### Loss & Training
DogFit loss: $\mathcal{L} = \|\epsilon_\theta(x_t|c,w) - [\epsilon + (w-1) \cdot \text{sg}(\epsilon_\theta(x_t|c) - \epsilon_{\theta_0}(x_t))]\|^2$. Validated on DiT/XL-2 and SiT/XL-2.

## Key Experimental Results

### Main Results (DiT/XL-2, Average over 6 Target Domains)

| Method | FID↓ | FD_DINOv2↓ | Sampling TFLOPS |
|--------|------|-----------|----------------|
| Fine-tuning | 19.14 | 461.45 | 366 |
| +CFG | 14.46 | 311.03 | 732 (2×) |
| +DoG | 13.09 | 245.31 | 732 (2×) |
| MG | 14.13 | 312.78 | 366 |
| **DogFit** | **12.34** | **246.01** | **366** |

DogFit surpasses DoG in FID while halving the sampling TFLOPS.

### Ablation Study
- Late-start improves FID by approximately 1–2 over full-training guidance.
- Cut-off further improves performance on SiT.
- The controllable guidance strength variant (DogFit + Control) incurs only a marginal performance drop compared to the fixed variant.

### Key Findings
- The source model indeed provides stronger marginal estimation than the unconditional model trained on the target domain.
- Internalizing guidance at training time not only reduces computation but actually outperforms inference-time guidance, owing to more thorough training-time optimization.
- The method is effective under both labeled and unlabeled transfer settings.

## Highlights & Insights
- The paradigm of **"learn the guidance direction during training, apply it for free at inference"** is practically appealing—effectively shifting inference cost to the training phase.
- The insight of **using the source model rather than the target model for marginal estimation** is theoretically well-motivated: the unconditional branch of the target model inevitably underfits on small domains.
- **Conditioning on guidance strength as a model input** unifies the advantages of both MG and DoG.

## Limitations & Future Work
- Source model weights must be retained during training to compute the guidance offset, incurring additional GPU memory overhead.
- Validation is limited to class-conditional generation; extension to text-conditional diffusion models (e.g., Stable Diffusion) remains to be explored.
- The timing of late-start and cut-off requires hyperparameter tuning.

## Related Work & Insights
- **vs. DoG**: Comparable performance with 2× faster inference.
- **vs. MG**: MG uses CFG-based guidance direction (weaker); DogFit uses the DoG direction (stronger). MG does not support runtime control; DogFit does.
- **vs. CFG distillation**: CFG distillation requires an additional training stage and architectural modifications, whereas DogFit is directly integrated into the fine-tuning objective.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of internalizing domain guidance into training is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 datasets + 2 backbones + multiple guidance baselines + comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear illustrations and thorough method comparisons.
- Value: ⭐⭐⭐⭐ A practical acceleration solution for transfer learning of diffusion models.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning](self-npo_data-free_diffusion_model_enhancement_via_truncated_diffusion_fine-tuni.md)
- [\[NeurIPS 2025\] DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models](../../NeurIPS2025/image_generation/deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)
- [\[AAAI 2026\] RelaCtrl: Relevance-Guided Efficient Control for Diffusion Transformers](relactrl_relevance-guided_efficient_control_for_diffusion_transformers.md)
- [\[AAAI 2026\] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models](melodia_training-free_music_editing_guided_by_attention_probing_in_diffusion_mod.md)
- [\[CVPR 2026\] Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping](../../CVPR2026/image_generation/memory-efficient_fine-tuning_diffusion_transformers_via_dynamic_patch_sampling_a.md)

<!-- RELATED:END -->
