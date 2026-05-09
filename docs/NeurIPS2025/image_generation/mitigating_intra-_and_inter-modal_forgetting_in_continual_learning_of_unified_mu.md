---
title: >-
  [Paper Note] Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models
description: >-
  [NeurIPS 2025][Image Generation][Unified Multimodal Generation Models] This paper proposes Modality-Decoupled Experts (MoDE), which decouples text and image adapters into independent T-MoE and V-Adapter subspaces, combined with knowledge distillation, to simultaneously mitigate intra-modal and inter-modal forgetting in continual instruction tuning of unified multimodal generation models.
tags:
  - NeurIPS 2025
  - Image Generation
  - Unified Multimodal Generation Models
  - Inter-modal Forgetting
  - Intra-modal Forgetting
  - LoRA Mixture of Experts
  - Knowledge Distillation
date: 2026-05-08
content_hash: 2f59b8f3815afba4
---

# Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models

**Conference**: NeurIPS 2025
**arXiv**: [2512.03125](https://arxiv.org/abs/2512.03125)
**Code**: [GitHub](https://github.com/Christina200/MoDE-official)
**Area**: Image Generation
**Keywords**: Unified Multimodal Generation Models, Inter-modal Forgetting, Intra-modal Forgetting, LoRA Mixture of Experts, Knowledge Distillation

## TL;DR

This paper proposes Modality-Decoupled Experts (MoDE), which decouples text and image adapters into independent T-MoE and V-Adapter subspaces, combined with knowledge distillation, to simultaneously mitigate intra-modal and inter-modal forgetting in continual instruction tuning of unified multimodal generation models.

## Background & Motivation

Unified Multimodal Generation Models (UMGMs, e.g., Chameleon, Janus-Pro) integrate visual understanding and image generation within a single autoregressive framework. However, they suffer from severe catastrophic forgetting when continually learning new tasks.

Existing continual learning research primarily focuses on **intra-modal forgetting**: forgetting prior tasks within the same output modality when learning new ones (e.g., sequentially learning multiple VQA tasks). UMGMs introduce a novel and underexplored challenge — **inter-modal forgetting**: fine-tuning on text understanding tasks degrades the model's image generation capability.

The authors empirically verify this phenomenon: after sequentially fine-tuning Chameleon on three VQA datasets, image generation quality degrades significantly and text-image alignment deteriorates. Theoretically, this stems from **modal gradient conflict**:

**Definition**: The gradient of the text generation loss $g_t = \nabla_\theta \mathcal{L}_t$ and the gradient of the image generation loss $g_v = \nabla_\theta \mathcal{L}_v$ conflict when $\langle g_v, g_t \rangle < 0$. In this case, the SGD update $\theta \leftarrow \theta - \eta g_t$ on the text task increases the visual loss: $\Delta \mathcal{L}_v = -\eta \langle g_t, g_v \rangle + \frac{\eta^2}{2} g_t^\top H_v g_t$, where the first term is positive under gradient conflict, leading to visual performance degradation.

Existing methods (e.g., CL-MoE, Model Tailor) cannot effectively address both types of forgetting simultaneously.

## Method

### Overall Architecture

MoDE integrates two types of lightweight adapters on top of frozen UMGM linear layers: T-MoE (Mixture-of-Experts LoRA) for text tokens and V-Adapter (a single LoRA) for image tokens, achieving modality decoupling. During continual instruction tuning, only MoDE components are trained while the original UMGM parameters remain frozen.

### Key Designs

1. **V-Adapter (Visual LoRA Adapter)**

   A LoRA module dedicated to processing image tokens for both visual understanding and image generation. It adopts the standard LoRA formulation:
   $\Delta W = \frac{\alpha}{r} BA$

   The modified linear transformation for input token representation $h$ is: $f(h) = hW^\top + \frac{\alpha}{r}hA^\top B^\top$. **Design Motivation**: Isolating image-related parameter updates in an independent subspace prevents interference with text updates.

2. **T-MoE (Text LoRA Mixture of Experts)**

   A MoE-LoRA module for text tokens, routing inputs to multiple experts via a router:
   $g_j(x) = \text{softmax}(xW_g)_j$
   $f(h) = hW^\top + \frac{\alpha}{r}\sum_{j=1}^{n} g_j(x) hA_j^\top B_j^\top$

   where $n$ is the number of experts. The routing mechanism enables different tasks to automatically activate different expert combinations, thereby mitigating intra-modal forgetting. **Motivation**: Expert diversity enables task-specific adaptation and avoids parameter overwriting.

3. **Theoretical Guarantee of Modality Decoupling**

   MoDE provably reduces inter-modal interference from $\mathcal{O}(\eta)$ to $\mathcal{O}(\eta^2)$:

   When T-MoE parameters $\phi$ are updated, the effect on the visual loss is:
   $\Delta \mathcal{L}_v = \frac{\eta^2}{2} \lambda_{\max}(\nabla^2_{\phi\phi}\mathcal{L}_v) \|\nabla_\phi \mathcal{L}_t\|^2$

   Since $\nabla_\phi \mathcal{L}_v = 0$ (T-MoE does not directly process visual tokens), the first-order conflict term vanishes, leaving only a second-order term.

### Loss & Training

- **T-MoE Training**: Standard cross-entropy loss $\mathcal{L}_{\text{CE}} = -\frac{1}{L}\sum_{i=1}^{L}\log p_\theta(X_i^{ans} | X^{img}, X^{ins}, X_{<i}^{ans})$

- **V-Adapter Training**: A weighted combination of cross-entropy loss and knowledge distillation loss:
  $$\mathcal{L}_{\text{V-Adapter}} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{KD}}$$

  The KD loss aligns softened logits between the teacher (frozen pretrained model) and student (V-Adapter-augmented model):
  $$\mathcal{L}_{\text{KD}} = \beta^2 \sum_{i=1}^{L} D_{\text{KL}}(\text{Softmax}(z_i^T/\beta) \| \text{Softmax}(z_i^S/\beta))$$

  Reference data is sampled from LAION-5B, with $\lambda=0.3$.

## Key Experimental Results

### Main Results

Sequential fine-tuning on Chameleon across 5 datasets (ScienceQA→TextVQA→ImageNet→GQA→VizWiz):

| Method | Text Align↑ | Image Align↑ | FID↓ | Understanding Acc.↑ | Forgetting↓ | Overall Δ↓ |
|--------|------------|-------------|------|---------------------|-------------|------------|
| Zero-shot | 0.2592 | 0.5205 | 52.13 | 22.48 | - | 34.84 |
| Seq LoRA | 0.2162 | 0.5150 | 56.12 | 28.43 | 35.33 | 28.57 |
| MoELoRA | 0.2248 | 0.5095 | 65.16 | 33.01 | 30.77 | 24.31 |
| CL-MoE | 0.2081 | 0.5150 | 65.87 | 32.86 | 30.95 | 24.46 |
| **MoDE** | **0.2458** | **0.5170** | **53.74** | **33.47** | **25.99** | **22.78** |

### Ablation Study

| Configuration | FID↓ | Understanding Acc.↑ | Forgetting↓ | Notes |
|--------------|------|---------------------|-------------|-------|
| Chameleon (original) | 52.13 | 22.48 | - | Baseline |
| + T-MoE LoRA | 51.28 | 33.03 | 28.65 | Text experts only; image fully preserved |
| + MoDE w/o KD | 54.61 | 33.07 | 26.49 | Without distillation; image quality drops |
| + MoDE (full) | 53.74 | 33.47 | 25.99 | KD effectively balances both capabilities |

### Key Findings

- MoDE is the only method capable of simultaneously mitigating intra-modal and inter-modal forgetting: FID 53.74 is close to the pretrained baseline of 52.13, while achieving the best understanding accuracy of 33.47.
- Although MoELoRA and CL-MoE improve understanding, shared parameters lead to severe image generation degradation (FID > 65 vs. pretrained 52.13).
- DualPrompt preserves image generation quality but yields insufficient understanding capability.
- Knowledge distillation is critical for maintaining image generation quality (FID 54.61 w/o KD vs. 53.74 with full MoDE).

## Highlights & Insights

- This work is the first to systematically identify and study inter-modal forgetting in UMGMs, filling a gap in continual learning research.
- The theoretical analysis of gradient conflict is concise and compelling, directly motivating the modality-decoupled design.
- MoDE's decoupled design provably reduces inter-modal interference by one order of magnitude ($\mathcal{O}(\eta) \to \mathcal{O}(\eta^2)$).
- The method is lightweight and plug-and-play, applicable to various Transformer-based UMGM architectures.

## Limitations & Future Work

- The validated task sequences are relatively short (5 tasks); performance under longer sequences remains to be verified.
- V-Adapter uses a single LoRA, which may also suffer from forgetting as image-related tasks accumulate.
- Knowledge distillation incurs additional storage overhead and requires reference data.
- Forward transfer across tasks is not explored.

## Related Work & Insights

- The gradient conflict analysis framework can be extended to other multi-task and multimodal learning scenarios.
- The decoupling design principle (independent parameter spaces for different modalities) provides guidance for the continual evolution of unified models.
- The strategy of using KD as a "memory anchor" can be combined with other continual learning methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The problem formulation of inter-modal forgetting is novel and significant, though the solution (MoE + KD) is a relatively mature technical combination.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple baselines, ablation studies, and qualitative results are comprehensive, though dataset and model diversity could be richer.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical chain from problem definition → theoretical analysis → method design is exceptionally clear.
- **Value**: ⭐⭐⭐⭐ Provides an important benchmark and solution for continual learning in UMGMs.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](../../ICCV2025/image_generation/joint_diffusion_models_in_continual_learning.md)
- [\[NeurIPS 2025\] Show-o2: Improved Native Unified Multimodal Models](show-o2_improved_native_unified_multimodal_models.md)
- [\[ICLR 2026\] Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models](../../ICLR2026/image_generation/uni-x_mitigating_modality_conflict_with_a_two-end-separated_architecture_for_uni.md)
- [\[NeurIPS 2025\] Mitigating Sexual Content Generation via Embedding Distortion in Text-conditioned Diffusion Models](mitigating_sexual_content_generation_via_embedding_distortion_in_text-conditione.md)

<!-- RELATED:END -->
