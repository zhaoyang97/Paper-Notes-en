---
title: >-
  [Paper Note] Show-o2: Improved Native Unified Multimodal Models
description: >-
  [NeurIPS 2025][Image Generation][Unified multimodal models] This paper presents Show-o2, a natively unified multimodal model built upon autoregressive modeling and Flow Matching. By constructing unified visual representa…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Unified multimodal models"
  - "autoregressive modeling"
  - "Flow Matching"
  - "3D causal VAE"
  - "visual understanding and generation"
date: 2026-05-08
content_hash: 77eb2f5d1e6e94d2
---

# Show-o2: Improved Native Unified Multimodal Models

**Conference**: NeurIPS 2025  
**arXiv**: [2506.15564](https://arxiv.org/abs/2506.15564)  
**Code**: [GitHub](https://github.com/showlab/Show-o)  
**Area**: Unified Multimodal Models / Image Generation  
**Keywords**: Unified multimodal models, autoregressive modeling, Flow Matching, 3D causal VAE, visual understanding and generation

## TL;DR

This paper presents Show-o2, a natively unified multimodal model built upon autoregressive modeling and Flow Matching. By constructing unified visual representations in a 3D causal VAE space via dual-path spatial(-temporal) fusion, Show-o2 supports multimodal understanding and generation across text, images, and video, with a two-stage training strategy that effectively preserves language knowledge.

## Background & Motivation

Large multimodal models (LMMs) and visual generation models have achieved impressive performance in visual understanding and image/video generation, respectively. Unified multimodal models (UMMs) attempt to integrate both capabilities within a single model. Existing approaches face the following challenges:

**Unified visual representation**: Multimodal understanding requires high-level semantic features (e.g., CLIP), while generation requires low-level structural details (e.g., VAE latents). These demands are fundamentally different. Existing methods either adopt a unified representation (Chameleon, Show-o) at the cost of one capability, or use decoupled representations (Janus series) at the cost of native unification.

**Scalability to images and video**: Most UMMs support only text and images; native support for video modality remains largely unexplored.

**Knowledge forgetting during training**: When training UMMs from a pretrained LLM, learning visual generation often leads to degradation of language knowledge unless large-scale text corpora are incorporated.

Show-o2's core innovation lies in constructing unified visual representations through a dual-path fusion mechanism in the 3D causal VAE space, supporting both images and video, while avoiding knowledge forgetting through a two-stage training strategy.

## Method

### Overall Architecture

Given interleaved text/image/video inputs, text is converted to embeddings via a tokenizer, and visual inputs are encoded into visual latents via a 3D causal VAE encoder. The visual latents are processed through dual-path extraction and spatial(-temporal) fusion to produce unified visual representations. Text embeddings and unified visual representations are concatenated into a sequence and fed into the base language model. A language head predicts text tokens via autoregressive modeling, while a Flow head generates images/video via Flow Matching. An omni-attention mechanism is employed: causal attention along the sequence dimension and full attention within visual representations.

### Key Designs

1. **Unified Visual Representation (Dual-Path Spatial-Temporal Fusion)**

   A 3D causal VAE encoder extracts visual latents, which are then processed through a dual-path architecture:

   - **Semantic path $\mathcal{S}(\cdot)$**: Shared ViT blocks from SigLIP (with an added 2×2 patch embedding) extract high-level semantic information. A pre-distillation step enables this path to extract semantic features from both clean and noisy visual latents:

   $\mathcal{L}_{\text{distill}} = -\frac{1}{n}\sum\log\text{sim}(\mathcal{S}(\mathbf{x}_t), \text{SigLIP}(\mathbf{X}))$

   where $\mathbf{x}_t = t \cdot \mathbf{x}_1 + (1-t) \cdot \mathbf{x}_0$, $t \sim [0,1]$. After training, the cosine similarity between semantic features extracted from clean latents and original SigLIP features reaches approximately 0.9.

   - **Projector $\mathcal{P}(\cdot)$**: A simple 2D patch embedding layer that retains complete low-level structural details.

   The two feature streams are merged via a spatial(-temporal) fusion mechanism:

   $\mathbf{u} = \text{STF}(\mathcal{S}(\mathbf{x}_t), \mathcal{P}(\mathbf{x}_t))$

   Concretely, the features are concatenated along the channel dimension, followed by RMSNorm and a two-layer MLP. In the video setting, semantic and low-level features are naturally aligned along the temporal dimension.

   **Design Motivation**: Understanding requires CLIP-level semantics, while generation requires VAE-level details. The dual-path design satisfies both requirements simultaneously within a unified latent space, and is naturally scalable to both images and video.

2. **Flow Head**

   A dedicated Flow head, consisting of several transformer layers with adaLN-Zero timestep modulation (analogous to DiT), is added alongside the language head to predict the velocity $\mathbf{v}_t = d\mathbf{x}_t / dt$. The training objective is:

   $\mathcal{L} = \alpha\mathcal{L}_{\text{NTP}} + \mathcal{L}_{\text{FM}}$

   where $\mathcal{L}_{\text{NTP}}$ denotes the next-token prediction loss and $\mathcal{L}_{\text{FM}}$ denotes the flow matching loss.

3. **Two-Stage Training Strategy**

    - **Stage 1**: Only the projector, spatial(-temporal) fusion module, and Flow head are trained, using approximately 66M image-text pairs, with interleaved and video data progressively introduced. The language model parameters are frozen to preserve language knowledge. $\alpha=0.2$.
    - **Stage 2**: Full model fine-tuning (excluding the VAE), using 9M high-quality understanding instruction data and 16M high-quality generation data. $\alpha=1.0$.

   **Design Motivation**: Training the visual generation components first before global fine-tuning eliminates dependence on large-scale text corpora.

   **Model Scaling**: The pretrained Flow head from the 1.5B model is transferred to the 7B model via a lightweight MLP transformation that aligns hidden dimensions, enabling rapid adaptation.

### Loss & Training

- Semantic path pre-distillation: 200K iterations, batch size 512, cosine schedule with lr 2e-5
- Stage 1 (1.5B): 150K iterations, 64 H100 GPUs, approximately 1.5 days
- Stage 2: approximately 35K iterations, approximately 15 hours
- 7B model: 128 H100 GPUs, approximately 2.5 days
- Generation data captions are dropped with probability 0.1 to enable classifier-free guidance

## Key Experimental Results

### Multimodal Understanding (Image)

| Model | Params | MME↑ | GQA↑ | SEED↑ | MMB↑ | MMMU↑ | MMStar↑ | AI2D↑ |
|-------|--------|------|------|-------|------|-------|---------|-------|
| Janus-Pro | 1.5B | 1444.0 | 59.3 | 68.3 | 75.5 | 36.3 | - | - |
| Show-o | 1.3B | 1097.2 | 58.0 | 51.5 | - | 27.4 | - | - |
| **Show-o2** | **1.5B** | **1450.9** | **60.0** | 65.6 | 67.4 | **37.1** | 43.4 | 69.0 |
| Janus-Pro | 7B | 1567.1 | 62.0 | 72.1 | 79.2 | 41.0 | - | - |
| TokenFlow-XL* | 14B | 1551.1 | 62.5 | 72.6 | 76.8 | 43.2 | - | 75.9 |
| **Show-o2** | **7B** | **1620.5** | **63.1** | 69.8 | 79.3 | **48.9** | **56.6** | **78.6** |

### Image Generation (GenEval)

| Model | Params | Training Data | Single Obj | Two Obj | Counting | Colors | Position | Color Attr | Overall↑ |
|-------|--------|--------------|-----------|---------|----------|--------|----------|-----------|---------|
| Janus-Pro | 7B | 144M | 0.99 | 0.89 | 0.59 | 0.90 | 0.79 | 0.66 | 0.80 |
| BAGEL | 14B | 1600M | 0.98 | 0.95 | 0.84 | 0.95 | 0.78 | 0.77 | 0.88 |
| **Show-o2** | **1.5B** | **66M** | 0.99 | 0.86 | 0.55 | 0.86 | 0.46 | 0.63 | 0.73 |
| **Show-o2** | **7B** | **66M** | 1.00 | 0.87 | 0.58 | 0.92 | 0.52 | 0.62 | **0.76** |

### Key Findings

1. The 7B Show-o2 surpasses Janus-Pro at the same scale and TokenFlow-XL at 14B on multimodal understanding, achieving MME of 1620.5 and MMMU of 48.9.
2. On generation tasks, Show-o2 trained on 66M data remains competitive with Janus-Pro trained on 144M data (0.76 vs. 0.80), though a gap remains compared to BAGEL trained on 1600M data (0.88).
3. The cosine similarity of semantic features after pre-distillation reaches 0.9, validating the feasibility of extracting CLIP-level semantics from VAE latent space.
4. The two-stage training strategy effectively preserves language knowledge without requiring large-scale text corpora.
5. Video understanding also performs well after subsequent fine-tuning (7B model: ActNet-QA 56.4, VideoMME 57.4/60.9).

## Highlights & Insights

- The **dual-path unified visual representation** is the central contribution: encoding both semantics and structural details within the VAE latent space eliminates the redundancy of separate CLIP and VAE encoders.
- Pre-distilling the semantic path into the VAE space is an elegant design choice — enabling the same set of latents to support both understanding and generation.
- Reusing the Flow head from the 1.5B model when scaling to 7B reduces large-model training cost.
- Native support for text, image, and video within a single unified model remains rare among current UMMs.

## Limitations & Future Work

- Image generation quality (GenEval 0.76) still lags behind state-of-the-art dedicated generation models (BAGEL 0.88, Mogao 0.89).
- Due to computational constraints, the 7B model did not incorporate interleaved or video training data; video generation capability is demonstrated only in the 1.5B model.
- The semantic path has a strong dependency on SigLIP, limiting flexibility in substituting the base visual encoder.
- The effect of scaling image resolution (432→1024) is not reported in detail.

## Related Work & Insights

- The dual-path fusion mechanism offers insights into how to simultaneously preserve semantic and structural information within a single latent space.
- The two-stage training approach (first freezing the LLM to learn visual generation, then fine-tuning globally) is a practical strategy for mitigating catastrophic forgetting.
- Comparisons with Transfusion, Chameleon, and related methods suggest that AR + Flow Matching constitutes a competitive hybrid paradigm.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The dual-path unified representation and semantic path distillation constitute a genuinely novel design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple benchmarks covering both understanding and generation are evaluated, though generation-side comparisons are somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ Architecture descriptions are clear and training details are comprehensive.
- **Value**: ⭐⭐⭐⭐ Provides a scalable reference design for natively unified multimodal models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[NeurIPS 2025\] Improved Training Technique for Shortcut Models (iSM)](improved_training_technique_for_shortcut_models.md)
- [\[NeurIPS 2025\] Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models](mitigating_intra-_and_inter-modal_forgetting_in_continual_learning_of_unified_mu.md)
- [\[CVPR 2026\] Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](../../CVPR2026/image_generation/learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)
- [\[ICLR 2026\] Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models](../../ICLR2026/image_generation/uni-x_mitigating_modality_conflict_with_a_two-end-separated_architecture_for_uni.md)

</div>

<!-- RELATED:END -->
