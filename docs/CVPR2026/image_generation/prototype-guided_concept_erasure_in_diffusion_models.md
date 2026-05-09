---
title: >-
  [Paper Note] Prototype-Guided Concept Erasure in Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Concept Erasure] To address the difficulty of thoroughly erasing broad concepts (e.g., violence, nudity) from diffusion models, this paper proposes a training-free erasure method based on concept prototypes. The method clusters concept-differential directions in the CLIP embedding space to obtain image-space prototypes, optimizes these into a text prototype space via cosine similarity, and at inference time selects the best-matching prototype as a negative guidance signal to suppress target concepts in a classifier-free guidance fashion.
tags:
  - CVPR 2026
  - Image Generation
  - Concept Erasure
  - Diffusion Model Safety
  - NSFW Content Filtering
  - Prototype Learning
  - Training-Free Inference
date: 2026-05-08
content_hash: 0dda5deb63d67ca6
---

# Prototype-Guided Concept Erasure in Diffusion Models

**Conference**: CVPR 2026
**arXiv**: [2603.08271](https://arxiv.org/abs/2603.08271)
**Code**: [https://github.com/Cocteau-23/Prototype-Guided-Concept-Erasure](https://github.com/Cocteau-23/Prototype-Guided-Concept-Erasure)
**Area**: Image Generation
**Keywords**: Concept Erasure, Diffusion Model Safety, NSFW Content Filtering, Prototype Learning, Training-Free Inference

## TL;DR
To address the difficulty of thoroughly erasing broad concepts (e.g., violence, nudity) from diffusion models, this paper proposes a training-free erasure method based on concept prototypes. The method clusters concept-differential directions in the CLIP embedding space to obtain image-space prototypes, optimizes these into a text prototype space via cosine similarity, and at inference time selects the best-matching prototype as a negative guidance signal to suppress target concepts in a classifier-free guidance fashion.

## Background & Motivation

**Background**: T2I models (e.g., Stable Diffusion) are trained on large-scale web data and inevitably learn unsafe concepts (nudity, violence, copyright-protected content, etc.). Concept erasure methods are broadly divided into training-based approaches (modifying model weights, e.g., ESD, RECE, MACE) and training-free approaches (inference-time intervention, e.g., SLD, Safree, AdaVD).

**Limitations of Prior Work**: Existing methods perform well on **narrow concepts** (e.g., specific entities such as Pikachu or Elon Musk), but degrade on **broad concepts** (e.g., "violence," "nudity"). Broad concepts encompass diverse visual manifestations — violence may involve bloodshed, gunfights, or riots — and a single unified signal cannot cover all modes.

**Key Challenge**: Prior methods implicitly assume that broad and narrow concepts share equivalent distributional properties and model them with a single or unified signal. This is viable for low-variance narrow concepts, but fails for high-variance, multi-faceted broad concepts — suppressing only the most salient instantiation (e.g., bloodshed in violence) while missing other semantic patterns (e.g., gunfights, riots).

**Key Insight**: Motivated by the observation that generative models organize semantics into structured low-dimensional neighborhoods rather than randomly scattered distributions, instances of a target concept in the embedding space should cluster into several compact regions. Cluster centroids can thus serve as "concept prototypes," each capturing a distinct salient mode of the concept.

**Core Idea**: Contrastive CLIP embeddings of generated images with and without the target concept are computed, clustered to obtain image-space concept prototypes, and then transferred to the text embedding space via cosine similarity optimization. At inference time, the best-matching prototype is selected as a negative guidance signal to precisely suppress each semantic sub-mode of the concept.

## Method

### Overall Architecture
A three-stage pipeline: (1) collect concept-related prompts and concept-contrastive prompts, and generate paired images with and without the target concept; (2) compute differential directions in the CLIP image embedding space and apply clustering to obtain image prototypes, then optimize these into text prototypes; (3) at inference time, select the best-matching prototype as negative guidance for denoising.

### Key Designs

1. **Concept Prototype Construction (Image Space)**

    - **Function**: Extract multi-modal directional representations of a concept from generated images.
    - **Mechanism**: For $N$ prompts, $M$ images with and $M$ images without the concept are generated per prompt. After CLIP encoding, all pairwise differentials $\mathcal{Z}_{\text{diff}} = \{z_{i,j} - z_{i,k}^{-}\}$ are computed, and k-means clustering yields $K$ image prototypes $\{p_{\mathbf{I}}^{(k)}\}_{k=1}^K$.
    - **Design Motivation**: The concept-contrastive prompt design is carefully crafted — only the concept keyword is removed while other descriptors (scene, lighting, etc.) are retained, ensuring that the differential direction purely reflects the concept itself.
    - **Key Parameters**: Broad concepts (e.g., violence) use $K=16$ prototypes; narrow concepts (style) use $K=1$; IP uses $K=2$.

2. **Text Prototype Optimization (Cross-Modal Transfer)**

    - **Function**: Transfer image-space prototypes into the text embedding space so they can be directly used to condition the LDM.
    - **Mechanism**: Each text prototype $p_{\mathbf{T}}^{(k)} \in \mathbb{R}^{L \times d}$ is a learnable soft prompt optimized by maximizing its CLIP cosine similarity with the corresponding image prototype: $\max_{p_{\mathbf{T}}^{(k)}} \frac{\langle p_{\mathbf{I}}^{(k)}, \mathcal{E}(p_{\mathbf{T}}^{(k)}) \rangle}{\|p_{\mathbf{I}}^{(k)}\| \|\mathcal{E}(p_{\mathbf{T}}^{(k)})\|}$
    - Backpropagation uses the EoT token embedding of the CLIP text encoder (encoder frozen; only prompt parameters updated), optimized for 2,000 steps at a learning rate of 5e-2.

3. **Prototype-Guided Inference (Concept Erasure)**

    - **Function**: Adaptively select the most relevant prototype at inference time for negative guidance.
    - **Mechanism**: The cosine similarity between the user prompt embedding and each prototype is computed; the prototype $p_{\mathbf{T}}^{(k^*)}$ with the highest similarity exceeding threshold $\tau$ is selected. The CFG formulation is modified as: $\tilde{\epsilon}_{\theta}(z_t, c) = \epsilon_{\theta}(z_t) + \alpha(\epsilon_{\theta}(z_t, c) - \epsilon_{\theta}(z_t)) - \beta(\epsilon_{\theta}(z_t, p_{\mathbf{T}}^{(k^*)}) - \epsilon_{\theta}(z_t))$
    - For multi-concept erasure, all prototypes are aggregated into a unified prototype bank.
    - **Design Motivation**: The threshold $\tau$ ensures that no unnecessary negative guidance is applied when the prompt is unrelated to the target concept, preserving normal generation quality.

### Loss & Training
- Backbone: SD v1.4; DDIM sampling with 30 steps; guidance scale 7.5.
- Data preparation: 400 prompt pairs per malicious concept; 100 pairs per artistic style/IP concept; 4 images per pair (fixed seed).
- Text prototype optimization: 2,000 steps; fully training-free (no modification to diffusion model weights).

## Key Experimental Results

### Main Results (I2P Dataset, Q16 Detection Rate ↓)

| Method | Type | Overall ↓ | Nudity ↓ | Violence ↓ | Self-harm ↓ |
|--------|------|-----------|----------|------------|-------------|
| SD v1.4 | Baseline | 35.6% | 54.5% | 40.1% | 35.5% |
| ESD | Training | 12.2% | 16.4% | 6.3% | 11.1% |
| TRCE | Training | 5.7% | 1.7% | 6.2% | 5.0% |
| Safree | Free | 8.8% | 5.3% | 9.6% | 7.2% |
| **Ours** | **Free** | **5.2%** | **1.7%** | **5.8%** | **3.8%** |

### Adversarial Robustness

| Method | Ring-a-Bell ↓ | P4D ↓ | UnDiff ↓ | FID ↓ |
|--------|--------------|-------|---------|-------|
| SD v1.4 | 71.3% | 91.3% | 63.8% | - |
| TRCE | 6.7% | 2.0% | 7.7% | 48.7 |
| Safree | 22.4% | 38.0% | 28.2% | 36.3 |
| **Ours** | **6.7%** | **14.5%** | **13.3%** | 45.1 |

### Key Findings
- Achieves an overall I2P detection rate of **5.2%**, the lowest among all methods (vs. TRCE at 5.7%), with consistently strong performance across all 7 sub-categories.
- Competitive under adversarial attacks despite not being specifically designed for them — matches TRCE on Ring-a-Bell (6.7%), underperforms TRCE on P4D (14.5% vs. 2.0%) but substantially outperforms Safree.
- **Strong cross-model generalizability**: outperforms Safree on both SDXL and SD 3.5; on SD 3.5, the P4D metric is reduced from Safree's 0.27 to 0.09.
- FID is slightly higher than Safree (45.1 vs. 36.3), suggesting that multi-prototype negative guidance may marginally affect generation diversity.

## Highlights & Insights
- **Multi-prototype modeling of broad concepts** is the core innovation — rather than blindly representing broad concepts with a single direction, k-means clustering in the embedding differential space captures distinct semantic sub-modes of the concept.
- **Concept-contrastive prompt design** is particularly elegant — removing only the target concept keyword while retaining all other descriptors ensures that differential directions purely reflect conceptual rather than contextual differences.
- **Cross-modal transfer** (image prototypes → text prototypes) leverages CLIP's aligned embedding space, requiring only soft prompt optimization without modifying the diffusion model.
- Fully training-free and compatible across multiple model architectures (SD1.4/SDXL/SD3.5), making deployment straightforward.

## Limitations & Future Work
- **Adversarial robustness is not an explicit optimization target**: performance under P4D attacks is notably weaker than TRCE; adversarial training could be incorporated to address this.
- The prototype count $K$ must be set manually (16 for broad concepts, 1 for narrow concepts), with no automatic determination mechanism.
- The threshold $\tau$ governs the trade-off between false erasure and missed erasure rates, requiring careful tuning.
- Concept-contrastive prompt generation relies on an LLM, which may be imprecise for concepts with ambiguous semantic boundaries.
- Slightly elevated FID suggests that negative guidance may partially affect image diversity and quality.

## Related Work & Insights
- **vs. Safree**: Safree projects text embeddings to avoid toxic concept subspaces; the proposed method applies multi-prototype negative guidance — the latter provides more comprehensive coverage on broad concepts (overall 5.2% vs. 8.8%).
- **vs. TRCE**: TRCE is a training-based method that modifies cross-attention weights; this work is training-free. TRCE shows stronger adversarial robustness (P4D) but requires model modification.
- **vs. AdaVD**: AdaVD performs value decomposition projection in cross-attention; this work applies negative guidance at the CFG level — the two approaches are orthogonal and potentially complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-prototype approach for modeling broad concepts is novel and intuitively well-motivated; the concept-contrastive differential and clustering pipeline is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers I2P, three adversarial attack benchmarks, and cross-model generalization comprehensively.
- Writing Quality: ⭐⭐⭐⭐ Motivation is articulated clearly; the multi-modal broad concept examples in Fig. 2 are convincing.
- Value: ⭐⭐⭐⭐ Offers practical value for T2I safety research; the training-free property facilitates deployment.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [\[CVPR 2026\] EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories](emma_concept_erasure_benchmark_with_comprehensive_semantic_metrics_and_diverse_c.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](../../ICLR2026/image_generation/speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)

<!-- RELATED:END -->
