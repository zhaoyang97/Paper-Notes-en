---
title: >-
  [Paper Note] Multimodal Dataset Distillation Made Simple by Prototype-Guided Data Synthesis
description: >-
  [ICLR 2026][multimodal distillation] This paper proposes PDS (Prototype-Guided Data Synthesis), the first training-free multimodal dataset distillation framework. It leverages CLIP's aligned embedding space to perform modality-specific clustering, applies the Hungarian algorithm for cross-modal prototype matching, and employs an unCLIP decoder to synthesize distilled images from image prototypes. On a distillation set of as few as 100 pairs, PDS surpasses all optimization-based methods at zero training cost while achieving state-of-the-art cross-architecture generalization.
tags:
  - ICLR 2026
  - multimodal distillation
  - CLIP
  - unCLIP
  - prototype learning
  - training-free distillation
date: 2026-05-08
content_hash: 09bb5f4c06222817
---

# Multimodal Dataset Distillation Made Simple by Prototype-Guided Data Synthesis

**Conference**: ICLR 2026
**arXiv**: [2602.19756](https://arxiv.org/abs/2602.19756)
**Code**: [GitHub](https://github.com/junhyeok9712/PDS)
**Area**: Information Retrieval
**Keywords**: multimodal distillation, CLIP, unCLIP, prototype learning, training-free distillation

## TL;DR

This paper proposes PDS (Prototype-Guided Data Synthesis), the first training-free multimodal dataset distillation framework. It leverages CLIP's aligned embedding space to perform modality-specific clustering, applies the Hungarian algorithm for cross-modal prototype matching, and employs an unCLIP decoder to synthesize distilled images from image prototypes. On a distillation set of as few as 100 pairs, PDS surpasses all optimization-based methods at zero training cost while achieving state-of-the-art cross-architecture generalization.

## Background & Motivation

**Background**: The success of vision-language models such as CLIP depends on large-scale image-text datasets like LAION-5B, which incur prohibitive training costs. Dataset distillation—compressing large datasets into a small set of synthetic samples—is well established for image classification, but multimodal distillation remains in its early stages, with only a handful of prior works such as MTT-VL, TESLA-VL, and LoRS.

**Limitations of Prior Work**: All existing multimodal distillation methods are optimization-based and suffer from three fundamental problems. First, the computational cost is enormous: they require repeatedly training models on full data while storing all intermediate parameters, becoming infeasible as dataset and model scale grow. Second, they are strongly architecture-dependent: jointly optimizing image pixels and text features essentially adds architecture-specific adversarial perturbations to the initialization images, producing distilled sets that look nearly identical to the originals; switching backbones (e.g., from NFNet to ResNet/ViT) necessitates complete re-distillation. Third, subset selection methods fail entirely at very small scales (e.g., 100 pairs) due to insufficient semantic diversity.

**Key Challenge**: Optimization-based methods are "heavy" (expensive and architecture-locked), while subset selection methods are "shallow" at extremely small scales (insufficient semantic diversity). A distillation approach that is both training-free and architecture-agnostic is needed.

**Key Insight**: The authors observe that CLIP's embedding space naturally aligns image and text modalities, enabling direct semantic prototype extraction via clustering. The key insight is that the unCLIP decoder can generate images directly from CLIP image embeddings—something standard Stable Diffusion cannot do—thereby bypassing pixel-space optimization entirely.

**Core Idea**: Construct cross-modal prototypes via CLIP embedding clustering and Hungarian matching, then synthesize distilled images from image prototypes using unCLIP, achieving zero-training multimodal dataset distillation.

## Method

### Overall Architecture

PDS is a three-stage pipeline. Given a large-scale image-text dataset $\mathcal{D} = \{(x_n, y_n)\}_{n=1}^N$, it produces a compressed distilled set $\mathcal{S} = \{(\tilde{x}_m, \tilde{y}_m)\}_{m=1}^M$ ($M \ll N$). The three stages are: (i) modality-specific clustering—extracting embeddings with CLIP encoders and clustering each modality independently; (ii) cross-modal prototype matching—solving a linear assignment problem to align image and text clusters; (iii) image synthesis—generating distilled images from image prototypes using an unCLIP decoder. No model parameters are trained throughout the entire process.

### Key Designs

1. **Modality-Specific Clustering**

   - **Function**: Extract semantically diverse representative prototypes from the large dataset.
   - **Mechanism**: CLIP image and text encoders are used to extract embeddings $\{(z_n^{\text{img}}, z_n^{\text{txt}})\}$ for all sample pairs. Cosine similarity between image-text pairs is computed and low-similarity (noisy or weakly aligned) pairs are filtered out. Mini-batch k-means clustering is then applied independently to image embeddings and text embeddings, with the number of clusters set to the target distillation size $M$, yielding $M$ image clusters and $M$ text clusters.
   - **Design Motivation**: The CLIP encoder is required rather than a VAE encoder, because VAE image embeddings and CLIP text embeddings do not share the same space. Experiments confirm that replacing CLIP with VAE causes IR@10 to collapse from 37.3% to 17.2%.

2. **Cross-Modal Cluster Matching**

   - **Function**: Establish a one-to-one correspondence between image clusters and text clusters.
   - **Mechanism**: A cost matrix $K \in \mathbb{R}^{M \times M}$ is constructed, where $K_{ij}$ is the negative count of image-text pairs shared by image cluster $C_i^{\text{img}}$ and text cluster $C_j^{\text{txt}}$. The Hungarian algorithm then solves the linear assignment problem $\min_P \sum_{ij} K_{ij} P_{ij}$ (where $P$ is a permutation matrix) to obtain the globally optimal one-to-one matching. For each matched cluster pair, only the shared image-text pair embeddings are retained; their mean yields the cross-modal prototype $(\tilde{z}_i^{\text{img}}, \tilde{z}_j^{\text{txt}})$.
   - **Design Motivation**: Simple cosine similarity matching cannot guarantee a globally optimal one-to-one correspondence. The Hungarian algorithm provides an exact solution in $O(M^3)$. For "pairless clusters" with no shared pairs, they can be retained at small distillation scales (with negligible impact) but should be discarded at large scales to avoid cross-modal misalignment.

3. **Image Synthesis via unCLIP Decoder**

   - **Function**: Generate high-quality distilled images from image prototype embeddings.
   - **Mechanism**: Since the U-Net in standard Stable Diffusion does not accept CLIP image embeddings as conditioning, the unCLIP architecture is adopted. Each image prototype $\tilde{z}_i^{\text{img}}$ is fed as a condition to the unCLIP decoder; the real caption most similar (by cosine similarity) to the text prototype is retrieved as auxiliary text conditioning. Classifier-free guidance is applied (guidance scale = 5.0, 100 sampling steps) to generate 224×224 images.
   - **Design Motivation**: Three comparisons establish the necessity of this design. (1) Retrieving real images from image prototypes fails to preserve semantic diversity. (2) Pure text-to-image generation with unCLIP loses the fine-grained visual information encoded in image prototypes. (3) CLIP inversion (pixel-space optimization) produces unrealistic images and takes 1,477 seconds, compared to only 9.7 seconds for PDS.

### Loss & Training

PDS itself involves no training or loss function optimization—this is its core advantage. For downstream evaluation, the synthesized distillation set is used to fine-tune CLIP with the standard InfoNCE contrastive loss. During evaluation, the text encoder is frozen; only the image encoder and a learnable linear projection layer are trained. All experiments are conducted on a single RTX 3090.

## Key Experimental Results

### Main Results: Cross-Architecture Generalization (Flickr30K, distilled with CLIP ViT-L/14)

| Eval Backbone | # Pairs | Method | IR@1 | IR@10 | TR@1 | TR@10 |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| ResNet-50 | 100 | TESLA-VL | 4.1 | 22.9 | 6.5 | 27.3 |
| ResNet-50 | 100 | LoRS | 6.3 | 28.0 | 9.1 | 34.5 |
| ResNet-50 | 100 | **PDS** | **7.9** | **37.3** | **10.2** | **39.0** |
| ResNet-50 | 300 | TESLA-VL | 10.3 | 40.6 | 14.9 | 48.8 |
| ResNet-50 | 300 | LoRS | 8.6 | 33.5 | 14.7 | 44.1 |
| ResNet-50 | 300 | **PDS** | **14.4** | **51.4** | **18.7** | **57.8** |
| ViT-Ti/16 | 100 | TESLA-VL | 2.1 | 13.1 | 2.6 | 13.7 |
| ViT-Ti/16 | 100 | LoRS | 2.8 | 16.1 | 5.2 | 20.5 |
| ViT-Ti/16 | 100 | **PDS** | **6.8** | **28.5** | **6.6** | **26.9** |
| ViT-Ti/16 | 300 | TESLA-VL | 5.1 | 24.5 | 6.1 | 27.3 |
| ViT-Ti/16 | 300 | LoRS | 4.1 | 20.7 | 6.2 | 25.7 |
| ViT-Ti/16 | 300 | **PDS** | **9.1** | **38.4** | **9.6** | **37.5** |

PDS achieves consistent improvements across all settings. For ResNet + 300 pairs, IR@10 exceeds the second-best by 10.8 pp and TR@10 by 9.0 pp. The advantage is even more pronounced on ViT (12.4 pp lead in IR@10 at 100 pairs), underscoring the severe architecture dependence of optimization-based methods.

### Ablation Study: Image Synthesis Strategy Comparison (100 pairs, Flickr30K, ResNet)

| Method | IR@1 | IR@10 | TR@1 | TR@10 | VRAM (GB) | Time (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Text-prototype retrieval | 5.2 | 27.1 | 6.4 | 28.2 | — | — |
| Image-prototype retrieval | 5.5 | 28.7 | 8.0 | 30.2 | — | — |
| unCLIP text-to-image | 5.2 | 26.7 | 6.4 | 28.9 | — | — |
| CLIP inversion (image-aligned) | 4.4 | — | 4.2 | — | 6.13 | 1477.7 |
| CLIP inversion (text-aligned) | 1.4 | — | 2.0 | — | 6.13 | 1477.7 |
| **PDS (full)** | **7.9** | **37.3** | **10.2** | **39.0** | **4.34** | **9.7** |

This table clearly demonstrates the contribution of each design decision: (1) conditioning on image prototypes rather than text prototypes alone yields substantial gains; (2) synthesized images outperform retrieved real images; (3) PDS is 150× faster than CLIP inversion with superior performance.

### Comparison with Subset Selection Methods (100 pairs, Flickr30K, ResNet)

| Method | IR@1 | IR@10 | TR@1 | TR@10 |
|:---|:---:|:---:|:---:|:---:|
| K-center | 2.9 | 16.8 | 5.3 | 24.1 |
| Herding | 3.6 | 20.1 | 6.7 | 28.2 |
| CLIP score filtering | 2.5 | 14.5 | 4.7 | 18.9 |
| LAION filtering | 2.4 | 14.5 | 4.7 | 19.3 |
| Image-based filtering | 2.2 | 13.6 | 4.0 | 16.2 |
| **PDS** | **7.9** | **37.3** | **10.2** | **39.0** |

At the extremely small distillation scale of 100 pairs, PDS outperforms the best subset selection method (Herding) by 17.2 pp in IR@10 and 10.8 pp in TR@10. Subset selection methods are fundamentally constrained to selecting real samples and cannot preserve semantic diversity through interpolation.

### Key Findings

- **Training-free distillation comprehensively outperforms optimization-based distillation**: PDS surpasses TESLA-VL and LoRS across all distillation set sizes and evaluation backbones without any training, challenging the assumption that distillation requires complex bilevel optimization.
- **Optimization-based methods are essentially perturbation injection**: Visualization reveals that distilled images from TESLA-VL/LoRS are nearly identical to their initialization images, effectively adding architecture-specific adversarial perturbations, which explains their poor cross-architecture generalization.
- **Image prototypes are the critical component**: Pure unCLIP text-to-image generation achieves only IR@10 = 26.7%, which jumps to 37.3% upon incorporating image prototype conditioning, demonstrating that image embeddings carry fine-grained visual information that text alone cannot capture.
- **CLIP alignment is a prerequisite for multimodal distillation**: Replacing CLIP with VAE for encoding (as in D4M and MGD3) drops IR@10 from 37.3% to 9.8%–17.2%, confirming that cross-modal alignment quality directly determines distillation effectiveness.
- **Handling pairless clusters**: At small distillation scales, pairless clusters are rare and their retention or removal has negligible effect; at large scales, they must be discarded to avoid introducing cross-modal misalignment noise.

## Highlights & Insights

- **Counter-intuitive finding: training-free outperforms trained methods**: PDS optimizes no parameters whatsoever, relying solely on the off-the-shelf capabilities of pretrained CLIP and unCLIP, yet surpasses methods that require repeated training iterations. This suggests that when pretrained embedding spaces are of sufficient quality, carefully designed utilization strategies (clustering + matching + decoding) are more effective than end-to-end optimization—an insight transferable to other data compression scenarios.
- **Precise positioning of the unCLIP decoder**: The authors astutely identify that standard Stable Diffusion cannot accept CLIP image embeddings as conditioning, and that unCLIP fills exactly this gap. The design principle of "selecting a generative architecture that matches the representational form of the prototype" reflects a deep understanding of the capability boundaries of generative models.
- **Remarkable efficiency advantage**: Image synthesis requires only 9.7 seconds and 4.34 GB of VRAM, versus 1,477 seconds and 6.13 GB for CLIP inversion—a 150× speedup. The entire distillation pipeline can be completed rapidly on a single RTX 3090, offering strong practical utility.

## Limitations & Future Work

- **Encoder lock-in**: PDS depends on the CLIP embedding space and cannot leverage stronger encoders such as SigLIP, as no corresponding unCLIP decoder exists for them. The authors identify developing conditional generators compatible with SigLIP embeddings as a future direction.
- **Limited domain transferability**: CLIP and unCLIP are primarily trained on natural images and may degrade in specialized domains such as medical imaging, where domain-adaptive fine-tuning would be required.
- **Underrepresentation of long-tail categories**: Cluster centroids naturally skew toward high-frequency semantics, potentially marginalizing rare concepts. The authors demonstrate in the appendix that PDS is more robust than subset selection on rare samples.
- **Evaluation limited to retrieval tasks**: The paper evaluates only image-text retrieval (R@k) and does not cover downstream tasks such as classification, VQA, or image captioning; broader generalizability remains to be validated.
- **Directions for improvement**: (1) Adaptive clustering strategies that dynamically allocate distillation samples per semantic cluster based on data distribution; (2) multi-round training-free prototype refinement that uses generated outputs to recalibrate prototypes iteratively.

## Related Work & Insights

- **vs. TESLA-VL / LoRS**: These methods use trajectory matching for bilevel optimization, jointly learning pixel and text features at high computational cost with architecture lock-in. PDS bypasses optimization entirely, substituting clustering + matching + generation, achieving better results at zero training cost.
- **vs. D4M / MGD3**: These training-free distillation methods for image classification use VAE encoders. Direct extension to multimodal settings fails due to the misalignment between VAE and CLIP embedding spaces (IR@10 drops by more than half), confirming that cross-modal alignment is a necessary condition for multimodal distillation.
- **vs. subset selection (Herding / K-center / CLIP filtering)**: These methods are confined to selecting subsets of real data, which yields severely insufficient semantic diversity at very small scales (100 pairs). PDS overcomes this limitation by interpolating in embedding space and synthesizing new images.
- **Promising follow-up direction**: If stronger encoders such as SigLIP acquire compatible conditional generators, the PDS framework can directly substitute components to achieve higher distillation quality.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First training-free multimodal distillation framework; first use of the unCLIP decoder for image synthesis in dataset distillation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage of cross-architecture generalization, extremely small-scale comparisons, three categories of baselines, and multiple ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Three-stage pipeline described clearly, motivation chain complete, comparisons conducted fairly.
- Value: ⭐⭐⭐⭐ — Direct practical utility for multimodal data efficiency; distillation achievable on a single GPU.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting](../../AAAI2026/information_retrieval/refeed_retrieval_feedback-guided_dataset_construction_for_style-aware_query_rewr.md)
- [\[ICLR 2026\] Leveraging Data to Say No: Memory Augmented Plug-and-Play Selective Prediction](leveraging_data_to_say_no_memory_augmented_plug-and-play_selective_prediction.md)
- [\[ICLR 2026\] RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)
- [\[NeurIPS 2025\] SuperCLIP: CLIP with Simple Classification Supervision](../../NeurIPS2025/information_retrieval/superclip_clip_with_simple_classification_supervision.md)
- [\[ICLR 2026\] FutureMind: Equipping Small Language Models with Strategic Thinking-Pattern Priors via Adaptive Knowledge Distillation](futuremind_equipping_small_language_models_with_strategic_thinking-pattern_prior.md)

</div>

<!-- RELATED:END -->
