---
title: >-
  [Paper Note] SOLAR: Self-supervised Joint Learning for Symmetric Multimodal Retrieval
description: >-
  [ICML2026][Multimodal VLM][Symmetric multimodal retrieval] SOLAR proposes the first two-stage self-supervised learning framework for "symmetric MM2MM retrieval" (where both query and document are image-text pairs and rol…
tags:
  - "ICML2026"
  - "Multimodal VLM"
  - "Symmetric multimodal retrieval"
  - "self-supervised joint learning"
  - "intersection-set difference decoupling"
  - "masked contrastive learning"
  - "multimodal embedding"
date: 2026-05-08
content_hash: 7a3cf0ce0dd5879a
---

# SOLAR: Self-supervised Joint Learning for Symmetric Multimodal Retrieval

**Conference**: ICML2026  
**arXiv**: [2605.15868](https://arxiv.org/abs/2605.15868)  
**Code**: The paper states "Code and benchmark will be available soon"; not yet officially open-sourced.  
**Area**: Multimodal VLM  
**Keywords**: Symmetric multimodal retrieval, self-supervised joint learning, intersection-set difference decoupling, masked contrastive learning, multimodal embedding  

## TL;DR
SOLAR proposes the first two-stage self-supervised learning framework for "symmetric MM2MM retrieval" (where both query and document are image-text pairs and roles are interchangeable). In the first stage, it learns an "intersection mask" to decouple shared and unique information between images and text via global-local alignment and QDA adaptive thresholding. In the second stage, it utilizes this mask to construct positive and hard-negative samples by masking different regions for contrastive learning. The authors also release a human-verified sym-MM2MM benchmark of 214 samples; SOLAR, with 0.2B parameters and 768-dimensional embeddings, outperforms the strongest 7.75B VLM baseline by 7.08 percentage points.

## Background & Motivation

**Background**: Multimodal retrieval is typically categorized into UM2MM, MM2UM, and MM2MM. Current general multimodal embedding models such as UniIR, VLM2Vec, MM-Embed, GME, and mmE5 typically adopt an asymmetric paradigm where the query is unimodal or has a specific structure while the content has another. Furthermore, they rely on supervised learning using human-annotated query-document pairs.

**Limitations of Prior Work**: Real-world scenarios often involve "symmetric MM2MM (sym-MM2MM)" retrieval where the query and content are structurally symmetric and semantically interchangeable. For example, in e-commerce, a user might search with a "front view image + back view description" to retrieve a "back view image + front view description." Existing asymmetric models perform poorly on sym-MM2MM and cannot be trained for "role swapping."

**Key Challenge**: The cost of natural annotation for sym-MM2MM is extremely high. Judging "semantic equivalence" is a subjective, fine-grained task, making large-scale human annotation expensive and slow. Meanwhile, synthetic data is limited by the capabilities of generative models and suffers from low-quality samples. This creates a tension between the "data bottleneck" and the necessity of web-scale self-supervision for scaling modern AI.

**Goal**: To enable the model to learn the ability to judge whether an image and text constitute the same semantic whole from readily available image-caption pairs without relying on any human sym-MM2MM annotations, while also releasing a benchmark for evaluating this task.

**Key Insight**: Any web image-text pair contains both "shared concepts covered by both modalities (intersection)" and "unique details appearing in only one modality (set difference)." If these can be automatically decoupled, samples can be generated programmatically: image-text pairs with the intersection masked should still be reconstructible from each other (positive samples), while masking the set difference loses unrecoverable information (hard negatives).

**Core Idea**: Use the "intersection mask" as a pivot to transform the semantic equivalence problem of symmetric retrieval into two learnable tasks: alignment of shared multimodal content and preservation of modality-unique content.

## Method

### Overall Architecture
The encoding side of SOLAR consists of five components: a vision encoder $\mathcal{E}_V$ (e.g., DINOv2 or CLIP-vision), a language encoder $\mathcal{E}_L$ (e.g., BGE-m3 or CLIP-text), two two-layer MLP adapters $\mathcal{A}_V, \mathcal{A}_L$ to project unimodal features into a shared space, and a VL-encoder $\mathcal{E}_{VL}$ composed of three attention layers for cross-modal fusion. During inference, the input image-text pair $\mathbf{X}=(\mathbf{I}, \mathbf{T})$ is processed via unimodal encoding and adapters to obtain patch-level visual features $\mathbf{V}$ and token-level text features $\mathbf{L}$. Local features $\mathbf{V}', \mathbf{L}'$ are concatenated with a learnable `[CLS]` token and fed into $\mathcal{E}_{VL}$, where the output at the `[CLS]` position serves as the final joint embedding $\mathbf{f}$.

Training is divided into two stages: Stage 1 learns an intersection mask that reliably distinguishes image-text intersections from set differences; Stage 2 uses this mask to automatically generate positive/hard-negative samples for contrastive learning. The entire process requires no human annotation and consumes only 800,000 unlabeled image-text pairs from LAION-5B.

### Key Designs

1. **Distillation-Guided Local Alignment + QDA Adaptive Threshold for Intersection Mask Generation (Core of Stage 1)**:

    - **Function**: For any image-text pair, output a 0/1 mask identifying "which patches/tokens belong to the shared intersection of the two modalities."
    - **Mechanism**: First, a metric signal is learned via Global-to-Local Alignment (GLA). For a positive pair, the average similarity between local features and the global representation of the "partner modality" should be higher than that of any in-batch negative sample, formulated as a hinge loss $\mathcal{L}_{L2V} = [\mathrm{mean}(\mathbb{S}_{L2V}^-) + \delta - \mathrm{mean}(\mathbb{S}_{L2V}^+)]_+$, with $\mathcal{L}_{V2L}$ defined symmetrically. Simultaneously, Local Distillation (LD) forces the student model's local features to maintain the same unimodal similarity ranking as strong unimodal teachers (DINOv2, BGE-m3) using Pearson correlation: $\mathcal{L}_\mathrm{LD}^L = 1 - \frac{1}{N}\sum_k \mathrm{corr}(\mathbf{S}_k^{\mathcal{T}}, \mathbf{S}_k)$. This ensures the GLA signal has a clean local foundation. Then, MaskGen collects similarity scores for each patch/token relative to the partner modality's global vector. Positive and in-batch negative samples form two Gaussian distributions. One-dimensional Quadratic Discriminant Analysis (QDA) is used to find the intersection point $\tau$ where Gaussian densities are equal (solving $\mathcal{N}(\tau; \mu^+, (\sigma^+)^2) = \mathcal{N}(\tau; \mu^-, (\sigma^-)^2)$). Positions with similarity above this threshold are marked as 1. Finally, an "evolutionary mask" $\mathbf{M} = \rho \mathbf{1} + (1-\rho) \hat{\mathbf{M}}$ is used, with $\rho$ annealing from 1 to 0 to prevent early-stage noise from causing model collapse.
    - **Design Motivation**: The high cost of human annotation necessitates a self-supervised approach. To generate masks automatically, a quantifiable signal of "where image and text align" is required; GLA provides this signal at minimal cost. LD addresses the circular dependency where unreliable student local features would degrade the GLA signal. QDA replaces fixed thresholds because similarity distributions vary across models and training stages. Evolutionary masking accounts for the increasing reliability of masks as training progresses.

2. **Synergistic Training via Masked ITC + Global Distillation (Mechanism of Stage 1)**:

    - **Function**: Drives mask accuracy while preventing the model from forgetting modality-unique information in the set difference.
    - **Mechanism**: The evolutionary masks $\mathbf{M}_V, \mathbf{M}_L$ from Stage 1 are applied to the self-attention of $\mathcal{E}_{VL}$. The `[CLS]` token is restricted to attending only to the intersection part, producing global embeddings $\mathbf{f}_V, \mathbf{f}_L$, which are optimized via a bidirectional InfoNCE-style Masked ITC loss $\mathcal{L}_\mathrm{ITC}$. To prevent the model from completely discarding set difference information, Global Distillation (GD) is introduced, requiring the student's "unmasked" global embedding to align with the teacher model's in-batch similarity structure via Pearson correlation: $\mathcal{L}_\mathrm{GD}^L = 1 - \mathrm{corr}(\mathbf{S}^\mathcal{T}, \mathbf{S})$. The total Stage 1 objective is $\mathcal{L} = \mathcal{L}_\mathrm{ITC} + \lambda_1 \mathcal{L}_{GLA} + \lambda_2 \mathcal{L}_{GD} + \lambda_3 \mathcal{L}_{LD}$.
    - **Design Motivation**: Masked ITC acts as a self-supervised signal for the mask: if the masked positions are indeed the intersection, the remaining content should allow cross-modal alignment. This creates a closed loop where alignment encourages better mask generation. GD acts as a counterweight to ITC, ensuring the final joint embedding retains modality-unique discriminative details (e.g., color, brand), which are critical for distinguishing hard negatives in sym-MM2MM retrieval.

3. **Segment-Based Automatic Positive/Hard-Negative Construction (Core of Stage 2)**:

    - **Function**: For each image-text pair $\mathbf{X}^i$, automatically produce a semantically equivalent but partially masked positive sample and a similar-looking but semantically broken hard negative sample.
    - **Mechanism**: On the text side, tokens with similarity higher than $\tau_L$ (intersection) are randomly masked to create the positive sample $\mathbf{M}_L^+$, while tokens below $\tau_L$ (set difference) are masked to create the negative sample $\mathbf{M}_L^-$. On the image side, due to patch redundancy, local visual features $\mathbf{V}'$ are first hierarchically clustered into coarse semantic segments $\mathbf{R}_k$. Each segment is scored for its relevance to the text: $s_k = \sum_{p \in \mathbf{R}_k} \mathbf{S}_{L2V}(p) / |\mathbf{R}_k|$. Segments scoring above $\tau_V$ form the intersection set for $\mathbf{M}_V^+$, and low-scoring segments form the set difference for $\mathbf{M}_V^-$. The anchor, positive samples, and three types of negative samples (constructed set-difference masked negatives, in-batch negatives, and offline-mined hard negatives) are fed into the contrastive loss: $\mathcal{L} = \frac{1}{N}\sum_i \log\frac{\sum_{j \in \mathbb{D}^{+i}} \exp(\langle \mathbf{f}^i, \mathbf{f}^j \rangle / \eta)}{\sum_{k \in \mathbb{D}^{+i} \cup \mathbb{D}^{-i}} \exp(\langle \mathbf{f}^i, \mathbf{f}^k \rangle / \eta)}$.
    - **Design Motivation**: Masking the intersection means "I removed the shared parts, but as long as the partner modality remains, the overall semantics can be reconstructed," which is the essence of a positive sample. Masking the set difference means "I removed the unique identifying details, causing unrecoverable information loss," which naturally creates a hard negative. This construction is more stable than synthetic image generation and does not depend on generative model limits. Using segments instead of single patches on the image side prevents ViT from reconstructing masked semantic info from neighboring patches, as seen in MAE-like tasks.

### Loss & Training
The total loss for Stage 1 is $\mathcal{L} = \mathcal{L}_\mathrm{ITC} + \lambda_1 \mathcal{L}_\mathrm{GLA} + \lambda_2 \mathcal{L}_\mathrm{GD} + \lambda_3 \mathcal{L}_\mathrm{LD}$, simultaneously optimizing masked alignment, global-local alignment, and dual-layer distillation while transitioning to hard masks via evolutionary annealing $\rho$. Stage 2 uses an InfoNCE-style contrastive loss with three types of negative samples for end-to-end training. All training is performed on 800,000 LAION-5B image-text pairs. The main encoders are fine-tuned using LoRA, while the VL-encoder and adapters are trained from scratch without any sym-MM2MM labels.

## Key Experimental Results

### Main Results

On the newly released sym-MM2MM benchmark (214 triplets + 1 million LAION candidate pool), the authors evaluated Recall@1/5/10, mR, Precision, and their average (Avg). The following table summarizes representative comparisons from Table 1 of the paper (R@1 / mR / Precision / Avg / #Params / #Dim):

| Method | Supervision | R@1 | mR | Precision | Avg | #Param | #Dim |
|------|---------|------|------|------|------|------|------|
| CLIP-SF | Supervised, encoder | 55.61 | 82.55 | 73.36 | 77.96 | 0.43B | 768 |
| MM-Embed | Supervised, VLM | 55.61 | 82.09 | 75.70 | 78.89 | 7.75B | 4096 |
| GME | Supervised, VLM | 56.07 | 80.37 | 74.77 | 77.57 | 7.75B | 3584 |
| UniME | Supervised, VLM | 59.81 | 83.02 | 73.36 | 78.19 | 7.49B | 3584 |
| mmE5 | Supervised, VLM | 57.94 | 84.58 | 76.64 | **80.61** | 10.12B | 4096 |
| Qwen3-VL-Embedding | Supervised, VLM | 56.54 | 81.15 | 74.77 | 77.96 | 7.75B | 4096 |
| CLIP-SF-ZS | Unsupervised, encoder | 53.27 | 80.22 | 71.03 | 75.62 | 0.15B | 512 |
| **SOLAR-B+D** (Ours) | Unsupervised, encoder | 72.90 | 87.54 | **85.51** | 86.53 | 0.71B | 768 |
| **SOLAR-C** (Ours) | Unsupervised, encoder | **77.57** | **90.81** | 84.58 | **87.69** | 0.20B | 768 |

SOLAR-C exceeds the strongest supervised VLM baseline mmE5 by 7.08 percentage points in Avg, while being ~50x smaller in parameters and more than 5x smaller in embedding dimensions. R@1 jumped from 59.81 (UniME supervised best) to 77.57, an improvement of nearly 18 points.

### Ablation Study

The table below shows the ablation of Stage 1 (selected from Table 2):

| Configuration | Avg after Stage 1 | Avg after Stage 2 | Gap vs Full Model |
|------|--------|--------|------|
| Full SOLAR (All losses) | 85+ | 86.53 | — |
| $\mathcal{L}_\mathrm{ITC}$ only | 79.5 | 81.5 | -5.0 |
| Without $\mathcal{L}_\mathrm{ITC}$ | 83.3 | 82.6 | -3.9 |
| Without $\mathcal{L}_\mathrm{GLA}$ | 80.8 | — | Sig. Drop |

### Key Findings
- Even after Stage 2 enhancement, the version with only $\mathcal{L}_\mathrm{ITC}$ remains 5 points lower than the full model in Avg, indicating that the GLA + LD + GD trio is indispensable for "growing" accurate intersection masks.
- Conversely, removing $\mathcal{L}_\mathrm{ITC}$ leads to a 3.9-point drop (less than removing GLA), suggesting that the alignment loss acts as an "amplifier" while GLA/LD act as "sensors" generating the signal; both are essential.
- SOLAR’s small models (0.15–0.71B) completely outperform 7.75B–10B VLMs in sym-MM2MM, strongly supporting the insight that "task-adaptive data generation mechanisms > general large models + general data."
- The improvement in the Precision metric (judging if positive samples beat hard negatives) is even more significant (85+ vs 73~76) than Recall@k, demonstrating that SOLAR’s true advantage lies in discriminative pairing—the core strength of the "intersection vs set difference" paradigm.

## Highlights & Insights
- **Turning geometric intuition of "intersection/difference" into executable algorithms**: The authors borrow the simple intuition of set theory (shared = intersection, unique = set difference) and operationalize it via GLA + QDA into a differentiable mask learning objective, making "interchangeability" a concrete training signal rather than an abstract concept.
- **Using the adversary as training data**: Generating positives by masking intersection and hard negatives by masking set difference provides a dual data synthesis idea. This directly addresses the industry-wide challenge of "how to construct hard negatives" in contrastive learning without relying on generative models, showing high potential for transfer to other symmetric tasks (e.g., dual-document comparison, multi-view 3D matching).
- **Practical combination of QDA thresholding + evolutionary annealing**: QDA handles "distribution shift," while annealing addresses "early-stage mask unreliability." Together, they enable stable convergence of unsupervised training on only 800,000 samples, a recipe directly applicable to other self-supervised tasks requiring "self-learned thresholds."
- **Small self-supervised models beating large supervised VLMs**: This conclusion has strong paradigmatic significance for tasks where annotation is expensive but web data is abundant. Aligning data generation with task structure is more cost-effective than simply increasing parameters.

## Limitations & Future Work
- The benchmark consists of only 214 triplets. Although the authors emphasized quality control via human + multimodal pipelines, the scale is small. Verification on larger (thousands or tens of thousands) human or semi-automatic benchmarks is necessary.
- Current mask generation assumes "every image-text pair contains both intersection and set difference." In extreme cases (e.g., a purely descriptive caption and matching image), the intersection equals the whole, which might cause mask degradation; the paper does not deeply discuss fallback strategies for such edge cases.
- The capacity of unimodal teachers (DINOv2, BGE-m3) sets the ceiling for the LD signal; whether SOLAR's advantages hold when switching to weaker unimodal backbones (e.g., lightweight SSL models) remains to be verified.
- Stage 2 hard negatives depend on the stability of Stage 1 thresholds. The paper does not provide an end-to-end joint training or multi-round alternating update version, which could further improve performance.

## Related Work & Insights
- **Comparison with UniIR, mmE5, etc.**: These models depend on supervised data with asymmetric roles; despite having 7.75B–10B parameters, they only reach 78–80 in Avg. SOLAR’s < 1B unsupervised model leads by 6–10 points, proving that "task-specific self-supervision" is a structural rather than engineering advantage in sym-MM2MM.
- **Comparison with CLIP/DINO**: CLIP's object is "global-to-global" alignment and does not distinguish between intersection and difference. SOLAR introduces a second-order structure of mask and set-difference preservation, ensuring the joint embedding is both "aligned for sharing" and "preserved for uniqueness."
- **Comparison with MAE/SimMIM**: MAE-like works use masking for reconstruction in representation pretraining. SOLAR uses masking as a contrastive signal for "semantic reconstructibility." While both use masking, the objectives are fundamentally different, revealing a new use for masking in multimodal alignment.
- **Comparison with synthetic data approaches** (e.g., Zhang et al. 2024): Synthetic routes are limited by generative model capabilities and require heavy filtering. SOLAR replaces "synthesis" with "masking web data," bypassing generative bottlenecks and making it possible to scale to hundreds of millions of LAION samples.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formalizes the symmetric MM2MM retrieval task for the first time and proposes a self-supervised decoupling framework, structurally distinct from existing multimodal embedding methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison with 10 SOTA supervised baselines on a new benchmark covering metrics like Recall, Precision, FPS, and parameters, with two-stage ablation. The small benchmark scale is the only drawback.
- Writing Quality: ⭐⭐⭐⭐ Clear task definition, derivation of motivation, and training mechanism with rigorous notation. A few subscripts in long formulas may be slightly demanding for the reader.
- Value: ⭐⭐⭐⭐⭐ High specific application value for e-commerce, content recommendation, and design matching. The success of the small-model self-supervised approach will influence future multimodal embedding research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition](../../CVPR2026/multimodal_vlm/trivia_self-supervised_fine-tuning_of_vision-language_models_for_table_recogniti.md)
- [\[ICLR 2026\] Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP](../../ICLR2026/multimodal_vlm/breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)
- [\[ICLR 2026\] U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning](../../ICLR2026/multimodal_vlm/u-marvel_unveiling_key_factors_for_universal_multimodal_retrieval_via_embedding_.md)
- [\[CVPR 2026\] Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](../../CVPR2026/multimodal_vlm/multi-modal_representation_learning_via_semi-supervised_rate_reduction_for_gener.md)
- [\[ACL 2026\] UniversalRAG: Retrieval-Augmented Generation for Multimodal Corpora](../../ACL2026/multimodal_vlm/universalrag_retrieval-augmented_generation_over_corpora_of_diverse_modalities_a.md)

</div>

<!-- RELATED:END -->
