---
title: >-
  [Paper Note] ATHA: Improving CLIP Adaptation in Source-Free Cross-Domain Few-Shot Learning by Breaking Tail Alignment
description: >-
  [ICML 2026][Multimodal VLM][CLIP Fine-tuning] ATHA proposes an asymmetric alignment paradigm for CLIP cross-domain few-shot fine-tuning: "align head tokens…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "CLIP Fine-tuning"
  - "Cross-Domain Few-Shot Learning"
  - "Vision-Text Alignment"
  - "Tail Token"
  - "Source-Free Adaptation"
date: 2026-05-08
content_hash: f9789dab31f0fb34
---

# ATHA: Improving CLIP Adaptation in Source-Free Cross-Domain Few-Shot Learning by Breaking Tail Alignment

**Conference**: ICML 2026  
**arXiv**: [2605.29776](https://arxiv.org/abs/2605.29776)  
**Code**: https://github.com/shuaiyi308/ATHA  
**Area**: Multimodal VLM / Cross-Domain Few-Shot  
**Keywords**: CLIP Fine-tuning, Cross-Domain Few-Shot Learning, Vision-Text Alignment, Tail Token, Source-Free Adaptation  

## TL;DR
ATHA proposes an asymmetric alignment paradigm for CLIP cross-domain few-shot fine-tuning: "align head tokens, push away tail tokens." Actively pushing semantically sparse patches away from text embeddings mitigates overfitting and improves 1-shot average accuracy from 55.92% to 58.35%.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) like CLIP learn semantically aligned image-text representations through contrastive pre-training, demonstrating strength in zero-shot tasks. For downstream adaptation, the mainstream approach is to **further strengthen the alignment between all patch tokens and corresponding text embeddings**—as seen in methods like SPARC, PACL, and Contrastive Localized Pre-Training. In Cross-Domain Few-Shot Learning (CDFSL) and its stricter variant, Source-Free CDFSL (SF-CDFSL, where source data is inaccessible during fine-tuning), the default assumption remains that "stronger alignment leads to better performance."

**Limitations of Prior Work**: The authors identify a **counter-intuitive phenomenon**: in cross-domain few-shot fine-tuning, deliberately pushing the patches with the lowest semantic similarity (referred to as tail tokens) away from the corresponding text embeddings consistently improves performance across four standard CDFSL benchmarks (ISIC, EuroSAT, CropDiseases, ChestX). This directly contradicts the mainstream "all-token alignment" paradigm, which suggests any such disruption should degrade performance.

**Key Challenge**: Under the dual constraints of large domain gaps and extremely sparse training data (1 or 5 shots per class), the model **lacks the capacity to extract sufficient semantics** from images to truly align tail tokens. Forcing tail token alignment results in "memorizing" specific pixel distributions of support set images (overfitting) rather than learning semantics. The paper uses CKA domain similarity to verify that standard fine-tuning causes an abnormal drop in source-target feature similarity (a typical signal of overfitting), whereas pushing tail tokens restores this similarity.

**Goal**: (1) Provide a principled explanation for "pushing tail tokens," (2) engineer this observation into an end-to-end trainable module that strengthens alignment for semantically rich patches while suppressing overfitting in semantically sparse patches, and (3) achieve SOTA on SF-CDFSL benchmarks.

**Key Insight**: Since the fundamental issue with tail tokens is that "forced alignment in the absence of semantics leads to noise memorization," while head token alignment still conveys useful transfer signals, the "alignment" process should be **stratified by token semantic relevance**: pull discriminative patches toward the most similar class text (pull), push weak patches away from the least similar class text (push), and leave the remaining tokens unchanged.

**Core Idea**: Use the "maximum class similarity" of a token as a proxy to dynamically identify Head/Tail tokens at each ViT layer, then perform asymmetric alignment using learnable intensity parameters $\alpha^{(l)}, \beta^{(l)}$ to "pull head and push tail."

## Method

### Overall Architecture
The base model is CLIP-ViT/B-16 utilizing LoRA (Low-Rank Adaptation) where the backbone is frozen, and only the LoRA matrices and a set of layer-wise learnable alignment intensities are trained. Given an input image $\mathbf{x}$ and $N$ target class names, the text encoder generates text embeddings $\mathbf{T}\in\mathbb{R}^{N\times D_t}$, which are then projected into the visual token space $\mathbf{T}'=\text{LayerNorm}(\mathbf{T})\mathbf{W}_p^\top\in\mathbb{R}^{N\times D}$ using CLIP's visual projection matrix $\mathbf{W}_p$ (shared across all layers). In the visual path, the image is partitioned into $L$ patches, passing through layer $l$ of the ViT to obtain $\mathbf{V}^{(l)}\in\mathbb{R}^{B\times(L+1)\times D}$. Within specified transformer blocks, ATHA computes the cosine similarity of each patch token to all class texts, identifies Head/Tail tokens for asymmetric modification, and feeds them back into the remaining parts of the block. Finally, the [CLS] token and text embeddings are used for cosine similarity, and the model is trained end-to-end using standard cross-entropy loss to optimize LoRA and $\{\alpha^{(l)},\beta^{(l)}\}$.

### Key Designs

1. **Based on Maximum Similarity Discriminative Token Selection**:

    - **Function**: Dynamically categorizes $L$ patch tokens into head, tail, or neutral classes at each ViT layer as the basis for subsequent asymmetric processing.
    - **Mechanism**: At layer $l$, token-class similarity is calculated as $s_{b,i,j}^{(l)}=\frac{{\mathbf{v}_{b,i}^{(l)}}^\top \mathbf{t}'_j}{\|\mathbf{v}_{b,i}^{(l)}\|\|\mathbf{t}'_j\|}$. For each token, the maximum similarity across all classes $s_{b,i}^{\max,(l)}=\max_j s_{b,i,j}^{(l)}$ is taken as its **transferability proxy**. Based on these values, the Top-$k_{\text{head}}$ are designated as Head Tokens ($k_{\text{head}}=\lfloor L\cdot\rho\rfloor$), the Bottom-$r_{\text{tail}}$ as Tail Tokens ($r_{\text{tail}}=\lfloor L\cdot\gamma\rfloor$), and others remain unchanged. The paper adopts $\rho=\gamma=0.1$.
    - **Design Motivation**: Similarity distribution plots demonstrate that pre-trained CLIP exhibits a clear bimodal structure (few heads + many tails) on the source domain, which is flattened after direct transfer to the target domain. "Maximum class similarity" serves as a **cheap and layer-dynamic** transferability metric that does not rely on additional learning signals, allowing each layer to re-identify tokens based on current feature distributions.

2. **Pull Head Asymmetric Head Alignment**:

    - **Function**: Actively pulls each Head token toward its most similar class text embedding to reinforce the alignment of semantically capable patches.
    - **Mechanism**: For Head token $i\in\mathcal{I}_{\text{head}}^{(l)}$, the index $j^+=\arg\max_j s_{b,i,j}^{(l)}$ is found, and the token is updated via $\tilde{\mathbf{v}}_{b,i}^{(l)}=\mathbf{v}_{b,i}^{(l)}+\alpha^{(l)}\cdot \mathbf{t}'_{j^+}$. $\alpha^{(l)}$ is the learnable pulling intensity for that layer. The initialization strategy sets $\alpha^{(l)}=0.8$ only for a specific layer (e.g., $l=8$) and $\alpha^{(l)}=0$ for others, allowing the model to first learn "where to reinforce" before fine-tuning other layers.
    - **Design Motivation**: Head tokens are already close to a specific class text; this alignment direction has authentic semantic support. Actively adding a text embedding for translation is equivalent to "taking a step in the correct semantic direction" within the visual feature space, reducing ambiguity during final classification. Learnable $\alpha^{(l)}$ allows the model to adjust layer-wise—preserving shallow layers while explicitly aligning deep layers.

3. **Push Tail Asymmetric Tail Alignment**:

    - **Function**: Actively pushes each Tail token away from its least similar class text embedding to explicitly break "meaningless alignment," preventing the memorization of noise patches as training sample features.
    - **Mechanism**: For Tail token $i\in\mathcal{I}_{\text{tail}}^{(l)}$, the index $j^-=\arg\min_j s_{b,i,j}^{(l)}$ is found, and the token is updated via $\tilde{\mathbf{v}}_{b,i}^{(l)}=\mathbf{v}_{b,i}^{(l)}-\beta^{(l)}\cdot \mathbf{t}'_{j^-}$. It can be proven via inner product that $\mathbf{v}'\cdot \mathbf{t}=\mathbf{v}\cdot\mathbf{t}-\beta\|\mathbf{t}\|^2 < \mathbf{v}\cdot \mathbf{t}$, effectively suppressing vision-text similarity. $\beta^{(l)}$ is initialized to a small value of $0.01$ for all layers to provide a gentle push-away starting point.
    - **Design Motivation**: This is the most counter-intuitive yet critical part of ATHA. CKA domain similarity experiments show that standard fine-tuning causes features to excessively absorb specific training sample information (abnormally low CKA), while "pushing tails" pulls the CKA back. This suggests the essence of tail alignment is **memorization rather than generalization**; explicitly pushing them away closes that memorization channel. Making push-away a learnable $\beta^{(l)}$ allows the model to automatically select the optimal intensity per layer.

### Loss & Training
- **Loss**: Standard image-text cross-entropy $\mathcal{L}_{\text{cross}}=-\frac{1}{N}\sum_i \log \frac{\exp(\text{sim}(\mathbf{f}_i,\mathbf{t}_i)/\tau)}{\sum_j \exp(\text{sim}(\mathbf{f}_i,\mathbf{t}_j)/\tau)}$, where $\mathbf{f}_i$ is the final visual embedding of the [CLS] token.
- **Trained Parameters**: LoRA low-rank matrices + layer-wise pairs $(\alpha^{(l)},\beta^{(l)})$, with the backbone frozen.
- **Optimizer & Hyperparameters**: AdamW, 100 epochs, data augmentation via random cropping and horizontal flipping. Follows a 5-way 1/5-shot episodic protocol, averaging across 800 episodes for 1-shot and 400 episodes for 5-shot.
- **Key Hyperparameters**: $\rho=\gamma=0.1$ (10% tokens for head/tail); $\alpha^{(8)}=0.8$ for single-layer bootstrap, and $\beta^{(l)}=0.01$ for all-layer bootstrap.

## Key Experimental Results

### Main Results
Evaluations on 4 cross-domain few-shot benchmarks (ISIC2018 Skin, EuroSAT Remote Sensing, CropDiseases, ChestX) for 5-way 1-shot / 5-way 5-shot:

| Method | Backbone | Shot | ISIC | EuroSAT | CropDiseases | ChestX | Ave. |
|------|----------|------|------|---------|--------------|--------|------|
| StepSTP (TPAMI-25) | ViT/CLIP | 1 | 32.97 | 70.01 | 84.84 | 22.84 | 52.68 |
| CLIP-LoRA (CVPRW-24) | ViT/CLIP | 1 | 35.23 | 81.41 | 85.32 | 21.73 | 55.92 |
| ReCIT (ICML-25, DINO) | ViT/DINO | 1 | 38.48 | 75.23 | 85.92 | 23.84 | 55.87 |
| REAP (ICML-25, DINO) | ViT/DINO | 1 | 38.67 | 75.97 | 85.33 | 24.17 | 56.04 |
| **CLIP-LoRA + ATHA(Ours)** | ViT/CLIP | 1 | **38.86** | **82.56** | **87.99** | **24.00** | **58.35** |
| StyleAdv-FT (CVPR-23) | ViT/DINO | 5 | 51.23 | 90.12 | 95.99 | 26.97 | 66.08 |
| FLoR (CVPR-24) | ViT/DINO | 5 | 53.06 | 90.75 | 96.47 | 27.02 | 66.83 |

On 1-shot, ATHA pushes the CLIP-LoRA baseline from 55.92% to 58.35% (+2.43 points), outperforming all previous ViT/CLIP and ViT/DINO methods across the 4 datasets. Specifically, it exceeds CLIP-LoRA by 1.15 points on EuroSAT and 2.67 points on CropDiseases.

### Ablation Study
The paper validates three points using distribution/CKA analysis:

| Configuration | Phenomenon | Conclusion |
|------|------|------|
| Pre-trained CLIP (Zero-shot) | Flat similarity distribution, weak discriminability | Domain gap causes head/tail bimodal structure to vanish |
| Standard Fine-tuning | Upward shift of the whole curve, significant drop in CKA | All-token alignment causes overfitting, pulling noise toward text |
| Push-away-tail Only | Tail drops, head continues to rise, CKA recovers | Pushing tail suppresses tail noise without harming head alignment |
| Full ATHA (Pull + Push) | Further head rise, further tail suppression | Pull-push synergy yields the +2.43 point end-to-end gain |

### Key Findings
- **Push is the primary source of gain**: Performing only "Push-away-tail" attains most of the performance improvement; "Pull-head" is a complementary addition. This aligns with the argument that "breaking harmful alignment > strengthening existing alignment."
- **CKA domain similarity is a valid overfitting metric**: Standard fine-tuning results in an abnormal CKA drop, which Push-away restores, providing quantitative evidence that "tail alignment = memorization."
- **Layer-wise initialization is critical**: Starting $\alpha$ at layer 8 (middle of the 12-layer ViT) and $\beta$ across all layers proves to be a stable progressive initialization strategy.
- **Robustness to $\rho,\gamma$**: A 10% ratio remains robust across all four datasets, suggesting that the head/tail boundary does not require fine-tuning.

## Highlights & Insights
- **Challenging the core belief of VLM adaptation**: While cross-modal alignment literature generally assumes "complete alignment is better," ATHA provides the first systematic evidence that "active anti-alignment" is the correct approach in few-shot + large domain gap scenarios.
- **Representation-based rather than loss-based alignment control**: By directly adding/subtracting text embeddings to manipulate patch representations, ATHA bypasses the indirect nature of contrastive loss, making the intensity and direction of alignment manipulation more controllable and layer-aware.
- **CKA as an "overfitting thermometer"**: This metric can be transferred to any source-free adaptation scenario to diagnose whether the model has fallen into the trap of "memorizing training samples."
- **Learnable asymmetric alignment intensities $(\alpha, \beta)$**: This concept can be generalized to any scenario requiring differential token treatment based on transferability—such as distinguishing between "rote memorization" and "generalized" prompt tokens in instruction tuning.

## Limitations & Future Work
- **Fixed head/tail ratios**: Although $\rho=\gamma=0.1$ is robust, semantic density varies across domains (medical vs. remote sensing), and adaptive ratios might yield further gains.
- **Validated only on LoRA**: It is unknown if other PEFT schemes (Prefix, Adapter) or full fine-tuning would benefit similarly.
- **Dependency on class names as semantic proxies**: In zero-shot scenarios where class names are missing or highly abstract (e.g., fine-grained codes), the "maximum similarity" judgment may be distorted.
- **Unexplained optimality of "pushing most dissimilar class text"**: While pushing any class text should reduce tail alignment, the authors choose the least similar class based on empirical best results rather than strict optimality analysis.

## Related Work & Insights
- **vs CLIP-LoRA (CVPRW-24)**: Both freeze the backbone and use LoRA, but CLIP-LoRA uses standard alignment; ATHA adds asymmetric alignment on top, gaining +2.43 points. This indicates "what modules to add" is more critical than "what parameters to tune."
- **vs SPARC / PACL (Dense Alignment)**: These aim for "every patch to be aligned," representing the mainstream fine-grained route. ATHA counters this directly: "Not every patch should be aligned." The performance difference is clear in cross-domain few-shot settings.
- **vs ReCIT / REAP (ICML-25)**: These use DINO instead of CLIP, averaging ~56%. ATHA surpasses them at 58.35% using CLIP, validating that CLIP’s vision-text alignment prior is more suitable for "alignment-manipulation" methods.
- **vs StepSTP (TPAMI-25)**: On the same backbone, ATHA is 5.67 points higher. StepSTP continues to perform all-token alignment, highlighting the value of the "breaking tail alignment" philosophy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Counter-intuitive discovery + systematic explanation + engineered solution. Challenges a long-held belief.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid results over 4 benchmarks × 2 shot settings and CKA analysis, though ablation could be even more granular.
- Writing Quality: ⭐⭐⭐⭐ Narrative flow from phenomenon to analysis to method is clear; figures effectively illustrate the core arguments.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to CLIP adaptation in low-resource settings; the observation may prompt a re-evaluation of the "alignment cult" in VLM literature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](../../CVPR2026/multimodal_vlm/mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)
- [\[ICML 2026\] Left-Right Symmetry Breaking in CLIP-style Vision-Language Models Trained on Synthetic Spatial-Relation Data](left-right_symmetry_breaking_in_clip-style_vision-language_models_trained_on_syn.md)
- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](../../ICCV2025/multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[ICML 2026\] VLA-Arena: An Open-Source Framework for Evaluating Vision-Language-Action Models](vla-arena_an_open-source_framework_for_benchmarking_vision-language-action_model.md)
- [\[ICML 2026\] AOEPT: Breaking the Implicit Modality-Reduction Bottleneck in Modality-Missing Prompt Tuning](aoept_breaking_the_implicit_modality-reduction_bottleneck_in_modality-missing_pr.md)

</div>

<!-- RELATED:END -->
