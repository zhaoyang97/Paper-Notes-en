---
title: >-
  [Paper Note] Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning
description: >-
  [CVPR 2025][Multimodal VLM][Generalized Zero-Shot Learning] This paper proposes the Visual and Semantic Prompt Collaboration Network (VSPCN). By concurrently learning visual and semantic prompts in a pre-trained ViT and designing a weak-fusion-at-shallow-layers and strong-fusion-at-deep-layers mechanism, it efficiently adapts ViT to extract semantic-relevant discriminative visual features, achieving state-of-the-art performance on CUB, SUN, and AWA2 benchmarks.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Generalized Zero-Shot Learning"
  - "Visual-Semantic Prompt"
  - "Prompt Tuning"
  - "ViT Adaptation"
  - "Knowledge Transfer"
date: 2026-05-08
content_hash: 0859a69157bb8b79
---

# Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.23030](https://arxiv.org/abs/2503.23030)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Generalized Zero-Shot Learning, Visual-Semantic Prompt, Prompt Tuning, ViT Adaptation, Knowledge Transfer

## TL;DR

This paper proposes the Visual and Semantic Prompt Collaboration Network (VSPCN). By concurrently learning visual and semantic prompts in a pre-trained ViT and designing a weak-fusion-at-shallow-layers and strong-fusion-at-deep-layers mechanism, it efficiently adapts ViT to extract semantic-relevant discriminative visual features, achieving state-of-the-art performance on CUB, SUN, and AWA2 benchmarks.

## Background & Motivation

Generalized Zero-Shot Learning (GZSL) requires the recognition of both seen and unseen classes, heavily relying on semantic information (such as class attributes) to transfer knowledge from seen to unseen classes. Traditional methods independently extract visual and semantic features using pre-trained backbones and align them afterward, which leads to insufficient alignment due to the independent extraction process.

Although fine-tuning backbones can enhance visual-semantic interactions, it faces two problems: (1) visual-semantic interactions only occur in the final layers of the network, offering limited influence on shallow layers; and (2) fine-tuning the entire backbone on limited seen-class data easily leads to overfitting to the seen classes. Although Transformer-based ZSL methods (e.g., ZSLViT) support multi-layer interactions, full fine-tuning still poses high risks of overfitting.

The key insight of this paper: unlike VPT which only learns visual prompts, concurrently learning visual prompts (encoding discriminative visual information) and semantic prompts (encoding class semantic information) allows them to collaboratively extract semantic-related visual features. Moreover, prompt tuning avoids the overfitting risks of full fine-tuning.

## Method

### Overall Architecture

VSPCN uses a pre-trained ViT-Base (ImageNet-1k) as the backbone. The input consists of five parts: the CLS token, visual prompts $f_{vp}$, semantic prompts $f_{sp}$, image tokens, and shared semantic attributes $S$ (encoded by GloVe). The shallow layers (first $l=6$ layers) utilize a weak fusion mechanism to initialize the prompts, while the deep layers (starting from the 7th layer) employ a strong fusion mechanism to continuously update them. The parameters of the ViT backbone are frozen, and only a small number of prompt-related parameters are trained.

### Key Designs

1. **Weak Prompt Fusion**:
    - **Function**: Injecting basic information into randomly initialized prompts in the shallow layers of the network.
    - **Mechanism**: Weak Visual Prompt Fusion (WVPF) aggregates information from image tokens using cross-attention: $\tilde{f}_{vp}^0 = \text{softmax}(\frac{Q_v^0 {K_v^0}^T}{\sqrt{D}}) V_v^0$, where the query originates from visual prompts, and the key/value from image tokens. Weak Semantic Prompt Fusion (WSPF) similarly aggregates information from shared semantic attributes $S$. After fusion, they are concatenated as $\tilde{F}^0 = [f_{cls}^0, \tilde{f}_{vp}^0, \tilde{f}_{sp}^0, f_1^0, \ldots, f_{N_v}^0]$ and fed into subsequent ViT layers.
    - **Design Motivation**: Shallow features are relatively low-level; a simple cross-attention mechanism is sufficient to provide a preliminary informational foundation for the prompts. Weak fusion is executed only once at the input layer.

2. **Strong Prompt Fusion**:
    - **Function**: Continuously supplementing prompt information in deep layers to prevent the decay of semantic influence as the layer depth increases.
    - **Mechanism**: Prompts are updated using a transformer with attention bias: $\tilde{f}_{vp}^l = [\alpha_v \text{softmax}(\frac{Q_v^l {K_v^l}^T}{\sqrt{D}}) + (1-\alpha_v) \text{softmax}(B_v^l)] V_v^l + f_{vp}^l$, where $B_v^l \in \mathbb{R}^{N_v}$ represents a learnable bias, and $\alpha_v$ controls the weight ratio between attention and bias. Semantic prompts are dynamically fused with the attributes $S^l$ updated by the adapter in a similar manner. During fusion, visual prompts interact exclusively with image tokens, without being distracted by other tokens.
    - **Design Motivation**: The semantic signals from weak fusion decay as the network deepens. Strong fusion re-injects visual and semantic info at every layer. The attention bias provides learnable prior positional information, supplementary to the attention mechanism.

3. **Semantic Adapter**:
    - **Function**: Learning instance-adaptive semantic features.
    - **Mechanism**: Utilizing cross-attention to allow semantic attributes to interact with the current image tokens: $S^l = \alpha_a \text{softmax}(\frac{Q_a^l {K_a^l}^T}{\sqrt{D}}) V_a^l + (1-\alpha_a) S^{l-1}$, where the query comes from the upper-level semantic attributes, and the key/value from the current image tokens.
    - **Design Motivation**: Globally shared semantic attributes treat all images identically. The adapter dynamically adjusts attribute weights based on different images, achieving adaptation of semantic features from the class-level to the instance-level.

### Loss & Training

The total loss is defined as $\mathcal{L} = \mathcal{L}_{BASE} + \lambda_{CED}\mathcal{L}_{CED} + \lambda_{SKD}\mathcal{L}_{SKD}$:

- **Base Loss** $\mathcal{L}_{BASE} = \mathcal{L}_{CLS} + \gamma \mathcal{L}_{AR}$: Classification cross-entropy (similarity between CLS token and semantic prototype) + semantic regression MSE (aligning CLS token with ground truth prototype).
- **Cross-Entropy Divergence Loss** $\mathcal{L}_{CED}$: Encourages visual prompts to learn discriminative information complementary to the CLS token. $\mathcal{L}_{ED} = \log(\frac{\mathcal{L}_{CE}(f_{vp}^M) + \mathcal{L}_{CE}(f_{cls}^M)}{\mathcal{L}_{KL}(\delta(f_{vp}^M), \delta(f_{cls}^M))} + 1)$, where the numerator ensures both are accurate and the denominator encourages diverse distributions.
- **Semantic Knowledge Distillation Loss** $\mathcal{L}_{SKD}$: JSD divergence + Euclidean distance, aligning semantic prompts to the corresponding class semantic prototypes.

During inference, a calibration strategy is used: $\tilde{y} = \arg\max_{\hat{y}}(f_{cls}^M \cdot a_{\hat{y}}^T + \tau \mathbb{I}_{\hat{y} \in \mathcal{Y}^u})$ to balance the bias between seen and unseen classes.

The model is trained on an NVIDIA RTX A4000 using the Adam optimizer with a learning rate of 0.001 and weight decay of 0.0001.

## Key Experimental Results

### Main Results (GZSL Harmonic Mean H)

| Dataset | VSPCN | ZSLViT | PSVMA | ZSCLR | MSDN | Gain vs. Second Best |
|--------|-------|--------|-------|-------|------|-----------|
| CUB | **75.7** | 73.6 | 73.8 | 72.4 | 68.1 | +1.9 |
| SUN | **53.8** | 47.3 | 52.3 | 48.7 | 41.3 | +1.5 |
| AWA2 | **77.6** | 74.2 | 75.4 | 73.4 | 67.7 | +2.2 |
| CUB (CZSL Acc) | **80.6** | 78.9 | - | 77.8 | 76.1 | +1.7 |
| SUN (CZSL Acc) | **75.3** | 68.3 | - | 66.3 | 65.8 | +7.0 |

### Ablation Study

| Configuration | CUB H | SUN H | AWA2 H | Description |
|------|-------|-------|--------|------|
| Baseline (ViT only) | 59.3 | 45.2 | 65.0 | No prompts |
| + Visual Prompt + WVPF + SVPF | 72.7 | 51.4 | 68.6 | Visual prompt contribution: +13.4 |
| + Semantic Prompt + WSPF + SSPF | 73.9 | 52.2 | 76.2 | Semantic prompt has a larger contribution: +14.6 |
| + Dual Prompts without Fusion | 65.6 | 48.9 | 67.2 | Without fusion, performance degrades below single-prompt configurations |
| Full VSPCN | **75.7** | **53.8** | **77.6** | Synergy of all components yields the optimal result |
| Without adapter (green ✓) | 74.9 | 52.9 | 72.2 | Adapter contributes 5.4% on AWA2 |

### Key Findings

- The contribution of semantic prompts (H gains of 14.6%/7.0%/11.2%) is larger than that of visual prompts (13.4%/6.2%/3.6%), indicating that injecting semantic information is more critical for GZSL.
- Performance drops below single-prompt baselines when dual prompts are not fused, indicating that the fusion mechanism is the core of their collaboration.
- Even when using ImageNet-1k ViT-Base, the proposed method outperforms methods using ImageNet-21k ViT-Large (71.0% vs 75.7% on CUB).
- The optimal settings are $\alpha_v=0.05$ (visual prompts rely almost entirely on attention) and $\alpha_s=0.8$ (semantic prompts rely more on the bias), demonstrating distinctly different fusion modes for the two prompt types.

## Highlights & Insights

- Prompt tuning is elevated from an "efficiency-oriented tool" to a "fundamental solution against overfitting"; freezing the backbone is significantly less prone to overfitting seen classes than fine-tuning it.
- The hierarchical design of weak fusion followed by strong fusion is intuitive: primary, shallow features utilize simple fusion, while fully developed, deep features undergo complex fusion.
- Attention map visualizations clearly display the complementary relationship: visual prompts focus on local regions, semantic prompts focus on semantic-related areas, and the CLS token synthesizes both.
- t-SNE visualizations demonstrate that VSPCN provides substantially better intra-class compactness and inter-class separation than ZSLViT.

## Limitations & Future Work

- Large-scale pre-trained VLMs such as CLIP were not used as backbones, leaving its efficacy on larger models unverified.
- Only GloVe was used for semantic attribute encoding; stronger encoders like BERT or CLIP text encoders could be explored.
- There are numerous hyperparameters ($\lambda_{CED}$, $\lambda_{SKD}$, $\alpha_v$, $\alpha_s$, $\alpha_a$, $\gamma$, $\eta_1$, $\eta_2$), requiring individual tuning for different datasets.
- The source code is not open-sourced, limiting reproducibility.

## Related Work & Insights

- Compared to methods learning only a single prompt (e.g., VPT, SP), the "collaborative dual prompts" approach is a key innovation that can be transferred to other vision-semantic tasks.
- Generative methods (GAN/VAE/Diffusion) and embedding-based methods have their respective pros and cons; VSPCN, as an embedding-based method, already matches or even outperforms generative alternatives.
- Combining VSPCN with generative methods is a promising direction: utilizing prompt-tuned features to guide the generation of unseen class features.

## Rating

- Novelty: ⭐⭐⭐⭐ The collaborative dual prompts and hierarchical fusion design are novel, but the overall architecture remains within the attention framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive experiments across three datasets, detailed ablation studies, hyperparameter analyses, visualizations, and thorough comparisons with generative methods.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, solid mathematical derivations, and high-quality charts/figures.
- Value: ⭐⭐⭐⭐ Achieved a new state-of-the-art in the GZSL field, with an inspiring methodological design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning](unem_unrolled_generalized_em_for_transductive_few-shot_learning.md)
- [\[CVPR 2025\] DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models](dpc_dual-prompt_collaboration_for_tuning_vision-language_models.md)
- [\[CVPR 2025\] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [\[CVPR 2026\] STiTch: Semantic Transition and Transportation in Collaboration for Training-Free Zero-Shot Composed Image Retrieval](../../CVPR2026/multimodal_vlm/stitch_semantic_transition_and_transportation_in_collaboration_for_training-free.md)
- [\[CVPR 2025\] Locality-Aware Zero-Shot Human-Object Interaction Detection](locality-aware_zero-shot_human-object_interaction_detection.md)

</div>

<!-- RELATED:END -->
