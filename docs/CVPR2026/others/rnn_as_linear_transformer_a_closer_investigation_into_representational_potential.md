---
title: >-
  [Paper Note] RNN as Linear Transformer: A Closer Investigation into Representational Potentials of Visual Mamba Models
description: >-
  [CVPR 2026][Others][Vision Mamba] This paper unifies Softmax Attention, Linear Attention, and Mamba into a single token-mixing matrix $Y=MX$. Through rank analysis, it proves that Mamba is a "low-rank approximation" of Softmax Attention, with its representational power strictly bounded between the two. The authors propose the Binary-AUC metric to quant
tags:
  - CVPR 2026
  - Others
  - Vision Mamba
date: 2026-05-08
content_hash: fc58c4b4af6bb3e6
---
# RNN as Linear Transformer: A Closer Investigation into Representational Potentials of Visual Mamba Models

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_RNN_as_Linear_Transformer_A_Closer_Investigation_into_Representational_Potentials_CVPR_2026_paper.html)  
**Code**: https://github.com/yangtiming/Dino-Mamba  
**Area**: Representation Theory / Vision Mamba  
**Keywords**: Vision Mamba, Rank Analysis, Linear Attention, Self-supervised DINO, Feature Map Evaluation

## TL;DR
This paper unifies Softmax Attention, Linear Attention, and Mamba into a single token-mixing matrix $Y=MX$. Through rank analysis, it proves that Mamba is a "low-rank approximation" of Softmax Attention, with its representational power strictly bounded between the two. The authors propose the Binary-AUC metric to quantify feature map quality and demonstrate that Vision Mamba trained via DINO self-supervision achieves 78.5% ImageNet linear probing accuracy.

## Background & Motivation

**Background**: Transformers have dominated vision tasks due to their global modeling capabilities, but their $O(L^2)$ complexity hinders scalability for high-resolution images. State Space Models (SSMs) like Mamba capture long-range dependencies with linear complexity. Combined with selective scanning and hardware-aware implementations, Mamba has become a popular Transformer alternative for vision backbones, segmentation, and medical imaging.

**Limitations of Prior Work**: Why Mamba works in vision remains largely a "black box." Existing studies (e.g., MILA, RALA) either interpret Mamba as a variant of linear attention by attributing success to forget gates or focus on gating mechanisms. These approaches essentially "force Mamba into the attention framework" without strictly defining its representational capacity based on its own structural properties.

**Key Challenge**: Although linear attention is computationally efficient, replacing softmax with a kernel function collapses the rank of the attention matrix to the projection dimension $D_{QK}$, significantly reducing expressivity. Softmax attention remains full-rank but incurs $O(L^2)$ costs. Structurally, Mamba resembles linear attention (both take the form $Y=MX$). Why does it perform significantly better? Where exactly does Mamba fall in the "expressivity vs. efficiency" trade-off?

**Goal**: (1) Provide a unified mathematical framework for the three token mixers and compare their rank upper bounds; (2) Create a quantifiable metric for feature map quality; (3) Validate the actual representational potential of Mamba under the self-supervised paradigm.

**Key Insight**: The authors observe that the SSD form of Mamba-2 can be written as a semi-separable matrix $M$, while Softmax and Linear Attention also follow the $Y=MX$ form. Since all three share the same shell, the differences lie in the structure of $M$. Consequently, the rank of the matrix serves as a unified ruler to measure their expressivity.

**Core Idea**: By viewing Mamba as "linear attention with a learnable causal mask," the authors prove that the learnable mask $L_M$ elevates the rank of off-diagonal blocks from $D_{QK}$ (in linear attention) to $\mathrm{rank}(L_M)\cdot N$. This establishes a strict hierarchy: Softmax > Mamba > Linear Attention. This theoretical conclusion is then quantified on real feature maps using a segmentation-mask-based AUC metric.

## Method

This work does not propose a new network but focuses on "Analysis + Measurement + Validation": first, a unified rank analysis establishes an expressivity hierarchy; second, Binary-AUC is designed to quantify feature map quality; and third, DINO self-supervision is used to train Vision Mamba to generate clean feature maps that support the theory. 

### Overall Architecture

The first step is **Unified Formalization**: given input $X\in\mathbb{R}^{L\times d}$, the outputs of all three token mixers are written as $Y=MX$, differing only in the construction of $M$. The second step is **Rank Analysis**: $M$ is partitioned into diagonal blocks (intra-chunk) and off-diagonal blocks (long-range dependencies). While all diagonal blocks are full-rank, differences emerge in the rank upper bounds of off-diagonal blocks, leading to the ranking: Softmax > Mamba > Linear Attention. The third step is **Empirical Measurement**: similarity maps between the [CLS] token and image tokens are used to calculate AUC against segmentation masks. DINO self-supervised training is employed to obtain cleaner feature maps, validating the theoretical hierarchy across classification, segmentation, detection, and robustness tasks.

### Key Designs

**1. Unified Matrix Form: Bringing Mamba to the Comparative Table**

To compare expressivity, the three must be made comparable. The authors prove that Softmax Attention, Linear Attention, and Mamba can all be represented as $Y=MX$, with different mixing matrices $M$:

$$M=\begin{cases} L_M\circ(C^\top B), & \text{Mamba}\\ L_{\text{Attn}}\circ\mathrm{softmax}(QK^\top), & \text{Self-Attn}\\ L_{\text{Attn}}\circ\big(\phi(Q)\phi(K)^\top\big), & \text{Lin-Attn}\end{cases}$$

Where $\circ$ denotes the Hadamard product. The key lies in the causal mask: Softmax and Linear Attention use a **fixed** lower-triangular mask of ones $L_{\text{Attn}}$, whereas Mamba derives from SSM recursion $h_t=A_t h_{t-1}+B_t x_t,\ y_t=C_t^\top h_t$. When expanded, $M_{ij}=L_{M,ij}\circ(C_i^\top B_j)$, where $L_{M,ij}=A_i\cdots A_{j+1}$ is a **learnable, data-dependent** mask generated by state transitions $A$. This unified form highlights that the essential difference between Mamba and Linear Attention is not $C^\top B$ vs. $\phi(Q)\phi(K)^\top$, but "Fixed Mask vs. Learnable Mask."

**2. Expressivity Hierarchy via Hadamard Rank Bounds: Mamba in the Middle**

Using the Hadamard rank bound $\mathrm{rank}(A\circ B)\le\mathrm{rank}(A)\cdot\mathrm{rank}(B)$, the authors constrain the rank of each mixing matrix. Partitioning $M$ into $C\times C$ sub-blocks, the diagonal blocks remain full-rank ($R_{\text{diag}}=C$) due to the lower-triangular structure. The off-diagonal blocks reveal the differences: Softmax's non-linearity turns the fixed $L_{\text{Attn}}$ into an "effectively learnable" mask, resulting in full-rank $R^{\text{off}}_{\text{Self}}=C$. In Linear Attention, the fixed mask has $\mathrm{rank}(L_{\text{Attn}})=1$ and the kernel product rank is limited by $D_{QK}$, so $R^{\text{off}}_{\text{Lin}}\le D_{QK}$. For Mamba, the learnable mask $\mathrm{rank}(L_M)\ge1$ and the $C^\top B$ rank is limited by state dimension $N$, resulting in $R^{\text{off}}_{\text{Mamba}}\le\mathrm{rank}(L_M)\cdot N$. This yields the hierarchy:

$$\underbrace{C}_{\text{Self-Attn}} > \underbrace{\mathrm{rank}(L_M)\cdot N}_{\text{Mamba}} > \underbrace{D_{QK}}_{\text{Lin-Attn}}$$

Intuition: Linear attention is double-constrained by a "fixed mask + low-rank kernel." Mamba replaces the fixed mask with a learnable one and allows the state dimension $N$ to scale efficiently with $O(LNJ)$ complexity (unlike $D_{QK}$ in linear attention, which is hindered by $O(LD_{QK}^2)$ costs). Thus, Mamba acts as a "low-rank approximation" of Softmax Attention—an attractive middle ground.

**3. Binary-AUC: Quantifying "Feature Map Quality"**

To validate the hierarchy, the authors proposed Binary-AUC. Multi-class segmentation labels are merged into binary foreground/background masks $\text{Mask}_{\text{label}}$. Feature maps are binarized using a threshold $t\in[0,1]$ to produce $\text{Mask}^t_{\text{feature}}$. Coverage rates $R(t,S)=\frac{|\text{Mask}^t_{\text{feature}}\cap S|}{|S|}$ are calculated to generate TPR/FPR curves, which are integrated to find the AUC:

$$\text{AUC}=\sum_i (\text{FPR}_{i+1}-\text{FPR}_i)\cdot\frac{\text{TPR}_{i+1}+\text{TPR}_i}{2},\quad \text{AUC}_{\text{norm}}=\max(\text{AUC},1-\text{AUC})$$

AUC=1 indicates perfect alignment with foreground truth; 0.5 is random. This metric represents the first objective evaluation of feature quality using [CLS]-token similarity maps at ImageNet scale.

### Training Strategy

The models are pre-trained on ImageNet-1k using the DINO self-supervised paradigm. Global/local crops are fed into Student $S$ and Teacher $T$ networks. Teacher parameters are updated via EMA. Teacher outputs are processed with centering and temperature softmax to get $P_t$ (stop-gradient), while $P_s$ is obtained from the student. Cross-entropy $-P_t\log P_s$ is minimized. The backbone uses Mamba-v2 (dimensions 256/512/768 for tiny/small/base, 24 layers with bidirectional scanning). Fixed position encodings in Vim are replaced with DINO's adaptive position encodings (bicubic interpolation) for multi-scale compatibility. The Linear Attention control group (LinearViT) replaces $\mathrm{softmax}(QK^\top)$ with $\phi(Q)\phi(K)^\top$ (row-wise softmax).

## Key Experimental Results

### Main Results

In ImageNet-1k linear probing (DINO pre-trained), the order Self-Attn > Mamba > Linear-Attn holds, consistent with the rank hierarchy. Mamba approaches ViT performance but requires slightly more parameters:

| Backbone | Mixer | #Param.(M) | Top-1 (%) |
|------|-------|------------|-----------|
| ViT-B | self-attn | 85 | 78.2 |
| LinearViT-B | linear attn | 85 | 74.7 |
| DinoVim-B | mamba-2 | 88 | 78.1 |
| DinoMa.-R.-B (re-probed) | mamba-2 | 88 | **78.5** |

On ADE20K semantic segmentation (UperNet) and COCO detection (Cascade Mask R-CNN), Mamba's advantage over linear attention is amplified due to high-resolution long-sequence requirements:

| Backbone | ADE20K mIoU(%) | COCO APᵇ | COCO APᵐ |
|------|----------------|----------|----------|
| LinearViT-B | 29.2 | 37.1 | 32.6 |
| DinoVim-B | 38.0 | 42.8 | 37.4 |
| ViT-B | 43.2 | 44.8 | 39.1 |

### Ablation Study

On ImageNet distribution shift variants, high-rank architectures generalize better. Mamba maintains robustness close to ViT despite linear complexity:

| Backbone | Sketch | ImageNet-A | ImageNet-R | Real |
|------|--------|-----------|-----------|------|
| LinearViT-B | 21.6 | 9.6 | 32.7 | 81.3 |
| DinoVim-B | 27.6 | 14.2 | 33.1 | 84.3 |
| ViT-B | 25.5 | 15.4 | 38.0 | 84.6 |

**Key Findings**:
- The ranking Self-Attn > Mamba > Linear-Attn holds across classification, segmentation, detection, and robustness, matching the rank hierarchy $C > \mathrm{rank}(L_M)\cdot N > D_{QK}$.
- Mamba's over linear attention is far more pronounced in long-sequence tasks (Seg +8.8 mIoU, Det +5.7 APᵇ) than in classification (+3.4), confirming its long-range modeling stems from higher rank.
- Binary-AUC strongly correlates with linear probing accuracy, suggesting it can serve as a proxy for model performance.

## Highlights & Insights
- **Unified Shell + Unified Ruler**: Unifying token mixers into $Y=MX$ and measuring them with matrix rank provides a clean analytical framework. It accurately identifies the difference as "Fixed vs. Learnable Masks + Scalable State Dimensions."
- **Learnable Mask as the Rank Source**: The most significant insight is that Mamba's gain over linear attention comes from the $L_M$ mask (product of state transitions), which raises the off-diagonal rank. This provides a clean algebraic explanation for SSM superiority.
- **Quantifying Qualitative Assessment**: Binary-AUC transforms subjective "saliency map" cleaning into a quantitative metric. This tool can be reused for interpretive diagnostics or as a proxy signal for NAS.

## Limitations & Future Work
- The rank analysis provides **upper bounds** rather than tight bounds. The hierarchy depends on typical configurations (e.g., $C=256, N=D_{QK}=64$).
- Binary-AUC requires segmentation ground truth, limiting its use to datasets with pixel-level annotations. Merging classes into foreground/background also loses fine-grained semantic info.
- Experiments were confined to ImageNet scale, 24-layer depth, and bidirectional scanning. Generalization to larger scales or different strategies remains to be tested.

## Related Work & Insights
- **vs. MILA**: MILA links Mamba to linear attention and attributes success to forget gates. This work takes the opposite approach, deriving a strict rank hierarchy from Mamba's semi-separable matrix structure.
- **vs. RALA**: RALA performs "rank enhancement" on linear attention. This paper explains that Mamba naturally possesses higher rank due to its learnable mask.
- **vs. DINOv2 + register**: While previous work used register tokens to denoise feature maps, they lacked quantitative measures. Binary-AUC objectively validates these improvements at scale.

## Rating
- Novelty: ⭐⭐⭐⭐ (Unified $Y=MX$ + Rank Hierarchy + Binary-AUC provides a cohesive new perspective).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers multiple tasks and diagnostic grains, though limited to ImageNet-1k).
- Writing Quality: ⭐⭐⭐⭐ (Clear theoretical derivations and intuitive logic).
- Value: ⭐⭐⭐⭐ (Provides a quantifiable algebraic explanation for "Why Vision Mamba works").

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction](../../CVPR2025/others/care_transformer_linear_attention.md)
- [\[ACL 2025\] The Hidden Attention of Mamba Models](../../ACL2025/others/the_hidden_attention_of_mamba_models.md)
- [\[CVPR 2026\] Event-based Visual Deformation Measurement](event-based_visual_deformation_measurement.md)
- [\[CVPR 2026\] 3D-Object Perception Transformer (3PT)](3d-object_perception_transformer_3pt.md)
- [\[CVPR 2026\] Neural Differentiation in Deep Networks: A Theoretical Framework for Expressivity and Representational Diversity](neural_differentiation_in_deep_networks_a_theoretical_framework_for_expressivity.md)

</div>

<!-- RELATED:END -->
