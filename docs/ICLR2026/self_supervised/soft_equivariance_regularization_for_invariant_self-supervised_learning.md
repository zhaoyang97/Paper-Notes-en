---
title: >-
  [Paper Note] Soft Equivariance Regularization for Invariant Self-Supervised Learning
description: >-
  [ICLR 2026][Self-Supervised Learning][equivariance] This paper proposes SER (Soft Equivariance Regularization), a layer-decoupled design that applies soft equivariance regularization to intermediate ViT layers while pres…
tags:
  - "ICLR 2026"
  - "Self-Supervised Learning"
  - "equivariance"
  - "invariance"
  - "ViT"
  - "regularization"
date: 2026-05-08
content_hash: 452d6c6c976ace9a
---

# Soft Equivariance Regularization for Invariant Self-Supervised Learning

**Conference**: ICLR 2026
**arXiv**: [2603.06693](https://arxiv.org/abs/2603.06693)  
**Code**: [https://github.com/aitrics-chris/SER](https://github.com/aitrics-chris/SER)  
**Area**: Self-Supervised Learning
**Keywords**: self-supervised learning, equivariance, invariance, ViT, regularization

## TL;DR
This paper proposes SER (Soft Equivariance Regularization), a layer-decoupled design that applies soft equivariance regularization to intermediate ViT layers while preserving the invariance objective at the final layer. Without introducing additional modules, SER consistently improves classification accuracy and robustness for invariant SSL methods (MoCo-v3, DINO, Barlow Twins).

## Background & Motivation

**Background**: The dominant paradigm in self-supervised learning (SSL) learns representations that are invariant to semantics-preserving augmentations (e.g., random cropping, color jittering) via contrastive learning or redundancy-reduction objectives. Representative methods include MoCo-v3, DINO, and Barlow Twins.

**Limitations of Prior Work**: Strong invariance learning suppresses transformation-related structural information (e.g., rotation, flip, and scale cues), which is useful for geometric robustness and spatially sensitive downstream tasks such as object detection. Prior works have attempted to incorporate equivariance objectives into invariant SSL, but typically apply both objectives to the **same final representation**.

**Key Challenge**: The final representation is typically spatially collapsed (e.g., the [CLS] token or global average pooling in ViT), making it ill-suited for defining spatial group actions. Enforcing equivariance at this layer conflicts with the invariance objective — the authors empirically find that pushing equivariance regularization to deeper layers increases equivariance scores but decreases linear evaluation accuracy on ImageNet-1k.

**Goal**: How can equivariance be elegantly incorporated into invariant SSL without modifying the baseline architecture or objective, while avoiding the invariance–equivariance trade-off?

**Key Insight**: The authors observe that invariance and equivariance should be enforced at different layers — a *layer decoupling* design. The spatial token maps at intermediate layers preserve the grid structure, making them naturally amenable to analytically defined group operations.

**Core Idea**: Apply soft equivariance regularization via analytical group operations on the spatial token maps of intermediate ViT layers, while keeping the original invariant SSL objective unchanged at the final layer.

## Method

### Overall Architecture
The SER pipeline proceeds as follows: input image → data augmentation → ViT encoder → intermediate layer produces spatial token map (equivariance regularization applied) → subsequent layers produce [CLS] token (invariant SSL loss applied) → output representation.

The key design decomposes the ViT encoder $f$ into two parts: $f = f^{(2)} \circ f^{(1)}$, where $f^{(1)}$ outputs a spatially structured token map (without the [CLS] token), and $f^{(2)}$ inserts the [CLS] token at its input and produces the final embedding.

### Key Designs

1. **Layer Decoupling Strategy**:

    - **Function**: Decouple invariance and equivariance objectives across different layers.
    - **Mechanism**: The spatial token map at an intermediate layer (e.g., layer 3) preserves the $H_f \times W_f$ grid structure, allowing token permutations to be directly defined via analytical group operations $\rho_g$. The [CLS] token at the final layer has collapsed spatial structure, and only the invariance loss is applied there.
    - **Design Motivation**: Experiments show that pushing equivariance regularization to deeper layers increases equivariance scores but degrades classification accuracy (Table 4), with the optimal position being an intermediate layer (layer 3 as the sweet spot).

2. **Analytical Feature-Space Group Operation $\rho_g$**:

    - **Function**: Directly define geometric transformations in feature space without learning additional transformation networks.
    - **Mechanism**: The group $\mathcal{G}$ consists of invertible geometric transformations (90° rotations, horizontal flips, anisotropic scaling without cropping). Discrete rotations and flips correspond to token permutations; scaling is implemented via deterministic grid resampling consistent with the input-space interpolation scheme.
    - **Design Motivation**: Avoids auxiliary transformation/action modules (e.g., EquiMod's transformation network) and requires no learning or prediction of per-sample transformation codes or labels, incurring only $1.008\times$ training FLOPs.

3. **Batch Partitioning and Augmentation Strategy**:

    - **Function**: Partition the mini-batch into $b_1$ (standard augmentation) and $b_2$ (invertible augmentation).
    - **Mechanism**: Standard SSL's RandomResizedCrop involves cropping, which is not invertible and does not form a group, making it impossible to define a valid relative transformation $g = g_2 g_1^{-1}$. Therefore, $b_2$ uses $\mathcal{T}_{eq} = \mathcal{T} \setminus \{\text{Random Crop}\} \cup \{\text{Rotation } 90°\}$, retaining photometric augmentations while replacing cropping with invertible geometric transformations. Both sub-batches participate in the invariance loss; only $b_2$ additionally participates in equivariance regularization.
    - **Design Motivation**: Ensures that the relative transformation $g$ in the equivariance loss is a well-defined group element.

4. **Delayed [CLS] Token Insertion**:

    - **Function**: Defer insertion of the [CLS] token from the input layer to after the equivariance regularization layer (i.e., to the input of $f^{(2)}$).
    - **Mechanism**: If the [CLS] token participates in attention from the first layer, it disrupts the spatial regularity of the intermediate token map.
    - **Design Motivation**: Preserves the pure spatial token map structure of $f^{(1)}$'s output, enabling precise definition of group operations.

### Loss & Training

The equivariance regularization uses a patch-wise NT-Xent contrastive loss:

$$\mathcal{L}_{\text{equiv}}^{i,j} = -\log \frac{\exp(s(z_{ij}, z'_{ij}))}{\exp(s(z_{ij}, z'_{ij})) + \sum_{m \neq i} \sum_n \exp(s(z_{ij}, z_{mn})) + \sum_{m \neq i} \sum_n \exp(s(z_{ij}, z'_{mn}))}$$

where $s(x,y) = \frac{1}{\tau} \frac{x^\top y}{\|x\| \|y\|}$ and $\tau$ is the temperature (0.3 for MoCo-v3/BT, 0.5 for DINO).

The total loss is: $\mathcal{L} = \mathcal{L}_{\text{inv1}} + \mathcal{L}_{\text{inv2}} + \lambda \mathcal{L}_{\text{equiv}}$

Training uses AdamW with batch size 2048, 100 epochs, 10-epoch warmup followed by cosine decay.

## Key Experimental Results

### Main Results

| Method | Views | ImageNet Top-1 | ImageNet-Sketch Top-1 | ImageNet-V2 Top-1 | ImageNet-R Top-1 |
|--------|-------|---------------|----------------------|-------------------|-----------------|
| MoCo-v3 | 2 | 68.44 | 17.65 | 56.54 | 18.59 |
| +AugSelf | 2 | 67.55 | 13.30 | 53.74 | 17.62 |
| +STL | 2 | 65.49 | 15.40 | 55.43 | 17.22 |
| **+SER** | **2** | **69.28** | **17.68** | **56.95** | **18.95** |
| +EquiMod | 3 | 68.95 | 14.81 | 56.31 | 16.54 |
| +E-SSL | 2+4 | 70.60 | 19.23 | 58.33 | 19.86 |
| **+SER** | **2+4** | **71.56** | **19.76** | **59.50** | **20.27** |

Under the strictly matched 2-view setting, SER is the only equivariant add-on that improves MoCo-v3 accuracy (+0.84); all other methods degrade performance.

### Ablation Study

| Configuration | Equiv Loss Layer | ImageNet Top-1 | Rotation Equiv ↑ |
|---------------|-----------------|----------------|------------------|
| MoCo-v3 (baseline) | - | 68.44 | 0.804 |
| MoCo + SER | Layer 3 | **69.28** | 0.840 |
| MoCo + SER | Layer 9 | 68.72 | 0.888 |
| MoCo + SER | Layer 12 | 68.18 | 0.924 |
| +SER, λ=0 (control) | Layer 3 | 68.82 | - |
| +SER, λ>0 (full) | Layer 3 | **69.28** | - |

### Key Findings
- **Layer decoupling is essential**: Equivariance regularization performs best at layer 3 (out of 12 ViT layers); pushing it to deeper layers harms classification accuracy even when equivariance scores are higher.
- **Layer decoupling is a general design principle**: Relocating EquiMod's equivariance objective from Layer 12 to Layer 3 improves Top-1 from 68.95→69.51; the same for AugSelf: 67.55→68.23.
- **The equivariance loss itself contributes**: A control experiment with λ=0 yields only +0.38 gain (attributable to batch partitioning/augmentation changes); enabling $\mathcal{L}_{\text{equiv}}$ yields an additional improvement to +0.84.
- Consistent gains across SSL methods: DINO +0.26, Barlow Twins +0.68.
- Larger gains on spatially sensitive tasks: COCO detection +1.7 mAP, ImageNet-C/P +1.11/+1.22.

## Highlights & Insights
- **Layer decoupling as a design principle**: Invariance and equivariance objectives should not be enforced at the same layer. This finding generalizes beyond SER — applying it to EquiMod and AugSelf also improves accuracy. It is a broadly transferable design principle for multi-objective regularization.
- **Analytical group operations as an alternative to learned modules**: By exploiting the regular patch grid structure of ViT, rotations and flips are directly implemented as token permutations, introducing no additional parameters — minimal yet effective.
- **Batch partitioning to handle non-invertible augmentations**: The batch is elegantly split into two parts — one following the standard augmentation pipeline (with cropping) and one using invertible augmentations. Both participate in the invariance loss; only the latter participates in equivariance regularization. This resolves the fundamental issue that cropping does not form a group.

## Limitations & Future Work
- Experiments are conducted only on ViT-S/16; larger models (ViT-B/L) and longer training schedules (300/800 epochs) are not evaluated.
- The group $\mathcal{G}$ covers only discrete transformations (90° rotations, flips, scaling); richer transformation groups such as continuous rotations remain unexplored.
- The $b_2$ partition excludes cropping, which may reduce augmentation diversity; an ideal solution would be to design an "invertible crop."
- The optimal layer position for equivariance regularization (layer 3) may shift with model scale and may require re-tuning.
- Although the additional computational cost is negligible (1.008×), the number of negatives in the patch-wise contrastive loss scales quadratically with batch size and spatial resolution.

## Related Work & Insights
- **vs. EquiMod**: EquiMod introduces an auxiliary transformation network to enforce equivariance at the final layer and requires an extra view (3-view); SER uses analytical operations at an intermediate layer, introduces no extra parameters, and achieves higher accuracy under the 2-view setting.
- **vs. E-SSL**: E-SSL uses 2+4 multi-crop to implicitly encourage equivariance; under the matched 2+4 view setting, SER still outperforms E-SSL (71.56 vs. 70.60).
- **vs. AugSelf**: AugSelf implicitly learns equivariance by predicting transformation parameters, yet degrades accuracy in the 2-view setting (67.55 < 68.44); SER improves accuracy under the same setting.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The layer decoupling design principle is insightful, though the core component (equivariant contrastive loss) is not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Strictly matched view comparisons, multiple SSL baselines, diverse datasets, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, rigorous experimental design, and consistent notation.
- **Value**: ⭐⭐⭐⭐ The layer decoupling principle has broad applicability, though absolute performance gains are modest.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](../../NeurIPS2025/self_supervised/t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)
- [\[ICLR 2026\] Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning](why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic.md)
- [\[ICLR 2026\] SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty](snap-uq_self-supervised_next-activation_prediction_for_single-pass_uncertainty_i.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](../../CVPR2026/self_supervised/mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[AAAI 2026\] Self-Supervised Inductive Logic Programming](../../AAAI2026/self_supervised/self-supervised_inductive_logic_programming.md)

</div>

<!-- RELATED:END -->
