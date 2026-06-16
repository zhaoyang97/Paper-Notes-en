---
title: >-
  [Paper Note] Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology
description: >-
  [NeurIPS 2025][Medical Imaging][Computational Pathology] This paper revisits end-to-end (E2E) learning with slide-level supervision in computational pathology…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Computational Pathology"
  - "End-to-End Learning"
  - "Multiple Instance Learning"
  - "Sparse Attention"
  - "ABMILX"
date: 2026-05-08
content_hash: 2d0a771dd780ad46
---

# Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology

**Conference**: NeurIPS 2025
**arXiv**: [2506.02408](https://arxiv.org/abs/2506.02408)  
**Code**: Unavailable  
**Area**: Medical Imaging
**Keywords**: Computational Pathology, End-to-End Learning, Multiple Instance Learning, Sparse Attention, ABMILX

## TL;DR

This paper revisits end-to-end (E2E) learning with slide-level supervision in computational pathology, and is the first to identify optimization difficulties induced by sparse-attention MIL under E2E training. It proposes ABMILX, which addresses this issue via multi-head attention and a global attention correction module, enabling E2E-trained ResNets to surpass state-of-the-art foundation models on multiple benchmarks.

## Background & Motivation

The dominant paradigm in computational pathology (CPath) is a **two-stage approach**: a pretrained encoder first extracts patch features offline from all patches in a whole-slide image, and a MIL aggregator then performs slide-level prediction. Recent foundation models (FMs) such as UNI (100M patches) and GigaPath (170K slides) have substantially improved offline feature quality; however, this paradigm suffers from fundamental limitations:

**Encoders are not fine-tuned for downstream tasks**: General-purpose pretrained features are insufficiently adapted to specific tasks.

**Disjoint optimization of encoder and MIL**: The two stages are trained independently, precluding end-to-end joint optimization.

**Saturating performance of foundation models**: Even scaling data to 170K slides and model size to 1B parameters yields limited gains on challenging benchmarks such as PANDA.

**End-to-end (E2E) learning** is the intuitive remedy—jointly training the encoder and MIL aggregator. Nevertheless, E2E learning remains largely overlooked due to high computational cost and suboptimal performance.

The key insight of this paper is: **the root cause of poor E2E performance is not a suboptimal sampling strategy, but rather overlooked optimization challenges introduced by sparse-attention MIL.** Specifically:

- Sparse attention (e.g., ABMIL) excels in the two-stage setting by focusing on critical regions among tens of thousands of patches.
- Under E2E training, however, the MIL module acts as a "soft instance selector," and its attention distribution governs the gradients received by the encoder. **Excessively sparse attention** causes the encoder to overfit to a limited set of discriminative regions and to be misled by redundant regions.
- Degraded features further corrupt attention accuracy, creating a **vicious optimization cycle**.

## Method

### Overall Architecture

The E2E training pipeline consists of: multi-scale random patch sampling → encoder feature extraction → ABMILX aggregation → task head prediction. The encoder parameters $\theta$, MIL parameters $\phi$, and task head parameters $\eta$ are jointly optimized:

$$\{\hat{\theta}, \hat{\phi}, \hat{\eta}\} \leftarrow \arg\min_{\theta, \phi, \eta} \sum_{i=1}^{n} \mathcal{L}(y_i, \hat{y}_i)$$

### Key Designs

1. **Multi-scale Random Instance Sampling (MRIS)**

   Simple random sampling is adopted in place of complex attention- or clustering-based sampling strategies, with multi-scale information incorporated. Given a target sample count $s$ and a scale set $\{I_1, \ldots, I_t\}$, the number of samples per scale is allocated according to ratios $\{\sigma_1, \ldots, \sigma_t\}$:

    $\hat{s}_j = \lceil s \times \sigma_j \rceil, \quad \sum_{j=1}^{t} \sigma_j = 1$

   Patches from different scales are uniformly resized to a common resolution and merged into the final sample set. This mimics the multi-scale diagnostic workflow of pathologists, while incurring substantially lower computational overhead than attention-based sampling (9 h vs. 68 h).

2. **Multi-Head Local Attention (MHLA)**

   Features $\mathbf{E}$ are partitioned into $m$ head features $\{\mathbf{H}^1, \ldots, \mathbf{H}^m\}$. Each head independently computes sparse attention $\mathbf{A}^j = \text{MLP}(\mathbf{H}^j)$ and aggregates within the head:

    $\mathbf{Z} = \text{Concat}(\mathbf{Z}^1, \ldots, \mathbf{Z}^m), \quad \mathbf{Z}^j = \text{Softmax}(\mathcal{G}(\mathbf{A}^j))^T \mathbf{H}^j$

   **Design Motivation**: Under E2E training, "false positives" in MIL attention tend to be randomly distributed. Voting across multiple independent heads **suppresses over-attention to redundant instances**, while providing broader discriminative region coverage from diverse feature subspaces.

3. **Global Attention Correction Module (Attention Plus, A+)**

   Patch-wise global similarity is leveraged to correct local sparse attention. An intra-head similarity matrix $\mathbf{U}^j$ is computed to propagate attention scores among similar patches:

    $\mathcal{G}(\mathbf{A}^j) = \mathbf{A}^j + \alpha \cdot \text{Softmax}\left(\frac{\mathbf{Q}^j {\mathbf{K}^j}^T}{\sqrt{\lceil D'/m \rceil}}\right) \mathbf{A}^j$

   where $\alpha$ is a learnable scaling factor. The key insight is that tissue regions sharing similar pathological characteristics typically exhibit highly similar features; therefore, attention assigned to discriminative patches should propagate to their similar neighbors. In the A+ module, propagation weights $P_{abx}(i) = \mathbf{A}_k^j \sum_{k=1}^{s} \mathbf{U}_{k,i}^j$ are jointly modulated by attention values and similarity scores, ensuring that only high-attention regions propagate substantially.

### Loss & Training

Standard losses are employed according to the specific task (cross-entropy for classification; NLL Survival Loss for survival analysis). E2E training requires no more than 10 RTX 3090 GPU-hours (on TCGA-BRCA). ImageNet-1K pretrained ResNet-18/50 is used as the encoder.

## Key Experimental Results

### Main Results — Subtype Classification

| Encoder | Method | E2E | TCGA-BRCA AUC | TCGA-NSCLC AUC |
|--------|------|-----|---------------|----------------|
| ResNet-50 | Best-of-two-stage | ✗ | 89.35 | 95.21 |
| CHIEF | RRTMIL | ✗ | 92.49 | 97.00 |
| UNI | RRTMIL | ✗ | 94.61 | 97.88 |
| GigaPath | RRTMIL | ✗ | 94.82 | 97.63 |
| **ResNet-18** | **ABMILX (ours)** | **✓** | **93.97** | **97.09** |
| **ResNet-50** | **ABMILX (ours)** | **✓** | **95.17** | **97.06** |

### Main Results — Cancer Grading & Survival Analysis

| Encoder | Method | E2E | PANDA Acc | LUAD C-index | BRCA C-index |
|--------|------|-----|-----------|-------------|-------------|
| ResNet-50 | Best-of-two-stage | ✗ | 62.72 | 64.15 | 64.93 |
| UNI | 2DMamba | ✗ | 76.37 | 61.05 | 64.69 |
| GigaPath | 2DMamba | ✗ | 75.72 | 64.49 | 65.35 |
| **ResNet-18** | **ABMILX** | **✓** | **78.34** | **64.91** | **67.78** |

### Ablation Study

| MIL in E2E | Sparsity | Classification Acc ↑ | Survival C-index ↑ |
|-----------|--------|----------|-------------|
| ABMIL | 80 (extreme) | 89.23 | 62.70 |
| TransMIL | 13 (global attention) | 91.44 | 63.42 |
| MHLA only ($\alpha=0$) | 61 | 91.58 | 63.80 |
| MHLA + A+ ($\alpha=1$) | 29 | 92.84 | 65.49 |
| **MHLA + A+ (learnable $\alpha$)** | **36** | **93.97** | **67.78** |

### Key Findings

1. **E2E-trained ResNet-50 surpasses foundation models**: on PANDA, E2E ResNet-50 (78.83%) > UNI (76.37%) > GigaPath (75.72%); on BRCA subtype classification, E2E ResNet-50 (95.17%) > GigaPath+RRTMIL (94.82%).
2. **MIL design has a far greater impact on E2E training than sampling strategy**: the gap between attention-based and random sampling is small (93.14 vs. 92.72), whereas the gap across different MIL designs is large (89.23 vs. 93.97).
3. **Sparsity is the root cause**: ABMIL's extreme sparsity (80) causes E2E optimization collapse; TransMIL partially alleviates this but its global attention is distracted by redundant instances; ABMILX achieves the optimal balance with moderate sparsity (36).
4. **Large inference speed advantage**: E2E ResNet-18 inference takes 1.7 s/slide vs. 83 s/slide for GigaPath, nearly 50× faster.
5. **External validation**: transferring from TCGA (training) to CPTAC (testing), E2E ResNet-50 (85.19 AUC) is on par with UNI+TransMIL (85.24 AUC), demonstrating generalization comparable to FMs.

## Highlights & Insights

- The paper is the first to identify the overlooked problem of "sparse-attention MIL inducing optimization difficulties under E2E training," with rigorous analysis and thorough experimental validation.
- ABMILX is elegantly designed: multi-head local attention suppresses sparsity-induced errors, while the global A+ module exploits patch similarity to propagate correct attention, all while preserving sparse characteristics.
- The work challenges the prevailing assumption that CPath requires large-scale foundation models: ImageNet-pretrained ResNets trained end-to-end can match FM-level performance.
- The cost analysis is compelling: no additional pretraining is required, total training time is 9 h (on RTX 3090), and inference is 50× faster.

## Limitations & Future Work

- Gains from E2E training on survival analysis tasks are modest, partly due to sampling size constraints; larger sampling budgets merit future investigation.
- The global attention matrix in A+ has $O(s^2)$ complexity, which may become a bottleneck at large sampling scales.
- Only ResNet encoders are evaluated; the effect of E2E training with Transformer-based encoders (e.g., ViT) remains unexplored.
- No comparison is made against recent E2E methods requiring multi-GPU clusters under equivalent computational budgets.

## Related Work & Insights

- This work provides a solid foundation for E2E learning in CPath: it identifies the optimization challenge and offers a simple yet effective solution.
- The design of "attention propagation + learnable sparsity" in ABMILX is generalizable to other weakly supervised tasks that rely on attention mechanisms.
- E2E learning may further benefit from upstream pretraining; combining foundation models with E2E fine-tuning is a natural next step.
- The vicious cycle of sparse attention under E2E training may also arise in other weakly supervised MIL settings.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to identify the root cause of E2E optimization difficulty and propose an elegant solution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six datasets, three task types, external validation, comprehensive ablations, and cost comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, in-depth analysis, and highly informative figures and tables.
- Value: ⭐⭐⭐⭐⭐ Has the potential to shift the CPath research paradigm away from "FM-centric, E2E-neglected" approaches.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Momentum Memory for Knowledge Distillation in Computational Pathology](../../CVPR2026/medical_imaging/momentum_memory_for_knowledge_distillation_in_computational_pathology.md)
- [\[NeurIPS 2025\] ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression](imagenet-trained_cnns_are_not_biased_towards_texture_revisiting_feature_reliance.md)
- [\[NeurIPS 2025\] THUNDER: Tile-level Histopathology image UNDERstanding benchmark](thunder_tile-level_histopathology_image_understanding_benchmark.md)
- [\[NeurIPS 2025\] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling](few-shot_learning_from_gigapixel_images_via_hierarchical_vision-language_alignme.md)
- [\[ICCV 2025\] AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images](../../ICCV2025/medical_imaging/aczerots_active_learning_for_zeroshot_tissue_segmentation_in.md)

</div>

<!-- RELATED:END -->
