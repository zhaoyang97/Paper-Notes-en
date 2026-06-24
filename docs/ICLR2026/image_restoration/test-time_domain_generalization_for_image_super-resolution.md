---
title: >-
  [Paper Note] Test-Time Domain Generalization for Image Super-Resolution
description: >-
  [ICLR 2026][Image Restoration][Test-Time Domain Generalization] For pixel-level tasks like Super-Resolution (SR), this paper proposes MC-TTDG: training a set of "domain-invariant codebooks + multiple domain-specific codebooks" on the source domain. During testing, target domain features are migrated via pixel-wise nearest neighbor codeword replacement to achieve fine-grained transfer, and a voting strategy selects the most suitable domain-specific codebook. This significantly…
tags:
  - "ICLR 2026"
  - "Image Restoration"
  - "Test-Time Domain Generalization"
  - "Image Super-Resolution"
  - "Multi-Codebook"
  - "Pixel-Level Feature Migration"
  - "Voting Selection"
date: 2026-05-08
content_hash: 81c2988b61806671
---

# Test-Time Domain Generalization for Image Super-Resolution

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=jBuMH3DOPQ](https://openreview.net/forum?id=jBuMH3DOPQ)  
**Code**: https://github.com/ZaizuoTang/MC-TTDG  
**Area**: Image Super-Resolution / Test-Time Domain Generalization / Low-Level Vision  
**Keywords**: Test-Time Domain Generalization, Image Super-Resolution, Multi-Codebook, Pixel-Level Feature Migration, Voting Selection

## TL;DR
For pixel-level tasks like Super-Resolution (SR), this paper proposes MC-TTDG: training a set of "domain-invariant codebooks + multiple domain-specific codebooks" on the source domain. During testing, target domain features are migrated via pixel-wise nearest neighbor codeword replacement to achieve fine-grained transfer, and a voting strategy selects the most suitable domain-specific codebook. This significantly improves cross-domain performance without requiring fine-tuning on the target domain.

## Background & Motivation

**Background**: Test-Time Domain Generalization (TTDG) is a lightweight solution for domain shift. It memorizes the "centroids" of the source distribution during training. During testing, instead of fine-tuning the network, it performs style transfer on target samples to align target feature means/variances with source centroids, enabling the source-trained model to generalize. Compared to traditional Domain Generalization (DG) which only learns invariant representations across multiple source domains and ignores target information, TTDG utilizes target samples during inference without expensive fine-tuning.

**Limitations of Prior Work**: Existing TTDG methods are almost entirely built on style transfer, which is a **global, coarse-grained** operation adjusting only the mean and variance of the entire feature map. While effective for high-level tasks like classification that rely on abstract global representations, it fails for pixel-level prediction tasks like SR. The authors' t-SNE visualization shows that target samples before and after style transfer almost completely overlap, failing to shift the distribution toward the source domain (Fig. 2).

**Key Challenge**: High-level tasks make decisions based on "global style," making statistical alignment sufficient. Conversely, low-level tasks (SR) require independent prediction for each pixel, demanding **pixel-wise fine-grained alignment**. Global style transfer is inherently mismatched with the requirements of low-level vision. Furthermore, existing methods using a single backbone or codebook to represent diverse source domains compress these distributions into a narrow feature space, leading to a **loss of domain-specific information** (Fig. 3a).

**Goal**: To adapt TTDG from high-level to low-level vision (SR), three issues must be addressed: (1) achieving pixel-level fine-grained migration; (2) representing multiple source domains without losing domain-specific information; (3) selecting the correct domain-specific codebook for the target domain during testing.

**Key Insight**: The authors introduce a **codebook**. A codebook consists of discrete codewords naturally suited for "pixel-wise nearest neighbor replacement," a local, discrete migration process that provides the fine granularity missing in style transfer. To the authors' knowledge, this is the first application of codebooks to TTDG and the first TTDG method specifically designed for low-level vision.

**Core Idea**: Utilize a "domain-invariant codebook + multiple domain-specific codebooks" to learn fine-grained representations in the source domain. During testing, apply pixel-wise nearest neighbor replacement for migration and use a voting mechanism to select the optimal domain-specific codebook.

## Method

### Overall Architecture
MC-TTDG splits the workflow into "Server-side Training + Edge-side Testing." In the training phase, a pre-trained SR network (Conv1 + Backbone + Decoder) is loaded and **frozen**. Only the newly added codebooks and decoupling modules are trained to reconstruct source domain features using both "domain-invariant + domain-specific" codebooks. The testing phase does not update network weights: target LR images pass through the same shallow feature extraction and decoupling. The domain-invariant codebook migrates invariant features, while the domain-specific codebook (selected via voting) migrates specific features. Both are summed and fed back into the frozen network to generate the SR image.

The process follows a pipeline: "shallow features → decoupling → dual-path codebook migration → summation → decoding." Training and testing share the same decoupling and migration operators, differing only in that training learns reconstruction across all source codebooks, while testing performs nearest neighbor replacement + voting on a single target image.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input LR Image<br/>(Source Training / Target Testing)"] --> B["Conv1 Shallow Feature fi"]
    B --> C["Representation Learning with Multi-Codebook (RLMC)<br/>Conv2 Decoupling into f_Base + f_Spe"]
    C -->|Invariant Feature| D["Pixel-level Nearest Neighbor Migration<br/>Pixel-wise replacement with closest codeword"]
    C -->|Specific Feature (Test-time)| E["Voting-based Specific Codebook Selection<br/>Multi-codebook pre-migration + Majority vote"]
    D --> F["Summation of two-path features<br/>Return to frozen Backbone+Decoder"]
    E --> F
    F --> G["Output SR Image"]
```

### Key Designs

**1. Representation Learning with Multi-Codebook (RLMC): Fine-grained Source Domain Decoupling**

This design addresses the loss of domain-specific information in single-codebook approaches. Instead of forcing all source domains into one codebook, the authors assign a **domain-specific codebook** $\text{Codebook}^{Spe}_i$ to each source domain, plus a shared **domain-invariant codebook** $\text{Codebook}^{Base}$. Specifically, the LR image from the $i$-th source domain is processed via shallow extraction $f_i = \text{Conv1}(LR^S_i)$, followed by convolutional decoupling: $f^{Base}_i = \text{Conv2}(f_i)$ as the invariant part, and $f^{Spe}_i = f_i - f^{Base}_i$ as the specific residual. Both paths are migrated via their respective codebooks to obtain $f^{BQ}_i$ and $f^{SQ}_i$, then summed and fed into the frozen backbone/decoder to produce $SR^S_i$.

This division of labor—invariant codebook as base and specific codebooks as offsets—allows domains to share common structures while retaining unique details. t-SNE visualization confirms that multi-codebook learning assigns independent feature spaces to each source domain (Fig. 3b), whereas a single codebook clusters them together. The training objective consists of four components: SR reconstruction loss $Loss_{SR}=|SR^S_i - HR^S_i|$, quantization loss $Loss_{Vq}$ for codewords, commitment loss $Loss_{Comm}$ to align entries, and an auxiliary classification loss $Loss_{Cls}=\text{CrossEntropy}(Pre, i)$, totaling $Loss_{All}=Loss_{SR}+\lambda Loss_{Vq}+\beta Loss_{Comm}+\gamma Loss_{Cls}$. Quantization/commitment losses use the gradient stop operator $sg(\cdot)$ to update codewords and layers. The classification branch uses $f^{Spe}_i$ only to prepare a discriminator for test-time voting.

**2. Pixel-level Nearest Neighbor Migration: Replacing Global Styles with Pixel-wise Codewords**

This design targets the coarse granularity of style transfer. Migration is defined as pixel-wise nearest neighbor retrieval: for a pixel feature $f_{x,y}$ at $(x,y)$, the closest codeword in the codebook is found via Euclidean distance and substituted:

$$\text{Transfer}(f_{x,y}, \text{Codebook}) = C_k,\quad k = \arg\min_{C_i \in \text{Codebook}} \|f_{x,y} - C_i\|_2 .$$

During testing, target domain invariant features are migrated using the invariant codebook $f^{BQ}_{Target} = \text{Transfer}(f^{Base}_{Target}, \text{Codebook}^{Base})$, and specific features use the selected specific codebook. Finally, $SR^T = \text{Decoder}(\text{Backbone}(f^{BQ}_{Target} + f^{SQ}_{Target}))$. Unlike style transfer which only modifies mean/variance, every pixel is independently mapped to source codewords. This fine granularity accurately aligns target distributions to the source domain (t-SNE in Fig. 2 shows migrated distributions falling within source clusters). Ablations show that style transfer yields nearly zero improvement over the baseline, while codebook migration provides significant gains.

**3. Voting-based Specific Codebook Selection: Correcting Classification Bias via Pre-migration**

With multiple domain-specific codebooks, the most suitable one must be selected at test time. A direct approach, like MoE, uses a classifier to pick the expert with the highest confidence. However, classifiers trained on source domains often fail on target samples due to domain shift (Table 3 shows Top-1 accuracy dropping to 0.06~0.29). The authors' insight is: rather than trusting a single judgment on **original** target features, first use every domain-specific codebook to perform **pre-migration**. This yields multiple features "already pulled toward specific source domains," which are then fed into the classifier to count votes. The codebook with the highest votes is selected; in case of a tie, it reverts to voting on un-migrated features. This ensemble-like approach stabilizes predictions under distribution shift, improving Top-1 accuracy to 0.35~0.49 and Top-2 accuracy to 0.78~0.88.

## Key Experimental Results

Datasets include DRealSR (P / IMG / Canon / Pan / Sony / DSC branches with distinct distributions) as well as Set5/Set14/B100/Urban/Manga109/DIV2K. Metrics used are PSNR / SSIM / LPIPS. Source domains are P, IMG, and Canon; target testing is on Pan, Sony, and DSC. Architectures include AdaCode, HAT, and MambaIR. The following results use MambaIR.

### Main Results

| Dataset (Target) | Metric | MC-TTDG (Ours) | Prev. SOTA (TTMG) | Gain |
|------------------|--------|----------------|-------------------|------|
| Pan              | PSNR↑  | 31.15          | 30.26             | +0.89 dB |
| Pan              | LPIPS↓ | 0.3593         | 0.4523            | Better |
| Sony             | PSNR↑  | 31.29          | 30.56             | +0.73 dB |
| Sony             | LPIPS↓ | 0.3157         | 0.4069            | Better |
| DSC              | PSNR↑  | 31.21          | 30.70             | +0.51 dB |
| DSC              | LPIPS↓ | 0.3583         | 0.4128            | Better |

Compared to TF-Cal, TSB, DG-PIC, TTDG, and TTMG (all based on coarse-grained style transfer), MC-TTDG leads across all metrics and target branches.

### Ablation Study

| Configuration | Pan PSNR | Sony PSNR | DSC PSNR | Description |
|---------------|----------|-----------|----------|-------------|
| Baseline      | 31.0263  | 30.7220   | 30.9117  | Frozen network, no migration |
| One codebook  | 31.1111  | 31.1411   | 30.9640  | Single codebook: loss of domain info |
| w/o Invariant | 30.9097  | 31.0144   | 30.9043  | Specific only: performance drop |
| Full RLMC     | 31.1571  | 31.2937   | 31.2118  | Invariant + Specific: best results |

| Migration Method         | Pan PSNR | Sony PSNR | DSC PSNR | Description |
|--------------------------|----------|-----------|----------|-------------|
| Baseline                 | 31.0263  | 30.7220   | 30.9117  | — |
| Single-center Style Trans| 31.0262  | 30.7171   | 30.9131  | Near zero gain |
| Multi-center Style Trans | 31.0261  | 30.7166   | 30.9118  | Near zero gain |
| Codebook Mig. (Ours)     | 31.1571  | 31.2937   | 31.2118  | Significant lead |

| Selection Method         | Pan Top1 | Sony Top1 | DSC Top1 | Description |
|--------------------------|----------|-----------|----------|-------------|
| Max Prediction Score     | 0.2907   | 0.0596    | 0.1660   | Classifier fails under shift |
| Voting (Ours)            | 0.3726   | 0.3545    | 0.4922   | Majority vote boosts accuracy |

### Key Findings
- **Granularity is critical for low-level migration**: Style transfer (single- or multi-center) yields almost zero gain. In contrast, pixel-wise codebook migration improves PSNR by 0.13~0.57 dB, proving that granularity is more important than the migration act itself for SR.
- **Invariant codebooks are indispensable**: Removing the invariant codebook causes performance to drop below the baseline in several cases (e.g., Pan 30.91 < 31.03), indicating that a shared base is necessary for effective decoupling.
- **Voting corrects bias**: Under distribution shift, the Top-1 accuracy for Sony was as low as 0.06 using direct scoring. Voting raised this to 0.35, with Top-2 approaching 0.78~0.88, demonstrating that "pre-migration + ensemble voting" effectively counters classifier inaccuracy on target domains.

## Highlights & Insights
- **Codebook as a Migration Operator**: Unlike VQ-VAE/VQGAN which use codebooks for generation or discrete representation, here the codebook is used for "pixel-level nearest neighbor replacement," providing much-needed fine granularity.
- **Base + Offset Decoupling**: Using an invariant codebook as a base and specific codebooks as offsets via a simple residual $f^{Spe}=f_i-f^{Base}$ is elegant and achieves clear domain separation in t-SNE.
- **Voting as Error Correction**: The idea that "source classifiers are untrustworthy on target data, so pre-migrate via experts and vote" is a valuable strategy for any test-time selection scenario under domain shift (e.g., MoE routing).
- **Plug-and-play External Module**: The system is an external addition to a frozen pre-trained SR network, making it ideal for deployments where the core model remains unchanged while codebooks are updated.

## Limitations & Future Work
- **Dependency on Domain Labels**: The method requires multiple source domains with explicit labels to build specific codebooks. Performance gain may diminish with a single source domain or ambiguous domain boundaries.
- **Linear Scalability of Codebooks**: One codebook per source domain means storage and pre-migration computational costs increase linearly with the number of source domains.
- **Modest Gain Magnitude**: While outperforming style-based methods, absolute PSNR gains are in the 0.1~0.9 dB range. Evaluation is focused on camera-specific branches; generalization to severe degradation (real blur, compression) requires further study.
- **Computational Redundancy**: Voting requires running migrations through all specific codebooks, creating overhead that was not fully analyzed in terms of the accuracy-speed trade-off.

## Related Work & Insights
- **vs Traditional DG**: Traditional DG learns invariant representations on source domains only. TTDG (this paper) utilizes target domain information during testing to perform explicit migration.
- **vs Style-based TTDG (TTMG / TTDG / TF-Cal)**: These use global mean/variance alignment for coarse-grained transfer. This paper uses pixel-level codeword replacement, which is far superior for pixel-level tasks like SR.
- **vs MoE Gating**: MoE typically uses a classifier (gate) to pick experts. This paper replaces single-gate scoring with "multi-codebook pre-migration + voting," significantly improving selection accuracy on target domains.

## Rating
- Novelty: ⭐⭐⭐⭐ First TTDG for low-level vision and first use of codebook for TTDG migration.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across architectures (AdaCode/HAT/MambaIR); extensive ablations on components and voting.
- Writing Quality: ⭐⭐⭐⭐ Clear mapping between challenges and solutions; effective use of t-SNE and architectural diagrams.
- Value: ⭐⭐⭐⭐ Establishes a new fine-grained migration paradigm for low-level test-time generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Domain-Adaptive Video Deblurring via Test-Time Blurring](../../ECCV2024/image_restoration/domain-adaptive_video_deblurring_via_test-time_blurring.md)
- [\[ICLR 2026\] Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content](exploring_real-time_super-resolution_benchmarking_and_fine-tuning_for_streaming_.md)
- [\[ICLR 2026\] Learning Domain-Aware Task Prompt Representations for Multi-Domain All-in-One Image Restoration](learning_domain-aware_task_prompt_representations_for_multi-domain_all-in-one_im.md)
- [\[ICLR 2026\] LinearSR: Unlocking Linear Attention for Stable and Efficient Image Super-Resolution](linearsr_unlocking_linear_attention_for_stable_and_efficient_image_super-resolut.md)
- [\[CVPR 2026\] Degradation-Consistent Test-Time Adaptation for All-in-One Image Restoration](../../CVPR2026/image_restoration/degradation-consistent_test-time_adaptation_for_all-in-one_image_restoration.md)

</div>

<!-- RELATED:END -->
