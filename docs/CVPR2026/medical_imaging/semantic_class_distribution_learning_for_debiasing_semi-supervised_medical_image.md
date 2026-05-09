---
title: >-
  [Paper Note] SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation
description: >-
  [CVPR 2026][Medical Imaging][Semi-supervised segmentation] This paper proposes SCDL, a plug-and-play semantic class distribution learning framework that addresses supervision bias and representation imbalance in semi-supervised medical image segmentation (SSMIS) via two components: Class Distribution Bidirectional Alignment (CDBA), which learns structured class-conditional feature distributions through proxy distributions, and Semantic Anchor Constraint (SAC), which guides proxy distributions toward true class semantics. SCDL achieves state-of-the-art performance on minority class segmentation.
tags:
  - CVPR 2026
  - Medical Imaging
  - Semi-supervised segmentation
  - class imbalance
  - distribution alignment
  - semantic anchor
  - plug-and-play
date: 2026-05-08
content_hash: d7e219d9418c158a
---

# SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.05202](https://arxiv.org/abs/2603.05202)
**Code**: [github.com/Zyh55555/SCDL](https://github.com/Zyh55555/SCDL)
**Area**: Medical Image Segmentation / Semi-Supervised Learning
**Keywords**: Semi-supervised segmentation, class imbalance, distribution alignment, semantic anchor, plug-and-play

## TL;DR

This paper proposes SCDL, a plug-and-play semantic class distribution learning framework that addresses supervision bias and representation imbalance in semi-supervised medical image segmentation (SSMIS) via two components: Class Distribution Bidirectional Alignment (CDBA), which learns structured class-conditional feature distributions through proxy distributions, and Semantic Anchor Constraint (SAC), which guides proxy distributions toward true class semantics. SCDL achieves state-of-the-art performance on minority class segmentation.

## Background & Motivation

**Background**: Semi-supervised medical image segmentation (SSMIS) leverages unlabeled data to reduce annotation burden, commonly through consistency regularization and contrastive representation learning.

**Limitations of Prior Work**: Medical segmentation data inherently suffers from severe class imbalance, with dominant organs occupying far more pixels than minor ones. Combined with semi-supervised mechanisms, this produces two compounding problems: (i) **supervision bias**—self-generated pseudo-labels and consistency constraints further reinforce learning of head classes, leaving tail classes underrepresented; (ii) **representation imbalance**—head-class features become compact while tail-class features drift into head-class-dominated regions, blurring class boundaries.

**Key Challenge**: Existing debiasing methods (re-weighting, output calibration) operate only at the loss or output level, without directly constraining class-conditional feature distributions. Unlabeled data is predominantly used for local consistency regularization rather than explicitly correcting skewed class-conditional feature distributions.

**Key Insight**: Learn a trainable proxy distribution for each semantic class in the embedding space, and use bidirectional alignment to provide consistent learning signals across all classes, including minorities.

**Core Idea**: Elevate debiasing from the loss/output level to the feature distribution level, reshaping class-conditional feature structure through bidirectional proxy distribution alignment and semantic anchor constraints.

## Method

### Overall Architecture

Existing segmentation network → encoder output embeddings $\mathbf{Z} \in \mathbb{R}^{B \times L \times D}$ → CDBA module maintains a learnable proxy distribution $\mathcal{N}(\mu_c, \text{diag}(\sigma_c^2))$ per class and performs bidirectional alignment → proxy samples generate three types of priors (distribution-weighted / center-similarity / token-sampled) injected into the decoder at multiple stages → SAC extracts semantic anchors from annotated regions to align proxies toward true semantics → plug-and-play, requiring no modification to the baseline training pipeline.

### Key Designs

1. **Class Distribution Bidirectional Alignment (CDBA)**

   - **Function**: Learns a proxy distribution for each class in the embedding space and pulls embeddings and proxies toward each other through bidirectional alignment.
   - **Mechanism**: Each class is modeled as a Gaussian proxy $p(u|c) = \mathcal{N}(\mu_c, \text{diag}(\sigma_c^2))$. Soft assignment $P(c|z) = \text{softmax}_c(\cos(z, \mu_c))$ allows each embedding to associate with multiple classes. The E2P loss $\mathcal{L}_{E2P} = \sum P(c|z) [1 - \cos(z, \mu_c)]$ pulls embeddings toward proxies, while the P2E loss $\mathcal{L}_{P2E} = \frac{1}{C}\sum_c \exp(-({\mathcal{E}_c^+ - \mathcal{E}_c^-}))$ drives each proxy to actively identify its own embeddings.
   - **Design Motivation**: Unidirectional alignment (E2P only) allows head-class proxies to dominate gradients, leaving minority-class embeddings still attracted to head-class regions. Bidirectional constraints ensure each proxy actively learns to discriminate its own embeddings, enabling minority-class proxies to form compact distributions as well.

2. **Semantic Anchor Constraint (SAC)**

   - **Function**: Provides reliable semantic supervision for each proxy from labeled data, preventing proxies from drifting away from true class semantics.
   - **Mechanism**: For each class $c$, the mean of encoder embeddings extracted from class-specific regions of annotated images serves as the semantic anchor: $\text{anchor}_c = \frac{1}{|\mathcal{Z}_c|}\sum_{z \in \mathcal{Z}_c} z$ (detached during backpropagation to prevent gradient flow into the encoder). The alignment loss is $\mathcal{L}_{SAC} = \frac{1}{C}\sum_c [1 - \cos(\mu_c, \text{anchor}_c)]$.
   - **Design Motivation**: Proxies are randomly initialized and trained primarily on unlabeled data, leaving them prone to semantic drift. Semantic anchors from labeled data provide the only reliable class-level supervisory signal.

3. **Proxy-Sampled Feature Augmentation**

   - **Function**: Samples from the learned proxy distributions to construct three complementary priors injected into different decoder stages.
   - **Mechanism**: (i) Distribution-weighted prior—samples $S$ times from each proxy to compute a weighted mean $\mathbf{r}^{dist}$, capturing distribution-level structure including variance; (ii) Center-similarity prior—uses proxy means directly for weighted aggregation $\mathbf{r}^{center}$, ignoring variance to provide a complementary signal; (iii) Token-sampled prior—locally perturbed sampling for robustness. The three priors are concatenated and injected into the decoder via a lightweight projection.
   - **Design Motivation**: Using only proxy means ignores distributional shape, while using only samples introduces noise. The three complementary priors ensure that both head and tail classes contribute effectively to segmentation.

### Loss & Training

Total loss = baseline segmentation loss + $\mathcal{L}_{E2P}$ + $\mathcal{L}_{P2E}$ + $\mathcal{L}_{SAC}$. SCDL is integrated as a plug-and-play module into four baselines: GenericSSL, DHC, GA-CPS, and GA-MagicNet. Weight decay is set to 1e-4. Experiments are conducted on an NVIDIA A40 GPU with batch size 4.

## Key Experimental Results

### Main Results

| Method | Synapse 20% DSC↑ | Synapse ASD↓ | AMOS 5% DSC↑ | AMOS ASD↓ |
|---|---|---|---|---|
| GenericSSL | 55.94 | 6.14 | 35.73 | 45.82 |
| **SCDL-GenericSSL** | **58.90** (+2.96) | **5.79** | **47.35** (+11.62) | **22.84** |
| GA-CPS | 66.29 | 5.44 | 50.90 | 13.77 |
| **SCDL-GA-CPS** | **67.50** (+1.21) | **3.32** | **61.57** (+10.67) | **10.08** |
| GA-MagicNet | 66.00 | 3.42 | 59.15 | 8.66 |
| **SCDL-GA-MagicNet** | **66.75** (+0.75) | **3.65** | **62.16** (+3.01) | **5.65** |
| DHC | 46.16 | 10.04 | 40.11 | 40.65 |
| **SCDL-DHC** | **49.17** (+3.01) | **10.59** | **49.28** (+9.17) | **17.47** |

### Ablation Study (Per-Class Dice on Synapse)

| Class | GA-CPS | SCDL-GA-CPS | Change | Type |
|---|---|---|---|---|
| Gallbladder (Ga) | 26.7 | 25.4 | -1.3 | Tail |
| Esophagus (Es) | 40.2 | 38.7 | -1.5 | Tail |
| Pancreas (PA) | 45.5 | **49.4** | **+3.9** | Tail |
| Right Adrenal Gland (RAG) | 44.7 | **49.2** | **+4.5** | Tail |
| Spleen (Sp) | 85.5 | **88.2** | +2.7 | Head |
| Liver (Li) | 92.7 | **93.7** | +1.0 | Head |

### Key Findings

- The most significant gains occur under AMOS 5% annotation (DSC up to +11.62%), confirming that SCDL is most advantageous when labeled data is extremely scarce.
- On the DHC baseline, ASD decreases from 40.65 to 17.47 ($\downarrow 23.18$), indicating substantial improvement in boundary-level accuracy.
- Tail classes such as pancreas and adrenal glands show notable improvements (+3.9/+4.5 Dice), though a few tail classes (gallbladder, esophagus) exhibit marginal declines, suggesting that proxy learning for extremely small structures still requires more supervisory signal.
- Consistent gains across four diverse baselines validate the generalizability of the framework.

## Highlights & Insights

- Elevating debiasing from the loss/output level to the feature distribution level is conceptually more fundamental: rather than instructing the model to "ignore" imbalance, SCDL explicitly restructures the feature space to impose organized class-conditional distributions.
- The bidirectional alignment design is elegant—E2P directs embeddings toward the correct proxy, while P2E drives each proxy to actively discriminate its own embeddings; the two optimization objectives are complementary.
- Detaching encoder gradients in the semantic anchor computation is a critical implementation detail—it updates only the proxies without interfering with the encoder's representation learning.

## Limitations & Future Work

- A few extremely small tail classes (e.g., gallbladder Dice declining from 26.7 to 25.4) remain unresolved, potentially requiring a minimum sample size guarantee.
- The Gaussian assumption for proxy distributions may be overly simplistic; mixture models or normalizing flows could capture more complex class-conditional structure.
- Validation is limited to CT multi-organ segmentation; other modalities such as pathology and fundus imaging remain untested.
- The number of proxies is tied to the number of classes, which may lack flexibility for fine-grained sub-class scenarios.

## Related Work & Insights

- **vs. CLD/SimiS**: Contrastive learning methods operate at the representation level but do not explicitly model class-conditional distributions; SCDL's proxy distributions provide more structured constraints.
- **vs. GA-MagicNet/GA-CPS**: Gradient aggregation methods reweight gradients to mitigate imbalance; SCDL complementarily reshapes feature structure at the distributional level.
- **vs. DHC**: Dynamic hybrid consistency debiases at the pseudo-label level, whereas SCDL constrains directly at the embedding level; the two approaches are orthogonal and can be combined.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The distribution-level debiasing perspective is novel; the combination of bidirectional alignment and semantic anchors is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive validation across four baselines and two datasets, including per-class analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Problem analysis is clear; the three-paradigm comparison figure is intuitive.
- **Value**: ⭐⭐⭐⭐ The plug-and-play design is practical and deployment-friendly.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[CVPR 2026\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[CVPR 2026\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [\[CVPR 2026\] Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](adaptation_of_weakly_supervised_localization_in_hi.md)

<!-- RELATED:END -->
