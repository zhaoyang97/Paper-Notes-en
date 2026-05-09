---
title: >-
  [Paper Note] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation
description: >-
  [CVPR 2026][Medical Imaging][Semi-supervised segmentation] This paper proposes SCDL (Semantic Class Distribution Learning), a plug-and-play module that learns structured class-conditional feature distributions and aligns them bidirectionally with learnable class proxies via Class Distribution Bidirectional Alignment (CDBA). Combined with Semantic Anchor Constraints (SAC), which leverage annotated data to guide proxies toward correct semantics, SCDL mitigates both supervision bias and feature representation bias in semi-supervised medical image segmentation (SSMIS), achieving notable improvements on tail-class organs.
tags:
  - CVPR 2026
  - Medical Imaging
  - Semi-supervised segmentation
  - class imbalance
  - distribution learning
  - proxy distribution
  - semantic anchors
date: 2026-05-08
content_hash: 75207e468d388d43
---

# Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.05202](https://arxiv.org/abs/2603.05202)
**Code**: [GitHub](https://github.com/Zyh55555/SCDL)
**Area**: Medical Imaging
**Keywords**: Semi-supervised segmentation, class imbalance, distribution learning, proxy distribution, semantic anchors

## TL;DR

This paper proposes SCDL (Semantic Class Distribution Learning), a plug-and-play module that learns structured class-conditional feature distributions and aligns them bidirectionally with learnable class proxies via Class Distribution Bidirectional Alignment (CDBA). Combined with Semantic Anchor Constraints (SAC), which leverage annotated data to guide proxies toward correct semantics, SCDL mitigates both supervision bias and feature representation bias in semi-supervised medical image segmentation (SSMIS), achieving notable improvements on tail-class organs.

## Background & Motivation

1. **Background**: Semi-supervised medical image segmentation (SSMIS) trains on a small amount of labeled data combined with abundant unlabeled data. Mainstream approaches include consistency regularization, contrastive learning, and pseudo-labeling. However, medical image datasets commonly exhibit severe class imbalance—large organs (e.g., liver) occupy the majority of pixels, while small organs (e.g., esophagus, adrenal glands) contribute very few.

2. **Limitations of Prior Work**: The combination of class imbalance and semi-supervised mechanisms introduces bias at two levels. (1) *Supervision signal bias*: pixel-level gradients dominated by large classes and the self-reinforcing nature of pseudo-labels both skew supervision toward head classes. (2) *Feature representation bias*: existing methods (re-weighting, output calibration) operate only at the loss or output layer, providing no direct constraints on class-conditional feature distributions, causing head-class features to be compact while tail-class features remain dispersed and are "absorbed" by head classes in the feature space.

3. **Key Challenge**: Unlabeled data is primarily used for local consistency regularization and rarely to explicitly correct the skew in class-conditional feature distributions—thus unlabeled data fails to help minority classes establish adequate feature representations, and imbalance persists.

4. **Goal**: To directly alleviate representation bias caused by class imbalance at the feature-space level, rather than addressing it only at the loss or output layer.

5. **Key Insight**: Learn a proxy distribution (Gaussian) for each semantic class, enforce bidirectional alignment so that embeddings are attracted to their corresponding proxy while proxies are repelled from non-target embeddings, and use semantic anchors from annotated regions to provide correct semantic supervision for the proxies.

6. **Core Idea**: By learning class-conditional proxy distributions and enforcing bidirectional alignment, SCDL directly reshapes the class distribution structure in the feature space, ensuring that minority classes also receive stable representation learning signals.

## Method

### Overall Architecture

SCDL is integrated into existing semi-supervised segmentation networks as a plug-and-play module. It comprises two core components: CDBA, which establishes a learnable proxy distribution for each class in the embedding space output by the encoder and enforces bidirectional alignment between embeddings and proxies; and SAC, which extracts semantic anchors from annotated regions to supervise proxy learning. Structured prior information constructed via proxy sampling is then injected into each decoder stage to enhance tail-class feature representations.

### Key Designs

1. **Class Distribution Bidirectional Alignment (CDBA)**

   - **Function**: Learn structured class-conditional feature distributions to mitigate representation bias.
   - **Mechanism**: Each semantic class $c$ is modeled by a learnable Gaussian proxy distribution: $p(u|c) = \mathcal{N}(\mu_c, \text{diag}(\sigma_c^2))$. A soft assignment of each token embedding to each proxy is computed as $P(c|z_{i,l}) = \text{softmax}_c(\cos(z_{i,l}, \mu_c))$. Bidirectional alignment consists of: (1) *Embedding-to-Proxy (E2P)*: $\mathcal{L}_{E2P} = \sum P(c|z) \cdot [1 - \cos(z, \mu_c)]$, pushing embeddings toward their soft-assigned proxies; (2) *Proxy-to-Embedding (P2E)*: $\mathcal{L}_{P2E} = \frac{1}{C}\sum \exp(-(\mathcal{E}_c^+ - \mathcal{E}_c^-))$, encouraging each proxy to discriminate between embeddings belonging and not belonging to that class.
   - **Design Motivation**: Soft assignments allow each embedding to influence the gradient updates of multiple proxies, eliminating the effect of class frequency differences—even when minority-class pixels are scarce, their proxies continue to receive learning signals through soft assignments. Bidirectional alignment ensures proxies are both attractive (E2P pulls embeddings closer) and discriminative (P2E induces repulsion).

2. **Proxy Sampling and Feature Enhancement**

   - **Function**: Utilize the learned proxy distributions to provide structured semantic priors to the downstream decoder.
   - **Mechanism**: Three types of priors are constructed: (1) *Distribution-weighted prior* $\mathbf{r}^{dist}$: $S$ samples are drawn from each proxy distribution; the average cosine similarity between each embedding and the sampled points is used as a weight to form a weighted combination of proxy means. (2) *Center similarity prior* $\mathbf{r}^{center}$: cosine similarities between embeddings and proxy means are directly used as weights for a deterministic combination. (3) *Token sampling prior* $\mathbf{z}^{sam}$: local perturbation sampling per token for robustness augmentation. The three priors are concatenated and injected into each decoder stage via a lightweight projection layer.
   - **Design Motivation**: The distribution-weighted prior retains variance information (uncertainty-awareness), the center prior provides a complementary deterministic signal, and their combination enables both head and tail classes to contribute effectively.

3. **Semantic Anchor Constraints (SAC)**

   - **Function**: Provide ground-truth class semantic guidance for randomly initialized proxy distributions.
   - **Mechanism**: For each class, class-aware embeddings are extracted from annotated regions by masking non-target regions with ground-truth masks before passing through the encoder; their mean serves as the semantic anchor: $\text{anchor}_c = \frac{1}{|\mathcal{Z}_c|}\sum_{z \in \mathcal{Z}_c} z$. A cosine similarity loss then aligns each proxy mean with its corresponding anchor: $\mathcal{L}_{SAC} = \frac{1}{C}\sum [1 - \cos(\mu_c, \text{anchor}_c)]$. Anchors are detached during backpropagation, ensuring SAC updates only the proxies without affecting the encoder.
   - **Design Motivation**: Without semantic constraints, randomly initialized proxies may learn incorrect class correspondences. SAC leverages the "certain signal" from limited labeled data to anchor the proxies; even sparse annotations suffice—what matters is that the anchor direction is correct, and precision can be further refined during training.

### Loss & Training

Total loss = baseline segmentation loss + $\mathcal{L}_{E2P}$ + $\mathcal{L}_{P2E}$ + $\mathcal{L}_{SAC}$. Weight decay for the SCDL module is set to 1e-4. Other configurations vary with the baseline method (e.g., GenericSSL, DHC, GA-CPS). Batch size is 4; training is performed on NVIDIA A40 GPUs.

## Key Experimental Results

### Main Results

Results on Synapse (20% labeled) and AMOS (5% labeled) datasets:

| Method | Synapse DSC↑ | Synapse ASD↓ | AMOS DSC↑ | AMOS ASD↓ |
|--------|-------------|-------------|----------|----------|
| GenericSSL baseline | 55.94 | 6.14 | 35.73 | 45.82 |
| SCDL-GenericSSL | **58.90 (+2.96)** | **5.79** | **47.35 (+11.62)** | **22.84** |
| DHC baseline | 46.16 | 10.04 | 40.11 | 40.65 |
| SCDL-DHC | **49.17 (+3.01)** | 10.59 | **49.28 (+9.17)** | **17.47** |
| GA-CPS baseline | 66.29 | 5.44 | 50.90 | 13.77 |
| SCDL-GA-CPS | **67.50 (+1.21)** | **3.32** | **61.57 (+10.67)** | 10.08 |
| GA-MagicNet baseline | 66.00 | 3.42 | 59.15 | 8.66 |
| SCDL-GA-MagicNet | **66.75 (+0.75)** | 3.65 | **62.16 (+3.01)** | **5.65** |

Notable gains on tail-class organs (Synapse, SCDL-DHC vs. DHC):

| Organ | DHC | SCDL-DHC | Gain |
|-------|-----|----------|------|
| Portal and Splenic Vein (PSV) | 30.7 | 42.6 | +11.9 |
| Esophagus (Es) | 14.7 | 23.5 | +8.8 |
| Right Adrenal Gland (RAG) | 27.9 | 36.7 | +8.8 |

More extreme recovery on AMOS (SCDL-DHC): right adrenal gland 0%→33.9%, left adrenal gland 0%→30.3%.

### Ablation Study

On Synapse (GA-CPS baseline):

| Configuration | DSC↑ | ASD↓ | Note |
|---------------|------|------|------|
| Baseline | 66.29 | 5.44 | GA-CPS |
| + CDBA | 66.77 (+0.48) | 6.24 | DSC improves but ASD increases |
| + CDBA + SAC | **67.50 (+1.21)** | **3.32** | ASD drops sharply by 2.92 upon adding SAC |

### Key Findings

- CDBA alone improves DSC but may degrade ASD (boundary quality); the addition of SAC is critical—it not only further improves DSC but also substantially improves boundary accuracy.
- Gains from SCDL are concentrated on tail classes and small organs: on AMOS with 5% labeled data, DHC's right/left adrenal gland Dice recovers from 0% to 33.9%/30.3%, demonstrating that SCDL effectively prevents extreme minority classes from being entirely neglected.
- Improvement margins are moderate on strong baselines (e.g., GA-MagicNet DSC=66.00, +0.75%) but substantial on weaker ones (GenericSSL AMOS +11.62%), indicating that SCDL is most effective at correcting severe class bias.
- ASD improvement is especially pronounced after adding SAC (6.24→3.32), suggesting that semantic anchor constraints contribute to better boundary geometric quality.

## Highlights & Insights

- **Plug-and-play design**: SCDL integrates seamlessly into any existing SSMIS method without modifying the baseline architecture, greatly enhancing its practical utility.
- **Soft assignment eliminates class-frequency bias**: Unlike hard assignment, each embedding influences all proxies through soft weights, ensuring minority-class proxies continuously receive gradient signals even under extreme scarcity.
- **Complementary design of three priors**: The distribution-weighted prior accounts for variance (uncertainty), the center prior accounts for the mean (determinism), and the token sampling prior enhances robustness—an insightful combination.
- **Using unlabeled data for distribution-level learning** rather than solely for consistency regularization represents an important paradigm shift: unlabeled data participates in modeling global class distributions.

## Limitations & Future Work

- Proxies adopt an isotropic Gaussian assumption (diagonal covariance matrix), which may lack the flexibility to represent complex class boundary shapes.
- The semantic anchor in SAC is computed as a simple mean, which may be insufficient for multi-modal distributions (e.g., large appearance variation of an organ across different imaging planes).
- In the ablation study, CDBA alone causes ASD to increase, suggesting that distribution alignment without semantic supervision may introduce instability.
- Validation is limited to CT multi-organ segmentation; experiments on other modalities such as MRI, pathology, and retinal imaging are absent.

## Related Work & Insights

- **vs. DHC**: DHC employs dynamic hybrid curriculum learning to address semi-supervised imbalance; SCDL-DHC improves upon it by 3%+ in DSC with larger gains on tail classes.
- **vs. GA-MagicNet/GA-CPS**: The GA series uses geometry-aware augmentation to handle imbalance; SCDL provides an orthogonal distribution-level solution that can be stacked with these approaches.
- **vs. CLD**: CLD applies contrastive distribution learning but primarily operates at the output layer; SCDL directly constrains class-conditional distributions in the embedding space.

## Rating

- Novelty: ⭐⭐⭐⭐ Bidirectional alignment of class proxy distributions combined with semantic anchor constraints is a novel combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic validation across two datasets and four baseline methods, though non-CT modalities are absent.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; the three-level bias analysis (supervision / representation / distribution) is thorough.
- Value: ⭐⭐⭐⭐ The plug-and-play module offers direct practical value to the semi-supervised medical segmentation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[CVPR 2026\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[CVPR 2026\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [\[CVPR 2026\] Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](adaptation_of_weakly_supervised_localization_in_histopathology_by_debiasing_pred.md)

</div>

<!-- RELATED:END -->
