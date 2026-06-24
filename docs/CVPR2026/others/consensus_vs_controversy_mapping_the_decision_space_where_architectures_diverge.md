---
title: >-
  [Paper Note] Consensus vs. Controversy: Mapping the Decision Space Where Architectures Diverge
description: >-
  [CVPR 2026][Architectural Discrepancy Analysis] The authors perform "disagreement forensics" on ImageNet using 12 pre-trained models from three major families (CNN, ViT, and MLP-Mixer). They find that while overall accuracies are nearly identical (mean 79.9%), architectural differences are concentrated in the most controversial 10% of images. This "controversial subset" exhibits ~4.5x higher disagreement than the "consensus subset," and intra-family consistency is significant…
tags:
  - "CVPR 2026"
  - "Architectural Discrepancy Analysis"
  - "Controversy Space"
  - "Ensemble Construction"
  - "Inductive Bias"
  - "ImageNet"
date: 2026-05-08
content_hash: 405faa8f1289dd49
---

# Consensus vs. Controversy: Mapping the Decision Space Where Architectures Diverge

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Consensus_vs._Controversy_Mapping_the_Decision_Space_Where_Architectures_Diverge_CVPR_2026_paper.html)  
**Code**: None  
**Area**: Model Analysis / Architectural Inductive Bias  
**Keywords**: Architectural Discrepancy Analysis, Controversy Space, Ensemble Construction, Inductive Bias, ImageNet

## TL;DR
The authors perform "disagreement forensics" on ImageNet using 12 pre-trained models from three major families (CNN, ViT, and MLP-Mixer). They find that while overall accuracies are nearly identical (mean 79.9%), architectural differences are concentrated in the most controversial 10% of images. This "controversial subset" exhibits ~4.5x higher disagreement than the "consensus subset," and intra-family consistency is significantly higher than inter-family, providing actionable guidance for model selection and ensemble construction.

## Background & Motivation
**Background**: CNNs, Vision Transformers, and MLP-Mixers originate from entirely different principles (local convolution, global attention, and pure MLP mixing), yet their top-1 accuracies on ImageNet have highly converged (72–84%). This leads to the perception that "architecture choice may no longer be critical."

**Limitations of Prior Work**: Aggregate accuracy, a single metric, masks the differences between models. Two models with 80% accuracy might fail on completely different images, but aggregate metrics fail to reveal this. Existing research has begun to challenge this "aggregate-only" view: Meding et al. found that 46% of ImageNet consists of "trivial images" (correctly classified by all) and 11.5% are "impossible images" (failed by all), with only 42.5% capable of distinguishing models. Conwell et al. observed across 224 models that "training recipes affect brain alignment more than architecture."

**Key Challenge**: While previous works ask which images are difficult for single models or whether aggregate similarity hides differences, **none have systematically characterized exactly on which images different architectures diverge**. In other words, where does the inductive bias of an architecture manifest? This "decision space where divergence occurs" has never been explicitly mapped.

**Goal**: To treat "disagreement" between models as signal rather than noise, explicitly mapping the "controversy space" within the image distribution—precisely locating and quantifying where architectural differences concentrate.

**Key Insight**: The core observation is that "not all images are equally informative for understanding architectural differences." By partitioning images into high-disagreement and low-disagreement tails based on a "controversy score," family-specific differences can be precisely isolated.

**Core Idea**: Instead of training new models, the authors utilize predictions from a set of off-the-shelf pre-trained models. By slicing the top-10% controversial subset and bottom-10% consensus subset based on per-image disagreement scores, they use statistical metrics to prove that **architectural divergence is highly localized and concentrated on a small fraction of high-information-density controversial images**.

## Method

### Overall Architecture
This is an analytical/empirical paper that does not involve training. The "forensics framework" consists of three steps: ① Constructing an ensemble of 12 models across three major families; ② Calculating a "disagreement score" for every image in the ImageNet validation set and slicing them into controversial/consensus subsets; ③ Performing family consistency, diversity, ensemble potential, calibration, and class-level analyses. The core involves defining two metrics and performing analyses around them.

**Problem Setup and Two Core Metrics**: Given $M=12$ pre-trained models $\{f_m\}$, each mapping an image to a probability distribution over $K=1000$ classes. For image $x_i$, the top-1 predicted class of the $m$-th model is $\hat{y}_i^{(m)}=\arg\max_k p_{i,k}^{(m)}$.

- **Agreement Score**: The proportion of models predicting the same top-1 class $A_i=\max_k \frac{1}{M}\sum_{m=1}^{M}\mathbb{1}[\hat{y}_i^{(m)}=k]$, ranging in $[1/M, 1]$. Higher values indicate consensus.
- **Disagreement Measure**: The standard deviation of the "maximum probability (confidence)" across models $D_i=\mathrm{std}(\{\max_k p_{i,k}^{(m)}\}_{m=1}^{M})$. This measures the dispersion of "highest confidence" among models—even if models share the same top-1 label, significant differences in confidence contribute to high disagreement.

**Controversial/Consensus Splitting and Controversy Score**: Images are sorted by $D_i$, where the highest $\alpha\%$ form the "controversial set" $\mathcal{C}_\text{contro}$ and the lowest $\alpha\%$ form the "consensus set" $\mathcal{C}_\text{cons}$ (fixed at $\alpha=10$). The **Controversy Score (CS)** quantifies the separation:

$$\mathrm{CS}=\frac{\mathrm{mean}(D_i: i\in\mathcal{C}_\text{contro})}{\mathrm{mean}(D_i: i\in\mathcal{C}_\text{cons})}$$

As this is a pure analysis workflow without a trainable pipeline, no architecture diagram is provided.

### Key Designs

**1. Cross-family 12-model Ensemble: Enabling Comparable Family Attribution**
To study differences between "architectural families," the ensemble must cover diverse architectures while having multiple representatives per family to distinguish architectural bias from individual model variance. The authors selected from `timm`: 7 CNNs (ResNet-50/101, Wide-ResNet-50, EfficientNet-B0/B2, MobileNetV3-Large, ConvNeXt-Tiny), 3 ViTs (ViT-Base/16, ViT-Small/16, DeiT-Base), and 2 MLPs (MLP-Mixer-B/16, gMLP-S16). All models use **pre-trained weights without fine-tuning**, with standardized 224x224 input, ImageNet normalization, and FP16 inference. Multiple members per family allow for the distinction between intra-family and inter-family consistency.

**2. Confidence Std. Dev. as Per-image Disagreement Score: Scalarizing Architecture Differences**
This is the core metric of the framework. $D_i$ is chosen over simple label-counting because it captures "soft disagreement." Even if all models predict the same top-1 label, if their confidence levels vary significantly (e.g., one at 0.95, another at 0.40), $D_i$ remains high, indicating the image sits at the intersection of different architectural comfort zones. The **Controversy Score** summarizes this separation into a single value (measured at 4.46) and is validated for robustness across perturbations.

**3. Family Consistency Matrix + Unsupervised Hierarchical Clustering: Emerging Family Structures**
To prove that divergence is structured along architectural lines rather than being random noise, the authors calculate pairwise agreement rates $\mathrm{Agr}(m_1,m_2)=\frac{1}{N}\sum_i \mathbb{1}[\hat{y}_i^{(m_1)}=\hat{y}_i^{(m_2)}]$. By applying hierarchical clustering to this consistency matrix **without providing family labels**, the resulting dendrogram automatically reconstructs the family structures (ResNets cluster together, ViTs form a separate branch, MLPs are most distant). This confirms that "architectural families" are real similarity groupings in the prediction-behavior space.

**4. ADER Metric: Quantifying Concentrations of Architectural Divergence**
To substantiate the claim that "differences are localized," the authors propose the **ADER (Architectural Divergence Explanation Ratio)**. It measures the proportion of inter-family disagreement that falls within the top 10% controversial images. If divergence were uniform, 10% of images should account for 10% of total disagreement. The observed ADER = 24.6 indicates that the concentration of inter-family disagreement in the controversy set is 24.6 times higher than expected by data volume alone.

### An Example: Origins of Controversy
Consider an image labeled as "controversial": After running softmax on 12 models, suppose ConvNeXt predicts 'palace' with 0.92 confidence, ResNet with 0.55, ViT-Base with 0.38, and MLP-Mixer with 0.30. The standard deviation of max confidences is large $\to$ high $D_i \to$ top-10% controversial set. Simultaneously, their top-1 predicted classes differ (high diversity). Such high-disagreement, high-diversity images are exactly where ensembles benefit most from combining different families. Conversely, for a clear image of a domestic cat where all models have 0.9+ confidence and identical predictions, $D_i$ is extremely low $\to$ consensus set—making the specific architecture or ensemble irrelevant.

## Key Experimental Results

### Main Results: Existence and Concentration of Controversy Space
Core findings across 50,000 ImageNet validation images and 12 models:

| Metric | Consensus Set (bottom 10%) | Controversial Set (top 10%) | Separation Factor | Significance |
|------|------|------|------|------|
| Mean Disagreement $D$ | 0.057 | 0.255 | **4.46×** (Controversy Score) | $p<0.001$ |
| Mean Diversity (Unique top-1 ratio) | 0.045 | 0.185 | ~4.1× | $p<0.001$ |
| Disagreement Std. Dev. | 0.005 | 0.024 | ~4.8× | — |

The disagreement distribution is right-skewed (mean 0.106, std 0.056), with most images cluster in the low-disagreement band (0.05–0.10) while a few extend to 0.30+, confirming that "controversy is highly concentrated." Overall accuracies range from 72.6% to 84.0% (mean 79.9%), but the structure of disagreement varies greatly.

### Family Consistency and Ensemble Analysis

| Analysis | Key Value | Description |
|------|---------|------|
| Intra-family vs. Inter-family Agreement | 83.5% vs. 80.2% | A systemic 3.3pp gap, stable across both controversial and consensus subsets. |
| Between ResNet Variants | >0.88 | Highest intra-family consistency, sharing the residual learning framework. |
| ViT-Base ↔ DeiT-Base | 93.4% | High intra-family consistency for ViTs. |
| ConvNeXt Mean Agreement | 85.3% | Highest overall, as it adopts Transformer-like designs. |
| MLP-Mixer / gMLP Mean Agreement | 78.8% / 83.5% | Lowest; lack of convolutions/attention creates the most unique decision boundaries. |
| Diversity vs. Disagreement Correlation | $r=0.82$ | High disagreement indicates different labels, not just scattered confidence. |
| Jaccard Coverage (6 / 8 models) | 47.8% / 57.1% | Even half the models fail to cover half the controversy set $\to$ diversity is non-redundant. |
| ADER | 24.6 | Inter-family disagreement concentration in 10% controversial images is 24.6x the data ratio. |

### Robustness, Calibration, and Cost-Accuracy Trade-off

| Verification | Result | conclusion |
|------|------|------|
| CS after Isotonic Calibration | 4.46 → **18.65** (ECE 0.089→0.044) | Calibration amplifies separation rather than weakening it; controversy is not a miscalibration artifact. |
| $\alpha$ Threshold Sensitivity (5%–20%) | CS between 4.4–4.8 (peak at 10%) | 10% threshold corresponds to the natural inflection point of the distribution. |
| OOD: ImageNet-A / -R | CS10 = 2.89 / 3.34; ADER10 = 9.82 / 10.14 | The controversy space phenomenon holds under distribution shift. |
| Sequential Gating (ConvNeXt→ResNet→ViT, $\tau=0.50$) | **83.7% Accuracy, avg. 1.18 model calls** | Saves 90% compute with only 0.3pp drop; ECE remains low at 0.028. |
| Class-level Controversy | Top-20 controversial class rates 30–42% | Classes like palace, steel drum, and lipstick are most prone to architectural disagreement. |

### Key Findings
- **Architectural divergence is localized, not uniform**: Only the top-10% of images contribute the vast majority of cross-model variance; on the remaining 90%, the choice of architecture or ensemble is largely negligible. This refutes the intuition that ensembles always need many models—expensive diversified models should only be invoked when uncertainty is detected.
- **Controversy $\neq$ Difficulty, and $\neq$ Label Noise**: In the consensus set, only 9.7% are "always correct trivial images," and in the controversial set, only 1.0% are "always wrong impossible images." Overlap with label error candidates is only 1.0%. Controversy captures a phenomenon of "high-confidence dispersion" independent of noise.
- **Calibration carries architectural fingerprints**: ViT-Small has the lowest ECE (0.020, 74.6% accuracy), while ResNet-101 has the highest (0.155, 81.9% accuracy). Transformers generally calibrate better than CNNs; architecture affects not just "what is predicted," but "how confidently."
- **Ensembles should cross families, not stack similar models**: Diversity is strongly correlated with disagreement ($r=0.82$). Different family models show sub-linear growth in covering the controversial set, implying that combining different families maximizes "collective intelligence" where it matters most.

## Highlights & Insights
- **Elevating "disagreement" from noise to a quantifiable research object**: By using a forensics framework that relies purely on off-the-shelf predictions, the authors turn the qualitative discussion of architectural bias into an empirical conclusion substantiated by statistical significance and cross-dataset replication.
- **Lightweight metrics ADER and Controversy Score are highly reusable**: Given prediction probabilities from any set of models, one can quantify how concentrated differences are and identify the specific samples causing them. This is directly applicable to selecting evaluation subsets or diagnosing new architectures.
- **"Calibration amplifies separation" is a robust counter-argument**: The authors preemptively addressed the concern that disagreement stems from miscalibration by showing CS increases post-calibration.
- **Direct engineering value via sequential gating**: Achieving a 90% reduction in compute for a marginal 0.3pp loss demonstrates how "controversy space" theory translates into efficient adaptive inference.

## Limitations & Future Work
- **Disagreement metric bound to top-1 confidence**: $D_i$ uses the standard deviation of max probabilities. The authors note that top-5 Jaccard disagreement has near-zero correlation ($r=0.012$), suggesting $D_i$ captures only one facet of architectural difference.
- **Lack of explanation for "why" images are controversial**: The framework precisely locates controversial images but remains statistical. It does not definitively attribute controversy to texture vs. shape bias, global vs. local info, or viewpoint anomalies.
- **Imbalance in model families**: The ensemble includes 7 CNNs, 3 ViTs, and 2 MLPs. This imbalance might affect the comparability of intra-family consistency rates, a point not fully discussed.
- **Scope limited to classification and ImageNet**: While OOD robustness was verified on ImageNet-A/-R, the generalizability to detection, segmentation, or generation remains unknown.

## Related Work & Insights
- **vs. Meding et al. (Dichotomous Data Difficulty)**: While they characterize image difficulty (trivial/impossible), this work characterizes architectural disagreement. The two are complementary—one focuses on single-model success/failure, the other on cross-model divergence.
- **vs. Conwell et al. (224 Models for Brain Alignment)**: They found training recipes mask architectural differences in aggregate; this work proves that architectural differences exist but are concentrated in specific subsets, providing the answer to "where" it matters.
- **vs. Geirhos et al. (Texture vs. Shape Bias)**: Geirhos used controlled synthetic data; this work locates bias manifestation within natural image distributions, a more direct approach for real-world deployment.

## Rating
- Novelty: ⭐⭐⭐⭐ Mapping "model disagreement" as a quantified "controversy space" with metrics like ADER/CS is a novel perspective; however, technical depth is limited as it uses existing models and basic statistics.
- Experimental Thoroughness: ⭐⭐⭐⭐ 12 models on the full ImageNet validation set, covering family structures, diversity, calibration, sensitivity, OOD, and gating.
- Writing Quality: ⭐⭐⭐⭐ Clear arguments, rigorous metric definitions, and convincing counter-argument experiments.
- Value: ⭐⭐⭐⭐ Provides actionable guidance for model selection and efficient ensembles (cross-family ensembles + adaptive gating saving 90% compute).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond Euclidean Gossip: KL-Barycentric Consensus on Heterogeneous and Imbalanced Images](beyond_euclidean_gossip_kl-barycentric_consensus_on_heterogeneous_and_imbalanced.md)
- [\[AAAI 2026\] Bayesian Network Structural Consensus via Greedy Min-Cut Analysis](../../AAAI2026/others/bayesian_network_structural_consensus_via_greedy_min-cut_analysis.md)
- [\[CVPR 2026\] HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition](hypevpr_exploring_hyperbolic_space_for_perspective_to_equirectangular_visual_pla.md)
- [\[ICML 2026\] Decision Tree Learning on Product Spaces](../../ICML2026/others/decision_tree_learning_on_product_spaces.md)
- [\[CVPR 2026\] InstantRetouch: Efficient and High-Fidelity Instruction-Guided Image Retouching with Bilateral Space](instantretouch_efficient_and_high-fidelity_instruction-guided_image_retouching_w.md)

</div>

<!-- RELATED:END -->
