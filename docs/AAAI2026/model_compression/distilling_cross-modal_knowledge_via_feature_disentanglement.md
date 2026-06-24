---
title: >-
  [Paper Note] Distilling Cross-Modal Knowledge via Feature Disentanglement
description: >-
  [AAAI 2026][Model Compression][Cross-modal distillation] This paper proposes Frequency-Decoupled Cross-Modal Knowledge Distillation (FD-CMKD), which decomposes teacher and student features into low-frequency (modality-shared semantics) and high-frequency (modality-specific details) components via Fourier transform, applies strong-consistency MSE and weak-consistency logMSE losses respectively, and introduces scale normalization along with shared classifier alignment to bridge…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Cross-modal distillation"
  - "frequency-domain feature disentanglement"
  - "knowledge transfer"
  - "scale consistency"
  - "shared classifier alignment"
date: 2026-05-08
content_hash: 43990dbbbe9e3b27
---

# Distilling Cross-Modal Knowledge via Feature Disentanglement

**Conference**: AAAI 2026
**arXiv**: [2511.19887](https://arxiv.org/abs/2511.19887)  
**Code**: [GitHub](https://github.com/Johumliu/FD-CMKD)  
**Area**: Model Compression / Cross-Modal Knowledge Distillation
**Keywords**: Cross-modal distillation, frequency-domain feature disentanglement, knowledge transfer, scale consistency, shared classifier alignment

## TL;DR

This paper proposes Frequency-Decoupled Cross-Modal Knowledge Distillation (FD-CMKD), which decomposes teacher and student features into low-frequency (modality-shared semantics) and high-frequency (modality-specific details) components via Fourier transform, applies strong-consistency MSE and weak-consistency logMSE losses respectively, and introduces scale normalization along with shared classifier alignment to bridge the feature space. FD-CMKD consistently outperforms existing distillation methods across multiple cross-modal scenarios including audio–visual, image–text, and semantic segmentation.

## Background & Motivation

**Knowledge distillation has achieved broad success in unimodal settings**: KD leverages a large teacher to supervise a smaller student, yielding strong results on image classification, object detection, and other unimodal tasks, yet direct transfer to cross-modal settings yields substantially degraded performance.

**Growing demand for cross-modal scenarios**: Real-world applications involve multi-modal data from vision, language, and audio, requiring knowledge transfer from a teacher of one modality to a student of another (e.g., visual→audio, text→visual).

**Core bottleneck — representational inconsistency**: Features of different modalities simultaneously encode modality-shared semantic information ("what") and modality-specific detail characteristics ("how"). Applying a uniform strong-alignment loss causes "representational conflict," forcing the student to distort its native modality's expressive capacity.

**Limitations of existing cross-modal distillation methods**: Methods such as C2KD focus primarily on logit-level distillation, neglecting the richer semantic and detail information embedded in intermediate features; they also handle hard samples with misaligned soft labels inadequately and are typically designed for specific scenarios or unidirectional distillation.

**Frequency-domain analysis provides a key insight**: The authors find that after applying Fourier transforms to features, low-frequency components exhibit inter-modal cosine similarities of 0.85–0.91, whereas high-frequency components have similarities near zero (or even negative), indicating that low frequencies carry modality-shared semantics while high frequencies carry modality-specific information.

**Necessity of differentiated distillation strategies**: Given that these two types of information are fundamentally different in nature, a one-size-fits-all alignment approach is inappropriate. Instead, strong consistency constraints should be applied to low-frequency components and weak consistency constraints to high-frequency components, while also addressing inter-modal feature scale discrepancies.

## Method

### Overall Architecture

The FD-CMKD framework consists of four core modules: (1) frequency-domain feature disentanglement, which decomposes teacher and student features into low- and high-frequency components; (2) differentiated frequency-domain distillation, applying distinct losses to each component; (3) scale consistency alignment, which eliminates inter-modal numerical scale discrepancies via feature normalization; and (4) shared classifier alignment, which further reduces distributional differences across modalities within a unified decision space.

### Key Designs

#### 1. Frequency-Decoupled Distillation

- **Function**: Transforms the raw feature $\mathbf{X}^m \in \mathbb{R}^D$ into the frequency domain via DFT, separates it into low-frequency $\mathbf{X}_{\text{low}}^m$ and high-frequency $\mathbf{X}_{\text{high}}^m$ components using binary mask filters, and reconstructs the spatial-domain features via IDFT.
- **Mechanism**: $\mathbf{X}_f^m = \text{DFT}(\mathbf{X}^m)$; the low-pass filter $\mathbf{M}_{\text{low}}$ retains the lower half of frequency components, the high-pass filter $\mathbf{M}_{\text{high}}$ retains the upper half, and $\mathbf{X}_{\text{low}}^m = \text{IDFT}(\mathbf{X}_f^m \cdot \mathbf{M}_{\text{low}})$.
- **Design Motivation**: Experiments confirm that cross-modal cosine similarity of low-frequency features reaches 0.91 on CREMA-D and 0.85 on AVE, substantially higher than that of the original features (0.84/0.74), while high-frequency similarity is near zero, demonstrating that the frequency domain naturally corresponds to the semantic–detail hierarchy.

#### 2. Differentiated Distillation Losses

- **Function**: Applies MSE to low-frequency features for strong consistency and logMSE to high-frequency features for weak consistency.
- **Mechanism**: Low-frequency loss $\mathcal{L}_{\text{low}} = \frac{1}{ND}\|\mathbf{X}_{\text{low}}^a - \mathbf{X}_{\text{low}}^b\|^2$; high-frequency loss $\mathcal{L}_{\text{high}} = \frac{1}{ND}\|\sigma(\mathbf{X}_{\text{high}}^a) - \sigma(\mathbf{X}_{\text{high}}^b)\|^2$, where $\sigma(x) = \text{sign}(x) \cdot \log(1+|x|)$ compresses gradients for large discrepancies.
- **Design Motivation**: High-frequency features contain noise and modality-specific information; the linearly growing gradient of MSE under large errors leads to overfitting to noise. The logMSE gradient plateaus at large differences, permitting high-frequency information to be "softly aligned" rather than forcibly matched.

#### 3. Scale Consistency Loss

- **Function**: Applies mean subtraction followed by L2 normalization to features, i.e., $\text{Std}(\mathbf{X}) = \frac{\mathbf{X} - \bar{\mathbf{X}}}{\|\mathbf{X} - \bar{\mathbf{X}}\|_2}$, eliminating inter-modal numerical range discrepancies.
- **Mechanism**: Mean subtraction can be implemented directly as DC-component filtering in the frequency domain, integrating seamlessly with the frequency-domain disentanglement. The distillation loss becomes $\mathcal{L}_{\text{low}} = \frac{1}{ND}\|\text{Std}(\mathbf{X}_{\text{low}}^a) - \text{Std}(\mathbf{X}_{\text{low}}^b)\|^2$.
- **Design Motivation**: Visualization reveals that audio-modality feature means are significantly higher than those of the visual modality; direct MSE would force the student features to shift toward the teacher's mean, disrupting the original distribution. Normalization refocuses the model on intrinsic discriminative features.

#### 4. Feature Space Alignment via Shared Classifiers

- **Function**: Two shared classifiers $\Phi_h$ and $\Phi_l$ are designed for high- and low-frequency components respectively; both teacher and student features are fed through the same classifiers, and cross-entropy loss aligns the decision boundaries.
- **Mechanism**: $\mathcal{L}_{\text{align}} = \text{CE}(\Phi_h(\mathbf{X}_{\text{high}}^a), y) + \text{CE}(\Phi_h(\mathbf{X}_{\text{high}}^b), y) + \text{CE}(\Phi_l(\mathbf{X}_{\text{low}}^a), y) + \text{CE}(\Phi_l(\mathbf{X}_{\text{low}}^b), y)$.
- **Design Motivation**: Scale alignment alone is insufficient, as the distributional shapes and class boundaries across modalities may still differ. The shared classifiers enforce that features from both modalities are comparable within the same decision space, reducing distributional divergence at the semantic level.

#### 5. Total Loss Function

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{align}} + \lambda_1 \mathcal{L}_{\text{low}} + \lambda_2 \mathcal{L}_{\text{high}}$$

where $\mathcal{L}_{\text{task}}$ is the student's classification cross-entropy loss, and $\lambda_1$, $\lambda_2$ are the frequency distillation loss weights.

## Key Experimental Results

### Table 1: Cross-Modal Classification (Top-1 Accuracy %)

| Method | Category | CREMA-D A | CREMA-D V | AVE A | AVE V | VGGSound A | VGGSound V | CrisisMMD T | CrisisMMD V |
|--------|----------|-----------|-----------|-------|-------|------------|------------|-------------|-------------|
| w/o KD | Uni | 62.4 | 66.8 | 63.7 | 38.8 | 68.9 | 44.9 | 77.4 | 70.2 |
| Logit | Logit | 61.7 | 62.6 | 60.0 | 39.1 | 65.7 | 45.4 | 78.5 | 70.5 |
| DIST | Logit | 62.2 | 64.0 | 62.4 | 40.3 | 66.4 | 45.5 | 78.3 | 71.3 |
| DKD | Logit | 61.0 | 61.4 | 60.5 | 38.1 | 64.4 | 44.5 | 79.0 | 70.7 |
| Feat | Feature | 60.9 | 64.3 | 58.7 | 39.6 | 67.7 | 45.5 | 77.7 | 70.8 |
| PKD | Feature | 60.4 | 64.8 | 58.0 | 41.0 | 62.9 | 46.9 | 77.5 | 70.9 |
| C2KD | Cross | 57.5 | 59.8 | 62.7 | 39.3 | 67.0 | 47.9 | 77.9 | 71.4 |
| **Ours** | **Cross** | **64.1** | **71.0** | **64.9** | **47.8** | **70.0** | **48.1** | **79.1** | **72.7** |

### Table 2: Ablation Study (CREMA-D / AVE Accuracy %)

| Freq | Align | Scale | LogMSE | CREMA-D A | CREMA-D V | AVE A | AVE V |
|------|-------|-------|--------|-----------|-----------|-------|-------|
| | | | | 60.9 | 64.3 | 58.7 | 39.6 |
| ✓ | | | | 60.8 | 68.7 | 61.0 | 43.3 |
| | ✓ | | | 60.9 | 67.9 | 63.2 | 41.3 |
| ✓ | ✓ | | | 61.8 | 68.7 | 62.4 | 45.8 |
| ✓ | | ✓ | | 62.2 | 70.0 | 62.4 | 44.8 |
| ✓ | ✓ | ✓ | | 62.2 | 70.6 | 62.4 | 46.0 |
| ✓ | ✓ | ✓ | ✓ | **64.1** | **71.0** | **64.9** | **47.8** |

### Table 3: Semantic Segmentation (NYU-Depth V2, mIoU %)

| Method | Depth | RGB |
|--------|-------|-----|
| Uni | 30.9 | 34.1 |
| DIST | 32.3 | 34.9 |
| DKD | 32.5 | 35.3 |
| C2KD | 31.8 | 34.8 |
| **Ours** | **33.2** | **36.9** |
| CIRKDv2 | 33.1 | 36.4 |
| CIRKDv2+Ours | **35.1** | **37.9** |

## Key Findings

1. **Effectiveness of frequency-domain disentanglement**: Cross-modal cosine similarity of low-frequency features (0.85–0.91) substantially exceeds that of the original features (0.74–0.84), while high-frequency similarity is near zero, validating the hypothesis that the frequency domain naturally corresponds to a semantic–detail hierarchy.
2. **Bidirectional distillation stability**: On CrisisMMD, DKD performs well on text but poorly on vision, while AFD shows the opposite pattern; FD-CMKD yields consistent improvements in both directions (T: 79.1%, V: 72.7%).
3. **Effective even from weak modality to strong modality**: On CREMA-D V, AVE A, and VGGSound A, most baselines underperform the no-distillation baseline, whereas the proposed method consistently yields positive gains.
4. **Validation of MSE vs. logMSE matching**: Ablation confirms that MSE for low frequency combined with logMSE for high frequency is the optimal pairing (CREMA-D V: 71.0%); full MSE drops to 70.5% and full logMSE drops to 68.0%.
5. **Frequency threshold of 1/2 is optimal**: Thresholds of 1/4 and 1/3 both underperform 1/2, indicating that a simple equal-split scheme is sufficiently effective.
6. **Composability with task-specific methods**: On NYU-Depth V2, CIRKDv2+Ours achieves 35.1/37.9 mIoU, surpassing either method used individually, demonstrating that frequency-domain disentanglement is an orthogonally complementary module.

## Highlights & Insights

- **Novelty of the frequency-domain perspective**: This is the first work to systematically leverage Fourier transforms to decouple cross-modal features into modality-shared and modality-specific components, with clear theoretical intuition and thorough experimental validation.
- **Elegant loss function design**: The logMSE formulation addresses the gradient explosion problem caused by high-frequency noise, forming a "strong–weak" complementary pairing with MSE.
- **Comprehensive and fair experimental evaluation**: The paper covers three types of scenarios — audio–visual (CREMA-D/AVE/VGGSound), image–text (CrisisMMD), and semantic segmentation (NYU-Depth V2) — using diverse architectures including ResNet, BERT, MobileNet, and DeepLabV3+.
- **Compelling t-SNE visualizations**: Conventional feature KD causes excessive overlap between modality features (losing modality-specific information), whereas the proposed method maintains clear separation between modality features while preserving shared semantic structure.
- **Elegant unification of scale normalization and DC filtering**: Mean subtraction is equivalent to DC component removal in the frequency domain, naturally embedding scale alignment into the frequency-domain disentanglement pipeline.

## Limitations & Future Work

1. **Fixed frequency threshold of 1/2**: The current approach uses a fixed binary mask; different datasets and modality pairs may require different thresholds. The authors acknowledge that adaptive learnable thresholds could be explored in future work.
2. **Validation limited to classification and segmentation tasks**: Generalizability to further downstream tasks such as detection, generation, and retrieval has not been verified.
3. **Limited backbone scale**: The largest networks evaluated are ResNet-18 and BERT-base; performance on large-scale models (e.g., ViT-L, LLaMA) remains unverified.
4. **Shared classifiers not used in the segmentation task**: The authors acknowledge that the shared classifier module was not applied to semantic segmentation due to the pixel-level classification nature of the task, suggesting limited generality of this component.
5. **No exploration of scenarios with more than two modalities**: Experiments are restricted to bimodal pairs; disentanglement strategies for three or more modalities remain to be investigated.
6. **Computational overhead of frequency-domain operations**: Although DFT/IDFT are theoretically lightweight, additional computational cost and training time comparisons are not reported.

## Related Work & Insights

- **C2KD (AAAI 2024)**: The first systematic cross-modal distillation framework, proposing bidirectional distillation and dynamic selection, but operating only at the logit level and neglecting intermediate features. FD-CMKD addresses this gap at the feature level.
- **FreeKD**: Uses frequency information for visual feature distillation in dense prediction tasks, but is restricted to the unimodal setting. This work extends the frequency-domain idea to cross-modal scenarios.
- **Modality Focusing Hypothesis (MFH)**: Theoretical analysis positing that modality-shared discriminative features are central to CMKD, providing a theoretical foundation for this paper's assumption that low frequency corresponds to modality-shared information.
- **Insights**: The frequency-domain disentanglement approach is generalizable to broader cross-modal tasks (e.g., decomposing visual tokens in VLM distillation), and the logMSE loss is applicable to other scenarios requiring "soft alignment."

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of frequency-domain disentanglement and differentiated losses is creative, and the core insight (low frequency = shared / high frequency = specific) is intuitively clear.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five datasets, three task types, detailed ablations and visualizations, covering bidirectional distillation and multiple architectures.
- Writing Quality: ⭐⭐⭐⭐ Logic is clear, with a complete derivation chain from observation → hypothesis → validation → design.
- Value: ⭐⭐⭐⭐ The method is general and composable with other approaches; frequency-domain disentanglement serves as an orthogonally complementary module.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Asymmetric Cross-Modal Knowledge Distillation: Bridging Modalities with Weak Semantic Consistency](asymmetric_cross-modal_knowledge_distillation_bridging_modalities_with_weak_sema.md)
- [\[ACL 2025\] Graph-guided Cross-composition Feature Disentanglement for Compositional Zero-shot Learning](../../ACL2025/model_compression/graph-guided_cross-composition_feature_disentanglement_for_compositional_zero-sh.md)
- [\[CVPR 2026\] Distilling Balanced Knowledge from a Biased Teacher](../../CVPR2026/model_compression/distilling_balanced_knowledge_from_a_biased_teacher.md)
- [\[AAAI 2026\] CTPD: Cross Tokenizer Preference Distillation](ctpd_cross_tokenizer_preference_distillation.md)
- [\[AAAI 2026\] Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers](distillation_dynamics_towards_understanding_feature-based_di.md)

</div>

<!-- RELATED:END -->
