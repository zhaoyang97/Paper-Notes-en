---
title: >-
  [Paper Note] I Am Big, You Are Little; I Am Right, You Are Wrong
description: >-
  [ICCV 2025][Minimal Pixel Sets] This work employs the causal-reasoning XAI tool rex to extract Minimal Pixel Sets (MPS) from image classification models…
tags:
  - "ICCV 2025"
  - "Minimal Pixel Sets"
  - "Image Classification"
  - "Model Comparison"
  - "Explainable AI"
  - "Causal Reasoning"
date: 2026-05-08
content_hash: 6d4e97e07628f4fb
---

# I Am Big, You Are Little; I Am Right, You Are Wrong

**Conference**: ICCV 2025
**arXiv**: [2507.23509](https://arxiv.org/abs/2507.23509)
**Code**: [ReX-XAI/ReX](https://github.com/ReX-XAI/ReX)
**Area**: Others (Explainable AI / Model Analysis)
**Keywords**: Minimal Pixel Sets, Image Classification, Model Comparison, Explainable AI, Causal Reasoning

## TL;DR

This work employs the causal-reasoning XAI tool rex to extract Minimal Pixel Sets (MPS) from image classification models, systematically comparing the "attentional focus" of 15 models across 5 architectures. Large models (EVA/ConvNext) are found to make classification decisions using as little as 5% of image pixels, and statistically significant differences in MPS size and spatial location are observed across architectures.

## Background & Motivation

**Background**: Despite the proliferation of visual classification model families (CNNs, ViTs, hybrid architectures) and scales, understanding of *how* these models reach decisions remains limited. Prior comparisons have focused primarily on accuracy and robustness, leaving a systematic study of *which pixels* models rely upon largely unaddressed.

**Limitations of Prior Work**: Jiang et al. used the SAG tool to extract patch-level "minimal sufficient explanations," but two critical issues exist:
1. SAG's notion of sufficiency is too permissive—it uses a confidence threshold rather than enforcing reproduction of the original classification.
2. SAG operates on fixed-size patches, precluding pixel-level precision.

**Goal**: This paper proposes extracting **Minimal Pixel Sets (MPS)** via the rex tool, offering two key advantages:
- Pixel-level minimality, unconstrained by patch size.
- Strict sufficiency: an MPS must cause the model to reproduce the original top-1 classification.

## Method

### Overall Architecture

The research pipeline proceeds as follows:
1. Select 15 pretrained ImageNet-1k models spanning 5 architectures.
2. Extract MPS for 1,000 images (500 validation + 500 test) using rex.
3. Compare MPS size and spatial distribution via statistical tests (Kruskal-Wallis H test, Friedman test).
4. Analyze the relationship between correct/incorrect classifications and MPS size.

### Key Designs

1. **Minimal Pixel Sets (MPS) Extraction**: rex adopts a causal inference framework, treating the model as a black-box causal model. The core algorithm proceeds as follows:

    - Partition the image into 4 superpixel regions.
    - Generate mutants by combinatorially masking regions (baseline = 0) and testing model outputs.
    - Rank pixels by causal responsibility (higher responsibility = greater contribution to classification).
    - Iteratively refine superpixel partitions (default: 20 iterations) to produce a responsibility landscape.
    - Incrementally add pixels in order of responsibility rank until the original top-1 classification is reproduced.
    - The resulting pixel set constitutes the MPS (approximately minimal but guaranteed sufficient).

2. **Model Architecture Selection**: Models span the full spectrum from small CNNs to large-scale Transformers:

    - **Inception** (CNN): V3, V4, ResNet-V2 (wide-network architectures)
    - **ResNet** (CNN): 152-B A1, A2, D (residual networks)
    - **ConvNext** (Modern CNN): V2 Large, V2 Huge v1/v2 (modernized ResNets)
    - **ViT** (Transformer): Large, Huge V1/V2 (standard vision Transformers)
    - **EVA** (Transformer): 02 Large V1/V2, Giant (large-scale pretrained ViTs, up to 1B parameters)

3. **Statistical Analysis Methods**:

    - **Kruskal-Wallis H test**: Cross-architecture differences in MPS size (non-parametric).
    - **Friedman test**: Within-architecture differences across model variants (paired data).
    - **Sørensen-Dice coefficient**: Overlap between MPS from different models.
    - **Hausdorff distance**: Spatial displacement between MPS.
    - **Bonferroni correction**: Control of Type I error under multiple comparisons.
    - **Linear mixed-effects model**: Controls for model accuracy as a confound when analyzing the effect of correct/incorrect classification on MPS size.

### Loss & Training

No training is performed in this work. All models use pretrained IN-1k weights from the Timm library (2024.02), converted to ONNX format for inference via ONNX Runtime. rex applies identical hyperparameters and random seeds across all models.

## Key Experimental Results

### Main Results

**MPS Size (as a proportion of image area)**

| Model | Overall Mean | Correct | Incorrect | Model Accuracy |
|-------|-------------|---------|-----------|----------------|
| ConvNext-V2 Huge v1 | **0.052** | 0.048 | 0.061 | 0.890 |
| EVA Giant | **0.056** | 0.052 | 0.065 | 0.894 |
| ConvNext-V2 Huge v2 | 0.068 | 0.063 | 0.082 | 0.894 |
| ConvNext-V2 Large | 0.089 | 0.075 | 0.122 | 0.880 |
| ViT Large | 0.099 | 0.098 | 0.111 | 0.900 |
| ViT Huge V2 | 0.103 | 0.102 | 0.113 | 0.872 |
| ResNet152-D | 0.137 | 0.130 | 0.155 | 0.828 |
| ViT Huge V1 | 0.158 | 0.154 | 0.170 | 0.882 |
| Inception V4 | 0.239 | 0.224 | 0.261 | 0.840 |
| Inception V3 | 0.247 | 0.231 | 0.271 | 0.800 |
| Inception-ResNet V2 | **0.254** | 0.246 | 0.265 | 0.814 |

Inception models produce MPS that are **3.6×** larger than those of ConvNext, a statistically significant difference.

### Ablation Study

**Cross-Architecture MPS Overlap (Dice coefficient, between best-performing models)**

| | EVA Giant | ConvNext | ViT Large | ResNet152 | Inception |
|--|-----------|----------|-----------|-----------|-----------|
| EVA Giant | 1.0 | 0.287 | 0.253 | 0.165 | 0.141 |
| ConvNext | 0.287 | 1.0 | 0.304 | 0.162 | 0.163 |
| ViT Large | 0.253 | 0.304 | 1.0 | 0.232 | 0.225 |
| ResNet152 | 0.165 | 0.162 | 0.232 | 1.0 | 0.282 |
| Inception | 0.141 | 0.163 | 0.225 | 0.282 | 1.0 |

MPS overlap across architectures is generally low (mostly < 0.3), indicating that different architectures attend to substantially different image regions.

### Key Findings

1. **Significant cross-architecture differences**: Kruskal-Wallis test yields $H(4)=1176.134, p<0.001$, rejecting the null hypothesis of equal MPS sizes across architectures.
2. **Within-architecture differences also present**: Friedman tests are significant ($p<0.01$) for ConvNext, EVA, ResNet, and ViT; only Inception models show no significant within-family variation ($p=0.36$).
3. **Incorrect classifications yield larger MPS**: The linear mixed-effects model shows that incorrect predictions increase mean MPS area by 2.6% ($p<0.01$).
4. **EVA Giant classifies using only 5.4% of pixels**, raising concerns about potential overfitting or "tunnel vision" in large models.
5. **Different models attend to different regions**: On the same image, the MPS of a ResNet may share zero overlap with that of other models (Dice = 0).

## Highlights & Insights

1. **Causal-reasoning perspective on model analysis**: MPS are grounded in a strict causal sufficiency definition, making them more operationally rigorous than attribution methods such as GradCAM or SHAP.
2. **"Tunnel vision" in large models**: EVA-Giant, with 1 billion parameters, relies on just 5% of pixels for classification, raising safety concerns in high-stakes domains such as medical imaging and autonomous driving.
3. **MPS as a model selection criterion**: Beyond accuracy and robustness, MPS characteristics offer a novel dimension for model selection.
4. **Larger MPS signals potential misclassification**: This can serve as a post-hoc diagnostic—an unusually large MPS for a given sample may flag a potentially erroneous prediction.

## Limitations & Future Work

- rex employs a zero baseline for masking, generating substantial amounts of out-of-distribution (OOD) images; alternative baselines (e.g., blurring) may alter conclusions.
- Computing a true MPS is NP-hard (DP-complete); rex yields only an approximation.
- Only top-1 MPS are analyzed; the multiplicity of explanations (a single image may admit several disjoint MPS) is not explored.
- The relationship between MPS size and model robustness is not quantitatively analyzed.
- The CalTech-256 validation subset is small (50 images), limiting statistical power.
- ImageNet-1k labels contain noise, which may confound the correct/incorrect classification analysis.

## Related Work & Insights

- **rex**: A black-box XAI tool grounded in actual causality (Halpern–Pearl framework).
- **SAG (Jiang et al.)**: Extracts patch-level combinatorial explanations but applies a weaker sufficiency criterion.
- **GradCAM / SHAP / LIME**: White-box/black-box attribution methods that rank pixel importance but do not guarantee sufficiency.
- **Insight**: Auditing "how much of the image a model actually uses" via MPS analysis prior to deployment constitutes a concise and effective model inspection strategy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First large-scale systematic comparison of 15 visual models using causal MPS.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five architectures, 15 models, 1,000 images, multiple statistical tests, CalTech cross-validation.
- **Writing Quality**: ⭐⭐⭐⭐ Research questions are clearly stated, statistical methodology is rigorous, and case analyses are illustrative.
- **Value**: ⭐⭐⭐ Opens a novel analytical perspective, but offers limited practical guidance (no improvement proposals are made).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ResNets Are Deeper Than You Think](../../NeurIPS2025/others/resnets_are_deeper_than_you_think.md)
- [\[ICCV 2025\] You Share Beliefs, I Adapt: Progressive Heterogeneous Collaborative Perception](you_share_beliefs_i_adapt_progressive_heterogeneous_collaborative_perception.md)
- [\[NeurIPS 2025\] Improving Forecasts of Suicide Attempts for Patients with Little Data](../../NeurIPS2025/others/improving_forecasts_of_suicide_attempts_for_patients_with_little_data.md)
- [\[ICLR 2026\] Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention](../../ICLR2026/others/key_and_value_weights_are_probably_all_you_need_on_the_necessity_of_the_query_ke.md)
- [\[ICCV 2025\] DeSPITE: Exploring Contrastive Deep Skeleton-Pointcloud-IMU-Text Embeddings for Advanced Point Cloud Human Activity Understanding](despite_exploring_contrastive_deep_skeletonpointcloudimutext.md)

</div>

<!-- RELATED:END -->
