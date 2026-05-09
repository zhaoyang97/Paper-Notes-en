---
title: >-
  [Paper Note] Normal-Abnormal Guided Generalist Anomaly Detection
description: >-
  [NeurIPS 2025][LLM Evaluation][generalist anomaly detection] NAGL is the first framework to incorporate mixed normal-and-abnormal reference samples into Generalist Anomaly Detection (GAD). Through two attention modules—Residual Mining (RM) and Anomaly Feature Learning (AFL)—it learns transferable anomaly patterns in residual space, substantially outperforming normal-reference-only methods in cross-domain scenarios with as few as 1 anomaly reference sample.
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - generalist anomaly detection
  - cross-domain transfer
  - residual learning
  - normal-abnormal reference
  - meta-learning
date: 2026-05-08
content_hash: df3bff52d3532981
---

# Normal-Abnormal Guided Generalist Anomaly Detection

**Conference**: NeurIPS 2025
**arXiv**: [2510.00495](https://arxiv.org/abs/2510.00495)
**Code**: [GitHub](https://github.com/JasonKyng/NAGL)
**Area**: LLM Evaluation
**Keywords**: generalist anomaly detection, cross-domain transfer, residual learning, normal-abnormal reference, meta-learning

## TL;DR

NAGL is the first framework to incorporate mixed normal-and-abnormal reference samples into Generalist Anomaly Detection (GAD). Through two attention modules—Residual Mining (RM) and Anomaly Feature Learning (AFL)—it learns transferable anomaly patterns in residual space, substantially outperforming normal-reference-only methods in cross-domain scenarios with as few as 1 anomaly reference sample.

## Background & Motivation

Visual anomaly detection is critical in industrial inspection and medical diagnosis. GAD aims to train a unified model on a source domain and transfer it directly to unseen target domains, addressing data scarcity and privacy constraints in the target domain.

Existing GAD methods (e.g., InCTRL, ResAD) share a key limitation:

**Reliance solely on normal reference samples**: the model lacks direct knowledge of anomalous features, limiting its discriminative capacity.

**Neglect of available anomaly samples**: in practice, a small number of anomalous samples (e.g., defective parts, diagnosed cases) are often accessible and contain valuable anomaly information that goes unused.

Intuitively, incorporating anomaly references should be beneficial, but naive approaches fail:
- **KNN-based methods** (far from normal + close to abnormal = anomaly): training-agnostic and lacking adaptability; experiments reveal a severe **false activation problem** where normal regions are incorrectly flagged as anomalous.
- Existing GAD methods rely on query-to-normal residuals to ensure transferability and cannot be straightforwardly extended to mixed normal-abnormal reference settings.

The core challenge is: how to effectively leverage anomaly reference information without introducing noise? The key insight of NAGL is that operating in **residual space** simultaneously preserves cross-domain transferability and anomaly discriminability.

## Method

### Overall Architecture

NAGL inference pipeline: given a query image and a set of normal and abnormal references →
1. A pretrained ViT backbone extracts features.
2. The query is matched against normal references via nearest-neighbor search → normal-guided score map $\mathcal{S}_n$.
3. The RM module mines anomaly patterns from normal-abnormal reference residuals → residual proxies.
4. The AFL module maps residual proxies onto the query image → anomaly proxies → anomaly-guided score map $\mathcal{S}_a$.
5. The two score maps are fused: $\mathcal{S} = \mathcal{S}_n + \mathcal{S}_a$ to yield the final anomaly localization.

### Key Designs

1. **Normal-Guided Anomaly Score**: PatchCore-style nearest-neighbor search is adopted. For each patch $f_i^q$ in the query feature $\mathcal{F}^q$, the nearest neighbor $f_*^n$ is retrieved from normal reference features $\mathcal{F}^n$, and the anomaly score is defined as $\mathcal{S}_n^i = \mathbf{d}(f_i^q, f_*^n)$ (cosine distance). This provides baseline anomaly localization, though discriminability is limited when only normal references are used.

2. **Residual Mining (RM) Module**: The objective is to extract transferable anomaly pattern representations from abnormal references. The residual between each abnormal reference and its nearest normal reference is first computed:
    $$\text{Res}(\mathcal{F}^a, \mathcal{F}^n) = \mathcal{F}^a - \mathcal{F}_*^n$$
   A cross-attention layer is then applied, with learnable proxies $\mathcal{P} \in \mathbb{R}^{M \times C}$ as Query, abnormal features $\mathcal{F}^a$ as Key, and the residual as Value:
    $$\widetilde{\mathcal{P}} = \mathbf{SA}_1\left(\text{Softmax}\left(\frac{\mathbf{Q}_1 \mathbf{K}_1^T}{\sqrt{d}} + \mathcal{M}'\right) \mathbf{V}_1\right)$$
   The attention mask $\mathcal{M}' = \alpha(1 - \mathcal{M}^a)$ (where $\alpha$ is a large negative value) ensures that attention focuses exclusively on residuals at anomalous regions. The resulting residual proxy $\widetilde{\mathcal{P}}$ captures variation patterns of abnormal samples in residual space. Operating in residual space ensures cross-domain transferability, as residual features exhibit similar distributions across domains.

3. **Anomaly Feature Learning (AFL) Module**: The residual proxies are applied to the query image to identify potential anomalies. The residual proxy $\widetilde{\mathcal{P}}$ serves as Query, the query-to-normal residual $\text{Res}(\mathcal{F}^q, \mathcal{F}^n)$ as Key, and the query feature $\mathcal{F}^q$ as Value:
    $$\widehat{\mathcal{P}} = \mathbf{SA}_2\left(\text{Softmax}\left(\frac{\mathbf{Q}_2 \mathbf{K}_2^T}{\sqrt{d}}\right) \mathbf{V}_2\right)$$
   The core mechanism is to identify anomalous regions in the query by measuring the similarity between the **reference anomaly-to-normal residual** and the **query-to-normal residual**. If a query region's residual pattern resembles a known anomaly residual, the corresponding visual features are likely anomalous. The resulting anomaly proxy $\widehat{\mathcal{P}} \in \mathbb{R}^{M \times C}$ encodes the most discriminative features in the query image. The final anomaly-guided score is:
    $$\mathcal{S}_a^i = \frac{1}{M} \sum_{m=1}^M 1 - \mathbf{d}(f_i^q, \widehat{\mathcal{P}}_m)$$

### Loss & Training

A meta-learning strategy is employed for training on the source domain. Each episode contains a normal-abnormal reference set and a query image. The loss combines segmentation and classification objectives:
$$\mathcal{L} = \mathcal{L}_{cls} + \lambda \mathcal{L}_{seg}$$
where:
- $\mathcal{L}_{seg} = \text{Focal}(\mathcal{S}, \mathcal{M}^q) + \text{Dice}(\mathcal{S}, \mathcal{M}^q)$: pixel-level anomaly localization.
- $\mathcal{L}_{cls} = \text{BCE}(s, y^q)$: image-level anomaly classification, with $s = \mathcal{T}_{0.01}(\mathcal{S})$ (mean of top 1% of score map).
- $\lambda = 1.0$.

Implementation details: ViT-S backbone (21M parameters, frozen); only attention modules are trained (24.4M total parameters). AdamW optimizer with initial learning rate $10^{-5}$, reduced by $10\times$ at epochs 10 and 15. Training converges in 20 epochs with 500 episodes per epoch. Input resolution $448 \times 448$; learnable proxy count $M=25$; normal references $K_1 \in [1,2,4]$; anomaly references $K_2 = 1$.

## Key Experimental Results

### Main Results

Cross-domain evaluation: VisA → MVTecAD and MVTecAD → VisA/BraTS.

| Setting | Method | MVTecAD Image AUROC | MVTecAD Pixel AUROC | VisA Image AUROC | VisA Pixel AUROC |
|---------|--------|-------------------|-------------------|-----------------|-----------------|
| $N^1$ | PatchCore | 83.4 | 92.0 | 79.9 | 95.4 |
| $N^1$ | WinCLIP | 93.1 | 95.2 | 83.8 | 96.4 |
| $N^1$ | ResAD | 84.8 | 93.4 | 80.9 | 95.9 |
| $N^1+A^1$ | **Ours** | **95.8** | **96.6** | **88.5** | **97.5** |
| $N^4$ | WinCLIP | 95.2 | 96.2 | 87.3 | 97.2 |
| $N^4+A^1$ | **Ours** | **97.1** | **97.0** | **91.2** | **97.8** |

Cross-domain medical evaluation (MVTecAD → BraTS):

| Setting | Method | Image AUROC | Pixel AUROC | Mean |
|---------|--------|------------|------------|------|
| $N^2$ | ResAD | 66.2 | 91.5 | 78.9 |
| $N^2+A^1$ | **Ours** | **82.1** | **96.8** | **89.5** |
| $N^4$ | PatchCore | 71.2 | 95.9 | 83.6 |
| $N^4+A^1$ | **Ours** | **84.9** | **97.1** | **91.0** |

### Ablation Study

| Configuration | MVTecAD Image | MVTecAD Pixel | VisA Image | VisA Pixel | Mean |
|--------------|-------------|-------------|-----------|-----------|------|
| i: Normal NN only | 93.2 | 94.5 | 81.5 | 95.3 | 91.1 |
| ii: Abnormal NN only | 70.1 | 83.3 | 58.8 | 84.5 | 74.2 |
| iii: Normal+Abnormal NN (w/o NAGL) | 90.7 | 92.1 | 77.2 | 93.5 | 88.4 |
| iv: Normal+Abnormal+NAGL (full) | **95.8** | **96.6** | **88.5** | **97.5** | **94.6** |

Efficiency comparison:

| Method | Parameters (M) | Training Time (H) | Inference Speed (FPS) |
|--------|--------------|-----------------|---------------------|
| InCTRL | 117.5 | 0.7 | 1.2 |
| ResAD | 59.2 | 20.6 | 7.8 |
| **Ours** | **24.4** | **0.3** | **17.1** |

### Key Findings

- **A single anomaly reference yields substantial gains**: the $N^1+A^1$ setting outperforms the best baseline using $N^4$ normal references (MVTecAD: +0.6 Image AUROC; VisA: +1.2 Image AUROC), achieving superior performance with fewer references.
- **False activation problem confirmed**: naively combining normal and abnormal references (setting iii) performs worse than using normal references alone (setting i) (91.1 → 88.4), corroborating that naive approaches introduce misleading noise.
- **NAGL effectively resolves false activations**: setting iv improves over setting iii by 6.2 mean AUROC, demonstrating that residual-space operation effectively filters noise.
- **Strong cross-domain generalization**: in the industrial-to-medical cross-domain scenario (BraTS), NAGL with $N^2+A^1$ surpasses ResAD with $N^2$ by 10.6 percentage points.
- **Outstanding efficiency**: NAGL uses only 1/5 the parameters of InCTRL, trains 69× faster than ResAD, and achieves 14× the inference speed of InCTRL.

## Highlights & Insights

- **Task-level novelty**: NAGL is the first to incorporate mixed normal-abnormal references into GAD, more closely reflecting real-world deployment conditions.
- **Residual space is the key**: directly fusing anomaly references in visual space causes false activations, whereas operating in residual space is inherently domain-transferable, as residual feature distributions are more stable across domains.
- **Minimal yet effective design**: a frozen backbone with only 2 trained attention modules (24.4M parameters) achieves state-of-the-art performance.
- **Elegant two-stage RM→AFL design**: anomaly pattern prototypes are first extracted in residual space via RM, then mapped back to visual space by AFL to localize specific anomalies.

## Limitations & Future Work

- The anomaly reference count is fixed at $K_2=1$; whether additional anomaly references yield further gains remains unexplored.
- The method assumes one reference sample per anomaly type; performance when anomaly types are not covered is unknown.
- The choice of learnable proxy count $M=25$ lacks an adaptive mechanism.
- Future work could explore natural language descriptions as a complement to visual normal/abnormal references.
- Extension to 3D anomaly detection and video anomaly detection is a promising direction.

## Related Work & Insights

- The comparison with PatchCore demonstrates that strong representation learning combined with anomaly guidance outperforms relying on large numbers of normal references.
- The cross-domain consistency of residual features warrants further investigation in other transfer learning tasks.
- The RM-AFL architecture is generalizable to other few-shot detection tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to leverage anomaly references in GAD; innovations at both the task and method levels.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Cross-domain evaluation on industrial and medical benchmarks; comprehensive ablations and efficiency comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, well-coordinated figures, and thorough explanation of the RM-AFL design.
- **Value**: ⭐⭐⭐⭐ Strong practical orientation, notable efficiency advantages, suitable for real-world deployment.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection](../../AAAI2026/llm_evaluation/refinevad_semantic-guided_feature_recalibration_for_weakly_supervised_video_anom.md)
- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](../../ICLR2026/llm_evaluation/towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[CVPR 2026\] Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](../../CVPR2026/llm_evaluation/weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)
- [\[NeurIPS 2025\] Rethinking Evaluation of Infrared Small Target Detection](rethinking_evaluation_of_infrared_small_target_detection.md)
- [\[NeurIPS 2025\] Robust Hallucination Detection in LLMs via Adaptive Token Selection](robust_hallucination_detection_in_llms_via_adaptive_token_selection.md)

<!-- RELATED:END -->
