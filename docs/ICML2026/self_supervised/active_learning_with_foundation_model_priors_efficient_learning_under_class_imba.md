---
title: >-
  [Paper Note] Active Learning with Foundation Model Priors: Efficient Learning under Class Imbalance
description: >-
  [ICML 2026][Self-Supervised Learning][Paper Note] This paper proposes PriorAL, which utilizes foundation model predictions as priors for joint decision-making with small models via a "Product of Experts." It employs imbalance-aware entropy filtering to partition the unlabeled pool into a "clean set (for free pseudo-labeling)" and a "noise set (for human annotation),"
tags:
  - ICML 2026
  - Self-Supervised Learning
date: 2026-05-08
content_hash: 1251260aedce0c69
---
# Active Learning with Foundation Model Priors: Efficient Learning under Class Imbalance

**Conference**: ICML2026  
**arXiv**: [2606.07630](https://arxiv.org/abs/2606.07630)  
**Code**: TBD  
**Area**: Active Learning / Label-Efficient Learning  
**Keywords**: Active Learning, Class Imbalance, Label Noise, Foundation Model Priors, Product of Experts

## TL;DR
This paper proposes PriorAL, which utilizes foundation model predictions as priors for joint decision-making with small models via a "Product of Experts." It employs imbalance-aware entropy filtering to partition the unlabeled pool into a "clean set (for free pseudo-labeling)" and a "noise set (for human annotation)," achieving over 50% savings in labeling costs on image/text tasks characterized by both class imbalance and label noise.

## Background & Motivation
**Background**: The core idea of active learning (AL) is to select only the most informative samples for manual annotation rather than labeling the entire dataset, thereby approaching full-supervision performance with minimal costs. Recent work has focused on class imbalance, encouraging balanced sampling to improve minority class coverage.

**Limitations of Prior Work**: Real-world datasets often present two concurrent difficulties: **extreme class imbalance** (massive majority classes vs. sparse minority classes) and **label noise** (incorrect oracle labels due to annotator fatigue or perceptual inconsistency). Under imbalance, naive budget allocation disproportionately favors majority classes, leaving minority classes under-annotated, while label noise further degrades minority class training. Existing imbalanced AL methods typically address imbalance but ignore noise.

**Key Challenge**: Foundation models (e.g., GPT, LLaMA, CLIP, SigLIP) carry rich priors that provide valuable labeling signals in low-resource or imbalanced scenarios. However, "Foundation Model AL" has not been systematically studied under the **simultaneous presence of imbalance and label noise**. Blindly trusting FM pseudo-labels introduces noise, while relying solely on small models is unreliable due to data scarcity.

**Goal**: Design a foundation model-informed AL algorithm that is robust to both class imbalance and label noise across image and text domains.

**Key Insight**: Rather than treating the foundation model as a Bayesian prior, its predictions are viewed as "extracted encoded knowledge" serving as prior information to guide sampling. Leveraging the classic **Product of Experts (PoE)** framework, the predictions of the foundation model and the small model are multiplied to jointly determine which samples to trust and which to query.

**Core Idea**: Utilize a "Foundation Model $\times$ Small Model" PoE for joint decision-making, paired with a category-frequency-weighted entropy to distinguish between a "clean set" for free pseudo-labels and a "noise set" for budgeted human annotation. This ensures the annotation budget is precisely targeted at samples that truly require it.

## Method

### Overall Architecture
PriorAL (Foundation Model Priors-Informed Active Learning) is an **iterative** pool-based AL algorithm. Given an imbalanced unlabeled pool $\mathcal{D}_U$, a pre-trained foundation model $M_L$, a small model $f$ to be trained, and a per-round budget $B$, each round $t$ consists of three phases: **prior labeling → imbalance-aware uncertainty sampling → small model training**. The key is that the foundation model and small model **jointly** influence sample selection.

```mermaid
graph TD
    A["Unlabeled Pool D_U"] --> B["Prior Labeling<br/>FM p_L + Pseudo-labels<br/>Small Model p_f"]
    B --> C["PoE Joint Decision<br/>p̄ ∝ p_f · p_L"]
    C --> D["Imbalance-aware Entropy Filtering<br/>Weighted by class frequency H_b"]
    D -->|Low H_b (High Confidence)| E["Clean Set D_C<br/>Free Pseudo-labels"]
    D -->|High H_b (Low Confidence)| F["Noise Set D_N"]
    F --> G["Small Model Uncertainty Sampling<br/>Top-B query oracle"]
    E --> H["Small Model Training<br/>D_C ∪ D_L"]
    G --> H
    H -->|f_t to next round| B
```

### Key Designs

**1. Product of Experts (PoE) Joint Decision: Multiplying instead of choosing**

The limitation of relying on either model alone is that foundation models have priors but lack task specificity, while small models fit the task but lack sufficient data. Utilizing Hinton’s PoE concept, the class probabilities are **multiplied element-wise and normalized** to obtain a joint distribution:

$$\bar{p}(x, y=i) = \frac{p_f(x,y=i)\cdot p_L(x,y=i)}{\sum_j p_f(x,y=j)\cdot p_L(x,y=j)}$$

where $p_L$ is the foundation model probability and $p_f$ is the small model probability from the previous round $f_{t-1}$. This multiplication represents "AND" logic: the joint distribution is sharp only if both experts agree on a class. If either expert is uncertain, the distribution flattens, increasing entropy. This $\bar{p}$ encodes both FM pre-trained priors and small model task fitting.

**2. Imbalance-aware Weighted Entropy Filtering: Retaining minority samples in the clean set**

Using standard joint entropy $H(x)=-\sum_i \bar{p}\log\bar{p}$ ignores class imbalance. The entropy is weighted by the frequency of the pseudo-label class: first, $w_i = |\{x\in\mathcal{D}^t_U : \hat{y}(x)=i\}|$ counts samples predicted as class $i$ by the FM. Then, the **imbalance-aware augmented entropy** is defined:

$$H_b(x) = -\sum_i \Big(\bar{p}(x,y=i)\log\bar{p}(x,y=i)\cdot \frac{w_{\hat{y}(x)}}{w_{\max}}\Big)$$

where $\hat{y}(x)=\arg\max_i p_L(x,y=i)$ and $w_{\max}=\max_i w_i$. The weight $w_{\hat y}/w_{\max}$ is $\approx 1$ for majority classes and much smaller for minority classes, thereby **lowering the $H_b$ for minority samples**. The top-$\rho|\mathcal{D}^t_U|$ samples with the highest $H_b$ form the noise set $\mathcal{D}_N$, while the rest enter the clean set $\mathcal{D}_C$. Consequently, high-entropy majority samples are pushed to the noise set for labeling, while minority samples more easily enter the clean set to be retained with zero-cost FM pseudo-labels.

**3. Small Model Uncertainty Sampling on the Noise Set: Precision budgeting**

While the clean set uses pseudo-labels, the algorithm must select which samples in $\mathcal{D}_N$ to label. Uncertainty scores are calculated using only the small model's probability from the previous round:

$$U_f(x) = -\sum_i p_f(x,y=i)\log p_f(x,y=i)$$

The top-$B$ samples with the highest $U_f$ (the ones the small model is least confident about) are sent to the oracle for true labels $\mathcal{D}_t$. Finally, the small model is trained on $\mathcal{D}_C \cup \mathcal{D}^{t+1}_L$. This structure implicitly mitigates oracle noise by assigning "hard" samples to humans and "easy" samples to pseudo-labels.

### Loss & Training
The small model $f$ is fine-tuned using standard cross-entropy on the training pool $\mathcal{D}_C \cup \mathcal{D}^{t+1}_L$. The algorithm runs for $T$ rounds with per-round budget $B$ and $M$ initial samples for cold start. The key hyperparameter is the noise set ratio $\rho$.

## Key Experimental Results

### Main Results
Experiments across **21 dataset configurations** (text: Trec, AGNews, SST-2; image: CIFAR-10, CIFAR-100, PathMNIST) covered "merge imbalance" and "long-tailed imbalance." PriorAL achieved superior performance with **50%+** labeling cost savings compared to strong AL baselines.

| Dataset (Setting) | Random | BADGE | CORESET | Ours (PriorAL) |
|-------------------|--------|-------|---------|----------------|
| Trec (merge, text) | 82.4 / 88.8 | 90.6 / 97.2 | 93.1 / 97.9 | **Best** |
| CIFAR-10 (merge, img) | 71.5 / 77.6 | 73.6 / 86.2 | 77.2 / 85.4 | **Best** |
| PathMNIST (LT, img) | 52.0 / 61.5 | 65.6 / 82.7 | 68.7 / 78.4 | **Best** |

*Note: Values represent balanced pool accuracy at 10% / 20% budget. See original paper Table 1 for exact metrics.*

### Ablation Study

| Configuration | Function | Impact of Removal |
|---------------|----------|-------------------|
| Full (PriorAL) | PoE + Weighted Entropy + Uncertainty | Complete performance; 50%+ savings |
| w/o PoE | Single model probability only | Loss of FM prior or task adaptation; poor filtering |
| w/o Imbalance Weight | Standard entropy $H(x)$ | Minority samples misclassified into noise set; poor balance |
| w/o Uncertainty | Random sampling in $\mathcal{D}_N$ | Budget wasted on known samples; decreased efficiency |

### Key Findings
- **Contribution of Disparity**: The "imbalance-aware weighting + clean/noise split" allows minority samples to be retained at zero cost while targeting the budget at truly difficult majority samples.
- **Cross-Domain Generality**: The unified pipeline works across both image and text domains, being the first to study FM-based AL under dual challenges of imbalance and noise.
- **Implicit Noise Resistance**: Assigning "hard" samples to humans and "reliable" ones to pseudo-labels naturally filters out erroneous oracle labels.

## Highlights & Insights
- **PoE as a Joint Judge**: The "AND" logic of multiplication ensures that if either expert is hesitant, entropy rises, making it more robust than single-model uncertainty.
- **Integrating Balance into Sampling**: Using class frequency to reweight entropy encodes the "balance" goal directly into the sampling criterion rather than post-hoc.
- **Reusable Paradigm**: The "free clean set + paid noise set" dichotomy is a highly reusable framework for compressing labeling costs.

## Limitations & Future Work
- **FM Prior Sensitivity**: If the foundation model performs poorly on the target domain, PoE and weighting may propagate incorrect pseudo-labels.
- **Weighting Intuition**: The efficacy of $w_{\hat y}/w_{\max}$ depends on the FM's pseudo-label distribution approximating the true distribution.
- **Hyperparameter $\rho$**: The noise set ratio requires further analysis regarding its robustness across different datasets.

## Related Work & Insights
- **vs GALAXY / DIRECT**: These focus on class balance via sampling but do not handle label noise or integrate FM priors as PriorAL does.
- **vs BADGE / CORESET**: These utilize single-model gradients/features; PriorAL is more efficient in imbalanced scenes due to dual-model joint decision-making.

## Rating
- Novelty: ⭐⭐⭐⭐ (First to study FM AL under dual imbalance/noise challenges).
- Experimental Thoroughness: ⭐⭐⭐⭐ (21 settings across domains).
- Writing Quality: ⭐⭐⭐ (Clear logic, though tables are dense).
- Value: ⭐⭐⭐⭐ (Significant 50%+ cost savings for real-world scenarios).

## Related Papers

- [\[CVPR 2026\] Parameter-efficient Continual Learning for Enhancing Plasticity without Forgetting under Limited Model Capacity](../../CVPR2026/self_supervised/parameter-efficient_continual_learning_for_enhancing_plasticity_without_forgetti.md)
- [\[CVPR 2026\] Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning](../../CVPR2026/self_supervised/temporal_imbalance_of_positive_and_negative_supervision_in_class-incremental_lea.md)
- [\[ICML 2026\] How 'Neural' is a Neural Foundation Model?](how_neural_is_a_neural_foundation_model.md)
- [\[ICML 2026\] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)
- [\[ICLR 2026\] Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](../../ICLR2026/self_supervised/test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)
- [\[ICML 2026\] PartCo: Part-Level Correspondence Priors Enhance Category Discovery](partco_part-level_correspondence_priors_enhance_category_discovery.md)
- [\[ICML 2026\] FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction](flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f.md)
- [\[ICML 2026\] A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning](a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive.md)
- [\[ICML 2026\] NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation Models](numleak_public_numeric_benchmarks_as_latent_labels_in_foundation_models.md)

</div>

<!-- RELATED:END -->
