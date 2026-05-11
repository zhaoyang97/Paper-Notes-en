---
title: >-
  [Paper Note] RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification
description: >-
  [Medical Imaging] This paper proposes RAxSS, a framework that integrates retrieval augmentation into the random sparse sampling (SSS) pipeline. By replacing uniform averaging with intra-window similarity-weighted aggrega…
tags:
  - "Medical Imaging"
date: 2026-05-08
content_hash: 2be589f5260d7b36
---

# RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification

## Metadata
- **Conference**: NeurIPS 2025
- **arXiv**: [2510.02936](https://arxiv.org/abs/2510.02936)
- **Code**: Unavailable
- **Area**: Medical Imaging
- **Keywords**: Time series classification, retrieval augmentation, sparse sampling, explainability, epileptic EEG

## TL;DR
This paper proposes RAxSS, a framework that integrates retrieval augmentation into the random sparse sampling (SSS) pipeline. By replacing uniform averaging with intra-window similarity-weighted aggregation, RAxSS maintains competitive performance on variable-length medical time series classification while providing an interpretable evidence chain spanning from "where" to "why."

## Background & Motivation

AI analysis of medical time series (heart rate, blood glucose, EEG, etc.) faces two persistent obstacles: the heterogeneity of clinical data and the demand for transparent explainability. Specific challenges include:

**Variable-length sequences**: Recording lengths vary substantially across patients and events, yet mainstream time series classification (TSC) methods are designed for fixed-length inputs.

**Limitations of uniform aggregation**: SSS handles variable-length inputs by randomly sampling fixed-length windows from long recordings and aggregating predictions, but uniform aggregation assumes all segments are equally informative—an unrealistic assumption for non-stationary, irregular real-world data.

**Insufficient explainability**: The explainability of SSS relies solely on visualization of local scores (indicating "where" signals occur) and cannot answer "why" a given region is trusted.

**Mechanism**: Drawing inspiration from retrieval-augmented methods for forecasting (RAFT)—which leverage similarity-based retrieval to improve performance on rare patterns and weak temporal correlations—RAxSS adapts this idea to classification, enabling selective weighting and interpretable evidence attribution.

## Method

### Overall Architecture

RAxSS unifies sampling, retrieval, and aggregation within the SSS framework. Long sequences are segmented into fixed-length windows, sampled proportionally to length (sequence $i$ is drawn with probability $p_i \propto T_i / \sum_j T_j$), and scored by a backbone network $f_\theta$. Intra-recording retrieval computes inter-window similarity, which replaces uniform averaging with similarity-weighted aggregation.

### Key Designs

1. **Retrieval-aware aggregation**: For each window $k$ in sequence $i$, the $m$ most similar yet distinct windows within the same sequence are retrieved (temporal overlap due to sliding-window extraction is permitted), using Pearson or cosine similarity:

$$\bar{s}_k = \frac{1}{m} \sum_{j \in N_k} s_k^{(j)}, \quad s_k^{(j)} = \phi(w_k, w_j)$$

2. **Temperature-scaled softmax weighting**: Support scores are converted to normalized weights via a softmax with temperature parameter $\tau$:

$$\alpha_k = \frac{\exp(\bar{s}_k / \tau)}{\sum_{t \in K_i} \exp(\bar{s}_t / \tau)} \in [0,1], \quad \sum_{k \in K_i} \alpha_k = 1$$

3. **Convex combination in probability space**: The sequence-level prediction is a convex combination of window-level posterior probabilities (guaranteeing outputs remain on the probability simplex):

$$\hat{p}^{(i)} = \sum_{k \in K_i} \alpha_k \, p_k$$

4. **From "where" to "why" explainability**: Beyond localization heatmaps, for each high-influence window $k$, the framework provides: (i) the aggregated support score $\bar{s}_k$; and (ii) a ranked neighbor leaderboard $\{(w_k^{(j)}, s_k^{(j)}) : j \in N_k\}$ with timestamps. Since $\partial \alpha_k / \partial s_k^{(j)} = \frac{1}{m\tau} \alpha_k(1 - \alpha_k) > 0$, increasing any neighbor's similarity strictly increases $\alpha_k$, making the leaderboard a faithful attribution explanation.

### Design Motivation

- **Selective amplification**: Windows with high support (patterns consistent with their neighborhood) receive higher weights, while noisy or anomalous windows are down-weighted.
- **Privacy-friendly**: Retrieval is strictly confined to the same recording/channel, with no access to external data.
- **Backbone-agnostic**: Compatible with any classifier, including Transformer variants.

## Key Experimental Results

### Main Results: Multi-Center iEEG Seizure Onset Zone Localization

| Model | F1 | AUC | Accuracy (%) |
|------|-----|------|-----------|
| **RAxSS (cosine)** | 0.6967±0.0791 | **0.8046±0.0346** | 69.76±5.25 |
| **RAxSS (pearson)** | 0.7275±0.0489 | 0.7980±0.0537 | 70.51±3.59 |
| SSS (reproduced) | **0.7437±0.0537** | 0.8035±0.0686 | **71.14±6.31** |
| SSS (original) | 0.7629 | 0.7999 | 72.35 |
| PatchTST | 0.7097 | 0.7852 | 66.83 |
| TimesNet | 0.6897 | 0.7174 | 65.98 |
| ModernTCN | 0.6938 | 0.7305 | 68.42 |
| DLinear | 0.6916 | 0.7044 | 68.41 |
| ROCKET | 0.6847 | 0.7481 | 69.27 |
| Mamba | 0.6452 | 0.7134 | 64.39 |
| GRUs | 0.6948 | 0.7340 | 65.85 |
| LSTM | 0.6709 | 0.7144 | 65.43 |

### Ablation Study: Similarity Function Selection

| Similarity Function | F1 Preference | AUC Preference | Characteristics |
|-----------|--------|--------|------|
| Cosine | Lower (0.697) | **Highest (0.805)** | Prioritizes discriminative capacity |
| Pearson | **Higher (0.728)** | Slightly lower (0.798) | Better F1/accuracy balance |

Clinicians may select accordingly: cosine favors AUC discrimination, while Pearson favors balanced detection.

### Key Findings

1. **RAxSS is competitive with SSS**: The cosine variant achieves the best AUC (0.8046), surpassing the reproduced SSS (0.8035) and all non-SSS baselines.
2. **Clear advantage over fixed-length methods**: Substantial improvements over PatchTST (0.7852 AUC), ROCKET (0.7481 AUC), and other fixed-length approaches.
3. **Explainability as the core added value**: RAxSS provides "why"-level explanations absent from SSS, while maintaining competitive performance.
4. **Cross-center robustness**: Validated on iEEG data from four medical centers: JHH, NIH, UMMC, and UMH.

## Highlights & Insights

1. **Transferring retrieval from forecasting to classification**: RAxSS is the first to adapt RAFT's retrieval augmentation paradigm from time series forecasting to variable-length classification—a clever methodological transfer.
2. **Mathematically guaranteed faithful explanations**: The strict monotonicity of $\alpha_k$ with respect to neighbor similarity ensures that the neighbor leaderboard constitutes a faithful attribution—not a post-hoc approximation.
3. **Theoretical guarantees from convex combination**: Aggregating in probability space ensures that outputs remain valid probability distributions, avoiding potential issues with logit-space aggregation.
4. **Controllability**: The temperature parameter $\tau$ and neighbor count $m$ provide tunable knobs for adjusting the granularity of explanations.

## Limitations & Future Work

- Retrieval is limited to the same channel/recording; cross-subject or cross-center pattern retrieval may enhance evidence quality but introduces privacy concerns.
- Similarity and temperature parameters are currently set manually; data-driven learning of these values would be desirable.
- F1 and accuracy do not surpass SSS, necessitating further tuning and calibration.
- Validation is confined to a single epilepsy iEEG task; generalization to other medical time series (ECG, blood glucose, etc.) remains to be explored.
- Systematic faithfulness stress testing (deletion/insertion tests, retrieval randomization, counterfactual probing) is lacking.

## Related Work & Insights

- **SSS (Mootoo et al.)**: The random sparse sampling framework upon which this work builds by introducing retrieval-based weighting.
- **RAFT**: Retrieval-augmented time series forecasting, leveraging historical patch similarity to improve predictions.
- **PatchTST, TimesNet**: Representative methods for fixed-length TSC.
- **Insight**: Explainability should not be a post-hoc decorative addition, but embedded within the aggregation mechanism itself—"the same evidence both determines the weight and explains the contribution."

## Rating
- Novelty: ⭐⭐⭐⭐☆ — The combination of retrieval augmentation and sparse sampling is novel, and the explainability design is elegant.
- Experimental Thoroughness: ⭐⭐⭐☆☆ — Only one dataset; ablation study lacks depth.
- Writing Quality: ⭐⭐⭐⭐☆ — Method description is clear and mathematical derivations are rigorous.
- Value: ⭐⭐⭐⭐☆ — Provides a practical framework for clinical time series classification that balances performance and explainability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/medical_imaging/thompson_sampling_via_fine-tuning_of_llms.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/medical_imaging/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](../../AAAI2026/medical_imaging/fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)
- [\[NeurIPS 2025\] Scaling Laws and Pathologies of Single-Layer PINNs: Network Width and PDE Nonlinearity](scaling_laws_and_pathologies_of_single-layer_pinns_network_width_and_pde_nonline.md)
- [\[NeurIPS 2025\] LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation](llm-assisted_emergency_triage_benchmark_bridging_hospital-rich_and_mci-like_fiel.md)

</div>

<!-- RELATED:END -->
