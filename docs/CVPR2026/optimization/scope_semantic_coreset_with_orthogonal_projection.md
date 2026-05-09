---
title: >-
  [Paper Note] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning
description: >-
  [CVPR 2026][Optimization][federated learning] This paper proposes SCOPE, a training-free federated coreset selection framework that leverages a frozen VLM (MobileCLIP-S2) with orthogonal projection embeddings to compute three scalar semantic metrics—representativeness, diversity, and boundary proximity—enabling globally-aware two-stage pruning that reduces communication bandwidth by 128–512× while surpassing full-data training.
tags:
  - CVPR 2026
  - Optimization
  - federated learning
  - coreset selection
  - VLM zero-shot
  - long-tail distribution
  - privacy preservation
date: 2026-05-08
content_hash: 9ac7a84fb7a0379f
---

# SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning

**Conference**: CVPR 2026
**arXiv**: [2603.12976](https://arxiv.org/abs/2603.12976)
**Code**: Unavailable
**Area**: Optimization
**Keywords**: federated learning, coreset selection, VLM zero-shot, long-tail distribution, privacy preservation

## TL;DR

This paper proposes SCOPE, a training-free federated coreset selection framework that leverages a frozen VLM (MobileCLIP-S2) with orthogonal projection embeddings to compute three scalar semantic metrics—representativeness, diversity, and boundary proximity—enabling globally-aware two-stage pruning that reduces communication bandwidth by 128–512× while surpassing full-data training.

## Background & Motivation

**Background**: Scientific federated datasets originate from distributed high-precision instruments (microscopes, spectrometers), exhibiting extreme class imbalance (long-tail distributions) and non-IID partitioning by nature. Federated learning mitigates privacy concerns but introduces data efficiency challenges. Coreset selection and data pruning are effective strategies for reducing communication and computational costs.

**Limitations of Prior Work**: (1) Local heuristic methods (FedCS, Herding) lack awareness of the global data distribution and may discard locally redundant but globally rare samples; (2) proxy-dataset-based methods (GCFL) require server-side data, violating privacy constraints; (3) gradient/loss-based methods (EL2N, GraND) amplify sensor noise and artifacts in scientific data; (4) methods requiring local warm-up training (FedCS, FedCore) themselves incur high computational overhead.

**Key Challenge**: In the federated setting, each client has only a local view yet requires global information to perform principled pruning; transmitting embedding vectors enables a global view but violates privacy and incurs high communication cost.

**Goal**: To achieve within the federated setting: (1) training-free coreset selection, (2) globally-aware cross-client class distribution estimation while transmitting only scalars rather than embeddings, and (3) robustness to extreme non-IID and long-tail imbalance.

**Key Insight**: A frozen vision–language model (MobileCLIP-S2) is used locally in a zero-shot manner to extract three scalar metrics per sample; only scalar statistics (mean/variance) are shared with the server to construct a global profile, which in turn guides local two-stage pruning.

**Core Idea**: The frozen VLM's orthogonal projection compresses each sample into three scalar semantic metrics; transmitting only scalar statistics enables global awareness; a two-stage pruning strategy first removes anomalies and then redundancies while protecting long-tail classes.

## Method

### Overall Architecture

Each client uses frozen MobileCLIP-S2 to extract three scalar metrics per sample (RS/DS/S$_{neg}$) → transmits only class-level scalar statistics (mean/variance) to the server → the server aggregates them into a Global Profile via the total variance formula → clients perform two-stage local pruning (consensus filtering + dynamic balancing) based on this profile → standard FedAvg training is conducted on the pruned data.

### Key Designs

1. **Three-Metric Orthogonal Projection Scoring**:
    - **Function**: Uses a frozen VLM in a zero-shot manner to compute three scalar semantic quality metrics per sample.
    - **Mechanism**:
        - Representativeness score $RS_i = v_{img,i} \cdot t_{c_i}$ (cosine similarity between the visual embedding and the ground-truth class text prototype — "Is it a good class prototype?")
        - Diversity score $DS_i = \|v_{res,i}\|_2$, where $v_{res,i} = v_{img,i} - RS_i \cdot t_{c_i}$ (norm of the orthogonal residual — "Does it carry novel features beyond the class definition?")
        - Boundary proximity $S_{neg,i} = \max_{j \neq c_i} v_{img,i} \cdot t_j$ (similarity to the most confusable incorrect class — "Is it prone to misclassification?")
    - **Design Motivation**: Although RS and DS are mathematically related ($DS = \sqrt{1-RS^2}$), they operate in distinct statistical spaces after independent normalization, providing a nonlinear redundancy penalty. The three metrics answer "Is it typical?", "Is it novel?", and "Is it hard?", respectively.

2. **Two-Stage Pruning**:
    - **Function**: First removes semantic anomalies (noise/sensor artifacts), then removes redundant samples while protecting long-tail classes.
    - **Mechanism**:
        - Stage 1 — **Consensus Filtering**: Anomaly score $AS_i = \hat{Z}_{S_{neg},i} - \hat{Z}_{RS,i}$ (z-score-normalized boundary proximity minus representativeness); high $AS$ indicates high confusion and low class representativeness, i.e., an anomaly. The top-$p_l$ samples are pruned.
        - Stage 2 — **Dynamic Balancing**: Redundancy score $R_i = \hat{Z}_{RS,i} - \hat{Z}_{S_{neg},i} - \hat{Z}_{DS,i}$ (high typicality + low confusion + low diversity = redundant). Redundancy pruning is applied only to globally over-represented classes ($T_c = f_c / W_c > \beta$), protecting globally rare classes.
    - **Design Motivation**: The two stages decouple two fundamentally distinct problems — anomalies are a *quality* issue (removed class-agnostically), while redundancy is a *quantity* issue (pruned only within over-represented classes). The global scarcity weight $W_c \propto (1/(F_c+\epsilon))^\gamma$ prevents long-tail classes from being incorrectly pruned.

3. **Global Profile Construction (Privacy-Preserving)**:
    - **Function**: The server aggregates global data distribution information from scalar statistics without receiving any embeddings.
    - **Mechanism**: Each client transmits only the per-class mean, variance, and sample count for the three metrics. The server aggregates cross-client statistics exactly via the total variance formula $[\sigma_{m,c}^{Global}]^2 = \frac{1}{N_c}\sum_k n_{k,c}[[\sigma_{m,c}^k]^2 + [\mu_{m,c}^k - \mu_{m,c}^{Global}]^2]$. Communication complexity is $O(C)$ rather than $O(C \times D)$.
    - **Design Motivation**: Naïve variance averaging underestimates heterogeneity — the total variance decomposition correctly captures both within-client and between-client variance. Scalar transmission achieves 128–512× bandwidth compression.

### Loss & Training

- The coreset selection phase is entirely zero-shot and training-free — only frozen MobileCLIP-S2 geometric projections are used.
- Subsequent federated training: standard FedAvg + SGD + cosine decay, 200 communication rounds; results are reported as the mean over the last 10 rounds.
- Hardware: single A100 GPU per edge node.

## Key Experimental Results

### Main Results

| Dataset | IR | α | $p_f$ | SCOPE | Best Baseline | Full Data |
|--------|-----|---|-------|-------|---------|--------|
| CIFAR-10 | 2 | 0.1 | 0.1 | **56.48%** | FedCore 55.96% | 55.63% |
| CIFAR-10 | 10 | 0.1 | 0.1 | **45.65%** | FedCore 44.98% | 45.07% |
| Tiny-ImageNet | 5 | 0.1 | 0.9 | **55.38%** | Forgetting 54.04% | 54.41% |
| UHCS | 10 | 0.1 | 0.1 | **95.36%** | FedCS 93.17% | 93.99% |
| UHCS | 10 | 0.1 | 0.9 | **92.62%** | EL2N 84.70% | 93.99% |

System efficiency: 128–512× communication bandwidth reduction; 7.72× speedup with ViT-B-16.

### Ablation Study

| Ablation Configuration | CIFAR-10 ($p_f$=0.9) | Change |
|----------|---------------------|------|
| Full SCOPE | 42.80% | — |
| w/o Global Profiling | 19.04% | **−23.76%** |
| w/o Consensus Filter | 40.33% | −2.47% |
| w/o Balancing Filter | 39.76% | −3.04% |

| VLM | Parameters | UHCS Accuracy |
|---------|--------|-----------|
| **MobileCLIP-S2** | **99M** | **94.54%** |
| ViT-H-14 | 986M | 92.35% |

### Key Findings

- SCOPE at $p_f$=0.1 (56.48%) surpasses full-data FedAvg (55.63%), indicating that full data containing noise and imbalance is itself harmful.
- Global Profiling is the overwhelmingly critical component — removing it causes a 23.76% collapse, confirming that federated coreset selection must be globally aware.
- The lightweight MobileCLIP-S2 (99M) outperforms the larger ViT-H-14 (986M), suggesting domain adaptation matters more than model scale.
- Baseline methods suffer catastrophic degradation at high pruning rates (wide error bars), whereas SCOPE remains stable (narrow error bars).
- Under severe heterogeneity (IR=10, α=0.1), SCOPE consistently matches or exceeds full-data training.

## Highlights & Insights

- Entirely training-free coreset selection — frozen VLM geometric scoring avoids the computational overhead of local warm-up training.
- Extremely communication-efficient — only scalar statistics are transmitted, achieving 128–512× bandwidth reduction, making the approach genuinely suitable for the privacy constraints of federated scenarios.
- The geometric intuition behind orthogonal projection decomposition (RS/DS/S$_{neg}$) is clear: sample quality is decomposed into three orthogonal dimensions of "typicality," "novelty," and "ambiguity."
- The decoupled two-stage pruning design is logically coherent — anomalies are a quality problem while redundancy is a quantity problem, and treating them separately is principled.

## Limitations & Future Work

- Relies on the quality of the VLM latent space — VLM representational capacity may be insufficient for specialized scientific domains (e.g., microscopy images).
- Assumes the class label set is known — not applicable to open-set or continual class-incremental scenarios.
- One-shot selection with no support for streaming or online adaptation — requires re-execution when data continuously grows.
- $\beta=0.5$ is fixed across all experiments; more extreme imbalance scenarios may require tuning.

## Related Work & Insights

- **FedCS (CVPR 2025)**: Requires local warm-up training and transmits full feature centroids; long-tail error rate 40.37% vs. SCOPE's 35.60%.
- **FedCore (ICC 2024)**: Requires warm-up training and degrades severely at high pruning rates.
- **EL2N/GraND**: Centralized methods that suffer catastrophic degradation under federated non-IID settings — prioritizing high-loss samples amplifies noise in scientific data.
- **Insights**: Using frozen VLMs for training-agnostic data quality assessment is a promising paradigm; the geometric approach of orthogonal projection decomposition is applicable to other data selection scenarios.

## Rating

- ⭐⭐⭐⭐ **Novelty**: The orthogonal projection three-metric design is novel; leveraging zero-shot VLMs for federated data selection is a creative idea.
- ⭐⭐⭐⭐⭐ **Experimental Thoroughness**: Evaluated on four datasets across multiple imbalance ratios, pruning rates, and backbones, with detailed ablations and system efficiency analysis.
- ⭐⭐⭐⭐ **Writing Quality**: Method formalization is clear; three research questions naturally motivate the design.
- ⭐⭐⭐⭐ **Value**: Practically valuable for data-efficient federated learning, with significant communication efficiency gains.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning](enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p.md)
- [\[CVPR 2026\] Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift](fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_.md)
- [\[ICLR 2026\] Learning to Recall with Transformers Beyond Orthogonal Embeddings](../../ICLR2026/optimization/learning_to_recall_with_transformers_beyond_orthogonal_embeddings.md)
- [\[CVPR 2026\] Dynamic Momentum Recalibration in Online Gradient Learning](dynamic_momentum_recalibration_in_online_gradient_learning.md)
- [\[CVPR 2026\] The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers](the_power_of_decaying_steps_enhancing_attack_stability_and_transferability_for_s.md)

<!-- RELATED:END -->
