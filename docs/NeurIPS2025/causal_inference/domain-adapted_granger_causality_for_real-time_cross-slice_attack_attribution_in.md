---
title: >-
  [Paper Note] Domain-Adapted Granger Causality for Real-Time Cross-Slice Attack Attribution in 6G Networks
description: >-
  [NEURIPS2025][Causal Inference][Granger causality] This paper proposes a domain-adapted Granger causality framework for 6G network slicing that integrates enhanced Granger causality testing with network resource contention modeling to enable real-time cross-slice attack attribution, achieving 89.2% accuracy and 87 ms response time across 1,100 attack scenarios, substantially outperforming existing statistical, deep learning, and causal discovery methods.
tags:
  - NEURIPS2025
  - Causal Inference
  - Granger causality
  - 6G network slicing
  - cross-slice attack attribution
  - resource contention
  - real-time security
date: 2026-05-08
content_hash: 2fd9113be0a60e47
---

# Domain-Adapted Granger Causality for Real-Time Cross-Slice Attack Attribution in 6G Networks

**Conference**: NEURIPS2025
**arXiv**: [2510.05165](https://arxiv.org/abs/2510.05165)
**Code**: To be confirmed (authors declare Apache 2.0 open-source release)
**Area**: Causal Inference
**Keywords**: Granger causality, 6G network slicing, cross-slice attack attribution, resource contention, real-time security

## TL;DR

This paper proposes a domain-adapted Granger causality framework for 6G network slicing that integrates enhanced Granger causality testing with network resource contention modeling to enable real-time cross-slice attack attribution, achieving 89.2% accuracy and 87 ms response time across 1,100 attack scenarios, substantially outperforming existing statistical, deep learning, and causal discovery methods.

## Background & Motivation

**Core Problem**: 6G networks partition shared physical infrastructure into multiple logical networks via network slicing to serve diverse services (eMBB/URLLC/mMTC), but resource sharing allows attacks to propagate across slices, making attribution extremely challenging.

**Limitations of Prior Work**: Existing approaches suffer from three major deficiencies: (a) correlation-based methods (e.g., Pearson correlation) yield high false positive rates (30.6%) and cannot distinguish causal from spurious associations; (b) deep learning methods (e.g., Transformer-XL) achieve relatively high accuracy but exceed 200 ms response time, failing to meet real-time requirements; (c) purely statistical causal methods (e.g., VAR-Granger) neglect resource contention—a domain-specific causal pathway.

**Opportunity and Limitation of Granger Causality**: Classical Granger causality infers causation from improvements in time-series prediction and is a canonical framework for temporal causal inference. However, when directly applied to multi-slice 6G scenarios, shared resources (CPU/memory/network) act as important confounders, generating numerous spurious causal relationships if left uncontrolled.

**Real-Time Constraint**: 6G security orchestration requires sub-100 ms response times, ruling out computationally expensive methods (PC algorithm: 156 ms; Transformer-XL: 234 ms).

**Interpretability Requirement**: Automated security orchestration systems require interpretable causal explanations to guide response strategies, which black-box deep learning methods cannot readily provide.

**Industrial Urgency**: In IoT and industrial automation scenarios, cross-slice attacks can cause real production-safety hazards (e.g., manufacturing control system latency violations triggering emergency shutdowns).

## Method

### Overall Architecture

The Domain-Adapted Granger Causality framework receives security telemetry streams $\{x_t^{(i)}\}$ from $N$ slices together with resource allocation data $A(t)$, and operates through three core components working in concert: (1) enhanced Granger causality testing to eliminate resource-induced confounding; (2) resource contention modeling to capture domain-specific causal pathways; and (3) integrated causal strength scoring combined with the Viterbi algorithm to extract the optimal attack path. The overall algorithmic complexity is $O(N^2 \cdot W \cdot (p+q+K) + N^3 \cdot \log N)$, which guarantees sub-100 ms execution under typical deployments ($N \leq 50$, $W = 300$).

### Module 1: Enhanced Granger Causality Testing

- **Function**: For each slice pair $(s_i, s_j)$, tests whether the telemetry time series of $s_i$ has statistically significant causal predictive power over $s_j$, while controlling for confounding effects of shared resources.
- **Mechanism**: Shared resource control terms $Z_t = [Z_{1,t}, \ldots, Z_{K,t}]^\top$ are introduced into the standard Granger regression. The unrestricted model is $Y_t = \sum \alpha_i Y_{t-i} + \sum \beta_j X_{t-j} + \sum \gamma_k Z_{k,t} + \varepsilon_t$; the restricted model ($H_0: \beta_j = 0$) is $Y_t = \sum \alpha_i Y_{t-i} + \sum \gamma_k Z_{k,t} + \eta_t$. The enhanced F-statistic is $F = \frac{(RSS_R - RSS_U)/q}{RSS_U/(T-p-q-K-1)}$, which follows an $F(q,\, T-p-q-K-1)$ distribution under $H_0$.
- **Design Motivation**: Shared resources are the primary confounding source in 6G scenarios—two slices may exhibit correlated telemetry solely because they share a CPU. The resource conditioning term $\gamma_k Z_{k,t}$ explicitly blocks the confounding path $X \leftarrow R \rightarrow Y$. Compared with standard Granger causality, resource conditioning yields an 8.2 percentage-point accuracy gain (74.1% → 82.3%).

### Module 2: Resource Contention Modeling

- **Function**: Captures causal pathways mediated through shared resource mechanisms that are missed by purely statistical methods, by modeling inter-slice resource contention intensity.
- **Mechanism**: Contention intensity is defined as $\rho_{ij}(t) = \sum_k w_k \cdot A_{ik}(t) \cdot A_{jk}(t) \cdot \sigma(U_{k,t} - \tau_k)$, where $A_{ik}(t)$ is the normalized resource allocation ratio, $U_{k,t}$ is the resource utilization rate, $w_k$ are learnable criticality weights, $\tau_k$ are contention thresholds, and $\sigma(\cdot)$ is the sigmoid function.
- **Design Motivation**: The product term $A_{ik} \cdot A_{jk}$ acts as a logical AND gate—contention arises only when both slices simultaneously use the same resource. The sigmoid term implements a threshold effect—contention becomes significant only after resource utilization exceeds the threshold. Linear weighted summation allows the model to automatically learn the relative importance of different resources (CPU/memory/network). Weights learned in experiments (CPU 0.45, memory 0.31, network 0.24) are intuitively consistent. This module contributes an additional 4.7 percentage-point accuracy gain (82.3% → 87.0%).

### Module 3: Integrated Causal Strength and Attack Path Extraction

- **Function**: Fuses statistical and domain causal evidence into a unified score and extracts the optimal attack path from the causal graph.
- **Mechanism**: The integrated causal strength is $\Gamma_{ij}(t) = \omega_1 \cdot \phi(F_{ij}(t)) + \omega_2 \cdot \rho_{ij}(t)$, where $\phi$ denotes min-max normalization and $\{\omega_1, \omega_2\}$ are learned via MLE (optimal values: $\omega_1 = 0.67$, $\omega_2 = 0.33$). Benjamini–Hochberg correction controls the FDR at 0.05; edges satisfying the threshold $\tau_{\text{causal}}$ form the causal graph $G$, and the Viterbi algorithm solves for the optimal path $C^* = \arg\max \prod \Gamma_{ij}(T)$.
- **Design Motivation**: Linear fusion offers better algorithmic stability and interpretability than nonlinear fusion in real-time systems. BH correction addresses the multiple comparison problem arising from $N(N-1)$ pairwise tests. Sensitivity analysis shows that accuracy fluctuates by only $\pm 1.5$ percentage points for $\omega_1 \in [0.55, 0.80]$, demonstrating robustness to parameter tuning.

### Loss & Training

- Parameters $\theta = \{w_1, \ldots, w_K,\, \tau_1, \ldots, \tau_K,\, \omega_1, \omega_2\}$ are learned by maximizing the regularized log-likelihood: $\mathcal{L}(\theta) = \sum_m \log P(C^{(m)} \mid X^{(m)}, A^{(m)}, \theta) - \lambda \|\theta\|_2^2$.
- Hyperparameters are selected via 5-fold cross-validation (optimal $\lambda^* = 10^{-3}$).
- Theoretical guarantees: Theorem 1 proves the asymptotic $F$-distribution of the enhanced F-statistic under $H_0$; Theorem 2 proves that causal relationships are uniquely identified with probability $\geq 1 - N(N-1)\alpha$ under the DAG assumption.

## Key Experimental Results

### Table 1: Main Performance Comparison (N = 1,100 attack scenarios, 5-fold cross-validation)

| Method | Accuracy (%) | Precision (%) | Recall (%) | FDR (%) | Response Time (ms) |
|---|---|---|---|---|---|
| Pearson Correlation | 72.9±1.8 | 69.4±2.1 | 76.2±1.9 | 30.6±2.1 | 21±3 |
| Transfer Entropy | 78.4±1.5 | 75.8±1.7 | 81.3±1.6 | 24.2±1.7 | 58±7 |
| VAR-Granger | 74.1±1.7 | 71.2±1.9 | 77.6±1.8 | 28.8±1.9 | 43±5 |
| PC Algorithm | 76.2±1.6 | 73.5±1.8 | 79.4±1.7 | 26.5±1.8 | 156±20 |
| GraphSAGE | 76.8±1.6 | 74.3±1.8 | 79.9±1.7 | 25.7±1.8 | 142±18 |
| LSTM-Attention | 79.1±1.4 | 76.7±1.6 | 82.2±1.5 | 23.3±1.6 | 167±22 |
| Transformer-XL | 81.3±1.3 | 78.9±1.5 | 84.1±1.4 | 21.1±1.5 | 234±31 |
| **Ours** | **89.2±0.9** | **87.6±1.1** | **91.1±1.0** | **12.4±1.1** | **87±9** |

### Table 2: Extended Metric Comparison

| Method | AUC-ROC | AUC-PR | F1 | MCC | Memory (MB) |
|---|---|---|---|---|---|
| Correlation | 0.742±0.018 | 0.689±0.021 | 0.726±0.019 | 0.461±0.024 | 12±2 |
| Transfer Entropy | 0.798±0.015 | 0.751±0.017 | 0.784±0.016 | 0.572±0.019 | 28±4 |
| Transformer-XL | 0.827±0.013 | 0.789±0.015 | 0.815±0.014 | 0.634±0.017 | 756±48 |
| **Ours** | **0.921±0.009** | **0.876±0.011** | **0.892±0.010** | **0.785±0.012** | **67±8** |

### Key Findings

1. The proposed method outperforms the strongest baseline, Transformer-XL, by 7.9 percentage points in accuracy ($p < 0.001$, Cohen's $d > 1.5$), while achieving 2.7× faster response (87 ms vs. 234 ms) and requiring only 1/11 of its memory.
2. Ablation studies quantify individual component contributions: resource conditioning +8.2 pp, resource contention modeling +4.7 pp, and integrated learning +2.2 pp, all significant at $p < 0.001$.
3. The framework maintains 84.3% accuracy under 60% partial observability, exhibiting graceful performance degradation.
4. In an industrial IoT attack case study, the complete 5-hop attack chain is reconstructed with 96.3% accuracy, zero false positives, and 73 ms response time, whereas the traditional correlation method produces 11 false alarms.

## Highlights & Insights

1. **Principled Integration of Statistical Causality and Domain Knowledge**: Rather than naively stacking components, the framework learns optimal fusion weights via MLE, and the necessity of each component is theoretically and empirically substantiated through ablation studies.
2. **Rare Achievement of Both Real-Time Performance and High Accuracy**: The 87 ms response time satisfies the strict real-time constraints of 6G security orchestration while surpassing the heavyweight Transformer-XL in accuracy.
3. **Interpretable Outputs**: Each causal edge is supported by dual explanations—an F-statistic p-value and a resource contention score—making the framework suitable for automated security decision-making.
4. **Elegant Design of Resource Contention Modeling**: The three-layer product–sigmoid–weighted structure is simultaneously compact and physically intuitive.

## Limitations & Future Work

1. The weak stationarity assumption may be violated during rapid nonlinear attack evolution; although resource conditioning partially mitigates this, fundamentally non-stationary scenarios (e.g., abrupt changes in attack patterns) may still cause failures.
2. Reliable attribution requires approximately 2 seconds of telemetry data, which may be insufficient for ultra-fast attacks.
3. The $O(N^3 \log N)$ complexity requires approximation algorithms for very large-scale deployments ($N > 50$).
4. The product-weighted linear structure of the resource contention model may inadequately capture more complex nonlinear interference patterns.
5. Experiments are conducted on a single production-grade testbed, lacking generalization validation across operators or geographic regions.

## Related Work & Insights

- **Classical Granger Causality [Granger, 1969; Shojaie & Fox, 2024]**: The core methodological foundation of this work; the enhancements lie in resource conditioning and domain fusion.
- **Transfer Entropy [Schreiber, 2000]**: An information-theoretic approach to temporal causal inference, achieving 78.4% accuracy in the experiments—inferior to the resource conditioning strategy proposed here.
- **Graph Neural Network Methods [GraphSAGE, Hamilton et al., 2017]**: Learn the spatial structure of slice relationships but lack temporal causal reasoning and resource contention modeling.
- **Granger Causality in IoT Security [Begum et al., 2025; Lv et al., 2024]**: Preliminary applications of Granger causality to IoT scenarios, without domain adaptation tailored to resource contention characteristics of 6G multi-slice networks.
- **Causal Discovery Methods [PC, DirectLiNGAM]**: Structural learning based on conditional independence testing; computationally expensive and unsuitable for real-time scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ — Innovatively combines Granger causality with network resource contention modeling, opening a new direction in 6G security research.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage across 1,100 scenarios, multiple baselines, ablation analysis, sensitivity analysis, and an industrial case study.
- Writing Quality: ⭐⭐⭐⭐ — Clear framework description, complete algorithmic pseudocode, rigorous theoretical proofs, and rich figures and tables.
- Value: ⭐⭐⭐⭐ — Provides a practical framework for 6G network security attribution; theoretical guarantees combined with real-time performance make it deployment-ready.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Revealing Multimodal Causality with Large Language Models](revealing_multimodal_causality_with_large_language_models.md)
- [\[NeurIPS 2025\] GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding](gst-unet_a_neural_framework_for_spatiotemporal_causal_inference_with_time-varyin.md)
- [\[NeurIPS 2025\] Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features](causality-induced_positional_encoding_for_transformer-based_representation_learn.md)
- [\[ACL 2026\] Cross-Modal Taxonomic Generalization in (Vision-) Language Models](../../ACL2026/causal_inference/cross-modal_taxonomic_generalization_in_vision-_language_models.md)
- [\[NeurIPS 2025\] Transferring Causal Effects using Proxies](transferring_causal_effects_using_proxies.md)

<!-- RELATED:END -->
