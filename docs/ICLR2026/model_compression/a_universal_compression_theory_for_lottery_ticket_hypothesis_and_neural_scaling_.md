---
title: >-
  [Paper Note] A universal compression theory for lottery ticket hypothesis and neural scaling laws
description: >-
  [ICLR 2026][Model Compression][Lottery Ticket Hypothesis] This paper proves a Universal Compression Theorem: any permutation-invariant function over $d$ objects can be asymptotically compressed to polylog(d) objects with…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Lottery Ticket Hypothesis"
  - "Neural Scaling Laws"
  - "Data Compression"
  - "Permutation-Invariant Functions"
  - "Theoretical Proof"
date: 2026-05-08
content_hash: 0b3dab3438143cf2
---

# A universal compression theory for lottery ticket hypothesis and neural scaling laws

**Conference**: ICLR 2026
**arXiv**: [2510.00504](https://arxiv.org/abs/2510.00504)  
**Code**: None  
**Area**: Model Compression
**Keywords**: Lottery Ticket Hypothesis, Neural Scaling Laws, Data Compression, Permutation-Invariant Functions, Theoretical Proof

## TL;DR
This paper proves a Universal Compression Theorem: any permutation-invariant function over $d$ objects can be asymptotically compressed to polylog(d) objects with error approaching zero (which is the optimal compression rate). From this theorem, the authors directly derive: (1) a proof of the dynamic lottery ticket hypothesis — any network can be compressed to polylogarithmic width while preserving its learning dynamics; (2) a dataset compression result — any dataset can be compressed to polylogarithmic size while preserving the loss landscape; and (3) an acceleration of power-law scaling laws to arbitrarily fast decay rates.

## Background & Motivation

### Limitations of Prior Work

**Background**: When training large-scale models, performance typically scales as a slow power law in the number of parameters and dataset size (neural scaling law), i.e., $L \sim d^{-\alpha}$. This raises a fundamental theoretical and practical question: **can comparable performance be achieved with significantly smaller models and significantly less data?**

This question is closely related to two widely studied topics in deep learning:

**Lottery Ticket Hypothesis (LTH)**: The conjecture proposed by Frankle & Carlin (2019) states that a randomly initialized neural network contains a "winning ticket" — a sparse subnetwork that, when trained in isolation, achieves performance comparable to the full network. Despite abundant empirical evidence, providing a theoretical proof of the **dynamic** version of the lottery ticket hypothesis — where the compressed subnetwork not only achieves comparable final performance but also preserves the entire learning dynamics trajectory — has remained an open problem.

**Acceleration of Neural Scaling Laws**: The existing power-law scaling ($L \sim d^{-\alpha}$) implies that performance improvements slow down as scale increases. Does a theoretical pathway exist to accelerate this slow decay to faster rates?

**Key Challenge**: These problems have been extensively explored empirically, but a unified theoretical explanation is lacking. Existing theoretical work either handles only special network architectures or relies on overly strong assumptions.

**Core Idea**: Starting from a compression theory for permutation-invariant functions, this paper provides a unified theoretical framework that simultaneously addresses both questions above.

## Method

### Overall Architecture
The centerpiece of this paper is a **Universal Compression Theorem**. The proof framework proceeds as follows:
1. First, it is shown that any universal permutation-invariant function over $d$ objects can be asymptotically compressed to a function over polylog(d) objects, with error approaching zero.
2. It is then proved that polylog(d) is the optimal compression rate (i.e., no further improvement is possible).
3. Finally, the theorem is applied separately to the domain of neural network parameters (yielding the lottery ticket hypothesis) and to the dataset domain (yielding acceleration of scaling laws).

### Key Designs
1. **Universal Compression Theorem**: The core claim is: given a universal permutation-invariant function $f$ over $d$ objects, there exists a function $\tilde{f}$ over polylog(d) objects such that $\|f - \tilde{f}\| \to 0$ as $d \to \infty$. Here, "permutation-invariant" means the function's output does not depend on the ordering of its inputs — a symmetry that is pervasive in neural networks.

   **Design Motivation**: Many structures in neural networks (e.g., neurons in fully connected layers, attention heads) possess intrinsic permutation invariance or equivariance, making a compression theory grounded in permutation invariance broadly applicable.

2. **Corollary (Ia) — Dynamic Lottery Ticket Hypothesis**: Derived directly from the Universal Compression Theorem: a large neural network can be compressed to polylogarithmic width while **preserving its learning dynamics**. This is stronger than the classical lottery ticket hypothesis — not only is final performance preserved, but the entire training trajectory (including loss curves, gradient flow, etc.) is maintained.

   Formally, for a network of width $d$, there exists a subnetwork of width polylog(d) such that the two exhibit identical learning dynamics trajectories under the same optimization algorithm, in an asymptotic sense.

3. **Corollary (Ib) — Dataset Compression and Scaling Law Acceleration**: Also derived from the Universal Compression Theorem: a large dataset can be compressed to polylogarithmic size while preserving the loss landscape of the corresponding model.

   More importantly, this implies that a neural scaling law of the form $L \sim d^{-\alpha}$ can be accelerated to an arbitrarily fast power-law decay, or ultimately to an exponential decay of the form $\exp(-\alpha' \sqrt[m]{d})$. This provides a theoretical basis for breaking the slow power-law scaling barrier.

### Loss & Training
This paper is a purely theoretical work and does not involve specific training strategies. The core contributions lie in mathematical proofs. Proof techniques include:
- Applying representation theory of permutation groups to analyze the structure of permutation-invariant functions
- Constructively establishing the compression mapping
- Proving the optimality of the polylog(d) compression rate (lower bound proof)
- Instantiating the abstract compression theorem in the concrete settings of neural networks and datasets

## Key Experimental Results

### Main Results
The paper centers on theoretical proofs; experiments primarily serve to validate the plausibility of the theoretical predictions.

| Validation Setting | Theoretical Prediction | Empirical Observation | Consistency |
|---|---|---|---|
| Network width compression | polylog(d) width suffices to preserve performance | Performance is well maintained after substantial compression | Consistent |
| Dataset compression | polylog(d) data points are sufficient | Loss landscape structure is preserved on small datasets | Consistent |
| Scaling law behavior | Decay can be accelerated beyond power-law | Accelerated scaling observed under appropriate conditions | Consistent |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Varying network width | Compression vs. performance curve | Validates the achievability of polylog compression rate |
| Varying dataset size | Loss landscape similarity | Validates preservation of loss landscape after dataset compression |
| Varying scaling exponent $\alpha$ | Decay rate after acceleration | Validates the theoretical prediction that scaling laws can be accelerated |

### Key Findings
- **polylog(d) is the optimal compression rate**: Both the achievability of polylog(d) and its optimality as a lower bound are established.
- **Dynamic lottery ticket hypothesis holds**: A theoretical proof of the dynamic version (preserving learning dynamics) is provided for the first time.
- **Scaling laws can be broken**: The power-law scaling $L \sim d^{-\alpha}$ can theoretically be accelerated to $\exp(-\alpha' \sqrt[m]{d})$, with profound implications for the efficiency of large-scale model training.
- **Permutation invariance is the key**: The source of compressibility lies in the permutation symmetry that is pervasive in neural networks.

## Highlights & Insights
- **Unified theoretical framework**: A single Universal Compression Theorem simultaneously addresses the lottery ticket hypothesis and scaling laws — two seemingly unrelated problems — demonstrating deep theoretical insight.
- **Constructive proof**: The proof is not merely existential but provides an explicit compression construction, enhancing its practical guiding significance.
- **Optimality guarantee**: The polylog(d) rate is proved to be the unimprovable optimal compression rate, establishing a theoretical ceiling for future compression methods.
- **Reinterpretation of scaling laws**: The slow power-law scaling is revealed to be breakable — through appropriate compression strategies, efficiency can be substantially improved.
- **Cross-disciplinary impact**: The theoretical results touch upon statistical physics (disordered systems), information theory, and machine learning.

## Limitations & Future Work
- **Theory-practice gap**: Although the proof is constructive, the algorithm for practically realizing this compression (especially for large Transformers) remains unclear.
- **Asymptotic nature**: The theorem holds in an asymptotic sense ($d \to \infty$); its applicability to finite-scale networks requires further analysis.
- **Permutation invariance assumption**: Not all neural network architectures possess perfect permutation invariance (e.g., positional encodings, fixed architectures); cases that violate this assumption warrant further discussion.
- **Practical operationalizability of scaling law acceleration**: While the theory establishes that scaling laws can be accelerated, how to realize this acceleration in practice remains to be explored.
- **Width compression only**: The analysis of depth compression (reduction in the number of layers) has not been addressed.

## Related Work & Insights
- **Frankle & Carlin (2019) Lottery Ticket Hypothesis**: The original version focuses on preserving static performance; this paper extends the result to preserving the dynamic learning process.
- **Kaplan et al. (2020) Neural Scaling Laws**: Empirically established scaling laws; this paper theoretically reveals the possibility of transcending their limitations.
- **Information Theory and Compression**: Deep connections exist with Kolmogorov complexity and rate-distortion theory in information theory.
- **Permutation-Equivariant Networks**: Works such as Deep Sets and Invariant Theory provide background for understanding permutation invariance.
- **Insights**: Symmetries such as permutation invariance are not merely tools for simplifying computation — they are a fundamental source of compression and generalization. Future model designs can more deliberately exploit such symmetries to improve efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs](../../ACL2026/model_compression/task-stratified_knowledge_scaling_laws_for_post-training_quantized_large_languag.md)
- [\[NeurIPS 2025\] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization](../../NeurIPS2025/model_compression/paretoq_improving_scaling_laws_in_extremely_low-bit_llm_quantization.md)
- [\[ICLR 2026\] Adaptive Width Neural Networks](adaptive_width_neural_networks.md)
- [\[NeurIPS 2025\] The Graphon Limit Hypothesis: Understanding Neural Network Pruning via Infinite Width Analysis](../../NeurIPS2025/model_compression/the_graphon_limit_hypothesis_understanding_neural_network_pruning_via_infinite_w.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](a_recovery_guarantee_for_sparse_neural_networks.md)

</div>

<!-- RELATED:END -->
