---
title: >-
  [Paper Note] A universal compression theory for lottery ticket hypothesis and neural scaling laws
description: >-
  [ICLR 2026][Model Compression][Paper Note] The paper proves a universal compression theorem: any permutation-invariant function can be asymptotically compressed to a $\text{polylog}(d)$ scale with error approaching zero (which is the optimal compression rate). This directly leads to the proof of the dynamic lottery ticket hypothesis—any network can be compresse
tags:
  - ICLR 2026
  - Model Compression
date: 2026-05-08
content_hash: 91392133705f17d6
---
# A universal compression theory for lottery ticket hypothesis and neural scaling laws

**Conference**: ICLR 2026  
**arXiv**: [2510.00504](https://arxiv.org/abs/2510.00504)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Lottery Ticket Hypothesis, Neural Scaling Laws, Data Compression, Permutation Invariant Functions, Theoretical Proof

## TL;DR
The paper proves a universal compression theorem: any permutation-invariant function can be asymptotically compressed to a $\text{polylog}(d)$ scale with error approaching zero (which is the optimal compression rate). This directly leads to the proof of the dynamic lottery ticket hypothesis—any network can be compressed to polylogarithmic width while maintaining invariant learning dynamics—as well as dataset compression to polylogarithmic size while maintaining the loss landscape, and the acceleration of power-law scaling laws to arbitrarily fast decay rates.

## Background & Motivation

**Background**: When training large-scale models, performance typically scales according to a slow power law regarding parameter count and dataset size (neural scaling law), i.e., $L \sim d^{-\alpha}$. Empirically, the exponent $\alpha$ for language models is mostly between 0.1–0.3. This raises a fundamental theoretical and practical question: **Can comparable performance be achieved with significantly smaller models and significantly less data?** The authors highlight this gap with an intuitive comparison—the human brain masters a native language with an input of approximately $10^8$ words, while contemporary large models often require trillions of tokens, a difference in data efficiency of about four orders of magnitude.

This problem is closely related to two widely discussed but separately studied topics in deep learning:

**Limitations of Prior Work 1: Lack of proof for the dynamic version of the Lottery Ticket Hypothesis**: The Lottery Ticket Hypothesis (LTH) by Frankle & Carlin (2019) suggests that randomly initialized networks contain "winning tickets"—sparse subnetworks that can match the full network's performance when trained independently. While empirical evidence is abundant, the **dynamic version** (where the compressed subnetwork not only matches final performance but also maintains the entire learning dynamics trajectory) has lacked theoretical proof.

**Limitations of Prior Work 2: Whether the slow power law of scaling laws can be broken**: The power-law scaling $L \sim d^{-\alpha}$ implies that as scale increases, returns diminish. Currently, there is no principled method to answer whether this slow decay can be accelerated and how fast it can go.

**Key Challenge**: These two phenomena have been explored empirically but lack a unified theoretical explanation; existing theories either cover only specific network architectures or rely on overly strong assumptions.

**Core Idea**: The authors found that both phenomena share the same underlying structure—permutation symmetry. By treating network neurons and data samples as interchangeable "objects," a single compression theorem for permutation-invariant functions can simultaneously address both questions.

## Method

### Overall Architecture
The entire paper revolves around a **Universal Compression Theorem**: any permutation-invariant function acting on $d$ objects can be asymptotically approximated by a function acting on only $\text{polylog}(d)$ objects, with the error tending to zero as $d\to\infty$. This logarithmic compression rate is proven to be the unimprovable optimal value. Interpreting the objects as neurons yields the dynamic lottery ticket hypothesis; interpreting them as data samples yields dataset compression and scaling law acceleration. Thus, two seemingly unrelated phenomena are unified under a single theorem.

### Key Designs

**1. Universal Compression Theorem: Converting Permutation Symmetry into Logarithmic Compression Rates**

The theorem starts from the pervasive permutation invariance in large models—swapping the order of neurons in a fully connected layer or heads in an attention mechanism does not change the function computed by the network. This redundancy is the root of compressibility. The theorem formally asserts: for any universal permutation-invariant function $f$ acting on $d$ objects, there exists a function $\tilde{f}$ depending on only $\text{polylog}(d)$ objects such that $\|f-\tilde{f}\|\to 0$ as $d\to\infty$. The proof uses the representation theory of permutation groups to characterize the structure of $f$ and constructively provides a compression mapping from $d$ dimensions to $\text{polylog}(d)$ dimensions. Thus, the conclusion is not just existential but provides a specific compression target. The accompanying lower bound proof shows that $\text{polylog}(d)$ is the limit: further compression prevents the error from vanishing, fixing the theoretical ceiling for all subsequent compression methods at the logarithmic level.

**2. Corollary (Ia) — Dynamic Lottery Ticket Hypothesis: Compressing to Logarithmic Width while Preserving the Training Trajectory**

By substituting network neurons for "objects" in the theorem, a stronger conclusion than the traditional LTH is obtained: for a network of width $d$, there exists a subnetwork of width only $\text{polylog}(d)$ that, under the same optimization algorithm, not only matches final performance but also remains asymptotically consistent across the entire learning dynamics trajectory (loss curves, gradient flow). The original version by Frankle & Carlin only guaranteed that "winning tickets" could match final accuracy after independent training (a static conclusion); here, the entire training process is preserved, hence the term "Dynamic Lottery Ticket Hypothesis," providing the first theoretical proof for this problem.

**3. Corollary (Ib) — Dataset Compression and Scaling Law Acceleration: Replacing Slow Power Laws with Fast Decay**

By substituting data samples for "objects," a large dataset can be compressed to logarithmic size while maintaining the loss landscape of the corresponding model. A more critical corollary falls on scaling laws: the empirical power law $L\sim d^{-\alpha}$ implies performance improves increasingly slowly as scale increases. The compression theorem demonstrates that this slow decay can be accelerated into arbitrarily fast power laws or even exponential forms $\exp(-\alpha'\sqrt[m]{d})$. This theoretically provides a possible path to break the slow power law and approach equivalent performance with significantly less data.

### Mechanism and Applicability
This paper is a purely theoretical work; the core contributions lie in the mathematical proofs. The technical route is: first, using a Deep Sets-style representation $f(w_1,\dots,w_d)=h\big(\sum_i g(w_i)\big)$ to write any smooth permutation-invariant function in an analyzable form; then, constructively providing a compression mapping from $d$ objects to $\text{polylog}(d)$ objects; establishing that the $\text{polylog}(d)$ rate is optimal within a constant factor via lower bound proofs; and finally, specializing the abstract theorem to neural network width (Corollary Ia) and dataset scale (Corollary Ib). Note that the theorem relies on smoothness and permutation symmetry assumptions. Structures that violate these (e.g., position embeddings, non-smooth ReLU) require additional discussion, and the conclusions hold in the asymptotic sense as $d\to\infty$.

## Key Experimental Results

### Main Results
As a theoretical paper, the experiments primarily serve to verify the plausibility of the theoretical predictions.

| Validation Scenario | Theoretical Prediction | Experimental Observation | Consistency |
|----------|---------|---------|--------|
| Network Width Compression | $\text{polylog}(d)$ width suffices for performance | Performance maintained after significant compression | Consistent |
| Dataset Compression | $\text{polylog}(d)$ data volume is sufficient | Loss landscape structure maintained on small datasets | Consistent |
| Scaling Law Behavior | Acceleration to faster-than-power-law decay | Accelerated scaling observed under suitable conditions | Consistent |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Different Network Widths | Compression vs. Performance curve | Verifies the achievability of the $\text{polylog}$ rate |
| Different Dataset Sizes | Loss landscape similarity | Verifies loss landscape preservation after dataset compression |
| Different Scaling Exponents $\alpha$ | Decay rate after acceleration | Verifies theoretical predictions of accelerated scaling laws |

### Key Findings
- **$\text{polylog}(d)$ is the optimal compression rate**: Not only is its reachability proven, but it is also shown to be a lower bound that cannot be further improved.
- **Dynamic Lottery Ticket Hypothesis holds**: Provides the first theoretical proof for the dynamic version (maintaining learning dynamics).
- **Scaling laws can be broken**: Theoretically, the power-law scaling $L \sim d^{-\alpha}$ can be accelerated to $\exp(-\alpha' \sqrt[m]{d})$, which has profound implications for the efficiency of large-scale model training.
- **Permutation invariance is key**: The source of compression capability lies in the widespread permutation symmetry within neural networks.

## Highlights & Insights
- **Unified Theoretical Framework**: Answers two seemingly unrelated questions—the lottery ticket hypothesis and scaling laws—with a single universal compression theorem, demonstrating deep theoretical insight.
- **Constructive Proof**: The proof is not merely existential but provides a specific compression construction, enhancing its practical guiding significance.
- **Optimality Guarantee**: Proves that $\text{polylog}(d)$ is the unimprovable optimal rate, setting a theoretical upper bound for future compression methods.
- **Rethinking Scaling Laws**: Reveals that slow power-law scaling is not unbreakable—efficiency can be significantly improved through appropriate compression strategies.
- **Cross-disciplinary Impact**: Results involve statistical physics (disordered systems), information theory, and machine learning.

## Limitations & Future Work
- **Gap between Theory and Practice**: Although the proof is constructive, the algorithm to implement this compression (especially for large Transformers) remains unclear.
- **Asymptotic Properties**: The theorem holds in the asymptotic sense ($d \to \infty$); applicability to finite-scale networks needs further analysis.
- **Permutation Invariance Assumption**: Not all neural network architectures possess perfect permutation invariance (e.g., position encodings, fixed architectures), necessitating more discussion on violations of this assumption.
- **Practical Operability of Scaling Law Acceleration**: While theoretically possible, how to achieve this acceleration in actual training remains to be explored.
- **Width Compression Only**: The analysis of depth compression (layer reduction) has not yet been covered.

## Related Work & Insights
- **Frankle & Carlin (2019) Lottery Ticket Hypothesis**: The original version focused on static performance preservation; this paper extends it to the preservation of the dynamic learning process.
- **Kaplan et al. (2020) Neural Scaling Laws**: Empirically established scaling laws; this paper theoretically reveals the possibility of breaking their limits.
- **Information Theory and Compression**: Deeply connected to Kolmogorov complexity and Rate-Distortion theory in information theory.
- **Permutation Equivariant Networks**: Works such as Deep Sets and Invariant Theory provide context for understanding permutation invariance.
- **Insight**: Symmetry (such as permutation invariance) is not just a tool for simplifying computation but a fundamental source of compression and generalization. Future model designs could more consciously exploit these symmetries to improve efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](../../ICML2026/model_compression/model_merging_scaling_laws_in_large_language_models.md)
- [\[ICLR 2026\] SGD-Based Knowledge Distillation with Bayesian Teachers: Theory and Guidelines](sgd-based_knowledge_distillation_with_bayesian_teachers_theory_and_guidelines.md)
- [\[ICML 2026\] LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws](../../ICML2026/model_compression/llms_as_noisy_channels_a_shannon_perspective_on_model_capacity_and_scaling_laws.md)
- [\[ACL 2026\] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs](../../ACL2026/model_compression/task-stratified_knowledge_scaling_laws_for_post-training_quantized_large_languag.md)
- [\[ICLR 2026\] UNITE: Universal Knowledge Integration from Task-Specific Experts](unite_universal_knowledge_integration_from_task-specific_experts.md)

</div>

<!-- RELATED:END -->
