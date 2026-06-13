---
title: >-
  [Paper Note] SHAP Meets Tensor Networks: Provably Tractable Explanations with Parallelism
description: >-
  [NeurIPS 2025][Audio & Speech][SHAP] This paper presents the first provably exact SHAP computation framework for Tensor Networks (TNs)…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "SHAP"
  - "Tensor Networks"
  - "Tensor Train"
  - "Explainability"
  - "Parallel Computation"
  - "Binarized Neural Networks"
  - "Parameterized Complexity"
date: 2026-05-08
content_hash: 18232ff99930b34f
---

# SHAP Meets Tensor Networks: Provably Tractable Explanations with Parallelism

**Conference**: NeurIPS 2025
**arXiv**: [2510.21599](https://arxiv.org/abs/2510.21599)  
**Code**: To be confirmed  
**Area**: Audio & Speech
**Keywords**: SHAP, Tensor Networks, Tensor Train, Explainability, Parallel Computation, Binarized Neural Networks, Parameterized Complexity

## TL;DR

This paper presents the first provably exact SHAP computation framework for Tensor Networks (TNs), proves that SHAP under the Tensor Train (TT) structure is parallelizable in polylogarithmic time (NC² complexity), and reveals via reductions that **width rather than depth** is the fundamental bottleneck for SHAP computation in binarized neural networks.

## Background & Motivation

1. **Intractability of SHAP computation**: Shapley values, as the most widely adopted post-hoc explanation method, are NP-Hard to compute exactly for black-box models such as neural networks; existing exact algorithms are limited to simple models like decision trees, which have insufficient expressive power.
2. **Limitations of approximation methods**: Approaches such as KernelSHAP, DeepSHAP, and FastSHAP reduce computational cost via sampling or heuristics, but provide no exactness guarantees and are therefore unreliable in high-stakes applications.
3. **Unique advantages of tensor networks**: Originating from quantum physics, TNs combine high expressive power with structured transparency—their function approximation capacity is comparable to certain neural network families, while their physics-inspired structure facilitates the derivation of efficient exact solutions.
4. **Dual demands of model compression and interpretability**: TNs are widely used for neural network abstraction and compression; providing exact SHAP algorithms for TNs simultaneously addresses interpretability after model compression.
5. **Theoretical gap in parallel computation**: Prior literature has never analyzed the **parallelizability** of SHAP from a computational complexity perspective—i.e., when SHAP can be completed in polylogarithmic time—a question critical to large-scale deployment.
6. **Lack of fine-grained analysis of SHAP hardness in neural networks**: Although it is known that SHAP for neural networks is NP-Hard, which structural parameters (depth vs. width vs. sparsity) constitute the bottleneck has not been studied at a fine-grained level.

## Method

### Overall Architecture: SHAP Computation for General Tensor Networks

- **Function**: Provides a provably exact SHAP computation framework for arbitrarily structured TNs.
- **Design Motivation**: The SHAP formula involves summation over all feature subsets (exponential cardinality), making direct computation infeasible; the structure of TNs must be exploited to avoid exponential enumeration.
- **Mechanism**:
  1. **Tensorized representation** (Proposition 1): Rewrites the SHAP formula as a contraction between the **modified weighted coalition tensor** $\tilde{\mathcal{W}}$ and the **marginal value tensor** $\mathcal{V}^{(M,P)}$, i.e., $\mathcal{T}^{(M,P)} = \tilde{\mathcal{W}} \times_S \mathcal{V}^{(M,P)}$.
  2. **Construction of the weighted coalition tensor** (Lemma 1): $\tilde{\mathcal{W}}$ can be represented as a sparse TT with core dimension $n_{in}^2$, constructible in $O(\log n_{in})$ time using $O(n_{in}^3)$ parallel processors.
  3. **Construction of the marginal value tensor** (Lemma 2): Introduces a "routing tensor" $\mathcal{M}^{(i)}$ to simulate the interventional mechanism—determining whether to use the input instance value or the distributional expectation based on whether each feature belongs to subset $S$—then contracts with the model TN and distribution TN.
  4. **Final SHAP matrix**: The SHAP values for input $x$ are obtained by contracting the marginal SHAP tensor with the rank-1 tensor of the input (Equation 3).

### Key Design 1: Efficient Parallel SHAP for Tensor Trains (TT)

- **Function**: Proves that SHAP computation belongs to the NC² complexity class when both the model and distribution are TT-structured.
- **Design Motivation**: SHAP for general TNs is #P-Hard (Proposition 2), necessitating structural restrictions for efficient algorithms; TT is the most widely used TN subfamily, featuring a one-dimensional chain topology.
- **Mechanism**:
  1. **Core theorem** (Theorem 1): Proves that when both $\mathcal{T}^M$ and $\mathcal{T}^P$ are TTs, the marginal SHAP tensor is itself a TT, with cores formed by contracting $\mathcal{M}^{(i)} \times \mathcal{I}^{(i)} \times \mathcal{P}^{(i)} \times \mathcal{G}^{(i)}$.
  2. **Parallel scan** (Proposition 3): TT contraction can be performed via the classical parallel prefix scan at depth $O(\log^2 n_{in})$; combined with the fact that matrix multiplication belongs to NC¹, the overall algorithm belongs to NC².
  3. **Reductions from other model families** (Theorem 2): Decision trees, tree ensembles, linear models, and linear RNNs can all be reduced to equivalent TTs within NC, thereby inheriting the NC² SHAP complexity—tightening all previously known polynomial-time results.

### Key Design 2: Fine-Grained Complexity Analysis of Binarized Neural Networks (BNNs)

- **Function**: Analyzes the influence of depth, width, and sparsity on the computational hardness of SHAP for BNNs using parameterized complexity theory.
- **Design Motivation**: Prior work only established that SHAP for neural networks is NP-Hard, without distinguishing which structural parameters are the bottleneck, providing no guidance for designing networks with interpretability guarantees.
- **Mechanism** (Theorem 3):
  1. **Fixed depth remains hard** (para-NP-Hard): Via reduction from 3SAT—any CNF formula can be encoded as a depth-2 BNN, and SHAP for CNFs is #P-Hard.
  2. **Fixed width is tractable** (XP): The BNN is converted layer-by-layer into a 2D TN and then back-contracted to an equivalent TT; compilation time is $O(R^W \cdot \text{poly}(D, n_{in}))$, which is polynomial for fixed $W$.
  3. **Fixed width and sparsity yields efficient computation** (FPT): With an additional constraint on reified cardinality $R$, the combinatorial explosion is confined to $R^W$ terms, giving a running time of $g(W,R) \cdot n^{O(1)}$, which is FPT.

## Key Experimental Results

This paper is a **theoretical work**; its core contributions are complexity-theoretic proofs rather than empirical evaluations. The following key theoretical results are summarized:

| Model Family | Previously Known Complexity | Ours | Distribution Class |
|:---|:---|:---|:---|
| Decision Trees | P (TreeSHAP) | **NC²** | TT distributions (incl. independent/empirical/Markov/HMM/Born Machine) |
| Tree Ensembles | P | **NC²** | TT distributions |
| Linear Models | P | **NC²** | TT distributions |
| Linear RNNs | P | **NC²** | TT distributions |
| Tensor Trains | First studied | **NC²** | TT distributions |
| BNN (fixed depth) | — | **para-NP-Hard** | Empirical/independent/TT |
| BNN (fixed width) | — | **XP** | Empirical/independent/TT |
| BNN (fixed width + sparsity) | — | **FPT** | Empirical/independent/TT |

**Key Findings**:

- For all classical ML models, SHAP is not only polynomial-time solvable but can also be computed in parallel in polylogarithmic time.
- In BNNs, width (rather than depth) is the critical dividing line for SHAP tractability: fixing depth leaves the problem NP-Hard, while fixing width renders it tractable.
- The distribution class supported by TT is far more expressive than the independent/empirical distributions used in prior algorithms, encompassing complex feature dependencies.

## Highlights & Insights

1. **First exact SHAP algorithm for tensor networks**, bridging the gap between highly expressive models and provably exact explanations.
2. **First analysis of the parallel computability of SHAP**, tightening classical results such as TreeSHAP from P to NC² and laying a theoretical foundation for large-scale parallel deployment.
3. **The "width > depth" insight for BNNs** is highly instructive—demonstrating that designing narrow and sparse binarized networks can simultaneously achieve compression and interpretability.
4. **Unified framework**: The TT-based reduction provides a unified treatment of SHAP complexity for diverse model families including decision trees, linear models, and RNNs.

## Limitations & Future Work

1. **Purely theoretical contribution**: No empirical validation of actual SHAP computation (runtime, GPU acceleration, etc.) is provided; practical deployability remains to be assessed.
2. **Restricted to binarized neural networks**: The BNN results cannot be directly generalized to full-precision networks, which are more prevalent in practice.
3. **Limitations of TT structure**: SHAP for general TNs remains #P-Hard; TT requires a one-dimensional chain topology, and some model families may not admit efficient reduction to TT.
4. **Other explanation methods not addressed**: Complexity comparisons with alternative explanation frameworks such as Banzhaf values and LIME are not explored.
5. **Distribution assumptions**: Although TT distributions are highly expressive, whether real-world data distributions can be well approximated by TTs requires further investigation.

## Related Work & Insights

### vs TreeSHAP (Lundberg et al., 2020)
TreeSHAP provides polynomial-time exact SHAP algorithms for decision trees and tree ensembles, but is limited to independent feature distributions or the internal structure of tree models. This paper tightens the complexity from P to NC² via TT reduction and supports the broader class of TT distributions (including Markov chains, HMMs, and other dependency structures), achieving theoretical improvements in both distributional expressiveness and computational efficiency.

### vs Van den Broeck et al. (2022) / Arenas et al. (2024)
These works explore SHAP complexity under knowledge compilation formalisms such as Boolean circuits and decomposable deterministic negation normal forms (d-DNNFs). The TN framework proposed in this paper subsumes these formalisms in expressive power (TTs can encode decision trees, linear models, and other model families), and introduces parallel complexity analysis for the first time, providing a more fine-grained computational-theoretic perspective.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First deep integration of tensor network theory with SHAP computation; pioneering analysis of parallel complexity)
- Experimental Thoroughness: ⭐⭐ (Purely theoretical; no empirical validation, though the theoretical proofs are rigorous and complete)
- Writing Quality: ⭐⭐⭐⭐ (Formally rigorous with clear illustrations; some proofs are overly brief and require consulting the appendix)
- Value: ⭐⭐⭐⭐ (Significantly advances XAI theory; the BNN width insight has practical implications for network design)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Toward Complex-Valued Neural Networks for Waveform Generation](../../ICLR2026/audio_speech/toward_complex-valued_neural_networks_for_waveform_generation.md)
- [\[ACL 2026\] Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages](../../ACL2026/audio_speech/indic-codecfake_meets_satyam_towards_detecting_neural_audio_codec_synthesized_sp.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)
- [\[NeurIPS 2025\] Target Speaker Extraction Through Comparing Noisy Positive and Negative Audio Enrollments](target_speaker_extraction_through_comparing_noisy_positive_and_negative_audio_en.md)
- [\[NeurIPS 2025\] VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction](vita-15_towards_gpt-4o_level_real-time_vision_and_speech_interaction.md)

</div>

<!-- RELATED:END -->
