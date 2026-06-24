---
title: >-
  [Paper Note] Bregman Conditional Random Fields: Sequence Labeling with Parallelizable Inference
description: >-
  [ACL 2025][CRF] Proposal of Bregman CRF (Bcrf), a novel discriminative model for sequence labeling based on mean regularization. It implements a parallelizable inference algorithm using iterative Bregman projection to replace the inherently sequential Viterbi/Forward algorithms in traditional CRFs. Bcrf achieves performance on par with standard CRFs on POS, NER, and word segmentation tasks while being faster, and outperforms Mean Field methods in scenarios with forbidden labe…
tags:
  - "ACL 2025"
  - "CRF"
  - "sequence labeling"
  - "Bregman projection"
  - "parallel inference"
  - "structured prediction"
date: 2026-05-08
content_hash: 675b6c064993928e
---

# Bregman Conditional Random Fields: Sequence Labeling with Parallelizable Inference

**Conference**: ACL 2025  
**arXiv**: [2506.00732](https://arxiv.org/abs/2506.00732)  
**Code**: [GitHub](https://github.com/FilippoC/lightspeed-crf)  
**Area**: Others  
**Keywords**: CRF, sequence labeling, Bregman projection, parallel inference, structured prediction

## TL;DR
Proposal of Bregman CRF (Bcrf), a novel discriminative model for sequence labeling based on mean regularization. It implements a parallelizable inference algorithm using iterative Bregman projection to replace the inherently sequential Viterbi/Forward algorithms in traditional CRFs. Bcrf achieves performance on par with standard CRFs on POS, NER, and word segmentation tasks while being faster, and outperforms Mean Field methods in scenarios with forbidden label transition constraints.

## Background & Motivation

**Background**: Sequence labeling is a core NLP task (POS tagging, NER, word segmentation), with Conditional Random Fields (CRFs) being the mainstream structured model. CRF inference relies on Viterbi (MAP inference) and Forward (marginal inference) algorithms.

**Limitations of Prior Work**: Viterbi and Forward are dynamic programming algorithms with a time complexity of $O(n|T|^2)$. However, they are inherently **sequential**—computation at each position depends on the previous position (dependencies across stages), which prevents full utilization of GPU parallelism. Although intra-stage parallelization (wavefront parallelization) is possible, the $n$ steps across stages must be executed serially.

**Key Challenge**: Modern deep learning heavily relies on GPU parallel acceleration (e.g., Transformers, SSMs), but the CRF layer remains a bottleneck for training and inference.

**Goal**: Design an alternative model that is performance-equivalent to standard CRFs but features parallelizable inference.

**Key Insight**: Inspired by entropic optimal transport (Cuturi, 2013) and SparseMAP (Niculae et al., 2018), this work uses **mean regularization** to replace the distribution regularization of standard CRFs, enabling inference to be decomposed into parallelizable KL projection subproblems.

**Core Idea**: Reformulate CRF inference from sequential dynamic programming to iterative Bregman projection—decompose the marginal polytope into the intersection of even and odd constraint sets, perform alternating projections, and fully parallelize the subproblems within each step.

## Method

### Overall Architecture
Bcrf replaces the standard CRF partition function $A_Y(\mathbf{w})$ with $B_Y(\mathbf{w}) = \max_{\mathbf{q} \in \text{conv}(Y)} \langle \mathbf{q}, \mathbf{w} \rangle + H(\mathbf{q})$. The key difference is that the regularization term $H(\mathbf{q})$ operates on marginal probability vectors (mean regularization) rather than the distribution vector (distribution regularization).

### Key Designs

1. **Mean Regularization vs. Distribution Regularization**

    - Standard CRF: $A_Y(\mathbf{w}) = \max_{\mathbf{p} \in \Delta(Y)} \langle \mathbf{p}, \mathbf{M}^\top\mathbf{w} \rangle + H(\mathbf{p})$, where $\mathbf{p}$ is an exponentially-dimensioned distribution.
    - Bcrf: $B_Y(\mathbf{w}) = \max_{\mathbf{q} \in \text{conv}(Y)} \langle \mathbf{q}, \mathbf{w} \rangle + H(\mathbf{q})$, where $\mathbf{q}$ is a polynomially-dimensioned marginal vector.
    - Design Motivation: Optimizing over marginal space allows the regularization term to have a simple analytical form, paving the way for efficient solvers.

2. **Graph-Theoretic Characterization of the Marginal Polytope**

    - Sequence labeling is modeled as a path-selection problem in a directed graph: each position corresponds to a cluster $V_i$, tags correspond to nodes within the cluster, and edges represent transitions between adjacent positions.
    - conv(Y) is equivalent to the set of non-negative vectors satisfying flow conservation constraints (entering cluster = 1, and flow conservation at each node).
    - By even-odd decomposition, conv(Y) is represented as $\mathcal{C}_{\text{even}} \cap \mathcal{C}_{\text{odd}}$.

3. **Iterative Bregman Projection (IBP) Inference**

    - Function: Computes $\nabla B_Y(\mathbf{w})$ by alternating projections onto $\mathcal{C}_{\text{even}}$ and $\mathcal{C}_{\text{odd}}$.
    - Mechanism: Projecting onto $\mathcal{C}_{\text{even}}$ is decomposed into $\lceil n/2 \rceil - 1$ independent, small-scale KL projections (each involving only the incoming and outgoing arcs of a single cluster), which are **fully parallelizable**.
    - Design Motivation: KL projections onto the intersection of two convex sets are guaranteed to converge via alternating projections (Bregman, 1967).
    - Complexity: $O(n/2)$ independent subproblems per step, each taking $O(|T|^2)$ time; $k$ iterations take $O(k \cdot n/2 \cdot |T|^2)$ total time but with a highly parallel inner structure.

4. **Fenchel-Young Loss + Partial Label Learning**

    - Function: Defines the supervised and weakly supervised loss functions for Bcrf.
    - Mechanism: The FY loss is defined as $\ell_{-H}(\mathbf{w}; \mathbf{y}) = -\langle \mathbf{w}, \mathbf{y} \rangle - H(\mathbf{y}) + B_Y(\mathbf{w})$, with gradients equal to $-\mathbf{y} + \nabla B_Y(\mathbf{w})$, which only requires calling the IBP algorithm.
    - Partial labels: $\tilde{\ell}_{-H}(\mathbf{w}; \tilde{Y}) = B_Y(\mathbf{w}) - B_{\tilde{Y}}(\mathbf{w})$, which requires two calls to IBP.
    - Design Motivation: Supports both full-label and partial-label scenarios in a unified framework.

## Key Experimental Results

### POS Tagging (4 Languages, Universal Dependencies)

| Method | Dutch | English | French | German | Parallelizable? |
|------|-------|---------|--------|--------|------|
| Unstructured | 93.4 | 90.8 | 96.0 | 94.0 | ✓ |
| CRF (Viterbi) | 94.7 | 91.9 | 96.2 | 94.3 | ✗ |
| Mean Field 5it | 93.8 | 90.6 | — | — | ✓ |
| **Bcrf 10it** | **94.5** | **91.7** | **96.1** | **94.2** | ✓ |

### NER (CoNLL, with Forbidden Transition Constraints)

| Method | F1 | Details |
|------|----|----|
| CRF (Viterbi) | ~91.5 | Gold standard |
| Mean Field | ~88.2 | Cannot handle forbidden transitions $\to$ severe degradation |
| **Bcrf** | **~91.3** | Close to CRF, correctly handles constraints |

### Key Findings
- **Bcrf $\approx$ CRF Performance**: Performs almost on par with standard CRF (gap <0.3%) on POS, NER, and word segmentation.
- **Bcrf $\gg$ Mean Field (with Constraints)**: The BIO tagging in NER contains forbidden transitions (e.g., I-PER cannot follow B-LOC). Mean Field (Mf) cannot model these constraints, leading to severe performance degradation. Bcrf handles these constraints correctly by encoding them as $-\infty$.
- **Convergence in ~10 Steps**: IBP basically converges after 10 iterations, with diminishing returns for extra steps.
- **Faster on GPUs**: Bcrf achieves faster wall-clock time than sequential Viterbi on long sequences, as its internal subproblems are fully parallelized.

## Highlights & Insights
- **An Elegant Bridge from Optimal Transport to NLP Structured Prediction**: Introducing Sinkhorn-like iterative projection methods from entropic OT into CRF represents a brilliant and elegant cross-domain transfer of technology.
- **The Ingenuity of Even-Odd Decomposition**: Splitting flow constraints into two groups based on the parity of positions makes subproblems within the same group completely independent and parallelizable. This observation is key to the feasibility of the entire method.
- **Partial Label Support**: Training CRFs with partial labels typically requires the Forward algorithm (which is sequential). Bcrf supports it naturally while maintaining parallelism.
- **Alignment with GPU Hardware Trends**: As models grow larger and GPU parallelism increases, the sequential bottleneck of CRF layers becomes more prominent. Bcrf's value will scale with hardware evolution.

## Limitations & Future Work
- **Approximate Inference**: IBP yields approximate solutions rather than exact MAP or marginal probabilities, with the approximation quality highly dependent on the number of iterations.
- **Temperature Hyperparameter $\tau$**: Emulating MAP requires a small $\tau$, but it cannot be 0 due to numerical instability, requiring manual tuning.
- **Limited to Linear Chain**: Restricted to first-order Markov dependencies, and has not yet been extended to semi-Markov or higher-order CRFs.
- **Insufficient Ablation on Transformer Integration**: The end-to-end comparison with encoder + Bcrf is not sufficiently comprehensive.
- **Theoretical Convergence Rate**: While convergence of IBP is guaranteed, its rates are not theoretically fully analyzed.

## Related Work & Insights
- **vs. Mean Field (Wang et al., 2020)**: Mf is also parallelizable but (a) is non-convex, (b) lacks convergence guarantees, and (c) cannot handle forbidden transitions. Bcrf consistently outperforms Mf.
- **vs. SparseMAP (Niculae et al., 2018)**: SparseMAP also uses mean regularization but employs DP (which is still sequential) as its inner loop solver. Bcrf's IBP is truly parallelizable.
- **vs. Entmax/Sparsemax**: They handle sparse distributions but do not directly address parallelization of sequence-level inference problems.
- **Insights**: The IBP framework could potentially be extended to tree-structured CRFs (such as constituency or dependency parsing) by dividing the flow constraints into parallelizable blocks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elegant theory, seamlessly introducing Bregman projections from optimal transport into CRF inference, with ingenious even-odd decomposition tricks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage including POS, NER, word segmentation, partial labels, and speed comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivations with a clear sequence from definition to proposition to algorithm.
- Value: ⭐⭐⭐⭐ Offers a GPU-friendly alternative to CRFs, possessing substantial practical value for NLP structured prediction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Fields to Random Trees](../../ICLR2026/others/from_fields_to_random_trees.md)
- [\[ACL 2025\] HelpSteer3: Human-Annotated Feedback and Edit Data to Empower Inference-Time Scaling](helpsteer3_human-annotated_feedback_and_edit_data_to_empower_inference-time_scal.md)
- [\[ACL 2025\] HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference](hata_trainable_and_hardware-efficient_hash-aware_top-k_attention_for_scalable_la.md)
- [\[ACL 2025\] CoAM: Corpus of All-Type Multiword Expressions](coam_corpus_of_all-type_multiword_expressions.md)
- [\[ICML 2025\] DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers](../../ICML2025/others/dsp_dynamic_sequence_parallelism_for_multi-dimensional_transformers.md)

</div>

<!-- RELATED:END -->
