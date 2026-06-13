---
title: >-
  [Paper Note] From Signal Degradation to Computation Collapse: Uncovering the Two Failure Modes of LLM Quantization
description: >-
  [ACL 2026][Model Compression][Post-Training Quantization] Through systematic mechanistic interpretability analysis, this paper reveals two qualitatively different failure modes in LLM quantization: signal degradation in…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Post-Training Quantization"
  - "Signal Degradation"
  - "Computation Collapse"
  - "Mechanistic Interpretability"
  - "Causal Tracing"
  - "Knowledge Recall"
  - "PTQ"
date: 2026-05-08
content_hash: 960c4df1efa3c385
---

# From Signal Degradation to Computation Collapse: Uncovering the Two Failure Modes of LLM Quantization

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.19884](https://arxiv.org/abs/2604.19884)  
**Code**: None  
**Area**: Model Quantization / Interpretability  
**Keywords**: Post-Training Quantization, Signal Degradation, Computation Collapse, Mechanistic Interpretability, Causal Tracing, Knowledge Recall, PTQ

## TL;DR

Through systematic mechanistic interpretability analysis, this paper reveals two qualitatively different failure modes in LLM quantization: signal degradation in 4-bit (complete computational patterns with compromised precision, partially repairable) and computation collapse in 2-bit (functional destruction of key components, requiring structural reconstruction).

## Background & Motivation

**Background**: Post-training quantization (PTQ) is a critical technology for the efficient deployment of LLMs. While 4-bit quantization is widely regarded as the optimal balance between precision and compression, 2-bit quantization typically triggers a catastrophic "performance cliff," where accuracy plummets to near zero.

**Limitations of Prior Work**: Existing research focuses on three directions: (1) macro-evaluation (measuring the magnitude of performance loss); (2) algorithmic improvement (numerical optimizations such as outlier suppression and rotation matrices); and (3) preliminary mechanistic exploration (layer/component sensitivity analysis). A common limitation is treating quantization damage solely as a "numerical problem" without exploring why internal model mechanisms fail.

**Key Challenge**: Is the catastrophic failure of 2-bit quantization a cumulative "quantitative change" of 4-bit degradation, or does it represent a qualitative shift? If it is a qualitative shift, it implies that current numerical optimization-based repair strategies are fundamentally misdirected for 2-bit scenarios.

**Goal**: This study aims to reveal internal mechanism differences in quantization failures through systematic mechanistic interpretability analysis (layer-wise information flow, causal paths, component functions, and representation spaces) and to verify that different failure modes require distinct repair strategies.

**Key Insight**: Quantization failure is likened to a signal processing problem—is the signal weakened by noise (degradation), or is the computation pipeline itself broken (collapse)?

**Core Idea**: The failures of 4-bit and 2-bit are essential differences rather than differences in degree. Signal degradation can be recovered through targeted training-free repairs, whereas computation collapse requires structural reconstruction (e.g., fine-tuning). This difference provides the strongest evidence for distinguishing the two modes.

## Method

**Overall Architecture**: Using Llama-3.1-8B as the primary subject, the study systematically compares the internal behaviors of FP16, 4-bit, and 2-bit models on a factual knowledge recall task (Pararel). A five-layer analysis establishes and verifies hypotheses: Macro Phenomena $\rightarrow$ Layer-wise Probing $\rightarrow$ Causal Analysis $\rightarrow$ Component/Representation Verification $\rightarrow$ Mechanism-oriented Intervention.

**Key Designs**:

1.  **Multi-level Knowledge Signal Tracing**
    *   **Function**: Traces the existence and causal transmission integrity of knowledge signals within the model.
    *   **Mechanism**: Employs Logit Lens to project hidden states into the vocabulary space layer-by-layer, observing changes in the probability and rank of the correct token. In 4-bit, signals appear in middle-to-late layers but with weakened intensity (degradation); in 2-bit, signals remain near zero (absence). Cross-model causal activation patching further verifies this: injecting "clean" FP16 activations into key positions of the quantized model (last subject token) allows 4-bit models to recover, whereas 2-bit models remain completely non-responsive.
    *   **Design Motivation**: Distinguishing "weakened signals" from "signals never generated" is core evidence for the two-mode hypothesis.

2.  **Component-level Functional Diagnosis (Attention + FFN Key-Value Memory)**
    *   **Function**: Pinpoints which specific components fail and how they fail.
    *   **Mechanism**: Uses normalized entropy (global concentration) + JSD divergence (focus deviation) for attention layers. For FFN layers, it uses the gating Sign Flip Rate (SFR, $>30\%$ indicates severe instability), Jaccard overlap of Top-1% activated neurons ($\approx 0.1$ indicates complete activation misalignment), and output cosine similarity ($\approx 0$ indicates complete semantic deviation). 2-bit quantization shows functional collapse across all indicators.
    *   **Design Motivation**: Attributing macro "signal loss" to specific component failures confirms whether the issue is a loss of precision or a loss of function.

3.  **Mechanism-Aware Two-Stage Repair vs. System Irreversibility Verification**
    *   **Function**: Verifies the fundamental difference in repairability between the two modes.
    *   **Mechanism**: A "Source Protection + Signal Recovery" scheme is designed for 4-bit: protecting the first few layers (8-bit for the first 2 layers in Llama/Mistral, $\approx 4.25$ avg bits; kurtosis-based selection for Qwen/Gemma, $\approx 4.1$ avg bits) + peak signal amplification ($\alpha$-times logit scale). These strategies and EORA low-rank compensation prove ineffective for 2-bit. A "domino experiment" shows that quantizing only the first 2 layers leads to a drop from $100\%$ to $41.65\%$.
    *   **Design Motivation**: Differences in repairability serve as the most direct and powerful practical evidence for distinguishing the two modes.

## Key Experimental Results

**4-bit Repair Experiments (Accuracy on Failure Subset)**:

| Model | Baseline (4-bit) | + Basic Repair | + Signal Amp (Final) |
| :--- | :--- | :--- | :--- |
| Llama3.1-8B | 0.00% | 67.91% | 75.19% ($\alpha=3$) |
| Mistral-7B | 0.00% | 66.86% | 81.26% ($\alpha=9$) |
| Qwen3-8B | 0.00% | 40.24% | 79.88% ($\alpha=7$) |
| Gemma2-9B | 0.00% | 33.85% | 64.08% ($\alpha=2$) |

**2-bit "Domino Effect" (Llama3.1-8B)**:

| Quantized Layers | Robust Subset | Failure Subset |
| :--- | :--- | :--- |
| None (FP16) | 100.00% | 100.00% |
| Layer 0 | 65.47% | 15.03% |
| Layers 0-1 | 41.65% | 5.29% |
| Layers 0-5 | 2.51% | 0.38% |

**Representation Space Analysis**:
*   4-bit: CKA maintains a clear diagonal structure, with activation subspace similarity to FP16 $>0.8$.
*   2-bit: CKA is almost entirely dark (structural collapse), with activation subspace similarity $\approx 0$.
*   4-bit error subspace alignment with signals is $\approx 0.3$ (resembling random noise).
*   2-bit error subspace alignment with signals is $\approx 0.8$ (directly interfering with core features).

**Key Findings**:
*   4-bit results in a "drop in answer rank" (correct answer remains in Top-5), while 2-bit results in "rank collapse" (dropping to thousands, equivalent to random guessing).
*   Architecture-dependent degradation: Llama/Mistral exhibit an "early representation bottleneck," while Qwen/Gemma show "uniform degradation."
*   2-bit models cannot process signals correctly even when receiving high-precision inputs—the components themselves have failed.
*   The distinction between the two failure modes is consistent across both GPTQ and AWQ methods.

## Highlights & Insights

*   **Valuable Framework for Qualitative Distinction**: This is the first systematic proof that 4-bit and 2-bit are not different degrees on the same continuum but two fundamentally different failure modes.
*   **Closed Loop from Diagnosis to Repair**: Mechanistic analysis directly guides the design of repair strategies, and the variance in repair effectiveness further validates the diagnosis.
*   **Compelling "Domino Experiment"**: Demonstrating that quantizing just the first two layers in 2-bit leads to catastrophic collapse—unrecoverable by 30 subsequent FP16 layers—visually illustrates the irreversibility of computation collapse.
*   **Deep Insight into Error Directions**: The high alignment of 2-bit quantization error with the signal subspace implies that noise is not random but systematically destroys the model's core features.

## Limitations & Future Work

*   The study focuses on weight-only quantization; failure modes in activation quantization remain to be explored.
*   Evaluations are anchored in factual recall tasks; performance in complex reasoning tasks needs verification.
*   Repair strategies require additional precision overhead ($\approx 4.1-4.25$ avg bits), and their practicality needs optimization.
*   The boundary between the two modes (behavior of 3-bit) warrants further research.
*   The cutoff point for failure modes may vary across different model architectures.

## Related Work & Insights

*   **GPTQ (Frantar et al., 2023)**: The most widely used weight-only PTQ method and the primary quantization baseline in this paper.
*   **Causal Tracing (Meng et al., 2022)**: A knowledge localization method extended here for cross-model repair experiments.
*   **Logit Lens (nostalgebraist, 2020)**: A tool for intermediate layer decoding.
*   **SpQR (Dettmers et al., 2023)**: A mixed-precision method echoed by the source protection strategy used in this paper.
*   **Insight**: Quantization research should move beyond numerical optimization; mechanistic understanding is vital for overcoming performance bottlenecks. Making 2-bit practical requires a shift from "compensation" to "reconstruction."

## Rating

*   **Novelty**: ★★★★★ — The systematic distinction and verification of two failure modes is a fresh and significant contribution.
*   **Experimental Thoroughness**: ★★★★★ — The evidence chain is complete, covering four models, multi-level analysis, and multiple metric validations.
*   **Writing Quality**: ★★★★★ — The narrative is extremely clear, progressing logically from phenomena to hypothesis, verification, and intervention.
*   **Value**: ★★★★☆ — Provides a critical diagnostic framework and mechanistic insights for future quantization research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Two-Stage Regularization-Based Structured Pruning for LLMs](two-stage_regularization-based_structured_pruning_for_llms.md)
- [\[ICLR 2026\] ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](../../ICLR2026/model_compression/paroquant_pairwise_rotation_quantization_for_efficient_reasoning_llm_inference.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[ICML 2026\] WUSH: Near-Optimal Adaptive Transforms for LLM Quantization](../../ICML2026/model_compression/wush_near-optimal_adaptive_transforms_for_llm_quantization.md)
- [\[ICLR 2026\] The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm](../../ICLR2026/model_compression/the_geometry_of_llm_quantization_gptq_as_babais_nearest_plane_algorithm.md)

</div>

<!-- RELATED:END -->
