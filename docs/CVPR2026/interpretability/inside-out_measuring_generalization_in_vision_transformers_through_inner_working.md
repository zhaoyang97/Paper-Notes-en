---
title: >-
  [Paper Note] Inside-Out: Measuring Generalization in Vision Transformers Through Inner Workings
description: >-
  [CVPR 2026][Interpretability][generalization measurement] This paper proposes generalization performance prediction metrics based on model-internal circuits…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "generalization measurement"
  - "circuit discovery"
  - "Vision Transformer"
  - "distribution shift"
  - "mechanistic interpretability"
date: 2026-05-08
content_hash: 483b418ee602ffdd
---

# Inside-Out: Measuring Generalization in Vision Transformers Through Inner Workings

**Conference**: CVPR 2026
**arXiv**: [2604.08192](https://arxiv.org/abs/2604.08192)
**Code**: [GitHub](https://github.com/deep-real/GenCircuit)
**Area**: Interpretability
**Keywords**: generalization measurement, circuit discovery, Vision Transformer, distribution shift, mechanistic interpretability

## TL;DR

This paper proposes generalization performance prediction metrics based on model-internal circuits, including Dependency Depth Bias (DDB) for pre-deployment model selection and Circuit Shift Score (CSS) for post-deployment performance monitoring, improving average correlation over existing proxy metrics by 13.4% and 34.1%, respectively.

## Background & Motivation

Reliable generalization evaluation is critical for machine learning deployment, especially in high-stakes settings with scarce annotations (e.g., medical imaging). The core challenges arise from two practical scenarios:

1. **Pre-deployment model selection**: How to select the best model without labeled target data? In-distribution (ID) accuracy is unreliable (the underspecification problem: models with similar ID accuracy can exhibit vastly different OOD performance).
2. **Post-deployment performance monitoring**: How to detect performance degradation under continuous distribution shift? Confidence-based metrics are unreliable (overconfidence problem: high confidence is assigned even to incorrect predictions).

Existing proxy metrics (e.g., confidence scores, accuracy-on-the-line, RANKME) analyze only the model's **external behavior** (output probabilities or feature quality), ignoring the **internal mechanisms** that produce these outputs.

**Core Idea**: This paper leverages circuit discovery techniques from Mechanistic Interpretability to extract generalization signals from internal computation paths — because *how* a model computes is more reflective of its generalization ability than *what* it computes.

## Method

### Overall Architecture

1. Extract continuous edge-weight circuits from ViTs via the EAP-IG method.
2. Aggregate circuits into an Inter-layer Dependency Matrix (IDM).
3. Pre-deployment: Discover "Generalization Motifs" via CCA and design the DDB metric.
4. Post-deployment: Monitor performance degradation via the circuit-shift measure CSS.

### Key Designs

1. **Continuous Circuit Definition and Discovery**:
    - **Function**: Represent the ViT computation graph as a continuous edge-weight map, quantifying the causal importance of each edge to model behavior.
    - **Mechanism**: For the ViT computation graph $\mathcal{G}=(\mathcal{V}, \mathcal{E})$, define the circuit as an edge-weight function $c(e) = \mathbb{E}_{x \sim \mathcal{D}}[KL(\mathcal{M}_{\setminus\{e\}}(x), \mathcal{M}(x))]$, i.e., the KL divergence of model output after removing edge $e$. Mean ablation is adopted over interchange ablation, as it is better suited for vision tasks. The EAP-IG method balances faithfulness and computational efficiency.
    - **Design Motivation**: Binary circuits discard fine-grained information; continuous relaxation preserves richer structural information necessary for generalization assessment. The entire process requires no labels.

2. **Dependency Depth Bias (DDB) — Pre-deployment Metric**:
    - **Function**: Quantify the model's relative reliance on deep vs. shallow features to predict OOD generalization ability.
    - **Mechanism**: Circuit edge weights are aggregated into an inter-layer dependency matrix $\Lambda_{ij}$. CCA is applied to discover cross-task "universal generalization motifs" — well-generalizing models rely on deep paths (∇-shaped), while poorly-generalizing models rely on shallow shortcuts (Δ-shaped). DDB is defined as the log ratio of the summed deep edge weights to shallow edge weights: $DDB = \log(\sum_{deep}/\sum_{shallow})$.
    - Three variants: DDB_global (global), DDB_deep (deep-to-deep connections), DDB_out (connections to output nodes).
    - **Design Motivation**: Deep layers encode more abstract, domain-invariant semantic representations, while shallow layers capture domain-specific spurious correlations. $\tau=0.3$ yields the best performance.

3. **Circuit Shift Score (CSS) — Post-deployment Metric**:
    - **Function**: Measure the degree of circuit shift on OOD data relative to an ID baseline, to predict performance degradation.
    - **Mechanism**: It is observed that the inter-layer topology of circuits remains stable post-deployment, but edge rewiring increases with distribution shift. CSS is defined as $CSS = d(\mathcal{R}(c_{ID}), \mathcal{R}(c_{OOD}))$, supporting two representation families: vectorized (cosine/ℓ2/SRCC distance) and graph-structured (Laplacian/NetLSD/Jaccard).
    - **Design Motivation**: Post-deployment comparison is made between circuits of the same model on different data; inter-layer topology no longer provides consistent signals (CCA yields contradictory generalization motifs across datasets), necessitating fine-grained rewiring pattern measurement. CSS(v, SRCC) performs best, indicating that **relative ranking changes** of edge weights are more reliable than absolute magnitudes.

### Loss & Training

No training is involved. Threshold calibration for CSS uses 39 corruption domains from CIFAR10-C as proxy data to simulate distribution shift; the CSS value corresponding to the corruption domain closest to the performance threshold $\delta$ is selected as the alert threshold $\delta'$.

## Key Experimental Results

### Main Results — Pre-deployment Model Selection

| Dataset | Metric | DDB_out (Ours) | Prev. SOTA | Gain |
|--------|------|---------------|---------|------|
| PACS (style shift) | R²/SRCC/KRCC | 0.862/0.897/0.731 | 0.765/0.878/0.720 (ID Acc) | +13% |
| Camelyon17 (institution shift) | R²/SRCC/KRCC | 0.748/0.820/0.646 | 0.588/0.802/0.628 (ATC) | +22% |
| Terra Incognita (geographic shift) | R²/SRCC/KRCC | 0.714/0.838/0.642 | 0.684/0.813/0.613 (DDB_global) | +5% |
| **Average** | Composite score | **0.766±0.029** | 0.632±0.047 (ID Acc) | **+13.4%** |

### Main Results — Post-deployment Performance Monitoring

| Dataset | Metric | CSS(v,SRCC) (Ours) | Prev. SOTA | Gain |
|--------|------|-------------------|---------|------|
| PACS | R²/SRCC/KRCC | 0.912/0.983/0.944 | 0.645/0.617/0.444 (ATC) | +78% |
| FMoW | R²/SRCC/KRCC | 0.723/0.750/0.722 | 0.428/0.717/0.611 (MDE) | +29% |
| Camelyon17 | R²/SRCC/KRCC | 0.519/0.807/0.608 | 0.036/0.273/0.187 (MDE) | +187% |
| ImageNet | R²/SRCC/KRCC | 0.953/0.961/0.855 | 0.942/0.957/0.861 (ATC) | +1% |
| **Average** | Composite score | **0.811±0.041** | 0.470±0.095 (ATC) | **+34.1%** |

### Ablation Study

| Configuration | R² | SRCC | KRCC | Note |
|------|-----|------|------|------|
| τ=0.1 | 0.744 | 0.743 | 0.562 | Shallow/deep boundary too narrow |
| τ=0.2 | 0.772 | 0.843 | 0.653 | Suboptimal |
| τ=0.3 | **0.798** | **0.862** | **0.684** | Best |
| τ=0.4 | 0.801 | 0.849 | 0.671 | Near-best |
| τ=0.5 | 0.772 | 0.838 | 0.673 | Mid-point boundary |

### Key Findings

- DDB closely aligns with training dynamics of OOD accuracy: well-generalizing models show DDB increasing from 2.6 to 4.1, while poorly-generalizing models stagnate around −0.9.
- Vectorized CSS substantially outperforms graph-structured CSS, indicating that fine-grained edge-weight patterns are more informative than coarse-grained topological similarity.
- Different datasets exhibit distinct circuit shift patterns: FMoW shows broad cross-layer changes, while Camelyon17 concentrates changes in deep layers.
- CSS alert F1 improves by approximately 45% over the best baseline within the clinically acceptable performance range (0.8–0.9).

## Highlights & Insights

- This work pioneers the transition of mechanistic interpretability from "post-hoc explanation" to "predictive metric," establishing a new paradigm for model evaluation.
- The "universal generalization motif" reveals an intuitively consistent regularity: well-generalizing models rely more on deep abstract features, while poorly-generalizing models rely on shallow surface features.
- CSS detects "silent failures" without labels, offering significant practical value in high-stakes domains such as healthcare.
- The transfer of circuit discovery from language models to vision models validates the existence of interpretable generalization patterns in vision Transformers.

## Limitations & Future Work

- The computational cost of circuit discovery is high, limiting the practicality of real-time post-deployment monitoring.
- Validation is restricted to the ViT architecture; applicability to CNNs or hybrid architectures remains unknown.
- Constructing the model zoo (72–144 ViTs) is costly, and such a large pool of candidate models may not be available in practice.
- The threshold calibration strategy relies on artificially constructed corruption datasets; its robustness under real-world distribution shifts requires further verification.

## Related Work & Insights

- **vs. Accuracy-on-the-Line**: This observation assumes a linear ID–OOD accuracy relationship, which is violated by the underspecification phenomenon; DDB avoids this issue by examining internal structure rather than external behavior.
- **vs. Confidence-based metrics (AC/ANE/MDE)**: Confidence scores only reflect output probability distributions and are severely affected by overconfidence; CSS measures changes in computation paths and is therefore more reliable.
- **vs. RANKME/α-ReQ**: Feature quality metrics assess representation space properties but do not account for how features are used by subsequent layers; DDB captures the full inter-layer dependency structure.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First application of circuit discovery to generalization prediction; paradigm-level innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three pre-deployment and four post-deployment datasets, multiple baselines, thorough ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear; the two scenarios are treated as self-contained, well-structured threads.
- **Value**: ⭐⭐⭐⭐ Provides practical guidance for model evaluation and monitoring, though computational cost limits applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Measuring the (Un)Faithfulness of Concept-Based Explanations](measuring_the_unfaithfulness_of_concept-based_explanations.md)
- [\[ICLR 2026\] Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](../../ICLR2026/interpretability/grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)
- [\[CVPR 2026\] SteelDefectX: A Coarse-to-Fine Vision-Language Dataset and Benchmark for Generalizable Steel Surface Defect Detection](steeldefectx_a_coarse-to-fine_vision-language_dataset_and_benchmark_for_generali.md)
- [\[NeurIPS 2025\] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](../../NeurIPS2025/interpretability/causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [\[ICLR 2026\] Implicit Statistical Inference in Transformers: Approximating Likelihood-Ratio Tests In-Context](../../ICLR2026/interpretability/implicit_statistical_inference_in_transformers_approximating_likelihood-ratio_te.md)

</div>

<!-- RELATED:END -->
