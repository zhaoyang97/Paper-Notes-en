---
title: >-
  [Paper Note] Adaptive Fidelity Estimation for Quantum Programs with Graph-Guided Noise Awareness
description: >-
  [AAAI 2026][Physics & Scientific Computing][Quantum fidelity estimation] This paper proposes QuFid, a framework that models quantum circuits as directed acyclic graphs (DAGs)…
tags:
  - "AAAI 2026"
  - "Physics & Scientific Computing"
  - "Quantum fidelity estimation"
  - "noise awareness"
  - "graph structure analysis"
  - "adaptive measurement"
  - "spectral complexity"
date: 2026-05-08
content_hash: e4a491ac4c6a9b5f
---

# Adaptive Fidelity Estimation for Quantum Programs with Graph-Guided Noise Awareness

**Conference**: AAAI 2026
**arXiv**: [2601.14713v1](https://arxiv.org/abs/2601.14713v1)  
**Code**: None  
**Area**: Quantum Computing / Quantum Program Testing
**Keywords**: Quantum fidelity estimation, noise awareness, graph structure analysis, adaptive measurement, spectral complexity

## TL;DR

This paper proposes QuFid, a framework that models quantum circuits as directed acyclic graphs (DAGs), characterizes noise propagation via control-flow-aware random walks, quantifies circuit complexity through spectral features of the propagation operator, and achieves adaptive measurement budget allocation — significantly reducing the number of measurement shots while maintaining fidelity accuracy.

## Background & Motivation

### Root Cause

**Background**: In the NISQ (Noisy Intermediate-Scale Quantum) era, quantum devices are constrained by decoherence, gate errors, and limited qubit counts. Fidelity estimation is a critical step for verifying the correctness of quantum programs. However, each fidelity estimation requires a large number of repeated measurements (shots), and the required shot count is difficult to determine in advance — too few leads to inaccurate estimates, while too many wastes scarce quantum resources. Existing methods either use a fixed shot count (heuristic) or rely on pre-trained noise models (e.g., QuCT, QuEst). Since quantum hardware noise is time-varying and device-heterogeneous, pre-trained approaches exhibit poor adaptability in dynamic environments.

### Limitations of Prior Work

**Goal**: How can one adaptively determine the minimum number of measurements needed to achieve a target fidelity accuracy — without relying on pre-trained noise models — based on quantum circuit structure and runtime statistical feedback? Specific challenges include:
1. **Noise heterogeneity**: Noise manifests differently across devices and time points, making static strategies unreliable.
2. **Circuit–hardware mismatch**: Transpilation alters the circuit's dependency structure and noise exposure patterns.
3. **Accuracy–latency trade-off**: A balance must be struck between statistical precision and rapid feedback.

## Method

### Overall Architecture

The QuFid pipeline consists of five steps:
1. Convert the quantum circuit into a DAG.
2. Compute structural deformation metrics induced by transpilation.
3. Construct a control-flow-aware noise propagation operator $P$.
4. Estimate circuit complexity via the spectral features of $P$.
5. Perform adaptive batched measurements based on complexity, with confidence-interval-driven early stopping.

### Key Designs

**1. Quantum Circuit Graph Construction**
- Each quantum gate corresponds to a node; directed edges connect non-commuting gates that share qubits.
- Directed edges encode execution order constraints: gate $v_j$ must execute after gate $v_i$.
- The graph is further converted into a MultiDiGraph to capture topological connectivity.

**2. Structural Deformation Metrics**
- The graphs $\mathcal{G}_0$ and $\mathcal{G}_t$ before and after transpilation are compared, yielding three deformation indicators:
    - $\Delta_{\text{deg}}$: Change in node degree distribution.
    - $\Delta_{\text{path}}$: Expansion of critical paths and long dependency chains.
    - $\Delta_{\text{conn}}$: Effective connectivity expansion.
- These metrics quantify the structural impact of transpilation in a noise-model-agnostic manner.

**3. Control-Flow-Aware Noise Propagation Model**
- A weighted adjacency matrix $A$ is constructed over the transpiled graph $\mathcal{G}_t$, where weights reflect deformation-modulated control-flow dependency strengths.
- The noise propagation operator is defined as $P = D^{-1}A$ (a row-stochastic matrix), where $P_{ij}$ represents the conditional probability that noise from gate $v_i$ affects gate $v_j$.
- Unlike uniform random walks, deformation-aware weights bias propagation toward structurally critical paths (long dependency chains and high fan-in regions).

**4. Spectral Complexity Estimation**
- The eigenvalues $\{\lambda_i\}$ of $P$ are computed, and circuit complexity is defined as $C(\mathcal{G}) = \sum_{i=1}^{k} |\lambda_i|$, taking the top $k$ dominant eigenvalues.
- Large spectral mass → strong long-range dependencies, slow noise decay → more measurements required.
- Small spectral mass → limited noise diffusion → fewer measurements sufficient for convergence.

**5. Adaptive Batch Size and Early Stopping**
- Batch size is set as $\mathcal{P} = C(\mathcal{G}) \cdot \log(\mathcal{D})$, where $\mathcal{D}$ is the transpiled circuit depth.
- Logarithmic scaling prevents over-allocation for deep but weakly coupled circuits.
- Measurements are executed iteratively, with fidelity statistics updated after each batch.

### Loss & Training

No training is involved. QuFid is a **training-free** online framework. The termination condition is based on a confidence interval:

$$\frac{z_\alpha \cdot \sigma}{\sqrt{|T|}} \leq \delta$$

where $\hat{F}$ is the empirical fidelity estimate, $\sigma$ is the standard deviation, $\delta$ is the target error bound, and $\alpha$ is the significance level (default 0.05). Sampling halts when the estimation error does not exceed $\delta$ at the $(1-\alpha)$ confidence level.

## Key Experimental Results

- **Platform**: IBM Quantum hardware (Sherbrooke, Kyiv, Brisbane; 127 qubits each).
- **Benchmarks**: 18 quantum circuit types (BV, QAOA, VQE, QFT, QKNN, QNN, QSVM, etc.), each tested at 4/6/8/10 qubits.
- **Baselines**: Fixed-shot baseline (10,000 shots), QuCT, QuEst.
- **Core Results**:
    - QuFid reduces measurement shots across all settings while keeping fidelity deviation within 0.01.
    - BV (4-qubit) requires only 592 iterations (vs. thousands for the fixed baseline); QKNN (10-qubit) requires 6,992 iterations, reflecting high structural complexity.
    - Compared to QuCT and QuEst, QuFid consumes significantly fewer shots with lower or comparable fidelity deviation.
    - QAOA case study: 4-qubit achieves a deviation of 0.00571 in 606 iterations; 10-qubit achieves 0.01821 in 7,344 iterations.

### Ablation Study

- **Varying deviation thresholds**: Relaxing the tolerance ($0.01 \to 0.02 \to 0.03$) consistently reduces measurement counts, with greater reductions for deeper circuits (e.g., Ising 10-qubit drops from 7,448 to 6,664).
- **Convergence behavior**: Fidelity rises rapidly in early iterations, undergoes brief fluctuation, then converges; deviation drops sharply before stabilizing.
- **Effect of circuit complexity**: Structurally simple circuits (BV, QPE, Simon) exhibit stable measurement requirements; complex circuits such as QKNN show steep growth in measurement demand as qubit count increases.

## Highlights & Insights

1. **Training-free**: QuFid requires no historical data or pre-trained noise models, making it suitable for cold-start scenarios and time-varying hardware environments.
2. **Theory-driven design**: The pipeline from DAG modeling to random walks to spectral analysis carries clear physical and statistical interpretations at each stage, ensuring strong explainability.
3. **Elegance of graph-level abstraction**: Using structural deformation between pre- and post-transpilation graphs as a proxy for noise modeling achieves noise-model-agnostic noise awareness.
4. **Real hardware validation**: Experiments are conducted on actual IBM Quantum backends rather than simulators.

## Limitations & Future Work

1. **Markov assumption**: The noise propagation model adopts a Markov abstraction and cannot fully capture long-range temporal correlations present on certain devices.
2. **Global spectral metric**: Spectral complexity is a global measure; introducing localized or qubit-level deformation metrics could further improve adaptivity.
3. **Single-objective optimization**: The current framework optimizes only the fidelity–shot-count trade-off; jointly optimizing fidelity, latency, and energy consumption is a direction for future work.
4. **Framework portability**: The current implementation is based on Qiskit/IBM; porting to other frameworks (e.g., MindSpore Quantum) has not yet been validated.

## Related Work & Insights

| Method | Requires Training | Noise Model Dependency | Optimization Target | Core Technique |
|--------|------------------|----------------------|--------------------|-|
| QuCT | Yes | Pre-trained data required | Fidelity prediction | Context + topological feature extraction |
| QuEst | Yes | Pre-trained data required | Fidelity prediction | Graph Transformer |
| **QuFid** | **No** | **None** | **Minimum shot count** | **Spectral complexity + adaptive stopping** |

The fundamental distinction between QuFid and QuCT/QuEst lies in their objectives: the latter predict fidelity values, whereas QuFid determines the minimum measurement budget. This makes QuFid more practical under dynamic hardware conditions.

## Related Work & Insights

- **Generality of graph spectral analysis**: The paradigm of modeling problems as DAG + random walk + spectral analysis is transferable to other settings requiring structure-aware resource allocation (e.g., layer importance estimation in neural network pruning).
- **Adaptive testing strategies**: The confidence-interval-driven early stopping mechanism can be directly adapted to software testing, model evaluation, and related domains.
- **Structural deformation metrics**: The idea of comparing graph structures before and after transformation is applicable to compiler optimization evaluation and structural change analysis following model quantization.

## Rating

- Novelty: ⭐⭐⭐⭐ Reframes fidelity estimation as a structure-aware adaptive testing planning problem; the spectral complexity-driven measurement budget allocation is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 18 circuit types × 4 scales × 3 real backends with broad coverage; however, comparisons with additional adaptive sampling methods are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, methodological derivations are rigorous, the physical significance of each module is well explained, and the Discussion section candidly addresses limitations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Data Verification is the Future of Quantum Computing Copilots](data_verification_is_the_future_of_quantum_computing_copilots.md)
- [\[AAAI 2026\] Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study](scientific_knowledge-guided_machine_learning_for_vessel_power_prediction_a_compa.md)
- [\[ICLR 2026\] HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning](../../ICLR2026/physics/hyperkkl_enabling_non-autonomous_state_estimation_through_dynamic_weight_conditi.md)
- [\[AAAI 2026\] Knowledge-Guided Masked Autoencoder with Linear Spectral Mixing and Spectral-Angle-Aware Reconstruction](knowledge-guided_masked_autoencoder_with_linear_spectral_mixing_and_spectral-ang.md)
- [\[ICLR 2026\] Sublinear Time Quantum Algorithm for Attention Approximation](../../ICLR2026/physics/sublinear_time_quantum_algorithm_for_attention_approximation.md)

</div>

<!-- RELATED:END -->
