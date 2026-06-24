---
title: >-
  [Paper Note] Robust Fitting on a Gate Quantum Computer
description: >-
  [ECCV2024][Physics & Scientific Computing][quantum computing] Robust fitting is implemented on a real gate quantum computer (IonQ Aria) for the first time: a quantum circuit is proposed for 1D $\ell_\infty$ feasibility testing, filling a critical gap in using Bernstein-Vazirani (BV) circuits to compute Boolean influence, and demonstrating how to accumulate 1D influence for high-dimensional non-linear models (e.g., fundamental matrix estimation).
tags:
  - "ECCV2024"
  - "Physics & Scientific Computing"
  - "quantum computing"
  - "robust fitting"
  - "consensus maximization"
  - "Boolean influence"
  - "gate quantum computer"
  - "Bernstein-Vazirani circuit"
date: 2026-05-08
content_hash: 53342c0f8459948b
---

# Robust Fitting on a Gate Quantum Computer

**Conference**: ECCV2024  
**arXiv**: [2409.02006](https://arxiv.org/abs/2409.02006)  
**Code**: TBD  
**Area**: Physics  
**Keywords**: quantum computing, robust fitting, consensus maximization, Boolean influence, gate quantum computer, Bernstein-Vazirani circuit

## TL;DR

Robust fitting is implemented on a real gate quantum computer (IonQ Aria) for the first time: a quantum circuit is proposed for 1D $\ell_\infty$ feasibility testing, filling a critical gap in using Bernstein-Vazirani (BV) circuits to compute Boolean influence, and demonstrating how to accumulate 1D influence for high-dimensional non-linear models (e.g., fundamental matrix estimation).

## Background & Motivation

- **Robust fitting** is a core problem in computer vision: estimating geometric models (such as fundamental matrices, homographies, 3D triangulation, etc.) from noisy data containing outliers, typically using the consensus maximization framework.
- This problem has been proven to be NP-hard and hard to approximate; RANSAC and its variants do not provide optimality guarantees.
- Quantum computing provides an alternative path to solving such hard problems:
    - **Adiabatic Quantum Computing (AQC)**: Doan et al. modeled it as hypergraph vertex cover (HVC) to be solved on a D-Wave quantum annealer; Farina et al. used quantum annealing for multi-model fitting.
    - **Gate Quantum Computing (GQC)**: Chin et al. proposed using BV circuits to compute Boolean influence to measure outlierness, theoretically achieving $O(N)$ parallel acceleration.
- **Key Gap**: Chin et al.'s method assumes that the $\ell_\infty$ feasibility test has a quantum implementation, but never provided a concrete circuit design, making verification on real GQC impossible.

## Core Problem

1. How to design a quantum circuit to implement the $\ell_\infty$ feasibility test, thereby making the BV circuit truly runnable?
2. How to extend the Boolean influence computed for 1D point fitting to high-dimensional non-linear robust fitting tasks?

## Method

### Review of Boolean Influence

Given a dataset $\mathcal{D} = \{p_i\}_{i=1}^N$, a binary vector $\mathbf{z} \in \{0,1\}^N$ denotes the subset selection. Define the minimax value:

$$g(\mathbf{z}) = \min_{\mathbf{x} \in \mathcal{M}} \max_{i: z_i=1} r_i(\mathbf{x})$$

$\ell_\infty$ feasibility test: $f(\mathbf{z}) = 0$ if and only if $g(\mathbf{z}) \le \epsilon$, meaning that the residuals of all points in the subset to some model do not exceed the threshold $\epsilon$.

The Boolean influence of the $i$-th data point is defined as the probability that flipping $z_i$ changes the feasibility:

$$\alpha_i = \Pr[f(\mathbf{z}) \neq f(\mathbf{z} \oplus \mathbf{e}_i)]$$

It has been theoretically proven that under certain conditions, the influence of inliers is strictly smaller than that of outliers, making it a reliable measure of outlierness.

### Accelerating Influence Computation via BV Quantum Circuits

- Classical methods need to iterate through $N$ data points for each sample $\mathbf{z}_j$, resulting in a computational complexity of $O(MN)$.
- The BV circuit uses $N+1$ qubits and utilizes quantum superposition and interference to concurrently obtain influence samples for all $N$ data points in a single measurement.
- Running it $M$ times approximates the influence, with the sampling error bound $\Pr(|\hat{\alpha}_i - \alpha_i| < \delta) > 1 - 2e^{-2M\delta^2}$.
- **Bottleneck**: The oracle $U_f$ in the BV circuit needs to implement the $\ell_\infty$ feasibility test, for which no concrete quantum circuit existed prior to this work.

### Contribution: Quantum Circuit for 1D $\ell_\infty$ Feasibility Testing

For the **1D point fitting** problem ($r_i(x) = |x - b_i|$), feasibility simplifies to:

$$f(\mathbf{z}) = 0 \iff \max(\{b_i | z_i=1\}) - \min(\{b_i | z_i=1\}) \le 2\epsilon$$

The circuit design ($U_f$) consists of the following key subcircuits:

| Subcircuit | Function |
|--------|------|
| **A / A⁻¹** | Select the maximum value from the sorted subset (one-hot encoded) |
| **B** | Select the minimum value from the sorted subset |
| **V₁ / V₂** | Encode data values into qubits |
| **S₁** | QFT-based quantum subtractor that computes the difference between the maximum and minimum values |
| **S₂** | Compare with $2\epsilon$ to output the feasibility decision |

Preprocessing: The data is sorted in descending order and shifted so that the minimum value is zero. The entire circuit consumes $3N + 2C + 1$ qubits ($N$ is the number of data points, $C$ is the bit precision of the data). Correct recovery of the ancillary qubits is ensured via $D$ and $D^{-1}$ (uncomputation), satisfying the BV circuit requirement.

### Accumulating Influence: Extending to Fundamental Matrix Estimation

Fundamental matrix estimation is converted into a linear regression problem through linearization (de-homogenization using the 8-point algorithm), and then the influence is accumulated through a RANSAC-style hypothesis sampling framework:

1. Sample $T$ minimum subsets (8 corresponding points each) and estimate the hypothesis fundamental matrix $\mathbf{F}_t$ using the 8-point method.
2. Compute the linearized residuals of all data points under $\mathbf{F}_t$ to serve as input data for 1D point fitting.
3. Apply Alg. 1/2 to compute the 1D influence for each hypothesis.
4. Accumulate the influence of each hypothesis via logarithmic averaging.

Model estimation stage: Uniformly sample thresholds $\gamma_h$ over normalized influence, perform least-squares fitting on points with influence $\le \gamma_h$, and return the model with the largest consensus set.

## Key Experimental Results

### Quantum Simulator Verification (Amazon Braket SV1)

- Data $\mathcal{D} = \{7,5,3,2\}$, $\epsilon=1$, 20 qubits.
- As the number of iterations $M$ increases, the approximate influence converges to the true values $\{0.50, 0.25, 0.25, 0.50\}$.

### Real Quantum Hardware Verification (IonQ Aria, 25 Qubits)

| Number of Data Points $N$ | Bit Precision $C$ | Number of Qubits | Aria Results | SV1 Results | Ground Truth |
|:---:|:---:|:---:|---:|---:|---:|
| 2 | 1 | 9 | 0.50, 0.56 | 0.51, 0.49 | 0.50, 0.50 |
| 3 | 3 | 16 | 0.49, 0.23, 0.53 | 0.50, 0.26, 0.52 | 0.50, 0.25, 0.50 |
| 4 | 3 | 19 | 0.48, 0.39, 0.41, 0.53 | 0.50, 0.25, 0.25, 0.49 | 0.50, 0.25, 0.25, 0.50 |

Small-scale circuits yield good results on real hardware; as the circuit size increases (approaching the 25-qubit limit), errors increase, primarily due to insufficient iterations ($M=50$) and the lack of error correction.

### Fundamental Matrix Estimation (Influence Accumulation)

ROC AUC classification scores on four datasets (TUM, KITTI, T&T, CPC):

| Dataset | RANSAC | Ours (Influence Alg.3) | Chin et al. |
|--------|:------:|:-----:|:-----:|
| TUM | 0.71 | **0.83** | 0.76 |
| KITTI | 0.86 | **0.94** | 0.86 |
| T&T | 0.72 | **0.82** | 0.82 |
| CPC | 0.75 | **0.87** | 0.85 |

Recalls (%) at SGD $\le 0.05$:

| Dataset | Ours | RANSAC | USAC-openCV |
|--------|:-------:|:------:|:-----------:|
| TUM | **62.8** | 60.9 | 54.0 |
| KITTI | 85.4 | 87.0 | 83.8 |
| T&T | **85.1** | 77.6 | 88.1 |
| CPC | **50.3** | 42.0 | 52.5 |

## Highlights & Insights

- **First Demonstration on Real GQC**: Achieving quantum robust fitting on IonQ Aria for the first time, turning theoretical potential into physical feasibility.
- **Filling a Key Gap**: Proposing a quantum circuit design for the $\ell_\infty$ feasibility test, resolving the fundamental issue of the BV circuit lacking an oracle implementation.
- **Bridge from 1D to High Dimensions**: Extending simple 1D influence to complex high-dimensional estimation tasks through RANSAC-style hypothesis sampling and logarithmic average accumulation.
- **Thorough Experimental Validation**: Correctness confirmed on simulators, feasibility confirmed on real hardware, and practicality confirmed on benchmark datasets.

## Limitations & Future Work

- The current quantum circuit is only applicable to **1D point fitting**; high-dimensional problems still require assistance from classical methods.
- IonQ Aria has only 25 qubits, which can only process extremely small-scale data ($N \le 4$), remaining far from practical utility.
- Operating in the current NISQ era, quantum noise is significant, and errors grow noticeably as the circuit scales.
- Quantum circuit resource consumption is $3N + 2C + 1$ qubits, growing linearly with the size of the data.
- Compared to mature RANSAC-like methods, the quantum approach currently offers no performance advantage, with its value lying in exploring alternative technical paths.
- Influence accumulation relies on RANSAC-style sampling ($T=1000$ hypotheses), which still incurs high classical computational overhead.

## Related Work & Insights

| Method | Quantum Paradigm | Problem Modeling | Hardware Verification | Applicable Scale |
|------|---------|---------|---------|---------|
| Chin et al. (2020) | GQC (BV circuit) | Boolean influence | Theory only | Theoretically scalable |
| Doan et al. (2022) | AQC (D-Wave) | Hypergraph vertex cover + QUBO | Run on D-Wave | Limited by QUBO variables |
| Farina et al. (2023) | AQC | Disjoint set cover + QUBO | Run on D-Wave | Limited by QUBO variables |
| **Ours** | **GQC (BV + new $U_f$)** | **Boolean influence** | **IonQ Aria** | 1D $N \le 4$, can accumulate to high-dimensional |

Core difference from AQC methods: GQC can implement arbitrary quantum algorithms (such as Shor's, Grover's) and is theoretically more versatile, but current hardware constraints are more stringent.

## Insights & Connections

- The hybrid quantum-classical framework is a pragmatic route in the NISQ era: utilizing quantum computers to accelerate subproblems while classical methods handle global optimization.
- The influence metric itself possesses independent value and can be decoupled from the quantum computing framework to serve as an outlier detection tool within classical methods.
- The accumulation strategy from 1D to high dimensions (hypothesis sampling + logarithmic averaging) could potentially be extended to other robust estimation tasks.
- If IBM's quantum roadmap (200-qubit fully error-corrected system by 2029) is realized, this method can scale to practical sizes.

## Rating
- Novelty: 8/10 — First to provide a quantum circuit for $\ell_\infty$ feasibility testing and validate it on real GQC, filling an important theoretical gap.
- Experimental Thoroughness: 7/10 — Comprehensive validation across simulators, real hardware, and benchmarks, though constrained to minimal scales by hardware limitations.
- Writing Quality: 8/10 — Clear problem definition, rigorous theoretical derivation, and a reasonable comparison between quantum and classical approaches.
- Value: 6/10 — Current practical value is limited, but it serves as a forward-looking exploration of quantum computing in vision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Robust and Interpretable Adaptation of Equivariant Materials Foundation Models via Sparsity-promoting Fine-tuning](../../ICLR2026/physics/robust_and_interpretable_adaptation_of_equivariant_materials_foundation_models_v.md)
- [\[ICLR 2026\] Sublinear Time Quantum Algorithm for Attention Approximation](../../ICLR2026/physics/sublinear_time_quantum_algorithm_for_attention_approximation.md)
- [\[NeurIPS 2025\] Quantum Doubly Stochastic Transformers](../../NeurIPS2025/physics/quantum_doubly_stochastic_transformers.md)
- [\[ICLR 2026\] Accelerating Inference for Multilayer Neural Networks with Quantum Computers](../../ICLR2026/physics/accelerating_inference_for_multilayer_neural_networks_with_quantum_computers.md)
- [\[AAAI 2026\] Data Verification is the Future of Quantum Computing Copilots](../../AAAI2026/physics/data_verification_is_the_future_of_quantum_computing_copilots.md)

</div>

<!-- RELATED:END -->
