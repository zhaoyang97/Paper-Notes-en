---
title: >-
  [Paper Note] ResQ: A Novel Framework to Implement Residual Neural Networks on Analog Rydberg Atom Quantum Computers
description: >-
  [ICCV 2025][Physics & Scientific Computing][Quantum Machine Learning] This paper proposes ResQ — the first framework to natively implement residual neural networks (ResNets) on analog Rydberg atom quantum computers by ex…
tags:
  - "ICCV 2025"
  - "Physics & Scientific Computing"
  - "Quantum Machine Learning"
  - "Residual Networks"
  - "Rydberg Atoms"
  - "Analog Quantum Computing"
  - "Neural ODE"
date: 2026-05-08
content_hash: 7c7259584fe73f03
---

# ResQ: A Novel Framework to Implement Residual Neural Networks on Analog Rydberg Atom Quantum Computers

**Conference**: ICCV 2025
**arXiv**: [2506.21537](https://arxiv.org/abs/2506.21537)  
**Code**: [github.com/positivetechnologylab/ResQ](https://github.com/positivetechnologylab/ResQ)  
**Area**: Quantum Computing / Physics-Inspired Computing
**Keywords**: Quantum Machine Learning, Residual Networks, Rydberg Atoms, Analog Quantum Computing, Neural ODE

## TL;DR

This paper proposes ResQ — the first framework to natively implement residual neural networks (ResNets) on analog Rydberg atom quantum computers by exploiting continuous-time Hamiltonian evolution, encoding input features and trainable parameters via piecewise parameterized laser pulses, achieving an average 50% improvement over classical models of equivalent scale on MNIST, FashionMNIST, and medical dataset classification tasks.

## Background & Motivation

### Core Problem

Quantum machine learning (QML) aims to accelerate or enhance machine learning models using quantum computing. Current QML approaches are predominantly based on gate-based digital quantum systems, which face a fundamental limitation when implementing ResNets — quantum gates permit only unitary linear transformations and cannot directly realize the additive skip connection $F(x)+x$.

### Limitations of Prior Work

**Limitations of gate-based approaches**: Digital quantum circuits consist of discrete operation sequences, making it difficult to express models with continuous dynamics such as ResNets and Neural ODEs.

**Hybrid approaches are not fully quantum**: The hybrid quantum-classical model of Liang et al. still relies on classical computation to realize residual connections.

**Continuous-variable QNNs**: Killoran et al. implement ResNets using a continuous-variable basis but do not exploit the unique capabilities of Rydberg atom systems.

**Existing Rydberg atom work**: The binary MNIST classification of Lu et al. employs a digital-analog hybrid scheme incompatible with commercially available devices that support only analog functionality.

### Root Cause

ResNets can be expressed via Neural ODEs as continuous-time differential equations $\frac{dx}{dt} = F(x, \theta)$. Analog Rydberg atom quantum computers naturally evolve in continuous time through the Schrödinger equation $i\hbar\frac{d}{dt}|\Psi(t)\rangle = H(t)|\Psi(t)\rangle$. The two share a **natural mathematical correspondence** that gate-based quantum systems inherently lack.

## Method

### Overall Architecture

The ResQ classification workflow:
1. Dimensionality reduction via PCA preprocessing
2. Scaling PCA features to the physical hardware constraint range
3. Encoding inputs and parameters into the global and local control parameters of the Rydberg Hamiltonian
4. Executing Hamiltonian evolution
5. Measuring all qubits and averaging $|1\rangle$-state probabilities as the prediction output
6. Gradient-based training using cross-entropy loss

### Key Designs

#### 1. **Rydberg Atom Hamiltonian Parameterization**
- **Function**: Encodes ResNet parameters and data into the physical control variables of the quantum system.
- **Mechanism**: The global Rydberg Hamiltonian is:

$$H(t) = \frac{\Omega(t)}{2}\sum_i(e^{i\phi(t)}|g\rangle_i\langle r|_i + h.c.) - \Delta(t)\sum_i \hat{n}_i + \sum_{i<j}\frac{C_6}{|\vec{p}_i - \vec{p}_j|^6}\hat{n}_i\hat{n}_j$$

supplemented by a local detuning term: $H_{local}(t) = -\delta(t)\sum_i h_i \hat{n}_i$

where $\Omega(t)$ (Rabi frequency), $\Delta(t)$ (detuning), and $\delta(t)$ (local detuning) are parameterized as time sequences, and $h_i \in [0,1]$ are per-site coupling coefficients.

- **Design Motivation**: These physical parameters naturally correspond to the learnable weights of a ResNet, with temporal evolution corresponding to network layers.

#### 2. **Piecewise Pulse Parameterization**
- **Function**: Divides each laser parameter into multiple piecewise linear pulse intervals.
- **Mechanism**: The amplitude in pulse interval $i$ is parameterized as $\theta_j \omega_i + \theta_{j+1}$, where $\theta_j, \theta_{j+1}$ are trainable parameters and $\omega_i$ is an input feature.
    - Hold time 0.15 μs + transition time 0.05 μs = 0.2 μs per interval
    - Maximum runtime 4.0 μs → up to 19 intervals
    - Three global parameters × 2 parameters per interval each = $6M$ trainable parameters
- **Design Motivation**: The linear combination allows each input feature to exert influence throughout the entire pulse duration while maintaining a trainable offset.

#### 3. **Local Detuning Feature Encoding**
- **Function**: Leverages per-site coupling coefficients $h_i$ for feature encoding and parameterization.
- **Mechanism**: $h_i$ values at even-indexed qubits encode input features; $h_i$ values at odd-indexed qubits serve as trainable parameters.
    - $N$ qubits → $N/2$ additional inputs + $N/2$ additional parameters
    - Total: $3 + N/2$ input features, $6M + N/2$ trainable parameters
- **Design Motivation**: The number of features scales **linearly** with the number of qubits without requiring increased circuit depth, which is an important scalability advantage.

### Loss & Training

- **Loss function**: Binary cross-entropy loss
- **Gradient computation**: Stochastic Pulse Gradient method — an unbiased gradient estimator for analog quantum programs
    - Repeatedly executes simulated trajectories with randomly inserted qubit rotations
    - 20 samples per gradient evaluation
- **Measurement**: Probability distributions estimated from 1000 repeated measurements
- **Optimizer**: Adam
- Configuration: $N=4$ atoms, 5 PCA features
- Deliberate feature assignment: high-variance PCA features assigned to global parameters ($\Omega, \Delta$); low-variance features assigned to local terms

## Key Experimental Results

### Main Results (vs. Classical Models of Equivalent Parameter Count)

| Task | ResQ | C-NN (1×) | C-ResNet (1×) | C-NODE (1×) | C-NN (100×) |
|------|------|-----------|---------------|-------------|-------------|
| PID Diabetes | **Best** | ↓52% | ↓57% | ↓36% | Comparable |
| MNIST 0v1 (Easy) | High accuracy | ↓Significant | ↓Significant | ↓Significant | Comparable |
| MNIST 4v9 (Hard) | Competitive | ↓Significant | ↓Significant | ↓ | Slightly lower |
| FashionMNIST (Easy) | Competitive | ↓37% | ↓ | ↓ | Slightly lower |
| FashionMNIST (Hard) | Competitive | ↓ | ↓ | ↓ | Comparable |

Overall: ResQ achieves an **average 50% improvement** over classical models of equivalent parameter count (C-NN: 56%, C-ResNet: 57%, C-NODE: 36%).

### Ablation Study on Atom Configurations

| Configuration | Chain | Ring/Square | Triangle | Optimal Spacing |
|------|------|---------|--------|---------|
| Performance trend | Competitive | Slightly better | Competitive | 12 μm (moderate spacing) |
| Interaction strength | — | — | — | Weaker interaction preferred |

### Ablation Study on Pulse Intervals

| No. of intervals | 1 | 3 | 5 | 7+ |
|--------|---|---|---|-----|
| PID accuracy | Baseline | Slightly better | On par | On par |
| Note | 3 intervals suffice; diminishing returns with more |

### Hardware Noise Robustness

- Inference executed on the QuEra Aquila 256-qubit real quantum computer
- Accuracy and F1 scores remain **within 1%** of ideal simulation results
- Predictions may flip due to hardware noise only for samples near the decision boundary (0.5)

### Key Findings

1. **ResQ significantly outperforms classical models in parameter efficiency**: average 50% improvement at equivalent parameter count, comparable to classical models with 10–100× more parameters.
2. **Weak interactions (large spacing) are more suitable for classification**: strong quantum entanglement impedes training.
3. **Intrinsic noise robustness**: ResQ's design confers inherent resistance to noise from analog quantum hardware.
4. **Constant execution time**: The Neural ODE formulation gives ResNet a runtime independent of problem scale, which does not hold for classical or digital quantum architectures.
5. **Linear feature scaling**: The number of input features grows linearly with qubit count without increasing circuit depth.

## Highlights & Insights

1. **A profound physics–computation correspondence**: The mapping chain ResNet → Neural ODE → Schrödinger equation → Rydberg atom evolution is both elegant and insightful.
2. **Validation on real quantum hardware**: Beyond simulation, actual inference is performed on the QuEra Aquila 256-qubit device.
3. **Quantum origin of parameter efficiency**: $N$ qubits can represent a $2^N$-dimensional state space, providing exponential expressive capacity.
4. **Practically motivated engineering choices**: Non-zero lower bounds enforce dynamic existence; PCA features are assigned to global/local parameters according to their importance.

## Limitations & Future Work

1. **Binary classification only**: The current framework relies on a single probability threshold; extension to multi-class classification is required.
2. **Only 4 atoms used**: Simulating more atoms is computationally prohibitive, limiting the practical scale of computation.
3. **Training is performed on a classical simulator**: Real quantum hardware is used only for inference; quantum-native training remains a challenge.
4. **PCA dimensionality reduction constraint**: 5 PCA features may discard important information, limiting performance on complex datasets.
5. **Experimental support for local detuning**: QuEra Aquila currently supports local detuning only on an experimental basis.
6. **Weak connection to the vision domain of ICCV**: Despite the use of visual datasets such as MNIST and FashionMNIST.

## Related Work & Insights

- Neural ODE (Chen et al., NeurIPS 2018) established the theoretical connection between ResNets and ODEs.
- Rydberg atom systems are currently the only quantum hardware supporting continuous-time Hamiltonian evolution with both local and global control.
- The Stochastic Pulse Gradient method provides a theoretical foundation for training analog quantum programs.
- The exponential state space of quantum computing may offer a new computational paradigm for large-scale vision tasks in the future.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First fully quantum implementation of ResNet on an analog Rydberg atom system, with deep physical insight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Includes real quantum hardware validation and multiple datasets, but limited to 4 atoms and binary classification.
- **Writing Quality**: ⭐⭐⭐⭐ — Physical background and method design are clearly explained, accessible to cross-disciplinary readers.
- **Value**: ⭐⭐⭐ — Strong academic novelty, but practical applicability and scalability are limited, with a weak connection to the vision domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DeltaPhi: Physical States Residual Learning for Neural Operators in Data-Limited PDE Solving](../../NeurIPS2025/physics/deltaphi_physical_states_residual_learning_for_neural_operators_in_data-limited_.md)
- [\[NeurIPS 2025\] From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics](../../NeurIPS2025/physics/from_black_hole_to_galaxy_neural_operator_framework_for_accretion_and_feedback_d.md)
- [\[NeurIPS 2025\] Exoplanet Formation Inference Using Conditional Invertible Neural Networks](../../NeurIPS2025/physics/exoplanet_formation_inference_using_conditional_invertible_neural_networks.md)
- [\[NeurIPS 2025\] Stable Minima of ReLU Neural Networks Suffer from the Curse of Dimensionality: The Neural Shattering Phenomenon](../../NeurIPS2025/physics/stable_minima_of_relu_neural_networks_suffer_from_the_curse_of_dimensionality_th.md)
- [\[NeurIPS 2025\] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](../../NeurIPS2025/physics/physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)

</div>

<!-- RELATED:END -->
