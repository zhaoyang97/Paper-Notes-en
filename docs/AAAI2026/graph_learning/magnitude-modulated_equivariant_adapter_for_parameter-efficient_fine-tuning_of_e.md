---
title: >-
  [Paper Note] Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks
description: >-
  [AAAI 2026][Graph Learning][Equivariant Graph Neural Networks] This paper proposes MMEA (Magnitude-Modulated Equivariant Adapter), a lightweight parameter-efficient fine-tuning method for spherical-harmonic-based equivariant GNNs. By employing scalar gating to independently modulate feature magnitudes along "order–multiplicity" channels, MMEA achieves state-of-the-art molecular potential energy prediction accuracy—surpassing both ELoRA and full fine-tuning—while strictly pres…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "Equivariant Graph Neural Networks"
  - "Parameter-Efficient Fine-Tuning"
  - "Molecular Potential Energy Prediction"
  - "Spherical Harmonics"
  - "PEFT"
date: 2026-05-08
content_hash: 336b59d98b431abc
---

# Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks

**Conference**: AAAI 2026
**arXiv**: [2511.06696](https://arxiv.org/abs/2511.06696)  
**Code**: [https://github.com/CLaSLoVe/MMEA](https://github.com/CLaSLoVe/MMEA)  
**Area**: Graph Learning
**Keywords**: Equivariant Graph Neural Networks, Parameter-Efficient Fine-Tuning, Molecular Potential Energy Prediction, Spherical Harmonics, PEFT

## TL;DR

This paper proposes MMEA (Magnitude-Modulated Equivariant Adapter), a lightweight parameter-efficient fine-tuning method for spherical-harmonic-based equivariant GNNs. By employing scalar gating to independently modulate feature magnitudes along "order–multiplicity" channels, MMEA achieves state-of-the-art molecular potential energy prediction accuracy—surpassing both ELoRA and full fine-tuning—while strictly preserving equivariance and using fewer trainable parameters.

## Background & Motivation

### Equivariant GNNs and Molecular Potential Energy Prediction

Density Functional Theory (DFT) is the standard computational approach in chemistry and materials science, but its cubic computational complexity limits large-scale simulation. Deep learning molecular potential energy models (e.g., MACE, NequIP, Equiformer) accelerate simulation by learning interatomic potentials while maintaining quantum mechanical accuracy.

Among these, **spherical-harmonic-based equivariant GNNs** are particularly powerful:
- They intrinsically respect rotational, translational, and permutation symmetries.
- They can model high-order physical information (not limited to scalars, but also vectors and higher-order tensors).
- They are highly sample-efficient: chemical accuracy can be achieved with only hundreds to thousands of local structures.

### Necessity and Challenges of Fine-Tuning

When the target system is underrepresented in pre-training data (e.g., rare chemical configurations), the accuracy of pre-trained models degrades. Fine-tuning can recover accuracy on small-scale task-specific data, but:

**Risks of full fine-tuning**: overfitting and catastrophic forgetting.

**Fatal flaw of conventional PEFT (LoRA, Adapter)**: These methods mix irreducible representations of different tensor orders, **breaking equivariance** and causing the model to lose its symmetry guarantees.

### Limitations of ELoRA

ELoRA is the first equivariant PEFT method, achieving equivariant fine-tuning by introducing path-dependent low-rank adapters into each tensor channel. However, ELoRA retains relatively high degrees of freedom within each tensor order—it permits mixing between multiplicity channels of different degrees.

**Core insight**: In a well-trained equivariant GNN, the multiplicity channels at each order already form a robust basis. Allowing them to mix freely during fine-tuning may distort the geometric structure of the pre-trained feature space.

### Physical Intuition Behind MMEA

Since the pre-trained model has already learned a good basis representation for each order, **modulating only the magnitude of each channel—rather than mixing channels—should suffice for adapting to new chemical environments**. This is analogous to adjusting the volume of each frequency band on a radio, rather than rearranging the frequencies themselves.

## Method

### Overall Architecture

MMEA inserts a lightweight gating module after equivariant linear layers. It modulates the magnitude of each "order × multiplicity" channel via scalar gains, strictly avoiding any mixing between different multiplicity channels.

### Key Designs

#### 1. **Node Feature Space Decomposition**: Understanding Equivariant Representation Structure

The node feature space is a direct sum of irreducible representations:
$$\mathcal{H} := \bigoplus_{\ell=0}^{L} \mathcal{H}^{(\ell)}, \quad \mathcal{H}^{(\ell)} := V^{(\ell)} \otimes \mathbb{R}^{1 \times m_\ell}$$

where $V^{(\ell)}$ is the irreducible representation of order $\ell$ (dimension $d_\ell = 2\ell+1$) and $m_\ell$ is the multiplicity. The SO(3) group action $g$ acts only on $V^{(\ell)}$:
$$g \cdot (v \otimes a) := (\rho^{(\ell)}(g)v) \otimes a$$

**Key observation**: The group action only rotates within $V^{(\ell)}$; the multiplicity space $\mathbb{R}^{1 \times m_\ell}$ is invariant. Therefore, scalar scaling along the multiplicity dimension does not break equivariance.

#### 2. **Lightweight Gating Network**: Generating Modulation Gains from Scalar Channels

**Function**: Takes only the $\ell=0$ (scalar) features $\mathbf{h}^{(0)}$ as input and generates scalar gains for all orders and all multiplicities via a two-layer MLP.

**Bottleneck projection**:
$$\mathbf{z} = \text{SiLU}(W_\downarrow \mathbf{h}^{(0)} + \mathbf{b}_\downarrow), \quad W_\downarrow \in \mathbb{R}^{r \times m_0}$$

**Expansion**:
$$[\boldsymbol{\gamma}^{(0)}, \boldsymbol{\gamma}^{(1)}, \ldots, \boldsymbol{\gamma}^{(L)}] = W_\uparrow \mathbf{z} + \mathbf{b}_\uparrow, \quad W_\uparrow \in \mathbb{R}^{(\sum_\ell m_\ell) \times r}$$

Each $\boldsymbol{\gamma}^{(\ell)} \in \mathbb{R}^{m_\ell}$ assigns a scalar gain to each multiplicity channel at order $\ell$. Here $r$ denotes the bottleneck dimension.

**Design Motivation**: Only scalar channels are used as input to preserve parameter efficiency and equivariance (scalars are invariant under SO(3) transformations).

#### 3. **Equivariant Modulation**: Channel-wise Scaling of Feature Magnitudes

For the scalar order ($\ell = 0$): additive modulation
$$\mathcal{A}_\Gamma^{(0)}(\mathbf{h}^{(0)}) = \mathbf{h}^{(0)} + \boldsymbol{\gamma}^{(0)}$$

For higher orders ($\ell \geq 1$): multiplicative scaling
$$\mathcal{A}_\Gamma^{(\ell)}(\mathbf{h}^{(\ell)}) = \sum_{k=1}^{m_\ell} \phi(\gamma_k^{(\ell)}) \mathbf{v}_k^{(\ell)} \otimes \mathbf{e}_k^{(\ell)}$$

where $\phi(x) = 1+x$ (residual scaling) or $\phi(x) = e^x$ (positive scaling).

The final modulated feature:
$$\mathbf{h}' := \mathcal{A}_\Gamma(\mathbf{h}) = \bigoplus_{\ell=0}^{L} \mathcal{A}_\Gamma^{(\ell)}(\mathbf{h}^{(\ell)})$$

**Core distinction between MMEA and ELoRA**:
- ELoRA permits low-rank mixing between different multiplicity channels within each order.
- MMEA applies only independent scalar scaling to each channel, with no mixing of different multiplicities whatsoever.

### Equivariance Proof (Summary)

The paper provides a rigorous mathematical proof that $\mathcal{A}_\Gamma(g \cdot \mathbf{h}) = g \cdot \mathcal{A}_\Gamma(\mathbf{h})$:
1. **Gain invariance**: $\boldsymbol{\gamma}^{(\ell)}$ is derived solely from scalar features $\mathbf{h}^{(0)}$, which are invariant under SO(3).
2. **Modulation equivariance**: Scalar scaling commutes with Wigner-D matrix rotations, because scaling acts on the multiplicity space (an invariant space) while rotation acts on the representation space.

### Loss & Training

- Weighted energy–force joint loss (ef loss)
- Energy weight: 1; force weight: 1000
- Adam optimizer with learning rate 0.005
- EMA decay 0.995, gradient clipping 100
- Same pre-trained model (MACE-OFF), datasets, and hyperparameters as ELoRA

## Key Experimental Results

### Main Results

#### rMD17 Dataset (10 organic molecules, 50 training samples)

| Molecule | Metric | Full | ELoRA | **MMEA** | MMEA vs. Full |
|----------|--------|------|-------|---------|---------------|
| Aspirin | E/F | 9.7/23.9 | 8.0/18.3 | **7.3/16.4** | ↓25%/↓31% |
| Azobenzene | E/F | 4.6/14.8 | 4.0/12.6 | **3.9/11.9** | ↓15%/↓20% |
| Benzene | E/F | 0.3/2.4 | 0.2/1.6 | **0.2/1.4** | ↓33%/↓42% |
| Naphthalene | E/F | 1.8/8.1 | 1.4/6.0 | **1.2/5.7** | ↓33%/↓30% |
| Paracetamol | E/F | 6.5/20.3 | 4.7/14.5 | **4.3/13.3** | ↓34%/↓34% |
| Salicylic | E/F | 4.3/17.2 | 3.2/13.3 | **2.9/12.0** | ↓33%/↓30% |
| Toluene | E/F | 1.8/8.8 | 1.4/6.2 | **1.2/5.4** | ↓33%/↓39% |
| Uracil | E/F | 2.9/15.8 | 2.1/12.3 | **2.0/10.7** | ↓31%/↓32% |

MMEA outperforms both Full and ELoRA across all 10 molecules. Compared to ELoRA, MMEA achieves an additional average improvement of ~6.6% in energy MAE and ~8.7% in force MAE.

#### 3BPA Dataset (300K training, multi-temperature generalization)

| Condition | Metric | Scratch | Full | ELoRA | **MMEA** |
|-----------|--------|---------|------|-------|---------|
| 300K | E/F | 3.0/8.8 | 3.3/7.8 | 3.0/7.5 | **2.7/7.5** |
| 600K | E/F | 9.7/21.8 | 7.3/16.6 | 6.5/15.5 | **6.5/15.4** |
| 1200K | E/F | 29.8/62.0 | 20.3/48.7 | 17.6/42.0 | **17.1/39.7** |
| Dihedral | E/F | 7.8/16.5 | 7.3/12.3 | 5.9/11.4 | **5.6/10.6** |

MMEA demonstrates superior generalization under high-temperature (out-of-distribution) conditions; the 1200K force error decreases from 42.0 (ELoRA) to 39.7, with notable improvement on the dihedral slice as well.

#### AcAc Dataset

| Condition | Metric | Scratch | Full | ELoRA | MMEA(r=16) | MMEA(r=32) |
|-----------|--------|---------|------|-------|------------|------------|
| 300K | E/F | 0.9/5.1 | 1.0/5.1 | 0.8/4.5 | **0.7/4.4** | **0.7/4.2** |
| 600K | E/F | 4.6/22.4 | 5.8/16.4 | 3.9/13.6 | 3.6/13.2 | **3.2/13.0** |

### Ablation Study

#### Ablation on rMD17-Aspirin

| Configuration | Energy MAE | Forces MAE | Note |
|---------------|-----------|------------|------|
| **MMEA (full)** | **7.3** | **16.4** | Best |
| Full fine-tuning | 9.7 | 23.9 | Baseline |
| w/o nonlinear activation | 7.6 | 16.4 | SiLU contributes marginally |
| w/o input-head reuse | 9.2 | 16.7 | Separate handling of scalar/higher-order is harmful |
| w/o scalar modulation | 12.9 | 30.5 | **Scalar channel modulation is critical** |
| w/o higher-order modulation | 8.3 | 16.6 | Higher-order modulation is valuable |
| Shared higher-order modulation | 7.6 | 16.6 | Independent modulation outperforms shared |
| Readout only | 23.8 | 36.8 | Tuning only 0.3% of parameters is insufficient |
| Adapter (conventional) | 11.0 | 26.3 | **Breaking equivariance degrades performance** |

**Most critical finding**: Removing scalar modulation causes catastrophic performance degradation (Energy MAE from 7.3 to 12.9), confirming that the scalar channel is the core of the modulation mechanism. The conventional Adapter performs even worse than full fine-tuning due to its violation of equivariance.

### Parameter Efficiency

| Method | Rank $r$ | Trainable Parameters | % of Full |
|--------|----------|----------------------|-----------|
| Full | / | 751,896 | 100% |
| ELoRA | 16 | 175,880 | 23.4% |
| **MMEA** | **16** | **151,354** | **20.1%** |
| MMEA | 32 | 201,258 | 26.7% |

MMEA (r=16) uses only 20.1% of the parameters of full fine-tuning—approximately 85% of ELoRA's parameter count—while achieving superior performance.

### Key Findings

1. In equivariant GNNs, channel-wise magnitude modulation is sufficient for adapting to new chemical environments without mixing channels.
2. Reducing degrees of freedom actually improves performance: MMEA uses fewer parameters than ELoRA yet achieves better results, primarily because it better preserves pre-trained knowledge.
3. Modulation of scalar channels ($\ell=0$) is the most critical component.
4. Convergence speed is substantially improved: MMEA reaches the loss level attained by ELoRA at ~epoch 200 by epoch 58.
5. When the target system deviates significantly from the pre-training distribution (e.g., using an inorganic pre-trained model to predict organic molecules), MMEA underperforms ELoRA and full fine-tuning.

## Highlights & Insights

1. **Profound "less is more" insight**: In the PEFT domain, MMEA demonstrates that fewer degrees of freedom (scalar scaling only) can yield better generalization by preserving the geometric structure of the pre-trained model.
2. **Physics-driven design**: Starting from the observation that "the equivariant representation space has already learned a good basis," the method adjusts magnitudes without altering directions.
3. **Rigorous equivariance proof**: Unlike empirically motivated engineering designs, MMEA offers complete mathematical guarantees.
4. **Practicality**: Integrated into the widely used e3nn framework for direct community adoption.
5. **Training efficiency**: Convergence is approximately 3–4× faster than ELoRA.

## Limitations & Future Work

1. **Limited out-of-distribution generalization**: When the target distribution diverges significantly from pre-training data, MMEA underperforms ELoRA, as reduced degrees of freedom become a disadvantage in this setting.
2. **No weight merging**: Unlike ELoRA, which can merge learned weights back into the backbone, MMEA incurs approximately 2.1% additional inference latency.
3. **Gating for fully connected tensor products**: How to effectively design gating for tensor products with multiple inputs remains an open problem.
4. **Absolute convergence speed**: Although faster than ELoRA, the absolute convergence speed still leaves room for improvement.
5. Adaptive rank selection strategies (using different ranks for different layers or orders) could be explored in future work.

## Related Work & Insights

- **ELoRA**: The first equivariant PEFT method and the direct predecessor of MMEA.
- **FiLM (Feature-wise Linear Modulation)**: MMEA's scalar gating shares conceptual similarity with FiLM, but operates under specific constraints imposed by the equivariant framework.
- **MACE**: The backbone model, an equivariant GNN with many-body interactions.
- **LoRA**: A seminal PEFT work, not directly applicable to equivariant networks.
- **BitFit**: An extremely minimal PEFT approach that updates only biases, sharing a philosophical alignment with MMEA's "minimal modulation" principle.

## Rating

- Novelty: ⭐⭐⭐⭐ — Original insight (preserving the multiplicity basis and modulating only magnitudes), though the method itself is relatively simple.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three standard datasets, 10 molecules, multi-temperature evaluation, detailed ablation and parameter analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear physical motivation, rigorous equivariance proofs, and transparent experimental setup.
- Value: ⭐⭐⭐⭐ — Directly applicable to the molecular simulation community, though the scope is limited to in-distribution fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](../../NeurIPS2025/graph_learning/smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)
- [\[ICLR 2026\] AdS-GNN - a Conformally Equivariant Graph Neural Network](../../ICLR2026/graph_learning/ads-gnn_-_a_conformally_equivariant_graph_neural_network.md)
- [\[ICLR 2026\] FS-KAN: Permutation Equivariant Kolmogorov-Arnold Networks via Function Sharing](../../ICLR2026/graph_learning/fs-kan_permutation_equivariant_kolmogorov-arnold_networks_via_function_sharing.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](adaptive_riemannian_graph_neural_networks.md)
- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)

</div>

<!-- RELATED:END -->
