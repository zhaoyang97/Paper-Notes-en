---
title: >-
  [Paper Note] ParalESN: Enabling Parallel Information Processing in Reservoir Computing
description: >-
  [ICML2026][Echo State Network] ParalESN injects LRU-style complex diagonal linear recurrences into the "untrained reservoir" of an Echo State Network (ESN). This allows traditional RC sequences to be parallelized over ti…
tags:
  - "ICML2026"
  - "Echo State Network"
  - "Linear Recurrence"
  - "Parallel Scan"
  - "Diagonal Complex Matrix"
  - "Fading Memory"
date: 2026-05-08
content_hash: cd1fafdf4fc55f66
---

# ParalESN: Enabling Parallel Information Processing in Reservoir Computing

**Conference**: ICML2026  
**arXiv**: [2601.22296](https://arxiv.org/abs/2601.22296)  
**Code**: https://github.com/nennomp/paralesn (Available)  
**Area**: Sequence Modeling / Reservoir Computing / State Space Models  
**Keywords**: Echo State Network, Linear Recurrence, Parallel Scan, Diagonal Complex Matrix, Fading Memory

## TL;DR
ParalESN injects LRU-style complex diagonal linear recurrences into the "untrained reservoir" of an Echo State Network (ESN). This allows traditional RC sequences to be parallelized over time and scaled to $10^5$ dimensions, while strictly maintaining the Echo State Property (ESP) and the universal approximation properties of fading memory filters.

## Background & Motivation

**Background**: Reservoir Computing (RC) functions by freezing a high-dimensional, random, nonlinear recurrent system and training only a linear readout, thereby circumventing the vanishing/exploding gradient problems inherent in RNN training. Its primary representative, the Echo State Network (ESN), relies on a state transition matrix $W_h$ initialized with a spectral radius below 1 to trigger the Echo State Property (ESP), ensuring the state is determined solely by the input.

**Limitations of Prior Work**: Traditional RC faces two critical bottlenecks. The first is **sequentiality**: the state update $h_t = (1-\tau)h_{t-1} + \tau\sigma(W_h h_{t-1} + W_{in} x_t)$ must proceed step-by-step through time, making it un-parallelizable on modern accelerators; this causes training time to scale linearly with sequence length. The second is **memory explosion**: a dense $W_h \in \mathbb{R}^{N_h \times N_h}$ causes out-of-memory (OOM) errors when the reservoir size $N_h$ reaches $10^5$, yet the capability of RC is highly dependent on reservoir dimensionality.

**Key Challenge**: The "dynamical richness" of RC originates from the non-linear activation composition of $W_h$, whereas being "parallelizable + memory-friendly" requires degrading the recurrence into a structured linear form compatible with associative scans. These two requirements are contradictory in the classical ESN framework—removing $\sigma$ seems to eliminate the non-linear expressive power.

**Goal**: This work aims to solve three sub-problems: (i) design a structured linear recurrence that supports parallel associative scans; (ii) prove this linear reservoir still satisfies ESP and is expressively equivalent to any linear ESN; (iii) reduce training costs by several orders of magnitude without sacrificing accuracy.

**Key Insight**: The authors note that deep State Space Models (S4, S5, Mamba) and Linear Recurrent Units (LRU) have proven that **complex-domain diagonal linear recurrences + non-linear readouts** can rival or exceed traditional RNNs/Transformers. Simultaneously, the fading memory filter theory of Grigoryeva and Ortega guarantees that as long as the readout layer is sufficiently expressive, a linear recurrence ESN remains a universal approximator. Combining these insights implies that the "untrained high-dimensional recurrence" of an ESN can be replaced with an LRU-style diagonal complex linear form, leaving non-linearity to a shared, lightweight mixing layer.

**Core Idea**: Reconstruct the untrained reservoir using a complex diagonal linear recurrence + ring input matrix + 1D convolutional mixing layer. This enables the recurrence to be parallelized via scanning, ensures memory grows linearly rather than quadratically with $N_h$, and theoretically proves its ESP and expressive equivalence to classical ESNs.

## Method

### Overall Architecture

ParalESN splits a block into two stages: (i) the **Reservoir**—a complex-domain linear recurrence (untrained); (ii) the **Mixing Layer**—a 1D convolutional non-linearity (untrained). Finally, a **Linear Readout** is appended as the only trainable component. The deep version (ParalESN deep) stacks multiple [Reservoir + Mixing Layer] blocks, where each layer receives the real-valued mixed state of the previous layer via a ring-topology input matrix. The entire chain only requires a closed-form solution (Ridge Regression / Least Squares) at the final layer.

Formally, the recurrence at step $t$ of layer $\ell$ is:

$$h^{(\ell)}_t = (1-\tau^{(\ell)}) h^{(\ell)}_{t-1} + \tau^{(\ell)}\left(\Lambda^{(\ell)}_h h^{(\ell)}_{t-1} + W^{(\ell)}_{in} z^{(\ell-1)}_t + b^{(\ell)}\right)$$

Where $\Lambda^{(\ell)}_h \in \mathbb{C}^{N_h \times N_h}$ is a **diagonal complex transition matrix**, $h^{(\ell)}_t \in \mathbb{C}^{N_h}$, and the post-mixed state is $z^{(\ell)}_t \in \mathbb{R}^{N_h}$. Since the recurrence is linear, the leakage coefficient can be absorbed into an equivalent transition matrix $\bar{\Lambda}^{(\ell)}_h = (1-\tau^{(\ell)})I + \tau^{(\ell)}\Lambda^{(\ell)}_h$. The update can be written as a first-order linear recurrence, satisfying the algebraic requirements for an associative scan, reducing time complexity from $O(T)$ to $O(\log T)$.

### Key Designs

1.  **Complex Diagonal Transition Matrix + LRU-style Initialization**:
    - **Function**: Replaces the dense random $W_h$ of ESN with a diagonal matrix $\bar{\Lambda}_h = \text{diag}(\lambda_1, \dots, \lambda_{N_h})$. Each eigenvalue $\lambda_i = \rho_i e^{i\theta_i}$ is initialized by sampling magnitude $\rho_i \sim \mathcal{U}[\rho_{min}, \rho_{max}]$ and phase $\theta_i \sim \mathcal{U}[\theta_{min}, \theta_{max}]$.
    - **Mechanism**: The diagonal form makes the spectral radius equal to the maximum magnitude $\max_i |\lambda_i|$. Thus, the necessary and sufficient condition for ESP simplifies to $|\lambda_i| < 1$ $\forall i$. Furthermore, the diagonal recurrence decomposes into $N_h$ independent scalar first-order recurrences, each satisfying the associativity required for parallel scans.
    - **Design Motivation**: Resolves two issues: parallelism (associative scan processes the entire sequence in $O(\log T)$ time) and memory (diagonal matrices require $N_h$ parameters instead of $N_h^2$). The magnitude/phase decoupled parameterization inherited from LRU allows fine control over the fading memory window and oscillation frequencies.

2.  **Ring Topology Input Matrix + Convolutional Mixing Layer**:
    - **Function**: The inter-layer input matrix $W^{(\ell>1)}_{in}$ uses a cyclic sparse structure—circularly shifting the input vector by one position followed by element-wise scaling. The mixing layer $f_{mix}$ uses a shared 1D convolutional kernel $W^{(\ell)}_{mix} \in \mathbb{C}^k$ sliding along the hidden dimension, followed by the real-part extraction and a $\tanh$ activation.
    - **Mechanism**: The ring matrix only requires storing $N_h$ scaling coefficients; multiplication is equivalent to "shift + element-wise," avoiding $N_h \times N_h$ dense matrices. The mixing layer uses a kernel of size $k$ ($k \ll N_h$) to enable interaction between state components with only $k+1$ parameters, independent of sequence length or hidden size.
    - **Design Motivation**: Diagonal recurrences evolve channels independently. A non-linear mixing mechanism is necessary to re-couple them, otherwise deep stacking is just a collection of independent channels. The ring input + shared kernel solves "channel coupling" and "memory explosion" simultaneously, which is critical for scaling to $10^5$ dimensions.

3.  **ESP and Universality Theoretical Guarantees**:
    - **Function**: Proves that ParalESN is expressively equivalent to "any linear recurrence ESN + MLP readout," thereby inheriting universal approximation on the class of fading memory filters.
    - **Mechanism**: Theorem 4.1 provides the necessary and sufficient ESP condition $|\lambda_i| < 1$. Proposition 4.2 proves that any $W_h \in \mathbb{C}^{N_h \times N_h}$ can be diagonalized and represented by a ParalESN (via eigenvalues and reparameterized weights) almost everywhere. Universality conclusions from Grigoryeva-Ortega for linear reservoirs are then extended to ParalESN.
    - **Design Motivation**: Addresses the core doubt regarding whether degrading RC to linear recurrences sacrifices expressivity. The proof demonstrates that diagonal constraints bring computational and memory advantages without losing expressive power.

### Loss & Training

Only the readout layer is trainable. For classification, the final state $y = f_{readout}(z^{(1)}_T, \dots, z^{(L)}_T)$ is solved via ridge regression. For regression, $y_t$ is output at each step. There is no Backpropagation (BP) and no gradient descent; the model is trained with one forward pass and one closed-form solution. Hyperparameters include $\rho_{min/max}$, $\theta_{min/max}$, $\tau$, input scale $\omega_{in}$, and kernel size $k$, tuned independently per layer.

## Key Experimental Results

### Main Results

| Task Type | Dataset | ParalESN | Traditional ESN/SOTA | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| Time-series Regression | MemCap / ctXOR / Mackey-Glass | Comparable or superior to ESN/Deep ESN | Similar bracket | Efficiency is the key gap |
| Sequence Classification | sMNIST ($N_h=10^5$) | Normal convergence | Traditional ESN OOM | ParalESN fits in VRAM |
| Long Sequences | Long Range Arena (LRA) | Competitive with fully trainable models | — | See Appendix G |
| Complexity | seq len $4^4 \to 4^8$ (128 units, 5 layers) | Time grows with $\log T$ | ESN grows linearly with $T$ | Order of magnitude speedup |

### Ablation Study

| Configuration | Reservoir Scale | Memory Performance | Key Finding |
| :--- | :--- | :--- | :--- |
| Traditional ESN | $10^5$ neurons | OOM | Dense $W_h$ explodes memory |
| ParalESN | $10^5$ neurons | Normal operation | Diagonal + ring reduces memory to linear |
| ParalESN (Shallow) | — | — | Significantly better than shallow ESN |
| ParalESN (Deep) | — | — | Matches Deep ESN performance; speed near single-layer ESN |

### Key Findings
*   **Logarithmic vs Linear**: In a 5-layer, 128-neuron setup, as sequence length increases from $4^4$ to $4^8$, traditional ESN time grows linearly, while ParalESN growth is roughly proportional to $\log T$—a direct benefit of the associative scan.
*   **OOM Boundary**: On sMNIST, scaling the reservoir to $10^5$ neurons causes traditional ESNs to crash, while ParalESN remains operational, pushing the scalability limit of RC up by an order of magnitude.
*   **Deep Versions are "Free"**: Deep ParalESN performance matches Deep ESN, but its speed is closer to a single-layer ESN. Stacking depth no longer introduces proportional linear latency, revitalizing the utility of deep RC.

## Highlights & Insights
*   **Theoretical Bridge**: Uses a clean diagonalization argument to link "classical RC ESN" and "modern SSM/LRU" into the same framework. Previously, the SSM and RC communities evolved independently; this paper shows they can share tools.
*   **Untrained + Parallel**: The biggest selling point of RC is "near-zero training cost," but it was previously hindered by sequentiality. ParalESN combines "no gradient training" with "associative scan acceleration" for a rare Pareto improvement: zero-gradient + GPU-friendly.
*   **Transferable Design**: The ring-topology input matrix and shared convolutional mixing layer are memory-efficient strategies applicable to any model aiming for extremely high hidden dimensions (e.g., long-context SSMs, massive RNNs), not just RC.

## Limitations & Future Work
*   The current mixing layer is a fixed random convolutional kernel; more complex coupling mechanisms (e.g., gating, attention) were not explored. Its expressive power might be the reason for the remaining gap between ParalESN and fully trainable models on the hardest tasks.
*   Experiments are focused on small-to-medium scale sequence tasks; verification on real-world massive text/speech/time-series foundation model scales is missing.
*   While complex-domain parameterization offers efficiency, the implementation's impact on engineering deployment (quantization, specialized inference hardware) requires systematic analysis. Integration with hardware-friendly RC (optical/electronic) is a natural next step.

## Related Work & Insights
*   **vs. Traditional ESN / Deep ESN / Res ESN**: These maintain dense non-linear recurrences for expressivity at the cost of serial training and quadratic memory growth. ParalESN achieves equivalent power with diagonal linear + convolution mixing for parallelization and linear memory.
*   **vs. LRU / S4 / S5 / Mamba**: These models share the complex diagonal linear recurrence + non-linear readout philosophy but are fully trainable and require complex initialization (e.g., HiPPO). ParalESN adapts this to an untrained + closed-form readout paradigm.
*   **vs. Structured RC / Simple Cycle Reservoir**: Early structured RCs used Hadamard or ring structures to compress $W_h$ but remained real-valued and serial. ParalESN is the first structured RC to enjoy parallelism through complex diagonalization.

## Rating
*   Novelty: ⭐⭐⭐⭐ Merging LRU's complex diagonal recurrence with RC is a clear and effective cross-domain combination.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers regression/classification/LRA, complexity curves, and OOM comparisons; theory and empiricism reinforce each other.
*   Writing Quality: ⭐⭐⭐⭐ Motivation, theory, architecture, and complexity analysis are clearly layered.
*   Value: ⭐⭐⭐⭐ Provides a scalable path for Reservoir Computing into the modern deep learning landscape, beneficial for both hardware RC and long-sequence modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Coupled Training with Privileged Information and Unlabeled Data](coupled_training_with_privileged_information_and_unlabeled_data.md)
- [\[ICML 2026\] Networked Information Aggregation for Binary Classification](networked_information_aggregation_for_binary_classification.md)
- [\[ICML 2026\] Structure-Induced Information for Rerooting Levin Tree Search](structure-induced_information_for_rerooting_levin_tree_search.md)
- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](../../AAAI2026/others/pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[NeurIPS 2025\] UniFormer: Unified and Efficient Transformer for Reasoning Across General and Custom Computing](../../NeurIPS2025/others/uniformer_unified_and_efficient_transformer_for_reasoning_across_general_and_cus.md)

</div>

<!-- RELATED:END -->
