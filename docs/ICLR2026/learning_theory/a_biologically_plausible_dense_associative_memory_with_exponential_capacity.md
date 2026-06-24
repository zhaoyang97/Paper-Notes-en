---
title: >-
  [Paper Note] A Biologically Plausible Dense Associative Memory with Exponential Capacity
description: >-
  [ICLR 2026][Learning Theory][Dense Associative Memory] By replacing the "winner-take-all" activation in the hidden layer of a dual-layer associative memory with a thresholded step activation, hidden neurons can participate in multiple memories simultaneously (distributed representation). This increases storage capacity from "linear in the number of hidden neurons" to "exponential in the number of hidden neurons" ($2^{N_h}$). The model was validated on MNIST/CIFAR-10…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Associative Memory"
  - "Computational Neuroscience"
  - "Dense Associative Memory"
  - "Modern Hopfield Networks"
  - "Exponential Capacity"
  - "Distributed Representation"
  - "Biological Plausibility"
date: 2026-05-08
content_hash: 39b307519b109687
---

# A Biologically Plausible Dense Associative Memory with Exponential Capacity

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=mRZOayQL1i](https://openreview.net/forum?id=mRZOayQL1i)  
**Code**: To be confirmed  
**Area**: Learning Theory / Associative Memory / Computational Neuroscience  
**Keywords**: Dense Associative Memory, Modern Hopfield Networks, Exponential Capacity, Distributed Representation, Biological Plausibility

## TL;DR
By replacing the "winner-take-all" activation in the hidden layer of a dual-layer associative memory with a thresholded step activation, hidden neurons can participate in multiple memories simultaneously (distributed representation). This increases storage capacity from "linear in the number of hidden neurons" to "exponential in the number of hidden neurons" ($2^{N_h}$). The model was validated on MNIST/CIFAR-10, demonstrating the ability to store tens of thousands of highly correlated images while maintaining biological plausibility.

## Background & Motivation

**Background**: Associative memory networks are a class of attractor models that encode memories as stable fixed points of network dynamics, allowing for the retrieval of full memories from partial or noisy inputs. The storage capacity of classical Hopfield networks grows only linearly with the number of neurons. "Dense Associative Memory" (DAM, also known as Modern Hopfield Networks) enhances capacity to super-linear or even exponential levels by introducing high-order interactions. However, naive implementations requiring non-linear high-order synaptic interactions are difficult to realize in biological circuits.

**Limitations of Prior Work**: Krotov & Hopfield (2021) proposed a **biologically plausible** dual-layer implementation where visible neurons correspond to features and hidden neurons act as intermediaries. It relies on standard pairwise synaptic interactions and allows high-order interactions between visible layers to "emerge" through the choice of hidden layer activation functions. However, it suffers from two major drawbacks: ① Capacity grows at most **linearly** with the number of hidden neurons $N_h$, which is inefficient for information storage. ② During inference, it exhibits **winner-take-all** (WTA) behavior, where only one hidden neuron is active at the converged fixed point. Consequently, the hidden layer learns "grandmother cell" local representations (one neuron tied to one memory) rather than more efficient distributed representations.

**Key Challenge**: The WTA nonlinearity locks each hidden neuron into a "one neuron = one memory" mapping, limiting the number of distinguishable hidden states to $N_h$ and fixing the capacity ceiling. To break this limit, hidden neurons must be reusable across multiple memories, and a single memory should be representable by a combination of hidden neurons.

**Key Insight**: The authors identifies that the root of the problem lies in the choice of the hidden layer nonlinearity. By replacing WTA (e.g., softmax, power-law, spherical normalization) with a **thresholded step activation** $\Theta(h_\mu - \theta)$ and shifting the operating regime from $N_v < N_h$ (classical Hopfield) to $N_v \gg N_h$ (visible layer much larger than hidden layer), the weight matrix effectively becomes an identity matrix under the Law of Large Numbers. This decouples the hidden neurons, allowing all $2^{N_h}$ binary combinations of hidden states to become stable fixed points.

**Core Idea**: Use a minimalist modification—"thresholded step activation + visible layer much larger than hidden layer"—to support distributed representations. This enables all $2^{N_h}$ binary patterns to serve as stable memories, elevating capacity from linear $N_h$ to exponential $2^{N_h}$.

## Method

### Overall Architecture

The network consists of a **bipartite structure**: $N_v$ visible neurons $v_i$ (representing features) and $N_h$ hidden neurons $h_\mu$ (intermediary units). The two layers are symmetrically connected with no lateral connections within a layer. The continuous-time dynamics are:

$$\tau_v \frac{dv_i}{dt} = -v_i + \frac{1}{\sqrt{N_h}}\sum_{\mu=1}^{N_h}\xi_{i\mu}\,\Theta(h_\mu - \theta), \qquad \tau_h \frac{dh_\mu}{dt} = -h_\mu + \frac{\sqrt{N_h}}{N_v}\sum_{i=1}^{N_v}\xi_{\mu i} v_i$$

where $\Theta(\cdot)$ is the standard Heaviside step function ($1$ if $z>0$, otherwise $0$), and synaptic weights are symmetric with $\xi_{\mu i} = \xi_{i\mu} \sim \mathcal{N}(0,1)$. The process involves a noisy input entering from the visible layer, the hidden layer quickly converging (assuming $\tau_h \ll \tau_v$) to a binary activation pattern $s_\mu = \Theta(h_\mu - \theta) \in \{0,1\}$, and the visible layer subsequently reconstructing the clean memory. The core of the paper is the **proof** that these simple dynamics possess $2^{N_h}$ stable fixed points with large basins of attraction in the $N_v \gg N_h$ regime.

### Key Designs

**1. Thresholded Step Activation: Replacing WTA with Distributed Representations**

This step directly addresses the "capacity fixed at $N_h$" and "grandmother cell" issues. Krotov-Hopfield activations (power-law, softmax) force a single winning hidden neuron. The proposed thresholded step activation $s_\mu = \Theta(h_\mu - \theta)$ **does not enforce competition**, allowing multiple hidden neurons to be active simultaneously. The hidden layer shifts from "one-hot" coding to "arbitrary binary code," where each neuron encodes "base components" shared across many memories. Theoretically, the capacity limit expands from "which single neuron is lit" ($N_h$) to "which group of neurons are lit" ($2^{N_h}$).

**2. $N_v \gg N_h$ Regime: Decoupling the Hidden Layer for $2^{N_h}$ Fixed Points**

This is the theoretical core. Substituting the fixed-point conditions, the effective update for the hidden layer is $s_\mu = \Theta\!\big(\sum_\nu J_{\mu\nu}s_\nu - \theta\big)$, where $J_{\mu\nu} = \frac{1}{N_v}\sum_i \xi_{\mu i}\xi_{i\nu}$. Unlike the classical Hopfield regime ($N_v < 0.138\,N_h$), this work assumes $N_v \gg N_h$. By the Marchenko–Pastur law, $J$ approaches the identity matrix, decoupling the hidden neurons. For finite $N_v$:

$$J_{\mu\nu} = \delta_{\mu\nu} + \frac{\zeta_{\mu\nu}}{\sqrt{N_v}}, \qquad \zeta_{\mu\nu} \sim \mathcal{N}(0, 1+\delta_{\mu\nu})$$

The fixed-point condition becomes $s_\mu = \Theta\!\big(s_\mu + q_\mu - \theta\big)$, where the perturbation $|q_\mu| = \big|\tfrac{1}{\sqrt{N_v}}\sum_\nu \zeta_{\mu\nu}s_\nu\big| \lesssim \sqrt{(N_h+1)/N_v}$. With an **optimal threshold $\theta = 1/2$**, the equation has solutions $s_\mu = 0$ and $s_\mu = 1$ for each $\mu$ as long as $|q_\mu| < 1/2$. The probability of no "bit flips" approaches 1 exponentially as $N_v$ increases. Thus, all $2^{N_h}$ binary combinations are fixed points.

**3. Stability and Large Basins of Attraction**

At equilibrium, $h_\mu = \sum_\nu J_{\mu\nu}s_\nu$. Since $J \approx I$, $h_\mu$ is either close to 0 or 1, far from the threshold $\theta = 1/2$. The **derivative of the step function is zero at equilibrium**, meaning the Jacobian vanishes and the fixed points are naturally stable. Analysis shows that as long as visible layer noise variance $\sigma_v^2 \ll N_v/N_h$, the hidden layer converges to the target binary pattern first, dragging the visible layer back to the clean memory. This loose upper bound implies a large basin of attraction and robustness to noise.

**4. Biological Plausibility: Local Activation, Asymmetric Weights, and Heterogeneous Thresholds**

Compared to Krotov-Hopfield, the activation functions here are **local**, and neuron activity remains within physiological ranges. Moreover, experiments demonstrate that **asymmetric weights and heterogeneous thresholds** still allow for stable retrieval. This is crucial as real neural circuits lack strict symmetry and neuron excitability varies, suggesting that memory dynamics do not require perfectly uniform parameters.

### A Practical Example: 60,000 MNIST Images via 50 Hidden Neurons

To illustrate: in the MNIST experiment with $N_v = 784$ and $N_h = 50$, linear models could store at most 50 memories. This network, leveraging $2^{50} \approx 10^{15}$ binary combinations as fixed points, successfully stored 60,000 highly correlated images and learned **57,913 unique minima**. Different styles of the digit "6" converge to different but overlapping hidden representations, capturing both shared category features and individual details.

### Loss & Training

While capacity analysis focuses on fixed random weights, a learning rule is needed for correlated data. The authors define **basic memories** as the weight vectors $\xi_\mu$. Complex memories are combinations of these basic components. Given target memories $\{v_m\}_{m=1}^M$, the weights $\xi$ and thresholds $\theta$ are optimized via:

$$(\xi,\theta) = \arg\min_{\xi,\theta}\sum_{m=1}^{M}\Big\|v_m - \frac{1}{\sqrt{N_h}}\sum_{\mu=1}^{N_h}\xi_\mu\,\Theta\!\big(\tfrac{\sqrt{N_h}}{N_v}\xi_\mu^\top v_m - \theta\big)\Big\|^2$$

During training, Xavier initialization is used, and a steep sigmoid approximates the non-differentiable $\Theta$ for gradient descent. The learned basic memories are approximately orthogonal, allowing a few components to represent a vast number of complex memories, thereby reducing redundancy.

## Key Experimental Results

### Main Results: High-Capacity Retrieval under Correlation

| Dataset | $N_v$ | $N_h$ | Memories Stored | Unique Minima Learned |
|---------|-------|-------|-----------------|-----------------------|
| MNIST | 784 | 50 | 60,000 | 57,913 |
| CIFAR-10 | 3,072 | 500 | 50,000 | 49,982 |

Even with highly correlated data and the CIFAR-10 setup partially violating the $N_v \gg N_h$ assumption, the network mapped nearly all memories to unique, stable, and interpretable minima. Learned thresholds (0.21 for MNIST, 0.43 for CIFAR-10) differed slightly from the theoretical 0.5 due to non-Gaussian data statistics.

### Classifiability of Recalled Representations (Generalization)

Non-linear classifiers were trained on recalled representations and tested on **unseen** samples:

| Representation | MNIST Accuracy | CIFAR-10 Accuracy |
|----------------|----------------|-------------------|
| Recalled Hidden | 95% | 40% |
| Recalled Visible | 98% | 56% |
| Original Image (Reference) | 99% | 88% |

### Key Findings

- **Exponential Capacity is Feasible**: Using 50 and 500 hidden neurons to store 60,000 and 50,000 correlated memories respectively breaks the "memories $\le$ hidden neurons" ceiling.
- **Simultaneous Memory and Generalization**: Unseen inputs are mapped to stable attractors that capture their features without being pulled into previously stored patterns.
- **Data-Dependent Category Structure**: MNIST hidden representations are highly separable (95%) because the raw pixel space has a strong categorical structure. For CIFAR-10, where categories are less correlated in pixel space, the visible representation accuracy (56%) significantly outperforms the hidden layer (40%).

## Highlights & Insights

- **Simplified Path to Exponential Capacity**: Changing the activation and flipping the operating regime reframes the capacity bottleneck from "single neuron choice" to "combinatorial group choice."
- **Rigorous Theoretical Grounding**: Utilizing the Marchenko–Pastur law to prove the decoupling of hidden neurons with Gaussian weights is an elegant application of random matrix theory.
- **Stability from Non-differentiability**: The fact that the step function's zero derivative provides clear stability guarantees for fixed points is a counter-intuitive yet robust finding.
- **Architecture Efficiency**: Compared to models requiring multiple modules for distributed coding, this work achieves exponential capacity within a single module.

## Limitations & Future Work

- **Training Rule Biological Implausibility**: While retrieval is biologically plausible, storing data currently relies on backpropagation-style global optimization.
- **The $N_v \gg N_h$ Constraint**: Exponential capacity depends on the visible dimension being significantly larger than the hidden dimension; this assumption's boundary was visible in CIFAR-10 experiments.
- **Gaussian Assumptions**: Theoretical results based on Gaussian weights and global thresholds deviate from empirical findings on real datasets.
- **Missing Biological Constraints**: Sparse connectivity and Dale’s law are not yet incorporated.

## Related Work & Insights

- **vs. Krotov & Hopfield (2021)**: Both are bipartite DAMs, but the latter uses WTA activation (linear capacity, local coding). This work adopts thresholded activation for exponential capacity and distributed coding.
- **vs. Chandra et al. (2025)**: While Chandra et al. use multiple WTA modules to achieve exponential capacity, this paper demonstrates that a single module is sufficient if the activation function is modified.
- **vs. Classical/Modern Hopfield**: Unlike models requiring complex high-order synaptic interactions, this implementation achieves exponential capacity using only standard pairwise synapses in a dual-layer framework.
- **vs. Transformers / Diffusion Models**: By identifying the bridge between dense associative memory and attention mechanisms, this work provides a high-capacity, biologically plausible example for modern deep learning architectures.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Transforming capacity from linear to exponential through minimalist changes is a major reframing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on MNIST/CIFAR-10, though the limits of the $N_v \gg N_h$ assumption were reached.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivations and honest disclosure of assumptions and limitations.
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm for biologically plausible associative memory with significant theoretical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dynamical properties of dense associative memory](dynamical_properties_of_dense_associative_memory.md)
- [\[ICLR 2026\] Transformers as Measure-Theoretic Associative Memory: A Statistical Perspective and Minimax Optimality](transformers_as_measure-theoretic_associative_memory_a_statistical_perspective_a.md)
- [\[ICLR 2026\] Adaptive Hopfield Network: Rethinking Similarities in Associative Memory](adaptive_hopfield_network_rethinking_similarities_in_associative_memory.md)
- [\[ICLR 2026\] Sublinear Spectral Clustering Oracle with Little Memory](sublinear_spectral_clustering_oracle_with_little_memory.md)
- [\[ICLR 2026\] Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity](tversky_neural_networks_psychologically_plausible_deep_learning_with_differentia.md)

</div>

<!-- RELATED:END -->
