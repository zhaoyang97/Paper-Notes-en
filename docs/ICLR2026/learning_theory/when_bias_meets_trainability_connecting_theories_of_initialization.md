---
title: >-
  [Paper Note] When Bias Meets Trainability: Connecting Theories of Initialization
description: >-
  [ICLR 2026][Learning Theory][Mean-Field Theory] This work demonstrates a rigorous mathematical equivalence between the core quantities of two independent theories characterizing randomly initialized wide networks: **Mean-Field (MF) theory**, which analyzes gradient stability/trainability, and **Initial Guessing Bias (IGB) theory**, which analyzes initial prediction preferences. It derives the counter-intuitive conclusion that the "Edge of Chaos" initialization…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Neural Network Initialization"
  - "Mean-Field Theory"
  - "Initial Guessing Bias (IGB)"
  - "Edge of Chaos (EOC)"
  - "Trainability"
  - "Vanishing/Exploding Gradients"
date: 2026-05-08
content_hash: c3a9f1a4d929f0a3
---

# When Bias Meets Trainability: Connecting Theories of Initialization

**Conference**: ICLR 2026  
**Paper**: When Bias Meets Trainability: Connecting Theories of Initialization  
**Code**: https://github.com/abassi98/igb_and_trainability  
**Area**: Learning Theory / Neural Network Initialization  
**Keywords**: Mean-Field Theory, Initial Guessing Bias (IGB), Edge of Chaos (EOC), Trainability, Vanishing/Exploding Gradients

## TL;DR
This work demonstrates a rigorous mathematical equivalence between the core quantities of two independent theories characterizing randomly initialized wide networks: **Mean-Field (MF) theory**, which analyzes gradient stability/trainability, and **Initial Guessing Bias (IGB) theory**, which analyzes initial prediction preferences. It derives the counter-intuitive conclusion that the "Edge of Chaos" initialization, which makes networks most trainable, corresponds precisely to the **most biased (rather than neutral)** state, where this bias is rapidly absorbed during early training.

## Background & Motivation

**Background**: Understanding the initial statistical properties of deep networks "before seeing data" is key to explaining their trainability. Two nearly parallel research lines exist: **Mean-Field theory** (Schoenholz, Poole, Hayou, etc.), which analyzes how signals and gradients propagate with depth in the infinite-width limit, identifying the "ordered/chaotic phase" transition. At the **Edge of Chaos (EOC)** between these phases, the signal propagation depth scale diverges, and the network is trainable from the first to the last layer, representing the optimal initialization. The other is the recently proposed **IGB theory** (Francazi et al. 2024), which suggests an untrained network may inherently "prefer" certain classes, assigning large regions of the input space to the same category (termed *prejudice*) or maintaining *neutrality*—depending solely on architecture and initialization.

**Limitations of Prior Work**: These two theories have operated independently. MF focuses on "gradient stability and trainability," while IGB focuses on "initial prediction bias." However, **the impact of initial prediction bias on trainability remains unclear**. Furthermore, the original IGB framework assumed zero bias variance $\sigma_b^2=0$, focused on single-node activations, and relied on the intuition that "neutral initialization learns fastest."

**Key Challenge**: The phase diagram coordinates of MF are $(\sigma_b^2, \sigma_w^2)$, while IGB uses bias intensity $\gamma$. They employ different "stochastic averaging" methods: MF fixes data and averages over the weight ensemble, whereas IGB fixes one initialization and passes the entire input distribution through the network. These different averaging orders seemed to preclude a direct connection.

**Goal**: (1) Mathematically map the core quantities of MF to those of IGB; (2) Use this correspondence to answer the relationship between "initial bias and trainability"; (3) Generalize IGB to non-zero bias and multi-node activations, while correcting errors in existing MF phase diagrams (e.g., for ReLU).

**Key Insight**: The authors observe that both theories hold in the "infinite-width" limit and study the same set of pre-activation distributions—one focusing on the "variance of activations around the center" and the other on the "variance of the center itself." If these variances are different slices of the same quantities, the theories can be unified.

**Core Idea**: Establish an equivalence theorem to translate MF's signal variance/covariance $(q_{aa}, q_{ab})$ into IGB's "activation drift ratio" $\gamma$. This allows reading both "trainability" and "prediction bias" from the same phase diagram, leading to the conclusion: **Optimal Trainable State (EOC) = Transient-Deep Prejudice State**.

## Method

### Overall Architecture

This is a purely theoretical work. The main thread is "welding" two initialization theories together and using the unified framework to reinterpret phase diagrams, derive counter-intuitive conclusions, and perform experimental validation. The process involves three steps: **Establishing magnitude equivalence between MF and IGB (Theorem 3.1) $\to$ Mapping "gradient stability (trainability)" and "prediction bias" onto the same phase diagram to show EOC is a biased state (Prop 4.1) $\to$ Generalizing the framework to general networks and activations**.

In MF, fixing data and averaging over weights, pre-activations at infinite width are i.i.d. Gaussian with mean 0 and variance $q_{aa}^{(l)}$. The relationship between different inputs $a, b$ is characterized by the correlation coefficient $c_{ab}^{(l)} = q_{ab}^{(l)}/\sqrt{q_{aa}^{(l)} q_{bb}^{(l)}}$. The ordered/chaotic phases are distinguished by $\chi_1 \equiv \partial c^{(l+1)}_{ab}/\partial c^{(l)}_{ab}\big|_{c=1}$ (or $\tilde\chi_1\equiv\partial q^{(l+1)}_{ab}/\partial q^{(l)}_{ab}\big|_{c=1}$ for unbounded activations): $\tilde\chi_1<1$ leads to vanishing gradients (ordered), $\tilde\chi_1>1$ to exploding gradients (chaotic), and $\tilde\chi_1=1$ is the EOC. In IGB, averaging over data first, the pre-activation of each node $i$ becomes Gaussian around a **non-zero center** $\mu_i^{(l)}$, which itself fluctuates with initialization (variance $\sigma_{\mu^{(l)}}^2$), while the activation variance around the center is $\sigma_{y^{(l)}}^2$. Bias strength is measured by the **activation drift ratio**:

$$\gamma^{(l)} \equiv \frac{\sigma_{\mu^{(l)}}^2}{\sigma_{y^{(l)}}^2}.$$

When $\gamma\ll 1$, node centers are drowned by noise (neutrality); when $\gamma\gg 1$, center drift dominates (prejudice). The critical point $\gamma^{(L)}=1$ separates "neutrality" ($\gamma<1$) from "prejudice" ($\gamma>1$).

### Key Designs

**1. MF↔IGB Equivalence Theorem: Mapping Core Quantities**

This is the foundation of the paper, resolving the conflict between different averaging methods. Theorem 3.1 proves that in the Mean-Field regime, given normalized inputs ($q^{(0)}_{aa}=1$, $q^{(0)}_{ab}=0$ for $a \neq b$), the two MF quantities in the infinite-data limit are exactly the two types of IGB variances:

$$q_{aa}^{(l)} = \sigma_{\mu^{(l)}}^2 + \sigma_{y^{(l)}}^2, \qquad q_{ab}^{(l)} = \sigma_{\mu^{(l)}}^2 \;(a\neq b),$$

yielding a direct relationship between the correlation coefficient and the drift ratio:

$$c_{ab}^{(l)} = \frac{\gamma^{(l)}}{1+\gamma^{(l)}}.$$

This mapping implies that while $q_{aa}, q_{ab}$ are stochastic in MF, they **concentrate to their means** in the "infinite-width, infinite-data" limit, becoming equivalent to $\sigma_y^2, \sigma_\mu^2$ in IGB.

**2. EOC as "Transient-Deep Prejudice" rather than Neutrality**

With $c=\gamma/(1+\gamma)$, "gradient stability" can be translated into "bias strength," leading to the counter-intuitive result (Prop 4.1). In both the ordered phase and EOC, $c=1$ is a stable fixed point, corresponding to $\gamma\to\infty$, or **deep prejudice**. However, the dynamics differ:

- **Ordered Phase** ($\tilde\chi_1<1$): Gradients vanish exponentially, initial conditions are "frozen," and prejudice is **persistent** (persistent-deep prejudice); the network cannot learn.
- **EOC** ($\tilde\chi_1=1$): Gradients are stable; although prejudice is strong, it is **rapidly absorbed** during early training (transient-deep prejudice), making it the optimal trainable state.
- **Chaotic Phase** ($\tilde\chi_1>1$): Gradients explode, and the network may appear biased or neutral.

Prop 4.1 concludes: **Regarding trainability, the optimal initialization (EOC) is not a neutral state but a transient-deep prejudice state.** This refutes the "neutrality is fastest" hypothesis.

**3. Generalization of IGB and Correction of MF Phase Diagrams**

The authors extend IGB to **non-zero bias variance** $\sigma_b^2$ and **multi-node activation functions** (e.g., max-pool/average-pool). They also correct omissions in previous ReLU phase diagrams, proving that for ReLU, the correlation coefficient $c^{(l)}$ **converges to 1 across the entire phase diagram**, indicating persistent deep prejudice. However, phases are still distinguishable by convergence rates and variance behavior: in the ordered phase, $\gamma^{(l)}$ **diverges exponentially**, whereas in the chaotic phase, it **diverges by power-law**.

**4. Per-Class Gradient Asymmetry**

A direct consequence is that initial bias makes gradient vanishing/explosion **class-dependent**. In the chaotic deep-prejudice phase of unbounded activations, the softmax probability is pushed toward one class, causing the gradient for the favored class to approach zero while other gradients explode. This **per-class gradient imbalance** significantly slows down learning.

### Loss & Training
Standard cross-entropy training is used. The practical application of the theoretical conclusions lies in **initialization hyperparameter selection**: tuning $(\sigma_w^2, \sigma_b^2)$ so gradients are stable along depth (i.e., EOC) ensures both gradient stability and the fastest absorption of initial bias, thereby **reducing the training steps required for hyperparameter tuning**.

## Key Experimental Results

### Main Results: Unified Phase Diagram and Classification

Table 1 characterizes each phase using both IGB and MF parameters:

| IGB ($\gamma$) | MF ($c$) | Gradient Sense ($\tilde\chi_1$) | Phase | Trainability |
|---|---|---|---|---|
| $\gamma=\infty$ | $c=1$ | $\tilde\chi_1<1$ | Ordered-deep prejudice | Vanishing grad, persistent bias |
| $\gamma=\infty$ | $c=1$ | $\tilde\chi_1=1$ | **Transient-deep prejudice (EOC)** | **Stable grad, transient bias, Optimal** |
| $\gamma=\infty$ | $c=1$ | $\tilde\chi_1>1$ | Chaotic-deep prejudice | Exploding grad, training failure |
| $1<\gamma<\infty$ | $0.5<c<1$ | $\tilde\chi_1>1$ | (chaotic) Prejudice | Exploding grad |
| $\gamma<1$ | $c<0.5$ | $\tilde\chi_1>1$ | (chaotic) Neutrality | Exploding grad, poor accuracy |

Theory validation (Fig. 2): In an MLP with width 10,000 and depth 100, the analytical IGB curves (calculated via $c=\gamma/(1+\gamma)$) match labels from MF Monte Carlo simulations, confirming the concentration of signal variances/covariances.

### Training Dynamics Validation

| Model / Setting | Task | Key Finding |
|---|---|---|
| Tanh MLP | Fashion-MNIST / CIFAR10 | EOC corresponds to the initial maximum bias state, but bias is absorbed quickly; neutral states perform poorly. |
| Vanilla ViT (no BN/LN/skip) | CIFAR10 | Gradients show the same phase transition behavior as MLPs. |
| Large ViT (ImageNet pre-tuned) | CIFAR100 | Original weights = weak IGB, optimal training; scaled weights $\times 0.5$ trigger strong IGB; $\times 1.5$ weaken IGB but unstable gradients slow training. |

### Key Findings
- **EOC = Fastest Learning + Highest Bias**: EOC is simultaneously the state of "fastest learning dynamics" and "maximum initial classification bias"; prejudice is transient and quickly absorbed.
- **Neutral Initialization is Inefficient**: Contrary to intuition, neutral states only appear in the chaotic phase with exploding gradients.
- **ResNet/Normalization blurs Phase Distinction**: Residual MLPs often have a single critical phase, explaining why modern architectures are "inherently easier to train."
- **Per-Class Gradient Imbalance**: In chaotic deep-prejudice phases, gradients for favored classes are zero while others explode, impacting learning quality.

## Highlights & Insights
- **Rosetta Stone for Theories**: Theorem 3.1 translates "trainability" (MF) and "prejudice" (IGB) into the same language ($c=\gamma/(1+\gamma)$).
- **Counter-intuitive Proposition**: Proving "optimal initialization must be biased" refutes common "neutrality = optimal" assumptions, providing insight for initialization design.
- **Practical Trick**: Tuning $(\sigma_w^2, \sigma_b^2)$ to achieve stable gradients (EOC) yields both stability and fast bias absorption, shortening hyperparameter tuning runs.
- **Warning on Short-Run Evaluations**: Explains why evaluating models with very few steps systematically favors certain classes due to unabsorbed IGB.

## Limitations & Future Work
- **Limit Assumptions**: Conclusions rely on "infinite-width, then infinite-data, then infinite-depth" sequences; real-world deviation for finite networks is not fully quantified.
- **Idealized Architectures**: Clear phase transitions require "vanilla" networks without BN/LN/skip, which are standard in modern architectures.
- **Distributional Assumptions**: Inputs are assumed to be i.i.d. Gaussian; extrapolation to structured real data requires caution.
- **Future Directions**: Developing diagnostic/initialization algorithms for "bias re-absorption" and studying how normalization layers shift phase boundaries.

## Related Work & Insights
- **vs. Mean-Field Theory (Schoenholz 2016 / Poole 2016)**: These works focus on "gradient stability"; this paper adds the "bias" dimension to EOC and corrects ReLU phase details.
- **vs. IGB (Francazi et al. 2024)**: Generalizes IGB to $\sigma_b^2 \neq 0$ and refutes the "neutrality = best" hypothesis using gradient stability analysis.
- **vs. Class Imbalance (Francazi et al. 2023)**: Provides an initialization-level mechanistic explanation for training difficulties in class-imbalanced scenarios via "per-class gradient imbalance."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ High conceptual contribution by unifying independent frameworks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across MLPs and ViTs, though focused on idealized architectures.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, though notationally dense.
- Value: ⭐⭐⭐⭐ Practical EOC tuning advice and warnings on evaluation bias.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] When Sample Selection Bias Precipitates Model Collapse](../../ICML2026/learning_theory/when_sample_selection_bias_precipitates_model_collapse.md)
- [\[ICLR 2026\] A New Initialization to Control Gradients in Sinusoidal Neural Networks](a_new_initialization_to_control_gradients_in_sinusoidal_neural_networks.md)
- [\[ICLR 2026\] Transformers with Endogenous In-Context Learning: Bias Characterization and Mitigation](transformers_with_endogenous_in-context_learning_bias_characterization_and_mitig.md)
- [\[ICLR 2026\] Overparametrization bends the landscape: BBP transitions at initialization in simple Neural Networks](overparametrization_bends_the_landscape_bbp_transitions_at_initialization_in_sim.md)
- [\[ICLR 2026\] Know When to Abstain: Optimal Selective Classification with Likelihood Ratios](know_when_to_abstain_optimal_selective_classification_with_likelihood_ratios.md)

</div>

<!-- RELATED:END -->
