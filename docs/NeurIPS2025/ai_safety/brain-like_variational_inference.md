---
title: >-
  [Paper Note] Brain-like Variational Inference
description: >-
  [NeurIPS 2025][AI Safety][variational inference] This paper proposes the FOND framework (Free energy Online Natural-gradient Dynamics), which derives spiking neural network inference dynamics from first principles via free energy minimization, and implements iPVAE (iterative Poisson VAE). iPVAE outperforms standard VAEs and predictive coding models in reconstruction–sparsity trade-off, biological plausibility, and OOD generalization.
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "variational inference"
  - "spiking neural network"
  - "predictive coding"
  - "free energy"
  - "Poisson VAE"
date: 2026-05-08
content_hash: 033f39a452fad992
---

# Brain-like Variational Inference

**Conference**: NeurIPS 2025
**arXiv**: [2410.19315](https://arxiv.org/abs/2410.19315)  
**Code**: [hadivafaii/IterativeVAE](https://github.com/hadivafaii/IterativeVAE)  
**Area**: Computational Neuroscience / Variational Inference
**Keywords**: variational inference, spiking neural network, predictive coding, free energy, Poisson VAE

## TL;DR
This paper proposes the FOND framework (Free energy Online Natural-gradient Dynamics), which derives spiking neural network inference dynamics from first principles via free energy minimization, and implements iPVAE (iterative Poisson VAE). iPVAE outperforms standard VAEs and predictive coding models in reconstruction–sparsity trade-off, biological plausibility, and OOD generalization.

## Background & Motivation
**Background**: Variational inference in machine learning (ELBO maximization) and neuroscience (free energy minimization) share mathematically equivalent formulations (ELBO = $-\mathcal{F}$), yet this equivalence has not yielded concrete, first-principles-driven algorithm or architecture designs.

**Limitations of Prior Work**: (a) Standard VAEs employ amortized inference (single encoder forward pass), which is inconsistent with the iterative/recurrent processing observed in the brain; (b) Predictive coding (PC) is iterative but communicates via continuous membrane potentials, inconsistent with the discrete spike-based communication of real neurons; (c) Existing models typically derive an architecture first and interpret it post hoc as variational inference, lacking a prescriptive methodology that derives architectures from theory.

**Key Challenge**: How can variational inference principles be translated into neural network architectures that are simultaneously computationally efficient and biologically plausible?

**Key Insight**: Following the spirit of the Bayesian Learning Rule, the paper treats distribution choice and parameterization as flexible design choices, while prescribing natural-gradient updates, online belief propagation, and iterative refinement as fixed components, deriving the architecture top-down.

**Core Idea**: By choosing Poisson distributions for the posterior and prior, using membrane potentials as dynamical variables, and performing natural gradient descent on the free energy, the framework naturally derives spiking network dynamics comprising three terms: feedforward drive, recurrent explaining away, and steady-state leak.

## Method

### Overall Architecture
The FOND framework operates on two levels:
- **Flexible choices** (determined by the modeler): distribution family (Poisson/Gaussian), parameterization (membrane potential $u$, firing rate $r = \exp(u)$)
- **Fixed prescription** (determined by FOND): (1) natural gradient descent, (2) online belief updates (current posterior becomes the next prior), (3) iterative refinement

Three concrete models are derived: iPVAE (Poisson), iGVAE (Gaussian), and $\text{iG}_\varphi$VAE (Gaussian + nonlinearity).

### Key Designs

1. **Poisson Free Energy and Its Natural Gradient**

    - Function: Derives the free energy gradient under a Poisson posterior/prior and Gaussian likelihood.
    - Core equation: $\mathcal{F} = \frac{1}{2}\|x - \Phi z\|_2^2 + \beta \sum_i (e^{u_i}(u_i - u_{0,i}) - (e^{u_i} - e^{u_{0,i}}))$
    - After applying the natural gradient (Fisher preconditioning $G(u) = \exp(u)$ cancels the $\exp(u)$ factor), a compact membrane potential dynamics is obtained: $\dot{u} \propto \Phi^T x - \Phi^T \Phi z(u) - \beta(u - u_0)$
    - Biological interpretation of the three terms: feedforward drive, recurrent explaining away, and steady-state leak.

2. **Spike-based Communication and Emergent Lateral Competition**

    - Function: Converts the dynamics from membrane potential space to firing rate space.
    - Mechanism: $r_{t+1,i} = r_{t,i} \cdot \frac{\exp(\Phi^T x)_i}{\exp(W_{ii} z_{t,i}) \prod_{j \neq i} \exp(W_{ij} z_{t,j})}$
    - The positive definiteness of $W = \Phi^T \Phi$ in the denominator guarantees self-inhibition ($W_{ii} > 0$) and mutual inhibition among neurons with overlapping tuning.
    - Key advantage: Recurrent interactions are mediated by discrete spikes $z$ rather than continuous values, making the model more biologically plausible than predictive coding.

3. **Online Inference: Temporally Evolving Prior**

    - Function: Enables continuous belief updating, where each step's posterior becomes the next step's prior.
    - Mechanism: Discrete-time update $u_{t+1} = u_t + \Phi^T x - \Phi^T \Phi z_t$; the KL term vanishes in the single-step update limit.
    - Biological significance: Corresponds to serial dependence in perception—prior stimuli influence current percepts.

4. **Training Scheme (Learning to Infer)**

    - Function: Learns generative model parameters $\Phi$ via BPTT (backpropagation through time).
    - Mechanism: During training, $T_{\text{train}}$ inference iterations are unrolled, and gradients accumulated across all steps are used for a single parameter update. $T_{\text{train}}$ can be interpreted as effective "depth."
    - At test time, $T_{\text{test}} = 1000$ steps are used, far exceeding the number of optimization steps at training time.

### Loss & Training
- Total loss = reconstruction loss (MSE) + $\beta$ × KL divergence (Poisson posterior vs. Poisson prior)
- $\beta$ controls the rate–distortion trade-off; larger $\beta$ yields sparser representations.
- Linear decoder $\hat{x} = \Phi z$ is used in the main experiments; nonlinear decoders are also supported.

## Key Experimental Results

### Main Results (Reconstruction–Sparsity Trade-off, van Hateren Natural Images)

| Model | R² (Reconstruction) | Sparsity (Zero Fraction) | Convergence Steps | Parameters |
|-------|---------------------|--------------------------|-------------------|------------|
| iPVAE | 0.83 | **77%** | 95 | Few (dictionary $\Phi$ only) |
| iG-VAE | **0.87** | 0% | 69 | Same |
| $\text{iG}_{\text{relu}}$-VAE | 0.82 | 58% | 75 | Same |
| P-VAE (amortized) | Lower | Lower | 1 | **25× more** (encoder network) |

### Key Comparisons

| Dimension | iPVAE | Standard PC | Amortized VAE |
|-----------|-------|-------------|---------------|
| Communication | Discrete spikes $z$ | Continuous membrane potential | Continuous values |
| Inference | Iterative natural gradient | Iterative vanilla gradient | Single forward pass |
| Sparsity | High (77%) | None | Low |
| V1-like features | ✓ Gabor filters | Partial | ✗ |
| OOD generalization | **Best** | Moderate | Poor |

### Ablation Study

| Configuration | Finding |
|---------------|---------|
| $T_{\text{train}} = 8/16/32$ | More training steps improve the reconstruction–sparsity frontier |
| Varying $\beta$ | Larger $\beta$ increases sparsity, as theoretically predicted |
| Linear vs. nonlinear decoder | Nonlinear decoders (MLP/CNN) yield better OOD performance |
| iPVAE vs. LCA | Statistically indistinguishable performance, but iPVAE is more robust to hyperparameters |

### Key Findings
- **All iterative VAEs consistently outperform their amortized counterparts**, despite the latter having orders of magnitude more parameters.
- The dictionary learned by iPVAE consists of V1-like Gabor filters and exhibits cortical response properties such as contrast-dependent response latency.
- iPVAE surpasses hybrid iterative-amortized VAEs (e.g., SVAE) in OOD generalization, likely due to learning compositional representations.
- On MNIST downstream classification, PVAE achieves approximately 98% accuracy, on par with supervised PCNs.

## Highlights & Insights
- **From first principles to concrete architecture**: Starting from free energy minimization, and applying Poisson distribution selection with a natural gradient prescription, the framework naturally derives a spiking network with feedforward drive, recurrent interaction, and leak terms—a paradigmatic example of prescriptive theory.
- **Emergent divisive normalization**: The multiplicative update rule in Eq. 8 naturally produces divisive normalization in the denominator, a computational primitive widely observed in the cerebral cortex.
- **Hardware-friendliness of integer spike counts**: The latent representations of iPVAE are integer-valued spike counts, making them naturally suited for low-power neuromorphic hardware.
- **Iterative inference advantages beyond accuracy**: Iterative methods perform particularly well in OOD settings because multi-step refinement can correct initial errors—an opportunity unavailable to amortized methods.

## Limitations & Future Work
- **Linear decoder in the main text**: Although the appendix extends to nonlinear decoders, the primary theory and experiments are based on the linear case.
- **Approximately 2× training time**: BPTT training for iterative inference incurs substantial overhead, though inference speed is on par with amortized methods at large batch sizes.
- **Limited evaluation datasets**: Main experiments are conducted on van Hateren image patches and MNIST; large-scale datasets (CelebA, CIFAR-10) are only briefly presented in the appendix.
- **Remaining gap in biological plausibility**: The straight-through gradient estimator and BPTT learning rule are themselves not biologically plausible.

## Related Work & Insights
- **vs. Predictive Coding (Rao & Ballard 1999)**: PC is also iterative but communicates via continuous membrane potentials; iPVAE uses discrete spike communication, which is more biologically plausible, and natural gradient descent converges faster than vanilla gradient descent.
- **vs. Standard VAE (Kingma & Welling 2014)**: VAEs perform single-step inference with an amortized encoder; iPVAE requires no encoder network, uses 25× fewer parameters, and generalizes better to OOD inputs.
- **vs. LCA (Rozell et al. 2008)**: LCA is the deterministic, non-spiking predecessor of iPVAE—the two achieve nearly identical performance, but iPVAE additionally benefits from a probabilistic framework and online learning.
- **vs. Bayesian Learning Rule (Khan & Rue 2023)**: FOND applies the BLR at the inference level—BLR unifies learning algorithms, while FOND unifies inference algorithms.

## Rating
- Novelty: ⭐⭐⭐⭐ The framework deriving concrete spiking networks from first principles is highly original, though the underlying techniques (natural gradient, Poisson VAE) already exist.
- Experimental Thoroughness: ⭐⭐⭐⭐ Model comparisons are comprehensive and systematic (9 models × multiple hyperparameters), but dataset scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is excellently organized, progressing logically from unified background → framework definition → concrete derivation → experimental validation.
- Value: ⭐⭐⭐⭐ Builds a meaningful bridge between computational neuroscience and machine learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Federated Variational Preference Alignment with Gumbel-Softmax Prior for Personalized User Preferences](../../ICML2026/ai_safety/federated_variational_preference_alignment_with_gumbel-softmax_prior_for_persona.md)
- [\[NeurIPS 2025\] DESIGN: Encrypted GNN Inference via Server-Side Input Graph Pruning](design_encrypted_gnn_inference_via_server-side_input_graph_pruning.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)
- [\[NeurIPS 2025\] Environment Inference for Learning Generalizable Dynamical System](environment_inference_for_learning_generalizable_dynamical_system.md)
- [\[NeurIPS 2025\] Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning](impact_of_dataset_properties_on_membership_inference_vulnerability_of_deep_trans.md)

</div>

<!-- RELATED:END -->
