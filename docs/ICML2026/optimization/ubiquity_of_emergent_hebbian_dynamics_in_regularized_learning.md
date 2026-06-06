---
title: >-
  [Paper Note] Ubiquity of Emergent Hebbian Dynamics in Regularized Learning
description: >-
  [ICML 2026][Optimization][Hebbian learning] Ours proves that near the steady state of L2 weight decay, the learning signals of **almost any** learning rule (SGD, Adam, DFA…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Hebbian learning"
  - "weight decay"
  - "learning signal alignment"
  - "noise-regularization phase diagram"
  - "neural plasticity"
date: 2026-05-08
content_hash: 88b11594879092de
---

# Ubiquity of Emergent Hebbian Dynamics in Regularized Learning

**Conference**: ICML 2026  
**arXiv**: [2505.18069](https://arxiv.org/abs/2505.18069)  
**Code**: None  
**Area**: Optimization Theory / Biological Plasticity Modeling  
**Keywords**: Hebbian learning, weight decay, learning signal alignment, noise-regularization phase diagram, neural plasticity

## TL;DR
Ours proves that near the steady state of L2 weight decay, the learning signals of **almost any** learning rule (SGD, Adam, DFA, or even random networks) **spontaneously** align toward the Hebbian direction, while sufficiently strong noise flips it to anti-Hebbian, with a clear phase boundary appearing at $\gamma \propto \sigma^2$.

## Background & Motivation

**Background**: Classical neuroscience considers Hebbian and anti-Hebbian plasticity as core mechanisms of brain learning—"fire together, wire together" is achieved through local homosynaptic rules (such as STDP) and prevented from diverging by homeostatic constraints (such as Oja's rule). In machine learning, deep networks are almost exclusively trained using gradient descent with weight decay and stochastic perturbations, which appear entirely different from biological mechanisms.

**Limitations of Prior Work**: Experimentally, once Hebbian/anti-Hebbian structures are observed in synaptic updates, they are often used as "counter-evidence" to prove that the brain cannot be performing global error-driven optimization. This draws a hard boundary between the two learning paradigms.

**Key Challenge**: Can one conclude that the underlying computation is Hebbian solely by "observing Hebbian-form updates"? In other words, is the Hebbian signature *identifiable*? Previous work by Ziyin et al. (2025b) hinted that weight decay is related to Hebbian alignment but only provided scattered observations without a unified mechanism or an explanation for when anti-Hebbian dynamics emerge.

**Goal**: (1) Provide a *generic* near-steady-state mechanism explaining why L2 weight decay forces learning signals to project toward the Hebbian direction; (2) Provide a dual mechanism explaining why noise flips it to anti-Hebbian; (3) Perform empirical validation across multiple architectures, optimizers, and tasks.

**Key Insight**: The authors notice that in biology, Hebbian is an **expansive** force that must be countered by homeostatic **contraction**; in deep learning, the learning signal must also exert expansion to resist the contraction of weight decay. **Once a system enters a near-steady state, two opposing forces must balance**, which imposes geometric constraints on the direction of the learning signal.

**Core Idea**: By treating weight decay as a "universal anti-Hebbian term," the steady-state condition $\mathbb{E}_x[g(x,\theta)] \approx \gamma W$ automatically forces the expected learning signal to have the same sign as the Hebbian direction $\bar H(W)=\mathbb{E}_x[h_b h_a^\top]$; noise introduces an additional quadratic term that reverses this inequality.

## Method

This paper is a purely theoretical and simulation-based study without a new algorithm; therefore, the "Method" section involves the mathematical derivation of two mechanisms and the details of the experimental paradigms.

### Overall Architecture

Consider a hidden layer $h_b = W h_a(x)$, where $h_a$ is the post-activation of the previous layer and $h_b$ is the pre-activation of the current layer. Updates with weight decay can be written as:

$$\Delta W \propto \underbrace{-(\nabla_{h_b}\ell)\, h_a^\top}_{\text{learning signal}} - \gamma W$$

Define the Hebbian direction $\Delta_{\rm Hebb} W = h_b h_a^\top$ and the anti-Hebbian direction $-h_b h_a^\top$. The core quantity is the **alignment** between the "learning signal" and the Hebbian update, characterized by the Frobenius inner product or cosine similarity. The overall argument is divided into two parts: Section 3 provides two theorems (specific and general forms) for positive alignment, and Section 4 provides a phase diagram where noise flips the alignment to negative.

### Key Designs

1.  **Hebbian Alignment Theorem under Steady State (Strong Form)**:
    - **Function**: Proves that for "standard" gradient learning signals, the expected alignment increases monotonically with weight decay and is strictly positive.
    - **Mechanism**: From the near-steady-state constraint $\mathbb{E}_x[(\nabla_{h_b}\ell) h_a^\top] + \gamma W \approx 0$, right-multiplying by $W^\top$ and substituting $h_b = W h_a$ yields $\mathbb{E}_x[(\nabla_{h_b}\ell) h_b^\top] = -\gamma W W^\top$, thus $\mathrm{Tr}\,\mathbb{E}_x[(\nabla_{h_b}\ell)^\top h_b] = -\gamma\,\mathrm{Tr}[WW^\top] < 0$. By assuming the "presynaptic norm is approximately constant" $\|h_a\|^2 \approx \mathbb{E}\|h_a\|^2$ (which holds under neural collapse or normalization), the Frobenius inner product between the learning signal and the Hebbian update simplifies to $\gamma\,\mathbb{E}\|h_a\|^2\,\mathrm{Tr}[WW^\top] > 0$.
    - **Design Motivation**: Directly characterizes **statistical correlation** (not just matching signs) and explicitly provides the monotonic dependence of alignment on $\gamma$, which serves as the theoretical basis for why increasing weight decay strengthens alignment in experiments.

2.  **Universal Hebbian Projection Theorem for Arbitrary Learning Rules (Weak Form)**:
    - **Function**: Removes the assumption that the "learning signal is the true negative gradient" and proves that **any** learning rule $g(x,\theta)$ satisfying the near-steady-state condition automatically gains a positive Hebbian projection.
    - **Mechanism**: Writing the general update as $\Delta W = g(x,\theta) - \gamma W$, the steady state gives $\mathbb{E}_x[g(x,\theta)] \approx \gamma W$. Directly calculating the inner product with $\bar H(W) := \mathbb{E}_x[h_b h_a^\top]$ yields: $\langle \mathbb{E}_x[g],\bar H\rangle_F = \gamma\langle W,\bar H\rangle_F = \gamma\,\mathbb{E}_x \|h_b\|^2 > 0$. This step is entirely independent of $g$ originating from a gradient; thus, Adam, DFA, and even a "Random NN" acting as a teacher satisfy this.
    - **Design Motivation**: Upgrades Hebbian alignment to a "**universal projection** in regularized learning"—this is the most counter-intuitive conclusion, implying that the observation of Hebbian signatures **cannot** be used to infer whether the underlying algorithm is performing gradient-based learning.

3.  **Noise-Induced anti-Hebbian Phase Transition**:
    - **Function**: Explains when anti-Hebbian dynamics appear and provides a phase diagram boundary between $\gamma$ and noise variance $\sigma^2$.
    - **Mechanism**: In linear regression $\ell(w)=\tfrac12 (w^\top x - y)^2$, injecting parameter noise $w = v + \epsilon$ at each step where $\epsilon \sim \mathcal N(0,\sigma I)$. The alignment between the SGD signal $\Delta_{\rm SGD} w = -x(w^\top x - y)$ and the Hebbian update $\Delta_{\rm Hebb} w = x w^\top x$ expands to $\mathbb{E}_\epsilon[(\Delta_{\rm SGD}w)^\top \Delta_{\rm Hebb}w] = -\|x\|^2[(v^\top x)^2 + \sigma^2\|x\|^2 - v^\top x\,y]$. The extra $-\sigma^2 \|x\|^4$ term causes the alignment to become negative as noise increases. With weight decay, the boundary is approximately $\approx -\sigma^2 c_0 + \gamma c_1$, which places the phase transition on a parabola $\gamma \propto \sigma^2$, consistent with the "white phase transition band" in Figure 4.
    - **Design Motivation**: Unifies the biological observation of "coexisting and switchable Hebbian/anti-Hebbian" phenomena into a phase diagram perspective of *contractive vs. expansive forces*. It also provides a testable prediction: **anti-Hebbian dominance should be observed in brain regions with strong ambient noise and weak weight decay**.

### Loss & Training

Ours does not introduce new training objectives; all experiments follow standard CE/MSE. The two standard setups are:
-   **SCE** (Standard Classification Experiment): A 2-layer 128-d tanh MLP trained on CIFAR-10, cross-entropy, default $\eta=0.01$, batch=256, 50 epochs.
-   **SRE** (Standard Regression Experiment): Student-teacher regression, 32-dimensional isotropic Gaussian input/output, 20k training samples + 2k validation; the transformer variant uses 2 layers, 4 heads, and 32-d token embedding.
The paper repeatedly emphasizes using a **large batch size (256)** for alignment measurement—because the theorems concern steady-state expectations, and small-batch SGD noise would mask the Hebbian signal.

## Key Experimental Results

### Main Results

**Table 1 (Excerpt)**: Fixed SRE setup, sweeping weight decay $\gamma$, 10 seeds, reporting mean±std of the cosine alignment between the 2nd layer learning signal and the Hebbian update.

| Model | Learning Rule | $\gamma=0$ | $\gamma=5\!\times\!10^{-5}$ | $\gamma=5\!\times\!10^{-4}$ | $\gamma=5\!\times\!10^{-3}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Regression MLP | Adam | $-0.02\pm0.00$ | $0.10\pm0.00$ | $\mathbf{0.66\pm0.01}$ | — |
| Regression MLP | SGD | $-0.10\pm0.01$ | $-0.06\pm0.01$ | $0.17\pm0.01$ | $\mathbf{0.59\pm0.01}$ |
| Regression MLP | DFA | $0.45\pm0.05$ | $0.45\pm0.04$ | $0.68\pm0.05$ | $\mathbf{0.87\pm0.00}$ |
| Regression MLP | Random NN | $0.00\pm0.00$ | $0.00\pm0.00$ | $0.05\pm0.00$ | $\mathbf{0.50\pm0.00}$ |
| Transformer | Adam | $-0.02\pm0.02$ | $0.50\pm0.24$ | $\mathbf{0.99\pm0.02}$ | — |
| Transformer | SGD | $0.00\pm0.01$ | $0.04\pm0.01$ | $0.47\pm0.06$ | $\mathbf{0.88\pm0.03}$ |

The most counter-intuitive row is Random NN—it "learns nothing" (the learning signal comes from the pseudo-error of a randomly initialized network output), but as long as $\gamma$ is increased to $5\!\times\!10^{-3}$, the alignment still reaches 0.5. "—" indicates weight collapse to 0 at that $\gamma$.

### Ablation Study

**Noise-Decay Phase Diagram (Summary of Figure 4)**: Fixed SRE, sweeping combinations of parameter noise $\sigma$ and weight decay $\gamma$, recording the sign of steady-state alignment.

| Configuration | Hebbian Alignment Sign | Description |
| :--- | :--- | :--- |
| Low noise + high $\gamma$ | Strongly Positive ($>0.5$) | Contraction dominant, aligns with Hebbian direction |
| High noise + low $\gamma$ | Strongly Negative ($<-0.3$) | Expansion dominant, flips to anti-Hebbian |
| Transition band $\gamma \approx \sigma^2$ | $\approx 0$ | Matches the parabolic prediction of the theorem |
| Very low noise + very low $\gamma$ | Near 0 | System not at steady state, alignment is non-directional |
| Various activations | Consistent Trend | Monotonic increase with $\gamma$ is robust across activations |

### Key Findings
-   **Regularization Determines Alignment Sign**: By increasing $\gamma$ from 0 to $5\!\times\!10^{-3}$, alignment in almost all (model, optimizer) combinations jumps from $\approx 0$ to 0.5-0.99, particularly extreme for Transformer+Adam (0.99).
-   **"Non-learning rules also align"**: Random NN and DFA also align (DFA even shows 0.45 at $\gamma=0$), supporting the weak form theorem—this is the strongest evidence for the non-identifiability of Hebbian signatures.
-   **Best Generalization is not Hebbian**: Figure 5 reveals that the solution with the lowest validation loss occurs near the phase transition band where both Hebbian and anti-Hebbian alignments are near 0; strong Hebbian alignment does not imply better learning.
-   **Early "Alignment Bump"**: Especially under ReLU + small learning rates, a sharp peak in Hebbian alignment appears early in training while weight norms remain monotonic—implying this phase arises from **feature alignment** rather than norm expansion.
-   **Steady-State Oscillations**: Updates for individual neurons oscillate between directions over the long term; models with stronger oscillations often generalize better, though the converse is not necessarily true.

## Highlights & Insights

-   **Reframing "Hebbian as an underlying mechanism" as a falsifiable proposition**: Previously, neuroscientists took the observation of Hebbian signatures as evidence against gradient descent. Ours provides clear counter-examples—Random NNs also "appear Hebbian." This shifts the identification problem from the biological level to the **signal-dynamic level**, which can be distinguished by measuring LTP/LTD alongside weight decay and noise.
-   **"Universal Projection" perspective**: Treating $\bar H(W)=\mathbb{E}_x[h_b h_a^\top]$ as a directional cone forced by steady-state regularization. Any algorithm satisfying $\mathbb{E}[g] \approx \gamma W$ is projected by this cone. This is a rare, minimalist "optimizer-agnostic" alignment theorem applicable to LoRA, batchnorm, and other implicit regularizations.
-   **Observable Phase Transitions**: The $\gamma \propto \sigma^2$ parabolic phase diagram provides a specific, verifiable prediction for biological experiments—if a brain region has high noise and slow weight decay, it should systematically favor anti-Hebbian dynamics, providing a concrete link between machine learning theory and neuroscience measurements.

## Limitations & Future Work

-   **"Near-Steady State" Assumption is not Universal**: The theorem relies on the expected update approaching zero. Large models in long-tail training rarely reach true steady states; the authors acknowledge that anti-Hebbian deviations observed in larger models might stem from the failure of the steady-state condition.
-   **"Presynaptic norm is approximately constant" is a strong assumption**: It depends on neural collapse or normalization layers. While the Appendix provides extensions for non-uniform $\gamma$, whether this holds for general high-dimensional representations requires more validation.
-   **Experiments limited to small models**: CIFAR-10 + 2-6 layer MLP/small transformers; did not touch LLM scales. Ours does not answer whether alignment remains observable in 100B parameter models.
-   **Lack of a "how to distinguish" experimental protocol**: The theory provides a dichotomy of mechanistic vs. emergent Hebbian, but does not provide a classifier ready for LTP/LTD data, leaving this to the neuroscience community.
-   **Future Directions**: Extending analysis to non-L2 penalties (e.g., sparsity penalties, which the authors mention enhance anti-Hebbian), replacing noise models with dual-source ambient + synaptic noise, and extending the alignment theorem to practical architectures with batchnorm/layernorm.

## Related Work & Insights

-   **vs. Xie & Seung 2003 (Contrastive Hebbian Algorithm)**: They proved CHA is equivalent to backpropagation in the equilibrium limit, but CHA is not a homosynaptic rule and violates Hebbian principles; the equivalence in ours is "phenomenological"—it *looks* Hebbian without requiring the underlying rule to be homosynaptic.
-   **vs. Ziyin et al. 2025b**: This is the closest prior work, which suggested a link between weight decay and Hebbian dynamics but offered only scattered observations. Ours upgrades this to a version with monotonicity and statistical correlation, paired with a dual theorem for anti-Hebbian dynamics.
-   **vs. DFA/Feedback Alignment (Lillicrap, Nøkland)**: That line of work attempts to design "biologically plausible gradient surrogates"; ours argues that whether a surrogate "looks Hebbian" cannot serve as a criterion for plausibility, reshaping the evaluation standards for such work.
-   **vs. Oja's rule / BCM / Bienenstock 1982**: Classical Hebbian rules require explicit normalization; ours provides an implicit normalization path—L2 weight decay directly forces alignment without needing an manual normalization term in the rule.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)
- [\[ICML 2026\] Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks](dynamics_and_representation_structure_of_local_approximations_to_gradient-based_.md)
- [\[ICML 2026\] Muon in Associative Memory Learning: Training Dynamics and Scaling Laws](muon_in_associative_memory_learning_training_dynamics_and_scaling_laws.md)
- [\[ICML 2026\] Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks](balancing_learning_rates_across_layers_exact_two-step_dynamics_and_optimal_scali.md)
- [\[ICML 2026\] Mirror Mean-Field Langevin Dynamics](mirror_mean-field_langevin_dynamics.md)

</div>

<!-- RELATED:END -->
