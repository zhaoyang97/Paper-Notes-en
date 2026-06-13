---
title: >-
  [Paper Note] Neural Low-Discrepancy Sequences
description: >-
  [ICML 2026][Robotics][Low-Discrepancy Sequences] NeuroLDS utilizes a small MLP that takes integer indices through sinusoidal positional encoding…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Low-Discrepancy Sequences"
  - "Quasi-Monte Carlo"
  - "MLP"
  - "Sobol'"
  - "Path Planning"
date: 2026-05-08
content_hash: 0ebfa7e65cff28fd
---

# Neural Low-Discrepancy Sequences

**Conference**: ICML 2026  
**arXiv**: [2510.03745](https://arxiv.org/abs/2510.03745)  
**Code**: https://github.com/camail-official/neuro-lds  
**Area**: Scientific Computing / Quasi-Monte Carlo / Neural Sampling  
**Keywords**: Low-Discrepancy Sequences, Quasi-Monte Carlo, MLP, Sobol', Path Planning  

## TL;DR
NeuroLDS utilizes a small MLP that takes integer indices through sinusoidal positional encoding, first regressing to Sobol' sequences and then fine-tuning with a closed-form $L_2$ discrepancy loss over all prefixes. It produces the first extensible neural low-discrepancy sequence supporting arbitrary lengths, outperforming Sobol'/Halton across 4D discrepancy metrics, Borehole integration, RRT motion planning, and Black–Scholes PDE solving.

## Background & Motivation

**Background**: Quasi-Monte Carlo (QMC) relies on low-discrepancy point sets/sequences to approximate integrals in $[0,1]^d$ with an error rate approaching $\mathcal{O}(N^{-1})$, compared to the $\mathcal{O}(N^{-1/2})$ of IID Monte Carlo. Classical constructions (Halton, Sobol', rank-1 lattice, digital nets) are based on number theory—using radical-inverse with prime bases or primitive polynomials over $\mathbb{F}_2$ to generate direction numbers. Recently, Message-Passing Monte Carlo (MPMC) formalized "finding minimum discrepancy point sets" as a differentiable optimization problem, using GNNs to learn an end-to-end mapping for a fixed $N$, achieving historically low discrepancy values on a small scale.

**Limitations of Prior Work**: MPMC can only provide "sets," not "sequences." Once trained for a fixed $N=1024$, adding a single point requires retraining the entire network. Conversely, incremental sampling planners like RRT require extensible sequences where "every prefix is as uniform as possible." Furthermore, the discrepancy of classical LDS is particularly low at $N=2^m$ but fluctuates significantly in the $2^m < N < 2^{m+1}$ interval—for instance, in van der Corput sequences, non-power-of-two $N$ values are consistently worse than the corresponding $2^m$.

**Key Challenge**: In QMC, "low discrepancy" and "extensibility" represent a structural contradiction. Sets can be globally optimized for extremely low discrepancy, but adding points destroys their uniformity; sequences must satisfy the strong constraint that "every prefix has low discrepancy," meaning their discrepancy curves are naturally inferior to the optimal set of the same length. MPMC focuses on the former, while classical Sobol'/Halton focus on the latter, but neither is Pareto optimal on the same curve.

**Goal**: To train a neural network $f_\theta: \{1,\dots,N\}\to [0,1]^d$ such that for any prefix $P\le N$, the discrepancy of $\{f_\theta(i)\}_{i=1}^P$ is minimized, and the discrepancy decreases smoothly across $N$ rather than exhibiting sawtooth oscillations.

**Key Insight**: The essence of classical LDS is "using the digital expansion of $i$ (radical-inverse / Gray-coded direction numbers) as input features to perform a deterministic transformation." This is naturally suited for neural network imitation—feeding $i$ into a network and letting it learn a "generalized digital rule." Discrepancy has a closed-form $L_2$ kernel representation (Eq. 2), which is fully differentiable and can serve directly as a loss function.

**Core Idea**: Represent the "index-to-point" mapping with a small MLP, using $K$-band sinusoidal features to simulate digital expansion at the input. Training occurs in two stages: first, supervised fitting to Sobol' as an inductive bias (to avoid collapsing into corners), followed by fine-tuning with an unsupervised loss based on the "sum of all prefix discrepancies."

## Method

### Overall Architecture
NeuroLDS is a deterministic sequence generator $f_\theta: \{1,\dots,N\}\to [0,1]^d$. The pipeline is as follows:

1. Index $i$ → $K$-band sinusoidal positional encoding $\psi_i \in \mathbb{R}^{1+2K}$;
2. $\psi_i$ passes through an $L$-layer MLP (ReLU + final layer sigmoid) → point $\mathbf{X}_i \in [0,1]^d$;
3. The overall sequence $\{\mathbf{X}_1,\dots,\mathbf{X}_N\}$ constitutes the generated LDS.

Training consists of two stages: minimizing MSE to regress to a Sobol' sequence (pre-training), followed by fine-tuning with a weighted sum of all prefix discrepancies as the loss.

### Key Designs

1. **Sinusoidal Index Encoding Mimicking Digital Expansion**:

    - **Function**: Exposes integer $i$ as a set of continuous, multi-frequency feature vectors for the downstream MLP.
    - **Mechanism**: $\psi(i) = [\,i/N,\; \sin(2^k\pi i/N),\; \cos(2^k\pi i/N)\,]_{k=0}^{K-1} \in \mathbb{R}^{1+2K}$, adopting the Fourier-feature approach from NeRF/Transformers. Each frequency axis $2^k\pi$ conceptually corresponds to a "base digit," similar to Halton's base-$b$ digit or Sobol's binary bit $g_k(i)$.
    - **Design Motivation**: Classical LDS achieve low discrepancy by mapping different bits of $i$ to different scales of the point to avoid conflicts; sinusoidal encoding is a continuous relaxation of this mechanism, allowing the MLP to freely combine frequency bands to produce novel "digital rules." Ablations show that larger $K \in \{8, 16, 32\}$ lead to more stable discrepancy curves (small $K$ results in large oscillations), at the cost of slightly increased training time.

2. **Two-Stage Training (Sobol' Pre-training + Closed-form $L_2$ Discrepancy Fine-tuning)**:

    - **Function**: Solves the degeneration problem where training from scratch with discrepancy loss collapses into a single corner of $[0,1]^d$.
    - **Mechanism**: Stage 1 minimizes $\mathcal{L}_{\text{pre}}(\theta) = \frac{1}{N}\sum_i \|f_\theta(\psi_i) - q_i\|_2^2$, where $q_i$ is a Sobol' sequence (discarding the first 128 burn-in points); Stage 2 minimizes the weighted sum of all prefix discrepancies $\mathcal{L}_{\text{disc}}(\theta) = \sum_{P=2}^N w_P \cdot D_2^\bullet(\{\mathbf{X}_i\}_{i=1}^P)^2$, where $D_2^\bullet$ is the closed-form kernel integral (star / sym / ctr / per / ext / asd). The computational complexity per prefix is $\mathcal{O}(dN^2)$, and weights $w_P$ are uniform by default.
    - **Design Motivation**: Ablation comparing "direct discrepancy loss training" and "Sobol' pre-training followed by fine-tuning" shows that the former collapses in 2/3/4 dimensions, while the latter converges stably—indicating that Sobol' topology acts as a successful strong inductive bias. Furthermore, the loss is inherently differentiable and does not rely on surrogate estimators.

3. **Prefix-wise Discrepancy Loss + Optional High-Dimensional Weights**:

    - **Function**: Directly encodes the "sequence" extensibility constraint into the training objective and supports coordinate importance weighting in high dimensions.
    - **Mechanism**: For any $P\le N$, the $L_2$ discrepancy is calculated using the closed-form kernel $(D_2^k(\{\mathbf{X}_i\}_{i=1}^P))^2 = \iint k\,d\boldsymbol{x}\,d\boldsymbol{y} - \frac{2}{P}\sum_i \int k(\mathbf{X}_i,\boldsymbol{y})\,d\boldsymbol{y} + \frac{1}{P^2}\sum_{i,j} k(\mathbf{X}_i,\mathbf{X}_j)$. All prefixes are included in the loss simultaneously. In high dimensions, a product weight kernel $\tilde k(\boldsymbol{x},\boldsymbol{y}) = \prod_j (1 + \gamma_j\, k(x_j,y_j))$ is used to suppress the influence of unimportant coordinates.
    - **Design Motivation**: Classical LDS discrepancy is only minimal at $N=2^m$ and fluctuates in between. Weighting all prefixes in the loss naturally "flattens" the curve. The Borehole case study verified that using $\boldsymbol{\gamma}$ estimated from sensitivity analysis allows NeuroLDS to further outperform NM-Greedy on anisotropic integration.

### Loss & Training
- **Stage 1**: MSE $\mathcal{L}_{\text{pre}}$ targeting the Sobol' sequence post burn-in.
- **Stage 2**: $\mathcal{L}_{\text{disc}}(\theta) = \sum_{P=2}^N w_P D_2^\bullet(\{\mathbf{X}_i\}_{i=1}^P)^2$, with $w_P$ default to uniform $1/(N-2)$; optional length-proportional weighting $w_P^* = 2P/(N^2+N-2)$—the latter performs better on long prefixes and slightly worse on short ones.
- **Kernels**: $\bullet \in \{\text{star, sym, ctr, per, ext, asd}\}$ are interchangeable; Optuna is used to tune the best hyperparameters (learning rate, width, depth, $K$) for each loss.

## Key Experimental Results

### Main Results

| Dataset | Metric | NeuroLDS (Ours) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| Borehole 8D Integration ($N=460$) | Absolute Error | **0.0657** | 0.1086 (Sobol') | ~40% error reduction |
| Borehole 8D Integration ($N=260$) | Absolute Error | **0.0239** | 0.4516 (Halton) | Significant lead |
| RRT Kinematic Chain (Width 0.64) | Success Rate % | **96.58** | 87.95 (Halton) | +8.6 |
| RRT Kinematic Chain (Width 0.40) | Success Rate % | **80.00** | 67.32 (Halton) | +12.7 |
| 2D Black–Scholes PDE Training | MSE ($\times 10^{-4}$) | **3.34** ($D_2^{\text{ctr}}$) | 4.04 (Sobol') | ~17% error reduction |

To reach the same average success rate as NeuroLDS, Sobol' requires 2.50× the points, Halton 1.55×, and uniform sampling 2.27×.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Full model (Pre-train + FT) | Stable convergence | Complete model |
| w/o Sobol' Pre-training (Direct) | Collapse to a corner | Direct discrepancy minimization fails in 2/3/4 dimensions |
| Index Encoding $K=8$ | High variance curve | Insufficient frequency bands to cover all scales |
| Index Encoding $K=32$ | Smoothest curve | Slightly increased training time |
| Linear Layers (No ReLU) | Failed Sobol' fit | Validates necessity of deep non-linearity |
| AR-GNN instead of MLP | Discrepancy degrades | Long context training signal decay |
| LSTM instead of MLP | Slightly better but 6× slower | Gains do not justify the cost |
| $w_P^*$ Length Weighting | Better for long prefixes | Aligns with intuition of weighting later stages |

### Key Findings
- Sobol' pre-training is critical—without it, the discrepancy loss causes the model to "collapse to a corner" in all dimensions, consistent with phenomena reported by Clément et al., 2025.
- In RRT, low discrepancy not only improves average success rates but is most crucial in "difficult-to-pass" scenarios like narrow passages (width 0.4)—validating the intuition that extensible LDS are better suited for incremental exploration than fixed sets.
- For the Black–Scholes PDE, continuous kernels (centered $D_2^{\text{ctr}}$ and average squared $D_2^{\text{asd}}$) show the greatest improvement, suggesting kernel choice should match the smoothness assumptions of the task.

## Highlights & Insights
- **Reinterpreting "Digital Expansion" as "Positional Encoding"**: Halton's radical-inverse digits and Sobol's Gray-code direction numbers both essentially map different bits of $i$ to different scales of the point. NeuroLDS achieves this with sinusoidal multi-frequency encoding but changes the "mapping rule" from hard-coded to learnable. This perspective connects QMC with NeRF and Transformers in terms of mathematical structure.
- **Discrepancy Closed-form as Loss**: Using the classical $L_2$ discrepancy $\mathcal{O}(dN^2)$ closed-form expression with automatic differentiation aligns theory perfectly with practice. It also retains flexibility for interchangeable kernels, including weighted anisotropic kernels.
- **Pre-training as a "Safety Anchor" for Inductive Bias**: Neural network optimization for non-convex losses naturally tends toward collapse. By first pulling the network to a "known good" initial manifold (Sobol'), subsequent discrepancy minimization becomes stable and yields actual progress. This pattern is valuable for other geometric optimization problems like optimal transport or sampling design.

## Limitations & Future Work
- Success depends on number-theoretic constructions like Sobol' or Halton as pre-training targets, meaning a "completely number-theory-free" ML-only LDS has not yet been achieved. How the choice of classical sequence and its inherent bias affects the final result remains an open question.
- Discrepancy calculation $\mathcal{O}(dN^2)$ remains expensive for large $N$. The paper only demonstrates up to $N=10^4$; scaling to $N=10^6$ (common for high-resolution QMC) requires approximate discrepancy or randomized acceleration.
- Verification is biased toward "traditional QMC-friendly" tasks (integration, PDE, RRT); its transferability to more "open" scenarios like RL exploration or generative model sample quality still needs testing.
- High-dimensional weighting $\boldsymbol{\gamma}$ relies on a-priori sensitivity knowledge. Finding weights in blind scenarios requires an initial coarse sampling, introducing a "startup" cost.

## Related Work & Insights
- **vs MPMC** (Rusch & Kirk, 2024): MPMC uses GNNs to learn optimal sets for a fixed $N$, achieving extremely low discrepancy but no extensibility. NeuroLDS uses MLP + index to learn sequences, with discrepancy slightly higher than MPMC at its target $N$ but decreasing smoothly throughout. They are complementary "Set SOTA" vs "Sequence SOTA."
- **vs Classical Sobol'/Halton**: Classical constructions are optimal at $N=2^m$ but oscillate in between. NeuroLDS weights all prefixes equally, lowering the overall curve and removing the sawtooth patterns.
- **vs NM-Greedy** (Chen et al., 2018): NM-Greedy also supports weighted discrepancy minimization via Nelder–Mead global search, but it cannot generalize—adding points requires rerunning. NeuroLDS can generate points for any length once trained.
- **vs Neural Fields (NeRF / SIREN)**: NeuroLDS implements the "index → point" relationship as a coordinate network, representing an interesting migration of INR concepts from "signal representation" to "sampling design."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The first method to truly generate extensible neural LDS, clarifying the link between digital expansion and positional encoding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers discrepancy, integration, planning, and PDEs, though $N$ scale is relatively small ($\le 10^4$) and limited to $d=8$.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear mathematical derivation with an appendix providing 6 kernel forms and Borehole formulas; low barrier to reproduction.
- **Value**: ⭐⭐⭐⭐⭐ Immediate impact on scientific computing pipelines requiring uniform sampling; code is open-sourced by MIT-CSAIL/Rus Lab.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models](neural_implicit_action_fields_from_discrete_waypoints_to_continuous_functions_fo.md)
- [\[ICLR 2026\] RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](../../ICLR2026/robotics/rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)
- [\[NeurIPS 2025\] BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning](../../NeurIPS2025/robotics/beast_efficient_tokenization_of_b-splines_encoded_action_sequences_for_imitation.md)
- [\[ICCV 2025\] COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation](../../ICCV2025/robotics/cosmo_combination_of_selective_memorization_for_low-cost_vision-and-language_nav.md)
- [\[AAAI 2026\] Towards Reinforcement Learning from Neural Feedback: Mapping fNIRS Signals to Agent Performance](../../AAAI2026/robotics/towards_reinforcement_learning_from_neural_feedback_mapping_.md)

</div>

<!-- RELATED:END -->
