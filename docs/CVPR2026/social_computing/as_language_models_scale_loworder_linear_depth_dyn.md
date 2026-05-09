---
title: >-
  [Paper Note] As Language Models Scale, Low-order Linear Depth Dynamics Emerge
description: >-
  [CVPR 2026][Social Computing][Transformer depth dynamics] This work treats the layer-wise forward pass of a Transformer as a discrete-time dynamical system and constructs a 32-dimensional low-order linear layer variant (LLV) surrogate to approximate the depth propagation dynamics of the last-token hidden state. The surrogate achieves a Spearman correlation of 0.995 in predicting per-layer intervention gains on GPT-2-large, and this linear identifiability monotonically increases with model scale (GPT-2 → medium → large). The closed-form optimal solution of the surrogate is further exploited to derive multi-layer activation steering schemes that require 2–5× less energy than heuristic intervention strategies.
tags:
  - CVPR 2026
  - Social Computing
  - Transformer depth dynamics
  - linear surrogate model
  - model scaling laws
  - activation intervention
  - system identification
date: 2026-05-08
content_hash: bca4fec788c05bfe
---

# As Language Models Scale, Low-order Linear Depth Dynamics Emerge

**Conference**: CVPR 2026
**arXiv**: [2603.12541](https://arxiv.org/abs/2603.12541)
**Code**: None (open-source configurations and scripts promised)
**Area**: Social Computing
**Keywords**: Transformer depth dynamics, linear surrogate model, model scaling laws, activation intervention, system identification

## TL;DR

This work treats the layer-wise forward pass of a Transformer as a discrete-time dynamical system and constructs a 32-dimensional low-order linear layer variant (LLV) surrogate to approximate the depth propagation dynamics of the last-token hidden state. The surrogate achieves a Spearman correlation of 0.995 in predicting per-layer intervention gains on GPT-2-large, and this linear identifiability monotonically increases with model scale (GPT-2 → medium → large). The closed-form optimal solution of the surrogate is further exploited to derive multi-layer activation steering schemes that require 2–5× less energy than heuristic intervention strategies.

## Background & Motivation

**State of the Field**: Activation steering has become a mainstream approach for controlling LLM behavior—injecting contrastive activation vectors at specific layers to modify attributes such as sentiment or toxicity. However, existing methods suffer from two major limitations: (1) the choice of intervention layer relies on heuristic per-layer scanning or fixed rules (e.g., "inject at the last layer"); (2) there is no theoretical guidance for allocating injection energy across layers in multi-layer interventions.

**Limitations of Prior Work**:
- The optimal intervention layer is task-dependent (middle layers are most effective for some tasks, later layers for others), making fixed rules universally suboptimal.
- Brute-force per-layer scanning is costly for large models and cannot guide coordinated multi-layer interventions.
- There is no systematic mathematical description of representation propagation along the depth dimension of Transformers.

**Root Cause**: Transformers are high-dimensional nonlinear systems and are typically treated as black boxes. However, if each layer's transformation can be locally approximated by a low-dimensional linear model given a prompt context, intervention design can be transformed from heuristic search into an optimal control problem with an analytic solution.

**Starting Point**: Drawing on system identification methods from control theory—treating depth as discrete time and the last-token hidden state as the system state—the paper applies Jacobian linearization of layer-wise transformations under frozen context, followed by Krylov subspace dimensionality reduction, to obtain a compact low-order linear surrogate.

## Method

### Overall Architecture

For a given prompt $p$, the depth-indexed hidden state of the last token is defined as $x_\ell(p) = h_\ell(p)[t(p),:] \in \mathbb{R}^H$. All non-last-token representations are frozen, and the frozen-context mapping $x_{\ell+1} = f_\ell(x_\ell; p)$ is defined under the prompt condition. This mapping is linearized via its Jacobian at the operating trajectory $\bar{x}_\ell(p)$, yielding $A_\ell(p)$. An intervention $u_\ell$ along the concept direction $v_\ell$ is injected, and after dimensionality reduction via the Krylov basis $P_\ell \in \mathbb{R}^{H \times d}$, a low-order LLV surrogate is obtained. This surrogate enables prediction of per-layer gain curves and derivation of optimal multi-layer intervention schemes.

### Key Designs

1. **Frozen-Context Local Linearization**
    - **Function**: Transforms the high-dimensional nonlinear Transformer block into an analytically tractable linear state transition.
    - **Mechanism**: All non-last-token representations are fixed; only the last-token state varies. The Jacobian $A_\ell(p) = \frac{\partial f_\ell}{\partial x_\ell}\big|_{\bar{x}_\ell}$ is computed at the operating point, yielding the linearized dynamics $\delta x_{\ell+1} \approx A_\ell(p) \delta x_\ell + A_\ell(p) v_\ell u_\ell$.
    - **Design Motivation**: Freezing the context ensures that only the depth propagation of the last token—the state directly affected by activation steering—is studied. The Jacobian is computed efficiently via JVP or central differences without explicitly constructing the full $H \times H$ matrix.

2. **Concept-Anchored Krylov Basis Dimensionality Reduction**
    - **Function**: Compresses the $H$-dimensional state space (e.g., 1280 dimensions in GPT-2-large) to $d=32$ dimensions.
    - **Mechanism**: The first column of the reduction basis $P_\ell$ is the concept direction $v_\ell$ (the difference of class-conditional means); the remaining 31 columns are constructed via reachability-inspired Krylov expansion—starting from the seed direction $A_\ell v_\ell$ after injection, iteratively propagating through the mean Jacobian, and orthogonalizing. The resulting LLV model is: $r_{\ell+1} \approx \bar{A}_\ell r_\ell + \bar{B}_\ell u_\ell$, where $\bar{A}_\ell = P_{\ell+1}^\top A_\ell P_\ell$ and $\bar{B}_\ell = P_{\ell+1}^\top A_\ell v_\ell$.
    - **Design Motivation**: The Krylov basis preferentially covers the reachable subspace actually activated by the intervention, which is systematically superior to a random orthogonal basis (verified by ablation). Placing the concept direction as the first column ensures the most attribute-sensitive direction is preserved after reduction.

3. **Minimum-Energy Multi-Layer Optimal Control**
    - **Function**: Derives a multi-layer intervention scheme that achieves a target concept shift $\Delta y_{\text{tar}}$ with minimum injection energy.
    - **Mechanism**: In the reduced model, the final concept shift is linear in the control vector: $\delta y \approx h^\top u$, where each component of $h$ is the predicted gain of the corresponding layer. The minimum-norm solution is closed-form: $u^{\star} = \frac{\Delta y_{\text{tar}}}{\|h\|_2^2} h$, i.e., energy is allocated proportionally to each layer's gain magnitude.
    - **Design Motivation**: Layers with higher gain receive more energy, and those with lower gain receive less—naturally balancing intervention efficiency. The required minimum amplitude is verified on the full model via one-dimensional bisection search.

### Loss & Training

No training is required—this is a purely analytical method. Concept directions are estimated from the class-conditional mean difference of annotated prompts (concept split, $n_{\text{concept}}=400$ per class). Jacobians are approximated via JVP or central differences (step size $2 \times 10^{-3}$). The model is identified on an operating split ($n_{\text{operating}}=200$ per class), and gains are evaluated on an independent held-out split ($n_{\text{eval}}=200$ per class), ensuring no data leakage.

## Key Experimental Results

### Main Results

**Scaling Laws: Gain Prediction Consistency of LLV Surrogate ($d=32$)**

| Model | Parameters | Hidden Dim H | Spearman↑ | Pearson↑ |
|---|:---:|:---:|:---:|:---:|
| GPT-2 | 124M | 768 | 0.77 | 0.68 |
| GPT-2-medium | 355M | 1024 | 0.81 | 0.74 |
| GPT-2-large | 774M | 1280 | **0.995** | **0.997** |

*Monotonically increasing consistency → larger models have local depth dynamics that can be more precisely described by a low-order linear surrogate.*

**Per-Task Gain Prediction on GPT-2-large ($d=32$, $\epsilon=0.1$)**

| Task | Spearman | Pearson |
|---|:---:|:---:|
| Amazon Polarity | 1.00 | 1.00 |
| Yelp Polarity | 1.00 | 1.00 |
| SST-2 | 0.99 | 0.99 |
| IMDB | 1.00 | 1.00 |
| Civil Comments Toxicity | 0.99 | 0.99 |
| TweetEval-Irony | 0.99 | 0.99 |
| TweetEval-Hate | 0.99 | 0.99 |

**Multi-Layer Control Energy Comparison (GPT-2-large, normalized to LLV optimal = 1.0×)**

| Intervention Strategy | Energy Multiple (median) |
|---|:---:|
| LLV Optimal | **1.0×** |
| Uniform-all | 2–5× |
| Last-layer-only | 10–100× |
| Random single-layer | 10–1000× |

### Ablation Study

| Ablation Dimension | Configuration | Consistency Change |
|---|---|---|
| Reduction basis | Krylov (default) | **Best** |
| | Random orthogonal basis | Significant drop on difficult tasks |
| Reduction dimension d | Very small → 32 → larger | Rapid improvement, then saturation |
| Perturbation magnitude ε | 0.01–0.5 | Stable over wide range |
| Concept direction | Class-conditional mean difference | Standard method, effective |

### Key Findings

- **"Larger models have more linear local depth dynamics"**: This is a counterintuitive yet empirically robust scaling law. Larger models are globally more complex, but their local depth responses can be captured more precisely by a compact linear surrogate.
- **Gain prediction not only identifies the optimal layer but also captures the full shape of the depth response**: including flat plateaus, non-monotonic landscapes, and other rich structures.
- **The optimal intervention depth is task-dependent**: some tasks exhibit monotonically increasing amplification at later layers, others show broad mid-to-late plateaus → the heuristic of "always inject at the last layer" is universally suboptimal.
- **LLV optimal control achieves the lowest or tied-lowest energy across all tasks**: Uniform-all is the strongest heuristic baseline but still requires 2–5× more energy.

## Highlights & Insights

- **Control-theoretic perspective on Transformer analysis**: Modeling depth propagation as a state-space system elevates intervention design from brute-force search to an optimal control problem with an analytic solution.
- **Scientific significance of the scaling law**: "Identifiability" is proposed as a new system-level metric for comparing model architectures and training strategies. Scale not only enhances capability but also increases the compressibility and predictability of local dynamics.
- **Closed-loop pipeline of analysis, design, and validation**: Analysis and design are conducted on the surrogate model → validation is performed on the full model → actual control effectiveness inversely confirms the surrogate's fidelity.

## Limitations & Future Work

- Validation is limited to the GPT-2 family (up to 774M parameters) → extension to billion-scale models such as LLaMA/Mistral is needed to verify whether the scaling law persists.
- The surrogate model is a prompt-conditioned local description → different prompts require re-identification, and computational overhead scales linearly with the number of prompts.
- Concept directions are estimated solely from class-conditional mean differences → complex concepts (e.g., "honesty") may require more refined direction estimation methods.
- Multi-concept simultaneous intervention scenarios are not addressed → potential coupling or conflicts between concepts remain unexplored.
- All 10 tasks are binary classification NLP tasks → applicability to more complex tasks (generation, reasoning) is unverified.

## Related Work & Insights

- **vs. Activation Addition (Turner et al., 2023)**: Provides intervention directions but offers no guidance on which layer is most effective → this paper predicts the complete depth gain curve.
- **vs. Linear Representation Hypothesis (Park et al., 2024)**: Explains why concepts can be linearly encoded (static representations) → this paper studies how perturbations propagate across layers (dynamical perspective).
- **vs. Moon (2024) controllability analysis**: Discusses controllability/observability of neural networks in general → this paper specifically identifies a reduced-order surrogate and validates actual control effectiveness.
- **Implications**: The low-rank nature of depth dynamics suggests the existence of redundant layers that could be safely skipped → providing theoretical guidance for layer-wise pruning; the optimal multi-layer intervention scheme has direct applications to safety alignment.

## Rating

⭐⭐⭐⭐⭐ (5/5)

The control-theoretic perspective on Transformer depth dynamics is highly novel. The finding that "larger models are more linear" is surprising yet empirically rigorous (10 tasks × 3 models × multi-dimensional ablations). Theory and experiments are tightly integrated, forming a complete closed loop from diagnosis (gain prediction) to design (optimal control)—making this a pioneering work in applying system identification theory to LLM analysis. The limitation to the GPT-2 family in terms of scale is the only notable shortcoming.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Learning from Synthetic Data via Provenance-Based Input Gradient Guidance](learning_from_synthetic_data_via_provenance-based_input_gradient_guidance.md)
- [\[CVPR 2026\] Bridging Pixels and Words: Mask-Aware Local Semantic Fusion for Multimodal Media Verification](bridging_pixels_and_words_mask-aware_local_semantic_fusion_for_multimodal_media_.md)
- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)
- [\[CVPR 2026\] Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning](revisiting_unknowns_towards_effective_and_efficient_open-set_active_learning.md)
- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](../../ACL2026/social_computing/spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)

<!-- RELATED:END -->
