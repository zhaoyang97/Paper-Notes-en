---
title: >-
  [Paper Note] A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State
description: >-
  [ICML 2026][LLM/NLP][Sampling error] This paper characterizes the information loss introduced by sampling from high-entropy distributions in GPT-style LLMs from a differential geometry perspective. It constructs $\mathfr…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Sampling error"
  - "differential geometry"
  - "parallel transport"
  - "world model"
  - "chess"
date: 2026-05-08
content_hash: 333fc58bc1456cc7
---

# A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State

**Conference**: ICML 2026  
**arXiv**: [2605.04899](https://arxiv.org/abs/2605.04899)  
**Code**: See supplementary materials  
**Area**: LLM Interpretability / NLP  
**Keywords**: Sampling error, differential geometry, parallel transport, world model, chess

## TL;DR
This paper characterizes the information loss introduced by sampling from high-entropy distributions in GPT-style LLMs from a differential geometry perspective. It constructs $\mathfrak{so}(n)$-valued 1-forms and parallel transport operators, and demonstrates in chess probing experiments that such geometric rotations are highly aligned with the world vectors learned by the model.

## Background & Motivation

**Background**: Autoregressive LLMs generate tokens via greedy or stochastic decoding. When the output distribution is concentrated (high confidence), sampling error is negligible; when the distribution is diffuse ("blurry points"), different samples can cause subsequent trajectories to diverge significantly.

**Limitations of Prior Work**: The sensitivity of LLMs to single-token perturbations is well known, but the intrinsic structure of this sensitivity has not been characterized—is it merely chaotic exponential divergence, or does it reflect geometric properties internal to the model?

**Key Challenge**: The internal state $z_t \in \mathbb{R}^n$ and the output distribution (discrete probabilities) are linked via projection/softmax/sampling, a highly nonlinear relationship. How can one describe this "internal → external → feedback to internal" coupling within a unified geometric framework?

**Goal**: To establish a testable relationship between the geometric properties of blurry points in the model's internal state and the world vectors obtained via probing.

**Key Insight**: Using the language of differential geometry, sampling uncertainty is modeled as a vector-valued degree on a manifold, injecting uncertainty into geometric actions via exterior products and parallel transport.

**Core Idea**: Parameterize blurring strength as a triple exterior product $A(z_t) = 4 z_t \wedge (p_1 v_1) \wedge (p_2 v_2)$, then contract with tangent vectors to obtain an $\mathfrak{so}(n)$-valued 1-form; its parallel transport imposes a testable rotation in hidden state space, coupled with the direction of the world vector.

## Method

### Overall Architecture

Three-layer structure: (1) Geometric modeling: exterior product and $\mathfrak{so}(n)$ structure; (2) Parallel transport and holographic measurement; (3) Experimental validation: confirming the coupling via clustering of world vector directions in chess tasks.

### Key Designs

1. **Upgrading Multiple Exterior Products to $\mathfrak{so}(n)$-Valued 1-Forms**:

    - **Function**: Converts the scalar magnitude of blurring into a directed geometric object acting on the hidden state.
    - **Mechanism**: Define $A(z_t) = 4 z_t \wedge (p_1 v_1) \wedge (p_2 v_2)$, whose Frobenius norm $\|A\|_F$ is proportional to the degree to which $z_t$ lies in the plane spanned by the top two token embeddings. Contracting with tangent vector $\mu$ yields $A_\mu(z_t) = 4 p_1 p_2 \big(-(\mu\cdot v_1)(z_t \wedge v_2) + (\mu \cdot v_2)(z_t \wedge v_1)\big)$, an element of the $\mathfrak{so}(n)$ Lie algebra (rotation generator).
    - **Design Motivation**: Scalar magnitude cannot encode direction; using antisymmetric tensors naturally introduces rotational actions, carrying directional uncertainty.

2. **Probability Charge and Parallel Transport Operator**:

    - **Function**: Parameterizes the strength of geometric action and computes cumulative rotation via closed path integration.
    - **Mechanism**: $4 p_1 p_2$ is termed "probability charge"—analogous to electric charge coupling in electromagnetism; it vanishes as $p_2 \to 0$ (high confidence) or $p_1 \ll p_2$, and reaches its maximum value 1 at $p_1=p_2=0.5$ (maximum uncertainty). The parallel transport operator along curve $\gamma$ is $U_\gamma = P\exp\big(-\int_0^1 A_{\dot\gamma(s)}(\gamma(s))\,ds\big)$ ($P$ denotes path ordering), measuring rotation of the hidden state in uncertain regions.
    - **Design Motivation**: Connects to gauge theory intuition, giving sampling uncertainty a physical analogue—"geometric curvature" as a carrier of information loss.

3. **Holography and Finite-Step Closed Path Measurement**:

    - **Function**: Under the constraint of not being able to continuously translate hidden states, measures local curvature via a closed square loop around a point.
    - **Mechanism**: The model only generates actual trajectories, not arbitrary paths, so point-to-point parallel transport is infeasible. Instead, an $\epsilon$-sized clover (four small squares combined to eliminate coordinate bias) is used to construct a closed loop, yielding a holographic operator $H_{z_t}$, reflecting local curvature $R = \partial_\mu A_\nu - \partial_\nu A_\mu - [A_\nu, A_\mu]$.
    - **Design Motivation**: Adopts standard techniques from lattice gauge theory, enabling extraction of local geometric information under restricted observations.

### Loss & Training
No new models are trained in this work; probes are trained as linear classifiers on frozen LLM hidden states to predict 737 chess piece positions.

## Key Experimental Results

| Setting | Task | Model | Key Metric | Result |
|---------|------|-------|------------|--------|
| Chess world model | 737 piece position classification | Qwen 32B | Mean accuracy | 81.2%–100% |
| Chess world model | Same as above | Mistral 24B | Mean accuracy | 76.0%–98.9% |
| Blurring sensitivity | Move selection difference | Qwen 32B | Position evaluation change | 4.5±1.5 log-cp |
| Geometric-semantic coupling | Rotation direction vs world vector | Qwen 32B | Mean $\|\cos\|$ | Top pieces >0.7, overall >0.5 |
| Geometric-semantic coupling | Board region clustering | Qwen 32B | Clustering purity | Board quadrants >85% |

### Key Findings
- **Coupling of World Vectors and Geometric Rotation**: At all branch points, the rotation direction and corresponding world vector have mean $|\cos|>0.5$, top pieces $>0.7$, far exceeding the random baseline $\sim 0.07$.
- **Geometric Mapping of Piece Importance**: High-value pieces (e.g., queen) correspond to the strongest blurring signals and largest rotation magnitudes; piece value is positively correlated with geometric strength.
- **Model Capability vs Geometric Signal**: The Mistral probe, with lower accuracy, also shows significantly weaker geometric signals, suggesting this is a feature of genuine structure learned by LLMs.
- **Sampling Error Is Not Pure Chaos**: Divergence directions are coupled with world structure, indicating this is not directionless chaos but a geometric projection of internal model representations.

## Highlights & Insights
- **Mathematical Elegance**: Sampling uncertainty is precisely characterized using Lie algebras and parallel transport, with tools borrowed from gauge theory fitting the context well.
- **New Dimension of Interpretability**: Links model brittleness to the geometry of the world model, providing a deeper explanation than "chaotic divergence."
- **Clever Task Choice**: Chess has a well-defined world model (legal positions), facilitating objective and quantitative validation of the coupling.
- **Cross-Model Consistency**: The coupling phenomenon appears in two different model families, suggesting it may be a general property of LLMs.

## Limitations & Future Work
- Validation is limited to the chess domain; it remains unclear whether the findings generalize to natural language or open domains.
- Probe training data is artificially balanced; its correlation with real text distributions is limited.
- The effect is significantly weaker on Mistral, indicating sensitivity to model choice and the need for more family-level validation.
- There is no rigorous proof yet for why geometric rotation must align with the world vector; this is currently an empirical observation.
- Future directions: extend to natural language tasks; design new decoding strategies using geometric insights; theoretically analyze how Lie group structures emerge from training.

## Related Work & Insights
- **vs Traditional Sensitivity Analysis**: Previous work only described exponential divergence; this paper further shows that the direction and pattern of divergence are linked to the world model.
- **vs Probe/World Model Work (e.g., OthelloGPT)**: Prior work verified the existence of world models; this paper explains why world models and sampling uncertainty are coupled.
- **Insights**: Differential geometry tools (parallel transport, holography) can be extended to other interpretability problems such as attention geometry and gradient manifold shapes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Highly original construction of differential geometry perspective and $\mathfrak{so}(n)$-valued forms.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Chess validation is thorough and quantitative, but single-domain limits generalizability.
- Writing Quality: ⭐⭐⭐⭐☆ Mathematically rigorous, but some derivations are challenging for readers without a differential geometry background.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for understanding LLM internal dynamics via geometry, with far-reaching impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Cylindrical Representation Hypothesis for Language Model Steering](the_cylindrical_representation_hypothesis_for_language_model_steering.md)
- [\[ICLR 2026\] Internal Planning in Language Models: Characterizing Horizon and Branch Awareness](../../ICLR2026/interpretability/internal_planning_in_language_models_characterizing_horizon_and_branch_awareness.md)
- [\[ICLR 2026\] Hidden Breakthroughs in Language Model Training](../../ICLR2026/interpretability/hidden_breakthroughs_in_language_model_training.md)
- [\[ICLR 2026\] Evolution of Concepts in Language Model Pre-Training](../../ICLR2026/interpretability/evolution_of_concepts_in_language_model_pre-training.md)
- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)

</div>

<!-- RELATED:END -->
