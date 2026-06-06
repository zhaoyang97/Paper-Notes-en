---
title: >-
  [Paper Note] A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State
description: >-
  [ICML 2026][LLM/NLP][Sampling Error] This paper characterizes the information loss introduced by sampling from high-entropy distributions in GPT-style LLMs from a differential geometric perspective. It constructs $\mathf…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Sampling Error"
  - "Differential Geometry"
  - "Parallel Transport"
  - "World Models"
  - "Chess"
date: 2026-05-08
content_hash: 7bc9ea6e0b883973
---

# A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State

**Conference**: ICML 2026  
**arXiv**: [2605.04899](https://arxiv.org/abs/2605.04899)  
**Code**: See supplementary material  
**Area**: LLM Interpretability / NLP  
**Keywords**: Sampling Error, Differential Geometry, Parallel Transport, World Models, Chess

## TL;DR
This paper characterizes the information loss introduced by sampling from high-entropy distributions in GPT-style LLMs from a differential geometric perspective. It constructs $\mathfrak{so}(n)$-valued 1-forms and parallel transport operators, demonstrating through chess probing experiments that this geometric rotation is highly aligned with the world vectors learned by the model.

## Background & Motivation

**Background**: Autoregressive LLMs generate tokens through greedy or stochastic decoding. When the output distribution is concentrated (high confidence), sampling error is negligible; however, when the distribution is diffuse ("blurry points"), different samplings cause subsequent trajectories to diverge significantly.

**Limitations of Prior Work**: While the sensitivity of LLMs to single-token perturbations is widely recognized, the internal structure of this sensitivity remains uncharacterized—is it merely chaotic exponential divergence, or does it reflect the internal geometric properties of the model?

**Key Challenge**: The relationship between internal states $z_t \in \mathbb{R}^n$ and the output distribution (discrete probabilities) via projection, softmax, and sampling is highly non-linear. The challenge lies in describing this "internal → external → internal perturbation" coupling within a unified geometric framework.

**Goal**: To establish a verifiable relationship between the geometric properties of "blurry points" inside the model and the world vectors obtained through probing.

**Key Insight**: Modeling sampling uncertainty as a directed measure on a manifold using the language of differential geometry. Uncertainty is "injected" into geometric actions via wedge products and parallel transport.

**Core Idea**: Parameterize blurring intensity as a triple wedge product $A(z_t) = 4 z_t \wedge (p_1 v_1) \wedge (p_2 v_2)$, then contract it with tangent vectors to obtain an $\mathfrak{so}(n)$-valued 1-form. Its parallel transport imposes testable rotations in the hidden state space that couple with the directions of world vectors.

## Method

### Overall Architecture

A three-layer structure: (1) Geometric modeling: wedge products and $\mathfrak{so}(n)$ structures; (2) Parallel transport and holographic measurement; (3) Experimental validation: confirming coupling relationships via directional clustering of world vectors in chess tasks.

### Key Designs

1.  **Upgrading Wedge Products to $\mathfrak{so}(n)$-valued 1-forms**:
    - **Function**: Transforms the scalar magnitude of blurring into a directed geometric object acting on hidden states.
    - **Mechanism**: Define $A(z_t) = 4 z_t \wedge (p_1 v_1) \wedge (p_2 v_2)$, where the Frobenius norm $\|A\|_F$ is proportional to the degree to which $z_t$ resides within the plane spanned by the top-two token embeddings. Contracting with a tangent vector $\mu$ yields $A_\mu(z_t) = 4 p_1 p_2 \big(-(\mu\cdot v_1)(z_t \wedge v_2) + (\mu \cdot v_2)(z_t \wedge v_1)\big)$, an element of the $\mathfrak{so}(n)$ Lie algebra (a rotation generator).
    - **Design Motivation**: Scalar magnitudes cannot encode direction; anti-symmetric tensors naturally introduce rotational effects that carry directional uncertainty information.

2.  **Probability Charge and Parallel Transport Operators**:
    - **Function**: Parameterizes the intensity of geometric action and calculates cumulative rotation through closed paths.
    - **Mechanism**: $4 p_1 p_2$ is termed the "probability charge"—analogous to charge coupling in electromagnetism; it tends to zero when $p_2 \to 0$ (high confidence) and reaches a maximum of $1$ at $p_1 = p_2 = 0.5$ (maximum uncertainty). The parallel transport operator along a curve $\gamma$ is $U_\gamma = P\exp\big(-\int_0^1 A_{\dot\gamma(s)}(\gamma(s))\,ds\big)$ (where $P$ denotes path ordering), measuring the rotation of hidden states in uncertain regions.
    - **Design Motivation**: Connects to intuitions from gauge theory, providing physical correspondence for sampling uncertainty—using "geometric curvature" as a carrier of information loss.

3.  **Holography and Finite-Step Closed Path Measurement**:
    - **Function**: Measures local curvature via closed quadrilateral paths around a point, overcoming the inability to continuously translate hidden states.
    - **Mechanism**: Since models generate actual trajectories rather than arbitrary paths, point-to-point parallel transport is impossible. Instead, a holographic operator $H_{z_t}$ is constructed using $\epsilon$-sized "clovers" (four combined small squares to cancel coordinate bias), reflecting the local curvature $R = \partial_\mu A_\nu - \partial_\nu A_\mu - [A_\nu, A_\mu]$.
    - **Design Motivation**: Borrows standard techniques from lattice gauge theory to extract local geometric information under restricted observations.

### Loss & Training
This work does not train new models. Probes are trained on frozen LLM hidden states using linear classifiers to predict 737 piece positions.

## Key Experimental Results

| Setting | Task | Model | Key Metric | Result |
| :--- | :--- | :--- | :--- | :--- |
| Chess World Model | 737 piece pos. classification | Qwen 32B | Avg. Accuracy | 81.2%–100% |
| Chess World Model | Same as above | Mistral 24B | Avg. Accuracy | 76.0%–98.9% |
| Blurring Sensitivity | Move selection difference | Qwen 32B | Pos. Eval Change | 4.5±1.5 log-cp |
| Geo-Semantic Coupling | Rotation Dir vs World Vector | Qwen 32B | Mean $\|\cos\|$ | Top pieces >0.7, Overall >0.5 |
| Geo-Semantic Coupling | Board partition clustering | Qwen 32B | Cluster Purity | Board quadrants >85% |

### Key Findings
- **Coupling of World Vectors and Geometric Rotation**: At all branch points, the rotation direction and the corresponding world vector show a mean $|\cos| > 0.5$ (top pieces $> 0.7$), significantly higher than the random baseline of $\sim 0.07$.
- **Geometric Mapping of Piece Importance**: High-value pieces (e.g., Queen) correspond to the strongest blurring signals and largest rotation magnitudes; there is a positive correlation between value and geometric intensity.
- **Model Capability vs. Geometric Signal**: Mistral, which has lower probe accuracy, also shows significantly weakened geometric signals, suggesting this is a feature of structures genuinely learned by LLMs.
- **Sampling Error is Not Pure Chaos**: The direction of divergence is coupled with the world structure, indicating this is not directionless chaos but a geometric projection of the model's internal representation.

## Highlights & Insights
- **Mathematical Elegance**: Precisely characterizes sampling uncertainty using Lie algebra and parallel transport, utilizing tools from gauge theory with high compatibility.
- **New Dimension of Interpretability**: Links model vulnerability to the geometry of world models, providing a deeper explanation than mere "chaotic divergence."
- **Clever Task Selection**: Chess provides an explicit world model (deterministic legal positions), facilitating objective and quantitative verification of coupling relationships.
- **Cross-Model Consistency**: The coupling phenomenon appears across different model families, suggesting it may be a universal property of LLMs.

## Limitations & Future Work
- Validated only in the single domain of chess; it remains unclear if this replicates in natural language or open domains.
- Probe training data faces human-balancing issues; its relevance to real text distributions is limited.
- Results on Mistral are significantly weaker, indicating sensitivity to model choice and requiring verification across more families.
- There is a lack of rigorous proof regarding why geometric rotation must align with world vectors; it remains an empirical observation.
- Future work: Extend to natural language tasks; design new decoding strategies based on geometric insights; theoretically analyze how Lie group structures emerge from training.

## Related Work & Insights
- **vs. Traditional Sensitivity Analysis**: Previous works only described exponential divergence; this paper further aligns the direction and pattern of divergence with world models.
- **vs. Probing/World Model Works (OthelloGPT, etc.)**: The latter verify the existence of world models; this paper explains why world models couple with sampling uncertainty.
- **Insights**: Differential geometry tools (parallel transport, holography) can be extended to other interpretability problems such as attention geometry or gradient flow shapes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The construction of the $\mathfrak{so}(n)$-valued form and the differential geometric perspective are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Validation in chess is thorough and quantitative, though the single domain limits generalizable conclusions.
- Writing Quality: ⭐⭐⭐⭐☆ Mathematically rigorous, though some derivations pose a high barrier for readers without a differential geometry background.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for understanding LLM internal dynamics through geometry, with far-reaching implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Escaping Mode Collapse in LLM Generation via Geometric Regulation](escaping_mode_collapse_in_llm_generation_via_geometric_regulation.md)
- [\[ICML 2026\] Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions](scheduling_llm_inference_with_uncertainty-aware_output_length_predictions.md)
- [\[ICML 2026\] The Cylindrical Representation Hypothesis for Language Model Steering](the_cylindrical_representation_hypothesis_for_language_model_steering.md)
- [\[ACL 2026\] Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context](../../ACL2026/llm_nlp/text-to-distribution_prediction_with_quantile_tokens_and_neighbor_context.md)
- [\[NeurIPS 2025\] Opinion Maximization in Social Networks by Modifying Internal Opinions](../../NeurIPS2025/llm_nlp/opinion_maximization_in_social_networks_by_modifying_internal_opinions.md)

</div>

<!-- RELATED:END -->
