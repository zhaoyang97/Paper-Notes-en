---
title: >-
  [Paper Note] Jacobian Scopes: Token-Level Causal Attributions in LLMs
description: >-
  [ACL 2026][Interpretability][Jacobian] The authors propose Jacobian Scopes—a unified framework using "the projection of the input token embedding to the final hidden state Jacobian onto a specific vector" as the token at…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Jacobian"
  - "vector-Jacobian product"
  - "Fisher information"
  - "effective temperature"
  - "token attribution"
date: 2026-05-08
content_hash: 63c359f4aa727a5c
---

# Jacobian Scopes: Token-Level Causal Attributions in LLMs

**Conference**: ACL 2026  
**arXiv**: [2601.16407](https://arxiv.org/abs/2601.16407)  
**Code**: https://huggingface.co/spaces/Typony/JacobianScopes (Online demo)  
**Area**: Interpretability / Causal Attribution / LLM Internal Mechanisms  
**Keywords**: Jacobian, vector-Jacobian product, Fisher information, effective temperature, token attribution

## TL;DR
The authors propose Jacobian Scopes—a unified framework using "the projection of the input token embedding to the final hidden state Jacobian onto a specific vector" as the token attribution strength. Accompanied by three scopes (Semantic / Fisher / Temperature) to explain how a "target logit / full prediction distribution / model confidence" is driven by each input token, the framework requires only a single backpropagation. On AOPC metrics, it performs on par with Input×Gradient and significantly outperforms Integrated Gradients.

## Background & Motivation
**Background**: The mainstream paths for LLM interpretability include attention visualization, activation patching, circuit tracing, or Sparse Autoencoders (SAE / Gemma Scope). Gradient-based attributions like Integrated Gradients, Input × Gradient, and SmoothGrad also exist. These methods have diverse objective functions and geometric assumptions, lacking a unified framework to define what exactly is being explained.

**Limitations of Prior Work**: (1) Gradient attribution methods often conflate "how a specific logit arises" with "how the entire prediction distribution arises," yielding weak explanatory power for tasks with non-unique predictions like translation. (2) Methods like Integrated Gradients (IG) require multiple integration steps (K forward + backward passes), which is computationally expensive. (3) Attention visualization only explains structural information and remains distant from the final prediction causal chain. (4) Almost no mainstream attribution method can explain "model confidence (temperature)," a dimension critical for ICL time series forecasting.

**Key Challenge**: Attribution methods require an **explicit explanandum**—is it the logit, the distribution shape, or the distribution width? Different objects correspond to different geometric directions $\bm{v}$, but existing works either hardcode this to a logit (IG) or use heuristics (attention), lacking a unified primitive that computes attribution for a "specified direction."

**Goal**: Construct a mathematically clear, single-backprop, and geometrically interpretable attribution primitive, and define three typical explananda (Semantic / Distribution / Confidence) under this primitive, each mapping to an easily calculated vector $\bm{v}$.

**Key Insight**: Observations show that all questions regarding "how input token $\bm{x}_t$ affects a certain output property" can be written as $\|\bm{v}^\intercal \bm{J}_t\|_2$, where $\bm{J}_t := \partial \bm{y} / \partial \bm{x}_t$ is the Jacobian from input to final hidden state, and $\bm{v}$ is the "direction" of interest. This collapses the family of attribution problems into a design choice of selecting $\bm{v}$.

**Core Idea**: Use the vector-Jacobian product (VJP) $\bm{v}^\intercal \bm{J}_t$ as a unified token attribution primitive. By varying $\bm{v}$ (e.g., unembed row / Fisher principal eigenvector / normalized hidden state), the authors derive Semantic / Fisher / Temperature Scopes, covering logit, full distribution, and confidence explanations with only one backpropagation pass each.

## Method

### Overall Architecture
The LLM is viewed as a function $f: \bm{X}_{1:T} \mapsto \bm{y} \in \mathbb{R}^{d_{\text{model}}}$, where $\bm{y}$ is the post-LN hidden state of the final layer; logit $\bm{z} = \bm{W}\bm{y}$, and prediction distribution $\bm{p} = \mathrm{softmax}(\bm{z})$. For each input position $t$, the input-to-output Jacobian is defined as $\bm{J}_t = \partial \bm{y} / \partial \bm{x}_t \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$. Computing $\bm{J}_t$ directly would require $d_{\text{model}}$ backprops. Instead, the authors use the VJP $\bm{v}^\intercal \bm{J}_t$—by constructing a scalar loss $\mathcal{L}(\bm{X}_{1:T}) = \bm{v}^\intercal \bm{y}$, a single backprop yields $\partial \mathcal{L} / \partial \bm{x}_t = \bm{v}^\intercal \bm{J}_t$ for all positions $t$. The unified attribution formula is $\mathrm{Influence}_t := \|\bm{v}^\intercal \bm{J}_t\|_2$, representing the "maximum displacement in direction $\bm{v}$ caused by an $\varepsilon$-norm perturbation on $\bm{x}_t$." Different Scopes are obtained simply by changing $\bm{v}$.

### Key Designs

1.  **Semantic Scope: Explaining the logit of a target token**:
    *   **Function**: Answers "why the model predicted 'truthful' over alternatives," locating input tokens that drive the target token's logit.
    *   **Mechanism**: Set $\bm{v} = \bm{w}_{\text{target}}$ (the row in the unembedding matrix corresponding to the target token). Thus, $\mathcal{L}_{\text{semantic}} = \bm{w}_{\text{target}}^\intercal \bm{y} = z_{\text{target}}$ is the logit of the target token. The attribution score is $\mathrm{Influence}_t^{\text{Sem}} = \|\bm{w}_{\text{target}}^\intercal \bm{J}_t\|_2$.
    *   **Design Motivation**: While Input × Grad and IG do this implicitly, Semantic Scope formalizes it as a VJP special case with a clear "target direction." It is ideal for scenarios with an explicit target word, such as uncovering the semantic reversal chain from "deceive" → "truthful" in LLaMA or revealing implicit political biases.

2.  **Fisher Scope: Explaining the entire prediction distribution**:
    *   **Function**: In scenarios with non-unique predictions (e.g., translation, where multiple synonyms are correct), it identifies which inputs most significantly alter the entire distribution shape.
    *   **Mechanism**: Use the Fisher Information Matrix (FIM) $\bm{F} = \bm{W}^\intercal (\mathrm{diag}(\bm{p}) - \bm{p}\bm{p}^\intercal) \bm{W}$ from information geometry. Perform eigendecomposition $\bm{F} = \bm{U}\bm{\Lambda}\bm{U}^\intercal$ and take the principal Fisher direction $\bm{u}_1$ as $\bm{v}$. The score is $\mathrm{Influence}_t^{\text{Fisher}} = \|\bm{u}_1^\intercal \bm{J}_t\|_2$. Theoretically, this is a rank-1 approximation of the total mutual information between $\bm{p}$ and $\bm{x}_t$.
    *   **Design Motivation**: When the target is not a specific token but rather a "category of tokens," logit attribution loses semantic cluster information. Fisher Scope uses the local metric of KL divergence (FIM) to automatically find the directions the LLM is most sensitive to in the output space. In translation tasks, it demonstrates LLaMA performing "word-level alignment + phrase-level cross-token reasoning."

3.  **Temperature Scope: Explaining model confidence**:
    *   **Function**: In scenarios where prediction distributions resemble Gaussian peaks (e.g., ICL time series forecasting), it identifies which inputs control the peak width (i.e., model certainty).
    *   **Mechanism**: Decompose the hidden state into norm and direction $\bm{y} = \|\bm{y}\|_2 \hat{\bm{y}}$. Then $\bm{z} = \beta_{\text{eff}} \hat{\bm{z}}$, where $\beta_{\text{eff}} = \|\bm{y}\|_2$ is the effective inverse temperature. The authors prove that when the softmax output is approximately Gaussian, $\beta_{\text{eff}}^{-1}$ is proportional to the variance. By setting $\bm{v} = \hat{\bm{y}}$, the score is $\mathrm{Influence}_t^{\text{Temp}} = \|\hat{\bm{y}}^\intercal \bm{J}_t\|_2$.
    *   **Design Motivation**: Previous methods did not explicitly explain the "sources of model confidence." ICL time series forecasting hinges on which historical segment determines future uncertainty. Temperature Scope answers this and validates the "context parroting" hypothesis.

### Loss & Training
This is a **purely post-hoc analysis** method; it involves no model training. Three Scopes correspond to specific scalar losses:

| Scope | $\bm{v}$ | Loss $\mathcal{L}$ |
|-------|---------|-------------------|
| Semantic | $\bm{w}_{\text{target}}$ | $z_{\text{target}}$ |
| Fisher | $\bm{u}_1$ (FIM principal eigenvector) | $\bm{u}_1^\intercal \bm{y}$ |
| Temperature | $\hat{\bm{y}}$ | $\beta_{\text{eff}} = \|\bm{y}\|_2$ |

**Implementation Details**: All parameter gradients are disabled; the single backprop accumulates only on input embeddings. The time cost is equivalent to one backprop pass (e.g., 0.027s vs. 0.069s forward pass on an RTX A4000).

## Key Experimental Results

### Main Results: AOPC Attribution Quality Comparison (LLaMA-3.2 3B)
AOPC (Area Over Perturbation Curve): The drop in target token log-prob after zeroing out the top-k% highest attribution tokens (more negative = more accurate attribution).

| Method | LAMBADA | IWSLT2017 DE→EN |
|--------|---------|-----------------|
| Random | $-0.23 \pm 0.01$ | $-0.19 \pm 0.01$ |
| Integrated Gradients | $-0.67 \pm 0.01$ | $-0.58 \pm 0.01$ |
| Input × Gradient | $-1.12 \pm 0.01$ | $-0.77 \pm 0.01$ |
| **Semantic Scope (Ours)** | $-1.16 \pm 0.01$ | $-0.78 \pm 0.01$ |
| **Temperature Scope (Ours)** | $\bm{-1.17 \pm 0.01}$ | $-0.76 \pm 0.01$ |
| **Fisher Scope (Ours)** | $\bm{-1.17 \pm 0.01}$ | $\bm{-0.80 \pm 0.01}$ |

### Ablation Study: Cross-model Scales + Relative Advantages of Scopes

| Evaluation Scenario | Semantic | Fisher | Temperature | Key Explanation |
|---------|----------|--------|------------|---------|
| Semantic/Bias Visualization | ✅ Best | – | – | Target word clear; Semantic gives precise tokens |
| Translation (Non-unique) | Blurry | ✅ Best | – | Fisher captures word-level + phrase-level alignment |
| Time Series ICL (Lorenz) | – | – | ✅ Best | Temperature reveals "history-match" patterns |
| Time Series ICL (Brownian) | – | – | ✅ Best | Temperature reveals "forgetting early context" |
| LLaMA-3.2 1B / 3B / Qwen2.5 1.5B / 7B | Exceeds IG | Exceeds IG | Exceeds IG | Robust across model scales/series |

### Key Findings
- **Three Scopes are complementary**: In ICL numerical prediction tasks, Semantic/Fisher Scopes give blurry attributions, while Temperature Scope accurately points to "which historical segment is being copied."
- **VJP single backprop is sufficient**: Attribution overhead is comparable to a single backprop, significantly faster than IG's K-step integration, while achieving better AOPC.
- **first-order sensitivity ≈ counterfactual relevance**: AOPC is a real intervention metric (zeroing tokens to see log-prob drop); the alignment with Jacobian indicates that first-order linear approximation is sufficient at the token level for LLMs.
- **Temperature Scope validates context parroting**: LLaMA's behavior in chaotic systems like Lorenz confirms it performs "nearest-neighbor search" in delayed-embedding space, providing direct attribution evidence for the hypothesis by Zhang & Gilpin (2025).
- **Attention sink interference**: Early tokens in Brownian experiments show high attribution due to the attention sink phenomenon, noted as a caveat.

## Highlights & Insights
- **VJP-as-attribution-primitive**: Treating the choice of direction $\bm{v}$ as the fundamental design makes this an elegant and extensible framework. Future explananda (e.g., SAE features, specific circuits) can be easily integrated by defining a corresponding $\bm{v}$.
- **Fisher Scope Information Geometry**: Using the FIM's principal direction to answer "which input changes the distribution most" is the first work to explicitly link distribution geometry with token attribution.
- **Temperature Scope for ICL**: For the first time, input attribution for "model confidence" is provided, translating ICL explanation from "pattern discovery" to "mechanistic causality."
- **Minimal Implementation + Interactive Demo**: One backprop + HuggingFace Spaces demo allows researchers to immediately visualize attributions on their own prompts with high reusability.

## Limitations & Future Work
- **First-order Linearity**: Jacobian only captures first-order causal relationships near input embeddings, lacking explanatory power for non-linear causal chains across many layers.
- **Architecture Blindness**: The method focuses on input-output relations without looking inside the transformer, thus it cannot identify which layer or attention head is responsible.
- **Backprop Dependency**: Requires a backward pass unlike pure forward methods (SAE activation), though the overhead is minimal.
- **Architecture Artifacts**: Phenomena like attention sinks can contaminate attributions, requiring correction based on architectural knowledge.
- **Future Directions**: Exploring higher-order spectral structures of the Fisher matrix; combining with SAE for "feature-level Scope"; extending to joint attribution of multiple tokens.

## Related Work & Insights
- **vs. Integrated Gradients (Sundararajan 2017)**: IG integrates along an interpolation path to satisfy axioms but requires K passes. Jacobian Scope uses local first-order but single pass, yielding better AOPC, suggesting "satisfying axioms ≠ better empirical accuracy."
- **vs. Input × Gradient (Shrikumar 2017)**: This work is a strict generalization—I×G is a special case where $\bm{v}$ is aligned with the input direction.
- **vs. activation patching / circuit tracing**: These are interventionist methods; Jacobian Scope is an observational method for token-level causality, serving as a complement.
- **vs. context parroting (Zhang & Gilpin 2025)**: Temperature Scope provides the first direct attribution evidence for their hypothesis.

## Rating
- Novelty: ⭐⭐⭐⭐ VJP as a unified primitive + two new Scopes (Fisher/Temperature) is an elegant geometric reframing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive AOPC tests across datasets and scales, though lacks 70B+ model validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent flow from formulas to geometric interpretation and case studies.
- Value: ⭐⭐⭐⭐ Provides plug-and-play tools and a framework of long-term significance for the interpretability community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models](meter_evaluating_multi-level_contextual_causal_reasoning_in_large_language_model.md)
- [\[ICML 2026\] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](../../ICML2026/interpretability/towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [\[NeurIPS 2025\] Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning](../../NeurIPS2025/interpretability/learning_to_focus_causal_attention_distillation_via_gradient-guided_token_prunin.md)
- [\[ACL 2026\] Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs](flattery_in_motion_benchmarking_and_analyzing_sycophancy_in_video-llms.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)

</div>

<!-- RELATED:END -->
