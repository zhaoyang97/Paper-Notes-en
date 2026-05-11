---
title: >-
  [Paper Note] As Language Models Scale, Low-order Linear Depth Dynamics Emerge
description: >-
  [CVPR 2026][Social Computing][activation intervention] This paper treats the layer depth of a Transformer as a discrete-time system, demonstrating that the inter-layer propagation and intervention response of GPT-2 can b…
tags:
  - "CVPR 2026"
  - "Social Computing"
  - "activation intervention"
  - "local linearization"
  - "depth dynamics"
  - "system identification"
  - "model scaling"
date: 2026-05-08
content_hash: 03433e6b4be69a64
---

# As Language Models Scale, Low-order Linear Depth Dynamics Emerge

**Conference**: CVPR 2026
**arXiv**: [2603.12541](https://arxiv.org/abs/2603.12541)
**Code**: None
**Area**: Social Computing
**Keywords**: activation intervention, local linearization, depth dynamics, system identification, model scaling

## TL;DR
This paper treats the layer depth of a Transformer as a discrete-time system, demonstrating that the inter-layer propagation and intervention response of GPT-2 can be approximated near a given context by a 32-dimensional low-order linear state-space surrogate. Notably, as model scale increases, this surrogate becomes more accurate. The framework further enables the derivation of energy-efficient multi-layer intervention strategies that outperform heuristic injection baselines.

## Background & Motivation
A conspicuous gap exists in current analysis methods for large language models: one line of work examines whether linearly separable "concept directions" exist within representations, while another investigates the effectiveness of activation steering, yet few studies model how a concept direction injected at a given layer actually propagates along depth and ultimately affects the output.

More concretely, it is already known that adding a direction vector to the residual stream can alter attributes such as sentiment, toxicity, and hate speech, yet prevailing practices remain largely empirical:

1. Performing a layer-by-layer sweep to identify the most effective layer.
2. Arbitrarily selecting the last layer, averaging injections across all layers, or choosing some intermediate layer.
3. Lacking a unified explanation for why certain layers are effective or why distributing intervention across multiple layers reduces "intervention energy."

The paper argues that what is truly missing is a *local, predictable, and controllable* perspective on depth dynamics. The authors treat the layer depth of a causal Transformer as discrete time, regard the hidden state of the last non-padding token as the system state, and pose a question rooted in systems control:

If a small perturbation is added to the residual stream at layer $k$, how does it affect the final concept readout after passing through all subsequent blocks?

This question matters for three reasons:

**Mechanistic understanding**: Without characterizing how perturbations propagate along depth, analysis remains confined to the static representational level of "a certain direction is useful."

**Intervention design**: If the gain curve across layers can be predicted, brute-force layer sweeps become unnecessary.

**Scaling laws**: Larger models are generally assumed to be more complex, but the authors hypothesize that larger models may in fact be locally more "regular" and therefore more amenable to compression by low-dimensional dynamical surrogates.

The authors' core observation is compelling: although LLMs are globally high-dimensional nonlinear systems, the local inter-layer propagation of the last token near a fixed prompt need not require an equally high-dimensional complex model to explain it. In other words, the dynamics that truly govern steering responses may be active only within a very small reachable subspace.

The paper accordingly addresses three coherent questions:

1. Can a low-order linear layer-variant surrogate be identified for a given context?
2. Can this surrogate accurately predict the steering sensitivity curve across layers?
3. Does this identifiability systematically improve as model scale increases?

The answers are affirmative, and the third point constitutes the paper's most valuable contribution: as models grow larger, the local low-order linear surrogate does not degrade—it becomes more accurate.

## Method

The methodological thread of this paper is straightforward: activation steering is reformulated as a local system identification and optimal control problem.

### Overall Architecture

The complete pipeline can be summarized in five steps:

1. Run the original model once on a given prompt to obtain the trajectory of the last token's hidden states along depth, $x_\ell(p)$.
2. Fix the hidden states of all non-final tokens, allow only the last token's state to vary by a small amount before the next block, and thereby obtain a prompt-conditioned, frozen-context mapping $x_{\ell+1} = f_\ell(x_\ell; p)$.
3. Locally linearize $f_\ell$ along this operating trajectory to obtain the per-layer Jacobian $A_\ell(p)$.
4. Combine the concept direction $v_\ell$ with a Krylov-style reachable subspace to construct a low-dimensional basis $P_\ell$, and project the high-dimensional dynamics into a low-order linear layer-variant (LLV) model.
5. Use the LLV surrogate to predict the gain curve for single-layer injection, and further derive the minimum-energy multi-layer actuation schedule that achieves a target output change.

This approach does not aim to fit the model's global behavior; rather, it fits a local but critical object: given the current context, how does a small perturbation to the last token propagate through subsequent layers to alter the final concept readout.

### Key Designs

1. **Treating depth as time and the last token as the system state**

   - *Function*: Define $x_\ell(p) = h_\ell(p)[t(p), :]$, where $t(p)$ is the position of the last non-padding token.
   - *Mechanism*: Rather than modeling the full hidden states of the entire sequence, the authors focus on the last token representation, which most directly influences the final readout, and treat the sequence of Transformer blocks as temporal evolution.
   - *Design Motivation*: Since the goal is to understand how activation steering affects final output, the last token state is closest to the decision end; this compression also allows "propagation along depth" to be described directly in state-space language.

2. **Freezing context and modeling only the local last-token response**

   - *Function*: For a fixed prompt, the hidden states of all non-final tokens are held constant; only the last token is perturbed, and the effect of the next block is observed.
   - *Mechanism*: This yields the prompt-conditioned mapping $x_{\ell+1} = f_\ell(x_\ell; p)$, which absorbs complex cross-token interactions into the "frozen context" and retains the local dynamics most relevant to steering.
   - *Design Motivation*: Without freezing the context, the full system state would be too large to identify; by doing so, the problem shifts from "globally explaining the Transformer" to "locally explaining depth propagation under the current context," making it tractable.

3. **Modeling the external input along the concept direction**

   - *Function*: The concept direction $v_\ell$ is estimated at each layer, and the main experiments consider only additive injections $u_\ell v_\ell$ along this direction.
   - *Mechanism*: Concept directions are derived from normalized class-mean differences on an independent concept split, so the "input" of steering is also defined in a data-driven manner.
   - *Design Motivation*: Many activation steering works assume the direction exists but do not model how it propagates. Here, direction estimation and dynamical propagation are unified, closing the loop among input, state, and output within a single framework.

4. **Constructing the concept-anchored low-dimensional basis $P_\ell$**

   - *Function*: The first column of the per-layer dimensionality reduction basis is fixed as the concept direction $v_\ell$; the remaining columns are constructed via a reachability-informed Krylov complement.
   - *Mechanism*: This is neither arbitrary PCA nor a random complement, but a basis built by recursively expanding directions that are genuinely reachable under steering, starting from the action of the average Jacobian.
   - *Design Motivation*: If the low-dimensional basis does not cover the subspace actually excited by the control input, even a small surrogate will miss critical propagation patterns. The Krylov-style complement ensures that the finite dimensional budget is spent on directions that steering actually uses.

5. **Obtaining the layer-variant linear surrogate (LLV)**

   - *Function*: After projection, the surrogate takes the form
     $$r_{\ell+1} \approx \bar{A}_\ell(p)\, r_\ell + \bar{B}_\ell(p)\, u_\ell, \quad r_\ell = P_\ell^\top \delta x_\ell.$$
   - *Mechanism*: Although the original model is nonlinear, its response near the operating trajectory can be approximated by a set of layer-varying linear matrices; this better reflects the fact that different Transformer layers serve different functions.
   - *Design Motivation*: The authors do not claim that "the Transformer is essentially linear"; rather, they emphasize that a *locally linear, layer-varying* approximation is sufficiently accurate. This more modest claim is both technically defensible and consistent with actual network structure.

6. **Connecting analysis and control via predicted gain**

   - *Function*: For a single-layer injection at layer $k$, the predicted final concept gain is
     $$g_k^{\text{pred}} \approx C\,\Phi(k+1, L)\,\bar{B}_k.$$
   - *Mechanism*: Once the reduced transition product $\Phi$ is available, the sensitivity of each layer can be read directly from the surrogate model without empirical per-layer measurement.
   - *Design Motivation*: This step transforms an "explanatory model" into an "actionable model." Rather than retrospectively explaining which layer is best, the framework predicts it in advance.

### Loss & Training

Strictly speaking, this work does not train a new model; it performs local identification on frozen pretrained GPT-2 variants, so there is no training loss in the conventional sense. The "learning" process is embodied in three components:

1. **Concept direction estimation**: The per-layer concept direction $v_\ell$ is computed as the normalized difference of class-conditional means of the last token representations on the concept split.
2. **Local Jacobian estimation**: Jacobian-vector products (JVPs) are used as the primary method; central finite differences serve as a fallback when necessary.
3. **Low-dimensional system identification**: $\bar{A}_\ell(p)$ and $\bar{B}_\ell(p)$ are obtained in the reduced basis, with reduced order fixed at $d = 32$ in the main experiments.

Several key experimental settings are worth noting:

1. For GPT-2-large main results, the concept batch size is 32 and the held-out batch size is 64.
2. The operating split is used for local dynamics identification; the evaluation split is used solely for assessing gain and control effectiveness—these two splits are strictly separated.
3. The Krylov complement size is 31, yielding a total reduced dimension of 32.
4. The gain evaluation magnitude in the main figures is primarily $\epsilon = 0.1$, with robustness verified over a wider range.

### Why This Approach Is Valid

The strongest aspect of this paper is not the complexity of its algorithm, but the principled decomposition of the problem.

First, the authors focus exclusively on local responses rather than attempting to explain the entirety of LLM computation, providing clear justification for linearization.

Second, the dimensionality reduction is not arbitrary; it is anchored to the concept direction and reachable subspace, so the reduced model preserves the dynamics most relevant to steering.

Finally, the value of the surrogate is assessed against two verifiable objectives:

1. Whether it can predict the full layerwise gain curve.
2. Whether it can yield a superior multi-layer actuation schedule.

That is, the paper does not offer an abstract "linearization-based explanation" in isolation; it requires that the explanation be empirically validated against the original nonlinear Transformer.

## Key Experimental Results

The paper evaluates on 10 binary NLP classification tasks: Amazon Polarity, Yelp Polarity, SST-2, IMDB, BoolQ, binarized MNLI, Civil Comments Toxicity, TweetEval-Irony, TweetEval-Hate, and TweetEval-Offensive.

Main results focus on GPT-2-large, with scaling behavior studied across GPT-2, GPT-2-medium, and GPT-2-large.

### Main Results

The most important quantitative results are not conventional classification accuracy, but the agreement between the predicted gain curve and the empirical full-model gain curve, as well as the energy efficiency of optimal control relative to heuristic strategies.

| Setting | Metric | Ours | Prev. SOTA / Baseline | Conclusion |
|---|---|---|---|---|
| GPT-2-large, $d=32$ | Layerwise gain prediction Spearman | 0.99 or 1.00 across reported datasets | Empirical layer sweep | Low-order LLV nearly perfectly reproduces the full gain curve |
| GPT-2-large, $d=32$ | Gain curve shape | Nearly overlaps the empirical curve | Coarse analysis targeting only the best layer | Surrogate predicts the complete response shape, not just the optimal layer |
| GPT-2 family scaling | Average Spearman | GPT-2: ~0.77; GPT-2-medium: ~0.81; GPT-2-large: 0.995 | Same reduced order = 32 | Larger models yield more identifiable low-order linear surrogates |
| GPT-2 family scaling | Average Pearson | GPT-2: ~0.68; GPT-2-medium: ~0.74; GPT-2-large: 0.997 | Same as above | Improvement in identifiability extends beyond rank correlation to magnitude agreement |
| GPT-2-large control | Energy required to achieve a target concept shift | LLV-optimal is lowest or tied for lowest | Uniform-all, last-layer-only, middle-only, early-only, random-layer | Control schedules derived from the linear surrogate significantly outperform all heuristics |

### Ablation Study

Although the paper does not adopt a typical modular stacking architecture, several key analyses serve as ablations:

| Configuration | Change in Key Metric | Remarks |
|---|---|---|
| Full LLV, $d=32$, Krylov complement | Best | Main setting; predicted curve closely matches empirical measurements |
| Reduced order decreased | Consistency degrades then saturates | Confirms that steering-relevant dynamics are genuinely low-dimensional, but underfitting occurs when order is too small |
| Random complement instead of Krylov complement | Worse prediction | Confirms that reachability-guided complement better captures true propagation subspace than random construction |
| Varying $\epsilon$ | Stable over a wide range | Empirical finite differences do not depend on a finely tuned operating point |
| Heuristic multi-layer / single-layer injection | Substantially higher energy | Demonstrates that the surrogate serves not only as an interpreter but also as a guide for control design |

### Key Findings

1. **The optimal intervention layer is not universal**: The shape of gain curves varies markedly across tasks—some exhibit monotonic increase toward later layers, others plateau in the mid-to-late region—making "always inject at the last layer" an unreliable strategy.
2. **The full response curve is what is being fitted**: The paper consistently emphasizes that the value of the surrogate lies not in identifying a single top-1 layer, but in accurately characterizing the complete sensitivity profile from early to late layers.
3. **The scaling law is counterintuitive**: It is commonly assumed that larger models are harder to interpret, yet here larger models are more readily described by low-order local linear systems.
4. **Control gains are substantial**: Compared to uniform-all, LLV-optimal typically reduces energy by approximately 2–5×; compared to last-layer-only, improvements often span one or more orders of magnitude.

## Highlights & Insights

1. This paper elevates activation steering from an "empirical heuristic" to a "local system control problem." The value of this reformulation lies in placing direction, propagation, and intervention allocation within a unified mathematical framework.
2. The finding that "larger models are more readily explained by low-order local surrogates" is striking. It suggests that scale brings not only capability gains but also increased regularity and compressibility of local dynamics.
3. The concept-anchored Krylov basis is an elegant design: the first column is forced to align with the concept direction, while the remaining dimensions are expanded around reachability, balancing semantic relevance with dynamical fidelity.
4. The paper places strong emphasis on the separation of the operating split from the evaluation split, preventing the same prompts from being used for both identification and validation, thereby lending greater credibility to the claim that "prediction is effective."
5. This work provides a useful intermediate abstraction: it is neither as static as pure probing nor as opaque as end-to-end control, but instead captures the "local propagation structure" as a tractable intermediate-level object.

## Limitations & Future Work

1. **Strong locality**: The authors themselves acknowledge that this is not a global linear replacement for the Transformer; the approximation is valid only near the prompt-conditioned operating trajectory. Whether it holds beyond this local neighborhood remains unclear.
2. **Last-token focus only**: This state definition is natural for next-token readout, but may be insufficient for mechanistic analyses requiring full-sequence interaction or cross-token aggregation.
3. **Validation limited to the GPT-2 family**: The scaling law currently holds only for GPT-2, GPT-2-medium, and GPT-2-large; it has not been verified on modern decoder-only LLMs, MoE architectures, or multimodal models.
4. **Tasks are primarily binary concept classification**: Concept directions are derived from binary class-mean differences, which is appropriate for tasks such as sentiment, toxicity, and hate speech detection, but may not transfer directly to open-ended generation, multi-label semantics, or complex reasoning.
5. **Single-objective control**: The current control objective is a target shift in the final concept score; more complex sequence-level behavioral constraints have not been addressed.
6. **Linear readout assumption remains implicit**: If certain semantic concepts do not correspond to stable linear directions, the definitions of both input and output for the surrogate would be affected.

Two directions for future improvement suggest themselves:

1. The framework could be applied to more modern open-source models to test whether "scale improves local identifiability" is specific to GPT-2 or reflects a more general post-training dynamical regularity.
2. The single concept direction could be extended to a multi-input control setting by replacing $v_\ell$ with a small control subspace $V_\ell$, enabling analysis of coupling effects in multi-attribute joint steering.

## Related Work & Insights

**vs. Activation Addition**: Prior work on activation addition identifies promising directions for injection but does not characterize how those directions propagate across layers; this paper fills in the propagation dynamics and intervention allocation problem.

**vs. Linear Representation Hypothesis**: The linear representation hypothesis concerns whether high-level concepts can be encoded along linear directions—a static question; this paper goes further by asking how perturbations along such directions evolve dynamically.

**vs. Local Jacobian Analysis**: Prior work has observed certain local linearity or geometric regularity in trained Transformers, but this paper concretizes that regularity into a verifiable, controllable low-order surrogate supported by scaling evidence.

Three implications for future research stand out:

1. Interpretability research on LLMs should move beyond "whether a concept can be linearly probed" toward studying the propagation structure of concept perturbations.
2. Scaling studies need not revolve solely around loss or benchmark accuracy; system-level properties such as "identifiability" and "compressibility" offer equally productive axes of inquiry.
3. For future work on steering or safety control, learning a usable low-order local surrogate first and then validating the derived control strategy against the original nonlinear model is a more principled route than heuristic actuation schedules.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The paper weaves activation steering, local linearization, reduced-order system identification, and scaling laws into a unified narrative—the contribution is both novel and coherent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Coverage across 10 tasks, three GPT-2 scales, and two evaluation dimensions (gain prediction and control) is solid, though the model family remains narrow.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The exposition is exceptionally clear, with tight alignment among problem definition, methodological structure, and core conclusions.
- **Value**: ⭐⭐⭐⭐⭐ Beyond explaining why certain layers are more amenable to intervention, the work yields actionable control strategies, providing strong inspiration for subsequent research on interpretable steering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning from Synthetic Data via Provenance-Based Input Gradient Guidance](learning_from_synthetic_data_via_provenance-based_input_gradient_guidance.md)
- [\[CVPR 2026\] Bridging Pixels and Words: Mask-Aware Local Semantic Fusion for Multimodal Media Verification](bridging_pixels_and_words_mask-aware_local_semantic_fusion_for_multimodal_media_.md)
- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)
- [\[CVPR 2026\] Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning](revisiting_unknowns_towards_effective_and_efficient_open-set_active_learning.md)
- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](../../ACL2026/social_computing/spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)

</div>

<!-- RELATED:END -->
