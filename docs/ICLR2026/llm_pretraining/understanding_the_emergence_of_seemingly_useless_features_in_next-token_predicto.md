---
title: >-
  [Paper Note] Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors
description: >-
  [ICLR 2026][LLM Pretraining][next-token prediction] By decomposing training gradient signals into three components — direct, pre-cached, and circuit sharing — this paper explains why Transformers trained with NTP learn f…
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "next-token prediction"
  - "feature emergence"
  - "pre-caching"
  - "circuit sharing"
  - "mechanistic interpretability"
date: 2026-05-08
content_hash: d04ea2c4cf1446b0
---

# Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors

**Conference**: ICLR 2026
**arXiv**: [2603.14087](https://arxiv.org/abs/2603.14087)
**Code**: [https://github.com/Markfryazino/useless-features-iclr-code](https://github.com/Markfryazino/useless-features-iclr-code)
**Area**: LLM Pre-training
**Keywords**: next-token prediction, feature emergence, pre-caching, circuit sharing, mechanistic interpretability

## TL;DR
By decomposing training gradient signals into three components — direct, pre-cached, and circuit sharing — this paper explains why Transformers trained with NTP learn features that appear "useless" for predicting the current next token. The framework is validated on OthelloGPT, small language models, and a pre-trained LLM (Gemma 2).

## Background & Motivation

**Background**: LLMs are trained with the next-token prediction (NTP) objective, i.e., learning $p(x_{t+1}|x_1 \cdots x_t)$. Intuitively, models should only learn features useful for predicting the next token. Studies on synthetic tasks have confirmed this — NTP training does appear to learn only immediately useful features.

**Limitations of Prior Work**: However, extensive empirical evidence shows that LLMs acquire features far beyond what immediate NTP requires — including abstract input feature reconstruction, "world models" (e.g., board states encoded by OthelloGPT), and multi-step lookahead capabilities. Why does the NTP objective drive the emergence of these "seemingly useless" features? Existing interpretability work has primarily adopted a teleological perspective (analyzing the algorithmic role of features in the trained model), without investigating the gradient signal sources during training.

**Key Challenge**: The NTP objective provides supervision only over the "next token," yet models learn features pertaining to "global state" and "future tokens." How does the gradient signal, optimizing only immediate prediction, drive the learning of these cross-positional features?

**Goal**: To explain the emergence mechanism of NTP-useless features from the perspective of gradient signal information flow. Specifically: (a) through what pathways do gradient signals reach parameters? (b) which pathways are responsible for learning "useless" features? (c) can the contribution of each mechanism be quantified?

**Key Insight**: Leveraging the computational graph structure of causally masked Transformers, gradient signals are decomposed into three independent pathways, with intervention (ablating mechanisms to observe their effects) and attribution (quantifying the influence of each mechanism) serving as the two analytical methods.

**Core Idea**: NTP-useless features emerge from the NTP objective via two mechanisms: pre-caching (loss signals from future positions back-propagate through attention) and circuit sharing (parameter sharing induces cross-positional feature transfer).

## Method

### Overall Architecture
A tripartite decomposition of information flow in the Transformer computational graph is proposed. Fixing a position $i$ and layer $k$ with residual stream $r_{\theta,i}^k(x)$, all gradient pathways are classified as: (1) **Direct**: paths through $r_{\theta,i}^k$ leading to $\hat{x}_{i+1}$ (immediate NTP loss); (2) **Pre-cached**: paths through $r_{\theta,i}^k$ leading to $\hat{x}_j$ ($j > i+1$, future position loss); (3) **Shared**: paths not through $r_{\theta,i}^k$ (indirect influence via parameter sharing). Together these form a complete decomposition of the gradient (Proposition 3.1).

### Key Designs

1. **Gradient Tripartite Decomposition (Proposition 3.1)**:

    - Function: Exactly decomposes the total gradient $\nabla_\theta L$ into three independent components: direct + pre-cached + shared.
    - Mechanism: Each component is defined via stop-gradient operations. Direct: $\nabla_\theta L_i - \nabla_\theta L_i^{sg(k,i)}$; Pre-cached: $\nabla_\theta \sum_{j \neq i} [L_j - L_j^{sg(k,i)}]$; Shared: $\sum_j \nabla_\theta L_j^{sg(k,i)}$. Their sum equals $\nabla_\theta L$ exactly.
    - Design Motivation: The direct component can only drive learning of NTP-useful features (as it relates solely to immediate prediction), while pre-cached and shared are the potential sources of "useless" feature emergence.

2. **Intervention Methods: Myopic Training and m-Untied Training**:

    - Function: Ablating pre-caching or circuit sharing to observe the effects of their absence.
    - Mechanism: **Myopic training** (proposed by Wu et al. 2024) blocks cross-positional gradient propagation — gradients are cut at the K and V matrices, so position $i$ is not incentivized to compute features useful for future positions. **m-Untied training** (proposed in this work) uses two independent sets of parameters for positions before and after $m$, blocking circuit sharing.
    - Design Motivation: Pre-caching increases expressive capacity (enabling complex multi-layer attention constructions), while circuit sharing facilitates cross-positional feature transfer (features that are NTP-useful at one position are encoded at another via shared parameters).

3. **Attribution Method: Feature Mismatch Influence**:

    - Function: Quantifies the specific contribution of each gradient component to feature emergence during training.
    - Mechanism: Feature mismatch is defined as $R(x|\theta_1, \theta_2, w_i^k) = \frac{1}{2}(\langle w_i^k, r_{\theta_1,i}^k(x) \rangle - \langle w_i^k, r_{\theta_2,i}^k(x) \rangle)^2$, and influence is defined as $I_i^k(\theta, x | w_i^k, \theta^*, G) = \frac{d}{d\varepsilon} R(x|\theta + \varepsilon G, \theta^*, w_i^k)|_{\varepsilon=0}$, where $G$ is a gradient component. An adaptation for the Adam optimizer maintains independent momentum for each of the three components, ensuring their step sizes sum to the actual optimizer step.
    - Design Motivation: Ablation alone cannot distinguish whether a mechanism is "necessary" from "how much it actually contributes." The attribution method provides a quantitative account of each mechanism's contribution by integrating influence at each training step.

4. **Large Model Inference: Intervention-based Influence Ratio $Q(w)$**:

    - Function: Estimates the ratio of pre-cached vs. direct influence in large models that cannot be retrained.
    - Mechanism: **Proposition 5.1** proves that by performing activation ablations on a trained model and computing $Q(w) = \frac{\sum_{j>i+1} d_j^{/i}}{d_{i+1}^{/i}}$ (sum of KL divergences at future positions after ablation / KL divergence at the immediate position), one can approximate the ratio of pre-cached to direct influence.
    - Design Motivation: Retraining large models is prohibitively expensive; however, by accessing only the final checkpoint, intervention experiments can still be used to infer the origins of features.

### Loss & Training
Standard NTP cross-entropy loss is used. The myopic and untied variants modify gradient propagation pathways via stop-gradient operations rather than modifying the loss function itself.

## Key Experimental Results

### Main Results

Analysis of influence on NTP-useful vs. NTP-useless features in OthelloGPT (95% confidence intervals):

| Gradient Component | NTP-useful Features | NTP-useless Features | Interpretation |
|---|---|---|---|
| Direct | [2.85, 12.38] | [-4.69, 2.74] | Direct only drives useful features |
| Pre-cached | [-1.99, 0.66] | [0.55, 3.05] | Pre-cached drives useless features |
| Shared | [4.80, 12.48] | [2.93, 9.91] | Shared contributes to both |
| Combined | [12.14, 19.05] | [4.42, 10.07] | Useful features are learned more strongly |

The influence of direct on NTP-useless features is not significantly different from zero, while pre-cached and shared make positive contributions to NTP-useless features — precisely explaining the source of fragility in OthelloGPT's "world model."

### Ablation Study

Effect of different training modes on the representational quality of NTP-useless features in toy tasks:

| Training Mode | Majority | Conditioned Majority | Notes |
|---|---|---|---|
| Standard training | High probe accuracy | High probe accuracy | All three mechanisms active |
| Myopic (no pre-cache) | Decreased | Substantially decreased | Pre-caching blocked |
| m-Untied (no sharing) | Decreased | Decreased | Circuit sharing blocked |
| Myopic + Untied | Worst | Complete failure | Cannot learn NTP-useless features |

Conditioned Majority requires a two-layer attention construction resembling induction heads; myopic training completely prevents the development of such circuits.

### Key Findings
- **Syntactic features are primarily driven by direct**: In small language models, pre-cached influence on POS tags and dependency labels is substantially lower than direct influence, indicating that simple syntactic features can be learned without pre-caching.
- **Pre-caching is essential for coherent text generation**: The loss of myopic models (3.29) is substantially higher than that of standard models (2.53), demonstrating that pre-caching is critical for complex language modeling.
- **Extreme $Q(w)$ values in Gemma 2 correlate with formal reasoning**: SAE features with extremely high or low $Q(w)$ are concentrated in code and formal domains. Pre-caching is especially important for tasks requiring the simulation of formal computational devices (e.g., AST parsing).
- **Pre-caching ≠ lookahead**: The correlation between $Q(w)$ and feature directions of lookahead predictors is negative — pre-cached features actually contribute less to lookahead prediction. This supports the "breadcrumbs" hypothesis: lookahead arises because different positions require similar features, rather than from explicit planning.

## Highlights & Insights
- **Shift from "static function" to "developmental process"**: Traditional interpretability asks "what does this feature do?"; this paper asks "how was this feature trained into existence?" This developmental perspective, analogous to the function vs. development distinction in neuroscience, provides a fundamentally new tool for understanding LLM internals.
- **Universality of the tripartite decomposition**: The direct/pre-cached/shared decomposition is grounded in the computational graph structure of causally masked Transformers and applies to all autoregressive models. It can serve as a general scaffold for analyzing the origins of any feature.
- **Causal explanation for the fragility of OthelloGPT's world model**: Previously it was known that "the world model is fragile"; now it is understood that "this is because NTP-useless board squares receive only pre-cached and shared signals, lacking direct signals." This constitutes the first statistically significant causal attribution for this phenomenon.

## Limitations & Future Work
- **High computational cost of attribution**: The method requires a full model retraining pass with three gradient components computed at each step, which is infeasible for large models.
- **$Q(w)$ is an approximation**: It is valid only in the local neighborhood of the trained model and does not represent the integrated influence across the entire training trajectory.
- **Gap between toy tasks and real LLMs**: Primary experiments are conducted on small models; large-model analysis is limited to the indirect $Q(w)$ metric.
- **Directions for improvement**: Developing more efficient attribution methods (e.g., requiring only intermediate checkpoints rather than full retraining); using gradient decomposition to discover previously unknown interpretable features (e.g., feature subspaces produced exclusively by pre-cached updates).

## Related Work & Insights
- **vs. Wu et al. (2024)**: They first proposed the concepts of pre-caching and myopic training. This work builds on their foundation by introducing circuit sharing, establishing a quantitative attribution framework, and discovering that pre-caching ≠ lookahead — representing a substantial deepening and correction of their findings.
- **vs. Vafa et al. (2025)**: They found that OthelloGPT's world model is fragile for board states sharing the same legal next moves. This paper provides a causal explanation from the gradient decomposition perspective: the direct influence on NTP-useless squares is zero, and learning relies solely on pre-cached and shared signals, making such features inherently more fragile.
- **vs. Bachmann & Nagarajan (2024)**: They identified a shortcoming of NTP (that useful features may not be learned). This paper analyzes the complementary direction: why NTP can learn features that exceed expectations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Proposes an entirely new framework for understanding feature emergence in Transformers (gradient tripartite decomposition + attribution method), with a distinctive and deep perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Progressively validated across toy tasks, OthelloGPT, small LMs, and Gemma 2 with a clear hierarchy, though large-model analysis is constrained by indirect metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative chain from intuition to formalization to experiments is exceptionally coherent, with precise concept definitions.
- Value: ⭐⭐⭐⭐⭐ Makes a foundational contribution to understanding how the NTP training objective produces "beyond-expectation" capabilities; the framework is broadly applicable to LLM interpretability research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Law of Data Reconstruction for Random Features (and Beyond)](a_law_of_data_reconstruction_for_random_features_and_beyond.md)
- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ICLR 2026\] Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization](understanding_and_improving_shampoo_and_soap_via_kullback-leibler_minimization.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[ICLR 2026\] Identifying and Evaluating Inactive Heads in Pretrained LLMs](identifying_and_evaluating_inactive_heads_in_pretrained_llms.md)

</div>

<!-- RELATED:END -->
