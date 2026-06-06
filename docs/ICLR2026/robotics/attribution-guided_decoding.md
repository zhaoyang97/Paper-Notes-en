---
title: >-
  [Paper Note] Attribution-Guided Decoding
description: >-
  [ICLR 2026][Robotics][attribution method] This paper proposes AGD, a decoding strategy that, at each generation step, selects from high-probability candidate tokens the one with the highest attribution score toward a use…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "attribution method"
  - "LRP"
  - "instruction following"
  - "factuality"
  - "entropy-gating"
  - "controlled decoding"
date: 2026-05-08
content_hash: f1afa67caa9fb207
---

# Attribution-Guided Decoding

**Conference**: ICLR 2026
**arXiv**: [2509.26307](https://arxiv.org/abs/2509.26307)  
**Code**: [GitHub](https://github.com/piotr-komorowski/AGD)  
**Area**: Robotics
**Keywords**: attribution method, LRP, instruction following, factuality, entropy-gating, controlled decoding

## TL;DR

This paper proposes AGD, a decoding strategy that, at each generation step, selects from high-probability candidate tokens the one with the highest attribution score toward a user-specified region of interest (ROI). This reframes attribution methods from passive analysis tools into active generation guidance mechanisms, achieving significant improvements on both instruction-following and factuality tasks.

## Background & Motivation

**Background**: LLM decoding strategies are a critical lever for controlling generation quality. Standard decoding methods (greedy, top-k, nucleus sampling) regulate output randomness but cannot directly guide the semantic properties of generated text. To enhance instruction following and factual accuracy, two families of methods have been proposed: (1) **Interventionist methods**, such as activation steering, which directly modify internal representations, and contrastive decoding approaches (CAD, DoLA), which modify output logits; (2) **Post-hoc methods**, which filter or rerank outputs.

**Limitations of Prior Work**: Interventionist methods directly modify the model's forward pass or logit distribution, pushing the model into out-of-distribution states, which leads to increased perplexity, repetitive outputs, and degraded text quality. This creates an undesirable trade-off in which users must choose between better control and higher generation quality. For instance, activation steering substantially harms output fluency and coherence while improving instruction following.

**Key Challenge**: How can generation be guided toward desired behaviors (e.g., instruction following, hallucination reduction) **without modifying the model's internal states or output distribution**? A mechanism is needed that achieves effective control without compromising generation quality.

**Goal**: (1) Propose a decoding guidance method that does not intervene in the model's forward pass; (2) Make the method flexibly applicable to diverse tasks (instruction following, factuality, in-context retrieval); (3) Reduce the computational overhead and quality degradation introduced by guidance.

**Key Insight**: The authors propose repurposing **attribution methods** from post-hoc explanation tools into forward guidance tools. Attribution methods can quantify the degree to which each candidate token "depends on" specific parts of the input. If candidate token A has a higher attribution score toward the user instruction than candidate token B, A is more "obedient" to the instruction—selecting A thus achieves instruction-guided generation without modifying any internal model states.

**Core Idea**: Reframe the decoding process as "finding, among candidate tokens, the one with maximum attribution toward a specified ROI," replacing probability maximization with an attribution-based selection mechanism.

## Method

### Overall Architecture

The AGD decoding pipeline consists of four steps: (1) Define the **Region of Interest (ROI)** $R$—this can be the token embeddings corresponding to the instruction portion of the input, a set of attention heads identified as storing specific knowledge, or token embeddings of a context document; (2) Perform a standard forward pass to obtain the probability distribution, select the top-k high-probability candidates to form the candidate set $\mathcal{C}_t$, and filter out tokens with probability below threshold $\pi_{\min}$; (3) For each candidate token $c \in \mathcal{C}_t$, use the attribution method (LRP) to backpropagate and compute attribution scores over each component in the ROI, summing them to obtain the total attribution score $S(c, R)$; (4) Select the candidate token with the highest attribution score as the output for the current step. The entire process neither modifies the model's forward pass nor alters logit values—it is a **selection-based** rather than an **interventionist** method.

### Key Designs

1. **LRP-Based Attribution Scoring Mechanism**

    - **Function**: Quantifies the degree to which each candidate token depends on the specified ROI, providing a principled basis for token selection.
    - **Mechanism**: The pre-softmax logit of candidate token $c$ is backpropagated using Layer-wise Relevance Propagation (LRP), yielding attribution scores $r_\omega$ for each model component $\omega \in \Omega$. The attribution scores of components within the ROI are summed to obtain the total score $S(c, R) = \sum_{\omega \in R} r_\omega$. A higher score indicates that token $c$ relies more heavily on information within the ROI. LRP (specifically the AttnLRP variant) is selected because it handles nonlinear components such as self-attention and layer normalization in Transformers more faithfully than simple gradient-based methods (I×G), while maintaining computational efficiency comparable to gradient methods—requiring only a single backward pass.
    - **Design Motivation**: Attribution methods naturally provide the information of "which part of the input determines a given output," and produce signed scores (positive/negative attribution). Positive attribution helps select tokens that depend on the instruction, while negative attribution helps avoid tokens that violate prohibitive constraints—a rich signal unavailable to purely probabilistic methods.

2. **Flexible ROI Definition**

    - **Function**: By varying the ROI definition, AGD is adapted to diverse tasks including instruction following, closed-book factuality, and open-book in-context retrieval.
    - **Mechanism**: For **instruction following**, the ROI $R_I$ is defined as the set of token embeddings corresponding to the instruction portion of the input (e.g., the system prompt). For **closed-book factuality**, the ROI $R_P$ is defined as a pre-identified set of parametric knowledge attention heads. For **open-book retrieval**, the ROI can be defined as context document token embeddings $R_C$ or in-context retrieval attention heads $R_{IC}$. All tasks share the same AGD algorithmic framework; only the ROI definition needs to be switched.
    - **Design Motivation**: Abstracting the ROI as an arbitrary subset of attributable model components elevates the method from a task-specific solution to a general framework. This "attributable component ↔ control objective" correspondence gives AGD strong flexibility, extensible to any control objective that can be quantified through attribution.

3. **Entropy-Based Adaptive Gating (Entropy-Gating)**

    - **Function**: Applies attribution guidance only when the model is uncertain, reducing computational overhead and protecting output quality.
    - **Mechanism**: The Shannon entropy $H(\mathbf{p}_t)$ of the output distribution at each step is computed. When $H(\mathbf{p}_t) < \tau$ (model is confident), greedy decoding is used directly; when $H(\mathbf{p}_t) \geq \tau$ (model is uncertain), AGD is activated. The threshold $\tau$ is set to the 80th percentile of token-level entropy on IHEval ($\tau = 1.734$). AGD thus applies guidance only at "critical branching points" where the model is hesitant—precisely the points that determine the trajectory of generation.
    - **Design Motivation**: Computing attribution at every step (requiring multiple backward passes) is computationally expensive. More importantly, forcing guidance when the model is already highly confident can disrupt already high-quality outputs—analogous to confusing a student who already knows the answer by imposing unnecessary hints. Adaptive gating achieves an excellent balance between effectiveness and efficiency.

### Loss & Training

AGD is a pure inference-time method that **requires no training or fine-tuning**. Fixed hyperparameters are: $k=5$ (candidate set size), $\pi_{\min}=0.05$ (minimum probability threshold), and $\tau=1.734$ (entropy-gating threshold). The same hyperparameters are used across all experiments, requiring no adjustment for different models or tasks.

## Key Experimental Results

### Instruction Following

Evaluated on three models (Llama 3.1 8B, Qwen 2.5 7B, Gemma 3 4B) across two benchmarks: IHEval and SysBench.

| Model | Method | PLA (IHEval) | QS | PLA*QS | SSR (SysBench) |
|------|------|-------------|-----|--------|----------------|
| Llama 3.1 | Greedy | 66.0 | 81.3 | 53.7 | 26.0 |
| Llama 3.1 | CAD | 73.9 | 72.6 | 53.7 | 32.3 |
| Llama 3.1 | AGD_LRP | **79.1** | 73.2 | **57.9** | 32.2 |
| Llama 3.1 | AGD_LRP_e | 74.5 | 76.4 | 56.9 | **33.9** |
| Qwen 2.5 | Greedy | 63.2 | 74.1 | 46.8 | 27.1 |
| Qwen 2.5 | AGD_LRP_e | **70.4** | 70.6 | **49.7** | **29.9** |
| Gemma 3 | Greedy | 84.7 | 82.3 | 69.7 | 33.3 |
| Gemma 3 | AGD_LRP_e | **86.7** | 81.4 | 70.6 | **36.0** |

### Factuality and In-Context Retrieval (Llama 3.1 8B)

| Setting | Method | TriviaQA | NQ | HotPotQA |
|------|------|----------|-----|----------|
| Closed-book | Greedy | 81.4 | 63.6 | 34.6 |
| Closed-book | DoLA | 81.2 | 63.8 | 34.3 |
| Closed-book | AGD_LRP_h | **82.4** | 63.0 | **39.6** |
| Open-book | Greedy | 89.4 | 83.5 | 81.3 |
| Open-book | CAD | 87.9 | 84.6 | 83.7 |
| Open-book | AGD_LRP_c | **91.4** | **87.9** | **87.9** |

### Key Findings

- **LRP substantially outperforms I×G**: AGD with LRP attribution consistently and substantially surpasses the I×G variant on instruction following. The AttnLRP rules for handling self-attention in Transformers provide more faithful attribution scores, which directly translates into more effective guidance.
- **Negative attribution signals are critical**: For constraints of the type "do not include certain words," violating candidate tokens produce **negative attribution scores** on the instruction portion, enabling the model to actively avoid such tokens. This is a distinctive advantage of AGD over simple probability manipulation methods.
- **Entropy gating significantly improves the quality–compliance trade-off**: On Llama 3.1, full AGD achieves PLA of 79.1 but QS drops to 73.2; the entropy-gated variant yields PLA of 74.5 with QS recovering to 76.4, with the composite metric PLA*QS differing by only 1%. In SysBench multi-turn dialogue, the entropy-gated variant's SSR (33.9) even surpasses full AGD (32.2), indicating that guiding only at critical points is more effective.
- **Substantial gains on open-book QA**: On HotPotQA (containing 80% distractor documents), AGD improves over greedy decoding by 6.6 points, demonstrating that the attribution mechanism helps the model locate relevant passages in noisy contexts.

## Highlights & Insights

- **Paradigm shift from explanation to guidance**: Repurposing attribution methods from post-hoc analysis tools into active guidance signals during generation represents a profound shift in perspective. Attribution methods have been used for decades to "explain why a model behaves as it does"; this paper is the first to use them to "determine how a model should behave."
- **Selection-based vs. interventionist**: AGD does not modify the model's forward pass or logit distribution—it makes a "more informed choice" among candidates the model already considers viable. This ensures that selected tokens always remain within the model's normal distribution, fundamentally avoiding the quality degradation associated with methods such as activation steering.
- **Unified ROI abstraction**: By uniformly abstracting control objectives as "subsets of attributable components," AGD becomes a general framework. As long as a target can be expressed as a set of input tokens or attention heads, AGD can provide guidance—from instruction following to factuality to in-context retrieval, switching requires only a change in ROI definition.

## Limitations & Future Work

- **Candidate set constraint**: As a selection mechanism, AGD cannot generate tokens that do not exist in the candidate set. If the "correct answer" is not among the model's top-k candidates, AGD cannot help.
- **Computational overhead**: Each candidate token requires a backward pass for attribution computation (though entropy gating mitigates this), resulting in non-trivial additional latency for long-form generation.
- **ROI definition relies on prior knowledge**: The ROI for instruction following is relatively natural (the system prompt), but identifying knowledge heads and in-context retrieval heads requires prior analysis, limiting transferability.
- **Validated only on models ≤8B**: The largest model evaluated has 8B parameters; scalability to larger models—particularly regarding the memory requirements of attribution computation—remains unknown.

## Related Work & Insights

- **vs. CAD (Context-Aware Decoding)**: CAD modifies the distribution by contrasting logits with and without the instruction, making it an interventionist method. AGD does not modify logits; it selects the highest-attribution token among high-probability candidates. AGD outperforms CAD on instruction following (Llama 3.1: 79.1 vs. 73.9 PLA), indicating that attribution signals are more effective than contrastive logit differences.
- **vs. Activation Steering**: Steering directly modifies internal representations, offering strong control at the cost of severe quality degradation. AGD's non-interventionist design fundamentally avoids this problem.
- **vs. DoLA**: DoLA reduces hallucination via layer-wise logit contrast, also an interventionist method. AGD significantly outperforms DoLA on closed-book HotPotQA (39.6 vs. 34.3), suggesting that attribution signals capture knowledge storage locations more precisely than inter-layer contrast.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Repurposing attribution methods as generation guidance tools is a highly inspiring contribution
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers three task types, three models, and multiple benchmarks, with thorough ablation and case analysis
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, polished figures, and well-articulated method motivation
- Value: ⭐⭐⭐⭐⭐ — A training-free, general-purpose decoding framework with broad implications for controllable LLM generation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories](../../AAAI2026/robotics/urbannav_learning_language-guided_urban_navigation_from_web-scale_human_trajecto.md)
- [\[NeurIPS 2025\] A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning](../../NeurIPS2025/robotics/a_snapshot_of_influence_a_local_data_attribution_framework_f.md)
- [\[AAAI 2026\] Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation](../../AAAI2026/robotics/affordance-guided_coarse-to-fine_exploration_for_base_placem.md)
- [\[ICLR 2026\] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)
- [\[ICLR 2026\] CLIP Behaves like a Bag-of-Words Model Cross-modally but not Uni-modally](clip_behaves_like_a_bag-of-words_model_cross-modally_but_not_uni-modally.md)

</div>

<!-- RELATED:END -->
