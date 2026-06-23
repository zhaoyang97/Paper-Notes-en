---
title: >-
  [Paper Note] How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning
description: >-
  [ICML 2026][Interpretability][function vector] This paper performs a prompt-granularity causal analysis of the formation mechanism of the function vector (FV) for n-shot prompts. It demonstrates that the FV can be linearly superimposed as a weighted sum of sub-FVs from individual examples, where the weights are determined by FV-head attention. Through 2×2 QK/V caus
tags:
  - ICML 2026
  - Interpretability
  - function vector
  - in-context learning
date: 2026-05-08
content_hash: 3209f5aa7496a063
---
# How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning

**Conference**: ICML 2026  
**arXiv**: [2605.16591](https://arxiv.org/abs/2605.16591)  
**Code**: None  
**Area**: Interpretability / In-Context Learning Mechanism / Function Vector  
**Keywords**: function vector, in-context learning, linear superposition, attention reweighting, QK/V causal decomposition

## TL;DR
This paper performs a prompt-granularity causal analysis of the formation mechanism of the function vector (FV) for n-shot prompts. It demonstrates that the FV can be linearly superimposed as a weighted sum of sub-FVs from individual examples, where the weights are determined by FV-head attention. Through 2×2 QK/V causal intervention, the study shows that contextualization primarily improves FV quality via the QK path (rather than V) by concentrating the model's attention on unambiguous demonstrations.

## Background & Motivation

**Background**: Function vectors (Todd et al. 2024; Hendel et al. 2023) have been established as the causal mechanism for in-context learning (ICL)—average activations on a set of FV-heads constitute an injectable "task direction" that can be added to the residual stream of a 0-shot prompt to recover few-shot behavior. Bakalova et al. (2025) further decomposed ICL into two stages: lower layers allow examples to absorb context from each other (contextualization), while middle layers aggregate these representations into the final token (aggregation).

**Limitations of Prior Work**: A mechanistic explanation at the prompt level regarding how FVs "emerge" from $n$ few-shot examples is currently lacking. Specifically: (i) Does the FV consist of independent contributions from each example that are then added together, or is nonlinear fusion required? (ii) Through which channel does contextualization improve ICL—by rewriting the Value content of each example or by modifying the routing (attention allocation) between Query and Key? (iii) What mechanism does the model use to assign weights to informative examples when both informative and ambiguous examples are present in the prompt?

**Key Challenge**: Existing theoretical literature (e.g., linear regression ICL, softmax retrieval) treats ICL as "retrieval based on similarity between query and key." However, actual LLM FV-head attention is strongly dominated by recency bias, and contextualization between examples clearly alters the final FV—thus, a simple "query-driven retrieval" view is insufficient.

**Goal**: (1) Provide evidence for the additivity of FVs at the prompt level (representationally and causally). (2) Identify the specific channels through which contextualization affects the FV. (3) Verify that contextualization improves quality through attention reweighting rather than simply injecting more information into the Value.

**Key Insight**: The authors introduce **uncontextualized ablation**—zeroing out cross-example attention edges via attention edge ablation while maintaining intra-example attention and attention to the final token. This serves as a "non-contextualized" counterfactual baseline. By comparing this with the full model, the causal contribution of contextualization can be cleanly isolated. This is combined with OLS fitting, Q/K/V patching, and Shapley decomposition to form a falsifiable causal chain.

**Core Idea**: Use a triad of **"linear superposition + attention reweighting + QK/V causal decomposition"** to decompose the black box of "few-shot prompt $\rightarrow$ FV" into a verifiable additive structure and localize the reweighting effect primarily to the Query–Key routing channel.

## Method

### Overall Architecture
The paper aims to answer how the function vector of an n-shot prompt "emerges" from individual examples and what changes during contextualization to improve the FV. The formation of the FV is viewed as an attention aggregation by FV-heads at the final separator position—each example writes its Value into the residual stream of the last token through attention, and the sum of all FV-head outputs yields $v_{FV}(p)$. The analysis proceeds in three layers: first, at the **representation layer**, OLS is used to fit $v_{FV}$ as a weighted sum of sub-FVs to check geometric additivity; next, at the **causal layer**, the reconstructed $\hat v_{FV}$ is injected into a 0-shot prompt to see if it can causally replace the true FV; finally, at the **mechanistic layer**, Q/K/V patching is used to decompose the effects of contextualization into QK routing and V content channels, quantified via Shapley values. Experiments are conducted on frozen gemma-2-{2b, 9b, 27b}, Llama-3.2-{1B, 3B}, and Llama-3.1-8B-Instruct across 10 task families (including ambiguous versions with a 2:1 ratio of ambiguous to unambiguous examples).

### Key Designs

**1. Per-prompt sub-FV + OLS Global Linear Superposition**
To determine if the FV is an additive sum of independent contributions, sub-FVs are obtained using attention masks. The last token $t_{n+1}$ is restricted to attend only to example $i$ and query $x_{n+1}$; the resulting read-out is the sub-FV $v_i$. OLS is then used across a batch of prompts to fit a set of **global** weights such that:
$$v_{FV}\approx \sum_{i=1}^n w_i v_i+\varepsilon$$
Because weights are fitted at the batch level rather than per-prompt, $w_i$ describes the average contribution of position $i$. Additivity is verified by cosine similarity and $R^2$ at the representation level and by the accuracy ratio of reconstructed vs. real FV injection at the causal layer.

**2. Uncontextualized Ablation**
To isolate the causal contribution of contextualization, the authors use attention edge ablation. Cross-example (cross-component) attention weights are zeroed out, while two types of edges are preserved: attention within the same example and attention to the final token. This completely severs information flow between examples while maintaining per-example encoding and final aggregation. This provides a clean counterfactual baseline to attribute differences solely to contextualization.

**3. 2×2 QK vs V Causal Decomposition + Shapley Values**
The contextualization state on an FV-head is represented by two binary variables $(QK, V)\in\{unc,ctx\}^2$. Using Q/K/V patching, Query/Key or Value activations are independently replaced to create four configurations: $F(0,0), F(0,1), F(1,0), F(1,1)$. The total gain from contextualization is $G=F(1,1)-F(0,0)$. Shapley values $\phi_{QK}$ and $\phi_V$—representing the average marginal contribution of each path—are then calculated to distribute $G$ between the two channels.

### Loss & Training
No models are trained; all analyses use causal intervention on frozen pretrained LLMs. OLS uses a closed-form solution for global fitting. FV injection layers and scaling $(\ell, \alpha)$ are determined via layer/scale sweeps for maximum accuracy. Attention smoothness is quantified using normalized attention entropy $\hat H=-\sum p_i\log p_i/\log n\in[0,1]$.

## Key Experimental Results

### Main Results
(Representative results on gemma-2-2b across 10 tasks, 3/5-shot.)

| Dimension | Metric | Results | Description |
|---|---|---|---|
| FV Linear Superposition (Representation) | Avg. cosine($v_{FV}, \hat v_{FV}$) | $\geq 0.925$ | OLS-reconstructed FVs are highly aligned across tasks and settings. |
| FV Linear Superposition (Representation) | Avg. $R^2$ | $\geq 0.875$ | As above. |
| FV Linear Superposition (Causal) | $Acc_{max}(\hat v_{FV}) / Acc_{max}(v_{FV})$ | $\approx 1$ (ctx) | Reconstructed FVs recover most causal effects of true FVs. |
| Robustness Extension | 20-shot linearity | Significant | Additive structure is not a short-prompt artifact. |
| Concentration on Ambiguous Tasks | Attn % to unambiguous examples | 32% (unc) → 61% (ctx) | Contextualization sharpens attention toward informative examples. |
| Positional Centroid Shift | $\Delta C$ (10-shot) | $-0.4\sim -0.6$ | Attention quality shifts toward earlier positions, mitigating recency bias. |
| Entropy Shift (Ambiguous) | $\Delta\hat H$ (3→10 shot) | $-0.08\sim -0.15$ | Contextualization acts as a "selection mechanism" in ambiguous scenarios. |

### Ablation Study

| Configuration | FV injection accuracy | Attn Allocation (amb vs unamb) | Conclusion |
|---|---|---|---|
| Unintervened (PRESENT-PAST-A) | 0.52 | 0.94 (unamb is high) | Full contextualized baseline. |
| Ablate examples only | 0.19 | unamb % drops, amb % rises | Example content determines "who to select." |
| Ablate $x_{n+1}$ only | 0.39 | unamb % stable, total mass drops | Query determines overall intensity, not preference. |
| 2×2 Decomposition (Shapley) | $\phi_{QK}$ consistently positive | — | QK routing leads to consistent quality gain. |
| 2×2 Decomposition (Shapley) | $\phi_V$ heterogeneous | — | V content changes are unstable and may be negative. |

### Key Findings
- **Additivity + Causality Hold**: FVs can be approximated by a linear combination of sub-FVs, and this combination is causally equivalent to the true FV. This simplifies the complex aggregation of ICL into a "weighted sum" model.
- **Contextualization exhibits dual behavior**: For standard prompts, it "flattens" attention to counter recency bias. For ambiguous prompts, it "sharpens" attention to select informative examples.
- **QK channel is the primary cause**: Shapley decomposition across six models shows that QK contributions are robust, while V contributions are heterogeneous. This refines the attribution of contextualization to attention routing.
- **Example-driven vs. Query-driven dichotomy**: In most ambiguous tasks, destroying example information harms FV quality more than destroying query information, suggesting that robust ICL relies on "self-nomination" from examples rather than query-active retrieval.

## Highlights & Insights
- **"Sub-FV + OLS" as a sharp dissection tool**: Using global weights $w_i$ reveals an inherent structural contribution based on position.
- **Uncontextualized ablation as a precise application of edge ablation**: By cutting cross-component edges, the semantic channel is isolated without destroying the model's overall execution capability.
- **2×2 Factorial + Shapley as a standard paradigm**: The study proves that QK and V contributions can differ significantly, suggesting this factorial approach should be used for analyzing other attention circuits.
- **"Ambiguous prompts" as a stress test**: While differences are hard to see in standard tasks due to recency bias, mixing ambiguous examples amplifies attention effects into measurable signals.

## Limitations & Future Work
- **Tasks restricted to token-level mapping**: It remains unknown if linear additivity holds for complex scenarios like CoT or long-context reasoning.
- **Ablation may be over-restrictive**: Edge masking shuts down both beneficial and harmful information flow; selective ablation by layer or head could provide more granularity.
- **Shapley calculation only considers QK/V**: Inter-head collaboration and MLP processing are currently treated as the "environment," which might lead to misestimation of specific channel contributions.

## Related Work & Insights
- **vs. Todd et al. 2024 / Hendel et al. 2023**: While they established the FV concept, this work explains how FVs are formed at a per-example granularity.
- **vs. Bakalova et al. 2025**: This work elevates their two-stage hypothesis (contextualization and aggregation) to the mechanistic level by identifying the QK routing role.
- **vs. Linear Regression Theories of ICL**: Theoretical literature focuses on "query-driven" similarity retrieval. This work's empirical finding of "example-driven" selection in ambiguous scenarios creates a tension that suggests theory needs to incorporate competition between examples.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Letting Tutor Personas Speak Up for LLMs: Learning Steering Vectors from Dialogue via Preference Optimization](../../ACL2026/interpretability/letting_tutor_personas_speak_up_for_llms_learning_steering_vectors_from_dialogue.md)
- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)
- [\[ICLR 2026\] Adaptive Concept Discovery for Interpretable Few-Shot Text Classification](../../ICLR2026/interpretability/adaptive_concept_discovery_for_interpretable_few-shot_text_classification.md)
- [\[ICLR 2026\] Causal Interpretation of Neural Network Computations with Contribution Decomposition](../../ICLR2026/interpretability/causal_interpretation_of_neural_network_computations_with_contribution_decomposi.md)
- [\[ICML 2026\] Singular Vectors of Attention Heads Align with Features](singular_vectors_of_attention_heads_align_with_features.md)

</div>

<!-- RELATED:END -->
