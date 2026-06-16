---
title: >-
  [Paper Note] How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning
description: >-
  [ICML 2026][Interpretability][function vector] This paper performs a prompt-level causal analysis of the formation mechanism of function vectors (FVs) in n-shot prompts. It demonstrates that FVs can be linearly superimposed as a weighted sum of sub-FVs from each example, where the weights are determined by FV-head attention. Through 2×2 QK/V causal interventions, i
tags:
  - ICML 2026
  - Interpretability
  - function vector
  - in-context learning
date: 2026-05-08
content_hash: b6a40dd003aa1169
---
# How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning

**Conference**: ICML 2026  
**arXiv**: [2605.16591](https://arxiv.org/abs/2605.16591)  
**Code**: None  
**Area**: Interpretability / In-Context Learning Mechanism / Function Vector  
**Keywords**: function vector, in-context learning, linear superposition, attention reweighting, QK/V causal decomposition

## TL;DR
This paper performs a prompt-level causal analysis of the formation mechanism of function vectors (FVs) in n-shot prompts. It demonstrates that FVs can be linearly superimposed as a weighted sum of sub-FVs from each example, where the weights are determined by FV-head attention. Through 2×2 QK/V causal interventions, it shows that contextualization primarily improves FV quality by narrowing attention onto unambiguous demonstrations via the QK path (rather than V).

## Background & Motivation

**Background**: Function vectors (Todd et al. 2024; Hendel et al. 2023) have been established as the causal mechanism for in-context learning (ICL)—the average activation across a set of FV-heads constitutes an injectable "task direction" that recovers few-shot behavior when added to the residual stream of a 0-shot prompt. Bakalova et al. (2025) further decomposed ICL into two stages: lower layers allow examples to absorb context (contextualization), while middle layers aggregate these representations into the final token (aggregation).

**Limitations of Prior Work**: A mechanical explanation at the prompt level for how FVs "emerge" from $n$ few-shot examples is currently missing. Specifically: (i) Does each example contribute independently to the FV, or is nonlinear fusion required? (ii) Through which channel does contextualization improve ICL—by rewriting the Value content of each example or by modifying the routing (attention allocation) between Query/Key? (iii) What mechanism does the model use to weight informative examples over ambiguous ones?

**Key Challenge**: Existing theoretical literature (linear regression ICL, softmax retrieval) treats ICL as "retrieval based on query and key similarity," but actual FV-head attention in LLMs is strongly dominated by recency bias, and contextualization between examples clearly alters the final FV—the "query-driven retrieval" view alone is insufficient.

**Goal**: (1) Provide evidence for the additivity of FVs at the prompt level (representation + causality); (2) identify the specific channels through which contextualization affects FVs; (3) verify whether contextualization improves quality via attention reweighting rather than simply injecting more information into Values.

**Key Insight**: The authors introduce **uncontextualized ablation**—zeroing out cross-example attention edges via attention edge ablation while maintaining intra-example attention and attention to the last token—as a counterfactual baseline. Comparing this with the full model isolates the causal contribution of contextualization. By combining this with OLS fitting, Q/K/V patching, and Shapley decomposition, the study forms a falsifiable causal chain.

**Core Idea**: Utilizing a triple approach of **"linear superposition + attention reweighting + QK/V causal decomposition"** to decompose the "few-shot prompt → FV" black box into a verifiable additive structure, locating the reweighting effect primarily in the Query–Key routing channel.

## Method

### Overall Architecture
This paper aims to answer how the function vector of an n-shot prompt is formed from individual examples and what contextualization modifies to improve the FV. The authors view FV formation as the aggregation of attention by FV-heads at the final separator position—each example writes its Value into the residual stream of the final token via attention, and the sum of all FV-head outputs yields $v_{FV}(p)$. The analysis proceeds in three layers: first, at the **representation layer**, OLS is used to fit $v_{FV}$ as a weighted sum of sub-FVs to check geometric additivity; next, at the **causal layer**, reconstructed $\hat v_{FV}$ is injected into 0-shot prompts to see if it can causally replace the true FV; finally, at the **mechanistic layer**, Q/K/V patching decomposes the effects of contextualization into QK routing and V content channels, using Shapley values to quantify the primary cause. Experiments are conducted on frozen gemma-2-{2b, 9b, 27b}, Llama-3.2-{1B, 3B}, and Llama-3.1-8B-Instruct across 10 task families.

### Key Designs

**1. Per-prompt sub-FV + OLS global linear superposition: Compressing "unknown nonlinear aggregation" into interpretable addition**

To determine if FVs are independent contributions, the authors use attention masks to obtain sub-FVs for each example: by restricting the final token $t_{n+1}$ to attend only to the $i$-th example and the query $x_{n+1}$, the resulting FV carries only the contribution of example $i$, denoted as $v_i$. OLS is then used across a batch of prompts to fit a set of **global** weights such that $v_{FV}\approx \sum_{i=1}^n w_i v_i+\varepsilon$. Since weights are fitted at the batch level rather than per-prompt, $w_i$ describes the average contribution of position $i$. Additivity is verified via cosine similarity and $R^2$ (representation), and the ratio of accuracy after injecting $\hat v_{FV}$ vs. $v_{FV}$ (causality).

**2. Uncontextualized ablation: Creating a clean counterfactual baseline via edge ablation**

To isolate the causal contribution of contextualization, the authors use attention edge ablation. They zero out attention edge weights between different prompt components but preserve two types of edges: internal attention within an example and attention from components to the final token. This completely cuts information flow between examples while leaving self-encoding and final aggregation intact. The difference between the uncontextualized and full models is then attributed solely to contextualization.

**3. 2×2 QK vs V causal decomposition + Shapley value: Decoupling routing and content channels**

Does contextualization improve the FV by changing attention routing or Value content? These channels are physically coupled. The authors represent the contextualization state on FV-heads as two binary variables $(QK, V)\in\{unc,ctx\}^2$. Through Q/K/V patching, they independently replace activations to obtain four configurations $F(0,0), F(0,1), F(1,0), F(1,1)$. The total benefit of contextualization $G=F(1,1)-F(0,0)$ is then distributed using Shapley values $\phi_{QK}$ and $\phi_V$, which represent the average marginal contribution of each path.

### Loss & Training
The study does not train models; all analyses are performed via causal interventions on frozen pretrained LLMs. OLS uses a closed-form solution. FV injection layers and scales $(\ell,\alpha)$ are determined via sweeps. Attention smoothness is quantified using normalized entropy $\hat H=-\sum p_i\log p_i/\log n\in[0,1]$.

## Key Experimental Results

### Main Results
(Representative results on gemma-2-2b across 10 tasks; full results in Appendix F/H.)

| Validation Dimension | Metric | Result | Description |
|---|---|---|---|
| FV Linear Superposition (Rep) | Avg cosine($v_{FV}, \hat v_{FV}$) | $\geq 0.925$ | OLS-reconstructed FVs are highly aligned across tasks and shots |
| FV Linear Superposition (Rep) | Avg $R^2$ | $\geq 0.875$ | High explained variance for the linear model |
| FV Linear Superposition (Causal) | $Acc_{max}(\hat v_{FV}) / Acc_{max}(v_{FV})$ | $\approx 1$ (ctx) | Reconstructed FVs recover most causal effects of true FVs |
| Robustness | 20-shot Linearity | Matches 5-shot | Additive structure is not an artifact of short prompts |
| Attention in Ambiguous Tasks | Weight on unambiguous examples | unc 32% → ctx 61% | Contextualization sharpens attention on informative examples |
| Attention in Normal Tasks | Entropy change $\Delta\hat H$ | $\approx 0$ | Contextualization rebalances positions without sharpening |
| Position Shift (Normal) | $\Delta C$ (10-shot) | $-0.4\sim -0.6$ | Quality mass shifts forward, mitigating recency bias |
| Entropy Drop (Ambiguous) | $\Delta\hat H$ (3→10 shot) | $-0.08\sim -0.15$ | Contextualization acts as a selection mechanism in ambiguous tasks |

### Ablation Study
| Configuration | FV injection accuracy | Attention Distribution (amb vs unamb) | Conclusion |
|---|---|---|---|
| Unintervened (PRESENT-PAST-A) | 0.52 | 0.94 (unamb ratio) | Full contextualized baseline |
| Ablate examples only | 0.19 | unamb ratio drops significantly | Example content determines "who to select" |
| Ablate $x_{n+1}$ only | 0.39 | unamb ratio stable | Query determines total attention mass, not preference |
| 2×2 Decomposition (Shapley) | $\phi_{QK}$ consistently positive | — | Routing (QK) consistently improves FV quality |
| 2×2 Decomposition (Shapley) | $\phi_V$ heterogeneous | — | Content (V) contribution is task-dependent and unstable |

### Key Findings
- **Dual Verification of Additivity and Causality**: FVs are not only geometrically close to linear combinations of sub-FVs but are also causally equivalent in injection experiments. This simplifies the complex aggregation of ICL into a "weighted sum" model.
- **Dual Roles of Contextualization**: In normal prompts, it "flattens" attention (anti-recency); in ambiguous prompts, it "sharpens" attention (selection). The mechanism switches roles based on task identifiability.
- **QK is the Main Driver, V is Secondary**: Shapley decomposition consistently shows QK contributions are robust, while V contributions vary, refining the attribution of contextualization to "routing."
- **Example-driven vs Query-driven Dichotomy**: In most ambiguous tasks, damaging examples hurts FV quality more than damaging the query, suggesting that robust ICL relies on examples "self-selecting" rather than the query performing active retrieval.

## Highlights & Insights
- **OLS as a Powerful Decomposition Tool**: Using global weights $w_i$ reveals a structural position-based contribution, making it applicable to other circuits where activations aggregate over tokens.
- **Precision of Edge Ablation**: Unlike full head ablations, cutting cross-component edges isolates semantic channels while preserving model execution capabilities, providing a cleaner causal intervention.
- **Factorial Paradigm for QK/V**: The 2×2 factorial + Shapley approach demonstrates that the contributions of routing and content can differ vastly, providing a blueprint for analyzing other attention heads.
- **Ambiguity as a Stress Test**: While recency dominates standard tasks, mixing ambiguous and unambiguous examples amplifies attention effects into measurable signals.

## Limitations & Future Work
- **Task Scope**: Limited to token-level mapping tasks; it remains unknown if linear additivity holds for CoT or long-context reasoning.
- **Ablation Restrictiveness**: Edge masking might cut beneficial early information flow alongside interference; selective ablation could provide more granularity.
- **Shapley Granularity**: Only considers QK and V; inter-head collaboration or MLP post-processing are treated as "environment," which might overlook higher-order interactions.
- **Anomalies like CAP-A**: The query-driven nature of certain tasks (e.g., capitalization) indicates that multiple regimes exist, requiring a more systematic classification.

## Related Work & Insights
- **Comparison with Todd et al. 2024 (FV Definition)**: While they established the FV as a "task direction," this work explains how FVs form at the prompt level and establishes per-example additivity.
- **Comparison with Bakalova et al. 2025 (Two-stage Hypothesis)**: This study upgrades the two-stage hypothesis (contextualization/aggregation) to a mechanistic level by identifying the QK routing path as the primary functional channel of contextualization.
- **Comparison with Linear Regression ICL Theories**: Theoretical work often implies a query-driven "similarity retrieval" model. This study's empirical finding of example-driven selection in ambiguous scenarios suggests that theories should incorporate inter-example competition.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

- **Todd et al., 2024**: Function Vectors in Large Language Models.
- **Bakalova et al., 2025**: The Two-Stage Mechanism of In-Context Learning.
- **Hendel et al., 2023**: In-Context Learning Creates Task Vectors.
</div>
<!-- RELATED:END -->

## Related Papers

- [\[ACL 2026\] Letting Tutor Personas Speak Up for LLMs: Learning Steering Vectors from Dialogue via Preference Optimization](../../ACL2026/interpretability/letting_tutor_personas_speak_up_for_llms_learning_steering_vectors_from_dialogue.md)
- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)
- [\[ICML 2026\] Singular Vectors of Attention Heads Align with Features](singular_vectors_of_attention_heads_align_with_features.md)
- [\[ICLR 2026\] Causal Interpretation of Neural Network Computations with Contribution Decomposition](../../ICLR2026/interpretability/causal_interpretation_of_neural_network_computations_with_contribution_decomposi.md)
- [\[ACL 2026\] Interpretability from the Ground Up](../../ACL2026/interpretability/interpretability_from_the_ground_up_stakeholder-centric_design_of_automated_scor.md)

</div>

<!-- RELATED:END -->
