---
title: >-
  [Paper Note] How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning
description: >-
  [ICML 2026][Interpretability][function vector] This paper performs a prompt-level causal analysis of the formation mechanism of function vectors (FV) in n-shot prompts. It proves that an FV can be linearly superposed as…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "function vector"
  - "in-context learning"
  - "linear superposition"
  - "attention reweighting"
  - "QK/V causal decomposition"
date: 2026-05-08
content_hash: 0b726e74778786d4
---

# How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning

**Conference**: ICML 2026  
**arXiv**: [2605.16591](https://arxiv.org/abs/2605.16591)  
**Code**: None  
**Area**: Interpretability / In-Context Learning Mechanism / Function Vector  
**Keywords**: function vector, in-context learning, linear superposition, attention reweighting, QK/V causal decomposition

## TL;DR
This paper performs a prompt-level causal analysis of the formation mechanism of function vectors (FV) in n-shot prompts. It proves that an FV can be linearly superposed as a weighted sum of sub-FVs from each example, where weights are determined by FV-head attention. Through a $2 \times 2$ QK/V causal intervention, the authors demonstrate that contextualization improves FV quality primarily by using the QK path (rather than V) to focus attention on unambiguous demonstrations.

## Background & Motivation

**Background**: Function vectors (Todd et al. 2024; Hendel et al. 2023) have been established as a causal mechanism for In-Context Learning (ICL)—average activations across a set of FV-heads constitute an injectable "task direction" that can be added to the residual stream of a 0-shot prompt to recover few-shot behavior. Bakalova et al. (2025) further decomposed ICL into two stages: lower layers allow examples to absorb context from one another (contextualization), while middle layers aggregate these representations into the final token (aggregation).

**Limitations of Prior Work**: A mechanistic explanation at the prompt level of how FVs "emerge" from $n$ few-shot examples is still lacking. Specifically: (i) Does each example contribute to the FV independently followed by summation, or is non-linear fusion required? (ii) Through which channel does contextualization improve ICL—by rewriting the Value content of each example or by modifying the routing (attention allocation) between Query and Key? (iii) When a prompt contains both informative and ambiguous examples, what mechanism allows the model to concentrate weight on the former?

**Key Challenge**: Existing theoretical literature (linear regression ICL, softmax retrieval) treats ICL as "retrieval based on Query and Key similarity." However, actual LLM attention in FV-heads is strongly dominated by recency bias, and contextualization between examples clearly alters the final FV—the simple "query-driven retrieval" view is insufficient.

**Goal**: (1) Provide evidence (representational and causal) for the additivity of FVs at the prompt level; (2) Identify the specific channels through which contextualization affects FVs; (3) Verify that contextualization improves quality through attention reweighting rather than simply injecting more information into the Value.

**Key Insight**: The authors introduce **uncontextualized ablation**—zeroing out cross-example attention edges via attention edge ablation while preserving intra-example attention and attention to the final token. This serves as a counterfactual "no contextualization" baseline. Comparing this with the full model isolates the causal contribution of contextualization. This is combined with OLS fitting, Q/K/V patching, and Shapley decomposition to form a falsifiable causal chain.

**Core Idea**: Utilizing a triad of **"linear superposition + attention reweighting + QK/V causal decomposition,"** the work decomposes the "few-shot prompt $\to$ FV" black box into a verifiable additive structure and locates the reweighting effect primarily within the Query–Key routing channel.

## Method

### Overall Architecture
The authors view FV formation as an aggregation process of FV-heads at the final separator position: each example writes its Value into the final token's residual stream via attention, and the sum of all FV-head outputs constitutes $v_{FV}(p)$. The analysis consists of three layers:

1.  **Representational Layer**: Extract $v_{FV}$ and sub-FVs $v_i$ for each example from a batch of prompts, using OLS to fit $v_{FV} \approx \sum_i w_i v_i$ to check if additivity holds.
2.  **Causal Layer**: Inject the OLS-reconstructed $\hat v_{FV} = \sum_i w_i v_i$ into 0-shot prompts and compare its injection accuracy with the real $v_{FV}$ to verify causal additivity.
3.  **Mechanistic Layer**: Use Q/K/V patching and $2 \times 2$ factorial interventions to decompose the effect of contextualization into the QK channel and the V channel, quantifying their contributions with Shapley values.

Models include gemma-2-{2b, 9b, 27b}, Llama-3.2-{1B, 3B}, and Llama-3.1-8B-Instruct. Tasks cover 10 families such as CC/PS/PC/PP/EN-FR. Tasks with an "-A" suffix are "ambiguous versions" containing ambiguous examples compatible with multiple candidate mappings mixed with unambiguous examples in a 2:1 ratio as a stress test. $n$-shot is set to 3/5/10.

### Key Designs

1.  **Per-prompt sub-FV + OLS Linear Superposition**:
    *   **Function**: Decomposes the FV of an n-shot prompt into a sum of independent contributions from each example.
    *   **Mechanism**: Sub-FVs $v_i$ are obtained by masking attention so the final token $t_{n+1}$ only attends to the $i$-th example and query $x_{n+1}$. OLS then fits global weights $w_i$ across prompts: $v_{FV} \approx \sum_{i=1}^n w_i v_i + \varepsilon$. Weights are fitted on a batch, describing the average contribution of position $i$. Both representational (cosine, $R^2$) and causal (injection accuracy ratio) metrics are used for validation.
    *   **Design Motivation**: To reduce the FV formation mechanism from an "unknown non-linearity" to an "interpretable additive structure." If additivity holds, studying contextualization becomes equivalent to "studying how weights $w_i$ change."

2.  **Uncontextualized Ablation + Attention Edge Masking**:
    *   **Function**: Constructs a clean counterfactual baseline where examples do not pass information to each other, but intra-example and aggregation information flows remain intact.
    *   **Mechanism**: Via attention edge ablation, attention weights between different prompt components (different examples) are zeroed out, while keeping (i) intra-component attention and (ii) attention to the final token.
    *   **Design Motivation**: Direct comparison between contextualized and uncontextualized models allows the difference to be attributed solely to contextualization. This axial causal intervention provides a ground truth for contextualization effects and source activations for Q/K/V patching.

3.  **2x2 QK vs V Causal Decomposition + Shapley Values**:
    *   **Function**: Decomposes the total gain $G = F(1,1) - F(0,0)$ from contextualization into the QK channel (routing) and the V channel (content).
    *   **Mechanism**: Contextualization states on FV-heads are treated as binary variables $(QK, V) \in \{unc, ctx\}^2$. Q/K/V patching independently replaces activations of Query/Key or Value from the contextualized source, yielding four configurations $F(0,0), F(0,1), F(1,0), F(1,1)$. Shapley values $\phi_{QK}, \phi_V$ are calculated to measure the average marginal contribution of each path.
    *   **Design Motivation**: Traditional ablation cannot distinguish if a component works by changing routing or content. Shapley decomposition decouples these physically coupled channels at the causal effect level, revealing that contextualization improves FV primarily via the QK path.

### Loss & Training
No models are trained in this work; all analyses are performed via causal interventions on frozen pretrained LLMs. OLS uses a closed-form global fit. FV injection parameters $(\ell, \alpha)$ are selected via layer/scale sweep for maximum accuracy. Entropy is quantified using normalized attention entropy $\hat H = -\sum p_i \log p_i / \log n \in [0, 1]$.

## Key Experimental Results

### Main Results
(Representative results on gemma-2-2b across 10 tasks, 3/5-shot.)

| Validation Dimension | Metric | Result | Note |
|---|---|---|---|
| FV Linear Superposition (Rep.) | Avg. cosine($v_{FV}, \hat v_{FV}$) | $\geq 0.925$ | High alignment across all tasks and settings |
| FV Linear Superposition (Rep.) | Avg. $R^2$ | $\geq 0.875$ | Strong linear fit |
| FV Linear Superposition (Causal) | $Acc_{max}(\hat v_{FV}) / Acc_{max}(v_{FV})$ | $\approx 1$ (ctx) | Reconstructed FV recovers most causal effects |
| Robustness Extension | 20-shot linearity | Remains significant | Additive structure is not a short-prompt artifact |
| Attention Focus (Amb. tasks) | Unambiguous example attn % | unc 32% $\to$ ctx 61% | Contextualization sharpens attention to informative examples |
| Attention Dist. (Normal tasks) | Entropy change $\Delta\hat H$ | $\approx 0$ (high) | Contextualization rebalances positions without sharpening |
| Position Center Shift | $\Delta C$ (10-shot) | $-0.4 \sim -0.6$ | Quality shifts from later positions forward, mitigating recency bias |
| Entropy Drop (Amb. tasks) | $\Delta\hat H$ (3 $\to$ 10 shot) | $-0.08 \sim -0.15$ | Contextualization acts as a "selection mechanism" in ambiguous settings |

### Ablation Study
| Configuration | FV injection accuracy | Attention Distribution (amb vs unamb) | Conclusion |
|---|---|---|---|
| Unintervened (PRESENT-PAST-A) | 0.52 | 0.94 (Higher unamb) | Full contextualized baseline |
| Ablate examples only | 0.19 | Unamb share drops, amb rises | Example content determines "who to select" (example-driven) |
| Ablate $x_{n+1}$ only | 0.39 | Unamb share stable, mass drops | Query determines overall intensity, not preference |
| Ablate $x_{n+1}$ (CAP-A task) | Higher drop than ablating ex. | Preference destroyed by $x_{n+1}$ | Exception: Query carries disambiguation signal in CAP-A |
| 2x2 Decomposition (Shapley) | $\phi_{QK}$ consistently positive | — | QK routing change $\to$ consistent FV quality gain |
| 2x2 Decomposition (Shapley) | $\phi_V$ heterogeneous | — | V content change $\to$ unstable, sometimes negligible gain |

### Key Findings
*   **Representational and Causal Additivity**: FVs are not only geometrically approximable by linear combinations of sub-FVs but are also causally equivalent. This simplifies complex aggregation into a weighted sum model.
*   **Duality of Contextualization**: On normal prompts, it "flattens" attention (countering recency); on ambiguous prompts, it "sharpens" attention (selecting informative examples).
*   **QK Channel Dominance**: Shapley values across 6 models show QK contributions are robust, while V contributions are heterogeneous. Contextualization improves ICL primarily by modifying routing.
*   **Example-driven vs. Query-driven Bifurcation**: In most ambiguous tasks, destroying examples hurts FV quality more than destroying the query, suggesting robust ICL relies on examples "self-nominating" rather than active query retrieval.

## Highlights & Insights
*   **The "Sub-FV + OLS" Tool**: Using global weights $w_i$ reveals a structured position-based contribution, providing a template for analyzing any circuit where activations aggregate across tokens.
*   **Precision of Edge Ablation**: Cutting only cross-component attention edges isolates specific semantic channels without destroying the model's overall execution capability—a "cut edges, not nodes" paradigm.
*   **Factorial + Shapley Standard**: This provides a rigorous way to decouple physically coupled QK and V channels in attention analysis, showing their contributions to task performance can differ vastly.
*   **Ambiguous Prompts as Stress Tests**: While attention differences are subtle in normal tasks due to recency bias, mixing ambiguous entries amplifies these effects into measurable signals.

## Limitations & Future Work
*   **Task Constraint**: Limited to token-level mappings; it remains unknown if linear additivity holds for complex reasoning tasks like Chain-of-Thought.
*   **Restrictive Ablation**: Edge masking stops both "good" and "bad" information flow; selective ablation by head or layer might provide finer resolution.
*   **Shapley Scope**: Only QK and V channels are considered; inter-head collaborations and MLP processing are treated as environment noise.
*   **Query-driven Exceptions**: CAP-A shows that some tasks are query-driven; a more systematic taxonomy of task regimes is needed.

## Related Work & Insights
*   **vs. Todd et al. 2024 / Hendel et al. 2023**: This work extends the definition of FV down to the per-example level and verifies causal additivity.
*   **vs. Bakalova et al. 2025**: Moves their two-stage (contextualization vs. aggregation) hypothesis from a descriptive level to a mechanistic level using $2 \times 2$ interventions.
*   **vs. Linear Regression Theory of ICL**: Theoretical work often assumes a query-driven "similarity retrieval" view. This paper's empirical finding of example-driven selection in ambiguous tasks suggests a need for theories to incorporate inter-example competition.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ 
*   Experimental Thoroughness: ⭐⭐⭐⭐ 
*   Writing Quality: ⭐⭐⭐⭐ 
*   Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Letting Tutor Personas Speak Up for LLMs: Learning Steering Vectors from Dialogue via Preference Optimization](../../ACL2026/interpretability/letting_tutor_personas_speak_up_for_llms_learning_steering_vectors_from_dialogue.md)
- [\[CVPR 2026\] SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling](../../CVPR2026/interpretability/subspacead_training-free_few-shot_anomaly_detection_via_subspace_modeling.md)
- [\[ICML 2026\] Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in modern Transformers](dissecting_multimodal_in-context_learning_modality_asymmetries_and_circuit_dynam.md)
- [\[ICML 2026\] Singular Vectors of Attention Heads Align with Features](singular_vectors_of_attention_heads_align_with_features.md)
- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)

</div>

<!-- RELATED:END -->
