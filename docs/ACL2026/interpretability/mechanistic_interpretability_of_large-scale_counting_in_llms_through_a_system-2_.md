---
title: >-
  [Paper Note] Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy
description: >-
  [ACL 2026][Interpretability][System-2] Addressing the failure of LLMs in large-scale counting (where single forward passes fail at $~10–30$ due to limited layer depth)…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "System-2"
  - "Counting"
  - "activation patching"
  - "causal mediation"
  - "attention knockout"
date: 2026-05-08
content_hash: e570d3c00eb606ba
---

# Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy

**Conference**: ACL 2026  
**arXiv**: [2601.02989](https://arxiv.org/abs/2601.02989)  
**Code**: To be confirmed  
**Area**: Mechanistic Interpretability / LLM Reasoning / Counting  
**Keywords**: System-2, Counting, activation patching, causal mediation, attention knockout

## TL;DR
Addressing the failure of LLMs in large-scale counting (where single forward passes fail at $~10–30$ due to limited layer depth), a simple test-time strategy—partitioning lists with `|` and prompting the model to count segments before summing—enables Qwen2.5/Llama3/Gemma3/GPT-4o/Gemini-2.5-Pro to jump from $0–20\%$ to $50–95\%$ accuracy in 50–100 object scenarios. Through attention analysis and four types of causal mediation experiments, the "segment counting -> intermediate aggregation -> final summation" three-stage circuit is localized to Layer 22 of Qwen2.5-7B (head 13 for segments, head 1 for aggregation).

## Background & Motivation
**Background**: LLMs perform well on simple arithmetic, but accuracy for naive counting tasks (e.g., "how many apples in the list") drops sharply when $N > 10$. Prior work (Hasani 2025b, Yehudai 2024) demonstrated this as an **architectural bottleneck** of Transformers: counting signals accumulate layer-by-layer (latent counter), saturating once they reach the layer depth limit. Furthermore, numerical representations in LLMs are sublinear/log-like, making larger numbers increasingly "fuzzy."

**Limitations of Prior Work**: (1) Simple Chain-of-Thought (CoT) provides limited help (requires structured input + CoT); (2) Training-side fixes (re-tokenization, specialized math models) treat symptoms rather than the root cause, as they remain depth-limited; (3) Existing test-time block-based techniques (LVLM-COUNT, Izadi 2025) verify behavioral improvements but fail to explain **internal mechanisms**—such as which heads and layers are activated by partitioning.

**Key Challenge**: (a) **Architecture vs. Task Scale** — A Transformer forward pass can only count up to its depth limit, whereas task scales can be infinite; (b) **Behavior vs. Mechanism** — While prompting is known to be effective, the underlying circuit structure remains unknown, preventing guaranteed controllable scaling.

**Goal**: (1) Propose "explicit partitioning + CoT summation" as a System-2 counting strategy, proving its efficacy across various LLMs; (2) Decouple the "segment count → write intermediate steps → aggregate" three-stage circuit using attention analysis, activation patching, masking ablation, and attention knockout; (3) Causally verify via cross-context patching that the final answer is directly regulated by intermediate step token embeddings.

**Key Insight**: The authors adopt Kahneman's Dual Process Theory: the implicit counting in a single LLM forward pass is treated as "System-1" (fast but depth-limited), while "partitioning + CoT step-by-step summation" is treated as "System-2" (slow, explicit, scalable). Mechanistic interpretability confirms that System-2 is implemented via specific heads/layers within the Transformer.

**Core Idea**: Explicitly partition the input using `|` and force the model to output `part1: x1, part2: x2, ...` before the sum. This keeps each partition within the model's "reliable counting range" (System-1 works), while System-2 handles integer summation (a step where almost all models succeed, as shown in Table 3 final-step accuracy 86–100%).

## Method

### Overall Architecture
The "Test-time System-2 Counting" pipeline: (1) At the input, partition a list of $N$ objects using `|` (fragment size $~6-9$ for open models, $~15-25$ for closed models); (2) Use a prompt requiring the model to follow a fixed output format: `part1: x1\npart2: x2\n... Final answer: x`; (3) The model implicitly counts partitions (System-1) and explicitly aggregates (System-2). No fine-tuning or external tools are required.

Mechanistic analysis utilizes four tools: (a) **CountScope probing** — Patch target token activations into a blank counting context to decode the implied count, localizing "where the latent count exists"; (b) **Token Zero-Ablation** — Zero out activations of the item + comma at the end of a partition to observe the drop in intermediate count probability; (c) **Layer-wise mask/unmask** — Identify which layers write the count; (d) **Attention knockout** — Block individual head attention to find critical heads for segments/aggregation; (e) **Cross-context patching** — Exchange intermediate token embeddings between two different contexts to see if the final answer changes accordingly (verifying causal direction).

### Key Designs

1. **Dual combination of explicit partitioning + forced intermediate steps**:
    - **Function**: Decomposes large-number tasks (impossible for a single forward pass) into sub-tasks within the model's "reliable range" and forces sub-results into materialized tokens for subsequent aggregation.
    - **Mechanism**: Adding `|` without CoT is actually **harmful** (Qwen2.5-7B accuracy at $N=11-20$ drops from 0.38 unstructured to 0.20 structured w/o steps), as partitioning resets the implicit counter but the model fails to aggregate. CoT without partitioning also helps little. Only the **combination** yields a jump from 0.38 to 0.95. Final-step accuracy (summation stage) exceeds $86\%$ for nearly all models, proving the bottleneck lies entirely in the intermediate count step—where System-1 still functions within segments.
    - **Design Motivation**: This combination is the minimum viable implementation of System-2. Partitioning provides a "controllable System-1 sub-task space," while CoT materializes sub-results as tokens for subsequent attention access.

2. **CountScope localizing latent counts at partition boundaries**:
    - **Function**: Identify "which token's hidden state stores the count information for each partition" to provide precise targets for causal intervention.
    - **Mechanism**: CountScope (Hasani 2025b) is a patching-based probe: injecting target activations into a blank counting context to see what number the LM generates. Experiments show count signals are stored with high confidence on the **last item token + last comma token** of each partition, and counters reset between partitions.
    - **Design Motivation**: Traditional logit-lens/tuned-lens are unreliable for decoding numbers. Task-conditioned probes like CountScope are necessary. Localizing to boundary tokens directly validates the mechanistic hypothesis of independent segment counting.

3. **Three-stage circuit + single-head causal attribution**:
    - **Function**: Dissect the specific attention pathways for "Information Storage (partition end tokens) → Information Transfer (intermediate step tokens) → Information Aggregation (final answer token)."
    - **Mechanism**: (a) Attention analysis reveals Layers 19-23 point strongly from intermediate tokens to corresponding partition boundary tokens; (b) Attention knockout identifies Layer 22 as critical: **Head 22-13** handles "partition end → intermediate step" transfer, while **Head 22-1** handles "intermediate step → final answer" aggregation; (c) Cross-context patching provides ultimate validation: swapping intermediate token embeddings from Context A with Context B changes A's final answer accordingly (e.g., $19 \to 21$), confirming token embeddings as causal mediators.
    - **Design Motivation**: Pure attention analysis offers only correlational evidence; activation patching provides causal conclusions. The discovery of different heads for different stages implies a division of labor within the LLM for this task.

### Loss & Training
**Pure inference-time**, no training involved. All interventions (CountScope probe / zero-ablation / attention knockout / cross-context patching) are implemented via forward hooks.

## Key Experimental Results

### Main Results: Behavioral Performance (Accuracy / MAE for $N=11–50$ (Open) and $N=51–100$ (Closed))

| Model | Input | Output | Acc $N=21-30$ | Acc $N=41-50$ | MAE $N=41-50$ |
|---|---|---|---|---|---|
| Qwen2.5-7B (28 layer) | Unstruct | w/o steps | 0.13 | 0.00 | 10.50 |
| Qwen2.5-7B | Unstruct | w/ steps | 0.11 | 0.00 | 9.68 |
| Qwen2.5-7B | Struct | w/o steps | 0.13 | 0.01 | 6.35 |
| **Ours (Qwen2.5-7B)** | **Struct** | **w/ steps** | **0.61** | **0.24** | **2.18** |
| **Ours (Llama3-8B)** | **Struct** | **w/ steps** | **0.54** | **0.26** | **2.20** |
| **Ours (Gemma3-27B)** | **Struct** | **w/ steps** | **0.85** | **0.50** | **2.25** |

| Closed Models ($N=51-100$) | Input/Output | Acc $N=91-100$ | MAE $N=91-100$ |
|---|---|---|---|
| GPT-4o, Unstruct, w/o steps | — | 0.24 | 4.26 |
| **GPT-4o, Struct, w/ steps** | — | **0.86** | **0.18** |
| Gemini-2.5-Pro, Unstruct, w/o steps | — | 0.20 | 2.70 |
| **Gemini-2.5-Pro, Struct, w/ steps** | — | **0.91** | **0.07** |

**Structured + w/ steps is the only universally effective configuration.**

### Ablation Study: Error Source Decomposition (Structured CoT setting)

| Model | Total Acc | Final-step Acc | Intermediate Acc |
|---|---|---|---|
| Qwen2.5-7B | 0.51 | **0.86** | 0.53 |
| Llama 3-8B | 0.49 | **0.96** | 0.48 |
| Gemma 3-27B | 0.71 | **0.93** | 0.76 |
| GPT-4o | 0.89 | **1.00** | 0.89 |
| Gemini-2.5-Pro | 0.94 | **0.97** | 0.94 |

Final-step Accuracy is consistently $\ge 86\%$, indicating the bottleneck is entirely the intermediate count.

### Key Findings
- **Failure modes differ across prompt combinations**: Partitioning without CoT causes models to output the "maximum partition size" as the answer (found in $13\%$ of Qwen2.5-7B and **$43.6\%$** of Llama3-8B errors), aligning with the hypothesis that models output the maximum latent count.
- **System-2 mechanism is staged**: CountScope shows partition counters reset after each `|`. Intermediate tokens aggregate these via Layer 22-Head 13, and the final answer aggregates intermediate tokens via Layer 22-Head 1.
- **Cross-model mechanistic consistency**: Parallel attention patterns were found in Llama3.2-8B (Layers 13-18) and Gemma3-4B (Layers 21-23), suggesting the System-2 circuit activated by prompting is a general Transformer capability rather than model-specific.
- **Counter-intuitive finding on CoT**: While CoT is often seen as a panacea, it provides almost no improvement for unstructured inputs (0.45 vs 0.38). Explicit structural separators are required to provide "stage boundaries" for attention heads.

## Highlights & Insights
- **Staged Computation Framework**: The paper decomposes System-2 behavior into "Storage → Transfer → Aggregation," mapping each to specific token positions and heads. This staged interpretation paradigm is extensible to other "divide and conquer" tasks like multi-step reasoning or long-doc summarization.
- **Input structure as stage boundaries**: Explicit separators create "stage anchors" in the token stream that attention heads can index, which is more reliable than letting the model generate its own stages.
- **Methodological contribution of CountScope**: The discovery that logit-lens/tuned-lens are unreliable for numerical decoding is significant; patching-style probes are superior for math/reasoning interpretability.
- **Bypassing architectural limits via test-time scaling**: Instead of increasing layers, externalizing computation into token sequences allows token-by-token generation to act as depth unrolling.

## Limitations & Future Work
- The study used synthetic noun-repetition data rather than natural prose.
- Requires prior knowledge of the model's "reliable partition size."
- Strategy is limited to near-independent sub-tasks (counting, step-by-step arithmetic) and may not generalize to strongly coupled multi-hop causal chains.
- Cross-model verification focused on layer-level attention rather than exhaustive head-level knockout for every model.

## Related Work & Insights
- **vs. CoT (Wei 2022)**: CoT provides steps but not necessarily stage boundaries; this paper proves boundaries are critical for saturation tasks.
- **vs. Hasani 2025b**: They proposed the layer-wise counter; this paper applies their tool to explain System-2 decomposition.
- **vs. Yehudai 2024**: They provided theoretical upper bounds for implicit counting; this paper bypasses these bounds empirically via test-time decomposition.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ The behavioral strategy is simple, but the **complete three-stage circuit identification** and causal validation are significant mechanistic contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Broad coverage across 5 models, multiple configurations, and rigorous causal interventions.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear structure (phenomenon → localization → causation).
- **Value**: ⭐⭐⭐⭐⭐ Directly guides prompt engineering for capacity-saturated tasks and enhances our understanding of System-2 internal circuits.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models](revitalizing_black-box_interpretability_actionable_interpretability_for_llms_via.md)
- [\[ACL 2026\] Why LLMs Hallucinate on Structured Knowledge: A Mechanistic Analysis of the Reasoning Process](why_llms_hallucinate_on_structured_knowledge_a_mechanistic_analysis_of_reasoning.md)
- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)
- [\[ICML 2026\] Beyond Additive Decompositions: Interpretability Through Separability](../../ICML2026/interpretability/beyond_additive_decompositions_interpretability_through_separability.md)
- [\[ACL 2026\] Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization](preference_heads_in_large_language_models_a_mechanistic_framework_for_interpreta.md)

</div>

<!-- RELATED:END -->
