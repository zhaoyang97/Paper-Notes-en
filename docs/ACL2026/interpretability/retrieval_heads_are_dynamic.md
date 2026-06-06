---
title: >-
  [Paper Note] Retrieval Heads are Dynamic
description: >-
  [ACL2026][Interpretability][retrieval heads] This paper demonstrates that retrieval heads in LLMs, responsible for extracting information from context…
tags:
  - "ACL2026"
  - "Interpretability"
  - "retrieval heads"
  - "dynamic attention heads"
  - "long context"
  - "HotpotQA"
  - "dynamic RAG"
date: 2026-05-08
content_hash: feb5beeec1635f81
---

# Retrieval Heads are Dynamic

**Conference**: ACL2026  
**arXiv**: [2602.11162](https://arxiv.org/abs/2602.11162)  
**Code**: No public code  
**Area**: Information Retrieval / Mechanistic Interpretability / Dynamic RAG  
**Keywords**: retrieval heads, dynamic attention heads, long context, HotpotQA, dynamic RAG

## TL;DR
This paper demonstrates that retrieval heads in LLMs, responsible for extracting information from context, are not a fixed set but change dynamically with each generation step. These dynamic heads cannot be replaced by static sets and can be predicted from hidden states, thereby enhancing dynamic RAG performance.

## Background & Motivation
**Background**: Mechanistic interpretability research has found functional differentiation in Transformer attention heads, such as induction heads, attention sinks, and retrieval heads. Retrieval heads are generally considered responsible for copying or extracting key information from the input context, serving as a crucial internal mechanism for long-context utilization and in-context retrieval.

**Limitations of Prior Work**: Previous methods for identifying retrieval heads mostly treat them as a static set: by calculating which heads frequently point to target information across a large number of samples and taking the top-k. This approach is suitable for global average analysis but ignores the fact that the state and requirements of each token change during autoregressive generation.

**Key Challenge**: If retrieval heads are actually step-specific, then static top heads are merely an average approximation. Certain long-tail heads that rarely appear in global statistics might perform irreplaceable retrieval functions at specific generation steps. Static pruning, static KV cache compression, or static RAG triggering strategies might mistakenly delete these critical mechanisms.

**Goal**: The authors organize the study around three claims: whether retrieval heads change dynamically; whether dynamic heads can be replaced by static heads; and whether the model's hidden states encode future retrieval head patterns. Subsequently, these findings are extended to HotpotQA multi-hop QA, and dynamic head selection is used to improve Dynamic RAG.

**Key Insight**: The paper redefines retrieval heads from dataset-level statistics to timestep-level behavior, directly observing which heads are retrieving target information at each generation step. This granularity is closer to the actual generation process of LLMs.

**Core Idea**: Treat retrieval heads as a dynamic sparse set determined by the current context and generation state. Use a hidden-state probe to predict the heads needed for the current step and apply this to more precise in-context retrieval.

## Method

### Overall Architecture
The paper first defines a copy-paste retrieval score on the Needle-in-a-Haystack (NIAH) task: a head's retrieval score is 1 if its maximum attention at the current step falls on the needle token and that token matches the next token to be generated. Based on this, the authors verify dynamics, irreplaceability, and hidden-state correlation. In HotpotQA, this is relaxed to a reasoning retrieval score (the proportion of attention allocated to supporting facts). Finally, dynamic heads are integrated into a modified version of DRAGIN, exposing only the context windows attended to by dynamic heads during retrieval, and comparing dynamic, static, and random strategies.

### Key Designs
1.  **Iterative definition of dynamic retrieval heads**:
    *   Function: Identify attention heads performing retrieval at the token level during the current generation step.
    *   Mechanism: In NIAH, the copy-paste retrieval score for head $h$ at timestep $t$ is $1[i^*\in I_{needle}\land x^t_{i^*}=\hat{y}]$, where $i^*$ is the position of the head's highest attention token. Unlike static top-k heads, this set is recomputed at every step.
    *   Design Motivation: Long-context retrieval is not continuously performed by the same batch of heads. Some heads activate only under specific tokens or contexts; average statistics underestimate them.

2.  **Irreplaceability test of dynamic heads**:
    *   Function: Verify whether dynamic heads are merely alternative views of static heads or possess independent functional necessity.
    *   Mechanism: For each generation step, a normal forward pass is performed to identify dynamic retrieval heads, which are then masked to regenerate the same token. Control groups mask an equivalent number of top static heads or random heads. Further experiments mask $k$ dynamic heads to see if the model activates static heads for compensation.
    *   Design Motivation: If static heads could replace dynamic ones, masking the dynamic set should not significantly degrade performance. Experiments show that even if the model compensatorily activates static top heads, NIAH accuracy drops from 1.0 to 0.0, indicating dynamic heads handle context-specific functions.

3.  **Hidden-state probe and Dynamic RAG application**:
    *   Function: Prove that dynamic head patterns are predictable and use predictions for retrieval control.
    *   Mechanism: The authors use temporal-offset CCA to measure the correlation between the final hidden state at timestep $n$ and future $n+k$ retrieval scores. They find top-1 canonical correlations of 0.966 for $k=0$, and 0.931/0.915 for $k=1,2$. An MLP probe is then trained to predict retrieval score patterns from the final hidden state. In Dynamic RAG, the probe predicts top-5 dynamic heads, and context windows are expanded based on these heads' attention peaks, showing only this local context to the model.
    *   Design Motivation: If dynamic heads could only be identified post-hoc, their engineering value would be limited; hidden state predictability means they can be used for online control, such as dynamic KV cache, dynamic retrieval triggering, or hallucination monitoring.

### Loss & Training
This is primarily an analysis paper and does not train new LLMs. The MLP probe performs binary classification in NIAH, with the final hidden state as input and binary retrieval scores of attention heads as targets. In HotpotQA, it performs regression to predict continuous reasoning retrieval scores. The Dynamic RAG component uses the probe to predict top-5 dynamic heads combined with the RIND retrieval triggering strategy from DRAGIN.

## Key Experimental Results

### Main Results
Statistical results on NIAH show that dynamic retrieval heads are distributed across a long tail far exceeding the top-20 static set, and the set of heads changes significantly between adjacent generation steps.

| Model | Dynamic heads per step | Activated heads / Total heads | Jaccard with top-20 static | Adjacent step Jaccard | Entropy |
|------|-------------------|---------------------------|--------------------------|----------------|---------|
| Llama-3.1-8B | 12.97 ± 7.69 | 238 / 1024 | 0.3512 | 0.2793 | 3.8154 |
| Llama-3.2-3B | 9.69 ± 6.18 | 149 / 672 | 0.3126 | 0.3188 | 3.0083 |
| Qwen3-8B | 20.18 ± 10.43 | 415 / 1152 | 0.4611 | 0.3668 | 4.1038 |
| Llama-2-13B | 6.20 ± 5.49 | 172 / 1600 | 0.2077 | 0.4979 | 4.8973 |
| Phi-4-mini | 6.13 ± 7.09 | 176 / 768 | 0.1845 | 0.5056 | 3.5532 |

The MLP probe can decode retrieval score patterns from hidden states, indicating that dynamic heads are not random fluctuations but predictable functional patterns.

| Model | Precision | Recall | F1 | AUPRC |
|------|-----------|--------|----|-------|
| Llama-3.1-8B | 0.8344 | 0.8353 | 0.8349 | 0.9173 |
| Llama-3.2-3B | 0.8564 | 0.8351 | 0.8456 | 0.9289 |
| Qwen3-8B | 0.8780 | 0.8362 | 0.8566 | 0.9339 |
| Llama-2-13B | 0.8455 | 0.8220 | 0.8336 | 0.9183 |
| Phi-4-mini | 0.8219 | 0.7865 | 0.8038 | 0.8862 |

### Ablation Study
On HotpotQA, the authors replaced the retrieval score with the supporting facts attention ratio and trained an MLP regressor. $R^2$ reached up to 0.8120, showing that dynamic retrieval signals in multi-hop reasoning are also predictable.

| Model | MSE ↓ | MAE ↓ | R2 ↑ |
|------|-------|-------|------|
| Llama-3.1-8B | 0.0023 | 0.0177 | 0.8120 |
| Llama-3.2-3B | 0.0036 | 0.0247 | 0.8015 |
| Qwen3-8B | 0.0050 | 0.0255 | 0.7200 |
| Llama-2-13B | 0.0009 | 0.0121 | 0.7669 |
| Phi-4-mini | 0.0014 | 0.0109 | 0.7333 |

The Dynamic RAG case study on HotpotQA compares dynamic heads, static heads, random heads, and no RAG. In most models, the dynamic strategy outperforms the static one.

| Model | Dynamic EM/F1 | Static EM/F1 | Dynamic Random EM/F1 | Fixed Random EM/F1 | w/o RAG EM/F1 |
|------|---------------|--------------|----------------------|--------------------|---------------|
| Llama-3.1-8B | 0.456 / 0.5586 | 0.398 / 0.5098 | 0.272 / 0.3670 | 0.272 / 0.3763 | 0.252 / 0.3257 |
| Llama-3.2-3B | 0.384 / 0.4993 | 0.428 / 0.5386 | 0.224 / 0.3143 | 0.226 / 0.3051 | 0.184 / 0.2439 |
| Qwen3-8B | 0.286 / 0.3580 | 0.278 / 0.3429 | 0.210 / 0.2804 | 0.210 / 0.2804 | 0.220 / 0.2961 |
| Llama-2-13B | 0.284 / 0.3838 | 0.278 / 0.3789 | 0.276 / 0.3762 | 0.272 / 0.3751 | 0.192 / 0.2750 |
| Phi-4-mini | 0.202 / 0.2690 | 0.186 / 0.2505 | 0.082 / 0.1090 | 0.086 / 0.1111 | 0.172 / 0.2331 |

### Key Findings
- The Jaccard similarity between dynamic heads and top-20 static heads is generally low. Expanding the static set to top-50 or top-100 actually decreases Jaccard similarity, suggesting dynamic heads are not simply a subset of a larger static collection.
- Masking dynamic retrieval heads harms performance on NIAH and HotpotQA significantly more than masking static or random heads, proving their functional necessity.
- The strong correlation between hidden states and future retrieval scores means the model encodes which retrieval mechanisms it will need before generation starts.
- Performance gains for Dynamic RAG are most evident in Llama-3.1-8B (F1 increases from 0.5098 static to 0.5586 dynamic); Llama-3.2-3B is an exception, likely because smaller models have fewer heads and each performs more mixed functions.

## Highlights & Insights
- The most interesting aspect is the re-characterization of retrieval heads from "fixed organs" to "state-activated functional roles". This is significant for interpretability, pruning, and retrieval-augmented reasoning.
- The hidden-state probe is a highly transferable tool. It transforms dynamic mechanisms from post-hoc analysis into online inference control signals.
- Insights for KV cache compression are direct: statically pruning heads with low average importance might delete long-tail but critical dynamic retrieval heads. Future compression strategies should retain heads based on timestep.
- Implications for hallucination detection: if a model lacks retrieval head activation when generating factual claims, it might be an internal signal that it is "not actually evidencing from context."

## Limitations & Future Work
- The Dynamic RAG experiments use attention masking to simulate context selection rather than actual retrieval and concatenation of external documents in a production setting; deployment feasibility requires further validation.
- The MLP probe is not an oracle; prediction errors may include sub-optimal heads in retrieval control, potentially introducing noise.
- Experiments primarily cover NIAH and HotpotQA, which are retrieval-intensive; it remains to be seen if the same mechanisms apply to long-form summarization, legal reasoning, or codebase QA.
- The paper focuses on attention heads but does not further explain if different dynamic heads form stable circuits or bind to specific semantic operations.

## Related Work & Insights
- **vs Wu et al. retrieval heads**: Early works used dataset statistics to identify fixed retrieval heads; this paper emphasizes that the same model activates different heads at different steps, and static sets are incomplete approximations.
- **vs QRHead / HeadKV**: These methods already focus on query-aware or head-level importance but are often used for static compression or re-ranking; this paper centers on token-level dynamics.
- **vs DRAGIN**: While DRAGIN decides when to retrieve, this paper further determines which internal retrieval heads and context windows should be relied upon.
- **Insights**: Future long-context inference could jointly model "when to retrieve, which context segments to retrieve, and which heads to preserve."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The dynamic retrieval head perspective clearly challenges the static head statistical paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers NIAH, HotpotQA, ablation, probes, and Dynamic RAG, though scene diversity could be expanded.
- Writing Quality: ⭐⭐⭐⭐☆ The three claims are well-organized with clear technical details and a clear main thread.
- Value: ⭐⭐⭐⭐☆ Highly inspiring for interpretability and reasoning system optimization; engineering implementation needs more verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context Language Models](from_interpretability_to_performance_optimizing_retrieval_heads_for_long-context.md)
- [\[ACL 2026\] Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization](preference_heads_in_large_language_models_a_mechanistic_framework_for_interpreta.md)
- [\[AAAI 2026\] Adaptive Evidential Learning for Temporal-Semantic Robustness in Moment Retrieval](../../AAAI2026/interpretability/adaptive_evidential_learning_for_temporal-semantic_robustnes.md)
- [\[ACL 2026\] Learning What Matters: Dynamic Dimension Selection and Aggregation for Interpretable Vision-Language Reward Modeling](learning_what_matters_dynamic_dimension_selection_and_aggregation_for_interpreta.md)
- [\[ICML 2026\] Singular Vectors of Attention Heads Align with Features](../../ICML2026/interpretability/singular_vectors_of_attention_heads_align_with_features.md)

</div>

<!-- RELATED:END -->
