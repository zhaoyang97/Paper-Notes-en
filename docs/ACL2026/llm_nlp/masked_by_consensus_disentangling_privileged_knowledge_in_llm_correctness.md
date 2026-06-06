---
title: >-
  [Paper Note] Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness
description: >-
  [ACL 2026][LLM/NLP][privileged knowledge] This paper discovers that "inter-model consistency" is a key confounding factor masking privileged knowledge by comparing the ability of self-probes (using the model's own hidden…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "privileged knowledge"
  - "correctness prediction"
  - "hidden state probing"
  - "inter-model consistency"
  - "domain specificity"
date: 2026-05-08
content_hash: 8f0699d73d134d67
---

# Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness

**Conference**: ACL 2026  
**arXiv**: [2604.12373](https://arxiv.org/abs/2604.12373)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: privileged knowledge, correctness prediction, hidden state probing, inter-model consistency, domain specificity

## TL;DR

This paper discovers that "inter-model consistency" is a key confounding factor masking privileged knowledge by comparing the ability of self-probes (using the model's own hidden states) and external probes (using hidden states from other models) to predict correctness. Eliminating this consistency reveals domain-specific privileged knowledge: it exists in factual tasks but is absent in mathematical reasoning.

## Background & Motivation

**Background**: Recent research suggests that LLMs can encode meta-information about their outputs through internal hidden states, including entity recognition, temperature reasoning, and cognitive state representations. A core question is: Do LLMs possess "privileged knowledge" about the correctness of an answer—internal signals of correctness inaccessible to external observers? Existing work shows linear probes can predict output correctness from hidden states with high precision.

**Limitations of Prior Work**: Contradictory conclusions exist regarding the presence of privileged knowledge. Some studies argue probes primarily detect retrieval activation patterns rather than correctness signals, noting external models achieve predictive performance comparable to self-probes, implying privileged knowledge does not exist. Conversely, other studies find that internal models indeed encode correct answer signals even when generating incorrect responses.

**Key Challenge**: The conclusion that "privileged knowledge does not exist" might result from confounded evaluation—when external models can use shared correctness patterns as proxy signals, true privileged knowledge (if it exists) may be masked.

**Goal**: Design a rigorous experimental framework to resolve this debate and determine if LLMs truly possess privileged knowledge about their own correctness.

**Key Insight**: The authors identify "inter-model consistency" as a key confounding factor—when models provide the same correct/incorrect labels for approximately 80% of questions, external probes can exploit the external model's own correctness patterns as a proxy for the target model's behavior. This confusion can be eliminated by constructing a "disagreement subset" (questions where models provide opposing labels).

**Core Idea**: While self-probes and external probes perform similarly on the full test set (no privileged advantage), a significant premium gap (~5%) appears in factual tasks within the disagreement subset, whereas it remains absent in mathematical reasoning—proving privileged knowledge is domain-specific.

## Method

### Overall Architecture

Given a target model $M_{target}$ and a question $q$, the model generates an answer and obtains a binary correctness label $y \in \{0,1\}$. A source model $M_{source}$ processes the same question $q$ to obtain hidden states $\mathbf{h}$, and a classifier (probe) $f$ is trained to predict $y$. Different configurations are created by varying the source model: self-probes ($M_{source} = M_{target}$) and external probes ($M_{source} \neq M_{target}$). The premium gap is defined as the advantage in AUC of the self-probe over the external probe.

### Key Designs

1.  **Formal Definition of Privileged Knowledge**:
    *   **Function**: Decomposes hidden states into observable and unobservable components to provide a theoretical basis for experiments.
    *   **Mechanism**: Hidden states $\mathbf{h} \approx \mathbf{z}_{public} \oplus \mathbf{z}_{private}$, where $\mathbf{z}_{public}$ captures inherent features of the input question (domain, entity type, etc.) accessible to any model, and $\mathbf{z}_{private}$ captures model-specific internal states (memory retrieval success, reasoning confidence, etc.). Privileged knowledge is defined as the signal in $\mathbf{z}_{private}$ related to correctness.
    *   **Design Motivation**: A clear definition is required to design experiments that eliminate confounding factors.

2.  **Disagreement Subset Evaluation**:
    *   **Function**: Eliminates inter-model consistency as a confounding factor and isolates the unique behavior of each model.
    *   **Mechanism**: Constructs a subset of questions where the target and source models provide opposing correctness labels ($y_{target} \neq y_{source}$). Key technique: Pros are still trained on the full training set and only filtered to the disagreement subset during inference. Training on the disagreement subset would introduce perfect negative correlation, allowing probes to exploit flipped signals.
    *   **Design Motivation**: Models share correct/incorrect labels on ~80% of questions; external probes can "free-ride" on this consistency, masking true privileged signals.

3.  **Per-Layer Premium Gap Analysis**:
    *   **Function**: Locates where privileged knowledge specifically emerges within the network.
    *   **Mechanism**: Extracts hidden states every 5 layers, trains probes separately, and calculates the per-layer premium gap (Self-probe AUC - Best External-probe AUC). The gap is plotted against normalized layer depth.
    *   *Design Motivation**: Confirms whether privileged signals align with known knowledge retrieval mechanisms—factual memory retrieval should emerge from middle layers.

### Experimental Setup

Three instruction-tuned models of similar scale (Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B) are used, plus an embedding model Qwen3-Embedding-8B as an external source. Five datasets are evaluated: factual knowledge (Mintaka, TriviaQA, HotPotQA) and mathematical reasoning (GSM1K, MATH). Linear probes (Logistic Regression + L2 regularization) are used with 10-fold nested stratified cross-validation, using AUC as the metric.

## Key Experimental Results

### Main Results

| Evaluation Mode | Factual Task Premium Gap | Math Reasoning Premium Gap |
| :--- | :--- | :--- |
| Full Test Set | ≈0 (No advantage in 22/33 cases) | ≈0 (No advantage in any model) |
| Disagreement Subset | **~5% (Significant in all 9 configs)** | ≈0 (No significant advantage) |

| Dataset Type | Inter-model Agreement Rate | Description |
| :--- | :--- | :--- |
| Factual Knowledge | ~80% | High consistency masks privileged signals |
| Math Reasoning | ~75% | No privileged signal even after eliminating consistency |

### Ablation Study

| Analysis Dimension | Factual Tasks | Math Reasoning | Description |
| :--- | :--- | :--- | :--- |
| Early Layers (0-0.25) | premium gap ≈ 0 | No consistent advantage | Surface/syntactic features; public information |
| Middle Layers (0.25-0.40) | premium gap turns positive | No consistent advantage | Factual retrieval signals begin to emerge |
| Deep Layers (0.40-1.0) | premium gap continues to grow | MATH ≈ 0, GSM1K negative | Factual privilege accumulates; none in math |
| MLP Probe | Qualitatively identical | Qualitatively identical | Nonlinearity does not change conclusions |
| Qwen-3-32B | Same trend | Same trend | Larger models do not change conclusions |

### Key Findings

*   **The conclusion of "no privileged knowledge" on full test sets is premature**: High inter-model consistency acts as a confounder masking the true signal. While Gemma performed best as an external probe in 7/9 factual configurations, this likely stems from encoding better public features $\mathbf{z}_{public}$ rather than an absence of privileged knowledge.
*   **Privileged knowledge is domain-specific**: Significant and consistent privileged advantages (~5%) exist in factual tasks but are entirely absent in mathematical reasoning. This suggests factual correctness depends on model-specific memory retrieval states, while mathematical correctness is determined by question structure observable to any model.
*   **Privileged signals accumulate starting from middle layers**: This is consistent with the information flow mechanism of knowledge retrieval—Chi et al. (2025) found knowledge recall is dominated by information flow from middle-layer subjects to answer tokens.
*   **Anomalous performance on GSM1K**: On the disagreement subset, external probes actually outperformed self-probes (negative premium gap), indicating mathematical difficulty is determined entirely by question structure rather than model-specific knowledge.

## Highlights & Insights

*   **Sophisticated Methodology**: Identifying "inter-model consistency" as a confounder and designing the "disagreement subset" to eliminate it reconciles previously contradictory findings. The "train on full, evaluate on subset" design notably avoids introducing new artifacts.
*   **The Factual vs. Math Dichotomy** provides deep insight into LLM internal mechanisms: Factual knowledge relies on model-specific parametric memory, while math reasoning relies on general computation of question structures. This explains why factual hallucinations are hard to detect externally while math errors are relatively transparent.
*   **Layer-wise Progressive Emergence** aligns with causal mechanism research, strengthening the interpretation that privileged knowledge originates from memory retrieval.
*   **The disagreement subset methodology is broadly transferable** to other research scenarios requiring the isolation of model-specific signals.

## Limitations & Future Work

*   Analysis is primarily limited to 7B-9B parameter models; larger models may exhibit different privileged knowledge patterns.
*   Only covers factual knowledge and math reasoning; hybrid domains like coding and commonsense reasoning remain unexplored.
*   Probing methods (linear and MLP) may not fully extract privileged signals; more complex classifiers might reveal more information.
*   The study is correlational, not causal—future work could verify via activation steering: intervening in the correctness direction within the residual stream should predictably modulate output correctness.

## Related Work & Insights

*   **vs. Xiao et al. (2025)**: They proposed a "generalized correctness model" suggesting cross-model predictors perform similarly to model-specific probes, concluding privileged knowledge does not exist. This paper shows that conclusion is masked by inter-model consistency.
*   **vs. Chi et al. (2025)**: They found hidden states are indistinguishable from correct answers during hallucinations, concluding LLMs do not encode correctness. This paper finds privileged signals in factual tasks within the disagreement subset; the two can be reconciled as Chi's findings may be affected by consistency confounding.
*   **vs. Kadavath et al. (2022)**: They found LLMs can predict their own correctness with high precision. This paper further distinguishes between true privileged knowledge and predictions based on public features.

## Rating

*   **Novelty**: ⭐⭐⭐⭐⭐ Identifies inter-model consistency confounder and designs disagreement subset methodology to reconcile major academic debates.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five datasets, three models, dual probes (linear/MLP), layer-wise analysis, and validation on larger models.
*   **Writing Quality**: ⭐⭐⭐⭐⭐ Clear problem definition, rigorous experimental design, and coherent logical derivation.
*   **Value**: ⭐⭐⭐⭐ Significant theoretical contribution to LLM self-awareness and interpretability; practical application value yet to be fully developed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding](../../ICLR2026/llm_nlp/stopping_computation_for_converged_tokens_in_masked_diffusion-lm_decoding.md)
- [\[AAAI 2026\] LoKI: Low-damage Knowledge Implanting of Large Language Models](../../AAAI2026/llm_nlp/loki_low-damage_knowledge_implanting_of_large_language_models.md)
- [\[NeurIPS 2025\] The Rise of Parameter Specialization for Knowledge Storage in Large Language Models](../../NeurIPS2025/llm_nlp/the_rise_of_parameter_specialization_for_knowledge_storage_in_large_language_mod.md)
- [\[NeurIPS 2025\] C²Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning](../../NeurIPS2025/llm_nlp/c2prompt_class-aware_client_knowledge_interaction_for_federated_continual_learni.md)
- [\[ACL 2026\] EVE: A Domain-Specific LLM Framework for Earth Intelligence](eve_a_domain-specific_llm_framework_for_earth_intelligence.md)

</div>

<!-- RELATED:END -->
