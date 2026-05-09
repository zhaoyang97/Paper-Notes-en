---
title: >-
  [Paper Note] Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness
description: >-
  [ACL 2026][LLM/NLP][privileged knowledge] By comparing the correctness-prediction performance of self-probes (using a model's own hidden states) against external probes (using hidden states from other models), this paper identifies inter-model agreement as the critical confounding factor that masks privileged knowledge. After controlling for agreement, domain-specific privileged knowledge is revealed: it exists in factual tasks but is absent in mathematical reasoning.
tags:
  - ACL 2026
  - LLM/NLP
  - privileged knowledge
  - correctness prediction
  - hidden-state probing
  - inter-model agreement
  - domain specificity
date: 2026-05-08
content_hash: 4ea69679a9be4090
---

# Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness

**Conference**: ACL 2026
**arXiv**: [2604.12373](https://arxiv.org/abs/2604.12373)
**Code**: None
**Area**: LLM/NLP
**Keywords**: privileged knowledge, correctness prediction, hidden-state probing, inter-model agreement, domain specificity

## TL;DR

By comparing the correctness-prediction performance of self-probes (using a model's own hidden states) against external probes (using hidden states from other models), this paper identifies inter-model agreement as the critical confounding factor that masks privileged knowledge. After controlling for agreement, domain-specific privileged knowledge is revealed: it exists in factual tasks but is absent in mathematical reasoning.

## Background & Motivation

**State of the Field**: Recent work has shown that LLMs encode meta-information about their outputs in internal hidden states, including entity recognition, temperature reasoning, and epistemic state representations. A central question is whether LLMs possess "privileged knowledge" about answer correctness—i.e., internal correctness signals inaccessible to external observers. Prior work has demonstrated that linear probes can predict output correctness from hidden states with high accuracy.

**Limitations of Prior Work**: The literature contains contradictory conclusions regarding the existence of privileged knowledge. Some studies argue that probes primarily detect retrieval activation patterns rather than correctness signals, and that external models can match the predictive performance of self-probes, suggesting privileged knowledge does not exist. Others find that models internally encode signals for the correct answer even when generating incorrect responses.

**Root Cause**: Conclusions that "privileged knowledge does not exist" may stem from confounded evaluation—when external models can exploit shared correctness patterns as proxy signals, any genuine privileged knowledge may be obscured.

**Paper Goals**: To design a rigorous experimental framework that resolves this debate and determines whether LLMs truly possess privileged knowledge about their own correctness.

**Starting Point**: The authors identify inter-model agreement as the key confounding factor. Since models assign identical correct/incorrect labels on approximately 80% of questions, external probes can leverage the external model's own correctness patterns as a proxy for the target model's behavior. Constructing a "disagreement subset"—questions on which models produce opposite labels—eliminates this confound.

**Core Idea**: On the full test set, self-probes and external probes perform comparably (no privileged advantage). On the disagreement subset, however, a substantial privileged advantage (~5%) emerges in factual tasks but not in mathematical reasoning, establishing that privileged knowledge is domain-specific.

## Method

### Overall Architecture

Given a target model $M_{target}$ and question $q$, the model generates an answer and receives a binary correctness label $y \in \{0,1\}$. A source model $M_{source}$ processes the same question $q$ to produce hidden states $\mathbf{h}$, and a classifier (probe) $f$ is trained to predict $y$. Different configurations are created by varying the source model: self-probes ($M_{source} = M_{target}$) and external probes ($M_{source} \neq M_{target}$). The premium gap is defined as the AUC advantage of the self-probe over the best external probe.

### Key Designs

1. **Formal Definition of Privileged Knowledge**:

    - Function: Decomposes hidden states into observable and unobservable components, providing a theoretical basis for the experiments.
    - Mechanism: Hidden states are decomposed as $\mathbf{h} \approx \mathbf{z}_{public} \oplus \mathbf{z}_{private}$, where $\mathbf{z}_{public}$ captures intrinsic features of the input question (domain, entity type, etc.) accessible to any model, and $\mathbf{z}_{private}$ captures model-specific internal states (memory retrieval success, reasoning confidence, etc.). Privileged knowledge consists of correctness-relevant signals within $\mathbf{z}_{private}$.
    - Design Motivation: A precise definition is necessary to design experiments that eliminate confounds.

2. **Disagreement Subset Evaluation**:

    - Function: Eliminates inter-model agreement as a confounding factor, isolating each model's unique behavior.
    - Mechanism: A subset of questions is constructed for which the target model and the source model produce opposite correctness labels ($y_{target} \neq y_{source}$). Crucially, probes are still trained on the full training set and filtered to the disagreement subset only at inference time. Retraining on the disagreement subset would introduce a perfect negative correlation, allowing external probes to exploit the inverted correctness signal of the external model.
    - Design Motivation: On approximately 80% of questions, models assign identical correct/incorrect labels, enabling external probes to "free-ride" on this agreement and obscure genuine privileged signals.

3. **Per-Layer Premium Gap Analysis**:

    - Function: Localizes where in the network privileged knowledge emerges.
    - Mechanism: Hidden states are extracted every fifth layer, separate probes are trained per layer, and the premium gap (self-probe AUC − best external probe AUC) is computed for each layer. The gap is plotted as a function of normalized layer depth.
    - Design Motivation: To verify whether privileged signals are consistent with known mechanisms of knowledge retrieval—if driven by factual memory retrieval, the signal should emerge from intermediate layers onward.

### Experimental Setup

Three instruction-tuned models of comparable scale are used (Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B), along with the embedding model Qwen3-Embedding-8B as an external source. Five datasets are evaluated: factual knowledge (Mintaka, TriviaQA, HotPotQA) and mathematical reasoning (GSM1K, MATH). Linear probes (logistic regression with L2 regularization) are used with 10-fold nested stratified cross-validation; AUC serves as the evaluation metric.

## Key Experimental Results

### Main Results

| Evaluation Setting | Factual Tasks Premium Gap | Math Reasoning Premium Gap |
|---|---|---|
| Full test set | ≈0 (no advantage in 22/33 model configurations) | ≈0 (no advantage across all models) |
| Disagreement subset | **~5% (significant across all 9 configurations)** | ≈0 (no significant advantage) |

| Dataset Type | Inter-Model Agreement Rate | Note |
|---|---|---|
| Factual knowledge | ~80% | High agreement masks privileged signal |
| Mathematical reasoning | ~75% | No privileged signal even after controlling for agreement |

### Ablation Study

| Analysis Dimension | Factual Tasks | Math Reasoning | Note |
|---|---|---|---|
| Early layers (0–0.25) | Premium gap ≈ 0 | No consistent advantage | Surface/syntactic features; public information |
| Middle layers (0.25–0.40) | Premium gap becomes positive | No consistent advantage | Factual retrieval signal begins to emerge |
| Deep layers (0.40–1.0) | Premium gap continues to grow | MATH ≈ 0, GSM1K negative | Factual privilege accumulates; none in math |
| MLP probe | Qualitatively identical | Qualitatively identical | Nonlinearity does not alter conclusions |
| Qwen-3-32B | Same trend | Same trend | Larger model does not alter conclusions |

### Key Findings

- **The conclusion of "no privileged knowledge" on the full test set is premature**: High inter-model agreement acts as a confounding factor that masks the true signal. Gemma performs best as an external probe in 7/9 factual configurations, but this likely reflects superior encoding of public features $\mathbf{z}_{public}$ rather than the absence of privileged knowledge.
- **Privileged knowledge is domain-specific**: A significant and consistent premium gap (~5%) exists in factual tasks but is entirely absent in mathematical reasoning. This suggests that factual correctness depends on model-specific memory retrieval states, whereas mathematical correctness is determined by problem structure and is therefore observable by any model.
- **Privileged signals accumulate from intermediate layers onward**: This is consistent with the mid-layer information-flow mechanism for knowledge retrieval—Chi et al. (2025) find that knowledge recall is dominated by mid-layer information flow from subject to answer tokens.
- **Anomalous behavior on GSM1K**: On the disagreement subset, external probes outperform self-probes (negative premium gap), indicating that mathematical problem difficulty is entirely determined by problem structure rather than model-specific knowledge.

## Highlights & Insights

- **The methodological contribution is particularly elegant**: Identifying inter-model agreement as a confounding factor and designing the disagreement subset to eliminate it reconciles previously contradictory findings. The design choice to train on the full dataset but evaluate only on the disagreement subset is especially careful, as retraining on the subset would introduce a new artifact.
- **The factual vs. mathematical domain dichotomy** provides deep insight into LLM internals: factual knowledge depends on model-specific parametric memory, while mathematical reasoning relies on general computation over problem structure—explaining why factual hallucinations are difficult to detect externally while mathematical errors are comparatively tractable.
- **The gradual emergence pattern in per-layer analysis** is consistent with causal mechanism studies, reinforcing the interpretation that privileged knowledge originates from memory retrieval processes.
- **The disagreement subset methodology is broadly transferable** to other research settings that require isolating model-specific signals.

## Limitations & Future Work

- The primary analysis is limited to 7B–9B parameter models; larger models may exhibit different patterns of privileged knowledge.
- Only factual knowledge and mathematical reasoning are examined; hybrid domains such as coding and commonsense reasoning remain unexplored.
- Probe-based methods (linear and MLP) may not fully extract privileged signals; more expressive classifiers could reveal additional information.
- The analysis is correlational rather than causal—future work could validate the findings through activation-steering experiments: intervening on the correctness direction in the residual stream should predictably modulate output correctness.

## Related Work & Insights

- **vs. Xiao et al. (2025)**: They propose a "generalized correctness model," arguing that cross-model predictors perform comparably to model-specific probes and therefore privileged knowledge does not exist. This paper shows that conclusion is confounded by inter-model agreement.
- **vs. Chi et al. (2025)**: They find that hidden states are indistinguishable between hallucinated and correct responses, concluding that LLMs do not encode correctness. This paper finds significant privileged signals in factual tasks on the disagreement subset; the two findings can be reconciled, as Chi et al.'s results may be affected by agreement-based confounding.
- **vs. Kadavath et al. (2022)**: They show that LLMs can predict their own correctness with high accuracy. This paper further distinguishes genuine privileged knowledge from predictions based on public features.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Identifying inter-model agreement as a confounding factor and designing the disagreement subset methodology reconciles an important academic debate.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Five datasets, three models, linear and MLP probes, per-layer analysis, and validation on a larger model.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem definition is clear, experimental design is rigorous, and the logical argumentation is tightly structured.
- Value: ⭐⭐⭐⭐ — Makes an important theoretical contribution to research on LLM self-awareness and interpretability; practical applications remain to be developed.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding](../../ICLR2026/llm_nlp/stopping_computation_for_converged_tokens_in_masked_diffusion-lm_decoding.md)
- [\[AAAI 2026\] LoKI: Low-damage Knowledge Implanting of Large Language Models](../../AAAI2026/llm_nlp/loki_low-damage_knowledge_implanting_of_large_language_models.md)
- [\[ACL 2026\] Detoxification for LLM from Dataset Itself](detoxification_for_llm_from_dataset_itself.md)
- [\[NeurIPS 2025\] The Rise of Parameter Specialization for Knowledge Storage in Large Language Models](../../NeurIPS2025/llm_nlp/the_rise_of_parameter_specialization_for_knowledge_storage_in_large_language_mod.md)
- [\[NeurIPS 2025\] C²Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning](../../NeurIPS2025/llm_nlp/c2prompt_class-aware_client_knowledge_interaction_for_federated_continual_learni.md)

<!-- RELATED:END -->
