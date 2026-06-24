---
title: >-
  [Paper Note] Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach
description: >-
  [ACL 2025][LLM Safety][Context Robustness] Proposes Grft (Gated Representation Fine-Tuning), a lightweight, plug-and-play gated representation fine-tuning method. With fewer than 200 training samples and only 0.0004% of the model parameters, it enables LLMs to exhibit human-like robust cognitive behavior when encountering contradictory or unhelpful external contexts.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Context Robustness"
  - "Representation Engineering"
  - "Gating Mechanism"
  - "RAG"
  - "Knowledge Conflict"
date: 2026-05-08
content_hash: 2bbb4af8df2f6e1f
---

# Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach

**Conference**: ACL 2025  
**arXiv**: [2502.14100](https://arxiv.org/abs/2502.14100)  
**Code**: None  
**Area**: LLM Safety  
**Keywords**: Context Robustness, Representation Engineering, Gating Mechanism, RAG, Knowledge Conflict

## TL;DR

Proposes Grft (Gated Representation Fine-Tuning), a lightweight, plug-and-play gated representation fine-tuning method. With fewer than 200 training samples and only 0.0004% of the model parameters, it enables LLMs to exhibit human-like robust cognitive behavior when encountering contradictory or unhelpful external contexts.

## Background & Motivation

Techniques such as Retrieval-Augmented Generation (RAG) enhance the factual accuracy of Large Language Models (LLMs) by providing external context, and they have been widely applied in domains like healthcare, law, and finance. However, LLMs face severe challenges when handling **imperfect evidence**:

**Over-reliance on external knowledge**: Even when an LLM possesses the correct internal knowledge, its accuracy plummets from ~99% to ~25-35% when presented with contradictory context.

**Distraction by unhelpful context**: Context that is semantically related but actually unhelpful for answering questions similarly causes severe performance degradation (dropping from ~99% to ~44-53%).

**Gap with human cognition**: Humans naturally weigh external information against internal knowledge, whereas LLMs lack this biological capacity.

The paper defines a **context-robust LLM** as one that should exhibit four types of behavior:
- (a) Relying on external context when internal knowledge is lacking
- (b) Utilizing both when internal and external knowledge match
- (c) Identifying conflicts and providing both answers when internal and external knowledge contradict
- (d) Ignoring the context and relying on internal knowledge when the context is unhelpful

Existing approaches (such as system prompts, ICL, CoT, and direct fine-tuning) fail to reliably achieve these behaviors.

## Method

### Overall Architecture

Grft introduces a lightweight intervention function on the hidden representations of the LLM, consisting of two components:

$$\text{Grft}(\mathbf{h}_l) = \mathbf{h}_l + \text{Gate}(\mathbf{h}_l) \cdot \text{Intervention}(\mathbf{h}_l)$$

**Core Idea**: The hidden representations of LLMs exhibit **inherently distinct patterns** when processing contradictory, matching, helpful, or unhelpful inputs. Precision intervention in the representation space can efficiently modify model behavior.

### Key Designs

**Gate Function**:

$$\text{Gate}(\mathbf{h}_l) = \sigma(\mathbf{W}_g \mathbf{h}_l + \mathbf{b}_g)$$

- Input: Hidden representation of the $l$-th layer $\mathbf{h}_l \in \mathbb{R}^d$
- Output: A scalar between 0 and 1, controlling the intervention strength
- Design Objective: Output a low value for "normal" inputs (unknown questions with helpful contexts, known questions with matching contexts); output a high value for "anomalous" inputs (contradictory/unhelpful contexts)
- Key feature: Adds only approximately 4.1K parameters

**Intervention**:

$$\text{Intervention}(\mathbf{h}_l) = \mathbf{R}^\top(\mathbf{W}\mathbf{h}_l + \mathbf{b} - \mathbf{R}\mathbf{h}_l)$$

- Adopts low-rank representation fine-tuning, learning interventions on representations in a low-dimensional space
- $\mathbf{W}$ and $\mathbf{R}$ are low-rank matrices (rank=4), and $\mathbf{b}$ is the bias offset
- The crucial difference from ReFT is the introduction of the Gate mechanism

**Training Data Construction** (requires only 100 known questions + 100 unknown questions):
- Unknown question pairs (gate label=0): The LLM does not know the answer, and a correct context is provided
- Matching samples (gate label=0): Context is consistent with internal knowledge
- Contradictory samples (gate label=1): Context conflicts with internal knowledge, aiming to output a response that identifies the contradiction and provides both perspectives
- Unhelpful samples (gate label=1): The context cannot assist in answering, aiming to ignore the context and reply using internal knowledge

### Loss & Training

The total loss consists of two parts:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{FT}}(\hat{y}_i, y_i) + \mathcal{L}_{\text{gate}}(\text{Gate}(\mathbf{h}_l^i), z_i)$$

- $\mathcal{L}_{\text{FT}}$: Standard cross-entropy loss to supervise the output behavior
- $\mathcal{L}_{\text{gate}}$: Binary cross-entropy loss to supervise the correct activation of the gating values

During training, the base model parameters are frozen, and only the learnable parameters of Grft are updated. Parameters set as: rank=4, batch size=5, trained for 100 epochs.

**Inference Strategy**:
- **Grft Direct Generation**: Standard generation using the model directly after intervention
- **Grft-requery**: When the output contains "CONTRADICTORY" or "UNHELPFUL" tokens, the original LLM is queried again to retrieve the internal answer

## Key Experimental Results

### Main Results

On the ConflictQA subset, using Llama-2-7B-Chat as the backbone model:

**Contradictory Context** (Known questions + contradictory evidence):

| Method | Short Context | Long Context |
|---|---|---|
| Original LLM | 34.55% | 25.33% |
| CoT | 41.83% | 36.02% |
| Astute-RAG | 59.60% | 46.70% |
| **Ours (Grft)** | **60.88%** | **61.19%** |
| **Ours (Grft-requery)** | **82.49%** | **88.15%** |

**Unhelpful Context** (Known questions + distracting/random evidence):

| Method | Random | Distract |
|---|---|---|
| Original LLM | 53.14% | 44.62% |
| **Ours (Grft)** | **73.22%** | **68.86%** |
| **Ours (Grft-requery)** | **97.98%** | **98.46%** |

**Normal Inputs (Matching/Helpful Context)**: Grft maintains high performance near the original LLM level (99.07%/98.23%/97.03%), demonstrating no performance degradation in normal scenarios.

### Ablation Study

- **Removing Gate** (Grft-W/O Gate): Effective on noisy inputs but degrades on normal inputs (helpful long context: 89.03% vs 98.23%), demonstrating that the Gate is crucial for avoiding over-intervention.
- **With Gate but without gate loss** (Grft-W/O Loss): Performance lies between having and not having the gate, indicating the importance of the supervisory signal of the gate loss.
- **Parameter Efficiency**: Grft requires only 36.9K parameters (0.0005%), significantly fewer than LoRA's 2.1M (0.0311%) and full fine-tuning's 6.74B.

### Key Findings

1. **Visualization of Gating Values**: The average gate value for contradictory inputs is >0.7 and close to 1.0 for unhelpful inputs, while matching/helpful inputs stay <0.3. The Gate successfully distinguishes normal inputs from anomalous ones.
2. **Generalizability**: Grft remains highly effective on the COUNTERFACT and NQ datasets (which were excluded from training) (COUNTERFACT contradictory: 62.52%, which is 20.28% higher than vanilla LLM).
3. **Consistency of Grft-requery**: The re-query strategy substantially enhances performance across all scenarios without compromising normal inputs.

## Highlights & Insights

- **Extreme Parameter Efficiency**: Achieving distinct behavioral alterations with only 0.0004% of model parameters and fewer than 200 training samples, which is credited to intervening in the representation space rather than the parameter space.
- **Exquisite Gating Mechanism**: Adaptively deciding whether to intervene avoids the "one-size-fits-all" performance degradation in normal scenarios, serving as the core improvement over ReFT.
- **Addressing RAG Robustness via Representation Engineering**: Unlike relying on modifying prompts or conducting large-scale fine-tuning, directly executing minimal intervention on the internal representations of LLMs is both highly novel and effective.
- The plug-and-play design allows for easy integration into existing RAG frameworks.

## Limitations & Future Work

- Currently, the method only processes a **single** context-question pair and does not consider more complex internal-external knowledge relationships in multi-document interaction scenarios.
- The experiments are mainly validated on Llama-2-7B and Llama-3-8B; generalization capability across a broader range of models needs further verification.
- The "binary" labels for contradiction detection might be oversimplified: in reality, the degree of contradiction may occupy a continuous spectrum.
- For cases where the internal knowledge of the LLM itself is flawed, this method might exhibit excessive reliance on the erroneous internal knowledge.
- During inference, Grft-requery requires querying the model twice, which escalates computational overhead.

## Related Work & Insights

- Representation Engineering (Zou et al., 2023) discovered characteristic patterns in LLM representations when handling contrastive concepts, serving as the core inspiration for this work.
- ReFT (Wu et al., 2024b) provides the fundamental framework for low-rank representation fine-tuning, onto which Grft adds a gating mechanism.
- Studies on knowledge conflict (Xie et al.; Ying et al., 2024b) reveal the vulnerability of LLMs during internal-external knowledge conflicts.
- This can inspire a broader range of LLM behavioral steering methods, achieving specific actions via targeted interventions in the representation space.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Combining a gating mechanism with representation fine-tuning to resolve RAG robustness is a highly novel perspective.
- **Value**: ⭐⭐⭐⭐⭐ — Extremely low cost, plug-and-play, immediately applicable for enhancing RAG systems.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Thorough ablations and adequate generalizability validation, though the range of test models can be extended.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem definition, with method motivations thoroughly explained.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] From Misleading Queries to Accurate Answers: A Three-Stage Fine-Tuning Method for LLMs](from_misleading_queries_to_accurate_answers_a_three-stage_fine-tuning_method_for.md)
- [\[ICML 2025\] TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs](../../ICML2025/llm_safety/tuco_measuring_the_contribution_of_fine-tuning_to_individual_responses_of_llms.md)
- [\[ICLR 2026\] Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!](../../ICLR2026/llm_safety/be_careful_when_fine-tuning_on_open-source_llms_your_fine-tuning_data_could_be_s.md)
- [\[ACL 2025\] Bias in the Mirror: Are LLMs' Opinions Robust to Their Own Adversarial Attacks](bias_in_the_mirror_are_llms_opinions_robust_to_their_own_adversarial_attacks.md)
- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/llm_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)

</div>

<!-- RELATED:END -->
