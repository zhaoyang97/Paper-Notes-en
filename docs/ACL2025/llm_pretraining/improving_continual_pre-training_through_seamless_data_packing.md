---
title: >-
  [Paper Note] Improving Continual Pre-training Through Seamless Data Packing
description: >-
  [ACL 2025][LLM Pretraining][Continual Pre-training] This paper proposes Seamless Packing (SP), a data packing strategy for continual pre-training. Through a two-stage method combining sliding window processing for long texts and the First-Fit-Decreasing (FFD) algorithm for packing short texts, SP preserves context continuity and minimizes truncation/padding, outperforming baseline methods in 99% of the experimental settings.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Continual Pre-training"
  - "Data Packing"
  - "Sliding Window"
  - "Context Continuity"
  - "Bin Packing"
date: 2026-05-08
content_hash: 7038d2bfaa805fa2
---

# Improving Continual Pre-training Through Seamless Data Packing

**Conference**: ACL 2025  
**arXiv**: [2505.22018](https://arxiv.org/abs/2505.22018)  
**Code**: [GitHub](https://github.com/Infernus-WIND/Seamless-Packing)  
**Area**: NLP / Pre-training  
**Keywords**: Continual Pre-training, Data Packing, Sliding Window, Context Continuity, Bin Packing

## TL;DR

This paper proposes Seamless Packing (SP), a data packing strategy for continual pre-training. Through a two-stage method combining sliding window processing for long texts and the First-Fit-Decreasing (FFD) algorithm for packing short texts, SP preserves context continuity and minimizes truncation/padding, outperforming baseline methods in 99% of the experimental settings.

## Background & Motivation

Continual pre-training is an effective strategy to adapt general LLMs to specific domains. Before training, variable-length texts must be packed into fixed-length sequences to enable parallel processing. The most common baseline approach is **Concatenation and Truncation (CT)**, which concatenates all documents end-to-end and segments them into equal-length sequences matching the model's maximum sequence length.

However, this straightforward approach suffers from two severe limitations:

**Contextual Fragmentation**: Arbitrary truncation splits semantically related content across sequence boundaries. For example, when "Event A will be held at some venue on some date" is split, the location/date information is detached from the event description, preventing the model from associating them.

**Truncation-Induced Hallucination**: In generation tasks such as summarization, the truncation of crucial explanatory context can lead to unfaithful model outputs.

An alternative strategy is using padding to avoid truncation, but padding consumes valuable sequence capacity without introducing any information. The authors formalize data packing as an optimization problem: **maximizing context continuity while minimizing truncation and padding under sequence length constraints**.

## Method

### Overall Architecture

Seamless Packing operates in two sequential stages:
- **Stage 1 (Sliding Window)**: Processes long texts, utilizing overlapping windows to achieve full-sequence coverage.
- **Stage 2 (FFD Packing with Dropping)**: Efficiently packs remaining short texts to minimize padding and truncation.

### Key Designs

#### 1. Stage 1: Sliding Window

**Function**: Generates continuous sequences dynamically for long texts qualifying for sliding windows, ensuring long documents fully span multiple sequences without being concatenated with other texts.

**Mechanism**: In traditional CT, any remainder of a long text divided by the sequence length is concatenated with the next document, leading to context fragmentation. SP introduces an appropriate amount of overlap between consecutive sequences to absorb the remainder, allowing the text to cover $n+1$ sequences instead of $n$.

**Key Parameters**: Maximum overlap rate $r_{max}$, which bounds the overlap between consecutive sequences. Given an original text length $L_{original}$ and $n$ complete segments:

$$L_{max\_overlap} = \lceil n \times r_{max} \times L_{seq} \rceil$$

When $L_{original} + L_{max\_overlap} \geq (n+1) \times L_{seq}$, the sliding window is applicable. The exact overlap is computed dynamically as:

$$L_{final\_overlap} = \lceil \frac{(n+1) \times L_{seq} - L_{original}}{n} \rceil$$

**Design Motivation**: 
- Fixed-stride traditional sliding windows fail to adapt to variable-length documents, yielding either incomplete coverage or excessive overlap.
- Adjusting parameters via $r_{max}$ instead of a fixed stride provides intuitive control over data redundancy.
- This design minimizes the mixture of unrelated document fragments within individual sequences.

**Theoretical Analysis**: If $n \geq \lceil 1/r \rceil$, all long texts can be processed utilizing sliding windows. For $r_{max}=0.3$, approximately 62% of the corpus can theoretically be packed in Stage 1.

#### 2. Stage 2: Packing with Dropping

**Function**: Packs the remaining short text segments from Stage 1 into fixed-length sequences.

**Mechanism**: This step models data packing as a bin-packing variant using the First-Fit-Decreasing (FFD) heuristic. A major innovation is allowing the bin capacity to slightly exceed the target sequence length ($c_{bin} = L_{seq} + c_{extra}$), discarding the minor overflow (dropping) rather than inserting padding.

**Algorithmic Workflow**:
1. Sort all short texts in descending order of length.
2. For each text, find the first bin capable of containing it.
3. Discard tokens that exceed $L_{seq}$.
4. For the few bins that cannot be filled, concatenate and re-segment their contents.

**Design Motivation**:
- Dropping avoids padding inefficiencies, ensuring model training is not diluted by meaningless padding tokens.
- $c_{extra}$ offers precise control over token loss. With proper configuration, dropped tokens represent a minor fraction (approx. 3.4%).
- FFD is preferred over Best-Fit-Decreasing (BFD) due to its early stopping mechanism, enabling 29% faster processing with comparable efficiency.

### Loss & Training

SP is a pure data pre-processing method that does not modify the underlying training objective, loss function, or optimization pipeline. It represents a complementary preprocessing step that integrates with data sampling and catastrophic forgetting mitigation.

## Key Experimental Results

### Main Results: Continual Pre-training Perplexity

| Model | Domain | CT | BFD | SP |
|------|------|-----|-----|-----|
| GPT2-812M | News | 10.79 | 10.28 | **10.13** |
| Llama3.2-1B | News | 11.48 | 10.52 | **10.07** |
| Llama3.2-1B | Finance | 6.27 | 5.71 | **5.52** |
| Qwen2.5-1.5B | Med | 6.71 | 6.47 | **6.43** |

SP achieves the lowest perplexity across all 9 evaluated model $\times$ domain combinations.

### Main Results: Downstream Task Performance (Full-Parameter Fine-Tuning)

| Task | Model | OM | CT | BFD | SP |
|------|------|------|------|------|------|
| BBC News | Llama3.2-1B | 97.22 | 97.44 | 97.54 | **97.64** |
| AG News | Qwen2.5-1.5B | 88.53 | 89.39 | 89.71 | **90.37** |
| Fin Sentiment | Llama3.2-1B | 87.25 | 88.18 | 88.18 | **88.68** |
| ChemProt | Qwen2.5-1.5B | 81.39 | 81.89 | 81.83 | **82.64** |
| PubMed Class | Qwen2.5-1.5B | 85.10 | 86.32 | 86.22 | **86.87** |

SP yields superior downstream performance in 99% of settings. While BFD degrades compared to CT in certain scenarios, SP consistently demonstrates robust improvements.

### Ablation Study

| Method | PubMed Class (3-Model Average) | ChemProt (3-Model Average) |
|------|------------------------|---------------------|
| FFD | 85.65 | 80.04 |
| BFD | 85.61 | 80.34 |
| BFD-m (Expanded Bin) | 85.57 | 80.53 |
| SP (Ours) | **86.16** | **81.48** |

- BFD-m > BFD: Increasing bin capacity to discard marginal portions is effective.
- SP > BFD-m: The sliding window stage contributes significantly to final downstream capacity.
- FFD $\approx$ BFD: The computational overhead introduced by BFD is redundant.

### Generalization Verification

| Setup | Results |
|------|------|
| Mixed Domains (News+Finance+Med) | SP achieves optimal performance across all 3 domains |
| General Domain (RedPajama → GLUE) | SP is optimal across tasks like MNLI/QNLI/RTE |
| Cross-lingual (French C4 → XNLI) | SP (69.04) > BFD (68.64) > CT (67.72) |
| LoRA Fine-Tuning | SP consistently outperforms BFD and CT |

### Hyperparameter Analysis

- **$r_{max}$ = 0.3** is optimal: higher thresholds lead to excessive training redundancy, while lower values offer insufficient coverage.
- **$c_{extra}$ = 50** is optimal (at $L_{seq}$ = 2048): larger limits result in unnecessary token truncation, whereas tighter bounds yield minor gains.

### Key Findings

1. **SP yields ~4x performance gain over BFD**: SP brings an average improvement of 0.96% compared to CT, while BFD only obtains 0.24%.
2. **BFD is unstable**: For every model, BFD performs worse than baseline CT in 2–4 downstream tasks, whereas SP remains consistently superior.
3. **Case Study**: When tested on synthetic event recollection, the model trained with SP recalled dates (3/5) and locations (5/5) correctly, whereas BFD-trained models suffered severe hallucinations.
4. **Consistency with Theory**: Stage 1 covers roughly 62% of the corpus, leaving only about 3.4% of tokens to be processed in Stage 2.

## Highlights & Insights

- **Novel Perspective on the Problem**: While prior research focuses on data curation, scheduling, and catastrophic forgetting, this paper addresses the fundamental but overlooked question of "how data is arranged/packed."
- **Simple yet Efficient Method**: The two-stage design decomposes long and short texts, providing complementary benefits without redundancy.
- **dropping > padding**: Under the SP framework, discarding a minor fraction of excess tokens is more effective than accumulating meaningless padding. This is a counter-intuitive but experimentally validated insight.
- **Extremely Comprehensive Experiments**: Covered 3 model architectures $\times$ 3 domains $\times$ 8 downstream tasks, supplemented with mixed-domain, general-domain, cross-lingual, and LoRA evaluations.

## Limitations & Future Work

1. **Incomplete Theoretical Explanation of Dropping**: A rigid mathematical framework explaining why dropping small portions performs better than standard padding padding is missing.
2. **Evaluated Only in Continual Pre-training**: The effectiveness during standard language modeling *from scratch* remains to be explored.
3. **Coding Domain Omitted**: Code corpora present structured syntax, which likely requires distinct packing considerations.
4. **Limited Model Scale**: Evaluations are restricted to models under 3B parameters; validation on larger scales remains future work.
5. **Content-Aware Packing**: Packing in this work relies solely on document length; considering semantic associations to cluster sequences remains an open direction.

## Related Work & Insights

- **Krell et al. (2021)**: Explored packing algorithms to maximize sequence throughput but overlooked context continuity.
- **Ding et al. (2024)**: Analyzed truncation impacts and proposed BFD; SP's Stage 2 builds directly upon their findings.
- **In-context Pretraining (Shi et al., 2024)**: Orders raw documents by semantic relation, indicating that data engineering techniques (e.g., document placement/packing) significantly influence final foundation capabilities.

## Rating

- **Novelty**: ⭐⭐⭐ — The methodology represents a clever integration of existing tools (sliding window + bin packing). The primary innovation lies in problem identification and execution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Solid, thorough validation across multiple architectures, domains, downstream benchmarks, ablations, and hyperparameter grids.
- **Writing Quality**: ⭐⭐⭐⭐ — Highly structured, presenting cohesive theoretical analyses backed by clear diagrams and extensive tables.
- **Value**: ⭐⭐⭐⭐ — Delivers a plug-and-play preprocessing solution with immediate practical guidelines for industrial-scale continual pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training](velocitune_a_velocity-based_dynamic_domain_reweighting_method_for_continual_pre-.md)
- [\[ACL 2025\] How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training](how_do_llms_acquire_new_knowledge_a_knowledge_circuits_perspective_on_continual_.md)
- [\[ICCV 2025\] ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training](../../ICCV2025/llm_pretraining/aceg_improving_generalization_of_scene_coordinate_regression.md)
- [\[ACL 2025\] Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation](synthesizing_post-training_data_for_llms_through_multi-agent_simulation.md)

</div>

<!-- RELATED:END -->
