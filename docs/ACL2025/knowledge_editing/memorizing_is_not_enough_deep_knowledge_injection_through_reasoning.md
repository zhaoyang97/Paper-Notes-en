---
title: >-
  [Paper Note] Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning
description: >-
  [ACL 2025][Knowledge Editing][Knowledge Injection] Proposes a four-level knowledge injection framework (Memorization → Retrieval → Reasoning → Association) and builds the DeepKnowledge synthetic evaluation platform. It systematically reveals the key factors for each level of knowledge injection: repetitive learning for memorization, diverse expressions for retrieval, and explicit reasoning patterns for deep reasoning and association, providing a complete method-level mapping…
tags:
  - "ACL 2025"
  - "Knowledge Editing"
  - "Knowledge Injection"
  - "Four-level Framework"
  - "Memorization-Retrieval-Reasoning-Association"
  - "Continual Pre-training"
  - "DeepKnowledge"
  - "Knowledge Types"
date: 2026-05-08
content_hash: 1edcb22285ec5afc
---

# Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning

**Conference**: ACL 2025  
**arXiv**: [2504.00472](https://arxiv.org/abs/2504.00472)  
**Code**: Undisclosed  
**Authors**: Ruoxi Xu, Yunjie Ji, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Ben He, Yingfei Sun, Xiangang Li, Le Sun  
**Institutions**: Institute of Software, Chinese Academy of Sciences, University of Chinese Academy of Sciences, a-m-team  
**Area**: Knowledge Injection / LLM Knowledge Management  
**Keywords**: Knowledge Injection, Four-level Framework, Memorization-Retrieval-Reasoning-Association, Continual Pre-training, DeepKnowledge, Knowledge Types

## TL;DR

Proposes a four-level knowledge injection framework (Memorization → Retrieval → Reasoning → Association) and builds the DeepKnowledge synthetic evaluation platform. It systematically reveals the key factors for each level of knowledge injection: repetitive learning for memorization, diverse expressions for retrieval, and explicit reasoning patterns for deep reasoning and association, providing a complete method-level mapping for LLM knowledge updates.

## Background & Motivation

**Background**:
   - LLMs capture knowledge from mass pre-training data, but staticity leads to outdated knowledge.
   - Continual Pre-training (CPT) is a common strategy to update domain knowledge.
   - Prior knowledge injection studies remain at a shallow level—mainly focusing on knowledge **memorization** (text completion) and **retrieval** (answering paraphrased questions).

**Core Problem**:
   - Knowledge injection is not a binary process but a continuous process from 0 to 1—prior work lacks a systematic definition of levels.
   - Shallow knowledge (memorization and retrieval only) cannot support reasoning tasks, leading to poor LLM performance in scenarios requiring deep reasoning.
   - The impact of different knowledge types (novel vs. incremental vs. updated) on injection effectiveness has not been systematically studied.

**Goal**: Establish a systematic mapping between knowledge injection levels and injection methods to guide efficient knowledge injection in practice.

## Method

### Four-Level Knowledge Injection Framework

| Level | Name | Definition | Capability Requirement |
|------|------|------|----------|
| Level 1 | **Knowledge Memorization (Memorization)** | Recall and repeat injected knowledge in its original form | Text completion |
| Level 2 | **Knowledge Retrieval (Retrieval)** | Correctly extract knowledge under different semantically equivalent formulations | Paraphrased QA |
| Level 3 | **Knowledge Reasoning (Reasoning)** | Apply injected knowledge to reasoning tasks | Multi-step reasoning |
| Level 4 | **Knowledge Association (Association)** | Jointly reason using both injected knowledge and existing knowledge | Cross-knowledge reasoning |

### DeepKnowledge Evaluation Platform Construction

#### Knowledge Acquisition

**Existing Knowledge Filtering**:
- Source: WikiFactDiff + MQuAKE
- Triple filtering criteria: uniqueness (subject-relation pair results are unique), non-recursiveness (subject $\neq$ object), chain-reasoning capability
- Manually select 16 groups of key reasoning relations
- 3-shot testing retains facts that the model can correctly recall → 26,477 valid knowledge facts

**Synthetic Knowledge Generation**:
- Use LLMs to generate fictional entity names (e.g., "FrankTown")
- Assign the same relation types as real knowledge to fictional entities
- Generate 109,860 synthetic knowledge facts

#### Four-Level Test Case Generation

1. **Memorization Test**: Remove the object from the training corpus to form cloze questions.
2. **Retrieval Test**: Use an LLM to rewrite the memorization test into 10 semantically equivalent questions.
3. **Reasoning Test**: Define two basic reasoning rules
    - **Combination**: Multi-hop knowledge aggregation
    - **Comparison**: Size comparison of knowledge
    - n-step reasoning = sampling n rules + filling knowledge + GPT-4 translation into natural language questions
4. **Association Test**: Similar to the reasoning test, but the questions must contain both newly injected knowledge and existing knowledge.

### Knowledge Types

| Type | Definition | Example |
|------|------|------|
| **Novel** | Completely new information about new entities | Newly proposed scientific theories |
| **Incremental** | Supplementary information for existing entities | A new book by a known author |
| **Updated** | Replacing outdated information of existing entities | A sports team getting a new coach |

### Injection Scenarios

| Scenario | Description |
|------|------|
| Duplicate | Same knowledge repeated multiple times (no modification) |
| Vanilla Paraphrase | LLM paraphrases the representation of knowledge |
| Style-enhanced Paraphrase | Paraphrasing with stylistic variations |
| Single-step Implicit Reasoning | Paraphrased knowledge + single-step reasoning QA |
| Single-step Explicit Reasoning | Paraphrased knowledge + single-step reasoning question + detailed reasoning process + answer |

All scenarios guarantee that each piece of knowledge is injected **20 times** to eliminate the impact of data volume differences.

### Training Settings

- **Model**: LLaMA 3-8B
- **Method**: Continual Pre-training (CPT) to avoid SFT-induced hallucinations
- **Data Ratio**: 1:1 mixture of training data and general instructions
- **Learning Rate**: 3e-5

## Experiments

### Key Finding 1: Repetitive Learning → Memorization

In the Duplicate scenario, the 0-shot memorization score improves stably with the number of repetitions, saturating at around **95 points**. However:
- Under the 3-shot setting, the memorization score is significantly lower than 0-shot → memorized knowledge is unstable and easily interfered with by context.
- Retrieval and reasoning scores under Duplicate are extremely low → memorized knowledge is isolated, lacking connection with other knowledge.

### Key Finding 2: Expression Diversity → Retrieval

Performance of knowledge retrieval scores under different injection scenarios:

| Injection Scenario | Retrieval Score Trend |
|----------|-------------|
| Duplicate | Consistently around 20, no improvement |
| Vanilla Paraphrase | Significant improvement |
| Style-enhanced Paraphrase | **Further substantial improvement (optimal)** |

- Expression diversity is the critical bridge from memorization to retrieval.
- Style-enhanced paraphrasing performs better than vanilla paraphrasing, indicating that the **heterogeneity** of expression (rather than just different phrasings) is key.

### Key Finding 3: Explicit Reasoning Patterns → Deep Reasoning (Table 1)

Injection effects on 2-step and 3-step reasoning tasks:

| Injection Scenario | Novel 2-step (3S-CoT) | Novel 3-step (3S-CoT) |
|----------|-------------------|-------------------|
| Duplicate | 3.3 | 3.7 |
| Style-enhanced Paraphrase | 31.3 | 24.7 |
| Single-step Implicit Reason | 34.3 | 31.7 |
| Single-step Explicit Reason | **41.0** | **49.3** |

**Key Conclusions**:
- Implicit reasoning improves zero-shot multi-step reasoning (28.7→41.7).
- Explicit reasoning performs best under 3-shot CoT (49.3 vs 31.7).
- Training with only single-step explicit reasoning generalizes to multi-step reasoning and new entities ← **Most important finding**

### Key Finding 4: LLMs Excel at Shallow Association, Deep Association Requires Explicit Reasoning (Table 2)

| Injection Scenario | Shallow Association 2-step (3S-CoT) | Deep Association 3-step (3S-CoT) |
|----------|-------------------|-------------------|
| Duplicate | 7.7 | 6.0 |
| Style-enhanced Paraphrase | 41.0 | 33.3 |
| Single-step Explicit Reason | **48.3** | **57.3** |
| Baseline (Uninjected Old Knowledge) | 64.0 | 55.3 |

- Paraphrased injection is sufficient to bring shallow association scores to around 45 (close to the baseline of 64).
- However, deep association (3-step) requires explicit reasoning injection to restore to baseline levels.

### Ablation Study: Impact of Knowledge Types

| Knowledge Type | Reasoning Performance | Cause Analysis |
|----------|---------|----------|
| Novel | Low | New entities lack existing reasoning frameworks |
| Updated | **High** | Existing reasoning frameworks of entities can be reused |
| Incremental | Medium | Lies in between |

**Insight**: Updated knowledge is easier to reach reasoning-level injection than Novel knowledge, as the model already possesses reasoning paths for the relevant entities.

### Ablation Study: General Instruction Ratio (Table 3)

| Training Ratio (Knowledge:Instruction) | Novel 3-step Reasoning (3S-CoT) |
|---------------------|----------------------|
| 2:1 | 6.3 |
| 1:1 | **49.3** |
| 1:2 | 54.7 |

General instruction data is crucial for knowledge reasoning—when the ratio shifts from 2:1 to 1:1, the 3-step reasoning score rockets from 6.3 to 49.3.

### Ablation Study: Expression Diversity Threshold

Increasing the number of paraphrased variants of the same knowledge (2→5) continuously improves the retrieval score until it **saturates at 4 variants**. Further increasing diversity beyond 4 variants yields no additional improvement → an optimal diversity threshold exists.

### Error Analysis

Main error sources for complex reasoning tasks:
- **Novel Knowledge**: 50%+ of errors stem from **incorrect question decomposition paths**
- **Updated Knowledge**: **Incorrect knowledge recall** is the main error source (conflict between old and new knowledge causes hallucinations)

## Highlights & Insights

1. **First systematic four-level knowledge injection framework**: Deconstructs the vague "knowledge update" into four clear levels: memorization → retrieval → reasoning → association, providing a unified evaluation metric for future research.
2. **Precise mapping from method to level**:
    - Memorization ← Repetitive training
    - Retrieval ← Diversified expressions
    - Reasoning ← Explicit reasoning patterns
    - Association ← Explicit reasoning + bridging old and new knowledge
3. **Generalization from single-step training to multi-step**: Utilizing only single-step explicit reasoning training data successfully achieves significant improvements in 2- or 3-step reasoning, demonstrating that reasoning capabilities can generalize.
4. **Practical guidance for knowledge types**: Updated knowledge is easier to inject at a deep level (leveraging existing reasoning paths), while Novel knowledge requires more explicit reasoning training.
5. **Crucial role of general instruction data**: A 1:1 mix of general instructions is a necessary condition for knowledge reasoning; training exclusively on knowledge leads to a complete loss of reasoning capabilities.

## Limitations & Future Work

1. **Only utilizing LLaMA 3-8B**: Experiments are limited to a single model; results may differ across various scales and architectures.
2. **Exploring only the CPT method**: The approach to knowledge injection is restricted to continual pre-training, without evaluating alternative methods such as LoRA fine-tuning, knowledge editing, or RAG.
3. **Limited types of reasoning operations**: Only two atomic reasoning operations, combination and comparison, are defined, leaving out richer reasoning types such as induction, analogy, and counterfactuals.
4. **Limitations of synthetic knowledge**: Knowledge about fictional entities may not fully reflect the complexity of real-world knowledge injection (e.g., commonsense reasoning, implicit knowledge).
5. **Fixed 20 injections per knowledge pack**: In practical scenarios, the frequency of knowledge occurrences varies significantly. A fixed frequency might obscure the difficulty of injecting long-tail knowledge.

## Related Work & Insights

- **Knowledge Memorization & Retrieval**: Carlini et al. (2021) training data extraction, Physics of LM Part 3.1 (Allen-Zhu & Li) knowledge storage and extraction, MQuAKE (Zhong et al., 2023) multi-hop knowledge editing
- **Knowledge Injection Methods**: Continual pre-training (Zhang et al., 2023; Jang et al., 2022), knowledge editing (Zhang et al., 2024)
- **Knowledge Reasoning**: Physics of LM Part 3.2 (Allen-Zhu & Li, 2023) knowledge manipulation, Grokked Transformers (Wang et al., 2024a) implicit reasoning
- **Knowledge & Hallucination**: Gekhman et al. (2024) SFT-induced hallucination, WikiFactDiff (Khodja et al., 2024) temporal knowledge difference
- **Knowledge-Augmented LLMs**: Adapting LLMs via Reading Comprehension (Cheng et al., 2023)

## Rating

⭐⭐⭐⭐ — Clear framework, systematic and comprehensive experiments, and findings with practical guiding value (especially the explicit reasoning → multi-step generalization and general instruction ratio). However, the experimental breadth is insufficient as it only uses a single model with the CPT method, and the ecological validity of synthetic knowledge is also limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A General Knowledge Injection Framework for ICD Coding](a_general_knowledge_injection_framework_for_icd_coding.md)
- [\[ACL 2025\] Structure-aware Domain Knowledge Injection for Large Language Models](structure-aware_domain_knowledge_injection_for_large_language_models.md)
- [\[ACL 2025\] Mitigating Negative Interference in Multilingual Sequential Knowledge Editing through Null-Space Constraints](mitigating_negative_interference_in_multilingual_sequential_knowledge_editing_th.md)
- [\[ACL 2025\] ToxEdit: Adaptive Detoxification Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing](adaptive_detoxification_safeguarding_general_capabilities_of_llms_through_toxici.md)
- [\[ACL 2025\] ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains](chainedit_propagating_ripple_effects_in_llm.md)

</div>

<!-- RELATED:END -->
