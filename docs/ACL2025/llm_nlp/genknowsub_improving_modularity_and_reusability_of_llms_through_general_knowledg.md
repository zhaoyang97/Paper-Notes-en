---
title: >-
  [Paper Note] GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction
description: >-
  [ACL 2025][LLM (Other)][LoRA] Proposes GenKnowSub (General Knowledge Subtraction), which trains a general knowledge LoRA on the Wikipedia corpus and subtracts it from task-specific LoRAs ($LoRA_{res}^i = LoRA_{ts}^i - LoRA_g$) to obtain purer residual modules. Combined with the Arrow routing algorithm to dynamically select the most relevant modules, it improves zero-shot transfer average accuracy by 1.6% on Phi-3, with even larger gains in cross-lingual scenarios (German +3.9…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LoRA"
  - "General Knowledge Subtraction"
  - "Arrow Routing"
  - "Zero-Shot Transfer"
  - "Modularity"
date: 2026-05-08
content_hash: c11998874f2a35da
---

# GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction

**Conference**: ACL 2025  
**arXiv**: [2505.10939](https://arxiv.org/abs/2505.10939)  
**Code**: [https://github.com/saharsamr/Modular-LLM](https://github.com/saharsamr/Modular-LLM)  
**Area**: Modular LLMs / Parameter-Efficient Transfer Learning  
**Keywords**: LoRA, General Knowledge Subtraction, Arrow Routing, Zero-Shot Transfer, Modularity

## TL;DR

Proposes GenKnowSub (General Knowledge Subtraction), which trains a general knowledge LoRA on the Wikipedia corpus and subtracts it from task-specific LoRAs ($LoRA_{res}^i = LoRA_{ts}^i - LoRA_g$) to obtain purer residual modules. Combined with the Arrow routing algorithm to dynamically select the most relevant modules, it improves zero-shot transfer average accuracy by 1.6% on Phi-3, with even larger gains in cross-lingual scenarios (German +3.9%, French +3.6%).

## Background & Motivation

**Background**: Modular zero-shot transfer is an important paradigm for validating LLM generalization to unseen tasks. A typical two-stage approach involves training task-specific modules through parameter-efficient fine-tuning methods like LoRA, followed by using a routing function to dynamically select or combine modules for new tasks. The Arrow algorithm is the current state-of-the-art post-hoc routing method, dynamically selecting the most relevant modules at the token level without requiring extra training.

**Limitations of Prior Work**: Task-specific LoRA modules store not only task-specific knowledge but also redundant general knowledge that has already been acquired during the base model's pre-training. This redundancy leads to: (1) different task modules being insufficiently distinctive, which makes it difficult for the routing algorithm to accurately differentiate them; (2) duplicated accumulation of general knowledge when combining modules, which harms generalization.

**Key Challenge**: Task-specific knowledge and general knowledge are entangled within LoRA modules, degrading the routing accuracy and zero-shot transfer performance of modular approaches.

**Goal**: Untangle general knowledge from task-specific modules to improve module distinctiveness and subsequent composition.

**Key Insight**: Leveraging the "forgetting via negation" principle from task arithmetic, a general knowledge LoRA is trained and then arithmetically subtracted from each task LoRA.

**Core Idea**: Subtracting general knowledge = purer task modules = better routing distinctiveness = superior zero-shot transfer.

## Method

### Overall Architecture

A three-stage pipeline: (1) train a library of task-specific LoRA modules on multi-task datasets; (2) train a general knowledge LoRA on the Wikipedia corpus, and subtract it from each task LoRA to obtain residual LoRAs; (3) employ the Arrow routing algorithm to dynamically select the top-k most relevant residual modules and perform weighted combination for each token at each layer.

### Key Designs

1. **General Knowledge LoRA Training and Subtraction (GenKnowSub)**:

    - **Function**: Trains a LoRA module utilizing causal language modeling objectives on a small-scale Wikipedia corpus to act as a "general knowledge signature," which is then subtracted from each task LoRA.
    - **Mechanism**: Formulated as $LoRA_{res}^i = LoRA_{ts}^i - LoRA_g$, where $LoRA_{ts}^i$ is the $i$-th task module, and $LoRA_g$ is the general knowledge module. It assumes that Wikipedia fine-tuning acts as a "flashback" mechanism, activating general knowledge already present in the base model and encoding it into LoRA. After subtraction, residual modules retain features unique to the task.
    - **Design Motivation**: General knowledge redundancy renders different task modules overly similar. Subtracting it increases inter-module distinctiveness, enhancing the routing capacity of Arrow. This represents a modular-level application of the "forgetting as learning" philosophy.
    - **Implementation**: One LoRA is trained per language (English, French, German) on 5,000 Wikipedia segments each, alongside their average $LoRA_{avg}$.

2. **Arrow Routing Algorithm (Dynamic Task Adaptation)**:

    - **Function**: Dynamically selects the top-k most relevant residual modules at each model layer for each input token and constructs weighted combinations to form the LoRA for the current token.
    - **Mechanism**: Performs Singular Value Decomposition (SVD) on each residual LoRA to extract the top right singular vector as the prototype. The input token is projected onto this prototype space, selecting the top-k modules with the highest similarity, which are then weighted and summed using softmax-normalized coefficients. The final forward pass is: $y_t^l = W_0^l x_t^l + B_t^l A_t^l x_t^l$.
    - **Design Motivation**: Token-level routing provides finer granularity than sentence-level routing, as different tokens within the same sentence may require knowledge from different task modules.

3. **Exploration of Cross-Lingual General Knowledge**:

    - **Function**: Generates general LoRAs by training on English, French, and German Wikipedia individually, as well as their average, and evaluates the effect of subtracting different language-specific LoRAs.
    - **Mechanism**: Even if task modules are exclusively trained on English, subtracting non-English (e.g., French) general LoRAs still yields performance improvements, indicating that general knowledge is shared cross-lingually.
    - **Design Motivation**: To validate the language-agnostic nature of GenKnowSub.

## Key Experimental Results

### Main Results (Phi-3 English Reasoning Benchmark, Zero-Shot)

| Method | PIQA | BoolQ | SWAG | HellaSwag | ARC-E | ARC-C | WG | OQA | BBH | Avg |
|------|------|-------|------|-----------|-------|-------|-----|-----|-----|-----|
| Phi-3 Baseline | 78.2 | 81.5 | 69.0 | 73.6 | 71.8 | 44.5 | 66.0 | 42.8 | 42.8 | 63.4 |
| Arrow | 80.2 | 80.0 | 69.0 | 71.9 | 80.5 | 53.9 | 66.0 | 47.4 | 41.2 | 65.6 |
| GenKnowSub (Avg) | **80.0** | **82.5** | **72.7** | **73.5** | **82.3** | **55.9** | 64.6 | **49.6** | **43.5** | **67.2** |

### Cross-Lingual Experiments (Phi-3 Zero-Shot)

| Language | Method | HellaSwag | ARC-C | XNLI | MMLU | Avg |
|------|------|-----------|-------|------|------|-----|
| German | Arrow | 48.6 | 40.9 | 43.5 | 35.4 | 42.1 |
| German | GenKnowSub (De) | 50.6 | 43.0 | 50.4 | 37.0 | **45.2** |
| French | Arrow | 55.3 | 41.6 | 44.4 | 34.8 | 44.0 |
| French | GenKnowSub (Fr) | 57.8 | 43.0 | 53.7 | 36.1 | **47.6** |

### Ablation Study

| Method | Avg (English 9 Benchmarks) |
|------|-----------------|
| Shared (Single LoRA) | 61.8 |
| Mean Normalization | 62.3 |
| Arrow | 65.6 |
| GenKnowSub (Avg) | **67.2** |

### Key Findings
- GenKnowSub outperforms Arrow by an average of 1.6% in English, 3.1% in German, and 3.6% in French, showing more prominent gains in cross-lingual settings.
- Mean Normalization (subtracting the average of task modules) shows inconsistent results, confirming that target-subtraction of "general knowledge" rather than just the "average" is crucial.
- Subtracting a general LoRA of a non-task language (such as French or German) is still effective for English tasks, demonstrating that general knowledge is shared across languages.
- GenKnowSub also outperforms Arrow on open-ended generation tasks (Super-Natural Instructions), scoring 46.91 vs 45.44 in Rouge-L.
- On Phi-2 (possessing weaker multilingual capabilities), GenKnowSub is effective only in English and fails cross-lingually, suggesting that this method relies on the base model's inherent multilingual capacity.

## Highlights & Insights
- **An intuitive counter-design of "improvement via subtraction"**—enhancing performance by removing information. The core insight is that redundant general knowledge blurs module distinctiveness, and its extraction actually refines routing precision. This represents an elegant application of task arithmetic to modular LLMs.
- **Portability of cross-lingual general knowledge**—subtracting French general knowledge improves English task performance, revealing that general knowledge acts as a cross-lingual "background noise." This provides practical guidelines for multilingual modular approaches.

## Limitations & Future Work
- Only validated on Phi-3 and Phi-2; larger scale models (such as Llama-3 70B) remain untested.
- Only covers three high-resource languages (English, French, and German); performance on low-resource languages is unknown.
- General LoRA training corpus is fixed at 5,000 Wikipedia segments; optimal data volumes and sources are yet to be explored.
- The subtraction is a parameter-free linear operation; future work could explore learning-based knowledge decoupling (e.g., orthogonalization).
- No systematic analysis was conducted on the choice of the k value in top-k routing.

## Related Work & Insights
- **vs Arrow (Ostapenko et al., 2024)**: Arrow directly routes raw task LoRAs; GenKnowSub subtracts general knowledge before routing, making subtraction the key enhancement.
- **vs Task Arithmetic (Ilharco et al., 2023)**: Task Arithmetic performs task vector addition/subtraction at the model parameter level; GenKnowSub applies general knowledge subtraction at the module level.
- **vs LoRAHub (Huang et al., 2024)**: LoRAHub requires training composition weights on downstream data; GenKnowSub operates in a completely zero-shot manner.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The concept of general knowledge subtraction is simple yet novel, and the discovery of cross-lingual portability is valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensively covered with 9 English benchmarks, cross-lingual tasks, open-ended generation, ablation studies, and validation on Phi-2.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly presented methodology and experiments, with a convincing ablation design.
- **Value**: ⭐⭐⭐⭐ Directly contributes to modular LLMs and zero-shot transfer learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes](improving_preference_extraction_in_llms_by_identifying_latent_knowledge_through_.md)
- [\[ACL 2025\] Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)
- [\[ACL 2025\] Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](knowledge_boundary_crosslingual.md)
- [\[ACL 2025\] Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](derta_decoupled_refusal.md)
- [\[ACL 2025\] A Training-free LLM-based Approach to General Chinese Character Error Correction](a_training-free_llm-based_approach_to_general_chinese_character_error_correction.md)

</div>

<!-- RELATED:END -->
