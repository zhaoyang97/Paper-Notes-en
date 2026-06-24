---
title: >-
  [Paper Note] Dynamic Label Name Refinement for Few-Shot Dialogue Intent Classification
description: >-
  [ACL 2025][Dialogue Systems][intent classification] Proposes a dynamic label name refinement method that utilizes LLMs to dynamically generate more distinctive intent label names (e.g., "Verify PAN" → "Verify PAN card details") based on retrieved examples in retrieval-based ICL intent classification. This effectively reduces confusion between semantically similar intents, consistently improving accuracy by 2.07%-7.51% across 6 datasets.
tags:
  - "ACL 2025"
  - "Dialogue Systems"
  - "intent classification"
  - "few-shot learning"
  - "label refinement"
  - "in-context learning"
  - "LLM"
  - "semantic disambiguation"
date: 2026-05-08
content_hash: f2b9a2923163463d
---

# Dynamic Label Name Refinement for Few-Shot Dialogue Intent Classification

**Conference**: ACL 2025  
**arXiv**: [2412.15603](https://arxiv.org/abs/2412.15603)  
**Code**: None  
**Area**: Others  
**Keywords**: intent classification, few-shot learning, label refinement, in-context learning, LLM, semantic disambiguation

## TL;DR

Proposes a dynamic label name refinement method that utilizes LLMs to dynamically generate more distinctive intent label names (e.g., "Verify PAN" → "Verify PAN card details") based on retrieved examples in retrieval-based ICL intent classification. This effectively reduces confusion between semantically similar intents, consistently improving accuracy by 2.07%-7.51% across 6 datasets.

## Background & Motivation

**Background**: Dialogue intent classification is a core component of task-oriented dialogue systems. Retrieval-based ICL (In-Context Learning) methods have become mainstream for few-shot intent classification: retrieving similar examples for each test query and constructing prompts for LLMs to classify.

**Limitations of Prior Work**: (1) In practical systems, the number of intent categories can reach dozens or hundreds, with many intents overlapping significantly in semantics—for example, in banking customer service, the cosine embedding similarity among "Verify PAN", "Bank verification details", and "Account not verified" is as high as 0.91; (2) The retrieved examples often exhibit significant semantic overlap across different intent categories, leading to greater model confusion; (3) Chain-of-Thought (CoT) methods are easily misled by similar label names when reasoning is required, sometimes even degrading performance.

**Key Challenge**: Label names themselves are not distinctive enough. When multiple intent labels are highly similar in the embedding space, it is difficult to distinguish them accurately, whether through direct classification or CoT reasoning.

**Goal**: Reduce semantic overlap between labels by dynamically refining intent label names, thereby boosting classification accuracy.

**Key Insight**: Utilize LLMs to analyze the semantic relationships among groups of retrieved examples and dynamically generate more descriptive and distinctive label names.

**Core Idea**: Let the LLM refine intent labels first (making names more distinctive) before performing ICL classification with the refined labels—using "renaming" to resolve "indistinguishability".

## Method

### Overall Architecture

The method consists of three steps: (1) Retrieval—use SentenceTransformer to retrieve the top-20 most similar training examples for each test query and group them by intent; (2) Label Refinement—use the LLM to analyze these example groups, determine if label names need refinement, and generate more descriptive labels; (3) Classification—construct an ICL prompt using the refined labels and examples to perform the final intent classification.

### Key Designs

1. **Retrieval-Based Example Selection**:

    - **Function**: Retrieves semantically similar training examples for each test input.
    - **Mechanism**: Encodes queries and the training set using a pre-trained SentenceTransformer (Reimers 2019) and retrieves the top-20 most similar examples based on cosine similarity. Examples are grouped by their original intents to provide context for subsequent label refinement.
    - **Design Motivation**: The retrieved examples are not only used for prompt construction in ICL but, more importantly, they provide "evidence" of semantic relationships between intents. It is precisely these similar yet distinct-intent examples that reveal the root causes of label confusion.

2. **Dynamic Label Refinement**:

    - **Function**: Leverages the LLM to analyze the examples within each intent group, deciding whether to refine the labels and generating new ones.
    - **Mechanism**: Designs a specialized prompt for each retrieved intent group, asking the LLM to evaluate the semantic relationship between the labels and the examples. The model determines if the original label is descriptive enough—if not, it generates an enhanced version; otherwise, it keeps the original. For example, "verify_pan" → "verify_pan_card_details" to increase its semantic distance from "bank_verification_details".
    - **Design Motivation**: Dynamic refinement (instead of static pre-definition) is necessary because different test queries retrieve different groups of examples. Adjustments must target specific points of confusion within the current context. Experiments confirm that dynamic refinement significantly outperforms static refinement.

3. **ICL Classification with Refined Labels**:

    - **Function**: Uses refined labels instead of original labels for the final classification.
    - **Mechanism**: Constructs a prompt containing examples with refined labels, the test query, and classification instructions. Examples are ordered from lowest similarity to highest (Milios et al. 2023). The same LLM is utilized for both label refinement and final classification to ensure consistency between refinement understanding and classification decisions.
    - **Design Motivation**: This two-step process allows the same LLM to first comprehend the differences between intents before classifying, which is more effective than a single-step classification approach.

## Key Experimental Results

### Main Results (Llama3-8b-inst)

| Dataset | Raw (Baseline) | CoT | Refined | Gain |
|--------|-----------|-----|-----------------|------|
| HWU64 | 88.10 | 87.17 (-0.93) | **89.03** | +0.93 |
| BANKING77 | 85.88 | 85.48 (-0.40) | **87.95** | +2.07 |
| CLINC150 | 95.03 | 95.13 (+0.10) | **95.51** | +0.48 |
| CUREKART | 89.76 | 89.76 (+0.00) | **91.94** | +2.18 |
| POWERPLAY11 | 70.87 | 67.00 (-3.87) | **76.10** | +5.23 |
| SOFMATTRESS | 82.61 | 81.10 (-1.51) | **85.40** | +2.79 |

### Ablation Study

| Configuration | Description | Conclusion |
|------|------|------|
| Original label similarity vs Refined label similarity (Llama3-8b) | 0.86 → 0.74 | Refinement significantly reduces semantic overlap between labels |
| Dynamic refinement vs Static refinement | Dynamic consistently outperforms static | Context-aware refinement is more effective |
| Small model refinement + Large model classification | Performance drops under cross-configurations | Refinement and classification should be completed by the same model |

### Cross-Model Consistency

| Dataset | Qwen2.5-7b Gain | Qwen2.5-1.5b Gain |
|--------|----------------|-------------------|
| BANKING77 | +1.62 | +4.03 |
| SOFMATTRESS | +3.61 | +7.51 |
| POWERPLAY11 | +3.56 | +2.26 |

### Key Findings
- Label refinement consistently improves performance across all 6 datasets and 3 models, whereas CoT actually degrades performance on 4/6 datasets.
- The gains are most significant on datasets with high semantic overlap: POWERPLAY11 (+5.23%) and SOFMATTRESS (+2.79%/7.51%), where intent labels are semantically most similar.
- Smaller models benefit more: Qwen2.5-1.5b achieves a 7.51% gain on SOFMATTRESS (vs. 2.79% for Llama3-8b), showing that label refinement effectively compensates for the semantic understanding deficiencies of smaller models.
- The embedding similarity of refined labels is consistently lower (Llama3: 0.86→0.74), verifying that the method indeed pulls intents apart in the semantic space.
- Cross-configuration experiments show that refinement and classification should be executed by the same model—different models hold different understandings of refined labels.

## Highlights & Insights
- The method is extremely simple yet highly effective—requiring only the insertion of a "label refinement" step into the standard ICL pipeline, with no fine-tuning, no extra data, and no architectural modifications. CoT is actually detrimental (averaging -1.2%), while label refinement is consistently effective.
- The embedding similarity analysis of refined labels provides an intuitively convincing explanation—it reduces the "crowdedness" of the label space, providing more "breathing room" for the classifier.

## Limitations & Future Work
- Each test sample requires an additional LLM call for label refinement, increasing inference latency and cost.
- Experiments were conducted only in a 10-shot setting, leaving its effectiveness in many-shot or zero-shot scenarios unverified.
- Refinement quality depends on the capability of the LLM—if the LLM has poor domain understanding, refinement may instead mislead the model.
- Comparison with training-based few-shot methods (e.g., prototypical networks) was not conducted.

## Related Work & Insights
- **vs Milios et al. 2023**: Foundational work for retrieval-based ICL intent classification. This paper adds a label refinement step on top of their framework.
- **vs CoT (Wei et al. 2023)**: CoT actually harms performance in intent classification (the reasoning chain is misled by similar labels). Label refinement directly addresses confusion at the label level rather than the reasoning level.
- **vs Sung et al. 2023; Cho et al. 2024**: Focus on the challenges of few-shot intent classification but address them from the model architecture perspective. This paper approaches the problem from the perspective of data representation (label names).

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of "renaming" is simple, intuitive, and effective. The angle of entry is fresh—solving classification confusion from the perspective of label naming rather than model capability.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 6 datasets + 3 models + multiple sets of ablation studies with good consistency, though it lacks comparisons with training-based methods.
- Writing Quality: ⭐⭐⭐⭐ Clear problem-oriented focus, and the example in Figure 1 is highly intuitive and easy to understand.
- Value: ⭐⭐⭐⭐ A plug-and-play label refinement strategy that is directly applicable to various ICL classification scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue](../../ACL2026/dialogue/context-agent_dynamic_discourse_trees_for_non-linear_dialogue.md)
- [\[ACL 2025\] ReflectDiffu: Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework](reflectdiffu_empathetic_response.md)
- [\[ACL 2026\] ReacTOD: Bounded Neuro-Symbolic Agentic NLU for Zero-Shot Dialogue State Tracking](../../ACL2026/dialogue/reactod_bounded_neuro-symbolic_agentic_nlu_for_zero-shot_dialogue_state_tracking.md)
- [\[ACL 2026\] Codebook-Injected Dialogue Segmentation for Multi-Utterance Constructs Annotation: LLM-Assisted and Gold-Label-Free Evaluation](../../ACL2026/dialogue/codebook-injected_dialogue_segmentation_for_multi-utterance_constructs_annotatio.md)
- [\[ACL 2025\] Dialogue Systems for Emotional Support via Value Reinforcement](dialogue_systems_for_emotional_support_via_value_reinforcement.md)

</div>

<!-- RELATED:END -->
