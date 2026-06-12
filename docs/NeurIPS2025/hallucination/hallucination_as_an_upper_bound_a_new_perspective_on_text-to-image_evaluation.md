---
title: >-
  [Paper Note] Hallucination as an Upper Bound: A New Perspective on Text-to-Image Evaluation
description: >-
  [NeurIPS 2025 (Workshop: Generative and Protective AI for Content Creation)][Hallucination Detection][Hallucination] This paper proposes a definition of hallucination in text-to-image (T2I) models as **bias-driven deviat…
tags:
  - "NeurIPS 2025 (Workshop: Generative and Protective AI for Content Creation)"
  - "Hallucination Detection"
  - "Hallucination"
  - "T2I Evaluation"
  - "Alignment Upper Bound"
  - "Bias Detection"
  - "Taxonomy"
date: 2026-05-08
content_hash: e5876081d001cd3d
---

# Hallucination as an Upper Bound: A New Perspective on Text-to-Image Evaluation

**Conference**: NeurIPS 2025 (Workshop: Generative and Protective AI for Content Creation)  
**arXiv**: [2509.21257](https://arxiv.org/abs/2509.21257)  
**Code**: None  
**Area**: Hallucination Detection  
**Keywords**: Hallucination, T2I Evaluation, Alignment Upper Bound, Bias Detection, Taxonomy

## TL;DR

This paper proposes a definition of hallucination in text-to-image (T2I) models as **bias-driven deviation**, establishes a taxonomy of three hallucination categories—attribute, relation, and object—and argues that hallucination evaluation serves as an "upper bound" for prompt alignment evaluation, thereby revealing hidden model biases.

## Background & Motivation

### State of the Field

Hallucination has been extensively studied in large language models (LLMs) and vision-language models (VLMs):

| Area | Hallucination Definition | Research Depth |
|:---|:---|:---|
| LLM | Generating content inconsistent with facts | Deep (numerous surveys and benchmarks) |
| VLM | Generating descriptions inconsistent with images | Actively developing (HaluEval, THRONE, etc.) |
| **T2I** | **Not clearly defined** | **Nearly absent** |

Existing T2I evaluation methods primarily focus on **alignment**:

- TIFA: QA-based prompt faithfulness
- GenEval: Compositional generation capability
- T2I-CompBench: Compositionality benchmark
- VQAScore: Visual question answering-based scoring

These methods only check "whether what the prompt requires is present," while ignoring "what the model generates beyond the prompt."

### Lower Bound vs. Upper Bound

The paper presents a key insight:

| Evaluation Dimension | Meaning | Type |
|:---|:---|:---|
| Alignment Evaluation | Are the elements required by the prompt present? | **Lower Bound** |
| Hallucination Evaluation | What has the model added beyond the prompt? | **Upper Bound** |

Focusing solely on alignment provides only a lower bound on model performance. A complete evaluation must also detect prompt-undriven content added by the model itself—i.e., hallucination.

## Method

### Overall Architecture

This paper is a **position paper** whose core contributions are conceptual:
1. Defining hallucination in T2I generation
2. Establishing a taxonomy of three hallucination categories
3. Distinguishing hallucination from alignment errors
4. Arguing for the necessity of hallucination evaluation as an upper bound

### Key Designs

#### Hallucination vs. Alignment Error

| Phenomenon | Alignment Error | Hallucination |
|:---|:---|:---|
| Definition | Failure to correctly render prompt-specified content | Addition of content not specified by the prompt |
| Example | "Red car" is generated as blue | "Car" generates pedestrians on the road |
| Direction | Model omission/error | Model addition |
| Source | Insufficient understanding/rendering capability | Internal model bias/prior |

#### Hallucination Taxonomy

##### 1. Object Hallucination

Generating entities not mentioned in the prompt.

Formally: let prompt $P$ specify object set $O = \{o_1, \ldots, o_n\}$. If the generated image contains a non-empty set $O'$ where $O' \cap O = \emptyset$, then $O'$ constitutes object hallucination.

| Prompt | Expected Content | Hallucinated Content | Bias Source |
|:---|:---|:---|:---|
| "a bowl of apples" | Bowl of apples | Oranges appear in the bowl | Scene completion bias |
| "a horse" | Horse | Rider appears on horse | Co-occurrence statistics |
| "a street with cars" | Street with cars | Pedestrians, bicycles appear | Scene completeness bias |

##### 2. Attribute Hallucination

The model assigns specific visual attributes to objects whose attributes are not specified in the prompt.

Formally: let prompt $P$ include object $o$ but no explicit attributes. If $o$ in the image possesses attribute $a'$ (not implied by $P$), then $a'$ constitutes attribute hallucination.

| Prompt | Expected Output | Hallucinated Attribute | Reflected Bias |
|:---|:---|:---|:---|
| "a doctor" | Doctor (neutral) | Male, white coat | Gender/occupational stereotype |
| "a wedding cake" | Wedding cake | White, multi-tiered | Cultural default |
| "a child" | Child | Smiling, outdoors, neat clothing | Idealized emotional default |

##### 3. Relation Hallucination

The model inserts relationships between objects that are not described in the prompt.

Formally: let prompt $P$ include objects $O = \{o_1, o_2\}$ with no explicit relation. If the image contains relation $r$ (not implied by $P$), then $r$ constitutes relation hallucination.

| Prompt | Expected Composition | Hallucinated Relation | Reflected Bias |
|:---|:---|:---|:---|
| "a man and a dog" | Man and dog co-present | Man walking dog (on leash) | Control/ownership association |
| "a woman and a laptop" | Woman and laptop | Woman typing | Work scenario association |
| "a child and a book" | Child and book | Child reading | Learning narrative association |

### Loss & Training

This paper involves no training. It is a conceptual framework paper intended to lay the groundwork for future T2I hallucination benchmarks and evaluation methods.

## Key Experimental Results

### Conceptual Framework Comparison

As a position paper, this work contains no conventional experiments. The core contribution lies in conceptual organization. The following compares existing evaluation dimensions:

| Evaluation Method | Detects Missing Objects | Detects Attribute Errors | Detects Relation Errors | Detects Extra Objects | Detects Implicit Bias |
|:---|:---|:---|:---|:---|:---|
| TIFA | ✓ | ✓ | Partial | ✗ | ✗ |
| GenEval | ✓ | ✓ | ✓ | ✗ | ✗ |
| T2I-CompBench | ✓ | ✓ | ✓ | ✗ | ✗ |
| VQAScore | ✓ | ✓ | Partial | ✗ | ✗ |
| iHallA | ✓ | ✓ | ✓ | Partial | ✗ |
| **Proposed Framework** | ✓ | ✓ | ✓ | **✓** | **✓** |

### Completeness Comparison of Evaluation Dimensions

| Dimension | Alignment Evaluation (Lower Bound) | Hallucination Evaluation (Upper Bound) |
|:---|:---|:---|
| Core Question | Is what the prompt requires present? | What extra content has the model added? |
| Captured Bias | Capability deficiency | Implicit bias and priors |
| Evaluation Direction | Missing content detection | Added content detection |
| Completeness | Necessary but insufficient | Complementary dimension |
| Existing Coverage | Extensive | **Nearly absent** |

### Key Findings

1. **Alignment evaluation is incomplete**: Existing T2I evaluation methods only check "whether anything is missing," not "whether anything extra has been added." A complete evaluation picture requires both.

2. **Hallucination reveals hidden biases**: Object hallucination reflects scene completion bias, attribute hallucination reflects social stereotypes, and relation hallucination reflects over-learned associations—all of which are entirely overlooked by current alignment evaluation.

3. **Independence of the three hallucination categories**: Object, attribute, and relation hallucination are three independent dimensions, each involving distinct evaluation challenges (entity detection vs. attribute recognition vs. relation reasoning).

4. **Implications for model deployment**: Hallucination undermines controllability, neutrality, and trustworthiness—factors critical to real-world deployment that are neglected in existing evaluations.

## Highlights & Insights

- **The lower/upper bound analogy is compelling**: Framing alignment as a lower bound and hallucination as an upper bound provides a clear conceptual framework for evaluation.
- **Social bias perspective**: Attribute hallucination directly connects to AI fairness issues (e.g., gender and cultural stereotypes).
- **Addressing an evaluation blind spot**: The paper explicitly identifies a systematic gap in T2I evaluation.
- **Practical guidance**: The taxonomy provides concrete categorical dimensions for constructing new T2I hallucination benchmarks.

## Limitations & Future Work

- **Lack of empirical validation**: As a position paper, no quantitative experiments or benchmark results are provided.
- **No evaluation methodology**: A taxonomy is proposed but no concrete detection methods or evaluation metrics are designed.
- **Ambiguous boundary of "hallucination"**: Scene completion by the model is sometimes reasonable (e.g., generating backgrounds); clearer criteria are needed for when such additions constitute hallucination.
- **No discussion of connections to VLM hallucination**: T2I hallucination evaluation may benefit from methods developed for VLM hallucination detection.
- **Workshop paper length constraints**: Many ideas are only briefly introduced without in-depth development.

## Related Work & Insights

- **Huang et al. (2025)**: Survey on LLM hallucination
- **Bai et al. (2024)**: VLM hallucination research
- **TIFA (Hu et al., 2023)**: QA-based T2I evaluation
- **GenEval (Ghosh et al., 2023)**: Compositional generation evaluation
- **T2I-CompBench (Huang et al., 2023)**: Compositionality benchmark
- **iHallA (Lim et al., 2025)**: Partially addresses T2I hallucination
- The proposed taxonomy can guide future benchmark design, particularly for bias auditing.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic framework for T2I hallucination
- **Technical Depth**: ⭐⭐ — Conceptual work with no technical methods or experiments
- **Practicality**: ⭐⭐⭐ — Provides direction for future benchmark design but is not directly applicable
- **Clarity**: ⭐⭐⭐⭐⭐ — Clear writing with vivid examples
- **Overall Score**: 6/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights](../../ACL2026/hallucination/rethinking_evaluation_for_llm_hallucination_detection_a_desiderata_a_new_rag-bas.md)
- [\[AAAI 2026\] Bridging Day and Night: Target-Class Hallucination Suppression in Unpaired Image Translation](../../AAAI2026/hallucination/bridging_day_and_night_target-class_hallucination_suppressio.md)
- [\[NeurIPS 2025\] Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers](generalization_or_hallucination_understanding_out-of-context_reasoning_in_transf.md)
- [\[NeurIPS 2025\] Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](causalllava_causal_disentanglement_for_mitigating_hallucinat.md)
- [\[NeurIPS 2025\] Intervene-All-Paths: Unified Mitigation of LVLM Hallucinations across Alignment Formats](intervene-all-paths_unified_mitigation_of_lvlm_hallucinations_across_alignment_f.md)

</div>

<!-- RELATED:END -->
