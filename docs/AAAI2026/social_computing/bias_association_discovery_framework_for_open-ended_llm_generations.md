---
title: >-
  [Paper Note] Bias Association Discovery Framework for Open-Ended LLM Generations
description: >-
  [AAAI 2026][Social Computing][Social Bias] This paper proposes the Bias Association Discovery Framework (BADF), which systematically extracts both known and unknown bias associations between demographic identities and descriptive concepts from LLM open-ended story generation, overcoming the limitation of prior methods that rely on predefined bias concepts.
tags:
  - "AAAI 2026"
  - "Social Computing"
  - "Social Bias"
  - "LLM Generation"
  - "Bias Association Discovery"
  - "Open-Ended Generation"
  - "Demographic Identity"
date: 2026-05-08
content_hash: 24970b4a93a8731e
---

# Bias Association Discovery Framework for Open-Ended LLM Generations

**Conference**: AAAI 2026
**arXiv**: [2508.01412](https://arxiv.org/abs/2508.01412)  
**Code**: [GitHub](https://github.com/JP-25/Discover-Open-Ended-Generation)  
**Area**: Human Understanding / LLM Bias Evaluation
**Keywords**: Social Bias, LLM Generation, Bias Association Discovery, Open-Ended Generation, Demographic Identity

## TL;DR

This paper proposes the Bias Association Discovery Framework (BADF), which systematically extracts both known and unknown bias associations between demographic identities and descriptive concepts from LLM open-ended story generation, overcoming the limitation of prior methods that rely on predefined bias concepts.

## Background & Motivation

**Background**: LLMs trained on large-scale real-world data inevitably encode social biases, leading to unfair representational harm across different demographic groups. Existing bias evaluation methods (e.g., cloze tests, multiple-choice QA) have made progress but largely rely on predefined identity–concept association pairs for measurement.

**Limitations of Prior Work**: Existing methods can only detect known biases (e.g., "elderly ↔ forgetful") and are unable to discover novel, unexpected bias associations latent in models. Even recent work such as BiasDora, which attempts open-ended discovery, is limited to word-level associations and cannot capture sentence-level or narrative-level complex bias patterns.

**Key Challenge**: LLMs are predominantly deployed in open-ended generation scenarios (e.g., story writing, dialogue), yet bias evaluation methods remain confined to constrained, template-based assessments—a severe mismatch between the evaluation paradigm and the usage paradigm.

**Goal**: To systematically discover and quantify both known and unknown bias associations from LLMs' open-ended free-form generated text.

**Key Insight**: Story generation is adopted as the open-ended task vehicle. By configuring diverse location and demographic identity combinations, LLMs are allowed to naturally expose their encoded social biases.

**Core Idea**: A three-stage framework, BADF, is designed—first extracting descriptive concepts from generated text, then filtering meaningful associations via frequency salience and statistical testing, and finally removing concepts that reflect only factual definitional exclusivity rather than bias.

## Method

### Overall Architecture

BADF comprises three core stages: (1) **Association Extraction**: comprehensively extracting descriptive concepts associated with demographic identities from open-ended generations; (2) **Salient Association Identification**: filtering statistically significant and identity-specific concepts via frequency scores and chi-square tests; (3) **Bias Association Identification**: removing concepts that reflect only factual exclusivity, ensuring the retained associations reflect model-learned biases.

The framework takes as input a carefully designed open-ended story generation experiment covering three major demographic categories—gender, race, and religion—across 10 location categories comprising 87 real-world locations, with over 29,000 two-character stories generated per model per setting.

### Key Designs

**Module 1: Multi-Stage Association Extraction Pipeline**

- **Function**: Extracts core descriptive concepts attributed to characters from each generated story, with fine-grained decomposition and canonicalization.
- **Mechanism**: A four-step pipeline is employed—(a) *Concept Extraction*: an LLM (Qwen3-32B) extracts core character attributes based solely on explicit textual evidence; (b) *Self-Refinement*: post-hoc verification removes hallucinated, redundant, and ambiguous entries; (c) *Fine-Grained Decomposition*: composite concepts are split into minimal meaningful atomic attributes (e.g., "casually chatting and appearing relaxed" → "talkative" + "relaxed"); (d) *Concept Canonicalization*: sentence embeddings are used to compute semantic similarity, clustering and merging semantically equivalent concepts.
- **Design Motivation**: Open-ended generations yield highly diverse concept expressions; direct comparison leads to substantial redundancy and omissions. The multi-stage process ensures extracted results are accurate, fine-grained, and cross-identity comparable.

**Module 2: Salient Association Identification (Dual Filtering Mechanism)**

- **Function**: Filters statistically significant and identity-discriminative associations from the large pool of extracted concepts.
- **Mechanism**: A dual approach is applied—(a) *Frequency Discriminability Score* $\mathcal{S}(Y,A)$: measures the gap between the frequency of concept $Y$ under identity $A$ and the minimum frequency across all other identities, normalized to $[0,1]$; (b) *Chi-Square Test*: tests whether the concept distribution is significantly associated with demographic identity within each location category ($p < 0.05$). Both conditions must be satisfied for retention.
- **Design Motivation**: Frequency alone is susceptible to random fluctuation, while statistical testing alone does not indicate the direction of association. The dual criterion ensures that retained associations are both identity-specific and statistically robust.

**Module 3: Bias Association Identification (Exclusivity Filtering)**

- **Function**: Filters out concepts that reflect factual definitions rather than model-learned bias.
- **Mechanism**: An LLM assesses whether each salient concept naturally and inevitably belongs exclusively to a given identity (e.g., the concept "female" belongs only to the female group by definition), and such "factually exclusive" concepts are removed.
- **Design Motivation**: Without this filter, associations such as "female ↔ female" would confound the analysis, misattributing universally factual exclusivity as bias patterns.

### Loss & Training

This paper does not involve model training. On the experimental design side, three sentiment constraint strategies are employed to guide LLM generation:

- **Baseline Setting**: No sentiment constraint; natural generation (results tend to be positive).
- **Balanced Sentiment Setting**: LLMs are prompted to generate stories encompassing both positive and negative experiences.
- **Negative Setting**: LLMs are prompted to produce narratives centered on difficulty, conflict, and disappointment.

An **Open-Box Setting** is also explored, applying the Patchscope technique to extract hidden representations from intermediate model layers, revealing latent biases not exposed through black-box generation.

## Key Experimental Results

### Main Results

| Demographic Category | Identity | Single-Character Baseline | Dual-Character Baseline | Balanced Sentiment | Negative Setting |
|---|---|---|---|---|---|
| Gender | Female | 169 | 277 | 423 | 524 |
| Gender | Male | 113 | 167 | 251 | 329 |
| Race | Asian | 335 | 684 | 651 | 674 |
| Race | Black | 435 | 590 | 591 | 632 |
| Religion | Buddhist | 777 | 755 | 702 | 735 |
| Religion | Muslim | 968 | 832 | 856 | 856 |

*(Numbers represent the count of bias associations discovered per identity across all locations under each setting.)*

### Ablation Study

Quality evaluation of LLM-assisted steps:
- Concept Extraction: Recall 0.9856, Precision 0.9330
- Fine-Grained Decomposition Accuracy: 0.9711
- Concept Canonicalization (Clustering): Homogeneity 1.0, Completeness 0.89, V-measure 0.94
- Exclusivity Filtering Accuracy: 0.98

### Key Findings

- Female identity is associated with a greater number of bias concepts and is more strongly linked to negative emotions such as "nervous" and "anxious," while male identity is associated with concepts such as "supportive" and "determination."
- The more negative the sentiment constraint, the greater the number of bias associations discovered (the negative setting yields approximately 90% more than the baseline).
- Different LLMs exhibit distinct bias patterns; Qwen3-8B produces the most bias associations across all categories.
- Bias associations discovered in the black-box and open-box settings differ substantially, indicating that some biases are concealed within the model's internal representations.

## Highlights & Insights

- This is the first systematic bias discovery framework targeting open-ended generation, breaking free from the constraint of predefined concept sets.
- The multi-stage pipeline is elegantly designed, with quality verification at each step and strong final evaluation metrics.
- Several unexpected bias associations are uncovered (e.g., "Black ↔ successful entrepreneur," "Asian ↔ communication difficulties") that cannot be detected through conventional evaluation.
- The experimental scale is large (29,000+ stories per model per setting) with broad coverage.

## Limitations & Future Work

- Concept extraction and exclusivity judgments rely on an LLM (Qwen3-32B), potentially introducing biases inherent to that model.
- Only English-language generation is covered; cross-lingual bias is not explored.
- Story generation represents a specific genre; different generation tasks (dialogue, QA) may expose distinct bias patterns.
- The frequency discriminability score uses the minimum value across other identities as baseline, which may lack robustness for categories with imbalanced numbers of identities.

## Related Work & Insights

- **BiasDora** (Raj et al.): A pioneer in open-ended bias discovery, but limited to word-level completion tasks.
- **BBQ** (Parrish et al.): A representative bias benchmark employing template-based multiple-choice questions.
- **Patchscope** (Ghandeharioun et al.): An interpretability technique for model internal representations, applied here for open-box bias probing.
- Insight: Bias evaluation should shift from "confirming known biases" to "discovering unknown biases," with open-ended generation as the key breakthrough modality.

## Rating

⭐⭐⭐⭐ (4/5)

The work is rigorous, systematic, and conducted at large scale. The proposed BADF framework offers methodological novelty and uncovers valuable, previously undiscovered bias associations. One point is deducted because the biases potentially introduced by the LLM-assisted steps themselves are not discussed in depth, and the scalability of the method to additional languages and task types remains unverified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Steering the Herd: A Framework for LLM-Based Control of Social Learning](../../ICLR2026/social_computing/steering_the_herd_a_framework_for_llm-based_control_of_social_learning.md)
- [\[AAAI 2026\] SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation](scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval.md)
- [\[NeurIPS 2025\] Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs](../../NeurIPS2025/social_computing/auto-search_and_refinement_an_automated_framework_for_gender_bias_mitigation_in_.md)
- [\[ACL 2026\] Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews](../../ACL2026/social_computing/justice_in_judgment_unveiling_hidden_bias_in_llm-assisted_peer_reviews.md)
- [\[AAAI 2026\] Argumentative Debates for Transparent Bias Detection](argumentative_debates_for_transparent_bias_detection_technic.md)

</div>

<!-- RELATED:END -->
