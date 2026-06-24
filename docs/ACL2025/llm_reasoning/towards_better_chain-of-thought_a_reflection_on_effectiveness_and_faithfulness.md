---
title: >-
  [Paper Note] Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness
description: >-
  [ACL 2025][Reasoning][Chain-of-Thought] This paper systematically analyzes the representation patterns of CoT from the dual perspectives of effectiveness and faithfulness. It finds that problem difficulty, information gain, and the monotonicity of information flow govern the effectiveness of CoT. It also reveals the mechanism of unfaithful CoT: the model recalls correct information from the question that was omitted by the CoT when predicting answers. Based on these insights…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Chain-of-Thought"
  - "Reasoning Effectiveness"
  - "Reasoning Faithfulness"
  - "Information Gain"
  - "Information Flow"
date: 2026-05-08
content_hash: 0ce1463a0c99ae2f
---

# Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness

**Conference**: ACL 2025  
**arXiv**: [2405.18915](https://arxiv.org/abs/2405.18915)  
**Code**: [Available](https://github.com/BugMakerzzz/better_cot)  
**Area**: LLM Reasoning  
**Keywords**: Chain-of-Thought, Reasoning Effectiveness, Reasoning Faithfulness, Information Gain, Information Flow

## TL;DR

This paper systematically analyzes the representation patterns of CoT from the dual perspectives of effectiveness and faithfulness. It finds that problem difficulty, information gain, and the monotonicity of information flow govern the effectiveness of CoT. It also reveals the mechanism of unfaithful CoT: the model recalls correct information from the question that was omitted by the CoT when predicting answers. Based on these insights, the QUIRE algorithm is proposed, which simultaneously improves both the effectiveness (+2.4%) and faithfulness (+5.6%) of CoT.

## Background & Motivation

CoT has demonstrated notable success in mathematical reasoning, but its performance on other tasks remains inconsistent or even detrimental. Existing evaluation studies present two major limitations:

**Shallow evaluation of effectiveness**: They merely report "on which tasks CoT is effective" (e.g., "tasks containing mathematical symbols") without deeply investigating the underlying factors.

**Lack of explanation in faithfulness evaluation**: They only evaluate whether a CoT is faithful, failing to explain why unfaithfulness occurs.

**Core Problem**: Why is CoT effective on certain tasks while ineffective on others? Why can a faulty CoT sometimes still lead to the correct answer? While these two questions appear independent, this paper reveals their intrinsic connection—enhancing faithfulness can directly improve effectiveness.

## Method

### Overall Architecture

The framework consists of three parts: (1) analyzing three key factors of CoT effectiveness; (2) explaining the mechanism behind CoT unfaithfulness; (3) designing the QUIRE algorithm.

### CoT Effectiveness Analysis

1. **Problem Difficulty**:
    - Problem difficulty is categorized into five levels based on the model's pass@1 rate without CoT.
    - **Conclusion Cl.1**: CoT yields significant improvements on difficult problems, but is almost ineffective or even harmful on simple tasks.
    - Mathematical reasoning has a high proportion of difficult problems (making CoT highly effective overall), whereas commonsense reasoning is dominated by simple problems (making CoT overall ineffective).
    - This aligns with intuition: simple problems do not require "thinking," and forcing CoT may introduce unnecessary distraction.

2. **Information Gain**:
    - Defined as $IG(C,Q) = H(C) - H(C|Q)$.
    - Larger $IG$ $\rightarrow$ CoT extracts more information from the question $\rightarrow$ CoT itself contributes less additional information.
    - **Conclusion Cl.2**: CoT is effective if and only if it provides additional information not present in the question itself.
    - Mathematical reasoning has the lowest $IG$ (the CoT contains many derivation steps = new information), while commonsense reasoning has the highest $IG$ (the CoT merely repeats known commonsense).

3. **Information Flow**:
    - Integrated Gradients Attribution (IGA) is utilized to track the attribution strength of each CoT step to the final answer.
    - Average Attribution Effect (AAE) and Monotonicity of Information Flow (MIF, measured by Spearman correlation coefficient) are defined.
    - **Conclusion Cl.3**: CoT is most effective when the information flow from CoT to the answer increases monotonically along the reasoning path.
    - Intuition: Good reasoning is "progressive," where each step accumulates contributions toward the final answer.

### CoT Unfaithfulness Analysis

The unfaithfulness phenomenon is explained by analyzing the information interactions among Question, CoT, and Answer jointly.

1. **Unfaithfulness Identification**:
    - The correctness of 50 CoT-Answer pairs is manually annotated.
    - Logical reasoning exhibits the most severe issues: in ProntoQA, 17 out of 50 samples (34%) demonstrate the pattern of incorrect CoT $\rightarrow$ correct answer.
    - Mathematical reasoning is virtually free of this issue.

2. **Question $\rightarrow$ CoT (Cl.4)**: Unfaithful CoT has lower $IG$ than faithful CoT $\rightarrow$ it omits correct information from the question.

3. **CoT $\rightarrow$ Answer (Cl.5)**: Unfaithful CoT has lower information transfer (AAE) to the answer than faithful CoT $\rightarrow$ the model relies less on the unfaithful CoT.

4. **Question $\rightarrow$ Answer (Cl.6, Key Finding)**:
    - When predicting the answer, the model directly "recalls" the correct information omitted by the CoT from the question.
    - Experiment: When sentences in the question are ranked by AAE, more omitted sentences achieve top-k AAE under the unfaithful setup $\rightarrow$ the model bypasses the CoT to extract information directly from the question.

### QUIRE Algorithm

1. **AAE Recall**:
    - First, generate a raw answer (using Self-Consistency) and calculate the AAE of each sentence in the question to the answer.
    - Select the top-k most relevant sentences as extra prompts to guide CoT generation.
    - Motivation: Proactively and explicitly feed the information that the model is likely to "secretly recall" directly into the CoT.

2. **IG Voting**:
    - Score each enhanced CoT using $IG(Q,C)$.
    - Higher $IG$ $\rightarrow$ CoT extracts more information from the question $\rightarrow$ fewer hallucinations.
    - Use the $IG$ score as the weighted voting weight for Self-Consistency.

## Key Experimental Results

### Main Results (Llama3.1-8B)

| Method | PW Acc | PW BertScore | PW FBS | PQA Acc | PQA BertScore | PQA FBS |
|------|:------:|:------:|:------:|:------:|:------:|:------:|
| CoT | 59.2 | 64.9 | 55.7 | 86.8 | 86.1 | 78.0 |
| Self-Consistency | 60.6 | 65.0 | 57.8 | 93.2 | 87.5 | 83.6 |
| Least-to-Most | 54.0 | 60.4 | 56.4 | 90.0 | 77.3 | 72.6 |
| Self-Refine | 51.6 | 65.9 | 53.4 | 88.5 | 91.5 | 84.5 |
| **QUIRE** | **63.0** | **66.6** | **58.0** | **95.0** | **92.7** | **89.2** |
| - AAE Recall | 60.2 | 65.1 | 57.0 | 95.0 | 87.5 | 84.6 |
| - IG Vote | 62.8 | 64.1 | 56.6 | 94.3 | 87.0 | 83.4 |

### Gemma2-9B Verification

| Method | PW Acc | PW FBS | PQA Acc | PQA FBS |
|------|:------:|:------:|:------:|:------:|
| CoT | 65.0 | 52.9 | 77.0 | 57.7 |
| SC | 31.0 | 50.3 | 81.0 | 60.5 |
| **QUIRE** | **65.0** | **56.3** | **92.5** | **69.5** |

QUIRE is similarly effective on Gemma2-9B, where ProntoQA FBS improves from 60.5 to 69.5.

### Key Findings

1. Three factors governing CoT effectiveness: more effective for difficult problems, more effective with more additional information, and more effective with monotonically increasing information flow.
2. In logical reasoning, 34% (ProntoQA) of samples suffer from unfaithfulness issues.
3. Root cause of unfaithfulness: when predicting answers, the model bypasses the CoT to directly "recall" the omitted information from the question.
4. Faithfulness and effectiveness are positively correlated: FBS +5.6% is accompanied by Acc +2.4%.
5. AAE Recall contributes significantly to Acc, while IG Vote contributes heavily to FBS, reinforcing their complementary roles.

## Highlights & Insights

1. **Unified Analytical Framework**: Connects effectiveness and faithfulness for the first time, identifying faithfulness as a key factor of effectiveness.
2. **Deep Application of Information Theory**: Employs two tools, Information Gain and Integrated Gradients Attribution, to "deconstruct mechanisms" rather than merely "observing phenomena."
3. **Explanation of the Unfaithfulness Mechanism**: Identifies that the model "bypasses" CoT to extract information directly from the question, providing key insights to understand the black box of LLM reasoning.
4. **Theoretical Duality of QUIRE**: AAE Recall $\leftarrow$ Cl.6 (proactively recalling missing information), IG Vote $\leftarrow$ Cl.2/Cl.4 (evaluating CoT quality via information gain).

## Limitations & Future Work

1. **Restricted to White-box Models**: IGA requires gradient information, making it inapplicable to closed-source models like GPT-4.
2. **Lack of Theoretical Proof**: The link from faithfulness to effectiveness is only validated empirically.
3. **Computational Overhead of IGA**: Multiple forward and backward passes lead to higher practical deployment costs.
4. **Task Coverage**: QUIRE is primarily verified on logical reasoning tasks; its performance on code and math tasks remains to be validated.

## Related Work & Insights

- Sprague et al. (2024) find that CoT is mainly useful in math/symbolic reasoning; this paper further explains "why."
- Bao et al. (2024) evaluate faithfulness through causal analysis; this paper goes a step further by explaining the mechanism of unfaithfulness.
- **Key Insight**: The effectiveness of CoT depends on its ability to provide valuable new information for a specific problem.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The unified analysis of effectiveness-faithfulness and the information-theoretic explanation of the unfaithfulness mechanism are highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 9 datasets, 4 models, and extensive supplementary experiments in the appendix.
- **Writing Quality**: ⭐⭐⭐⭐ — The logical progression from analysis to methodology is clear and coherent.
- **Value**: ⭐⭐⭐⭐⭐ — Contributes significantly to the understanding of the nature of CoT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning](unveiling_the_key_factors_for_distilling_chain-of-thought_reasoning.md)
- [\[ACL 2025\] Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions](improving_chain-of-thought_reasoning_via_quasi-symbolic_abstractions.md)
- [\[ACL 2025\] DRT: Deep Reasoning Translation via Long Chain-of-Thought](drt_deep_reasoning_translation_via_long_chain-of-thought.md)
- [\[ACL 2025\] Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning](enhancing_chain-of-thought_reasoning_with_critical_representation_fine-tuning.md)
- [\[ACL 2025\] TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning](tract_regression_cot.md)

</div>

<!-- RELATED:END -->
