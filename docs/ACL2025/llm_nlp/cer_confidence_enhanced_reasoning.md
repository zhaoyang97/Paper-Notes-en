---
title: >-
  [Paper Note] CER: Confidence Enhanced Reasoning in LLMs
description: >-
  [ACL 2025][LLM (Other)][Uncertainty reasoning] Proposes the Confidence Enhanced Reasoning (CER) framework, which quantifies the confidence of key tokens (numerical values in math tasks or proper nouns in open-domain tasks) in each intermediate step of CoT reasoning. It evaluates the reliability of the entire reasoning chain using the product of step-wise confidence and replaces simple majority voting with confidence-weighted aggregation, achieving improvements over self-consi…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Uncertainty reasoning"
  - "confidence aggregation"
  - "multi-step reasoning"
  - "self-consistency improvement"
  - "key tokens"
date: 2026-05-08
content_hash: aa78d3aca24151ed
---

# CER: Confidence Enhanced Reasoning in LLMs

**Conference**: ACL 2025  
**arXiv**: [2502.14634](https://arxiv.org/abs/2502.14634)  
**Code**: [https://github.com/sharif-ml-lab/CER](https://github.com/sharif-ml-lab/CER)  
**Area**: LLM Reasoning  
**Keywords**: Uncertainty reasoning, confidence aggregation, multi-step reasoning, self-consistency improvement, key tokens  

## TL;DR
Proposes the Confidence Enhanced Reasoning (CER) framework, which quantifies the confidence of key tokens (numerical values in math tasks or proper nouns in open-domain tasks) in each intermediate step of CoT reasoning. It evaluates the reliability of the entire reasoning chain using the product of step-wise confidence and replaces simple majority voting with confidence-weighted aggregation, achieving improvements over self-consistency by up to 7.4% on math and 5.8% on open-domain tasks.

## Background & Motivation
**Background**: Self-consistency (SC) is the primary method for improving LLM reasoning accuracy, which selects the final answer via majority voting after generating multiple reasoning chains. However, SC assigns the same weight to all reasoning chains regardless of their reliability.

**Limitations of Prior Work**: (a) SC fails when the majority of chains converge to an incorrect answer, as voting cannot correct systematic errors; (b) the quality of different reasoning chains varies significantly, yet SC treats them equally, giving a high-quality chain the same weight as a low-quality one; (c) there is a lack of lightweight methods to quantify uncertainty during the reasoning process.

**Key Challenge**: Certain steps in a reasoning chain are particularly critical to the final answer (e.g., numerical values in key calculation steps), but existing methods do not distinguish the importance of different steps or tokens.

**Goal**: To identify critical decision points in multi-step reasoning, quantify their confidence, and replace majority voting with confidence-weighted aggregation.

**Key Insight**: Leveraging the LLM's own token probabilities to estimate the confidence of intermediate answers—the probabilities of key tokens (numbers or entity names) reflect the model's level of certainty.

**Core Idea**: Key token probabilities $\rightarrow$ step-wise confidence $\rightarrow$ reasoning chain reliability $\rightarrow$ weighted voting.

## Method

### Overall Architecture
CER consists of three components: (1) **Key Token Identification**—identifying key tokens (numerical values in math, proper nouns in open-domain tasks) in each reasoning step; (2) **Step-to-Chain Confidence Calculation**—computing step-wise confidence using key token probabilities and obtaining the entire chain's confidence via a step-wise aggregation function (e.g., product); (3) **Confidence-Weighted Aggregation**—using chain-level confidence as weights for weighted voting to select the final answer.

### Key Designs

1. **Key Token Identification and Confidence Computation**:

    - Function: To locate "decisive tokens" in intermediate reasoning steps and estimate their confidence.
    - Mechanism:
        - Mathematical Tasks: Identifying numerical results generated in each step (e.g., calculating an intermediate value "= 125") and taking the average probability of those numerical tokens.
        - Open-Domain Tasks: Identifying proper nouns or entities appearing in each step (e.g., "Albert Einstein") and taking the average probability of those entity tokens.
    - Design Motivation: The correctness of these key tokens determines the final accuracy of the reasoning chain. If the confidence of intermediate computational results is low, the entire chain is likely unreliable.

2. **Step-wise Aggregation Function**:

    - Function: To aggregate step-wise confidence into the confidence of the entire chain.
    - Mechanism: $\text{Chain Confidence} = f(c_1, c_2, ..., c_n)$, where $f$ can be a product (the most strict—any single low-confidence step drags down the entire score), minimum, mean, etc.
    - Best Choice: **Product**—because an error in an earlier step of a reasoning chain propagates to all subsequent steps.
    - Design Motivation: The error propagation characteristic of multi-step reasoning—the product naturally models "one wrong step ruins the whole chain."

3. **Path-wise Aggregation**:

    - Function: To replace simple majority voting with chain-level confidence-weighted voting.
    - Mechanism: $\text{Final Answer} = \arg\max_a \sum_{\text{chain}_i \text{ gives } a} w_i$, where $w_i$ is the confidence of chain $i$.
    - vs Self-Consistency: In SC, $w_i = 1$ (equal weight); in CER, $w_i$ reflects the quality of the reasoning chain.
    - Design Motivation: Reasoning chains with high confidence should exert a greater influence on the final answer—a single "certain" correct chain should outweigh two "uncertain" incorrect ones.

### Loss & Training
- **No Training Required**—directly utilizing the token probabilities of the LLM.
- Requires only white-box access (to retrieve token probabilities).
- Applicable to any LLM that supports token probability outputs.

## Key Experimental Results

### Main Results

| Method | GSM8K | MATH | AQuA | TriviaQA | NQ |
|------|-------|------|------|----------|-----|
| CoT (baseline) | Baseline | Baseline | Baseline | Baseline | Baseline |
| Self-Consistency | +3-5% | +2-4% | +2-3% | +1-3% | +1-2% |
| **CER** | **+7-12%** | **+5-8%** | **+4-7%** | **+3-7%** | **+2-6%** |
| Extra Gain of CER vs SC | **+2-7.4%** | **+2-5%** | **+1-4%** | **+2-5.8%** | **+1-4%** |

### Ablation Study

| Aggregation Function | Performance | Explanation |
|---------|------|------|
| Product (Recommended) | Best | Correctly models error propagation |
| Minimum | Second Best | Too conservative |
| Mean | Average | Not sensitive enough |
| Equal Weight (=SC) | Baseline | Does not differentiate chain quality |

### Key Findings
- **Confidence weighting consistently outperforms equal-weight voting** across all five datasets and four LLMs, validating the importance of differentiating reasoning chain quality.
- **The improvement is more pronounced in math tasks** (up to 7.4%), as the confidence of numerical tokens is a strong signal of reasoning correctness.
- Utilizing the product as the step-wise aggregation function is the most effective, aligning with the error propagation model of multi-step reasoning.
- Significant improvements are also observed even on small models (7B), indicating that the method is not dependent on model scale.
- Extremely low computational overhead—only requires reading token probabilities and performing basic operations on top of SC.

## Highlights & Insights
- The intuition that **"not all reasoning chains are equal"** is elegantly operationalized: quantifying confidence with token probabilities and replacing simple voting with weighted voting.
- **Key token identification** is the core of the method: not all token probabilities are informative; only the probabilities of "decisive tokens" (numerical values/entities) reflect reasoning quality.
- **Using product aggregation to model error propagation** is a reasonable inductive bias—multi-step reasoning indeed suffers from the "one wrong step ruins all" effect.
- The method is highly lightweight: no training or additional model calls are needed; it simply requires reading existing probabilities.
- It can be orthogonally combined with other reasoning enhancement methods (e.g., ToT, MCTS), providing better evaluation signals for their candidate selection.

## Limitations & Future Work
- Requires white-box access (token probabilities), which black-box APIs (e.g., GPT-4) do not provide.
- The identification of key tokens relies on simple rules (numerical values/entities); more complex reasoning may involve other types of critical tokens.
- The method is ineffective when all chains exhibit high confidence but produce incorrect answers (systemic overconfidence).
- Validated only on multiple-choice and short-answer tasks; the effectiveness in open-ended generation scenarios remains unknown.

## Related Work & Insights
- **vs Self-Consistency**: SC uses equal-weight voting, whereas CER uses confidence-weighted voting—a simple yet effective improvement.
- **vs Semantic Entropy**: SE estimates uncertainty using semantic clustering but does not differentiate steps; CER estimates at the step-level key token—offering finer granularity.
- **vs Disentangling Memory & Reasoning**: That paper separates "knowing" and "reasoning"; CER quantifies "certainty" and "uncertainty"—complementary perspectives.
- **vs Calibration Confidence (ACL2025 generation)**: That paper focuses on confidence calibration for generation tasks; CER focuses on reasoning tasks—bringing a similar concept to different applications.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of key token confidence and weighted voting is simple and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation studies with 5 datasets × 4 models and various aggregation functions.
- Writing Quality: ⭐⭐⭐⭐ The method is clearly described, and the example in Figure 1 is intuitive.
- Value: ⭐⭐⭐⭐ Lightweight reasoning enhancement that can be applied immediately.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models](direct_confidence_alignment_aligning_verbalized_confidence_with_internal_confide.md)
- [\[ACL 2025\] Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning](boosting_llms_molecular_structure_elucidation_with_knowledge_enhanced_tree_searc.md)
- [\[ACL 2025\] SQLong: Enhanced NL2SQL for Longer Contexts with LLMs](sqlong_enhanced_nl2sql_for_longer_contexts_with_llms.md)
- [\[ACL 2025\] Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs](do_language_models_mirror_human_confidence_exploring_psychological_insights_to_a.md)
- [\[ACL 2025\] Are Your LLMs Capable of Stable Reasoning?](are_your_llms_capable_of_stable_reasoning.md)

</div>

<!-- RELATED:END -->
