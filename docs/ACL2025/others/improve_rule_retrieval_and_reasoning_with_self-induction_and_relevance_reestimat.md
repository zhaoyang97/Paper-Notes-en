---
title: >-
  [Paper Note] Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate
description: >-
  [ACL 2025][Rule Retrieval] To address the semantic gap between queries (concrete instantiated facts) and rules (abstract variable formulations) in rule retrieval, this paper proposes SIAR (Self-Induction Augmented Retrieval) and $R^3$ (Rule Relevance ReEstimate). By mapping queries into the rule semantic space and re-evaluating rule relevance, these two methods significantly improve both rule retrieval and downstream reasoning performance.
tags:
  - "ACL 2025"
  - "Rule Retrieval"
  - "Semantic Alignment"
  - "Self-Induction"
  - "Reranking"
  - "LLM Reasoning"
date: 2026-05-08
content_hash: 22def8b3e47ff67f
---

# Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate

**Conference**: ACL 2025  
**arXiv**: [2505.10870](https://arxiv.org/abs/2505.10870)  
**Code**: None  
**Area**: Others  
**Keywords**: Rule Retrieval, Semantic Alignment, Self-Induction, Reranking, LLM Reasoning

## TL;DR

To address the semantic gap between queries (concrete instantiated facts) and rules (abstract variable formulations) in rule retrieval, this paper proposes SIAR (Self-Induction Augmented Retrieval) and $R^3$ (Rule Relevance ReEstimate). By mapping queries into the rule semantic space and re-evaluating rule relevance, these two methods significantly improve both rule retrieval and downstream reasoning performance.

## Background & Motivation

Rule-based reasoning is an important way to enhance the capabilities of LLMs: rules are first induced from experiences, and then relevant rules are retrieved to assist in reasoning. However, existing research primarily focuses on the generation and application of rules, severely neglecting the critical middle step: **rule retrieval**.

Rule retrieval differs fundamentally from traditional knowledge retrieval:
- **Traditional Retrieval**: Queries and target passages typically share keywords or semantic similarities (e.g., "Who list of US presidents" $\to$ passages containing "US president").
- **Rule Retrieval**: Queries are concrete instantiated facts (e.g., "California environmental law mandates recycling"), whereas rules are in abstract variable forms (e.g., "If regulation $Y$ applies to region $Z$, then person $X$ in region $Z$ must comply with regulation $Y$"). A massive semantic gap exists between the two.

The authors demonstrate the severity of this issue through experiments: retrieving rules using standard retrieval methods to assist in reasoning actually degrades performance (compared to not using rules). However, if the ground-truth rules (golden rules) are directly provided, a 7B model improves by $31.54\%$, and a 72B model improves by $23.67\%$. This indicates that **the issue is not that rules are useless, but that retrieval is inaccurate**.

## Method

### Overall Architecture

The method is integrated into the standard retrieve-then-reason pipeline:
1. **Pre-retrieval**: SIAR leverages the self-induction of LLMs to generate hypothetical rules for query expansion.
2. **Post-retrieval / Pre-reasoning**: $R^3$ utilizes the LLM to re-evaluate the relevance of retrieved rules and rerank them.
3. **Reasoning**: The reranked top-$k$ rules are used to assist downstream LLM reasoning.

### Key Designs

#### 1. SIAR: Self-Induction Augmented Retrieval

**Function**: Before retrieval, a hypothetical rule (self-induced rule, SI) is generated from the query using the self-induction capability of LLMs, which is then used to augment the retrieval query.

**Mechanism**: If the query set and the rule set are viewed as two almost non-overlapping semantic subspaces—the former consisting of instantiated concrete facts, and the latter composed of abstract conceptual knowledge—the role of self-induction is to project the query as much as possible into the rule subspace, allowing the query to better match rules with similar underlying logic during retrieval.

**Implementation**: Few-shot in-context learning is used to guide the LLM to perform induction and abstraction on the query by:
- Summarizing facts in the query
- Abstracting concrete entities into variables
- Hypothesizing potential reasoning relationships

**Two Usage Options**:
- **SIAR (w/ SI)**: Using only the SI as the new query
- **SIAR (w/ SI + input)**: Concatenating the SI and the original query as the new query

**Design Motivation**: Traditional retrieval (whether sparse or dense) directly matches original queries with abstract rules, yielding poor results due to mismatched semantic spaces. Self-induction acts as a "space transformation" that maps queries from the factual space to the rule space.

#### 2. R3: Rule Relevance ReEstimate

**Function**: For the top-$n$ rule list retrieved by SIAR, the LLM is used to re-evaluate the relevance of each rule to the original query, followed by reranking and selecting the top-$k$ rules.

**Mechanism**: Retrievers can only evaluate semantic similarity and cannot judge whether a rule actually helps in reasoning. $R^3$ compensates for this by having the LLM evaluate two dimensions: (1) whether the abstract knowledge in the rule can be instantiated into the facts in the query, and (2) whether the rule helps in reasoning.

**Implementation**: Inspired by RankGPT, the LLM is directly prompted to output the reranked list of rules (rather than performing pairwise comparisons), reducing the number of API calls and accelerating processing.

**Design Motivation**: Although SIAR can improve retrieval ranking, LLMs have limited induction capabilities and may still produce inaccurate SIs for complex queries. $R^3$ provides a second layer of defense, evaluating the quality of rules from an application perspective (rather than just a semantic perspective).

### Methodology Characteristics

The entire method is purely prompting-based and **requires no training**. It is highly generalizable and can be paired with different retrievers (sparse, dense, or LLM-based retrievers) and LLMs of various scales.

## Key Experimental Results

### Main Results: Retrieval Performance (Natural Language Rule Base)

| Method | CLUTRR R@1 | ULogic R@1 | CAIL2018 R@1 |
|------|-----------|-----------|-------------|
| Vanilla (sparse) | 6.67 | 68.91 | 25.30 |
| +SIAR (72B, SI+input) | 11.06 | 74.82 | 74.70 |
| +SIAR-R3 (72B, SI+input) | **14.31** | **92.17** | **86.14** |
| Vanilla (dense) | 2.10 | 30.36 | 9.04 |
| +SIAR (72B, SI) | 11.74 | 64.82 | 76.51 |
| +SIAR-R3 (72B, SI) | **14.03** | **88.19** | **81.32** |

### Main Results: Reasoning Performance

| Method | CLUTRR | ULogic | CAIL2018 |
|------|--------|--------|----------|
| Direct (No Rules) | 38.36 | 93.01 | 80.12 |
| Golden Rule | 89.03 | 94.58 | 98.90 |
| Vanilla sparse retrieval | 37.60 | 93.13 | 73.49 |
| SIAR (sparse) | 49.14 | 94.21 | 86.14 |
| SIAR-R3 (sparse) | **51.71** | **95.90** | **86.75** |
| Vanilla dense retrieval | 30.53 | 90.00 | 72.89 |
| SIAR (dense) | 49.81 | 95.06 | 86.75 |
| SIAR-R3 (dense) | **51.05** | **95.78** | **84.94** |

(The above are all results of Qwen2.5-72B-Instruct)

### Ablation Study

| Factor | Finding |
|------|------|
| Open-source vs. Closed-source LLMs | Qwen2.5-72B performs comparably to GPT-4o, and even better in some scenarios |
| 72B vs. 7B | The induction and reranking capabilities of larger models are significantly stronger |
| Sparse vs. Dense Retrieval | Sparse retrieval is superior in most scenarios (keyword matching is more effective for rules) |
| SI vs. SI+input | Sparse retrieval suits SI+input, while dense retrieval is better suited for SI only |
| Different Retrievers | Significant improvements are observed across BM25, BGE, and BGE-Gemma2 |
| Doubled Rule Base | The method remains effective even when the number of rules is doubled |

### Key Findings

1. **Rules are highly valuable for reasoning, but retrieval is the bottleneck**: Golden rules bring a $23\text{--}31\%$ improvement, whereas vanilla retrieval actually degrades performance.
2. **SIAR gains more in dense retrieval**: Dense retrieval R@1 increases from $2.10$ to $11.74$ ($+9.64$), indicating that dense retrieval is more severely affected by the semantic gap.
3. **$R^3$ is more effective on simpler datasets**: $R^3$ brings massive improvements on ULogic and CAIL2018 (up to $+43.25$ R@1), while only larger models benefit on CLUTRR.
4. **Sparse retrieval generally outperforms dense retrieval**: Many concepts in rule scenarios are poorly represented in dense vector spaces, making keyword matching more precise.

## Highlights & Insights

- **Clear and Important Problem Definition**: The paper systematically points out the fundamental difference between rule retrieval and traditional knowledge retrieval (the concrete vs. abstract semantic gap), addressing an overlooked yet critical problem.
- **Elegant Theoretical Intuition**: The concept of "mapping queries from the factual subspace to the rule subspace" is simple yet powerful, similar to the philosophy of HyDE but applied to rule scenarios.
- **Zero-shot/Training-free and Highly Versatile**: Being a purely prompting-based method, it can be seamlessly plugged into any retrieve-then-reason workflow.
- **In-depth Analysis**: Cross-analyses of different retriever types, rule formats, and model scales provide rich practical guidance.

## Limitations & Future Work

1. **Limited Scale of Rule Base**: The maximum size is only 1,048 rules, which is far from real-world scenarios (e.g., tens of thousands of laws and regulations).
2. **Reliance on LLM Induction Ability**: The self-induction quality of smaller models (7B) is noticeably insufficient, and even $R^3$ cannot effectively compensate for this.
3. **Computational Cost**: SIAR requires an extra LLM inference step, and $R^3$ requires another—which can be uneconomical for large-scale query scenarios.
4. **Unexplored Rule Combinations**: Real-world reasoning might require chained combinations of multiple rules, while the current method only retrieves top-$k$ independent rules.
5. **Potential for Training Lightweight Retrieval Models**: SIs generated by SIAR could be used as training signals to fine-tune dense retrievers, internalizing the space mapping capability.

## Related Work & Insights

- **HyDE (Gao et al.)**: Uses an LLM to generate hypothetical documents before retrieval; SIAR can be viewed as its variant in the rule domain.
- **RankGPT**: The reranking concept of $R^3$ is inspired by it, but the evaluation criteria are extended from semantic relevance to reasoning utility.
- **ExpNote/Hypothesis Search**: Related works on rule generation and application, whereas this paper focuses on the overlooked retrieval stage.
- **Insight**: When the retrieval targets and queries belong to different semantic spaces, the "generate-then-retrieve" paradigm holds universal value.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first to systematically study the rule retrieval problem. The spatial mapping concept of SIAR possesses theoretical elegance.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive cross-experiments covering three datasets, multiple retrievers, and multiple model scales.
- **Writing Quality**: ⭐⭐⭐⭐ — Figure 1 clearly illustrates the problem, and the method description is concise.
- **Value**: ⭐⭐⭐⭐ — Identifies an important problem and provides effective solutions, though the scale of the rule base limits its impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Learning to Reason Over Time: Timeline Self-Reflection for Temporal Reasoning](tiser_timeline_self_reflection_temporal.md)
- [\[ACL 2025\] Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home](adaptive_retrieval_without_self-knowledge_bringing_uncertainty_back_home.md)
- [\[CVPR 2026\] NeuroRule: Bridging Vision and Logic with Differentiable Rule Induction](../../CVPR2026/others/neurorule_bridging_vision_and_logic_with_differentiable_rule_induction.md)
- [\[ACL 2025\] Hard Negative Mining for Domain-Specific Retrieval in Enterprise Systems](hard_negative_mining_for_domain-specific_retrieval_in_enterprise_systems.md)
- [\[ACL 2025\] ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval](acord_an_expert-annotated_retrieval_dataset_for_legal_contract_drafting.md)

</div>

<!-- RELATED:END -->
