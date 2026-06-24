---
title: >-
  [Paper Note] Beyond Output Matching: Bidirectional Alignment for Enhanced In-Context Learning
description: >-
  [ACL 2025][LLM (Other)][In-Context Learning] Proposes Bidirectional Alignment (BiAlign). Building upon traditional knowledge distillation that aligns only output distributions, it introduces input preference alignment. By utilizing ranking loss, the student model learns the teacher model's preference ranking over different ICL exemplars. BiAlign consistently outperforms baselines across five language understanding, reasoning, and coding tasks…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "In-Context Learning"
  - "Knowledge Distillation"
  - "Bidirectional Alignment"
  - "Input Preference"
  - "Ranking Loss"
  - "Exemplar Selection"
date: 2026-05-08
content_hash: ff76e2a9af3ee169
---

# Beyond Output Matching: Bidirectional Alignment for Enhanced In-Context Learning

**Conference**: ACL 2025  
**arXiv**: [2312.17055](https://arxiv.org/abs/2312.17055)  
**Code**: Not publicly available  
**Area**: In-Context Learning / Knowledge Distillation  
**Keywords**: In-Context Learning, Knowledge Distillation, Bidirectional Alignment, Input Preference, Ranking Loss, Exemplar Selection

## TL;DR

Proposes Bidirectional Alignment (BiAlign). Building upon traditional knowledge distillation that aligns only output distributions, it introduces input preference alignment. By utilizing ranking loss, the student model learns the teacher model's preference ranking over different ICL exemplars. BiAlign consistently outperforms baselines across five language understanding, reasoning, and coding tasks, yielding a 20% improvement on GSM8K and an 18% improvement on LogiQA.

## Background & Motivation

**Background**: LLMs exhibit excellent performance in few-shot scenarios through ICL, but deploying large models (175B) is costly (requiring at least 350GB of GPU memory). Thus, knowledge distillation to transfer capabilities from large models (teachers) to small models (students) has become an important research direction.

**Limitations of Prior Work**: Existing distillation methods only focus on **output-side** alignment—either training the student on teacher-generated outputs or matching the teacher's token-level probability distribution. However, ICL performance is highly sensitive to the **input-side** (exemplar selection), where different combinations of exemplars can result in dramatic performance variations, ranging from near-random to surpassing fine-tuned SOTA.

**Key Challenge**: Existing distillation methods only teach the student "what to output" but do not teach them "what kind of input exemplars to prefer". Consequently, the student cannot benefit from exemplars of varying quality in the way the teacher does.

**Goal**: To enhance the ICL capabilities of the student model by aligning input preferences.

**Key Insight**: Analogous to how reward models in RLHF learn "which outputs to prefer", BiAlign enables students to learn "which input exemplars to prefer".

**Core Idea**: Bidirectional Alignment = Output Distribution Alignment (KL divergence) + Input Preference Alignment (ranking loss), enabling the student to learn both "what to output" and "which inputs are superior".

## Method

### Overall Architecture

Divided into two stages: (1) Upstream ICL alignment—aligning the student and the teacher on a source task set $\mathcal{T}^{\text{src}}$; (2) Downstream ICL evaluation—assessing the ICL capabilities of the aligned student on a target task set $\mathcal{T}^{\text{tgt}}$ that has no overlap with the source tasks.

### Key Designs

1. **Token-level Output Distribution Alignment**: Computes the KL divergence between the student and the teacher over the entire ICL sequence (including exemplars and test items), rather than only on the target answer portion. This ensures there are sufficient tokens in a batch to maintain in-weights capabilities.

$$\mathcal{L}^{\text{KL}} = \sum_{i=1}^{m} \sum_{j=1}^{t} D_{\text{KL}}(P_j(\mathcal{V}|\hat{X}_i, \theta_T) \| P_j(\mathcal{V}|\hat{X}_i, \theta_S))$$

2. **Input Preference Metric**: The preference score of a model for a set of exemplars $R_{ij}$ is defined as the probability of generating the correct answer $\hat{y}_i$ given the exemplar set and the test input $\hat{x}_i$. That is, $Q^T(R_{ij}) = P(\hat{y}_i | R_{ij}, \hat{x}_i, \theta_T)$. The intuition is that the model prefers exemplar sets that better facilitate correct answer generation.

3. **Exemplar Subset Sampling**: Divides all $k$-shot exemplars into two groups, $G_{\text{sim}}$ and $G_{\text{dissim}}$, based on their semantic similarity to the test sample. Then, $N$ subsets containing various numbers of similar exemplars are sampled from the power set ($N \ll 2^k$, with $N=4$ in experiments).

4. **Ranking Loss**:

$$\mathcal{L}^{\text{rank}} = \sum_i \sum_{R^+, R^- \in R_i^{\text{all}}} \max\{0, \underbrace{\frac{\log Q^S(R^-) - \log Q^S(R^+)}{\max \log Q^S - \min \log Q^S}}_{\text{Left: 学生偏好差异}} + \underbrace{\frac{1}{N-1}(\text{rank}(Q^T(R^-)) - \text{rank}(Q^T(R^+)))}_{\text{Right: 教师排名差异}}\}$$

    - The Left part measures the student's normalized preference difference between positive and negative exemplar sets.
    - The Right part reflects the teacher's **relative ranking** difference between positive and negative exemplar sets (using the rank function instead of raw scores to reduce the impact of magnitude variations in scores).
    - Positive/negative polarity is determined by the teacher's preference score: those with higher preference scores are considered positive.

5. **Total Loss**: $\mathcal{L} = \mathcal{L}^{\text{KL}} + \lambda \mathcal{L}^{\text{rank}}$

### Training Data

Uses CrossFit (a large multi-task few-shot dataset) to construct 12K ICL training instances, with each instance containing a random number of exemplars (ranging from 4 to 10) to enhance the model's generalization across various prompt lengths.

## Key Experimental Results

### Main Results (Table 1, Student: Llama 2-7B)

| Method | MMLU | BBH | GSM8K | LogiQA | HumanEval | Avg |
|------|------|-----|-------|--------|-----------|-----|
| Vanilla | 45.4 | 39.5 | 15.2 | 30.3 | 14.6 | 29.0 |
| FT (meta-training) | 46.4 | 39.8 | 15.6 | 31.7 | 14.2 | 29.5 |
| Output-Align (13B teacher) | 46.3 | 39.3 | 15.4 | 32.2 | 14.0 | 29.4 |
| **BiAlign (13B teacher)** | **47.5** | **41.0** | **16.8** | **33.9** | **15.6** | **31.0** |
| Output-Align (70B teacher) | 47.1 | 39.8 | 16.4 | 33.2 | 14.6 | 30.2 |
| **BiAlign (70B teacher)** | **49.5** | **43.2** | **18.3** | **35.7** | **16.6** | **32.7** |

- BiAlign consistently outperforms all baselines across all tasks.
- 13B Teacher: +2.0% average relative gain; 70B Teacher: +3.7% average relative gain.
- Tasks requiring more reasoning benefit more: GSM8K +20.4%, LogiQA +17.8% (with the 70B teacher).

### Mathematical Reasoning Difficulty Gradient (Table 2)

| Difficulty | ASDiv (Easy) | SVAMP | GSM8K | AQUA-RAT (Hard) |
|------|-----------|-------|-------|---------------|
| Relative Gain | 6.0% | 5.6% | 10.5% | 11.5% |

- The harder the task, the larger the improvement yielding from BiAlign—input preference alignment provides fine-grained supervision.

### Additional Verification

- **Larger Student Model (13B)**: BiAlign still outperforms Output-Align (40.9 vs 38.8).
- **Other Backbone Models**: Llama 3-8B (63.9 vs 61.7), Phi-3-mini (69.1 vs 67.4).
- **Computational Overhead**: Training FLOPs are approximately 2.3x that of Output-Align, but BiAlign remains superior under identical FLOP budgets.
- **Preference Consistency**: The agreement rate on top/bottom-preferred subsets between the BiAlign student and its teacher is significantly higher than that of Output-Align.
- **Inference Stage**: No additional overhead.

## Highlights & Insights

1. **Pioneering Input Preference Alignment**: Uncovers a neglected dimension in knowledge distillation—students must learn not only "what to output" but also "which inputs are superior," forming an intriguing duality with preference learning in RLHF.
2. **Design of the Ranking Loss**: Employs a rank function instead of raw scores for alignment, mitigating the impact of scale fluctuations in scores, which is validated empirically.
3. **Highly Effective for Reasoning Tasks**: Input preference alignment offers fine-grained supervision, demonstrating the most significant gains in tasks that require reasoning.
4. **Diversity in ICL Prompt Lengths**: Randomly uses 4-10 exemplars during training, enhancing generalization over varying numbers of exemplars, including zero-shot scenarios (which likely explains the improvements on HumanEval).
5. **Complementary to ICP**: BiAlign seamlessly integrates with In-Context Pretraining, further boosting ICL performance.

## Outperforming Limitations & Future Work

1. The ranking loss introduces approximately 2.3x additional training GPU computational overhead.
2. The subset sampling strategy is relatively simple (random grouping based on similarity), and superior sampling strategies may exist.
3. Only explores scenarios with fixed student architectures, without considering neural architecture search.
4. The effectiveness when distilling from ultra-large-scale teachers (e.g., 400B+) remains unverified.
5. The impact of the choice of the source task set on final results is not analyzed in depth.

## Related Work & Insights

- **ICL**: Works like MetaICL enhance ICL capabilities through supervised/self-supervised training.
- **Knowledge Distillation**: Approaches like GKD and DistilBERT focus on output distribution alignment.
- **RLHF/Preference Learning**: DPO, RRHF, etc., learn output preferences; BiAlign extends the concept of preference to the input side.
- **Exemplar Selection**: Numerous works study the impact of exemplar selection in ICL; BiAlign explicitly incorporates this factor into the distillation framework.

## Rating

⭐⭐⭐⭐ — Offers an innovative perspective (input preference is an overlooked dimension) with a well-designed ranking loss and comprehensive experimental coverage (across multiple tasks, model sizes, and backbones). The main limitations reside in the extra computational overhead and the somewhat incremental nature of the methodological novelty (adding a ranking loss to KL distillation). Overall, a solid work on ICL distillation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Beyond In-Context Learning: Aligning Long-form Generation of LLMs via Task-Inherent Attribute Guidelines](beyond_in-context_learning_aligning_long-form_generation_of_large_language_model.md)
- [\[ACL 2025\] Cross-Modal Alignment for LLM-Enhanced Spoken Language Understanding](cross-modal_alignment_for_llm-enhanced_spoken_language_understanding.md)
- [\[ACL 2025\] Exploring Explanations Improves the Robustness of In-Context Learning](exploring_explanations_improves_the_robustness_of_in-context_learning.md)
- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](leveraging_in-context_learning_for_political_bias_testing_of_llms.md)
- [\[ICML 2025\] Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence](../../ICML2025/llm_nlp/beyond_induction_heads_in-context_meta_learning_induces_multi-phase_circuit_emer.md)

</div>

<!-- RELATED:END -->
