---
title: >-
  [Paper Note] Does Self-Consistency Improve the Recall of Encyclopedic Knowledge?
description: >-
  [ACL 2026][LLM Reasoning][Self-Consistency] Applying the "==" heuristic from Sprague et al. at the subject level, the authors split the 57 subjects of MMLU into two subsets: **symbolic reasoning** and **knowledge recall*…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Self-Consistency"
  - "Chain-of-Thought"
  - "MMLU Split"
  - "Knowledge Recall"
  - "Symbolic Reasoning"
date: 2026-05-08
content_hash: f0bb8d02819cc459
---

# Does Self-Consistency Improve the Recall of Encyclopedic Knowledge?

**Conference**: ACL 2026  
**arXiv**: [2604.19395](https://arxiv.org/abs/2604.19395)  
**Code**: N/A (repository not released)  
**Area**: LLM Reasoning / Knowledge Recall / Evaluation  
**Keywords**: Self-Consistency, Chain-of-Thought, MMLU Split, Knowledge Recall, Symbolic Reasoning

## TL;DR
Applying the "==" heuristic from Sprague et al. at the subject level, the authors split the 57 subjects of MMLU into two subsets: **symbolic reasoning** and **knowledge recall** (approximately a 1:2 ratio). They empirically demonstrate that self-consistency (SC) is not only effective for symbolic reasoning—a domain where CoT already excels—but also provides **consistent gains in knowledge recall** (+2.48 for $n=20$). This pushes the overall MMLU accuracy of GPT-4o to 88.93%. The mechanism is explained using the "ratio of the majority answer" as a confidence signal (Pearson $\rho \approx 0.42$).

## Background & Motivation

**Background**: Chain-of-Thought (CoT) is a standard for LLM evaluation, but Sprague et al. (2025) reported that 95% of CoT gains on MMLU originate from questions involving symbolic reasoning (formulas/numerical values); questions relying purely on "encyclopedic knowledge recall" see almost no benefit. Self-Consistency (SC) builds on CoT by sampling multiple reasoning paths and applying a majority vote, primarily validated previously in arithmetic and commonsense reasoning.

**Limitations of Prior Work**: Since SC is entirely built upon CoT, and CoT is reportedly ineffective for knowledge recall questions (e.g., "How high is Mount Fuji?"), is SC also useless for knowledge recall? Although Chung et al. (2024) tested SC on MMLU, they **did not report symbolic reasoning and knowledge recall separately**. These signals were conflated, obscuring the true contribution of SC across different problem types.

**Key Challenge**: MMLU is a hybrid task. Its 57 subjects include both typical symbolic reasoning (math/physics/econometrics) and pure knowledge recall (history/law/medical facts). However, the official supercategory classification (STEM/humanities) is based on academic disciplines and is **not orthogonal to reasoning types** (e.g., econometrics is grouped under humanities). Without a clean a priori split, the real utility of SC for knowledge recall cannot be answered.

**Goal**: (RQ1) Can knowledge recall questions benefit from multiple reasoning paths? (RQ2) If so, through what mechanism?

**Key Insight**: Rather than training a new question classifier, the authors reuse the "==" heuristic from Sprague et al. (2025) (which labels a question as symbolic reasoning if "=" appears in the question or answer). They upgrade this from a **post-hoc instance-level model-dependent** classification to a stable **a priori subject-level model-agnostic** split.

**Core Idea**: Using a simple yet stable a priori subject-level dichotomy, MMLU is split into reasoning and knowledge halves. Mirrored validation is performed using GSM8K (pure symbolic reasoning) and MedMCQA (almost pure knowledge recall) as prototypical benchmarks. Re-testing SC on this clean experimental setup proves its effectiveness for knowledge recall and formalizes the "majority answer ratio" as a confidence signal.

## Method

### Overall Architecture
The method involves no model training; all experiments are zero-shot reasoning. The workflow: (1) Use the "==" heuristic to perform a priori subject-level clustering on the 57 MMLU subjects, yielding reasoning and knowledge subsets (ratio ~1:2); (2) Validate the split on prototypical benchmarks GSM8K/MedMCQA—CoT gains on the reasoning subset and GSM8K should be significantly higher than on the knowledge subset and MedMCQA; (3) Run Direct Answer (DA) vs CoT vs CoT+SC ($n \in \{3, 5, 20\}$) on GPT-4o (2024-08-06) / GPT-4o-mini / Qwen2.5-32B-Instruct using nucleus sampling ($top-p=0.9$); (4) Introduce the confidence signal $s = \frac{|\text{majority answer}|}{|\text{valid answers}|}$ and conduct Pearson correlation analysis with accuracy.

### Key Designs

1.  **a priori Subject-Level Split (MMLU Reasoning vs Knowledge)**:
    - **Function**: Deconstructs MMLU into two subsets with pure reasoning types, allowing the utility of SC to be measured separately across different problem types.
    - **Mechanism**: Reuses the instance-level heuristic from Sprague et al. (2025)—if a question/answer contains mathematical equality symbols like "=", it is treated as a symbolic reasoning sample. However, the authors aggregate this signal at the **subject level** (a subject is classified as reasoning if the "==" occurrence rate exceeds a threshold) and propagate the classification within subject clusters (e.g., if college math is classified, elementary math is too). This results in a reasoning:knowledge ratio of ≈ 1:2. Validation in Appendix E using CoT gain curves shows an AUC of 0.96, proving the heuristic is highly consistent with the subset where CoT actually provides the most gain.
    - **Design Motivation**: Post-hoc instance-level classification depends on model output (different models classify differently, making results non-reproducible). Subject-level a priori classification is stable, model-agnostic, and easy for other researchers to reuse.

2.  **Cross-Benchmark Mirror Validation (GSM8K ↔ MMLU-Reasoning, MedMCQA ↔ MMLU-Knowledge)**:
    - **Function**: Uses two reasoning-pure external benchmarks to "anchor" and validate the correctness of the MMLU split.
    - **Mechanism**: GSM8K is pure arithmetic reasoning; MedMCQA consists of medical MCQs where only 16 out of 4,183 questions contain "==" (≈0.4%), making it pure knowledge recall. If the MMLU split is correct, the gain patterns of CoT on MMLU-Reasoning and GSM8K should match, as should the patterns for MMLU-Knowledge and MedMCQA. Data in Table 1 confirms this: CoT gains +37.3 on GSM8K / +14.9 on MMLU-Reasoning; +1.69 on MedMCQA / +1.56 on MMLU-Knowledge.
    - **Design Motivation**: As MMLU is a hybrid dataset, it is difficult to prove subset "purity" internally. Introducing two prototypical benchmarks with clear boundaries for mirror validation is a much more credible and cost-effective method than manual re-annotation.

3.  **Answer Consistency as Confidence Signal $s$ and Mechanism Analysis**:
    - **Function**: Upgrades the SC majority vote from a "voting mechanism" to a quantifiable confidence signal and validates its correlation with accuracy.
    - **Mechanism**: Define $s = \frac{\text{count of majority answer}}{\text{number of valid answers}}$; for $\{A, A, C\}$, the confidence for majority A is $s = 2/3$. The Pearson correlation $\rho$ between $s$ and prediction accuracy on MMLU is 0.40 at $n=5$ and 0.42 at $n=20$ (0.46 for reasoning, 0.42 for knowledge). This indicates that the majority ratio is a reliable confidence signal that holds **across problem types**. Thus, SC works on knowledge recall not by "exploring and synthesizing paths," but by "filtering out unstable paths that produce different conclusions."
    - **Design Motivation**: Explains why SC is effective for knowledge recall, refuting the intuition that "knowledge recall is a single-step deduction where multiple paths are meaningless." Qualitative examples show that even for knowledge questions, LLMs hallucinate multiple plausible but conflicting justifications; SC effectively filters this instability.

### Loss & Training
Zero training. Inference configuration: 14,042 test samples for MMLU, 1,319 for GSM8K, and 4,183 validation samples for MedMCQA. For GPT-4o-mini, a 285-sample dev set was used for zero/few-shot comparisons. Max output: 1000 tokens for CoT, 20 tokens for DA. Significance tests used paired bootstrap resampling ($p < 0.05$).

## Key Experimental Results

### Main Results
**Accuracy of GPT-4o on MMLU / GSM8K / MedMCQA (%)**:

| Prompt / Sampling | MMLU All | MMLU Reasoning | MMLU Knowledge | GSM8K | MedMCQA |
|-------------------|----------|----------------|----------------|-------|---------|
| DA, nucleus | 83.26 | 75.45 | 85.56 | 46.93 | 75.07 |
| CoT, nucleus | 87.86 (+4.60) | 90.38 (+14.93) | 87.12 (+1.56) | 84.23 (+37.30) | 76.76 (+1.69) |
| CoT + SC ($n=5$) | 88.64 (+5.38) | 91.32 (+15.87) | 87.85 (+2.29) | 84.31 (+37.38) | 77.67 (+2.60) |
| CoT + SC ($n=20$) | **88.93 (+5.67)** | **91.94 (+16.49)** | **88.04 (+2.48)** | **84.46 (+37.53)** | 77.41 (+2.34) |

The bulk of CoT improvements comes from reasoning (+14.93 vs +1.56), consistent with Sprague et al. However, SC still adds +0.92 on knowledge (87.12 → 88.04), achieving $p < 0.05$ significance. SC also adds +0.6~0.9 on MedMCQA, showing consistent directionality.

### Ablation Study

| Configuration | MMLU dev (GPT-4o-mini) All / Reasoning / Knowledge | Description |
|------|--------------------------------------------------|------|
| 0-shot CoT | 80.35 / 84.44 / 78.46 | baseline |
| 0-shot CoT+SC ($n=5$) | 83.16 / 86.67 / 81.54 | zero-shot + SC overall +2.81 |
| 0-shot CoT+SC ($n=20$) | 82.81 / 88.89 / 80.00 | reasoning subset +4.45, max gain |
| 4-shot CoT | 80.35 / 80.00 / 80.51 | equal to 0-shot CoT |
| 4-shot CoT+SC ($n=20$) | 82.46 / 85.56 / 81.03 | few-shot + SC overall +2.11 |
| Pearson $\rho$ ($n=5$) | All 0.40 / Reasoning 0.43 / Knowledge 0.40 | effective confidence signal |
| Pearson $\rho$ ($n=20$) | All 0.42 / Reasoning 0.46 / Knowledge 0.42 | correlation increases with $n$ |

### Key Findings
- **SC is effective for knowledge recall**: On MMLU-Knowledge, $n=20$ SC is +0.92 higher than vanilla CoT, achieving statistical significance and overturning the assumption that SC only helps symbolic reasoning.
- **Majority ratio $s$ is a cross-task confidence signal**: With $\rho \approx 0.42$, it is more robust to first-token bias than logit-based confidence (e.g., the "My answer is C" phenomenon in Wang et al. 2024a).
- **Zero-shot can outperform few-shot**: Table 3 shows 0-shot CoT+SC on GPT-4o-mini is comparable to or better than 4-shot, suggesting arbitrary answer extraction combined with SC is more flexible than fixed exemplar constraints.
- **GPT-4o hits 88.93% on MMLU**: Best in class for models of its time; shows SC remains an underrated simple trick in the GPT-4 era.
- **Cost is linear computation**: $n=5$ spends 5× inference compute for a +0.7 gain in knowledge; returns are diminishing, requiring trade-offs for industrial deployment.
- **Failure Qualitative Observation**: Even for knowledge questions, LLMs generate multiple "plausible sounding" but conflicting explanations. SC stabilizes output by filtering paths that lead to minority answers.

## Highlights & Insights
- **"a priori Split" Research Paradigm**: By defining a stable, model-independent question classification before measuring utility, it creates a "reusable experimental bench"—other researchers can apply the reasoning/knowledge split to test their own methods.
- **Cross-Benchmark Mirror Validation**: Using GSM8K + MedMCQA as "anchors" for pure reasoning types to validate MMLU subsets is a low-cost, high-credibility validation technique that can be transferred to any hybrid benchmark.
- **majority vote → Confidence Signal**: Formalizing majority voting as $s$ and measuring its correlation with accuracy provides a **mechanistic explanation** for SC beyond simple metric comparison. This elevates an "engineering trick" to an "interpretable method."
- **Refuting the "CoT doesn't help knowledge recall $\Rightarrow$ SC won't either" inference**: The authors use counterfactuals to show that CoT and SC solve different problems—CoT upgrades "no reasoning" to "reasoning," while SC upgrades "unstable reasoning" to "stable reasoning." The latter is useful for all types (even 1-step deduction).

## Limitations & Future Work
- **Subject-level split is a coarse approximation**: The authors admit some subjects have mixed question types, making subject-level splitting sub-optimal. Instance-level classifiers could be more precise.
- **SC cost scales linearly with $n$**: $n=5$ for +0.7 knowledge accuracy is often not worth it for industrial use; the paper does not discuss compute optimization (e.g., early stopping/adaptive $n$).
- **Task format limited to MCQA + open-ended numerical**: Though universal SC can extend to any task, this paper does not test SC on generative tasks like summarization or translation.
- **Missing comparisons with stronger paradigms**: Comparison with more complex reasoning enhancements like self-refine or Tree-of-Thought is missing, making it hard to judge marginal gains over stronger baselines.
- **GPT-4o Black-box**: Analysis cannot access internal representations, leaving mechanistic explanations at the prompt-output behavioral level.

## Related Work & Insights
- **vs Sprague et al. 2025 ("To CoT or not to CoT")**: They used instance-level post-hoc classification to argue "CoT mainly helps symbolic reasoning." This paper upgrades to a priori subject-level splitting and extends the analysis from CoT to SC, reaching a stronger conclusion that "SC helps both."
- **vs Chung et al. 2024 (Flan-PaLM SC on MMLU)**: They ran SC on MMLU but lacked subject-level decomposition; this paper is the first to explicitly quantify SC's contribution to "knowledge recall" subsets.
- **vs Wang et al. 2023 (Original SC paper)**: The original work focused on arithmetic/commonsense reasoning; this paper extends to knowledge recall, showing SC is more broadly applicable than originally envisioned.
- **vs MMLU-Pro / MMLU-CF / MMLU-Redux**: These works focus on instance-level data correction; this paper focuses on subject-level grouping. Both are orthogonal and can be combined.

## Rating
- Novelty: ⭐⭐⭐ Does not propose new technology, but the research question is sharp and the approach is clever. The "a priori subject-level split + cross-benchmark mirror validation" is a valuable methodological contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 models × 3 benchmarks × multiple $n$ × 0-shot/4-shot × Pearson analysis + qualitative cases + bootstrap significance tests. Highly dense for a small paper.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem statement is clear, and the motivation chain (CoT → SC → knowledge recall) is seamless. Tables are minimalist but information-dense; an exemplar of short paper writing.
- Value: ⭐⭐⭐⭐ Directly actionable—all LLM evaluation studies should report "knowledge subset improvement" as a separate dimension; $s$ as a general confidence signal is ready for calibration research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning](self-consistency_from_only_two_samples_cot-pot_ensembling_for_efficient_llm_reas.md)
- [\[ACL 2026\] Reliability-Aware Adaptive Self-Consistency for Efficient Sampling in LLM Reasoning](reliability-aware_adaptive_self-consistency_for_efficient_sampling_in_llm_reason.md)
- [\[ACL 2026\] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)
- [\[NeurIPS 2025\] A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning](../../NeurIPS2025/llm_reasoning/a_theoretical_study_on_bridging_internal_probability_and_sel.md)
- [\[ACL 2026\] Towards Effective In-context Cross-domain Knowledge Transfer via Domain-invariant-neurons-based Retrieval](towards_effective_in-context_cross-domain_knowledge_transfer_via_domain-invarian.md)

</div>

<!-- RELATED:END -->
