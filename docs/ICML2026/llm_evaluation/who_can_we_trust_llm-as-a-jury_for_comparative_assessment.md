---
title: >-
  [Paper Note] Who can we trust? LLM-as-a-jury for Comparative Assessment
description: >-
  [ICML 2026][LLM Evaluation][LLM-as-a-jury] This paper points out that the reliability of multiple LLM judges in pairwise comparisons varies significantly. It proposes the BT-$\sigma$ model with judge-specific discrimination parameters, which simultaneously learns the ranking of candidate outputs and the reliability of each LLM judge without human calibration labels, thereby aligning more closely with human rankings than simple averaging or standard Bradley-Terry aggregation.
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "LLM-as-a-jury"
  - "Bradley-Terry"
  - "Reviewer Reliability"
  - "Pairwise Comparison"
  - "Unsupervised Calibration"
date: 2026-05-08
content_hash: 006d2eb7096034c3
---

# Who can we trust? LLM-as-a-jury for Comparative Assessment

**Conference**: ICML 2026  
**arXiv**: [2602.16610](https://arxiv.org/abs/2602.16610)  
**Code**: No public code  
**Area**: LLM Evaluation / Comparative Automated Evaluation  
**Keywords**: LLM-as-a-jury, Bradley-Terry, Reviewer Reliability, Pairwise Comparison, Unsupervised Calibration  

## TL;DR
This paper points out that the reliability of multiple LLM judges in pairwise comparisons varies significantly. It proposes the BT-$\sigma$ model with judge-specific discrimination parameters, which simultaneously learns the ranking of candidate outputs and the reliability of each LLM judge without human calibration labels, thereby aligning more closely with human rankings than simple averaging or standard Bradley-Terry aggregation.

## Background & Motivation
**Background**: LLM-as-a-judge has become a common tool for evaluating NLG, summarization, dialogue responses, and open-ended generation. Compared to direct scoring, pairwise comparison is generally more stable. Consequently, many systems involve one or more LLM judges determining whether candidate output $i$ is superior to $j$, then aggregating these comparison results into a global ranking.

**Limitations of Prior Work**: The quality of multiple LLM judges is inconsistent. Some models exhibit length bias, some are sensitive to candidate ordering, and some show severe cycle contradictions across different evaluation dimensions. Common probability or voting averages assume all judges are equally reliable, weighting noisy models and high-quality models equally, which causes the final ranking to be dragged down by inconsistency probabilities.

**Key Challenge**: Pairwise comparisons should ideally satisfy a certain global ranking structure, but the preference probabilities provided by LLMs often violate transitivity, commutativity, and calibration consistency. Directly using soft probabilities preserves more information but also amplifies inconsistencies; using only hard decisions is more robust but discards probability intensity.

**Goal**: The authors aim to simultaneously recover the global skill ranking of candidate items and the reliability or discriminative power of each judge from the pairwise comparison probabilities of multiple LLM judges, without relying on human-annotated calibration sets.

**Key Insight**: Starting from the Bradley-Terry model, the paper first analyzes when soft BT self-calibrates and when it fails due to probability inconsistency. It then treats "judge reliability" as a model parameter rather than manually specifying weights before aggregation.

**Core Idea**: A learnable discrimination scale $\sigma_k$ is added to each LLM judge, making reliable judges more sensitive to skill differences and naturally down-weighting noisy judges, forming an unsupervised reliability-aware BT aggregation.

## Method
The main logic of the paper is clear: first, LLM comparison probabilities are placed into the Bradley-Terry framework to show that standard soft BT in multi-judge scenarios is equivalent to matching average probabilities; then, it is noted that average probabilities cannot express reliability differences between judges; finally, BT-$\sigma$ is proposed to learn item skills and judge discriminators within the same likelihood.

### Overall Architecture
The input consists of a set of candidate generation results and preference probabilities from multiple LLM judges for all candidate pairs. For each pair $(i,j)$ and judge $k$, the model observes $p_{ij}^{(k)}$, the probability that judge $k$ believes $i$ is superior to $j$. The output includes global ranking scores $s_i$ for candidate items and reliability parameters $\sigma_k$ for each judge. During evaluation, the candidate ranking is compared against human rankings using Spearman rank correlation (SRC).

The method first performs a symmetric debiasing: if the same candidate pair yields $p_{ij}$ and $p_{ji}$ under two different orders, it uses $p'_{ij}=\frac{1}{2}(p_{ij}+1-p_{ji})$ to enforce basic order consistency. Subsequently, methods like hard BT, soft BT, Temp-BT, and BT-$\sigma$ are compared on the same set of debiased comparisons.

### Key Designs

**1. Using probability consistency to diagnose the boundaries of hard BT vs. soft BT**: The paper addresses a counter-intuitive phenomenon—why soft BT, which preserves probability intensity, sometimes performs worse than hard BT, which only considers the win/loss direction. The standard Bradley-Terry assumes $P(i\succ j)=\sigma(s_i-s_j)$, and soft BT fits this structure using observed probabilities $p_{ij}$. The authors prove: when the evaluation probabilities are self-consistent (generated from a global skill vector), temperature scaling only scales the skill space and does not change the ranking; soft BT implicitly completes self-calibration. In this case, hard BT and soft BT yield the same ranking. However, real LLM probabilities often violate transitivity and commutativity; fitting these contradictory intensities amplifies noise. Hard BT, by discarding magnitude, acts as a more noise-resistant estimator. This diagnosis is the starting point: the issue is not the BT structure, but the varying signal quality of different judges, which should not be treated equally.

**2. BT-$\sigma$: A learnable discrimination scale $\sigma_k$ for each reviewer**: This is the core of the paper. It is proven that feeding all judges' probabilities directly into soft BT is equivalent to averaging the probabilities first and then fitting soft BT, which fails to capture reliability differences. BT-$\sigma$ inserts a discrimination scale $\sigma_k$ for each judge $k$ into the soft BT likelihood: $\mathcal{L}(\mathbf{s},\{\sigma_k\})\propto\prod_k\prod_{(i,j)}\sigma((s_i-s_j)/\sigma_k)^{p_{ij}^{(k)}}(1-\sigma((s_i-s_j)/\sigma_k))^{1-p_{ij}^{(k)}}$. $\sigma_k$ controls the sensitivity of judge $k$ to skill differences: a smaller $\sigma_k$ indicates higher sensitivity, more self-consistent probabilities, and higher reliability; a larger $\sigma_k$ indicates flatter and noisier probabilities. All $\{s_i\}$ and $\{\sigma_k\}$ are jointly maximized in the same likelihood without any human labels. This is essentially an unsupervised version of temperature scaling—where the calibration signal comes from the multi-judge comparison structure itself, automatically granting reliable judges more weight. The paper emphasizes that $\sigma_k$ is only meaningful in "multi-judge + soft probability" scenarios; under single-judge or hard BT, the global scale $\sigma_k$ is absorbed by item skills.

**3. Validating that $\sigma_k$ captures reliability and extension to aspects**: Since $\sigma_k$ could just be a mathematical degree of freedom, it must be proven to correspond to "reliability." The authors correlate the learned $1/\sigma_k$ with the independent SRC of the judge and $(1-\text{CycleRate})$ (where CycleRate counts triplets like $i\succ j\succ k\succ i$). If consistent judges learn larger $1/\sigma_k$, the model captures actual reliability rather than overfitting. The paper also proposes BT-$\sigma$-asp, learning a separate scale for each "judge $\times$ aspect" pair to test if reliability varies by dimension; experiments find a single $\sigma_k$ per judge is usually sufficient, suggesting reliability is largely stable across dimensions.

### Loss & Training
BT-$\sigma$ directly maximizes the joint likelihood above. Parameters include all item skills $\{s_i\}$ and judge discriminators $\{\sigma_k\}$. Optimization is performed using L-BFGS-B with random initialization for $s_i$ and $\sigma_k$, typically converging within 100 iterations. Temp-BT serves as a supervised baseline requiring human labels to fit temperatures; BT-$\sigma$ uses no human labels.

## Key Experimental Results

### Main Results
The paper tests on SummEval, Topical-Chat, and NovelEval. The main table reports Spearman correlation for SummEval and Topical-Chat. SummEval includes coherence, consistency, fluency, and relevance; Topical-Chat includes coherency, continuity, engagingness, and naturalness.

| Dataset | Metric | Ours (BT-σ) | Prev. Strong Baseline | Gain |
|--------|------|------|----------|------|
| SummEval COH | SRC | 57.38 | soft BT 53.94 / Temp-BT 56.21 | +3.44 over unsupervised soft BT |
| SummEval FLU | SRC | 42.99 | soft BT 42.69 / Temp-BT 41.88 | Slight lead |
| SummEval REL | SRC | 54.15 | soft BT 53.11 / Temp-BT 55.14 | Better than soft BT, lower than supervised Temp-BT |
| Topical-Chat CNT | SRC | 56.30 | soft BT 53.87 / Temp-BT 52.21 | +2.43 pts vs soft BT |
| Topical-Chat NAT | SRC | 60.56 | soft BT 58.20 / Temp-BT 60.65 | Close to supervised calibration |
| SummEval ALL | SRC | 50.50 | soft BT 49.40 / Crowd-BT 48.35 | Overall lead |

### Ablation Study
Ablations focus on whether discriminators represent judge reliability and if aspect-specific discriminators are necessary.

| Configuration | Key Metric | Description |
|------|---------|------|
| SummEval, $1/\sigma_k$ vs judge SRC | ALL PCC 72.21 / SRC 85.71 | Discriminator highly correlates with independent judge performance |
| Topical-Chat, $1/\sigma_k$ vs judge SRC | ALL PCC 67.41 / SRC 59.52 | Positive correlation maintained across tasks |
| SummEval, $1/\sigma_k$ vs $1-\text{CycleRate}$ | ALL PCC 90.29 / SRC 95.24 | More consistent judges learn larger $1/\sigma_k$ |
| BT-σ-asp vs BT-σ | Small gain on SummEval, mixed on Topical-Chat | Aspect-relative reliability exists but gives limited gains |
| hard BT-σ on Topical-Chat ENG | SRC 67.36 | Hard decision + reliability modeling is more stable in high-noise dimensions |

### Key Findings
- For a single LLM judge, hard BT often matches or exceeds soft BT, indicating raw probability magnitudes are not always trustworthy. After multi-judge aggregation, soft BT becomes stronger as noise from different models partially cancels out.
- The advantage of BT-$\sigma$ comes from explicitly modeling judge heterogeneity. Instead of simply averaging probabilities, it allows different temperature slopes for different judges in the likelihood, naturally attenuating unreliable models.
- $1/\sigma_k$ correlates extremely highly with cycle consistency (up to 95.24 SRC on SummEval ALL). This is strong evidence that the discriminator captures "ease of generating preference cycles," a core reliability dimension.

## Highlights & Insights
- The paper transforms "LLM judge credibility" from engineering empirical knowledge into learnable parameters. While many pipelines manually select models or use majority voting, BT-$\sigma$ provides a probabilistic alternative requiring no human labels.
- The explanation of hard BT vs. soft BT is valuable. It serves as a reminder that probability outputs are not necessarily better than binary preferences; if the probabilities themselves do not satisfy a global ranking structure, preserving intensity may just preserve noise.
- The interpretability of $\sigma_k$ is comprehensive. The authors do not just report scores but verify correlations with cycle inconsistency, making the method a diagnostic tool.

## Limitations & Future Work
- BT-$\sigma$ is still based on the assumption of a global Bradley-Terry skill. If candidate outputs involve context-dependent, non-transitive human preferences or multi-modal preference groups, a single skill vector may be oversimplified.
- The paper mainly targets offline NLG benchmark comparisons. In real-world open-ended evaluation, prompt rubrics and length biases are more complex; the stability of $\sigma_k$ needs further testing.
- Temp-BT still holds advantages in some dimensions, indicating that if high-quality annotations exist, supervised calibration remains valuable. Future work could explore semi-supervised combinations.
- BT-$\sigma$ estimates judge-level reliability and does not directly handle instance-level reliability. Certain judges may fail only on specific types of samples, requiring finer-grained conditional discriminators.

## Related Work & Insights
- **vs. Avg-Prob / majority voting**: Simple averaging treats all judges equally; this work learns soft weights via $\sigma_k$ and enforces a global ranking structure.
- **vs. hard / soft Bradley-Terry**: Standard BT only learns item skill; this work incorporates the judge's probability scale into the model.
- **vs. supervised temperature scaling**: Temp-BT requires human labels for fitting; BT-$\sigma$ uses pairwise structures for self-supervised learning of discriminators.
- **vs. Crowd-BT / annotator aggregation**: Crowdsourcing models usually assume repeated annotations and latent ground truth; this work handles LLM soft probabilities in ranking recovery.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Embeds judge reliability into BT likelihood; identifies the core issue with a concise model.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple NLG datasets, judges, and dimensions with reliability analysis.
- Writing Quality: ⭐⭐⭐⭐☆ Theory, formulas, and experimental phenomena are well-connected; explanation of hard/soft BT is very clear.
- Value: ⭐⭐⭐⭐☆ Practical for automated evaluation systems as a lightweight module for aggregation and judge diagnosis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The ACUTE Protocol: Operationalizing Language Model Activations for Better Calibration, Utility, and Trust](the_acute_protocol_operationalizing_language_model_activations_for_better_calibr.md)
- [\[ICML 2026\] Who Flips? Self- and Cross-Model Counterarguments Reveal Answer Instability in LLMs](who_flips_self-_and_cross-model_counterarguments_reveal_answer_instability_in_ll.md)
- [\[ACL 2025\] CoV-Eval: Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective](../../ACL2025/llm_evaluation/cov_eval_evaluating_llms_from_code_security_perspective.md)
- [\[ICLR 2026\] Rethinking LLM Evaluation: Can We Evaluate LLMs with 200× Less Data?](../../ICLR2026/llm_evaluation/rethinking_llm_evaluation_can_we_evaluate_llms_with_200_less_data.md)
- [\[ACL 2026\] Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation](../../ACL2026/llm_evaluation/teaching_language_models_to_forecast_research_success_through_comparative_idea_e.md)

</div>

<!-- RELATED:END -->
