---
title: >-
  [Paper Note] Who can we trust? LLM-as-a-jury for Comparative Assessment
description: >-
  [ICML 2026][LLM Evaluation][LLM-as-a-jury] This paper points out that the reliability of different LLM judges varies significantly in pairwise comparisons. It proposes the BT-$\sigma$ model with judge-specific discrimination parameters to simultaneously learn candidate output rankings and the reliability of each LLM judge without manual calibration labels, achi
tags:
  - ICML 2026
  - LLM Evaluation
  - LLM-as-a-jury
  - Bradley-Terry
date: 2026-05-08
content_hash: 7c1453d78fd04624
---
# Who can we trust? LLM-as-a-jury for Comparative Assessment

**Conference**: ICML 2026  
**arXiv**: [2602.16610](https://arxiv.org/abs/2602.16610)  
**Code**: No public code  
**Area**: LLM Evaluation / Comparative Automatic Evaluation  
**Keywords**: LLM-as-a-jury, Bradley-Terry, Reviewer Reliability, Pairwise Comparison, Unsupervised Calibration  

## TL;DR
This paper points out that the reliability of different LLM judges varies significantly in pairwise comparisons. It proposes the BT-$\sigma$ model with judge-specific discrimination parameters to simultaneously learn candidate output rankings and the reliability of each LLM judge without manual calibration labels, achieving a closer alignment with human rankings than simple averaging or standard Bradley-Terry aggregation.

## Background & Motivation
**Background**: LLM-as-a-judge has become a common tool for evaluating NLG, summarization, dialogue responses, and open-ended generation. Compared to direct scoring, pairwise comparisons are generally more stable. Consequently, many systems employ one or more LLM judges to determine if candidate output $i$ is superior to $j$, aggregating these results into a global ranking.

**Limitations of Prior Work**: The quality of multiple LLM judges is inconsistent. Some models exhibit length bias, others are sensitive to the order of candidates, and some show severe cyclical contradictions across different evaluation dimensions. Common probability or voting averages assume all judges are equally reliable, weighting noisy models and high-quality models equally, which causes the final ranking to be degraded by inconsistency.

**Key Challenge**: Pairwise comparisons should ideally satisfy a global ordering structure, but the preference probabilities provided by LLMs often violate transitivity, exchangeability, and calibration consistency. Directly using soft probabilities preserves more information but also amplifies inconsistencies; using only hard decisions is more robust but discards probabilistic intensity.

**Goal**: This paper aims to recover the global skill ranking of candidate items and the reliability or discriminative power of each judge from the pairwise comparison probabilities of multiple LLM judges, without relying on human-annotated calibration sets.

**Key Insight**: Starting from the Bradley-Terry model, the paper first analyzes when soft BT self-calibrates and when it fails due to probability inconsistency. It then treats "judge trustworthiness" as a model parameter rather than a manually assigned weight before aggregation.

**Core Idea**: A learnable discriminative scale $\sigma_k$ is added for each LLM judge. This makes reliable judges more sensitive to skill differences while naturally downweighting noisy judges, resulting in an unsupervised, reliability-aware BT aggregation.

## Method
The main logic of the paper is clear: first, incorporate LLM comparison probabilities into the Bradley-Terry framework to show that standard soft BT in multi-judge scenarios is equivalent to matching the average probability; then, point out that average probabilities fail to represent reliability differences between judges; finally, propose BT-$\sigma$ to learn item skill and judge discriminators within a single likelihood.

### Overall Architecture
The input consists of a set of candidate generations and the preference probabilities for all candidate pairs from multiple LLM judges. For each pair $(i,j)$ and judge $k$, the model observes $p_{ij}^{(k)}$, the probability that judge $k$ believes $i$ is better than $j$. The output includes global ranking scores $s_i$ for candidates and reliability parameters $\sigma_k$ for each judge. Evaluation is performed using the Spearman rank correlation between the candidate ranking and human score rankings.

The method first applies symmetric debiasing: if the same pair of candidates yields $p_{ij}$ and $p_{ji}$ under two different orderings, the basic temporal consistency is enforced using $p'_{ij}=\frac{1}{2}(p_{ij}+1-p_{ji})$. Subsequently, methods like hard BT, soft BT, Temp-BT, and BT-$\sigma$ are compared on the same set of debiased comparisons.

### Key Designs

**1. Diagnosing the boundaries of hard BT vs. soft BT via probability consistency**: The paper first addresses a counter-intuitive phenomenon—why soft BT, which retains probability intensity, sometimes performs worse than hard BT, which only considers the winner. Standard Bradley-Terry assumes $P(i\succ j)=\sigma(s_i-s_j)$, and soft BT fits this structure using observed probabilities $p_{ij}$. The authors prove that when the review probabilities are self-consistent (generated by some global skill vector), temperature scaling only scales the skill space and does not change the ranking. In this case, soft BT implicitly completes self-calibration, and hard BT yields the same ranking as soft BT. However, real LLM probabilities often violate transitivity and exchangeability, making them unexplainable by a single skill vector. In such cases, soft BT amplifies noise by fitting contradictory intensities, while hard BT becomes a more robust estimator by discarding magnitude and retaining only direction. This diagnosis is the starting point of the paper: the problem lies not in the BT structure, but in the varying quality of signals from different reviewers, which cannot be treated with equal weight.

**2. BT-$\sigma$: Learnable discriminative scale $\sigma_k$ for each reviewer**: This is the core of the paper. The authors prove that feeding all reviewers' probabilities directly into soft BT is equivalent to averaging the probabilities first and then fitting a soft BT, which fails to capture reliability differences. BT-$\sigma$ inserts a discriminative scale $\sigma_k$ for each reviewer $k$ into the soft BT likelihood:
$$\mathcal{L}(\mathbf{s},\{\sigma_k\})\propto\prod_k\prod_{(i,j)}\sigma((s_i-s_j)/\sigma_k)^{p_{ij}^{(k)}}(1-\sigma((s_i-s_j)/\sigma_k))^{1-p_{ij}^{(k)}}$$
$\sigma_k$ controls the sensitivity of reviewer $k$ to skill differences. A smaller $\sigma_k$ indicates the reviewer is more sensitive to differences, more self-consistent, and more trustworthy. A larger $\sigma_k$ indicates flatter and noisier probabilities. All $\{s_i\}$ and $\{\sigma_k\}$ are jointly maximized in the same likelihood without any human labels. This is essentially an unsupervised version of temperature scaling—calibration signals come from the multi-reviewer comparison structure itself rather than human annotations, automatically giving reliable reviewers higher weight. The paper also emphasizes that $\sigma_k$ is only meaningful in "multi-reviewer + soft probability" scenarios; in single-reviewer or hard BT cases, the global scale $\sigma_k$ is absorbed by item skill, losing information.

**3. Validating that $\sigma_k$ captures reliability and extending to aspect dimensions**: To ensure $\sigma_k$ is not just a mathematical degree of freedom, the authors prove it corresponds to "reliability." They analyze the correlation between the learned $1/\sigma_k$ and the reviewer's independent SRC, as well as $1-\text{CycleRate}$ (CycleRate measures the proportion of directed cycles like $i\succ j\succ k\succ i$). If more consistent reviewers learn larger $1/\sigma_k$, the model has captured true reliability rather than over-fitting a benchmark. The paper also proposes BT-$\sigma$-asp, which learns separate scales for each "reviewer $\times$ evaluation dimension" pair. Experiments show that one $\sigma_k$ per reviewer is generally sufficient, suggesting reliability is largely stable across dimensions.

### Loss & Training
BT-$\sigma$ directly maximizes the joint likelihood mentioned above. Parameters include all item skills $\{s_i\}$ and judge discriminators $\{\sigma_k\}$. The authors use L-BFGS-B for optimization, with random initialization for $s_i$ and $\sigma_k$, typically converging within 100 iterations. Temp-BT serves as a supervised reference, requiring human annotations to fit temperatures for each judge/aspect. BT-$\sigma$ does not use human labels and relies solely on LLM pairwise probabilities.

## Key Experimental Results

### Main Results
Testing was conducted on SummEval, Topical-Chat, and NovelEval. The main table reports Spearman correlation for SummEval and Topical-Chat. SummEval includes coherence, consistency, fluency, and relevance; Topical-Chat includes coherency, continuity, engagingness, and naturalness.

| Dataset | Metric | Ours (BT-$\sigma$) | Prev. Strong Baseline | Gain |
| :--- | :--- | :--- | :--- | :--- |
| SummEval COH | SRC | 57.38 | soft BT 53.94 / Temp-BT 56.21 | +3.44 over unsupervised soft BT |
| SummEval FLU | SRC | 42.99 | soft BT 42.69 / Temp-BT 41.88 | Slight lead |
| SummEval REL | SRC | 54.15 | soft BT 53.11 / Temp-BT 55.14 | Better than soft BT, lower than supervised Temp-BT |
| Topical-Chat CNT | SRC | 56.30 | soft BT 53.87 / Temp-BT 52.21 | +2.43 vs soft BT |
| Topical-Chat NAT | SRC | 60.56 | soft BT 58.20 / Temp-BT 60.65 | Close to supervised calibration |
| SummEval ALL | SRC | 50.50 | soft BT 49.40 / Crowd-BT 48.35 | Overall lead |

### Ablation Study
Ablations and analyses focus on whether the learned discriminator represents judge reliability and if aspect-specific discriminators are necessary.

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| SummEval, $1/\sigma_k$ vs judge SRC | ALL PCC 72.21 / SRC 85.71 | Discriminator highly correlated with independent judge performance |
| Topical-Chat, $1/\sigma_k$ vs judge SRC | ALL PCC 67.41 / SRC 59.52 | Maintains positive correlation across tasks |
| SummEval, $1/\sigma_k$ vs $1-\text{CycleRate}$ | ALL PCC 90.29 / SRC 95.24 | More consistent judges learn larger $1/\sigma_k$ |
| BT-$\sigma$-asp vs BT-$\sigma$ | Slight gain on SummEval, mixed on Topical-Chat | Aspect-related reliability exists but offers limited gains |
| hard BT-$\sigma$ on Topical-Chat ENG | SRC 67.36 | Hard decision + reliability modeling is more stable in high-noise dimensions |

### Key Findings
- For a single LLM judge, hard BT often matches or exceeds soft BT, indicating that raw probability magnitudes are not always trustworthy. After multi-judge aggregation, soft BT becomes stronger, as noise from different models partially cancels out.
- The advantage of BT-$\sigma$ stems from explicitly modeling judge heterogeneity. Instead of simply averaging probabilities, it allows different temperature curves for different judge probabilities within the likelihood, naturally weakening unreliable models.
- The correlation between $1/\sigma_k$ and cycle consistency is extremely high, especially for SummEval ALL SRC at 95.24. This provides strong evidence that the discriminator captures the reliability dimension of "likelihood to produce preference cycles."

## Highlights & Insights
- The paper transforms "LLM judge reliability" from an engineering heuristic into learnable parameters. While many evaluation pipelines manually select models or use simple majority voting, BT-$\sigma$ provides a probabilistic modeling alternative that requires no human labels.
- The explanation of hard BT vs. soft BT is insightful. It serves as a reminder that probabilistic outputs are not inherently better than binary preferences; when probabilities do not satisfy a global ranking structure, retaining intensity may mean retaining noise.
- The interpretability of $\sigma_k$ is well-addressed. Rather than just reporting aggregate scores, the authors verify the correlation between the discriminator and judge performance/cycle inconsistency, making the method a useful diagnostic tool.

## Limitations & Future Work
- BT-$\sigma$ is still predicated on the assumption of a global Bradley-Terry skill. If candidate outputs involve context-dependent, non-transitive human preferences or multi-modal preference groups, a single skill vector may be overly simplistic.
- The paper primarily focuses on offline benchmarks for NLG. In real-world open-ended evaluation, judge prompts, rubrics, candidate lengths, and safety constraints are more complex; whether $\sigma_k$ remains stable requires further testing.
- Temp-BT still holds an advantage in some dimensions, suggesting that supervised calibration remains valuable if high-quality annotations are available. Future work could explore semi-supervised combinations of limited labels and BT-$\sigma$.
- BT-$\sigma$ estimates judge-level reliability and does not directly address instance-level reliability. Some judges might only fail on specific sample types, requiring a more fine-grained conditional discriminator.

## Related Work & Insights
- **vs. Avg-Prob / majority voting**: Simple averaging treats all judges equally. Ours learns soft weights via $\sigma_k$ and enforces a global ranking structure.
- **vs. hard / soft Bradley-Terry**: Standard BT only learns item skills. Ours incorporates the judge's probability scale into the model, making the trustworthiness of soft probabilities variable.
- **vs. supervised temperature scaling**: Temp-BT requires human labels to fit temperatures. BT-$\sigma$ uses the pairwise comparison structure to learn the discriminator in a self-supervised manner, making it more suitable for reference-free evaluation.
- **vs. Crowd-BT / annotator aggregation**: Crowdsourcing models typically assume repeated labeling and latent ground truth. Ours is designed for LLM soft probability comparisons, directly handling ranking recovery in generative evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Embedding judge reliability into the Bradley-Terry soft comparison likelihood targets the right problem with a concise model.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple NLG datasets, judges, and evaluation dimensions, including reliability correlation analysis. Instance-level failure analysis could be strengthened.
- Writing Quality: ⭐⭐⭐⭐☆ Theoretical motivation, formulas, and experimental phenomena are naturally connected. The explanation of hard/soft BT is particularly clear.
- Value: ⭐⭐⭐⭐☆ Practical for automatic evaluation systems; can serve as a lightweight module for LLM-as-a-jury aggregation and judge diagnostics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoV-Eval: Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective](../../ACL2025/llm_evaluation/cov_eval_evaluating_llms_from_code_security_perspective.md)
- [\[ACL 2026\] Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation](../../ACL2026/llm_evaluation/teaching_language_models_to_forecast_research_success_through_comparative_idea_e.md)
- [\[ACL 2026\] SCAN: Structured Capability Assessment and Navigation for LLMs](../../ACL2026/llm_evaluation/scan_structured_capability_assessment_and_navigation_for_llms.md)
- [\[ACL 2025\] Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge?](../../ACL2025/llm_evaluation/can_external_validation_tools_improve_annotation_quality_for_llm-as-a-judge.md)
- [\[ACL 2026\] Zero-shot Large Language Models for Automatic Readability Assessment](../../ACL2026/llm_evaluation/zero-shot_large_language_models_for_automatic_readability_assessment.md)

</div>

<!-- RELATED:END -->
