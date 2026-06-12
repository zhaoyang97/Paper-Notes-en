---
title: >-
  [Paper Note] Who can we trust? LLM-as-a-jury for Comparative Assessment
description: >-
  [ICML 2026][LLM Evaluation][LLM-as-a-jury] This paper identifies significant variance in reliability among multiple LLM judges during pairwise comparisons. It proposes the BT-σ model with judge-discriminator parameters t…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "LLM-as-a-jury"
  - "Bradley-Terry"
  - "Reviewer Reliability"
  - "Pairwise Comparison"
  - "Unsupervised Calibration"
date: 2026-05-08
content_hash: 49d054a3158825e3
---

# Who can we trust? LLM-as-a-jury for Comparative Assessment

**Conference**: ICML 2026  
**arXiv**: [2602.16610](https://arxiv.org/abs/2602.16610)  
**Code**: No public code  
**Area**: LLM Evaluation / Comparative Automated Evaluation  
**Keywords**: LLM-as-a-jury, Bradley-Terry, Reviewer Reliability, Pairwise Comparison, Unsupervised Calibration  

## TL;DR
This paper identifies significant variance in reliability among multiple LLM judges during pairwise comparisons. It proposes the BT-σ model with judge-discriminator parameters to simultaneously learn candidate output rankings and the reliability of each LLM judge without human calibration labels, achieving rankings closer to human judgment than simple averaging or standard Bradley-Terry aggregation.

## Background & Motivation
**Background**: LLM-as-a-judge has become a standard tool for evaluating NLG, summarization, dialogue responses, and open-ended generation. Compared to direct scoring, pairwise comparisons are generally more stable; thus, many systems task one or more LLM judges to determine if candidate output $i$ is superior to $j$, aggregating these results into a global ranking.

**Limitations of Prior Work**: The quality of multiple LLM judges is inconsistent. Some models exhibit verbosity bias, some are sensitive to candidate ordering, and others suffer from severe cyclic contradictions across different evaluation dimensions. Common probability or vote averaging methods assume all judges are equally reliable, weighting noisy and high-quality models equally, which causes final rankings to be degraded by inconsistency.

**Key Challenge**: Pairwise comparisons are expected to satisfy a global ordering structure, yet the preference probabilities provided by LLMs frequently violate transitivity, commutativity, and calibration consistency. Using soft probabilities directly retains more information but amplifies inconsistencies, while using only hard decisions is more robust but loses information regarding preference strength.

**Goal**: The authors aim to recover global skill rankings of candidate items and the reliability or discriminative power of each judge from the pairwise comparison probabilities of multiple LLM judges without relying on human-annotated calibration sets.

**Key Insight**: Starting from the Bradley-Terry model, the paper first analyzes when soft BT self-calibrates and when it fails due to probability inconsistency. It then treats "judge trustworthiness" as model parameters rather than manually assigning weights before aggregation.

**Core Idea**: A learnable discrimination scale $\sigma_k$ is added to each LLM judge, allowing reliable judges to be more sensitive to skill differences while naturally down-weighting noisy judges, forming an unsupervised reliability-aware BT aggregation.

## Method
The research follows a clear trajectory: placing LLM comparison probabilities into a Bradley-Terry framework to show that standard soft BT in multi-judge scenarios is equivalent to matching average probabilities; demonstrating that average probabilities fail to represent reliability differences; and finally proposing BT-σ to learn item skills and judge discriminators within a single likelihood.

### Overall Architecture
The input consists of a set of candidate generations and preference probabilities for all candidate pairs from multiple LLM judges. For each pair $(i,j)$ and judge $k$, the model observes $p_{ij}^{(k)}$, the probability that judge $k$ prefers $i$ over $j$. The output includes global ranking scores $s_i$ for candidates and reliability parameters $\sigma_k$ for judges. Evaluation is conducted via Spearman rank correlation (SRC) between candidate rankings and human score rankings.

The method first performs a symmetrization debias: if the same pair yields $p_{ij}$ and $p_{ji}$ under different orders, $p'_{ij}=\frac{1}{2}(p_{ij}+1-p_{ji})$ is used to enforce basic order consistency. Subsequently, hard BT, soft BT, Temp-BT, and BT-σ are compared based on these debiased results.

### Key Designs
1. **Explaining Differences between Hard BT and Soft BT via Probability Consistency**:
    - **Function**: Explains why hard BT sometimes outperforms soft BT and links this phenomenon to logical inconsistencies in LLM judges.
    - **Mechanism**: Standard BT assumes $P(i\succ j)=\sigma(s_i-s_j)$. Soft BT fits this structure using probability $p_{ij}$. If probabilities originate from a global skill vector, temperature scaling only scales skills without changing rankings. However, if LLM probabilities contain cycles or non-transitivity, soft BT must fit contradictory magnitudes, whereas hard BT remains noise-resistant by preserving only direction.
    - **Design Motivation**: This provides a diagnostic basis for reliability modeling: the issue lies not with the BT structure itself, but with the varying quality of signals from different judges, which cannot be treated with equal weight.

2. **BT-σ Judge-Specific Discriminator**:
    - **Function**: Learns the reliability of each judge without human labels and uses it to modulate their influence on global rankings.
    - **Mechanism**: BT-σ extends soft BT into $\mathcal{L}(\mathbf{s},\{\sigma_k\})\propto\prod_k\prod_{(i,j)}\sigma((s_i-s_j)/\sigma_k)^{p_{ij}^{(k)}}(1-\sigma((s_i-s_j)/\sigma_k))^{1-p_{ij}^{(k)}}$. A smaller $\sigma_k$ indicates a judge who is more sensitive and consistent regarding skill differences, while a larger $\sigma_k$ indicates flatter, noisier probabilities.
    - **Design Motivation**: This acts as unsupervised temperature calibration where the signal comes from the comparison structure itself rather than human labels, automatically weighting reliable judges more heavily during aggregation.

3. **Reliability Diagnosis and Aspect-Dependent Extensions**:
    - **Function**: Verifies that the learned $\sigma_k$ has interpretable reliability meaning rather than being a purely mathematical degree of freedom.
    - **Mechanism**: The paper analyzes the correlation between $1/\sigma_k$ and the judge’s standalone SRC, as well as $1-\text{CycleRate}$. It also proposes BT-σ-asp, learning separate discriminators for each judge-aspect pair to check if reliability varies by evaluation dimension.
    - **Design Motivation**: Strong correlations between $1/\sigma_k$ and independent performance/cyclic consistency would prove that the model captures actual reliability rather than merely overfitting a benchmark.

### Loss & Training
BT-σ directly maximizes the joint likelihood described above, with parameters including all item skills $\{s_i\}$ and judge discriminators $\{\sigma_k\}$. The authors use L-BFGS-B for optimization with random initialization for $s_i$ and $\sigma_k$, typically reaching convergence within 100 iterations. Temp-BT serves as a supervised reference requiring human labels to fit temperatures for each judge/aspect; BT-σ uses no human labels and relies solely on LLM pairwise probabilities.

## Key Experimental Results

### Main Results
The paper evaluates on SummEval, Topical-Chat, and NovelEval. The main tables report Spearman correlation for SummEval (coherence, consistency, fluency, relevance) and Topical-Chat (coherency, continuity, engagingness, naturalness).

| Dataset | Metric | Ours (BT-σ) | Prev. SOTA / Baselines | Gain |
|--------|------|------|----------|------|
| SummEval COH | SRC | 57.38 | soft BT 53.94 / Temp-BT 56.21 | +3.44 vs unsupervised soft BT |
| SummEval FLU | SRC | 42.99 | soft BT 42.69 / Temp-BT 41.88 | Marginal lead |
| SummEval REL | SRC | 54.15 | soft BT 53.11 / Temp-BT 55.14 | Better than soft BT, lower than supervised Temp-BT |
| Topical-Chat CNT | SRC | 56.30 | soft BT 53.87 / Temp-BT 52.21 | +2.43 vs soft BT |
| Topical-Chat NAT | SRC | 60.56 | soft BT 58.20 / Temp-BT 60.65 | Close to supervised calibration |
| SummEval ALL | SRC | 50.50 | soft BT 49.40 / Crowd-BT 48.35 | Overall lead |

### Ablation Study
Analysis focuses on whether learned discriminators represent reliability and if aspect-specific discriminators are necessary.

| Configuration | Key Metric | Description |
|------|---------|------|
| SummEval, $1/\sigma_k$ vs judge SRC | ALL PCC 72.21 / SRC 85.71 | Discriminator highly correlates with independent judge performance |
| Topical-Chat, $1/\sigma_k$ vs judge SRC | ALL PCC 67.41 / SRC 59.52 | Positive correlation maintained across tasks |
| SummEval, $1/\sigma_k$ vs $1-\text{CycleRate}$ | ALL PCC 90.29 / SRC 95.24 | Consistent judges learn higher $1/\sigma_k$ |
| BT-σ-asp vs BT-σ | Marginal gains on SummEval | Aspect-related reliability exists but yields limited gains |
| hard BT-σ on Topical-Chat ENG | SRC 67.36 | Hard decision + reliability modeling is robust in high-cycle-noise dimensions |

### Key Findings
- On individual LLM judges, hard BT often matches or exceeds soft BT, suggesting that raw probability magnitudes are not always trustworthy. After multi-judge aggregation, soft BT becomes stronger, suggesting that noise from different models partially cancels out.
- The advantage of BT-σ stems from explicitly modeling judge heterogeneity. Instead of simple averaging, it allows different temperature curves for judge probabilities within the likelihood, naturally attenuating unreliable models.
- $1/\sigma_k$ is extremely correlated with cyclic consistency, specifically reaching an ALL SRC of 95.24 on SummEval. This strongly suggests the discriminator captures the reliability dimension of "propensity for preference cycles."

## Highlights & Insights
- The paper transforms "LLM judge trustworthiness" from an engineering heuristic into a learnable parameter. While many pipelines use manual model selection or simple majority voting, BT-σ provides a probabilistic alternative requiring no human labels.
- The explanation of hard vs. soft BT is highly valuable. It serves as a reminder that probabilistic output is not inherently better than binary preferences; when probabilities do not satisfy a global ranking structure, retaining intensity may mean retaining noise.
- The interpretability of $\sigma_k$ is thoroughly examined. Rather than just reporting aggregate scores, the authors validate the discriminator against judge performance and cyclic inconsistency, making the method a viable diagnostic tool.

## Limitations & Future Work
- BT-σ still assumes a global Bradley-Terry skill structure. If context-dependent, non-transitive human preferences, or multi-modal preference populations exist, a single skill vector may be oversimplified.
- The paper primarily targets offline NLG benchmarks. In real-world open-ended evaluations, judge prompts, rubrics, length, and safety constraints are more complex; the stability of $\sigma_k$ requires further testing.
- Temp-BT still holds an advantage in certain dimensions, indicating that supervised calibration remains valuable if high-quality labels are available. Semi-supervised combinations of limited labels and BT-σ are a potential future direction.
- BT-σ estimates judge-level reliability and does not directly address instance-level reliability. A judge might fail only on specific sample types, requiring finer-grained conditional discriminators.

## Related Work & Insights
- **vs Avg-Prob / majority voting**: Simple averaging treats all judges equally; this work learns soft weights via $\sigma_k$ while enforcing a global ranking structure.
- **vs hard / soft Bradley-Terry**: Standard BT only learns item skills; this work incorporates judge probability scales into the model to allow variable credibility of soft probabilities.
- **vs supervised temperature scaling**: Temp-BT requires human labels for fitting; BT-σ uses the pairwise structure for self-supervised discriminator learning, making it more suitable for reference-free evaluation.
- **vs Crowd-BT / annotator aggregation**: Crowdsourcing models usually assume repeated labels and latent truth; this work targets LLM soft probability comparisons to handle ranking recovery in generative evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Precisely targets judge reliability within a Bradley-Terry soft comparison likelihood with a concise model.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple datasets, judges, and dimensions with strong reliability correlation analysis; instance-level failure analysis could be strengthened.
- Writing Quality: ⭐⭐⭐⭐☆ Natural flow between theoretical motivation, formulas, and experimental phenomena; the hard/soft BT explanation is particularly clear.
- Value: ⭐⭐⭐⭐☆ Highly practical for automated evaluation systems; serves as a lightweight module for LLM-as-a-jury aggregation and judge diagnostics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation](../../ACL2026/llm_evaluation/teaching_language_models_to_forecast_research_success_through_comparative_idea_e.md)
- [\[ACL 2026\] SCAN: Structured Capability Assessment and Navigation for LLMs](../../ACL2026/llm_evaluation/scan_structured_capability_assessment_and_navigation_for_llms.md)
- [\[ACL 2026\] Zero-shot Large Language Models for Automatic Readability Assessment](../../ACL2026/llm_evaluation/zero-shot_large_language_models_for_automatic_readability_assessment.md)
- [\[ACL 2026\] Can We Predict Before Executing Machine Learning Agents?](../../ACL2026/llm_evaluation/can_we_predict_before_executing_machine_learning_agents.md)
- [\[ICML 2026\] Resolution Diagnostics for Paired LLM Evaluation](resolution_diagnostics_for_paired_llm_evaluation.md)

</div>

<!-- RELATED:END -->
