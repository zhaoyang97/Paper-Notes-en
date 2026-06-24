---
title: >-
  [Paper Note] A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI
description: >-
  [ACL 2025][AIGC Detection][Human Judgment Distribution] This paper proposes using LLM-generated NLI explanations to substitute expensive human explanations for approximating Human Judgment Distributions (HJD). Experiments demonstrate that with the guidance of human label distributions, LLM-generated explanations achieve comparable performance to human explanations across metrics like KL divergence and JSD. Furthermore, the approach generalizes well to datasets without human e…
tags:
  - "ACL 2025"
  - "AIGC Detection"
  - "Human Judgment Distribution"
  - "NLI"
  - "LLM Explanation Generation"
  - "Label Variation"
  - "Annotation Disagreement"
date: 2026-05-08
content_hash: 8da30f2fa37b217d
---

# A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI

**Conference**: ACL 2025  
**arXiv**: [2412.13942](https://arxiv.org/abs/2412.13942)  
**Code**: [https://github.com/mainlp/MJD-Estimator](https://github.com/mainlp/MJD-Estimator)  
**Area**: Annotation Disagreement / Human Judgment Distribution  
**Keywords**: Human Judgment Distribution, NLI, LLM Explanation Generation, Label Variation, Annotation Disagreement

## TL;DR

This paper proposes using LLM-generated NLI explanations to substitute expensive human explanations for approximating Human Judgment Distributions (HJD). Experiments demonstrate that with the guidance of human label distributions, LLM-generated explanations achieve comparable performance to human explanations across metrics like KL divergence and JSD. Furthermore, the approach generalizes well to datasets without human explanations (MNLI) and out-of-domain test sets (ANLI).

## Background & Motivation

**Background**: Annotation disagreements among humans in NLI represent genuine semantic ambiguity rather than noise—different annotators can reasonably give different judgments (entailment/neutral/contradiction) for the same premise-hypothesis pair. The Human Judgment Distribution (HJD) captures this distribution information through a large number of annotators per instance (e.g., 100 annotators per instance in ChaosNLI), which is valuable for training NLI classifiers.

**Limitations of Prior Work**: Chen et al. (2024) demonstrated that LLMs can effectively approximate HJDs (termed MJD Estimator) using a few human labels + human explanations. However, collecting human explanations is far more expensive than collecting labels alone—each explanation requires annotators to elaborate on their reasoning process, and most NLI datasets do not contain explanation annotations at all.

**Key Challenge**: The MJD Estimator relies on human explanations to work effectively, but human explanations are the most expensive bottleneck in the overall pipeline.

**Goal**: Can LLM-generated explanations be of sufficient quality to replace human explanations for approximating HJDs?

**Key Insight**: Prompt the LLM to exhaustively generate all possible explanation reasons for each NLI label, then select a corresponding number of explanations based on the human label distribution to construct label-explanation pairs as inputs for the MJD Estimator.

**Core Idea**: The source of explanations (human vs. LLM) does not matter; what matters is the presence of the explanations—"A rose by any other name would smell as sweet".

## Method

### Overall Architecture

Three steps: (1) LLMs generate multiple explanations for each NLI label (entailment, neutral, contradiction) of a given premise-hypothesis pair. (2) Selection strategies (either label-free or label-guided) choose an appropriate subset of explanations. (3) The selected label-explanation pairs are fed into the MJD Estimator (utilizing first-token probabilities) to output Model Judgment Distributions (MJD), which are then compared with the ground truth HJD.

### Key Designs

1. **Model Explanation Generation**:

    - Function: Prompts the LLM (Llama3 / GPT-4o) to list all possible explanation reasons for each label (entailment/neutral/contradiction) of a given premise-hypothesis pair.
    - Mechanism: Label variation in NLI implies there can be multiple plausible reasons for the same label (e.g., multiple annotators might choose "entailment" but for different reasons). Consequently, the LLM is requested to exhaustively list multiple justifications instead of generating only a single one.
    - Design Motivation: Exhaustive generation provides a sufficient candidate pool for subsequent selection strategies.

2. **Explanation Selection Strategies**:

    - Function: Selects a suitable subset from the candidate explanation pool to serve as input for the MJD Estimator.
    - Mechanism:
        - **Label-Free Selection**: Selects 1 explanation per label uniformly, totaling 3 explanations, without using any human labels (acting as a baseline).
        - **Label-Guided Selection**: Selects the corresponding number of explanations matching the human label distribution. For example, if 3 out of 5 annotators choose entailment, 1 choosing neutral, and 1 choosing contradiction, then 3 entailment explanations + 1 neutral explanation + 1 contradiction explanation are selected.
    - Selection Mode: First (selecting the first $k$ explanations generated by the LLM) vs. Longest (selecting the longest $k$ explanations).
    - Design Motivation: Since label information is low-cost and readily available in most datasets, using labels to guide explanation selection can compensate for the absence of human explanations.

3. **MJD Estimation and Evaluation**:

    - Function: Feeds the selected label-explanation pairs into an LLM using MCQA prompts, and extracts the MJD based on first-token probabilities.
    - Mechanism: Permutation averaging is employed over label order, explanation order, and combinations to eliminate positional bias (A-bias, length bias, and sequential bias).
    - Evaluation Metrics: KL divergence, JSD, and TVD to measure distributional distances; downstream classification performance on ANLI is also evaluated using a BERT/RoBERTa model trained on the resulting MJD.

## Key Experimental Results

### Main Results (Llama3, VariErr Label-Guided, vs Human Explanations)

| Explanation Source | KL ↓ | JSD ↓ | TVD ↓ | D.Corr ↑ |
|---------|------|-------|-------|----------|
| No explanation (Llama3 only) | 0.259 | 0.262 | 0.284 | 0.689 |
| + Human explanations | 0.238 | 0.250 | 0.269 | 0.771 |
| + Model explanations (Label-Free) | 0.295 | 0.278 | 0.310 | 0.744 |
| + Model explanations (VariErr-Guided) | 0.234 | 0.247 | 0.266 | 0.760 |
| + Model explanations (MNLI-Guided) | 0.242 | 0.251 | 0.275 | **0.849** |

### Downstream Tasks (RoBERTa Performance on ANLI)

| Training Distribution Source | CE Loss ↓ | Weighted F1 ↑ |
|-------------|-----------|---------------|
| ChaosNLI HJD | 0.922 | 0.653 |
| Llama3 + Human explanations MJD | 1.019 | 0.616 |
| Llama3 + Model explanations (MNLI-Guided) MJD | 1.018 | **0.645** |

### Key Findings
- Label-guided model explanations and human explanations achieve highly comparable quality in HJD approximation; under MNLI guidance, the D.Corr metric even outperforms human explanations (0.849 vs. 0.771).
- Self-supervised label-free selection underperforms as a baseline but still outperforms the completely explanation-free baseline.
- The performance of "First" and "Longest" selection modes is similar, despite only showing an 18.9% overlap, indicating the robustness of the results to selection strategies.
- Classifiers trained on MJD outperform those trained on hard single labels when evaluated on the out-of-domain ANLI test set, proving the downstream utility of modeling human label variation.

## Highlights & Insights
- **The core insight that "names do not matter"** — the source of the explanations (human vs. LLM) is less critical than the sheer presence of the explanations. This implies that most NLI datasets can obtain closely approximated HJDs at zero additional human annotation cost. This finding has potential generalization value for all tasks featuring considerable annotation disagreement.
- **A practical path to dramatic cost reduction** — reducing the requirements from 100 annotators/instance (ChaosNLI) to only 4-5 labels + LLM-generated explanations. This dramatic reduction makes the HJD approach feasible for real-world deployment.
- **Effective out-of-domain generalization** — the approach is not only effective on datasets that natively possess human explanations but also generalizes successfully to datasets without explanations (MNLI) and out-of-domain environments (ANLI), showing high practicality.

## Limitations & Future Work
- The proposed method is verified only on NLI tasks. Other tasks with prominent label variation (e.g., sentiment analysis, toxicity detection) remain untested.
- A small number of human labels (4-5 per instance) are still required to guide the selection; the completely label-free mode underperforms.
- The explanation selection heuristics ("First" / "Longest") are quite simplistic; semantic-similarity-based selection strategies could be explored in future work.
- The ground truth HJD relies heavily on ChaosNLI (annotated by 100 humans); the quality and representativeness of this ground truth itself are not questioned.
- Only two models, Llama3 and GPT-4o, were explored as explanation generators.

## Related Work & Insights
- **vs. Chen et al. (2024)**: That work requires human explanations for MJD estimation. This paper demonstrates that LLM-generated explanations are viable, comparable substitutes.
- **vs. Lee et al. / Madaan et al.**: Direct prompting for distribution prediction in LLMs yields inconsistent improvements. This paper's explanation-guided approach is far more robust.
- **vs. Pavlovic & Poesio (2024)**: Also attempted LLM-based HJD approximation, but produced mixed results. This paper achieves superior consistency through the explanation mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐ Replacing human explanations with LLM-generated ones to approximate HJD is an elegant, novel, and highly impactful finding.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple scenarios and models (ChaosNLI, MNLI, VariErr, and ANLI), using permutation averaging to eliminate biases.
- Writing Quality: ⭐⭐⭐⭐ The metaphorical title is fitting, the methodology is clearly articulated, and the experiments are logically organized.
- Value: ⭐⭐⭐⭐ Provides direct, practical value for optimizing annotation costs and modeling human label variation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Scoring to Explanations: Evaluating SHAP and LLM Rationales for Rubric-based Teaching Quality Assessment](../../ACL2026/aigc_detection/from_scoring_to_explanations_evaluating_shap_and_llm_rationales_for_rubric-based.md)
- [\[ACL 2025\] Comparing LLM-generated and human-authored news text using formal syntactic theory](llm_vs_human_formal_syntax.md)
- [\[ACL 2025\] HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring](haco-det_a_study_towards_fine-grained_machine-generated_text_detection_under_hum.md)
- [\[ACL 2025\] Learning to Rewrite: Generalized LLM-Generated Text Detection](learning_to_rewrite_generalized_llm-generated_text_detection.md)
- [\[ACL 2025\] Low-Perplexity LLM-Generated Sequences and Where To Find Them](low-perplexity_llm-generated_sequences_and_where_to_find_them.md)

</div>

<!-- RELATED:END -->
