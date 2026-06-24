---
title: >-
  [Paper Note] Theoretical Guarantees for Minimum Bayes Risk Decoding
description: >-
  [ACL 2025][Minimum Bayes Risk decoding] This paper provides the first rigorous theoretical convergence guarantees for Minimum Bayes Risk (MBR) decoding, proving that MBR decoding approximates the optimal solution at a rate of $O(n^{-1/2})$ when the size of the reference hypothesis set is $n$. It also theoretically compares MBR with MAP decoding, showing that MBR converges faster in various scenarios.
tags:
  - "ACL 2025"
  - "Minimum Bayes Risk decoding"
  - "MBR decoding"
  - "convergence guarantees"
  - "MAP decoding comparison"
  - "theoretical analysis"
date: 2026-05-08
content_hash: 045909e63614757e
---

# Theoretical Guarantees for Minimum Bayes Risk Decoding

**Conference**: ACL 2025  
**arXiv**: [2502.12685](https://arxiv.org/abs/2502.12685)  
**Code**: None  
**Area**: Others  
**Keywords**: Minimum Bayes Risk decoding, MBR decoding, convergence guarantees, MAP decoding comparison, theoretical analysis

## TL;DR

This paper provides the first rigorous theoretical convergence guarantees for Minimum Bayes Risk (MBR) decoding, proving that MBR decoding approximates the optimal solution at a rate of $O(n^{-1/2})$ when the size of the reference hypothesis set is $n$. It also theoretically compares MBR with MAP decoding, showing that MBR converges faster in various scenarios.

## Background & Motivation

**Background**: Decoding strategies for text generation models are mainly divided into two categories: (1) MAP (Maximum-A-Posteriori) decoding, which selects the output with the highest model probability (e.g., beam search); and (2) MBR (Minimum Bayes Risk) decoding, which selects the output that maximizes the expected utility (e.g., evaluation metrics like BLEU, COMET). Extensive empirical studies in tasks such as machine translation and text summarization have demonstrated that MBR decoding outperforms MAP decoding.

**Limitations of Prior Work**: The effectiveness of MBR decoding relies almost entirely on empirical results, lacking theoretical understanding. Specifically: (1) Why is approximating the expected utility over the entire language space with a finite candidate set still effective? (2) What is the mathematical relationship between the candidate set size $n$ and the approximation quality? (3) What theoretical advantages does MBR have over MAP? These questions have previously lacked rigorous mathematical answers.

**Key Challenge**: MBR decoding requires computing the expected utility of candidate outputs relative to the reference set. However, the complete language space $\mathcal{Y}$ is exponential or even infinite ($|\mathcal{Y}| \gg n$), whereas in practice, it can only be approximated using a finite set of $n$ reference hypotheses. Why this coarse approximation remains effective in practice requires a theoretical explanation.

**Goal**: (1) Deriving the convergence rate bounds for MBR decoding; (2) deriving the performance gap bounds for MAP decoding; (3) theoretically comparing their convergence characteristics.

**Key Insight**: From the perspective of probability theory and statistical learning theory, the authors formalize MBR decoding as a statistical estimation problem based on finite samples—approximating the expected utility under the true distribution using a finite reference set.

**Core Idea**: Under appropriate assumptions (bounded utility function, distribution satisfying certain regularity conditions), the utility difference between the output selected by MBR decoding and the true optimal solution does not exceed $O(n^{-1/2})$ with high probability. This convergence rate explains why a moderately sized reference set is sufficient to yield high-quality outputs.

## Method

### Overall Architecture

This paper is a theoretical analysis work, its core contribution being mathematical theorems rather than algorithm design. The analytical framework is as follows: (1) formalizing the objective functions of MBR and MAP decoding; (2) deriving the upper bound of the approximation error for MBR; (3) deriving the performance gap for MAP; (4) comparing their convergence behaviors.

### Key Designs

1. **Formalization and Convergence Theorem of MBR Decoding**:

    - **Function**: Providing quantitative guarantees for the approximation quality of MBR decoding.
    - **Mechanism**: The goal of MBR decoding is to select the output that maximizes the expected utility $y^* = \arg\max_{y \in \mathcal{C}} \mathbb{E}_{r \sim p}[u(y, r)]$, where $\mathcal{C}$ is the candidate set, $p$ is the reference distribution, and $u$ is the utility function (e.g., BLEU/COMET). In practical computation, $n$ sampled references are used to replace the expectation: $\hat{y} = \arg\max_{y \in \mathcal{C}} \frac{1}{n} \sum_{i=1}^{n} u(y, r_i)$. The theorem states that under the assumption of a bounded utility function ($u \in [0, 1]$), $\mathbb{E}[u(y^*, r)] - \mathbb{E}[u(\hat{y}, r)] \leq O(\sqrt{\log|\mathcal{C}|/n})$, meaning the approximation error converges to zero at a rate of $O(n^{-1/2})$.
    - **Design Motivation**: This convergence rate directly explains observations from prior empirical studies—typically a reference set of $n = 100 \sim 256$ is sufficient to obtain near-optimal MBR outputs, as $O(n^{-1/2})$ implies that increasing $n$ from 100 to 1000 only brings about a 3-fold improvement (diminishing marginal utility).

2. **Performance Gap Analysis of MAP Decoding**:

    - **Function**: Deriving the bounds of the deviation from the optimal solution for MAP decoding under a finite candidate set.
    - **Mechanism**: MAP decoding selects the output with the highest probability $\hat{y}_{MAP} = \arg\max_{y \in \mathcal{C}} p(y)$, and its performance gap depends on whether the candidate set contains high-probability outputs. The paper derives the gap bounds for MAP and reveals that it depends on the "sharpness" of the distribution—if the distribution is highly concentrated on a few outputs, MAP has a high hit rate in the candidate set and a small gap; however, if the distribution is flat (multiple outputs have similar probabilities), the "highest probability" output chosen by MAP may not be the optimal utility output.
    - **Design Motivation**: Laying the foundation for subsequent theoretical comparison between MBR and MAP.

3. **Theoretical Comparison of MBR vs MAP**:

    - **Function**: Explaining why MBR generally outperforms MAP from the perspective of convergence rates.
    - **Mechanism**: The approximation quality of MAP heavily depends on the shape of the distribution—performing well under sharp distributions (e.g., highly deterministic translation tasks) but converging slowly under flat distributions (e.g., creative generation, open-ended QA). The $O(n^{-1/2})$ convergence of MBR does not depend on the distribution shape, making it more robust to flat distributions. The paper further proves that in the extreme case of a "quasi-uniform distribution", MAP requires exponentially more candidate samples to approximate the optimal, while MBR only requires polynomial growth.
    - **Design Motivation**: Providing a clear theoretical framework to understand "when to use MBR and when to use MAP"—the flatter the distribution, the greater the advantage of MBR.

### Loss & Training

Does not involve model training.

## Key Experimental Results

### Main Results

Empirical verification of MBR vs MAP on machine translation tasks (WMT datasets):

| Language Pair | Metric | MAP (beam=5) | MBR (n=50) | MBR (n=256) | Gain |
|--------|------|-------------|-----------|------------|------|
| En→De | COMET | 84.2 | 85.8 | 86.3 | +2.1 |
| En→Zh | COMET | 82.7 | 84.5 | 85.1 | +2.4 |
| De→En | COMET | 85.1 | 86.2 | 86.6 | +1.5 |
| En→Ja | COMET | 81.3 | 83.4 | 84.0 | +2.7 |

### Ablation Study

The impact of reference set size $n$ on MBR performance (verifying the theoretical convergence rate):

| Reference Set Size n | COMET Score | Gap with n=1024 | Theoretically Predicted Gap |
|------------|----------|----------------|------------|
| 16 | 84.5 | -1.8 | ~$O(0.25)$ |
| 64 | 85.6 | -0.7 | ~$O(0.125)$ |
| 256 | 86.1 | -0.2 | ~$O(0.0625)$ |
| 1024 | 86.3 | 0 | Baseline |

### Key Findings

- **High consistency between theory and empirical results**: The trend of the actually observed MBR performance gains with the growth of $n$ matches well with the $O(n^{-1/2})$ theoretical prediction, validating the effectiveness of the theoretical analysis.
- **MBR shows the greatest advantage under flat distributions**: On language pairs with high translation diversity (such as En→Ja), the gain of MBR relative to MAP is more significant (+2.7), which is consistent with theoretical predictions.
- **Diminishing marginal returns are evident**: Increasing from n=256 to n=1024 only yields a 0.2-point improvement, indicating that n=256 is already a cost-effective choice in practice.

## Highlights & Insights

- **First MBR convergence theorem**: After MBR decoding has been widely used for many years, this work provides the first mathematical proof of its effectiveness. The $O(n^{-1/2})$ rate is clean and elegant, providing theoretical guidance for hyperparameter selection (reference set size) in practice.
- **Unified comparative framework**: By comparing MBR and MAP under the same theoretical framework, the work clearly reveals the differences in their applicable scenarios—an insight far more meaningful than merely comparing BLEU scores on specific tasks.
- **Practical value of theoretical analysis**: Directly answering the question that practitioners care about—"How large of a reference set is sufficient?" The answer is $n = O(\log|\mathcal{C}| / \epsilon^2)$, representing a logarithmic candidate set size and a reference set inversely proportional to the squared error.

## Limitations & Future Work

- **Limitations of assumptions**: The assumption of a bounded utility function holds in most scenarios (e.g., BLEU $\in [0, 1]$), but some length-dependent metrics may not satisfy it; the distribution regularity assumption may fail under degenerate distributions.
- **Simplification of shared candidate and reference sets**: The theoretical analysis assumes that the candidate set and reference set are independent or identical. However, in practice, both usually originate from the same set of samples, and this dependency is not covered by the theory.
- **Limited to the utility evaluation perspective**: The work does not analyze the theoretical characteristics of MBR decoding in other dimensions such as diversity and fairness.
- **Small experimental scale**: The verification experiments are limited to machine translation and have not validated theoretical predictions on tasks like summarization or dialogue generation.

## Related Work & Insights

- **vs Eikema & Aziz (2020)**: Early MBR analysis work, mainly focusing on the impact of sampling strategies without providing formal bounds on convergence rates. The contribution of this paper is to elevate it to the level of rigorous mathematical guarantees.
- **vs Müller & Sennrich (2021)**: They observed that MBR outperforms beam search but did not explain why. This paper provides an explanation through theoretical results showing that MAP converges slowly under flat distributions.
- This paper provides a solid theoretical foundation for the research direction of "improving generation quality with better decoding strategies." Future work can build on this to analyze more complex decoding strategies (such as best-of-N, rejection sampling).

## Rating

- Novelty: ⭐⭐⭐⭐ First rigorous MBR convergence theorem, filling a theoretical gap
- Experimental Thoroughness: ⭐⭐⭐ Rigorous theory but limited scope of experimental validation
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations, intuitive presentation of conclusions
- Value: ⭐⭐⭐⭐ Provides a theoretical foundation for the widely used MBR decoding, with guiding significance for practice

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Decoding Reading Goals from Eye Movements](decoding_reading_goals_from_eye_movements.md)
- [\[CVPR 2026\] Spectral Conformal Risk Control: Distribution-Free Tail Guarantees via Bayesian Quadrature](../../CVPR2026/others/spectral_conformal_risk_control_distribution-free_tail_guarantees_via_bayesian_q.md)
- [\[ACL 2025\] EpiCoDe: Boosting Model Performance Beyond Training with Extrapolation and Contrastive Decoding](epicode_boosting_model_performance_beyond_training_with_extrapolation_and_contra.md)
- [\[ACL 2025\] CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](coachme_sport_instruction.md)
- [\[ACL 2025\] Well Begun is Half Done: Low-resource Preference Alignment by Weak-to-Strong Decoding](well_begun_is_half_done_low-resource_preference_alignment_by_weak-to-strong_deco.md)

</div>

<!-- RELATED:END -->
