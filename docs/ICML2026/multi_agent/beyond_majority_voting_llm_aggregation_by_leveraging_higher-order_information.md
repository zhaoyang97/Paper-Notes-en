---
title: >-
  [Paper Note] Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information
description: >-
  [ICML 2026][Multi-Agent][LLM aggregation] This paper proposes two algorithms for aggregating LLM responses by leveraging higher-order information—Optimal Weight (OW) based on first-order accuracy information and Inverse Surprising Popularity (ISP) based on second-order correlation information. These methods are provably superior to Majority Voting (MV) under label-free conditions and demonstrate consistent improvements on UltraFeedback, MMLU, and healthcare datasets.
tags:
  - "ICML 2026"
  - "Multi-Agent"
  - "LLM aggregation"
  - "multi-agent reasoning"
  - "information aggregation"
  - "Bayesian optimality"
  - "unsupervised labeling"
date: 2026-05-08
content_hash: 77e7f1f21dce4ca2
---

# Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information

**Conference**: ICML 2026  
**arXiv**: [2510.01499](https://arxiv.org/abs/2510.01499)  
**Code**: None  
**Area**: Multi-agent  
**Keywords**: LLM aggregation, multi-agent reasoning, information aggregation, Bayesian optimality, unsupervised labeling  

## TL;DR

This paper proposes two algorithms for aggregating LLM responses by leveraging higher-order information—Optimal Weight (OW) based on first-order accuracy information and Inverse Surprising Popularity (ISP) based on second-order correlation information. These methods are provably superior to Majority Voting (MV) under label-free conditions and demonstrate consistent improvements on UltraFeedback, MMLU, and healthcare datasets.

## Background & Motivation

**Background**: Multi-agent LLM reasoning (e.g., LLM debate, LLM council) has been widely adopted. In aggregating responses from multiple models, the vast majority of existing works use Majority Voting (MV) as the standard strategy.

**Limitations of Prior Work**: MV is a "zero-order" method that relies solely on the frequency of raw responses, completely ignoring the heterogeneity in competence among different LLMs (e.g., some models have 90% accuracy while others have 60%) and the correlations between responses. This means weak and strong models have equal weight, and results are easily misled when multiple weak models commit the same error.

**Key Challenge**: Utilizing model accuracy (first-order information) for weighting typically requires a large amount of labeled data, which is unavailable in unsupervised scenarios like auto-labeling or prediction markets. Meanwhile, the classic Surprisingly Popular (SP) method, which utilizes second-order correlation info without labels, actually performs worse than MV in LLM contexts—because LLMs lack the systematic biases found in human crowds, the signal utilized by SP yields counterproductive results.

**Goal**: Design aggregation algorithms that do not rely on labels, leverage higher-order information, and are theoretically provably superior to MV.

**Key Insight**: The authors observe that after randomly shuffling option orders, the joint distribution exhibits a symmetric structure (all incorrect options are equally probable), which allows for the derivation of closed-form optimal solutions. Furthermore, the failure of SP stems from its "bias amplification" direction being exactly opposite to what is needed for LLMs; reversing the calculation direction of SP corrects this.

**Core Idea**: Utilize the inverse sigmoid function of accuracy as the optimal weight to achieve Bayesian optimal aggregation; when labels are absent, reverse the direction of SP (calculating scores from a "counterfactual" perspective) to utilize second-order information and surpass MV.

## Method

### Overall Architecture

The paper addresses the problem of aggregating answers $a_1, \ldots, a_N$ from $N$ LLMs for a $K$-way multiple-choice question into a result more accurate than MV. The method begins with a preprocessing step—randomly shuffling option orders for each question so that all incorrect options appear with equal statistical probability, ensuring a symmetric joint distribution structure that enables closed-form optimal solutions. Aggregation follows three paths based on available information: if model accuracies are known, use OW for Bayesian optimal weighting; if no labels are available, use ISP which relies on inter-response correlations, or first estimate accuracies via ISP/fitting followed by OW (i.e., OW-I / OW-L). Finally, the selected answer is mapped back to the original option order.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input: N LLM responses a_1 … a_N for K-way questions"]
    A --> B["Preprocessing: Random option shuffling<br/>Ensures equal error prob & symmetric joint distribution"]
    B -->|Known Accuracy| OW["Optimal Weight (OW)<br/>Inv-Sigmoid Weighting ω_i = σ_K⁻¹(x_i)<br/>Bayesian Optimal"]
    B -->|No Labels · 2nd Order Correlation| ISP["Inverse Surprising Popularity (ISP)<br/>Reverse SP direction, score with counterfactual prob"]
    B -->|No Labels · Estimate Accuracy First| EST["OW-L / OW-I<br/>OW-L: Fit 2nd order prob to solve x_i<br/>OW-I: Use ISP results as pseudo-labels to estimate x_i"]
    EST -->|Substitute into σ_K⁻¹(x_i)| OW
    OW --> OUT["Inverse map to original options<br/>Output aggregated answer"]
    ISP --> OUT
```

### Key Designs

**1. Optimal Weight (OW): Bayesian Optimal Weighted Voting with Known Accuracy**

The primary flaw of MV is assigning equal voice to all models; a strong model (90% accuracy) is weighted the same as a weak one (60%), allowing a few weak models to bias the outcome. OW assigns each agent $i$ a weight $\omega_i = \sigma_K^{-1}(x_i)$, where $x_i$ is its accuracy and $\sigma_K(x) = e^x / (K-1+e^x)$ is the generalized sigmoid for $K$ classes. It then selects $f_{OW} = \arg\max_s \sum_i \sigma_K^{-1}(x_i) \cdot \mathbb{1}\{a_i = s\}$. The key is not just the intuition that "higher accuracy means higher weight," but that under the symmetric structure induced by shuffling, this set of inverse sigmoid weights is the necessary and sufficient solution for posterior probability maximization. It is the Bayesian optimum among all possible aggregators, not just a heuristic weighting scheme. Two corollaries follow: for $K=2$, it reduces to the inverse logistic, aligning with the Bradley-Terry model; for homogeneous agents (equal accuracy), it reduces to MV—indicating MV is only optimal for special cases like self-consistency (multiple samples from the same model).

**2. Inverse Surprising Popularity (ISP): Surpassing MV via Correlations Without Labels**

While OW is effective, it requires accuracy data, which is often missing in auto-labeling scenarios. The classic Surprisingly Popular (SP) method can utilize second-order correlation information $\mathbb{P}(A_i|A_j)$ without labels, but it underperforms MV on LLMs because SP assumes groups have a systematic bias toward underestimating the correct answer—a premise that does not hold for LLM ensembles. ISP fixes this by reversing the prediction direction: while SP calculates "predicting how others will answer," ISP calculates "predicting my own response given a counterfactual answer from others." The score is defined as $S_{ISP}(s,i) = \frac{1}{N-1}\sum_{j \neq i} \frac{1}{K-1}\sum_{a \neq a_j} \mathbb{P}(A_i=s|A_j=a)$, and the option with the largest advantage $Adv_{ISP}(s) = \sum_i \mathbb{1}\{a_i=s\} - \sum_i S_{ISP}(s,i)$ is chosen. Using counterfactual conditional probabilities biases the prediction toward incorrect options, which effectively amplifies the advantage of the correct answer. The paper establishes the chain inequality $\mathbb{E}[Adv_{ISP}(s^*)] \geq \mathbb{E}[Adv_{MV}(s^*)] \geq \mathbb{E}[Adv_{SP}(s^*)]$, explaining why ISP outperforms MV while MV outperforms the original SP.

**3. OW-L / OW-I: Bridging to the First-Order Framework via Accuracy Estimation**

ISP works label-free but only utilizes second-order information. The Bayesian optimal OW still requires accuracy. These two variants fill the gap when labels are missing: OW-L solves for accuracies $\hat{x}_1, \ldots, \hat{x}_N$ by minimizing the mean squared error between empirical and theoretical conditional probabilities; OW-I uses ISP results as pseudo-labels to estimate accuracy based on agent consistency. Both then substitute $\hat{x}_i$ into the weight formula $\sigma_K^{-1}(\hat{x}_i)$. This transforms "label-free second-order info" into "optimal first-order weights required by OW." Experiments show both variants perform similarly and outperform direct ISP application.

## Key Experimental Results

### Main Results (Simulated Data)

| Method | $K=2$ | $K=4$ | $K=6$ | $K=8$ | $K=10$ |
|------|-------|-------|-------|-------|--------|
| MV | 85.13% | 92.64% | 94.22% | 94.85% | 95.54% |
| SP | 79.94% | 90.52% | 92.68% | 93.66% | 94.40% |
| Single Best | 90.34% | 89.94% | 90.31% | 89.95% | 90.05% |
| **ISP (Ours)** | **90.48%** | **94.45%** | **95.78%** | **96.23%** | **96.49%** |
| OPT (clairvoyant) | 91.37% | 94.94% | 96.05% | 96.46% | 96.81% |

### Main Results (4 Strong Models on Real Datasets)

| Method | UltraFeedback | MMLU | ARMMAN |
|------|--------------|------|--------|
| MV | 72.21% | 89.32% | 85.24% |
| ISP | 73.26% | 90.01% | 85.78% |
| **OW-L** | **73.66%** | **90.37%** | **85.78%** |
| **OW-I** | **73.66%** | **90.37%** | **85.78%** |
| Single Best (oracle) | 73.14% | 91.02% | 85.32% |

### Key Findings

- ISP outperforms MV across all $K$ values, with the gap narrowing as $K$ increases ($\Theta(1/K)$), consistent with theoretical predictions.
- In 16 model combinations, OW-L outperformed MV in 97.92% of cases, with absolute gains up to 14.20%; MV was never the best performer.
- T-test results show statistical significance with t-statistics of 12.53 (UltraFeedback), 23.39 (MMLU), and 3.22 (ARMMAN), all p-values < 0.001.
- On "hard" subsets (MMLU-hard, where at least two models selected the same wrong option), OW-L/OW-I improved over MV by over 7% (17.23% → 24.79%), indicating higher value for higher-order information in difficult scenarios.

## Highlights & Insights

- **Bayesian Optimality of Inverse Sigmoid Weights**: The simple weighting scheme $\omega_i = \sigma_K^{-1}(x_i)$ is actually the optimal aggregator among all possibilities. This provides theoretical backing for using Bradley-Terry models in RLHF.
- **Counter-intuitive Design of Inverse SP**: While classic SP failed for LLMs, the authors diagnosed the cause (lack of human-like systematic bias) and reversed the direction to create ISP, a valuable problem-solving pattern.
- **Bridging Information Hierarchies**: OW-L/OW-I convert label-free second-order signals into first-order optimal weights, a strategy that can be adapted to other unsupervised aggregation tasks.

## Limitations & Future Work

- Theoretical analysis assumes conditional independence (LLMs are independent given the correct answer), which might be violated in practice, though experiments show robustness.
- All models use the same global weight for all questions, ignoring task-specific expertise; prompt-specific weighting is a future direction.
- Designed for closed-form multiple-choice questions ($K$ options); extension to open-ended generation is unclear.
- Theoretical assumptions regarding position bias (LLMs being unaffected by option order) may not hold for weaker models, necessitating further theoretical work.

## Related Work & Insights

- **Surprising Popularity (Prelec et al., 2017)**: Classic second-order aggregation method; this paper proves SP < MV for LLMs and offers ISP as a fix.
- **Bradley-Terry Model**: Corollary 3.2 links OW to BT models, supporting the effectiveness of BT in RLHF.
- **Self-Consistency (Wang et al., 2022)**: Corollary 3.3 proves MV is optimal for homogeneous agents, meaning complex aggregation is unnecessary for self-consistency scenarios.
- **Insight**: This framework is a drop-in replacement for MV in multi-LLM systems, with negligible computational overhead (seconds on CPU), suitable for API-based workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs](systematic_failures_in_collective_reasoning_under_distributed_information_in_mul.md)
- [\[ICML 2026\] Voting Protocols as Coordination Mechanisms for Role-Constrained Multi-Agent Tutoring Systems](voting_protocols_as_coordination_mechanisms_for_role-constrained_multi-agent_tut.md)
- [\[NeurIPS 2025\] Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning](../../NeurIPS2025/multi_agent/adaptive_coopetition_leveraging_coarse_verifier_signals_for_resilient_multi-agen.md)
- [\[ACL 2025\] Voting or Consensus? Decision-Making in Multi-Agent Debate](../../ACL2025/multi_agent/voting_or_consensus_decision-making_in_multi-agent_debate.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](../../AAAI2026/multi_agent/beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)

</div>

<!-- RELATED:END -->
