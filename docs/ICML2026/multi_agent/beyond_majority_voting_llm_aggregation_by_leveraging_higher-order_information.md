---
title: >-
  [Paper Note] Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information
description: >-
  [ICML 2026][Multi-Agent][Paper Note] This paper proposes two LLM answer aggregation algorithms that leverage higher-order information: Optimal Weight (OW), based on first-order accuracy information, and Inverse Surprising Popularity (ISP), based on second-order correlation information. The methods are proven to outperform Majority Voting (MV) without requ
tags:
  - ICML 2026
  - Multi-Agent
date: 2026-05-08
content_hash: 7d89ff6163d9a319
---
# Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information

**Conference**: ICML 2026  
**arXiv**: [2510.01499](https://arxiv.org/abs/2510.01499)  
**Code**: None  
**Area**: Multi-Agent  
**Keywords**: LLM aggregation, multi-agent reasoning, information aggregation, Bayesian optimality, unsupervised labeling  

## TL;DR

This paper proposes two LLM answer aggregation algorithms that leverage higher-order information: Optimal Weight (OW), based on first-order accuracy information, and Inverse Surprising Popularity (ISP), based on second-order correlation information. The methods are proven to outperform Majority Voting (MV) without requiring ground-truth labels, with consistent improvements validated on UltraFeedback, MMLU, and healthcare datasets.

## Background & Motivation

**Background**: Multi-agent LLM reasoning (e.g., LLM debate, LLM council) has been widely adopted. In aggregating answers from multiple models, the vast majority of existing works directly employ Majority Voting (MV) as the standard aggregation strategy.

**Limitations of Prior Work**: MV is a "zero-order" method that relies solely on the frequency of raw answers, completely ignoring the heterogeneity in capabilities among different LLMs (e.g., one model having 90% accuracy while another has 60%) and the correlations between answers. This implies that a weak model and a strong model hold equal weight in voting, and the final result is easily misled when multiple weak models commit the same error.

**Key Challenge**: Utilizing model accuracy (first-order information) for weighting typically requires a large amount of labeled data for estimation, which is unavailable in unsupervised scenarios such as auto-labeling or prediction markets. Conversely, the classic Surprisingly Popular (SP) method utilizes second-order correlation information without labels but performs worse than MV in LLM contexts. This is because LLMs do not exhibit the same systematic biases as human crowds, causing the signals utilized by SP to produce counterproductive effects.

**Goal**: Design aggregation algorithms that do not rely on labels, leverage higher-order information, and are theoretically provable to outperform MV.

**Key Insight**: The authors observe that by randomly shuffling the order of options, the joint distribution exhibits a symmetric structure (where all incorrect options are equally probable). This structure allows for the derivation of closed-form optimal solutions. Furthermore, the failure of SP stems from its "bias amplification" direction being exactly opposite to what is required in LLM scenarios; inverting the calculation direction of SP corrects this.

**Core Idea**: Utilize the inverse sigmoid function of accuracy as the optimal weight to achieve Bayesian optimal aggregation. In the absence of labels, invert the prediction direction of SP (calculating scores from a "counterfactual" perspective) to utilize second-order information to surpass MV.

## Method

### Overall Architecture

This paper addresses the problem of aggregating answers $a_1, \ldots, a_N$ from $N$ LLMs for a $K$-option multiple-choice question into a single answer that is more accurate than Majority Voting. The method begins with a preprocessing step: randomly shuffling the option order for each question. This ensures that all incorrect options are statistically equally likely, creating a symmetric structure in the joint distribution of answers, which allows subsequent optimal solutions to be expressed in closed form. Aggregation is then performed based on the available information: if model accuracies are known, Optimal Weight (OW) is used for Bayesian optimal weighting; if no labels are available, Inverse Surprising Popularity (ISP) is used by relying on inter-model correlations, or accuracies are first estimated via ISP/fitting to revert to the OW framework (referred to as OW-I / OW-L). Finally, the selected answer is mapped back to the original option order via an inverse permutation.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input: Answers a_1 … a_N from N LLMs for a K-option question"]
    A --> B["Preprocessing: Randomly shuffle options<br/>Ensures equal error probability and symmetric joint distribution"]
    B -->|Known Accuracy| OW["Optimal Weight (OW)<br/>Inverse sigmoid weighting ω_i = σ_K⁻¹(x_i)<br/>Bayesian optimal"]
    B -->|No Labels · Second-order Correlation| ISP["Inverse Surprising Popularity (ISP)<br/>Inverts SP direction, scoring via counterfactual conditional probability"]
    B -->|No Labels · Estimate Accuracy First| EST["OW-L / OW-I<br/>OW-L solves x_i via second-order probability fitting<br/>OW-I estimates x_i using ISP results as pseudo-labels"]
    EST -->|Substitute into σ_K⁻¹(x_i)| OW
    OW --> OUT["Inverse permutation mapping to original options<br/>Output aggregated answer"]
    ISP --> OUT
```

### Key Designs

**1. Optimal Weight (OW): Achieving Bayesian Optimality in Weighted Voting with Known Accuracies**

The primary flaw of Majority Voting is assigning equal weight to all models, where a strong model with 90% accuracy is equivalent to a weak one with 60%. OW assigns each agent $i$ a weight $\omega_i = \sigma_K^{-1}(x_i)$, where $x_i$ is its accuracy and $\sigma_K(x) = e^x / (K-1+e^x)$ is the generalized sigmoid for $K$ classes. It then selects the option with the highest weighted sum: $f_{OW} = \arg\max_s \sum_i \sigma_K^{-1}(x_i) \cdot \mathbb{1}\{a_i = s\}$. The significance is not merely the intuition that higher accuracy yields higher weight, but that under the symmetric information structure induced by shuffling, these inverse sigmoid weights are the necessary and sufficient solution for maximizing posterior probability, representing the Bayesian optimum among all possible aggregators. Two corollaries follow: for $K=2$, it reduces to the inverse logistic, aligning with the Bradley-Terry model; when agents are homogeneous (identical accuracy), it reduces to MV, indicating that MV is only optimal in special cases like self-consistency.

**2. Inverse Surprising Popularity (ISP): Surpassing MV via Correlations Without Labels**

While OW is effective, it requires knowledge of accuracies, which are unavailable in unsupervised scenarios. The classic Surprisingly Popular (SP) method utilizes second-order correlation information $\mathbb{P}(A_i|A_j)$ without labels, yet it performs worse than MV for LLMs because SP assumes the crowd systematically underestimates the correct answer—a premise that does not hold for LLM groups. ISP modifies this by inverting the SP prediction direction: while SP calculates "what an agent predicts others will answer," ISP calculates the prediction "as if other agents provided a counterfactual answer." The score is defined as $S_{ISP}(s,i) = \frac{1}{N-1}\sum_{j \neq i} \frac{1}{K-1}\sum_{a \neq a_j} \mathbb{P}(A_i=s|A_j=a)$, and the option maximizing the advantage function $Adv_{ISP}(s) = \sum_i \mathbb{1}\{a_i=s\} - \sum_i S_{ISP}(s,i)$ is chosen. Using counterfactual conditional probabilities biases the prediction score toward incorrect options, thereby magnifying the advantage of the correct answer. The paper establishes the chain inequality $\mathbb{E}[Adv_{ISP}(s^*)] \geq \mathbb{E}[Adv_{MV}(s^*)] \geq \mathbb{E}[Adv_{SP}(s^*)]$, explaining why ISP outperforms MV while MV outperforms the original SP.

**3. OW-L / OW-I: Bridging to the First-Order Optimal Framework by Estimating Accuracies**

ISP works without labels but only utilizes second-order information, whereas the Bayesian optimal OW requires accuracy. These two variants estimate the missing accuracies to revert to OW: OW-L solves for agent accuracies $\hat{x}_1, \ldots, \hat{x}_N$ by minimizing the mean squared error between empirical and theoretical conditional probabilities; OW-I uses the ISP aggregation results as pseudo-labels to calculate the consistency rate of each agent as an accuracy estimate. Once $\hat{x}_i$ is estimated, it is substituted back into the weight formula $\sigma_K^{-1}(\hat{x}_i)$. This translates "unsupervised second-order information" into the "first-order optimal weights required for OW." Experiments show both variants perform similarly on real data and exceed the performance of direct ISP usage.

## Key Experimental Results

### Main Results (Simulated Data)

| Method | $K=2$ | $K=4$ | $K=6$ | $K=8$ | $K=10$ |
|------|-------|-------|-------|-------|--------|
| MV | 85.13% | 92.64% | 94.22% | 94.85% | 95.54% |
| SP | 79.94% | 90.52% | 92.68% | 93.66% | 94.40% |
| Single Best | 90.34% | 89.94% | 90.31% | 89.95% | 90.05% |
| **ISP (Ours)** | **90.48%** | **94.45%** | **95.78%** | **96.23%** | **96.49%** |
| OPT (clairvoyant) | 91.37% | 94.94% | 96.05% | 96.46% | 96.81% |

### Main Results (Real Datasets with 4 Strong Models)

| Method | UltraFeedback | MMLU | ARMMAN |
|------|--------------|------|--------|
| MV | 72.21% | 89.32% | 85.24% |
| ISP | 73.26% | 90.01% | 85.78% |
| **OW-L** | **73.66%** | **90.37%** | **85.78%** |
| **OW-I** | **73.66%** | **90.37%** | **85.78%** |
| Single Best (oracle) | 73.14% | 91.02% | 85.32% |

### Key Findings

- ISP outperforms MV across all values of $K$, with the performance gap narrowing as $K$ increases ($\Theta(1/K)$), consistent with theoretical predictions.
- Across 16 model combinations, OW-L outperforms MV in 97.92% of cases, with absolute improvements of up to 14.20%; MV never achieved the best performance in any combination.
- Hypothesis testing yields t-statistics of 12.53 (UltraFeedback), 23.39 (MMLU), and 3.22 (ARMMAN), with p-values all < 0.001, indicating statistically significant improvements.
- On "hard" subsets (MMLU-hard, where at least two models selected the same incorrect option), OW-L/OW-I improved over MV by more than 7% (17.23% → 24.79%), demonstrating the higher value of higher-order information in difficult scenarios.

## Highlights & Insights

- **Bayesian Optimality of Inverse Sigmoid Weighting**: The seemingly simple weighting scheme $\omega_i = \sigma_K^{-1}(x_i)$ is actually the optimal solution among all possible aggregators. This provide theoretical endorsement for the use of Bradley-Terry models in RLHF. The conclusion is elegant and directly applicable.
- **Counter-intuitive Design of Inverting SP**: While classic SP is effective for human crowds but fails for LLMs, the authors analyze the underlying cause (lack of systematic human-like bias in LLMs) and invert the direction to derive ISP. This "diagnose failure → targeted modification" approach is highly instructive.
- **Bridging Information Levels**: OW-L/OW-I convert "unsupervised second-order information" into "first-order optimal weights," a strategy that can be migrated to other unsupervised aggregation scenarios.

## Limitations & Future Work

- Theoretical analysis relies on the conditional independence assumption (LLMs are independent given the correct answer). While experiments show effectiveness even when this is violated, a formal robustness bound is lacking.
- The use of global weights for all questions does not account for variance in model capabilities across different domains (e.g., a model might excel at math but struggle with linguistics). Prompt-specific weights are a clear direction for improvement.
- The framework currently handles closed multiple-choice questions ($K$ options); extension to open-ended generation tasks remains unclear.
- Theoretical assumptions regarding position bias (LLMs being unaffected by option order) do not fully hold for weaker models. Although no explicit debiasing was required in experiments, theoretical guarantees for weak models need strengthening.

## Related Work & Insights

- **Surprising Popularity (Prelec et al., 2017)**: A classical second-order aggregation method; however, the authors prove SP is inferior to MV in LLM contexts, positioning ISP as a targeted improvement.
- **Bradley-Terry Model**: Corollary 3.2 establishes the link between OW and BT models, providing theoretical support for their effectiveness in RLHF.
- **Self-Consistency (Wang et al., 2022)**: Corollary 3.3 proves that MV is optimal when agents are homogeneous, implying no complex aggregation is needed for self-consistency scenarios.
- **Insights**: This framework can serve as a direct drop-in replacement for MV in multi-LLM systems with minimal computational overhead (mere seconds on a CPU), making it suitable for API-based applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICML 2026\] Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs](systematic_failures_in_collective_reasoning_under_distributed_information_in_mul.md)
- [\[ACL 2025\] Voting or Consensus? Decision-Making in Multi-Agent Debate](../../ACL2025/multi_agent/voting_or_consensus_decision-making_in_multi-agent_debate.md)
- [\[NeurIPS 2025\] Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning](../../NeurIPS2025/multi_agent/adaptive_coopetition_leveraging_coarse_verifier_signals_for_resilient_multi-agen.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](../../AAAI2026/multi_agent/beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [\[ACL 2026\] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](../../ACL2026/multi_agent/scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)

</div>

<!-- RELATED:END -->
