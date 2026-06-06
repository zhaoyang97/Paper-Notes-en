---
title: >-
  [Paper Note] Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information
description: >-
  [ICML 2026][Multi-Agent][LLM Aggregation] This paper proposes two LLM answer aggregation algorithms that leverage higher-order information: Optimal Weight (OW), based on first-order accuracy information…
tags:
  - "ICML 2026"
  - "Multi-Agent"
  - "LLM Aggregation"
  - "Multi-Agent Reasoning"
  - "Information Aggregation"
  - "Bayesian Optimality"
  - "Unsupervised Annotation"
date: 2026-05-08
content_hash: 6753e5fb283dbc3e
---

# Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information

**Conference**: ICML 2026  
**arXiv**: [2510.01499](https://arxiv.org/abs/2510.01499)  
**Code**: None  
**Area**: Multi-Agent  
**Keywords**: LLM Aggregation, Multi-Agent Reasoning, Information Aggregation, Bayesian Optimality, Unsupervised Annotation  

## TL;DR

This paper proposes two LLM answer aggregation algorithms that leverage higher-order information: Optimal Weight (OW), based on first-order accuracy information, and Inverse Surprising Popularity (ISP), based on second-order correlation information. The authors demonstrate that these methods are provably superior to Majority Voting (MV) without requiring labels, and validate consistent improvements across UltraFeedback, MMLU, and healthcare datasets.

## Background & Motivation

**Background**: Multi-agent LLM reasoning (e.g., LLM debate, LLM council) has been widely adopted. In aggregating multiple model responses, the vast majority of existing works use Majority Voting (MV) as the standard aggregation strategy.

**Limitations of Prior Work**: MV is a "zero-order" method that relies solely on the frequency of raw answers, completely ignoring the heterogeneity of capabilities among different LLMs (where some models have 90% accuracy and others only 60%) as well as the correlations between responses. This implies that weak and strong models hold equal weight in voting, and the final result can be easily misled when multiple weak models commit the same error.

**Key Challenge**: Utilizing model accuracy (first-order information) for weighting requires a large amount of labeled data to estimate accuracy, which is unavailable in unsupervised scenarios such as auto-annotation or prediction markets. Meanwhile, the classic Surprisingly Popular (SP) method, although it utilizes second-order correlation information without needing labels, actually underperforms MV in LLM scenarios. This occurs because LLMs do not exhibit the same systematic biases as human groups, causing the signals utilized by SP to have a counterproductive effect.

**Goal**: Design aggregation algorithms that do not rely on labels, utilize higher-order information, and are provably superior to MV.

**Key Insight**: The authors observe that after randomly shuffling the order of options, the joint distribution possesses a symmetric structure (all incorrect options are equally probable). This structure allows for the derivation of a closed-form optimal solution. Furthermore, the reason SP fails is that its "bias amplification" direction is exactly opposite to what is required in LLM scenarios; reversing the calculation direction of SP can correct this.

**Core Idea**: Use the inverse sigmoid function of accuracy as the optimal weight to achieve Bayesian optimal aggregation. When labels are unavailable, the prediction direction of SP is reversed (calculating scores from a "counterfactual" perspective) to leverage second-order information and surpass MV.

## Method

### Overall Architecture

Given the answers $a_1, \ldots, a_N$ from $N$ LLMs for a multiple-choice question with $K$ options, the system first randomly shuffles the option order for each problem (a preprocessing step) to ensure the joint distribution exhibits symmetry. Then, an aggregator is selected based on the available information level: OW is used when accuracy is known; ISP is used when labels are absent, or accuracy is estimated via ISP before applying OW (OW-I / OW-L). The final result is output by mapping back to the original option order using an inverse permutation.

### Key Designs

1.  **Optimal Weight (OW) — First-order Optimal Aggregation**:

    - **Function**: Calculates the Bayesian optimal weighted vote when the accuracy $x_i$ of each LLM is known.
    - **Mechanism**: Assigns weight $\omega_i = \sigma_K^{-1}(x_i)$ to the $i$-th agent, where $\sigma_K(x) = e^x / (K-1+e^x)$ is the generalized sigmoid function. The aggregation formula is $f_{OW} = \arg\max_s \sum_i \sigma_K^{-1}(x_i) \cdot \mathbb{1}\{a_i = s\}$. When $K=2$, this reduces to the inverse of the logistic function, establishing a theoretical link with the Bradley-Terry model.
    - **Design Motivation**: Under the symmetric information structure induced by random shuffling, this set of weights is precisely the necessary and sufficient solution for maximizing posterior probability, representing the Bayesian optimum among all aggregators (not limited to linear weighting). When all agents are homogeneous, OW reduces to MV, indicating that MV is only optimal in scenarios like self-consistency sampling.

2.  **Inverse Surprising Popularity (ISP) — Second-order Unsupervised Aggregation**:

    - **Function**: Aggregates using only the conditional probabilities $\mathbb{P}(A_i|A_j)$ between agents, without requiring any labels.
    - **Mechanism**: Unlike classic SP, which calculates "how each agent predicts others will answer," ISP calculates "how I would predict if other agents gave a counterfactual answer." Specifically, the ISP score is $S_{ISP}(s,i) = \frac{1}{N-1}\sum_{j \neq i} \frac{1}{K-1}\sum_{a \neq a_j} \mathbb{P}(A_i=s|A_j=a)$, and the option with the maximum advantage function $Adv_{ISP}(s) = \sum_i \mathbb{1}\{a_i=s\} - \sum_i S_{ISP}(s,i)$ is selected.
    - **Design Motivation**: Classic SP is inferior to MV in LLM scenarios because LLMs lack the systematic bias of human groups that often underestimate correct answers. ISP reverses the prediction direction—using counterfactual conditional probabilities instead of true conditional probabilities—biasing the prediction score toward the incorrect direction and thereby amplifying the advantage of the correct answer. It is theoretically proven that $\mathbb{E}[Adv_{ISP}(s^*)] \geq \mathbb{E}[Adv_{MV}(s^*)] \geq \mathbb{E}[Adv_{SP}(s^*)]$.

3.  **OW-L / OW-I — Estimating First-order Weights from Second-order Information**:

    - **Function**: Approximately obtains the accuracy parameters required for OW in unsupervised scenarios.
    - **Mechanism**: OW-L solves for accuracies $\hat{x}_1, \ldots, \hat{x}_N$ by minimizing the mean squared error between empirical and theoretical conditional probabilities. OW-I uses the aggregation results of ISP as pseudo-labels and directly calculates the consistency rate of each agent with these pseudo-labels as the accuracy estimate. Once accuracies are estimated, both methods substitute them into the OW weight formula $\sigma_K^{-1}(\hat{x}_i)$ for aggregation.
    - **Design Motivation**: While OW is Bayesian optimal, it requires labels to estimate accuracy. These two methods "bridge" second-order information to the first-order optimal framework. Experiments show that both perform consistently on real-world data and outperform direct use of ISP.

## Key Experimental Results

### Main Results (Simulated Data)

| Method | $K=2$ | $K=4$ | $K=6$ | $K=8$ | $K=10$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MV | 85.13% | 92.64% | 94.22% | 94.85% | 95.54% |
| SP | 79.94% | 90.52% | 92.68% | 93.66% | 94.40% |
| Single Best | 90.34% | 89.94% | 90.31% | 89.95% | 90.05% |
| **ISP (Ours)** | **90.48%** | **94.45%** | **95.78%** | **96.23%** | **96.49%** |
| OPT (clairvoyant) | 91.37% | 94.94% | 96.05% | 96.46% | 96.81% |

### Main Results (Real Datasets with 4 Strong Models)

| Method | UltraFeedback | MMLU | ARMMAN |
| :--- | :--- | :--- | :--- |
| MV | 72.21% | 89.32% | 85.24% |
| ISP | 73.26% | 90.01% | 85.78% |
| **OW-L** | **73.66%** | **90.37%** | **85.78%** |
| **OW-I** | **73.66%** | **90.37%** | **85.78%** |
| Single Best (oracle) | 73.14% | 91.02% | 85.32% |

### Key Findings

- ISP outperforms MV across all values of $K$, and the gap between the two narrows as $K$ increases ($\Theta(1/K)$), consistent with theoretical predictions.
- Across 16 model combinations, OW-L outperforms MV in 97.92% of cases, with a maximum absolute gain of 14.20%; MV was never the best in any combination.
- Hypothesis testing yields t-statistics of 12.53 (UltraFeedback), 23.39 (MMLU), and 3.22 (ARMMAN), with all p-values < 0.001, indicating statistically significant improvements.
- On the "hard distractors" subset (MMLU-hard, where at least two models chose the same wrong option), OW-L/OW-I improved over MV by more than 7% (17.23% $\rightarrow$ 24.79%), indicating that higher-order information is more valuable in difficult scenarios.

## Highlights & Insights

- **Bayesian Optimality of Inverse Sigmoid Weights**: The seemingly simple weighting scheme $\omega_i = \sigma_K^{-1}(x_i)$ is actually the optimal solution (not limited to linear) among all possible aggregators. this provides theoretical endorsement for the use of Bradley-Terry models in RLHF. This conclusion is elegant and directly applicable.
- **Counter-intuitive Design by Reversing SP**: Classic SP is effective in human groups but fails in LLMs. The authors deeply analyze the reasons (LLMs lack human systematic bias) and reverse the direction to derive ISP. This "diagnose failure cause $\rightarrow$ targeted modification" approach is highly commendable.
- **Bridging Second-order Information to First-order Optimality**: OW-L/OW-I transform "unlabeled second-order information" into "first-order optimal weights that require labels." This idea of indirectly utilizing information hierarchies can be transferred to other unsupervised aggregation scenarios.

## Limitations & Future Work

- Theoretical analysis relies on the conditional independence assumption (LLMs are independent given the correct answer). While experiments show effectiveness even when assumptions are violated, formal robustness bounds are lacking.
- All models use the same global weights for the same problem, without considering differences in model capabilities across different problem types (e.g., a model might excel at math but struggle with linguistic understanding). Prompt-specific weighting is a clear direction for improvement.
- Only closed-ended multiple-choice questions ($K$ options) are handled; extension to open-ended generation tasks is currently unclear.
- The theoretical assumption regarding position bias (that LLMs are unaffected by option order) does not hold perfectly for weak models. Although experiments show effectiveness without explicit debiasing, theoretical guarantees for weak models need strengthening.

## Related Work & Insights

- **Surprising Popularity (Prelec et al., 2017)**: A classic aggregation method based on second-order information, but the authors prove that SP is inferior to MV in LLM scenarios. ISP is a targeted improvement over SP.
- **Bradley-Terry Model**: Corollary 3.2 establishes a link between OW and BT models, providing theoretical support for the effectiveness of BT models in RLHF.
- **Self-Consistency (Wang et al., 2022)**: Corollary 3.3 proves that MV is optimal when agents are homogeneous, meaning more complex aggregation is unnecessary in self-consistency scenarios.
- **Insight**: This framework can serve as a direct drop-in replacement for MV in multi-LLM systems, with computational overhead limited to a few seconds at the CPU level, making it suitable for API-based scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs](systematic_failures_in_collective_reasoning_under_distributed_information_in_mul.md)
- [\[ACL 2026\] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](../../ACL2026/multi_agent/scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)
- [\[ACL 2026\] Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](../../ACL2026/multi_agent/collaborative_multi-agent_scripts_generation_for_enhancing_imperfect-information.md)
- [\[ICML 2026\] ProtocolBench: Which LLM MultiAgent Protocol to Choose?](protocolbench_which_llm_multiagent_protocol_to_choose.md)
- [\[ICML 2026\] MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
