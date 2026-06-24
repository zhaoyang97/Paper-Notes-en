---
title: >-
  [Paper Note] An Efficient and Precise Training Data Construction Framework for Process-Supervised Reward Model in Mathematical Reasoning
description: >-
  [ACL 2025][Reasoning][Process-Supervised Reward Model] This paper proposes the EpicPRM framework, which quantifies the contribution of each reasoning step through perplexity-based Monte Carlo estimation and utilizes adaptive binary search to efficiently locate the first incorrect step. It constructs Epic50k, a high-quality process-supervised dataset (with only 50k annotated steps), which trains a PRM that performs comparably to or even outperforms models trained on PRM800k.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Process-Supervised Reward Model"
  - "Data Construction"
  - "Monte Carlo Estimation"
  - "Binary Search"
  - "EpicPRM"
date: 2026-05-08
content_hash: 42193aacc0eea475
---

# An Efficient and Precise Training Data Construction Framework for Process-Supervised Reward Model in Mathematical Reasoning

**Conference**: ACL 2025  
**arXiv**: [2503.02382](https://arxiv.org/abs/2503.02382)  
**Code**: [https://github.com/xiaolizh1/EpicPRM](https://github.com/xiaolizh1/EpicPRM)  
**Area**: LLM Reasoning / Mathematical Reasoning  
**Keywords**: Process-Supervised Reward Model, Data Construction, Monte Carlo Estimation, Binary Search, EpicPRM

## TL;DR
This paper proposes the EpicPRM framework, which quantifies the contribution of each reasoning step through perplexity-based Monte Carlo estimation and utilizes adaptive binary search to efficiently locate the first incorrect step. It constructs Epic50k, a high-quality process-supervised dataset (with only 50k annotated steps), which trains a PRM that performs comparably to or even outperforms models trained on PRM800k.

## Background & Motivation

**Background**: Enhancing the mathematical reasoning capabilities of LLMs is a core challenge in the AI field. Process supervision, which trains a PRM (Process-Supervised Reward Model) by providing correctness annotations for each step in a reasoning chain, has been proven more effective than outcome supervision.

**Limitations of Prior Work**: Existing process-supervised data construction methods face a dilemma between cost and quality: (1) Human annotation methods (e.g., PRM800k) yield high quality but annotating 800k steps requires massive human and financial resources, making it hard to scale to new domains; (2) Automatic annotation methods (e.g., Math-Shepherd) use Monte Carlo (MC) estimation to evaluate step correctness, but require sampling a large number of rollouts for each step, which is computationally expensive and has annotation precision limited by the sample size.

**Key Challenge**: MC estimation approximates the probability of a step being correct by the ratio of successful rollouts $M/N$. However, when the sample size $N$ is not large enough, this estimate is imprecise (e.g., flipping a coin twice and getting heads both times does not mean a 100% probability). Moreover, performing MC estimation on all steps of an incorrect reasoning chain is wasteful—in practice, only the first incorrect step needs to be located.

**Goal**: Improve the precision of process-supervised annotations while reducing computational costs.

**Key Insight**: (1) Replace counting with perplexity to estimate MC probability more accurately; (2) Replace step-by-step search with adaptive binary search to locate the first incorrect step more efficiently.

**Core Idea**: Define the correctness of a step as its "contribution" to the final answer. The contribution is quantified using a perplexity-based MC estimation (MC_PPL), and an adaptive binary search algorithm dynamically adjusted to question difficulty is used to efficiently find the first incorrect step.

## Method

### Overall Architecture
The EpicPRM framework consists of three steps: (1) Use multiple LLMs to generate diverse CoT reasoning chains; (2) Apply adaptive binary search combined with MC_PPL on each incorrect chain to locate the first incorrect step; (3) Annotate all steps before the first incorrect step as correct, and those after as incorrect, forming the training data.

### Key Designs

1. **Perplexity-based Monte Carlo Estimation (MC_PPL)**:

    - **Function**: More precisely estimate the probability of reaching the correct answer from a given reasoning state.
    - **Mechanism**: Traditional MC estimation uses $M/N$ (where $M$ of $N$ rollouts are correct) to approximate the probability, which exhibits high variance under small sample sizes. This paper replaces counting with perplexity weights: $MC_{PPL}(s_t, \theta_{1:K}) = \frac{1}{K} \sum_{k=1}^{K} \frac{\sum_{m=1}^{M} \log PPL(j; s_t, \theta_k)}{\sum_{n=1}^{N} \log PPL(j; s_t, \theta_k)}$. Here, $PPL$ directly calculates the probability of the model generating each rollout. Compared to simple counting, PPL weights consider the generation probability of each rollout, preventing "accidentally correct but extremely low-probability" rollouts from misleading the estimation.
    - **Design Motivation**: Perplexity can be directly obtained from the model without extra sampling. When a correct rollout has an extremely low generation probability, the counting method overestimates the correctness probability, whereas the PPL method does not.

2. **Step Contribution Quantification**:

    - **Function**: Distinguish whether a step is a "contributing correct step" or an "incorrect step that happens to have no impact".
    - **Mechanism**: Define the contribution of step $s_t$ as $MC_{PPL}(s_t) - MC_{PPL}(s_{t-1})$. If a step does not make a positive contribution to the probability of the final correct answer (contribution $\le 0$), it is labeled as a potential error, even if subsequent rollouts happen to yield the correct answer. This solves the issue of mislabeling "error step + self-correction" as correct (as shown in the case in Figure 1 of the paper).
    - **Design Motivation**: Traditional methods assume "if a correct answer can be reached from a step, then the step is correct," ignoring the self-correction capability of the completer. In reality, a clearly incorrect step may still lead to a correct answer due to the LLM's self-correction ability.

3. **Adaptive Binary Search**:

    - **Function**: Efficiently locate the first incorrect step in a reasoning chain.
    - **Mechanism**: Model "finding the first incorrect step" as a search problem in an ordered sequence. Utilizing binary search reduces the number of MC estimations from $O(n)$ (step-by-step search) to $O(\log n)$. The key improvements are in adaptivity: (1) Adjusting the search starting point based on question difficulty—errors in difficult questions usually appear in earlier steps, while errors in simple questions appear in later steps, so the initial position of binary search is adjusted based on question difficulty; (2) Dynamically adjusting the sample size at each search point based on the confidence level of MC_PPL values—sampling less at highly confident positions and more near the boundaries.
    - **Design Motivation**: Experiments reveal that the ratio of the first incorrect step's position to the total steps is low for difficult questions (about 0.3) and high for simple questions (about 0.7). Leveraging this prior knowledge can reduce search steps.

### Loss & Training
PRM training uses standard process supervision loss—predicting the correctness label (correct/incorrect) for each annotated step using cross-entropy loss.

## Key Experimental Results

### Main Results

| PRM Training Data | Data Size | MATH best-of-N | GSM8K best-of-N |
|------------|--------|----------------|-----------------|
| PRM800k (Human) | 800k steps | 68.4 | 82.1 |
| Math-Shepherd | ~440k steps | 66.2 | 80.8 |
| **Epic50k** | **50k steps** | **69.1** | **82.5** |

### Ablation Study

| Configuration | MATH BoN | Description |
|------|---------|------|
| Full EpicPRM | 69.1 | Complete framework |
| Replace MC_PPL with M/N | 65.8 | PPL Contribution: +3.3 |
| Replace Binary Search with Step-by-Step Search | 68.7 | Comparable accuracy but 64% higher cost |
| Without Step Contribution | 66.5 | Contribution filtering is crucial |
| Without Multi-Model Rollouts | 67.2 | Multi-model diversity helps |

### Key Findings
- Epic50k achieves better PRM performance using less than 10% of PRM800k's data, strongly demonstrating that "data quality > data quantity".
- MC_PPL yields a 3.3-point improvement compared to the traditional counting method, showing that perplexity indeed provides a more precise MC estimate.
- Adaptive binary search reduces the computational cost of annotation by 64.39% compared to step-by-step search, with negligible accuracy loss (<0.5 points).
- The strong correlation between question difficulty and the position of the first incorrect step is an interesting empirical finding, providing a prior for search optimization.

## Highlights & Insights
- Replacing counting with perplexity for MC estimation is a simple yet effective improvement that exploits the LLM's capability to directly output token probabilities. This idea can be generalized to all scenarios using MC estimation.
- The concept of "step contribution" elegantly addresses the annotation noise caused by the LLM's self-correction capability. This has been an overlooked but highly impactful issue in the PRM literature.
- Adaptive binary search reduces annotation cost by 64%, making the construction of high-quality process-supervised data practically feasible.

## Limitations & Future Work
- Currently only validated in mathematical reasoning; validation in other domains requiring process supervision, such as code generation and scientific reasoning, is still needed.
- Adaptive search relies on the prior assumption that "errors in difficult questions occur earlier," which may not hold for certain question types.
- MC_PPL requires the completer model to output token-level probabilities, which is not applicable to closed-source API models.
- The data quality of Epic50k may be limited by the capability of the completer model, and a stronger completer could yield better data.

## Related Work & Insights
- **vs PRM800k (Lightman et al.)**: PRM800k relies heavily on human annotation, while EpicPRM is fully automated, reduces data size by 20x, and yields better quality.
- **vs Math-Shepherd**: Math-Shepherd uses traditional MC estimation, which has limited precision. EpicPRM's PPL estimation and contribution filtering provide higher-quality annotations.
- **vs OmegaPRM**: OmegaPRM introduced the binary search concept, but EpicPRM improves on it by incorporating adaptivity and PPL-based estimation.

## Rating
- Novelty: ⭐⭐⭐⭐ PPL replacing counting and step contribution are valuable technical innovations, but overall it is an improvement over existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation studies are comprehensive, but evaluation benchmarks are limited (only MATH and GSM8K).
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, and the mathematical derivations are rigorous.
- Value: ⭐⭐⭐⭐⭐ Significantly lowers the cost of high-quality PRM data, holding major practical value for mathematical reasoning research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](../../NeurIPS2025/llm_reasoning/unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[ACL 2025\] ProcessBench: Identifying Process Errors in Mathematical Reasoning](processbench_identifying_process_errors_in_mathematical_reasoning.md)
- [\[ACL 2025\] Dynamic and Generalizable Process Reward Modeling (DG-PRM)](dgprm_dynamic_process_reward.md)
- [\[NeurIPS 2025\] DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](../../NeurIPS2025/llm_reasoning/dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)
- [\[ICML 2026\] GRPO is Secretly a Process Reward Model](../../ICML2026/llm_reasoning/grpo_is_secretly_a_process_reward_model.md)

</div>

<!-- RELATED:END -->
