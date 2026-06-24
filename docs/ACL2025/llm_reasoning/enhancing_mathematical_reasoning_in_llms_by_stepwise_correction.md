---
title: >-
  [Paper Note] Enhancing Mathematical Reasoning in LLMs by Stepwise Correction
description: >-
  [ACL 2025][Reasoning][Mathematical Reasoning] This paper proposes StepCo (Stepwise Correction), an iterative "verify-and-correct" framework. It leverages a Process Supervised Verifier (PSV) to sequentially locate the first erroneous step in an LLM's reasoning path and trigger corrections by the LLM. Using GPT-4o as the backbone, StepCo achieves an average accuracy of 94.1% across 8 mathematical reasoning benchmarks, outperforming the Best-of-10 method by +2.4 percentage point…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Mathematical Reasoning"
  - "Stepwise Correction"
  - "Process Supervised Verifier"
  - "Iterative Verification"
  - "Error Propagation"
date: 2026-05-08
content_hash: d0ca461c3d30dbe2
---

# Enhancing Mathematical Reasoning in LLMs by Stepwise Correction

**Conference**: ACL 2025  
**arXiv**: [2410.12934](https://arxiv.org/abs/2410.12934)  
**Code**: [https://github.com/wzy6642/StepCo](https://github.com/wzy6642/StepCo)  
**Area**: LLM Reasoning  
**Keywords**: Mathematical Reasoning, Stepwise Correction, Process Supervised Verifier, Iterative Verification, Error Propagation

## TL;DR
This paper proposes StepCo (Stepwise Correction), an iterative "verify-and-correct" framework. It leverages a Process Supervised Verifier (PSV) to sequentially locate the first erroneous step in an LLM's reasoning path and trigger corrections by the LLM. Using GPT-4o as the backbone, StepCo achieves an average accuracy of 94.1% across 8 mathematical reasoning benchmarks, outperforming the Best-of-10 method by +2.4 percentage points while reducing token consumption by 77.8%.

## Background & Motivation
LLMs solve mathematical problems by generating intermediate reasoning steps (Chain-of-Thought). However, errors in reasoning paths are **propagative** — an error in one step cascades and affects all subsequent steps. Existing Best-of-N decoding methods let LLMs independently generate multiple reasoning paths and select the optimal one. However, this "repeated independent sampling" often leads to **repeating the same mistakes** — when the correct answer is not in the sample set, Best-of-N inevitably fails.

The authors observe that in 21.2% of Best-of-10 error cases, the correct reasoning path is not in the sample set at all, and LLMs tend to make the same errors at the same locations. The core contradiction is: without external feedback, LLMs cannot self-correct (Huang et al., 2024).

The key insight of this paper is: instead of independent multiple samplings, it is better to use an external verifier to precisely locate the erroneous step and then only correct that step and the subsequent parts. This both addresses error propagation and significantly reduces token consumption.

## Method

### Overall Architecture
The workflow of StepCo:
1. Given a problem, the LLM generates an initial reasoning path $r^{(0)}$
2. Enter the iterative verify-and-correct loop (up to $T=5$ rounds):
    - **Verification Phase**: The PSV evaluates the "probability of leading to the correct answer" $p(s_i)$ for each step, and finds the first step $s_k$ where the probability is below a threshold $\theta$.
    - **Correction Phase**: Retain the correct steps before $s_k$, and feed the erroneous step along with its probability as feedback to let the LLM correct $s_k$ and the subsequent steps.
3. If the probabilities of all steps exceed $\theta$, the current path is accepted; otherwise, the above process is repeated.

### Key Designs

1. **Training of the Process Supervised Verifier (PSV)**

    - Core Idea: Construct training data using an automatic process labeling method, and then fine-tune Llama-3-8B as the PSV.
    - Data Construction Process: For each question and gold standard answer, two types of demonstrations — $D^+$ (encouraging correct steps) and $D^-$ (encouraging exploration of erroneous/alternative steps) — are used at each step to expand reasoning paths, forming a complete binary tree.
    - Quality score of each step = the ratio of leaf nodes in the subtree rooted at that step that match the correct answer: $\pi(s_m) = \frac{\sum_{s_\ell \in \mathcal{L}_m^q} \mathbb{I}(s_\ell = a)}{|\mathcal{L}_m^q|}$
    - Fine-tune Llama-3-8B (LoRA) using MSE loss.

2. **Error Localization in the Verification Phase**

    - For each step in the reasoning path, concatenate the question and the prefix steps as input to the PSV to predict the probability of this step leading to the correct answer.
    - Find the first step whose probability is below the threshold $\theta$ (default 0.5), and mark it and subsequent steps as needing correction.
    - Key constraint: $k^{(t)} \geq k^{(t-1)}$, ensuring that verification does not backtrack to steps that have already passed verification.

3. **Precise Feedback in the Correction Phase**

    - Instead of simply having the LLM regenerate from scratch, explicit feedback is provided: "The probability of step $s_k$ leading to the correct answer is $p(s_k)$, please correct step $s_k$ and the following steps."
    - Keep the correct steps unchanged and only correct the erroneous parts, avoiding "modifying the correct steps and disrupting what is already good."

### Loss & Training
- PSV training uses MSE loss: $\mathcal{J} = \frac{1}{|\mathcal{D}|}\sum_{j=1}^{|\mathcal{D}|}(\text{PSV}(x_j) - y_j)^2$
- Fine-tune Llama-3-8B using LoRA.
- StepCo itself is an inference framework and does not require additional training of the backbone LLM.

## Key Experimental Results

### Main Results (8 mathematical reasoning datasets, GPT-3.5-Turbo / GPT-4o)

| Method | SVAMP | GSM8K | MATH500 | AQuA | Average |
|------|-------|-------|---------|------|------|
| Zero-Shot-CoT | 76.7/90.4 | 78.6/94.6 | 37.9/74.0 | 51.3/72.8 | 72.9/86.7 |
| Best-of-10 | 85.5/93.9 | 85.3/94.5 | 42.1/77.0 | 66.1/81.1 | 80.1/91.7 |
| CRITIC | 83.3/93.5 | 79.2/95.4 | 44.9/74.9 | 63.8/80.2 | 78.4/90.4 |
| **StepCo** | **89.7/96.0** | **87.0/96.4** | **56.9/80.4** | **72.4/84.7** | **84.7/94.1** |

### Efficiency Comparison

| Method | Average Accuracy (GPT-4o) | Token Consumption | Compared to Best-of-10 |
|------|-------------------|----------|---------------|
| Best-of-10 | 91.7% | 100% | — |
| StepCo (1 round) | 92.3% | ~22.2% | +0.6, 78% saved |
| StepCo (5 rounds) | **94.1%** | ~22.2% | +2.4, 77.8% saved |

### Ablation Study

| PSV Model | Average Accuracy (GPT-3.5-Turbo) |
|---------|-------------------------|
| StepCo (DiVeRSe) | 82.6 |
| StepCo (Math-Shepherd) | 83.8 |
| StepCo (Ours PSV) | **84.7** |

### Generalization on Non-Mathematical Reasoning (GPT-3.5/GPT-4o)

| Method | HotpotQA EM | HotpotQA F1 | CSQA ACC |
|------|------------|------------|---------|
| Best-of-10 | 32.9/52.0 | 44.1/57.1 | 73.0/83.4 |
| StepCo | **35.0/53.0** | **47.4/58.7** | **74.3/84.9** |

### Key Findings
- **Correction Analysis**: StepCo modifies the correct-to-incorrect ratio to only 5.3% (on GSM8K), whereas Self-Correct goes as high as 14.5%, indicating that the quality of external feedback is crucial.
- **Difficulty Analysis**: On the highest difficulty level (Level 5) of MATH500, StepCo still achieves 29.1%, significantly outperforming all baselines.
- **Threshold Sensitivity**: The performance is best when $\theta=0.5$. A larger $\theta$ increases the number of iterations but slightly decreases the accuracy.
- **Open-source Model Compatibility**: Using Llama-3-8B as the backbone LLM, StepCo outperforms Best-of-10 by +1.4 on MATH500.
- **21.2% of Best-of-10 errors** occur because the correct path is not in the sampled set — this is an inherent defect of the Best-of-N method.

## Highlights & Insights
- The core argument that **iterative correction is superior to repeated sampling** is highly persuasive: Best-of-N repeats the same mistakes, whereas StepCo uses external feedback to break this loop.
- The **automatic process labeling method** is ingenious: constructing a binary tree using positive/negative demonstrations and automatically calculating the quality score of each step via leaf node distribution eliminates the need for human annotation.
- **Generalization of PSV to non-mathematical tasks**: Although trained only on mathematical data, the PSV is also effective on HotpotQA and CSQA, indicating that the concept of "step quality" has some generalizability.
- The **77.8% token savings** are highly significant for actual deployment.

## Limitations & Future Work
- The PSV is trained only on English mathematical tasks; multilingual and broader reasoning types (such as code, scientific reasoning) have not been validated.
- For tasks where answers are not numerical or entity-type (such as open-ended generation), the current method of evaluation is difficult to apply.
- The PSV is based on Llama-3-8B; for extremely complex reasoning (such as competition-level mathematics), the PSV itself might fail to deliver accurate judgments.
- The threshold $\theta$ requires manual tuning depending on the task.
- **Research Idea**: StepCo could be combined with Reinforcement Learning (RL) — using step-level rewards provided by the PSV for online RL, allowing the model itself to learn to generate reasoning paths with fewer errors, rather than merely correcting them during inference.

## Related Work & Insights
- Math-Shepherd (Wang et al., 2024): Constructs process supervision data via Monte Carlo estimation. StepCo goes a step further by performing iterative correction on top of this framework.
- CRITIC (Gou et al., 2024): Representative work in self-correction with external feedback. StepCo is more precise at the step level.
- OmegaPRM (Luo et al., 2024): Automatic process labeling method, sharing a similar concept of binary tree construction to StepCo.
- The failure cases of Self-Correct (14.5% correcting correct steps to incorrect ones) once again validate the conclusion of Huang et al. (2024) that "self-correction is impossible without external feedback."

## Rating
- Novelty: ⭐⭐⭐⭐ The iterative verification-correction framework combined with automatic process labeling has a clear mindset and effectively distinguishes itself from the Best-of-N paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 mathematical benchmarks + 2 non-mathematical tasks, 3 LLM backbones, and multiple ablations/analyses.
- Writing Quality: ⭐⭐⭐⭐ The task motivation is clearly articulated, the method description is rigorous, and the experimental analysis is thoroughly conducted.
- Value: ⭐⭐⭐⭐ Highly practical, and the 77.8% token savings have direct significance for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning](self-error-instruct_generalizing_from_errors_for_llms_mathematical_reasoning.md)
- [\[ACL 2025\] Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](safe_math_reasoning.md)
- [\[ACL 2025\] BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving](bpp-search_enhancing_tree_of_thought_reasoning_for_mathematical_modeling_problem.md)
- [\[ACL 2025\] ProcessBench: Identifying Process Errors in Mathematical Reasoning](processbench_identifying_process_errors_in_mathematical_reasoning.md)
- [\[ACL 2025\] Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](mclm_multilingual_test_time_scaling.md)

</div>

<!-- RELATED:END -->
