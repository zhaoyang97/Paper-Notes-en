---
title: >-
  [Paper Note] GPO: Learning from Critical Steps to Improve LLM Reasoning
description: >-
  [NeurIPS 2025][LLM Reasoning][Critical step identification] GPO estimates the advantage function for each step in a reasoning trajectory via Monte Carlo simulation to identify "critical steps" (the turning points where t…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "Critical step identification"
  - "reinforcement learning"
  - "reasoning optimization"
  - "advantage function"
  - "process-level optimization"
date: 2026-05-08
content_hash: fce8ad1f926dbffe
---

# GPO: Learning from Critical Steps to Improve LLM Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2509.16456](https://arxiv.org/abs/2509.16456)  
**Code**: Available (provided with the paper)  
**Area**: LLM Reasoning
**Keywords**: Critical step identification, reinforcement learning, reasoning optimization, advantage function, process-level optimization

## TL;DR
GPO estimates the advantage function for each step in a reasoning trajectory via Monte Carlo simulation to identify "critical steps" (the turning points where the model makes errors), then resets from those critical steps and resamples new trajectories for training. This plug-and-play approach consistently improves multiple optimization algorithms—including PPO, DPO, KTO, SimPO, and ORPO—on reasoning tasks.

## Background & Motivation

**Background**: Improvements in LLM reasoning capability currently rely primarily on post-training methods such as PPO (online RL) and DPO/SimPO/KTO (offline preference optimization), which have been validated by systems such as DeepSeek-R1 and OpenAI O1.

**Limitations of Prior Work**: These methods optimize entire reasoning trajectories as monolithic units. However, errors in LLM reasoning typically originate from a specific "critical step"—one at which correct handling leads to success while incorrect handling causes all subsequent reasoning to collapse. Whole-trajectory optimization cannot effectively focus on these critical steps.

**Key Challenge**: Although Satori and similar works introduce trajectory reset, the reset point is selected **randomly** without truly identifying which step is critical, resulting in low signal efficiency.

**Goal**: How can critical steps in reasoning trajectories be automatically localized, and how can more informative training data be constructed?

**Key Insight**: Drawing on the notion of critical state identification from Explainable RL (XRL), each reasoning step is treated as an action in an MDP, and the absolute value of the advantage function is used to measure the importance of each step—the step with the largest advantage value is designated the "critical step."

**Core Idea**: Use the advantage function to localize critical steps in reasoning trajectories, reset and resample trajectories from those steps, and focus training on the key turning points where the model most needs to learn.

## Method

### Overall Architecture
GPO proceeds as follows: given a problem $x$, the current policy $\pi$ generates a reasoning trajectory $y = (y_0, y_1, \ldots, y_{K-1})$ (split into multiple steps by newlines); the advantage function $A^\pi(x, y_{0:i-1}; y_i)$ is estimated for each step; the step $y_m$ with the largest advantage value is selected as the critical step; the trajectory is truncated at $y_m$ and a new trajectory $y'$ is resampled using $\pi$; finally, the new trajectory is added to the training data for optimization with PPO, DPO, or other algorithms.

### Key Designs

1. **Critical Step Identification**:

    - **Function**: Identify the step with the greatest influence on the final outcome within a reasoning trajectory.
    - **Mechanism**: The reasoning process is modeled as an MDP. The advantage function for each step is defined as $A^\pi(x, y_{0:i-1}; y_i) = Q^\pi(x, y_{0:i-1}; y_i) - Q^\pi(x, y_{0:i-2}; y_{i-1})$ and estimated via Monte Carlo (MC) simulation. Specifically, multiple complete trajectories are sampled from each step onward, their accuracy rates are aggregated to approximate Q-values, and the step with the largest advantage value is selected.
    - **Design Motivation**: A large advantage value indicates that the difference in success rates between "continuing from this step" and "continuing from the previous step" is maximal, marking this step as the watershed between success and failure. Compared to the random selection in Satori, this approach precisely localizes the model's weaknesses.

2. **Trajectory Reset & Resample**:

    - **Function**: Truncate the original trajectory at the identified critical step and resample subsequent reasoning paths.
    - **Mechanism**: The correct reasoning prefix preceding the critical step is retained, and the model is allowed to explore anew from this critical position, generating new trajectories. For PPO, new trajectories are added directly to the online buffer; for DPO, correct and incorrect continuations form preference pairs.
    - **Design Motivation**: Training data generated in this way exhibits greater diversity at critical decision points, providing the model with richer experience precisely where learning is most needed.

3. **Plug-and-Play General Framework**:

    - **Function**: GPO integrates seamlessly with PPO (Procedure-I) and DPO/KTO/SimPO/ORPO (Procedure-II).
    - **Mechanism**: For online methods (PPO), resampled trajectories are added directly to the buffer and trained with the original reward signal. For offline methods (DPO), two trajectories are sampled from the critical step to form a new preference pair. Hyperparameters remain unchanged across both settings.
    - **Design Motivation**: Rather than replacing existing methods, GPO augments them, maximizing its applicability across the broad landscape of existing approaches.

### Loss & Training
GPO does not introduce a new loss function; instead, it modifies how training data are constructed. Theoretically, for DPO, GPO is equivalent to advantage-weighted RL with the advantage function as weights (Theorem 5.3):
$$\max_\pi \mathbb{E}\left[\sum_i \log \pi(y_i|x, y_{0:i-1}) \cdot \exp\!\left(A^{\pi_{ref}}(x, y_{0:i-1}; y_i) / \beta\right)\right].$$
For online PPO, the authors prove that the advantage-weighted sampling strategy tightens the regret bound.

## Key Experimental Results

### Main Results
Base model: DeepSeek-R1-Distill-Qwen-7B, fine-tuned with LoRA.

| Algorithm | BBH | MATH | GSM8K | MMLU | MMLUPro | AIME-24 | AIME-25 |
|-----------|-----|------|-------|------|---------|---------|---------|
| Base Model | 59.97 | 71.60 | 86.50 | 54.09 | 38.80 | 13.33 | 16.67 |
| PPO | 61.82 | 79.60 | 86.96 | 56.66 | 47.47 | 26.67 | 23.33 |
| **GPO-PPO** | **63.48** | **87.80** | **87.44** | **59.39** | **51.05** | **30.00** | **26.67** |
| DPO | 63.20 | 82.40 | 86.05 | 57.08 | 48.28 | 20.00 | 20.00 |
| **GPO-DPO** | **64.25** | **86.80** | **88.48** | **58.93** | **51.93** | **26.67** | **26.67** |
| KTO | 62.86 | 77.20 | 89.31 | 59.42 | 49.02 | 20.00 | 20.00 |
| **GPO-KTO** | **64.31** | **79.60** | **90.25** | **61.35** | **50.52** | **23.33** | **26.67** |
| SimPO | 61.97 | 72.20 | 86.58 | 56.93 | 45.70 | 20.00 | 23.33 |
| **GPO-SimPO** | **62.58** | **74.00** | **88.35** | **57.44** | **47.74** | **23.33** | **26.67** |

### Ablation Study

| Configuration | BBH | MATH | Note |
|---------------|-----|------|------|
| PPO | 61.82 | 79.60 | Baseline |
| PPO + Random Reset (Satori) | ~62 | 79.9 | Random reset |
| **GPO-PPO** | **63.48** | **87.9** | Advantage-guided reset |
| DPO | 63.20 | 82.40 | Baseline |
| DPO + Random Reset (Satori) | ~63.5 | ~83.5 | Random reset |
| **GPO-DPO** | **64.25** | **86.8** | Advantage-guided reset |

### Key Findings
- GPO yields the most substantial gains on MATH: PPO → GPO-PPO improves by 8.2% and DPO → GPO-DPO improves by 4.4%, indicating that critical step identification is especially important for mathematical reasoning.
- GPO produces consistent improvements across all five optimization algorithms and seven datasets, validating the generality of the approach.
- Performance improves continuously as the number of MC samples increases from 2 to 12, saturating beyond 12; using the default of 4 samples already yields strong results.
- Consistent gains are observed across models ranging from 1.5B to 70B parameters, demonstrating effectiveness at different scales.
- In a user study, 50 participants selected GPO-identified critical steps as the most important across five problems at rates of 44%, 68%, 88%, 76%, and 56%, closely aligning with human judgment.

## Highlights & Insights
- **Transferring critical state identification from XRL to LLM reasoning**: The analogy is elegant—reasoning trajectory = RL trajectory, reasoning step = action, critical step = critical state. This cross-domain transfer is a valuable paradigm worth emulating.
- **Plug-and-play design**: GPO modifies only how training data are constructed without altering any hyperparameters or loss functions of the underlying optimization algorithm, enabling broad adoption. This "change the data, not the algorithm" strategy is a highly practical design paradigm.
- **Dual validation via theory and experiment**: The approach is supported by theoretical regret-bound guarantees as well as comprehensive experiments spanning multiple algorithms, datasets, and model scales, complemented by a user study for qualitative validation.

## Limitations & Future Work
- MC simulation introduces approximately 1.8–1.9× computational overhead (roughly doubling PPO training time), with greater cost for longer reasoning chains.
- The current approach uses a simple heuristic of splitting reasoning steps by newlines and merging short steps; more complex reasoning structures (tree- or graph-structured reasoning) may require finer-grained segmentation strategies.
- Generalized Advantage Estimation (GAE) could be explored as an alternative to MC simulation to reduce variance and computational cost.
- Evaluation of critical steps currently relies on downstream performance and time-consuming human evaluation, lacking automated assessment metrics.

## Related Work & Insights
- **vs. Satori**: Satori also employs trajectory reset but selects reset points randomly. GPO uses the advantage function to precisely localize critical steps, and GPO-PPO (87.9%) substantially outperforms Satori-style random reset (79.9%) on MATH.
- **vs. Step-DPO**: Step-DPO applies step-level DPO at each token; GPO operates at the level of reasoning steps (which carry greater semantic meaning) and is theoretically proven to be equivalent to advantage-weighted RL.
- **vs. VinePPO**: VinePPO also employs fine-grained credit assignment, but the key distinction of GPO is that it generates new training data through resampling rather than merely reweighting gradients.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The core idea is clear and elegant (transferring critical state identification from XRL), though trajectory reset has precedent in works such as Satori.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five algorithms × seven datasets × multiple model scales, with ablation studies and a user study—extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure, complementary theory and experiments, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ A plug-and-play general strategy with strong practical utility, though computational overhead limits applicability at very large scales.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning](../../ACL2026/llm_reasoning/templaterl_structured_template-guided_reinforcement_learning_for_llm_reasoning.md)
- [\[NeurIPS 2025\] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)
- [\[NeurIPS 2025\] Curriculum Abductive Learning](curriculum_abductive_learning.md)
- [\[NeurIPS 2025\] On Learning Verifiers and Implications to Chain-of-Thought Reasoning](on_learning_verifiers_and_implications_to_chain-of-thought_reasoning.md)
- [\[NeurIPS 2025\] The Impact of Quantization on Large Reasoning Model Reinforcement Learning](the_impact_of_quantization_on_large_reasoning_model_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
