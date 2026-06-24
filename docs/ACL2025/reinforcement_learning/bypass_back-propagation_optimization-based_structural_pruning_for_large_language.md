---
title: >-
  [Paper Note] Bypass Back-propagation: Optimization-based Structural Pruning for Large Language Models via Policy Gradient
description: >-
  [ACL 2025][Reinforcement Learning][Structural Pruning] This paper proposes a policy gradient-based structural pruning method for LLMs. By learning Bernoulli pruning masks in the probability space, it directly optimizes the loss function of the pruned model without requiring any backpropagation through the LLM itself, relying solely on forward inference to complete pruning optimization.
tags:
  - "ACL 2025"
  - "Reinforcement Learning"
  - "Structural Pruning"
  - "Policy Gradient"
  - "Backpropagation-Free"
  - "Bernoulli Distribution"
  - "LLM Compression"
date: 2026-05-08
content_hash: c849aa0f50354873
---

# Bypass Back-propagation: Optimization-based Structural Pruning for Large Language Models via Policy Gradient

**Conference**: ACL 2025  
**Link**: [ACL Anthology](https://aclanthology.org/2025.acl-long.1421/)
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: Structural Pruning, Policy Gradient, Backpropagation-Free, Bernoulli Distribution, LLM Compression

## TL;DR

This paper proposes a policy gradient-based structural pruning method for LLMs. By learning Bernoulli pruning masks in the probability space, it directly optimizes the loss function of the pruned model without requiring any backpropagation through the LLM itself, relying solely on forward inference to complete pruning optimization.

## Background & Motivation

**Background**: Large language models have parameters ranging from billions to hundreds of billions, incurring high deployment costs. Model pruning is one of the primary methods for compressing LLMs. Current mainstream LLM pruning methods (e.g., SparseGPT, Wanda) mainly operate in the post-training stage without requiring expensive weight fine-tuning.

**Limitations of Prior Work**: Pruning criteria of existing methods typically rely on **handcrafted heuristic metrics** (such as weight magnitudes, gradient information, activation statistics, etc.). These metrics are essentially indirect estimations of "which parameters are unimportant," which can lead to suboptimal pruning decisions. An ideal pruning scheme should directly optimize the task loss of the pruned model, but this requires backpropagation through the LLM, which is computationally prohibitive.

**Key Challenge**: Optimization-based pruning can achieve better results but requires backpropagation through the entire LLM, leading to unacceptable memory and computational overhead. Conversely, heuristic-based pruning is highly efficient but yields suboptimal results.

**Goal**: To design a structural pruning method that is optimization-based (directly minimizing task loss) while bypassing the need for backpropagation through the LLM.

**Key Insight**: Pruning masks are treated as random variables parameterized by a Bernoulli distribution. A policy gradient estimator is utilized to decouple the gradient calculation from the LLM. Only forward passes of the LLM are needed to evaluate loss under different masks, and the parameters of the Bernoulli distribution are then updated using the REINFORCE algorithm.

**Core Idea**: Replace backpropagation through the LLM with policy gradient optimization, searching for the optimal pruning mask in the probability space to enable optimization-based structural pruning using only forward passes.

## Method

### Overall Architecture

The input is a pre-trained LLM and a calibration dataset; the output is a pruned compact model. The method consists of three steps: (1) Initialize the Bernoulli distribution parameters corresponding to each prunability structure unit (attention heads, intermediate FFN dimensions); (2) Iteratively sample pruning masks, evaluate loss, and update the distribution parameters using policy gradients; (3) Determine the pruning scheme based on the final distribution.

### Key Designs

1. **Bernoulli parameterized pruning masks**:

    - **Function**: Convert discrete pruning decisions (keep/remove) into a continuous probabilistic optimization problem.
    - **Mechanism**: Associate each prunability unit (e.g., the $h$-th attention head in the $l$-th layer) with a Bernoulli distribution parameter $\theta_{l,h} \in [0,1]$, representing the probability of retaining that unit. The pruning mask $m_{l,h} \sim \text{Bernoulli}(\theta_{l,h})$ is a binary variable sampled from this distribution. Pruning decisions are indirectly optimized by optimizing the $\theta$ parameters.
    - **Design Motivation**: Relax the combinatorial optimization problem (the NP-hard optimal pruning problem) into a continuous optimization problem, making gradient-based methods applicable.

2. **Policy Gradient Estimator**:

    - **Function**: Compute gradients for pruning masks without backpropagating through the LLM.
    - **Mechanism**: Model the pruning process as a reinforcement learning problem—where the Bernoulli parameters $\theta$ represent the policy, the sampled mask $m$ represents the action, and the negative loss of the pruned model serves as the reward. The gradient is estimated using the REINFORCE algorithm: $\nabla_\theta J = \mathbb{E}_{m \sim p_\theta}[\nabla_\theta \log p_\theta(m) \cdot R(m)]$, where $R(m)$ can be computed with a single forward pass. Crucially, the gradient $\nabla_\theta \log p_\theta(m)$ does not involve LLM parameters and is calculated entirely within the low-dimensional parameter space of the Bernoulli distribution.
    - **Design Motivation**: Decouple "quality evaluation" (requires a forward pass) from "decision optimization" (takes place only in the $\theta$ space), bypassing the need for backpropagation through the LLM.

3. **Global Heterogeneous Pruning and Metric Initialization**:

    - **Function**: Automatically assign different pruning rates to different layers and optionally leverage existing heuristic metrics for initialization.
    - **Mechanism**: Since each structural unit is associated with an independent Bernoulli parameter, different layers naturally obtain distinct retention probabilities after optimization, realizing heterogeneous pruning. Additionally, the Bernoulli parameters can be initialized using existing metrics (e.g., gradient-based importance scores) rather than starting from a uniform distribution, accelerating optimization convergence.
    - **Design Motivation**: Uniform pruning rates (removing the same ratio per layer) perform poorly in practice because different layers exhibit varying redundancies. Metric initialization combines the prior knowledge of heuristic methods with the precise search capability of optimization methods.

### Loss & Training

Variance reduction techniques (baseline subtraction) are utilized to alleviate the high variance of REINFORCE. Multiple masks are sampled per iteration to estimate the gradient. The total number of iterations remains small (due to the low-dimensional parameter space), and the overall computational cost mainly stems from the multiple forward passes of the LLM.

## Key Experimental Results

### Main Results

| Model | Pruning Rate | Dataset | Ours PPL | SparseGPT PPL | Wanda PPL | LLM-Pruner PPL |
|------|--------|--------|---------|---------------|-----------|----------------|
| LLaMA-7B | 20% | WikiText2 | Outperforms Baseline | Baseline | Baseline | Baseline |
| LLaMA-7B | 50% | WikiText2 | Significantly Outperforms | Baseline | Baseline | Baseline |
| LLaMA-2-7B | 20% | C4 | Outperforms Baseline | Baseline | Baseline | Baseline |
| LLaMA-3-8B | 20% | WikiText2 | Consistently Outperforms | Baseline | Baseline | - |
| Mistral-7B | 20% | WikiText2 | Best | Baseline | Baseline | - |
| Vicuna-7B | 50% | WikiText2 | Clear Advantage | Baseline | Baseline | Baseline |

### Ablation Study

| Configuration | WikiText2 PPL | Description |
|------|--------------|------|
| Policy Gradient Optimization (Ours) | Best | Direct optimization of the target loss |
| Random Initialization + Optimization | Suboptimal | Slower convergence but achieves close final performance |
| Metric Initialization + Optimization | Best | Combines the dual advantages of prior knowledge + optimization |
| Metric Only (No Optimization) | Poor | Upper bound of heuristic methods |
| Uniform Pruning Rate | Poor | Different layers exhibit varying redundancies |
| Heterogeneous Pruning Rate (Ours) | Significantly Better | Automatically adapts to the redundancy of each layer |

### Key Findings
- Policy gradient optimization yields superior pruning schemes compared to all heuristic metrics, with the advantages being more pronounced at higher pruning rates (50%).
- Heterogeneous pruning (different pruning rates per layer) delivers significant gains over uniform pruning; optimization results reveal that layers at different depths indeed possess varying levels of redundancy.
- Metric initialization accelerates convergence, although starting from random initialization can also eventually converge to a close solution.
- Computational cost analysis demonstrates that the total number of forward passes required by the proposed method remains within an acceptable range (several hundred runs), which is far below that of fully backpropagation-based pruning.

## Highlights & Insights
- The core concept of **"bypassing backpropagation via policy gradients"** is brilliant: by treating pruning masks as RL actions, it utilizes the "backpropagation-free" property of REINFORCE to optimize combinatorial decisions. This approach can be extended to other scenarios requiring optimization of discrete structural decisions (e.g., NAS, feature selection).
- **Automating global heterogeneous pruning** addresses a crucial issue in practice: determining historical per-layer pruning ratios, which previously required manual tuning or additional search processes.
- The method displays high generalizability, and is theoretically applicable to the structural pruning of any Transformer architecture.

## Limitations & Future Work
- Although the high-variance issue of policy gradient is mitigated by baseline subtraction, it may still introduce optimization instability.
- Experiments were primarily conducted on 7B–13B models; the effectiveness and efficiency on larger scales (70B+) remain to be validated.
- Evaluation is confined to perplexity (PPL), omitting performance metrics on downstream tasks (e.g., reasoning, code generation).
- Integration with post-pruning recovery techniques like knowledge distillation remains unexplored.

## Related Work & Insights
- **vs SparseGPT**: SparseGPT efficiently solves unstructured pruning using second-order approximations but is less applicable to structural pruning; this work focuses on structural pruning from an optimization perspective.
- **vs Wanda**: Wanda utilizes weight-activation product as its pruning index, proving simple and efficient but remaining a heuristic method; the proposed optimization method can build upon Wanda's results as an initialization for further improvement.
- **vs LLM-Pruner**: LLM-Pruner also implements structural pruning but depends on gradient information (requiring backpropagation); this work completely bypasses backpropagation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The core idea of bypassing backpropagation via policy gradients for structural pruning is brilliant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple models and pruning rates, though lacking downstream task evaluations.
- Writing Quality: ⭐⭐⭐⭐ The methodology description is clear and the mathematical derivations are thorough.
- Value: ⭐⭐⭐⭐ Introduces a novel optimization paradigm for the structural pruning of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models](../../ICLR2026/reinforcement_learning/spg_sandwiched_policy_gradient_for_masked_diffusion_language_models.md)
- [\[ICLR 2026\] Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models](../../ICLR2026/reinforcement_learning/remix_reinforcement_routing_for_mixtures_of_loras_in_llm_finetuning.md)
- [\[ACL 2025\] MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](maporl_multi-agent_post-co-training_for_collaborative_large_language_models_with.md)
- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](../../NeurIPS2025/reinforcement_learning/mmada_multimodal_large_diffusion_language_models.md)
- [\[ICLR 2026\] ResT: Reshaping Token-Level Policy Gradients for Tool-Use Large Language Models](../../ICLR2026/reinforcement_learning/rest_reshaping_token-level_policy_gradients_for_tool-use_large_language_models.md)

</div>

<!-- RELATED:END -->
