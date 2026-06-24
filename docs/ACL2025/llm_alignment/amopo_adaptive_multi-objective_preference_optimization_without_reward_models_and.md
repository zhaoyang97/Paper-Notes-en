---
title: >-
  [Paper Note] AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models
description: >-
  [ACL 2025 (Findings)][LLM Alignment][Multi-objective preference alignment] This paper proposes the AMoPO framework, which achieves dimension-aware adaptive weight allocation by modeling the generation space as a Gaussian distribution. It completes multi-objective preference alignment without relying on reward models or reference models, outperforming the state-of-the-art (SOTA) by 28.5% on the HelpSteer2 dataset, and validating scalability on 7B, 14B, and 32B models.
tags:
  - "ACL 2025 (Findings)"
  - "LLM Alignment"
  - "Multi-objective preference alignment"
  - "adaptive weights"
  - "Gaussian distribution"
  - "reward-model-free"
  - "preference optimization"
date: 2026-05-08
content_hash: 5e9ae0abdb31c085
---

# AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2506.07165](https://arxiv.org/abs/2506.07165)  
**Code**: [https://github.com/Javkonline/AMoPO](https://github.com/Javkonline/AMoPO)  
**Area**: LLM Alignment  
**Keywords**: Multi-objective preference alignment, adaptive weights, Gaussian distribution, reward-model-free, preference optimization

## TL;DR

This paper proposes the AMoPO framework, which achieves dimension-aware adaptive weight allocation by modeling the generation space as a Gaussian distribution. It completes multi-objective preference alignment without relying on reward models or reference models, outperforming the state-of-the-art (SOTA) by 28.5% on the HelpSteer2 dataset, and validating scalability on 7B, 14B, and 32B models.

## Background & Motivation

Preference alignment of Large Language Models (LLMs) is a current research hotspot, aiming to align model outputs with human preferences across multiple dimensions (e.g., helpfulness, correctness, instruction following). However, existing methods suffer from two core problems:

**Difficulty in multi-dimension balancing**: Existing multi-objective alignment methods (such as MODPO) struggle to effectively balance the trade-offs between different preference dimensions. Simple fixed weights or linear combinations fail to capture the priority differences of varied dimensions across different samples—some samples may require more focus on correctness, while others may require more focus on helpfulness.

**Computation overhead from auxiliary model dependency**: Methods like DPO require a reference model, while RLHF requires a reward model. These auxiliary models not only increase computational and storage costs but may also introduce additional alignment errors.

The key challenge lies in how to dynamically balance multiple preference dimensions without introducing additional models. The key insight of AMoPO is to **utilize dimension-aware generation metrics as implicit rewards**, combining a **Gaussian distribution-based adaptive weighting mechanism** to allow the model to automatically adjust the optimization priorities of different dimensions based on the current generation quality.

## Method

### Overall Architecture

The training pipeline of AMoPO:
- **Input**: Preference pair data (chosen/rejected) containing multi-dimensional preference annotations, where each data sample carries dimension scores such as helpfulness, correctness, and instruction following.
- **Data Processing**: For each data sample, construct 4 groups (number of dimensions + 1) of chosen/rejected pairs, with each group using a different dimension-aware prompt to guide the model to focus on a specific dimension.
- **Forward Pass**: Compute the log probabilities of all 8 sequences (4 dimensions $\times$ 2 chosen/rejected) in a single forward pass.
- **Loss Computation**: Compute losses for each dimension based on ORPO/SimPO, generate adaptive weights via Gaussian sampling, and obtain the final loss through weighted summation.
- **Output**: The aligned LLM.

### Key Designs

1. **Dimension-Aware Prompting**:

    - **Function**: Construct specialized system prompts for each preference dimension to guide the model in understanding the current optimization direction.
    - **Mechanism**: For the "no specific dimension" general version, a comprehensive prompt is used; for specific dimensions (e.g., helpfulness), evaluation criterion descriptions (on a 1-5 scale) are injected, and the score of that dimension is embedded into the prompt: `"focus on the {dimension} dimension... based on the evaluation value of {score}"`
    - **Design Motivation**: By using prompts, the model is made aware of the dimension currently being optimized during generation, achieving implicit dimension awareness without requiring additional reward models to evaluate the quality of each dimension.

2. **Reference-Model-Free Loss Function Based on ORPO/SimPO**:

    - **Function**: Compute preference losses without relying on a reference model.
    - **Mechanism**: The ORPO loss combines SFT loss and odds ratio loss:
    $\mathcal{L}_{ORPO} = -\log p_{chosen} + \beta \cdot (-\log\sigma(\log\frac{p_{chosen}/(1-p_{chosen})}{p_{rejected}/(1-p_{rejected})}))$
      The single-dimension loss of SimPO is:
    $\mathcal{L}_{SimPO} = -\log\sigma(\beta \cdot (\log p_{chosen} - \log p_{rejected} - \gamma/\beta))$
    - **Design Motivation**: ORPO directly embeds preference signals into the language modeling objective, and SimPO achieves reference-model-free alignment through length-normalized log probability differences.

3. **Gaussian Adaptive Weighting**:

    - **Function**: Dynamically allocate loss weights according to the generation quality of each dimension in the current batch.
    - **Mechanism**:
      1. Extract the token-level probabilities of the chosen and rejected sequences for each dimension.
      2. Calculate the mean $\mu$ and standard deviation $\sigma$ of non-zero token probabilities for each sequence.
      3. Sample from the Gaussian distribution $\mathcal{N}(\mu, \sigma)$ to obtain the quality metric of the sequence.
      4. Add the sampled values of chosen and rejected sequences to obtain the raw weight for each dimension.
      5. Apply softmax normalization to all dimension weights.
      6. Multiply the normalized weights by the corresponding dimension losses.
    - **Design Motivation**: Dimensions with poorer generation quality (where token probability distribution is more dispersed) receive higher weights, guiding the model to prioritize optimizing weak dimensions; the introduction of stochastic sampling increases exploration and avoids trapping the model in fixed-weight patterns.

### Loss & Training

The final loss is the sum of weighted losses across all dimensions, optionally incorporating an SFT auxiliary loss:

$$\mathcal{L}_{total} = \sum_{d \in D} w_d \cdot \mathcal{L}_d + \gamma_{ftx} \cdot \mathcal{L}_{SFT}$$

where $w_d$ is the Gaussian adaptive weight, $D$ = {no_object, helpfulness, correctness, instruction_following}, and $\gamma_{ftx}$ controls the proportion of the SFT loss.

Training is based on the LLaMA-Factory framework, utilizing DeepSpeed for distributed training.

## Key Experimental Results

### Main Results

| Model | Method | HelpSteer2 Total Score | Gain |
|------|------|-----------------|------|
| Qwen2.5-7B | DPO | baseline | - |
| Qwen2.5-7B | ORPO | baseline+α | - |
| Qwen2.5-7B | SimPO | baseline+β | - |
| Qwen2.5-7B | MODPO | baseline+γ | - |
| Qwen2.5-7B | **AMoPO** | **best** | **+28.5%** |
| Qwen2.5-14B | AMoPO | - | Continuous improvement |
| Qwen2.5-32B | AMoPO | - | Continuous improvement |

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| AMoPO (full) | Optimal | Full framework, Gaussian adaptive weight + multi-dimension |
| Remove adaptive weight (equal weight) | Degradation | Fixed weights fail to adapt to the dimension priorities of different samples |
| Reduce number of dimensions | Degradation | Demonstrates the importance of multi-dimensional coverage |
| Single-dimension optimization only | Significant degradation | Validates the necessity of multi-objective optimization |
| 7B → 14B → 32B | Continuous improvement | Proves the scaling ability of AMoPO |

### Key Findings
- AMoPO outperforms methods requiring auxiliary models (e.g., DPO) without using any reward models or reference models.
- The Gaussian adaptive weighting mechanism dynamically adjusts weights of each dimension during training, automatically discovering and prioritizing the optimization of weaker dimensions.
- The framework exhibits good scalability, with performance continuously improving from 7B to 32B models.
- Dimension-aware prompt injection significantly enhances the independent optimization of each dimension.

## Highlights & Insights

- **Using Gaussian sampling for weight allocation** is a novel trick: it leverages the statistical features of token probability distributions (mean and variance) to reflect generation quality, introducing randomness through sampling to avoid rigid weights. This idea can be transferred to other multi-objective learning scenarios.
- **Dimension-aware prompt** encodes preference dimension information into the input, enabling the model to focus on specific dimensions based on the prompts during inference, which achieves a certain degree of controllable generation.
- The overall framework is built on LLaMA-Factory, making it easy to reproduce and extend.

## Limitations & Future Work

- Currently validated only on the HelpSteer2 dataset with preference dimensions fixed to helpfulness/correctness/instruction_following; generalization to more dimensions needs further validation.
- Gaussian sampling has unstable statistics when the batch size is small, potentially causing significant fluctuations in weights.
- Correlations and conflict relationships between dimensions are not explicitly modeled; simple independent Gaussian variables may miss inter-dimension interactions.
- The selection of dimension-aware prompts during inference relies on user specification, lacking an automatic dimension selection mechanism.

## Related Work & Insights

- **vs DPO**: DPO requires a reference model, while AMoPO achieves reference-model-free training through ORPO/SimPO variants, reducing GPU memory footprint by half.
- **vs MODPO**: MODPO uses fixed linear combinations for multi-objective mixing, whereas AMoPO achieves dynamic balance through Gaussian adaptive weights, yielding significantly superior performance.
- **vs ORPO**: The original ORPO is single-objective. AMoPO extends it to a multi-objective version, incorporating dimension awareness and adaptive weights.

## Rating

- Novelty: ⭐⭐⭐⭐ Gaussian sampling for adaptive weight allocation is a novel entry point, though the overall framework is highly combinatory.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete multi-scale model experiments and ablation studies, but validated on only a single dataset.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and complete description of the method.
- Value: ⭐⭐⭐⭐ Possesses practical value for multi-objective alignment scenarios, with code open-sourced and reproducible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](automixalign_adaptive_data_mixing.md)
- [\[ACL 2025\] LPOI: Listwise Preference Optimization for Vision Language Models](lpoi_listwise_preference_optimization_for_vision_language_models.md)
- [\[ACL 2025\] T-REG: Preference Optimization with Token-Level Reward Regularization](t-reg_preference_optimization_with_token-level_reward_regularization.md)
- [\[ICML 2026\] Steerable Cultural Preference Optimization of Reward Models](../../ICML2026/llm_alignment/steerable_cultural_preference_optimization_of_reward_models.md)
- [\[ACL 2025\] Robust Preference Optimization via Dynamic Target Margins](robust_preference_optimization_via_dynamic_target_margins.md)

</div>

<!-- RELATED:END -->
