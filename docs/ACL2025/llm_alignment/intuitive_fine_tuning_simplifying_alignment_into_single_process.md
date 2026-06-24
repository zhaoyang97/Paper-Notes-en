---
title: >-
  [Paper Note] Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process
description: >-
  [ACL 2025][LLM Alignment][SFT] By unifying the analysis of SFT and Preference Optimization (PO) through an MDP framework, SFT is identified as a special case of PO where preference estimation and transition optimization are insufficient. IFT (Intuitive Fine-Tuning) is proposed to utilize temporal residual connections, allowing the model to achieve alignment performance comparable to or even surpassing the SFT+PO pipeline without requiring preference data.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "SFT"
  - "Preference Optimization"
  - "Unified Alignment"
  - "MDP"
  - "Temporal Residual Connection"
date: 2026-05-08
content_hash: ed65922422d5120c
---

# Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process

**Conference**: ACL 2025  
**arXiv**: [2405.11870](https://arxiv.org/abs/2405.11870)  
**Code**: [https://github.com/TsinghuaC3I/Intuitive-Fine-Tuning](https://github.com/TsinghuaC3I/Intuitive-Fine-Tuning)  
**Area**: LLM Alignment  
**Keywords**: SFT, Preference Optimization, Unified Alignment, MDP, Temporal Residual Connection

## TL;DR
By unifying the analysis of SFT and Preference Optimization (PO) through an MDP framework, SFT is identified as a special case of PO where preference estimation and transition optimization are insufficient. IFT (Intuitive Fine-Tuning) is proposed to utilize temporal residual connections, allowing the model to achieve alignment performance comparable to or even surpassing the SFT+PO pipeline without requiring preference data.

## Background & Motivation

**Background**: LLM alignment typically consists of two stages: SFT to learn formats and PO (e.g., DPO/PPO) to align preferences. These two stages are executed sequentially due to differences in paradigms (loss functions, data formats, and auxiliary models).

**Limitations of Prior Work**: SFT is highly efficient but yields limited performance, whereas PO achieves better results but requires annotated preference data and auxiliary models. Their sequential combination fails to leverage their synergistic advantages.

**Key Challenge**: When predicting each token, SFT uses the prefix of the target answer as the prior state. However, this deviates from the model's own distribution, leading to a biased estimation of model preferences and insufficient transition optimization.

**Core Idea**: By incorporating temporal residual connections $\hat{s_i^\theta} = (1-\lambda)s_i^* + \lambda\pi_\theta(s_{i-1}^*)$ to mix the model's own predictions into the prior state, the model obtains a much closer estimation of true preferences while retaining the data efficiency of SFT.

## Method

### Overall Architecture
IFT introduces a lightweight modification to standard SFT: at each token position, the model's own predicted embedding for the current position (passed via temporal residual connections) is mixed with the ground truth embedding as the prior state. This allows the model to perceive its own generation preferences during training, instead of only seeing the "perfect human-written answers."

### Key Designs

1. **Unified Analysis from an MDP Perspective**:

    - Function: Unveil the fundamental connections and differences between SFT and PO.
    - Mechanism: Define preference estimation $\mathcal{P}(\rho_0)$ as the model's entire response preference to instructions, and transition optimization as aligning the transition matrix.
    - Key Findings: When predicting the $n$-th token, SFT uses $s_{n-1}^*$ (the intermediate state of the human answer) as the prior. In actual deployment, however, the model uses $s_{n-1}^\theta$ (its own generated intermediate state), leading to an overestimation of model preferences.
    - Design Motivation: This overestimation makes preference alignment in SFT insufficient, necessitating corrective PO afterwards.

2. **Intuitive Preference Estimation**:

    - Function: Obtain preference estimations close to those of PO without requiring preference data.
    - Mechanism: Introduce distribution perturbation $\delta_\theta(s_i^*) = (1-\lambda)s_i^* + \lambda\pi_\theta(s_{i-1}^*)$ such that the preference estimation is expressed as $\hat{\mathcal{P}_\theta} = [(1-\lambda)\mathcal{P}_\theta^{sft} + \lambda\mathcal{P}_\theta^{truly}]$.
    - Design Motivation: Setting $\lambda=0$ degenerates to SFT, while $\lambda=1$ approaches true preference, allowing a flexible trade-off between efficiency and accuracy.

3. **Dynamic Relation Propagation**:

    - Function: Allow the prediction accuracy of early tokens to influence the optimization of subsequent tokens.
    - Mechanism: Reconstruct the loss function using a differentiable cumulative sum, enabling early token predictions to affect subsequent gradients via the residual connection.
    - Design Motivation: Emulate the effect of online sampling in PO—if the model deviates from the correct path at some point, all subsequent tokens should account for this deviation.

### Loss & Training
$\mathcal{L}_{IFT} = \mathbb{E}[-\sum_{n=0}^{N}\sum_{i=n}^{N}\log\mathcal{T}_\theta(a_i^*, \delta_\theta(s_i^*))]$. Only positive samples are used (the same data format as SFT), with no requirement for preference annotations, reference models, or online sampling. The only hyperparameter is $\lambda$.

## Key Experimental Results

### Main Results

| Method | AlpacaEval LC% | MT-Bench | GSM8K | Data Requirement |
|------|---------------|----------|-------|---------|
| SFT | Lower | Lower | Baseline | Only Positive Samples |
| SFT+DPO | Higher | Higher | Higher | Positive + Preference Pairs |
| IFT (Ours) | Comparable / Better | Comparable / Better | **Better** | Only Positive Samples |

### Ablation Study

| Configuration | Key Findings | Description |
|------|---------|------|
| $\lambda$ Sensitivity | 0.3-0.5 is optimal | Too small degenerates to SFT, too large causes instability |
| Cumulative Sum vs. Simple Sum | Cumulative sum is significantly better | Dynamic relation propagation is effective |
| Frozen Lake | IFT learns the optimal policy | SFT falls into suboptimal path on 4x4 grid |

### Key Findings
- IFT achieves performance comparable to SFT+PO using only positive samples, eliminating the need for preference annotations and offering extremely high data efficiency.
- The advantages are particularly pronounced in reasoning and fact-following tasks, which are highly sensitive to the "ability to recover from deviation."
- In the Frozen Lake toy game, SFT always uses the ground truth prior and thus learns a suboptimal path, whereas IFT successfully learns the optional policy.
- The computational overhead of the temporal residual connection is minimal, with negligible impact on training speed.

## Highlights & Insights
- The unified analysis under the MDP framework clearly uncovers the limitations of SFT, making it instantly transparent why SFT is a "special case" of PO.
- The temporal residual connection is elegant and simple, introducing almost no computational overhead while theoretically shifting preference estimation from biased to unbiased.

## Limitations & Future Work
- The choice of $\lambda$ may require task-specific tuning.
- The theoretical analysis is based on the assumption of greedy decoding; conclusions might require modification under sampling decoding.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The unified MDP perspective and the design of IFT are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive multi-task validation accompanied by toy experiment explanations.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivation.
- Value: ⭐⭐⭐⭐⭐ Simplifying the alignment pipeline holds immense practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models](prmbench_a_fine-grained_and_challenging_benchmark_for_process-level_reward_model.md)
- [\[ACL 2025\] Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation](retrieval-augmented_fine-tuning_with_preference_optimization_for_visual_program_.md)
- [\[ACL 2025\] M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](m2s_multiturn_to_singleturn_jailbreak_in.md)
- [\[ACL 2025\] Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization](fine-grained_video_dubbing_duration_alignment_with_segment_supervised_preference.md)
- [\[ICLR 2026\] Anchored Supervised Fine-Tuning](../../ICLR2026/llm_alignment/anchored_supervised_fine-tuning.md)

</div>

<!-- RELATED:END -->
