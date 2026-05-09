---
title: >-
  [Paper Note] CAP: Controllable Alignment Prompting for Unlearning in LLMs
description: >-
  [ACL 2026][Reinforcement Learning][LLM unlearning] This paper proposes the CAP framework, which trains a lightweight SLM to generate controllable prompt prefixes that guide a frozen LLM to selectively forget target knowledge. Without modifying model parameters, CAP achieves reversible and transferable knowledge unlearning in LLMs.
tags:
  - ACL 2026
  - Reinforcement Learning
  - LLM unlearning
  - prompt-driven
  - controllable alignment
  - knowledge erasure
date: 2026-05-08
content_hash: 4c4d0fb66cb84bed
---

# CAP: Controllable Alignment Prompting for Unlearning in LLMs

**Conference**: ACL 2026
**arXiv**: [2604.21251](https://arxiv.org/abs/2604.21251)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: LLM unlearning, prompt-driven, reinforcement learning, controllable alignment, knowledge erasure

## TL;DR

This paper proposes the CAP framework, which trains a lightweight SLM to generate controllable prompt prefixes that guide a frozen LLM to selectively forget target knowledge. Without modifying model parameters, CAP achieves reversible and transferable knowledge unlearning in LLMs.

## Background & Motivation

**Background**: LLMs trained on unfiltered corpora inevitably retain sensitive information. Regulations such as GDPR mandate selective knowledge unlearning. Existing approaches primarily achieve unlearning by modifying model parameters.

**Limitations of Prior Work**: (1) Retraining- and gradient-based methods incur high computational costs; (2) unlearning boundaries are uncontrollable, often causing general performance degradation; (3) these methods strictly require access to model weights and are thus inapplicable to closed-source models; (4) existing non-invasive methods rely on empirically designed prompts and lack a systematic end-to-end training framework.

**Key Challenge**: Parameter-modification methods are direct but costly and irreversible, whereas parameter-free methods (e.g., prompt engineering) are lightweight but lack controllability and systematic optimization.

**Goal**: To design an end-to-end prompt-driven unlearning framework that achieves precise, controllable, and reversible knowledge erasure without modifying LLM parameters.

**Key Insight**: The unlearning problem is reformulated as an inference-time control problem — a lightweight SLM is trained as a policy network to generate input-conditioned control prefixes that steer the behavior of the frozen LLM.

**Core Idea**: The SLM generates two types of prompt prefixes for each input query (forgetting prompts and retention prompts). Through a variational information bottleneck (VIB) contrastive objective and Beam PPO reinforcement learning optimization, the LLM is guided to suppress target knowledge while preserving general capabilities.

## Method

### Overall Architecture

CAP consists of two stages: (1) prompt generator optimization — an SLM is trained via RL to generate effective forgetting/retention prompt prefixes; (2) inference stage — the frozen SLM generates prompt prefixes, which are combined with Self-Check instructions to guide the LLM's final output.

### Key Designs

1. **Dual Prompt Prefix Mechanism (Forgetting + Retention)**:

    - Function: Guides the LLM to suppress target knowledge and preserve general capabilities, respectively.
    - Mechanism: The SLM generates $n$ forgetting prompt candidates $\mathcal{P}_f^k$ and $n$ retention prompt candidates $\mathcal{P}_r^k$ for each query. Each candidate is concatenated with the query and fed to the frozen LLM, yielding a forgetting answer set and a retention answer set.
    - Design Motivation: The dual-prompt design decouples forgetting and retention into two independently optimizable directions, avoiding the conflict between the two objectives within a single prompt.

2. **Variational Information Bottleneck Contrastive Objective (VIB)**:

    - Function: Guides the optimization directions of forgetting and retention from an information-theoretic perspective.
    - Mechanism: For the forgetting branch, the mutual information between LLM outputs and labels is minimized (via a variational upper bound using KL divergence); for the retention branch, mutual information is maximized (via an InfoNCE lower bound). The two branches are jointly optimized, with $\beta$ controlling the trade-off.
    - Design Motivation: Modeling forgetting (information compression) and retention (information preservation) directly at the information-theoretic level is more theoretically grounded than heuristic reward-based approaches.

3. **Beam PPO Reinforcement Learning Optimization**:

    - Function: Enhances the stability and diversity of policy exploration.
    - Mechanism: A beam of $k$ anchor policies is maintained. The current policy $\pi_\theta$ is regularized by the minimum KL divergence with respect to all anchor policies, preventing the local optima and policy collapse associated with standard PPO.
    - Design Motivation: Standard PPO lacks stability in prompt generation; Beam PPO provides broader coverage of the parameter space through multi-path exploration.

### Loss & Training

The total reward function is $\mathcal{R} = \lambda_{VIB} \cdot \mathcal{R}_{VIB} + \lambda_{label} \cdot \mathcal{R}_{label} + \lambda_{len} \cdot \mathcal{R}_{len}$, where the VIB reward guides information compression/preservation, the label reward evaluates forgetting/retention alignment, and length regularization encourages concise prompts close to the target length. The B-PPO objective augments the standard PPO clipping loss with multi-anchor KL regularization.

## Key Experimental Results

### Main Results

| Model | Method | RWKU ASG↓ | WMDP Bio Acc↓ | MMLU Acc↑ |
|------|------|----------|--------------|----------|
| Zephyr-7B | Original | 63.0 | 63.7 | 54.1 |
| Zephyr-7B | NPO | 28.9 | 43.1 | 48.6 |
| Zephyr-7B | ICUL | 30.3 | 44.9 | 44.5 |
| Zephyr-7B | **CAP** | **6.2** | **24.8** | **51.5** |
| GPT-4.1 | ICUL | 36.7 | 38.6 | 81.5 |
| GPT-4.1 | **CAP** | **7.5** | **35.9** | **80.6** |
| Claude-Sonnet-4 | **CAP** | **7.4** | **30.1** | **84.2** |

### Ablation Study

| Configuration | Forgetting Acc↓ | Retention Acc↑ | Note |
|------|----------|----------|------|
| w/o IB + Standard PPO | 37.5 | 49.8 | No structured reward |
| + IB + B-PPO (Full CAP) | 24.8 | 51.5 | Best balance |
| Forgetting VIB only | 25.6 | 44.7 | Retention performance degraded |
| Retention VIB only | 38.6 | 52.2 | Forgetting capability weakened |
| Random selection vs. Self-Check | 26.2 / 24.8 | 48.5 / 51.5 | Self-Check provides stability fine-tuning |

### Key Findings
- CAP reduces ASG from 63.0 to 6.2 on Zephyr-7B for generative tasks, substantially outperforming all baselines.
- On discriminative tasks, CAP significantly reduces WMDP accuracy while preserving MMLU performance close to the original level.
- CAP transfers seamlessly to closed-source models (GPT-4.1, Claude-Sonnet-4, DeepSeek-V3, etc.) using only discrete prompts.
- Beam size $k=4$, candidate count $n=3$, and maximum prompt length $L=16$ constitute the optimal hyperparameter configuration.
- Various SLMs (Qwen3-0.6B, Qwen2.5-0.5B, Gemma3-1B) can all effectively guide unlearning, demonstrating the model-agnostic nature of the approach.

## Highlights & Insights
- The core innovation lies in shifting unlearning from parameter space to output space, achieving reversible unlearning via discrete prompts — removing the prompt generator restores the original model.
- The VIB contrastive objective elegantly unifies forgetting (compression) and retention (preservation) from an information-theoretic perspective, surpassing heuristic rewards.
- The Beam PPO improvement over standard PPO has general utility beyond unlearning tasks.
- Hidden-state visualizations intuitively demonstrate how prompts redirect internal activations from knowledge regions to safe/refusal regions.

## Limitations & Future Work
- The two-stage inference (SLM prefix generation + LLM output generation) introduces marginal latency overhead.
- Generated control prefixes consume a small portion of the LLM's context window.
- The SLM is fixed to Qwen3-0.6B in the main experiments; although other SLMs are shown to be effective, optimal SLM selection remains underexplored.
- Robustness under adversarial attacks, while superior to baselines, is not yet perfect.

## Related Work & Insights
- **vs. LLMU/NPO**: These methods require modifying LLM parameters and are inapplicable to closed-source models; CAP requires no parameter modification.
- **vs. ICUL**: ICUL drives unlearning via in-context learning but lacks negative samples and adapts poorly to adversarial distributions; CAP achieves stronger generalization through RL-optimized prompts.
- **vs. SPUL**: SPUL employs soft prompt tuning but still requires gradient backpropagation; CAP uses discrete prompts and requires no access to LLM gradients.
- **vs. Pawelczyk et al.**: Their classifier-based non-invasive approach depends on classifier accuracy; CAP's end-to-end optimization is more reliable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ End-to-end prompt-driven unlearning paradigm with elegant VIB + Beam PPO design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 7 LLMs (including closed-source), multiple datasets, comprehensive ablations, and sensitivity analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological exposition with complete theoretical derivations.
- Value: ⭐⭐⭐⭐⭐ Significant practical value for the knowledge unlearning problem in closed-source LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reasoning Boosts Opinion Alignment in LLMs](../../ICLR2026/reinforcement_learning/reasoning_boosts_opinion_alignment_in_llms.md)
- [\[ACL 2026\] Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF](reverse_constitutional_ai_a_framework_for_controllable_toxic_data_generation_via.md)
- [\[ACL 2026\] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment](healing_entropy_collapse_enhancing_exploration_in_few-shot_rlvr_via_hybrid-domai.md)
- [\[ACL 2026\] Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning](easy_samples_are_all_you_need_self-evolving_llms_via_data-efficient_reinforcemen.md)
- [\[ACL 2026\] RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization](rl-plus_countering_capability_boundary_collapse_of_llms_in_reinforcement_learnin.md)

</div>

<!-- RELATED:END -->
