---
title: >-
  [Paper Note] CAP: Controllable Alignment Prompting for Unlearning in LLMs
description: >-
  [ACL 2026][LLM Safety][LLM Unlearning] The authors propose the CAP framework, which trains a lightweight SLM to generate controllable prompt prefixes that guide a frozen LLM to selectively unlearn target knowledge. This…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "LLM Unlearning"
  - "Prompt-driven"
  - "Reinforcement Learning"
  - "Controllable Alignment"
  - "Knowledge Elimination"
date: 2026-05-08
content_hash: 466d4b02ae889320
---

# CAP: Controllable Alignment Prompting for Unlearning in LLMs

**Conference**: ACL 2026  
**arXiv**: [2604.21251](https://arxiv.org/abs/2604.21251)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: LLM Unlearning, Prompt-driven, Reinforcement Learning, Controllable Alignment, Knowledge Elimination

## TL;DR

The authors propose the CAP framework, which trains a lightweight SLM to generate controllable prompt prefixes that guide a frozen LLM to selectively unlearn target knowledge. This approach achieves reversible and transferable LLM unlearning without modifying model parameters.

## Background & Motivation

**Background**: LLMs are trained on unfiltered corpora and inevitably retain sensitive information. Regulations such as GDPR require selective knowledge unlearning. Existing methods primarily achieve this by modifying model parameters.

**Limitations of Prior Work**: (1) Retraining and gradient-based methods are computationally expensive; (2) Unlearning boundaries are uncontrollable, often leading to overall performance degradation; (3) These methods strictly depend on access to model weights, making them unavailable for closed-source models; (4) Existing non-intrusive methods rely on empirical prompt design and lack a systematic end-to-end training framework.

**Key Challenge**: While parameter-modifying methods are direct, they are costly and irreversible. Conversely, non-intrusive methods like prompt engineering are lightweight but lack controllability and systematic optimization.

**Goal**: Design an end-to-end prompt-driven unlearning framework that achieves precise, controllable, and reversible knowledge unlearning without modifying LLM parameters.

**Key Insight**: The unlearning problem is transformed into an inference-time control problem—training a lightweight SLM as a policy network to generate input-conditioned control prefixes that guide the output behavior of a frozen LLM.

**Core Idea**: The SLM generates two types of prompt prefixes (forget prompts and retain prompts) for each input query. These are optimized using a Variational Information Bottleneck (VIB) contrastive objective and Beam PPO reinforcement learning, enabling the LLM to suppress target knowledge while maintaining general capabilities.

## Method

### Overall Architecture

CAP consists of two phases: (1) Prompt generator optimization—using RL to train an SLM to generate effective forget/retain prompt prefixes; (2) Inference phase—the frozen SLM generates prompt prefixes which, combined with a Self-Check instruction, guide the final output of the LLM.

### Key Designs

1. **Dual Prompt Prefix Mechanism (Forget + Retain)**:

    - **Function**: Guides the LLM to suppress target knowledge and maintain general capabilities, respectively.
    - **Mechanism**: For each query, the SLM generates $n$ forget prompt candidates $\mathcal{P}_f^k$ and $n$ retain prompt candidates $\mathcal{P}_r^k$. These are concatenated with the query and input into the frozen LLM to obtain sets of forget and retain answers.
    - **Design Motivation**: The dual-prompt design decouples unlearning and retention into two independently optimizable directions, avoiding conflicts between the two within a single prompt.

2. **Variational Information Bottleneck Contrastive Objective (VIB)**:

    - **Function**: Guides the optimization directions of unlearning and retention through an information-theoretic approach.
    - **Mechanism**: The unlearning branch minimizes the mutual information between the LLM output and the label (via a variational upper bound KL divergence), while the retention branch maximizes mutual information (via an InfoNCE lower bound). Both branches are jointly optimized, with $\beta$ controlling the trade-off.
    - **Design Motivation**: Modeling unlearning (information compression) and retention (information preservation) at the information-theoretic level is more theoretically grounded than purely heuristic reward-based methods.

3. **Beam PPO Reinforcement Learning Optimization**:

    - **Function**: Enhances the stability and diversity of policy exploration.
    - **Mechanism**: The framework maintains a beam of $k$ anchor policies. The current policy $\pi_\theta$ is regularized by the minimum KL divergence relative to all anchor policies, preventing local optima and policy collapse common in standard PPO.
    - **Design Motivation**: Standard PPO lacks stability in prompt generation; Beam PPO provides greater parameter space coverage through multi-path exploration.

### Loss & Training

The total reward function is $\mathcal{R} = \lambda_{VIB} \cdot \mathcal{R}_{VIB} + \lambda_{label} \cdot \mathcal{R}_{label} + \lambda_{len} \cdot \mathcal{R}_{len}$, where the VIB reward guides information compression/retention, the label reward evaluates unlearning/retention alignment, and length regularization encourages concise prompts close to an ideal length. The B-PPO objective function adds multi-anchor KL regularization to the standard PPO clip loss.

## Key Experimental Results

### Main Results

| Model | Method | RWKU ASG↓ | WMDP Bio Acc↓ | MMLU Acc↑ |
|------|------|----------|--------------|----------|
| Zephyr-7B | Original | 63.0 | 63.7 | 54.1 |
| Zephyr-7B | NPO | 28.9 | 43.1 | 48.6 |
| Zephyr-7B | ICUL | 30.3 | 44.9 | 44.5 |
| Zephyr-7B | **Ours** | **6.2** | **24.8** | **51.5** |
| GPT-4.1 | ICUL | 36.7 | 38.6 | 81.5 |
| GPT-4.1 | **Ours** | **7.5** | **35.9** | **80.6** |
| Claude-Sonnet-4 | **Ours** | **7.4** | **30.1** | **84.2** |

### Ablation Study

| Configuration | Forget Acc↓ | Retain Acc↑ | Description |
|------|----------|----------|------|
| W/o IB + Std PPO | 37.5 | 49.8 | No structured reward |
| + IB + B-PPO (Full CAP) | 24.8 | 51.5 | Best balance |
| Forget VIB only | 25.6 | 44.7 | Retention performance compromised |
| Retain VIB only | 38.6 | 52.2 | Unlearning capability weakened |
| Random selection vs Self-Check | 26.2/24.8 | 48.5/51.5 | Self-Check fine-tunes stability |

### Key Findings
- CAP reduces the ASG in generative tasks from 63.0 to 6.2 (Zephyr-7B), significantly outperforming all baselines.
- In discriminative tasks, CAP substantially lowers WMDP accuracy while maintaining MMLU performance near original levels.
- CAP transfers seamlessly to closed-source models (GPT-4.1, Claude-Sonnet-4, DeepSeek-V3, etc.) using only discrete prompts.
- The optimal hyperparameter configuration was found to be beam size $k=4$, candidate count $n=3$, and maximum prompt length $L=16$.
- Different SLMs (Qwen3-0.6B, Qwen2.5-0.5B, Gemma3-1B) can effectively guide unlearning, demonstrating the model-agnostic nature of the method.

## Highlights & Insights
- Shifting unlearning from the parameter space to the output space via discrete prompts is the core innovation—removing the prompt generator restores the original model, ensuring reversibility.
- The VIB contrastive objective elegantly unifies unlearning (compression) and retention (preservation) from an information-theoretic perspective.
- The improvements of Beam PPO over standard PPO have general utility beyond unlearning tasks.
- Hidden state visualizations intuitively demonstrate how prompts redirect internal activations from knowledge regions to safety/refusal regions.

## Limitations & Future Work
- The two-stage inference (SLM prefix generation + LLM output generation) introduces marginal latency overhead.
- The generated control prefixes occupy a small portion of the LLM's context window.
- While Qwen3-0.6B was used as the primary SLM and others were validated, the selection of the optimal SLM has not been fully explored.
- Robustness under adversarial attacks is better than baselines but still not perfect.

## Related Work & Insights
- **vs LLMU/NPO**: These methods require modifying LLM parameters and are inapplicable to closed-source models; CAP requires no parameter modification.
- **vs ICUL**: ICUL uses in-context learning for unlearning but lacks negative samples, showing poor adaptability to adversarial distributions; CAP's RL-optimized prompts offer stronger generalization.
- **vs SPUL**: SPUL uses soft-prompt tuning but still requires gradient backpropagation; CAP uses discrete prompts without requiring access to LLM gradients.
- **vs Pawelczyk et al.**: They propose a classifier-based non-intrusive method but depend on classifier accuracy; CAP's end-to-end optimization is more reliable.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ End-to-end prompt-driven unlearning paradigm with elegant VIB + Beam PPO design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 7 LLMs (including closed-source), multiple datasets, comprehensive ablation, and sensitivity analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Methodological exposition is clear with complete theoretical derivation.
- **Value**: ⭐⭐⭐⭐⭐ Significant practical value for the problem of unlearning in closed-source LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Inoculation Prompting: Eliciting Traits from LLMs during Training Can Suppress Them at Test-Time](../../ICLR2026/llm_safety/inoculation_prompting_eliciting_traits_from_llms_during_training_can_suppress_th.md)
- [\[ACL 2026\] Can Persona-Prompted LLMs Emulate Subgroup Values? An Empirical Analysis of Generalisability and Fairness in Cultural Alignment](can_persona-prompted_llms_emulate_subgroup_values_an_empirical_analysis_of_gener.md)
- [\[ICML 2026\] Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility](../../ICML2026/llm_safety/multilingual_unlearning_in_llms_transfer_dynamics_and_reversibility.md)
- [\[CVPR 2026\] SineProject: Machine Unlearning for Stable Vision–Language Alignment](../../CVPR2026/llm_safety/sineproject_machine_unlearning_for_stable_vision_language_alignment.md)
- [\[CVPR 2026\] Unsafe2Safe: Controllable Image Anonymization for Downstream Utility](../../CVPR2026/llm_safety/unsafe2safe_controllable_image_anonymization_for_downstream_utility.md)

</div>

<!-- RELATED:END -->
