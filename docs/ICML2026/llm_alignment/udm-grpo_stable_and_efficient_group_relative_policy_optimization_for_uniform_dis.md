---
title: >-
  [Paper Note] UDM-GRPO: Stable and Efficient GRPO for Unified Discrete Diffusion Models
description: >-
  [ICML 2026][LLM Alignment][Discrete Diffusion Models] By defining the final clean sample as the action and reconstructing trajectories using the forward process…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "Discrete Diffusion Models"
  - "Policy Optimization"
  - "Text-to-Image Generation"
  - "Training Stability"
date: 2026-05-08
content_hash: a053bfdd80c7c3d4
---

# UDM-GRPO: Stable and Efficient GRPO for Unified Discrete Diffusion Models

**Conference**: ICML 2026  
**arXiv**: [2604.18518](https://arxiv.org/abs/2604.18518)  
**Code**: https://github.com/Yovecent/UDM-GRPO  
**Area**: Diffusion Models / Text-to-Image Generation / RL Alignment  
**Keywords**: Discrete Diffusion Models, Policy Optimization, Text-to-Image Generation, Training Stability

## TL;DR
By defining the final clean sample as the action and reconstructing trajectories using the forward process, this work successfully integrates GRPO into discrete diffusion models for the first time, addressing training instability and achieving SOTA on multiple benchmarks like GenEval.

## Background & Motivation

**Background**: Diffusion models have demonstrated excellence in text-to-image generation. GRPO has proven efficient in enhancing LLM reasoning capabilities, and recently, Flow-GRPO successfully applied it to continuous diffusion models.

**Limitations of Prior Work**: Directly applying GRPO to Uniform Discrete Diffusion (UDM) models results in severe training instability—rewards fluctuate significantly after an initial 500-step rise, and KL divergence grows sharply, leading to total collapse.

**Key Challenge**: First, noisy and unreliable intermediate prediction signals at various timesteps should not serve as RL optimization targets. Second, a severe distribution shift exists between the model's self-generated backward trajectories and the forward distribution used during pre-training, leading to out-of-distribution (OOD) training.

**Goal**: To develop the first UDM-GRPO framework capable of stable and efficient RL optimization for discrete diffusion models.

**Key Insight**: Starting from the specific structure of the UDM sampling process, this work proposes eliminating these two issues by unifying the final clean sample as the action and reconstructing trajectories through the forward process.

**Core Idea**: Replace intermediate noisy predictions with final clean samples as RL actions and construct training trajectories using the forward process of model-generated clean samples rather than the backward process.

## Method

### Overall Architecture
Based on the GRPO framework, UDM-GRPO introduces two core modifications tailored to the specificities of discrete diffusion models: (1) the action definition is changed from intermediate predictions $x_1^t$ to the final clean sample $\hat{x}_1$; (2) the trajectory source is shifted from the backward process $\mathcal{X}_{\text{backward}}$ to the forward process $\mathcal{X}_{\text{forward}}$.

### Key Designs

1.  **Final Samples as Actions**:
    - **Function**: Unifies actions across all timesteps as $a_t \triangleq \hat{x}_1$ to avoid learning noisy intermediate predictions.
    - **Mechanism**: In standard diffusion pre-training, the target is consistently the clean image, ensuring consistency between actions and reward signals. By maximizing the probability $p_\theta(\hat{x}_1|\hat{x}_t,c)$, the optimization signal points accurately toward the target defined by the rewards.
    - **Design Motivation**: Early timesteps exhibit high prediction entropy and noise; direct optimization of these intermediate states forces the model to learn erroneous signals.

2.  **Forward Process Trajectory Reconstruction**:
    - **Function**: Replaces the backward sampling process with the forward diffusion process $\mathcal{X}_{\text{forward}}$ of the generated clean sample $\hat{x}_1$ as the RL optimization trajectory.
    - **Mechanism**: Given $\hat{x}_1$ and any noisy timestep $t$, the corresponding $\hat{x}_t \sim p_t(x|\hat{x}_1)$ is generated via the forward process. This ensures the trajectory distribution aligns perfectly with the forward distribution seen during pre-training.
    - **Design Motivation**: Cumulative prediction errors in the backward process cause the trajectory distribution to deviate severely from the pre-training distribution, leading to instability when the model learns in OOD states.

3.  **Step Reduction and CFG-Free Strategy**:
    - **Function**: (a) Step reduction strategy randomly selects 3 consecutive timesteps for optimization within high-noise early stages of diffusion; (b) CFG-Free strategy eliminates the joint optimization of conditional and unconditional models.
    - **Mechanism**: Early timesteps have high variance and larger exploration space, requiring focused optimization, while late timesteps are nearly deterministic with lower optimization gains.
    - **Design Motivation**: Dispersed gradients across all timesteps slow down convergence; CFG dual-objective optimization consumes significant computational resources.

## Key Experimental Results

### Main Results

| Model | GenEval | PickScore | OCR |
|------|---------|-----------|-----|
| URSA (Base) | 0.69 | 21.79 | 0.08 |
| SD3.5-L | 0.71 | 22.91 | 0.68 |
| FLUX.1-Dev | 0.66 | 22.84 | 0.59 |
| URSA + UDM-GRPO | **0.96** | **23.81** | **0.57** |

### Ablation Study

| Model | Action Definition | Trajectory Source | GenEval | PickScore | OCR |
|------|---------|---------|---------|-----------|-----|
| Base URSA | - | - | 0.69 | 21.79 | 0.08 |
| Direct Adapt. | $x_1^t$ | backward | 0.84 | 21.99 | 0.23 |
| Improved Action | $\hat{x}_1$ | backward | 0.89 | 23.10 | 0.23 |
| Improved Traj. | $\hat{x}_1$ | forward | 0.94 | 23.51 | 0.34 |
| Full Method | $\hat{x}_1$ | forward+CFG-Free | 0.96 | 23.81 | 0.57 |

### Key Findings
- The FID of forward trajectories is consistently lower than that of backward trajectories compared to the pre-training distribution, validating the distribution alignment hypothesis.
- GenEval scores across all six individual metrics reached above 0.95.
- OCR accuracy improved significantly from 8% to 57%.

## Highlights & Insights
- **Deep Diagnosis of Root Causes**: Precisely identifies two distinct sources of instability through entropy visualization and quantitative FID analysis.
- **Elegant Unification of Design**: Using the final clean sample as a unified action definition resolves intermediate prediction noise while naturally aligning RL objectives with pre-training objectives.

## Limitations & Future Work
- The method still requires 32 A100 GPUs for training, entailing high computational costs.
- While the CFG-Free strategy improves stability, it leads to a noticeable performance dip in the initial stages.
- Evaluation was limited to T2I tasks; applicability to other discrete generative tasks (text, audio, etc.) remains to be verified.

## Related Work & Insights
- **vs Flow-GRPO**: Designed for continuous diffusion; direct application to discrete models causes instability.
- **vs DDPO**: DDPO operates on the entire backward trajectory within a multi-step MDP setting; this work unifies the process into a final sample action.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First successful resolution of the UDM-GRPO integration challenge.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three tasks, multiple benchmarks, and comprehensive ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear problem diagnosis and reproducible experiments.
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm for combining discrete generation with RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] F-TIS: Harnessing Diverse Models in Collaborative GRPO](f-tis_harnessing_diverse_models_in_collaborative_grpo.md)
- [\[ACL 2026\] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](../../ACL2026/llm_alignment/mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)
- [\[ICML 2026\] $f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses](f-divergence_regularized_rlhf_two_tales_of_sampling_and_unified_analyses.md)
- [\[ICML 2026\] Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance](toward_stable_value_alignment_introducing_independent_modules_for_consistent_val.md)
- [\[ICLR 2026\] Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends](../../ICLR2026/llm_alignment/group-relative_reinforce_is_secretly_an_off-policy_algorithm_demystifying_some_m.md)

</div>

<!-- RELATED:END -->
