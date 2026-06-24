---
title: >-
  [Paper Note] Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation
description: >-
  [CVPR 2025][Image Generation][Prompt Optimization] Proposes PAG (Prompt Adaptation with GFlowNets), which reformulates prompt adaptation as a probabilistic inference problem. By using GFlowNets to sample from the reward distribution instead of maximizing the reward, and combining three key techniques—flow reactivation, reward-prioritized sampling, and progressive reward decomposition—to solve the mode collapse issue, PAG generates text-to-image prompts that are both high in q…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Prompt Optimization"
  - "GFlowNets"
  - "Text-to-Image Generation"
  - "Diversity"
  - "Probabilistic Inference"
date: 2026-05-08
content_hash: 08908dc172314f73
---

# Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation

**Conference**: CVPR 2025  
**arXiv**: [2502.11477](https://arxiv.org/abs/2502.11477)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Prompt Optimization, GFlowNets, Text-to-Image Generation, Diversity, Probabilistic Inference

## TL;DR
Proposes PAG (Prompt Adaptation with GFlowNets), which reformulates prompt adaptation as a probabilistic inference problem. By using GFlowNets to sample from the reward distribution instead of maximizing the reward, and combining three key techniques—flow reactivation, reward-prioritized sampling, and progressive reward decomposition—to solve the mode collapse issue, PAG generates text-to-image prompts that are both high in quality and diverse.

## Background & Motivation

1. **Background**: Text-to-image diffusion models (such as Stable Diffusion) are capable of generating high-quality images, but it remains difficult for users to write prompts that trigger desired attributes (e.g., aesthetic quality, user intent). Prompt adaptation automatically optimizes user prompts to generate better images.
2. **Limitations of Prior Work**: Pioneering work such as Promptist utilizes PPO reinforcement learning to optimize prompts, but the reward maximization nature of RL causes the policy to converge to deterministic actions—the generated prompts always append similar suffixes, yielding extremely low diversity and degenerating into simple heuristics.
3. **Key Challenge**: Reward maximization concentrates the policy on narrow high-reward regions, failing to explore diverse modes in the prompt space. However, directly fine-tuning language models with GFlowNets leads to mode collapse caused by the loss of neural plasticity.
4. **Goal** (a) Replace RL with GFlowNets to achieve diverse prompt sampling; (b) resolve the mode collapse problem during the GFlowNet fine-tuning process.
5. **Key Insight**: Discovered that the proportion of "dormant neurons" continuously rises during GFlowNet fine-tuning (similar to the progressive hardening of brain circuits in neuroscience), preventing the model from learning from diverse samples.
6. **Core Idea**: Reformulate prompt adaptation from reward maximization to a probabilistic inference problem of sampling from a reward distribution, maintaining the exploration ability of GFlowNets via a three-pronged approach: flow reactivation, prioritized sampling, and progressive reward decomposition.

## Method

### Overall Architecture
The input is the user's initial prompt $\mathbf{x}$. A language model (GPT-2) acts as the GFlowNet policy to generate the adapted prompt $\mathbf{y}$, which is fed into a diffusion model (SD v1.4) to generate images and compute rewards. The policy is then fine-tuned via GFlowNet training objectives so that its sampling probability is proportional to the reward. The entire process does not modify the parameters of the diffusion model.

### Key Designs

1. **GFlowNet Probabilistic Inference Framework**:
    - **Function**: Ensure that the policy's probability of sampling prompts is proportional to the reward, rather than merely maximizing the reward.
    - **Mechanism**: The optimal policy can be analytically derived as $p_\theta^*(\mathbf{y}|\mathbf{x}) \propto p_{\text{ref}}(\mathbf{y}|\mathbf{x}) \cdot \exp(\frac{1}{\beta}r(\mathbf{x},\mathbf{y}))$, which is the reference policy multiplied by the exponent of the reward. This aligns exactly with the sampling objective of GFlowNet (i.e., sampling from an unnormalized density function). The reward function consists of two parts: aesthetic quality enhancement (LAION predictor) and text-to-image correlation (CLIP similarity).
    - **Design Motivation**: Compared to the reward maximization of RL, sampling based on the reward proportion in GFlowNet inherently guarantees the diversity of solutions—high-reward regions are sampled frequently but do not dominate exclusively.

2. **Flow Reactivation + Reward-Prioritized Sampling**:
    - **Function**: Resolve the loss of neural plasticity and training instability during GFlowNet fine-tuning.
    - **Mechanism**: Flow reactivation periodically re-initializes the parameters of the final layer of the flow function every $M$ steps (without resetting the forward policy) to awaken dormant neurons. However, a pure reset causes the model to forget discovered high-reward regions. Therefore, reward-prioritized sampling is introduced to sample from the experience replay buffer proportionally to the reward $P_\mathcal{B}(\mathbf{x},\mathbf{y}) = \frac{\exp(R(\mathbf{x},\mathbf{y}))}{\sum \exp(R(\mathbf{x},\mathbf{y}))}$, ensuring fast recovery of knowledge in high-quality regions after the reset.
    - **Design Motivation**: Empirical results show that the proportion of dormant neurons continuously increases during GFlowNet training (from ~30% to ~80%), which directly triggers a sharp drop in output diversity. Flow reactivation clears solidified paths, while prioritized sampling preserves knowledge—the two are complementary.

3. **Progressive Reward Decomposition**:
    - **Function**: Provide fine-grained learning signals for each step in the sequence generation process to address the credit assignment problem.
    - **Mechanism**: Leveraging the property that the likelihood of the reference policy can be decomposed token-by-token, the reward is extended from being defined only on terminal states to all intermediate states: for non-terminal states $R(\mathbf{x}, y_{0:t}) = p_{\text{ref}}(y_{0:t}|\mathbf{x})$, while terminal states are multiplied by the exponent of the reward. This allows the use of the Forward-Looking Detailed Balance objective for flow matching at each step, providing local gradient signals at every step.
    - **Design Motivation**: Standard GFlowNet only receives a reward after generating the complete prompt. The model struggles to determine which intermediate token choices led to good or bad outcomes, which makes it conservatively exploit known patterns rather than explore new paths.

### Loss & Training
Uses the Forward-Looking Detailed Balance (FL-DB) objective to constrain flow conservation at each transition step in the trajectory. Trained for 10K steps with a batch size of 256, policy learning rate of 1e-5, and flow function learning rate of 1e-4. Online sampling and buffered prioritized sampling each account for 50%.

## Key Experimental Results

### Main Results
Evaluated reward and diversity across 4 datasets (Lexica/DiffusionDB/COCO/ChatGPT):

| Method | Reward | Diversity | Description |
|------|--------|-----------|------|
| SFT | 0.66 | 0.14 | Supervised fine-tuning baseline |
| Promptist | 0.70 | 0.02 | PPO reinforcement learning, extremely poor diversity |
| Rule-Based | 0.59 | 0.12 | Simple suffix concatenation |
| GFlowNets (vanilla) | 0.81 | 0.20 | Naive GFlowNets |
| **PAG (Ours)** | **0.83** | **0.29** | ImageReward metric |

### Ablation Study

| Configuration | Reward (COCO) | Diversity (COCO) | Description |
|------|--------------|-----------------|------|
| PAG Full Model | 0.88 | 0.32 | All components |
| w/o Reset (Flow Reactivation) | 0.70 | 0.27 | Accumulation of dormant neurons |
| w/o PRT (Prioritized Sampling) | 0.30 | 0.22 | Severe performance degradation |
| w/o FL (Reward Decomposition) | 0.56 | 0.16 | Largest loss of diversity |

### Key Findings
- **Reward-Prioritized Sampling (PRT) is the most critical component**: Removing it causes the reward to plummet from 0.88 to 0.30, demonstrating that maintaining continuous access to high-quality experiences is vital for GFlowNet training.
- **Reward decomposition contributes the most to diversity**: Removing it drops diversity from 0.32 to 0.16, validating the effectiveness of fine-grained credit assignment in resisting mode collapse.
- **Cross-model zero-shot transfer**: When applying the policy trained on SD v1.4 directly to SD3, PAG significantly outperforms all baselines (0.95 vs. 0.76), demonstrating the robustness brought by diverse prompts.
- **Comparison with diffusion model fine-tuning**: PAG is comparable in aesthetic quality to DDPO (which directly fine-tunes the diffusion model), but its generated styles are far more diverse than those of DDPO.

## Highlights & Insights
- **Discovery of the loss of neural plasticity**: It systematically reveals for the first time in GFlowNet fine-tuning of language models that dormant neurons cause mode collapse—a finding that provides valuable insights for all GFlowNet applications.
- **Probabilistic Inference vs. Reward Maximization**: Transitioning prompt optimization from an RL paradigm to a probabilistic inference paradigm is a paradigm-level innovation. This concept can be transferred to any generative task requiring both high quality and diversity (e.g., code generation, molecular design).
- **Practical value of prompt adaptation**: It requires no modifications to the diffusion model parameters, can be directly transferred to closed-source models, and incurs computational costs only at the level of fine-tuning GPT-2—highly engineering-friendly.

## Limitations & Future Work
- The policy model uses a relatively small GPT-2; switching to a larger language model could generate more creative prompts.
- The current reward function solely considers aesthetics and relevance, without covering more complex human preferences such as safety or style-diversity preferences.
- Every reward evaluation requires generating images (3 images per prompt), leaving the training cost still relatively high.
- Applications in other conditional generation tasks (e.g., video generation, 3D generation) have not yet been explored.

## Related Work & Insights
- **vs. Promptist**: Promptist uses PPO to maximize reward, leading to deterministic behavior, whereas PAG uses GFlowNets to sample proportionally to the reward, inherently guaranteeing diversity.
- **vs. DPO-Diff**: DPO-Diff optimizes negative prompts via gradients, which is complementary rather than competitive; the advantage of PAG lies in its zero-shot transfer capability.
- **vs. DDPO**: DDPO directly fine-tunes diffusion model parameters, yielding good performance but being non-transferable and computationally intensive; PAG operates in the prompt space, making it lightweight and transferable.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Reformulates prompt adaptation as a probabilistic inference problem, and discovers and resolves the loss of plasticity issue in GFlowNet fine-tuning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Thorough experiments across multiple datasets, multiple reward functions, cross-model transfer, and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Deep problem analysis and clear motivation explanation.
- **Value**: ⭐⭐⭐⭐ Provides significant reference value for both the GFlowNet community and T2I optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows](diverseflow_sample-efficient_diverse_mode_coverage_in_flows.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](../../ICLR2026/image_generation/diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[CVPR 2025\] ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts](shapewords_guiding_text-to-image_synthesis_with_3d_shape-aware_prompts.md)
- [\[CVPR 2025\] PreciseCam: Precise Camera Control for Text-to-Image Generation](precisecam_precise_camera_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] Minority-Focused Text-to-Image Generation via Prompt Optimization](minority-focused_text-to-image_generation_via_prompt_optimization.md)

</div>

<!-- RELATED:END -->
