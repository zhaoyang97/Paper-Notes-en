---
title: >-
  [Paper Note] Visually-Guided Policy Optimization for Multimodal Reasoning
description: >-
  [ACL2026][Reinforcement Learning][Multimodal Reasoning] VGPO employs hidden state similarity to locate vision-related tokens during RLVR training…
tags:
  - "ACL2026"
  - "Reinforcement Learning"
  - "Multimodal Reasoning"
  - "GRPO"
  - "Visual Attention"
  - "Visual Forgetting"
date: 2026-05-08
content_hash: facac6fc841ae6f4
---

# Visually-Guided Policy Optimization for Multimodal Reasoning

**Conference**: ACL2026  
**arXiv**: [2604.09349](https://arxiv.org/abs/2604.09349)  
**Code**: https://github.com/wzb-bupt/VGPO  
**Area**: reinforcement_learning  
**Keywords**: Multimodal Reasoning, Reinforcement Learning, GRPO, Visual Attention, Visual Forgetting

## TL;DR
VGPO employs hidden state similarity to locate vision-related tokens during RLVR training, subsequently strengthening visual focus through late-stage visual compensation and intra/inter-trajectory advantage re-weighting. This allows Qwen2.5-VL-7B to surpass GRPO/DAPO and existing vision-enhanced RL methods in mathematical multimodal reasoning and vision-dependent tasks.

## Background & Motivation
**Background**: Methods such as RLVR, GRPO, and DAPO have significantly improved the step-by-step reasoning capabilities of VLMs, particularly in tasks with verifiable answers like mathematics, geometry, and visual question answering. Current multimodal reasoning research typically focuses on final answer rewards, rollout diversity, KL/entropy regularization, or external visual verifiers.

**Limitations of Prior Work**: The reasoning process of VLMs remains strongly text-dominated. During the generation of long reasoning chains, models may briefly attend to images initially but increasingly rely on the problem text and previously generated tokens. Visual token activations become sparse, leading to visual fact forgetting, hallucinations, or erroneous reasoning based on language priors.

**Key Challenge**: Multimodal reasoning requires models to consistently utilize visual evidence across long chains, whereas standard RL only rewards final answer correctness regardless of whether the model faithfully processed the image. Existing vision-enhancement methods often introduce special tokens, additional forward passes, noisy image comparisons, or auxiliary models, resulting in high training costs and system complexity.

**Goal**: The authors aim to directly incorporate "whether the model consistently attends to the image during reasoning" into policy optimization without introducing extra models or external visual verification processes. This ensures the model pursues both answer correctness and sufficient usage of visual evidence.

**Key Insight**: The observation is made that the similarity between the hidden states of generated tokens and image tokens can serve as an endogenous Visual Focus Score. When the model genuinely utilizes visual information, this similarity increases, and the corresponding image attention regions are typically semantically reasonable.

**Core Idea**: Construct visual attention signals using the model's own hidden states and transform them into weighting factors for the RL advantage function. This allows rewards for correct answers to propagate along reasoning trajectories that are more visually faithful.

## Method
VGPO can be understood as adding a "visual faithfulness modulator" layer onto RL frameworks like DAPO/GRPO. While original RLVR only considers the final reward of each rollout (e.g., exact match), VGPO redistributes advantages at both the token and trajectory granularities without changing the verifiable reward itself. It assigns higher update weights to vision-related tokens and trajectories with stronger overall visual focus.

### Overall Architecture
Given an image $I$, a text question $q$, and an answer $a$, the policy model samples a set of reasoning trajectories. First, VGPO derives visual prototypes from the hidden states of image tokens and calculates the similarity between each generated token and the visual prototypes to form a Visual Focus Score. Then, Visual Attention Compensation applies a linear enhancement to high-scoring visual tokens in the later stages of reasoning to counteract temporal visual forgetting. Finally, Dual-Grained Advantage Re-Weighting embeds this visual compensation signal into the policy objective: intra-trajectory weighting distinguishes token-level visual importance, while inter-trajectory weighting distinguishes the cumulative visual focus of the entire response.

### Key Designs
1. **Visual Focus Score**:
	- **Function**: Determines whether each generated token is related to visual evidence without external annotations or auxiliary models.
	- **Mechanism**: Aggregates the hidden states of input image tokens into a visual prototype $\mu_v$, then calculates the cosine similarity between the current generated token's hidden state $h_{i,t}$ and $\mu_v$. The visual focus score is defined as $\rho_{i,t}=0.5(\mathcal{S}(h_{i,t},\mu_v)+1)$, normalized to $[0,1]$. Mean-pooling is used as the default construction for the visual prototype.
	- **Design Motivation**: To strengthen visual faithfulness, it is necessary to identify which tokens in the reasoning chain are actually "thinking about the image." Hidden state similarity provides a cheap, endogenous signal that can be integrated into end-to-end training.

2. **Visual Attention Compensation**:
	- **Function**: Specifically compensates for the decay of visual attention in the later stages of long reasoning.
	- **Mechanism**: Instead of using $\rho_{i,t}$ directly, which might underestimate late-stage visual tokens due to natural attention decay, VGPO constructs $w_{i,t}=\rho_{i,t}[1+G_i(\rho_{i,t})\beta t/T_i]$. Here, $t/T_i$ linearly enhances compensation based on generation position, and $G_i$ is active only for tokens in the latter part of the trajectory that fall within the top-$\kappa$ visual scores. Default hyperparameters are $\beta=0.3$, $\gamma=0.5$, and $\kappa=0.2$.
	- **Design Motivation**: Early reasoning naturally tends to attend to images; indiscriminate strengthening could interfere with problem understanding. Late-stage compensation targets actual visual forgetting, avoiding treating all tokens as visual tokens.

3. **Dual-Grained Advantage Re-Weighting**:
	- **Function**: Rewards visually faithful reasoning at both the token and trajectory levels.
	- **Mechanism**: Within a trajectory, $w_{i,t}$ is min-max normalized and the trajectory mean is subtracted to obtain $\psi_{i,t}$, granting higher advantage to tokens with visual activation above the trajectory average. Between trajectories, the cumulative compensation score $s_i=\sum_t w_{i,t}$ is calculated, then normalized and centered within the rollout group to obtain $\phi_i$. The final advantage is $\hat{A}^{\mathcal{V}}_{i,t}=\hat{A}_i(1+\psi_{i,t})(1+\phi_i)$.
	- **Design Motivation**: Focusing only on local tokens ignores whether the response consistently references the image; focusing only on the whole trajectory fails to accurately assign credit to key visual steps. Combining both granularities produces a more refined optimization signal.

### Loss & Training
The base optimization follows the group-relative policy optimization style of GRPO/DAPO: multiple responses are sampled for each question, a binary reward is obtained based on exact match, and advantages are normalized within the group. VGPO replaces the standard advantage with the visually-modulated advantage $\hat{A}^{\mathcal{V}}_{i,t}$. Experiments use Qwen2.5-VL 3B, 7B, and 32B. Training data includes ViRL39K, Geo3K, and MMK12. Training incorporates 2 epochs, a learning rate of $1\times 10^{-6}$, a rollout batch size of 512, a maximum response length of 2,048, and an evaluation temperature of 0.

## Key Experimental Results

### Main Results
The main experiments compare the Qwen2.5-VL-7B base model, GRPO, DAPO, VGPO, and existing 7B multimodal reasoning methods. VGPO achieves the best average performance across both general mathematical/geometric reasoning and vision-dependent multimodal reasoning tasks.

| Method | Avg-Math↑ | Avg-Vision↑ | Gain vs. Base | Note |
|--------|------|------|----------|------|
| Qwen2.5-VL-7B | 50.0 | 48.7 | - | Without RL post-training |
| + GRPO | 62.6 | 58.8 | Math +25.2%, Vision +20.7% | Group-relative answer reward only |
| + DAPO | 63.8 | 59.6 | Math +27.6%, Vision +22.4% | Stronger RL baseline |
| PAPO-D-7B | 65.5 | 60.4 | - | Vision-enhanced RL method |
| VPPO-RL-7B | 65.7 | 61.3 | - | KL-aware vision enhancement |
| + VGPO (Ours) | 66.6 | 63.3 | Math +33.2%, Vision +30.0% | Best performance on both metrics |

| Setting | Avg-Math↑ | Avg-Vision↑ | Note |
|------|---------|------|------|
| Qwen2.5-VL-3B + DAPO | 55.3 | 48.3 | Small model baseline |
| Qwen2.5-VL-3B + VGPO | 57.7 | 53.6 | Significant gains in vision tasks |
| Qwen2.5-VL-32B + DAPO | 68.4 | 64.8 | Large model baseline |
| Qwen2.5-VL-32B + VGPO | 70.7 | 66.7 | Gains persist at 32B scale |
| 7B + DAPO w/ Geo3K 2.1K | 57.4 | 54.8 | Small training set scenario |
| 7B + VGPO w/ Geo3K 2.1K | 60.4 | 55.8 | Outperforms DAPO with limited data |
| 7B + DAPO w/ MMK12 6.4K | 60.8 | 58.8 | Medium training set scenario |
| 7B + VGPO w/ MMK12 6.4K | 62.4 | 60.3 | Generalizes across different datasets |

### Ablation Study

| Configuration | Avg-Math↑ | Avg-Vision↑ | Overall↑ | Note |
|------|---------|------|------|------|
| DAPO baseline | 63.8 | 59.6 | 62.2 | No visual advantage re-weighting |
| + Intra-trajectory | 66.1 | 62.5 | 64.6 | Token-level weighting is effective |
| + Inter-trajectory | 65.3 | 62.0 | 64.0 | Trajectory-level accumulation helps |
| + Intra & Inter | 66.6 | 63.3 | 65.3 | Components are complementary |

| Compensation Strategy | Avg-Math↑ | Avg-Vision↑ | Overall↑ | Note |
|------|---------|------|------|------|
| DAPO baseline | 63.8 | 59.6 | 62.2 | No visual compensation |
| Step-Function | 64.7 | 60.7 | 63.1 | Abrupt compensation causes instability |
| Exponential | 65.1 | 61.0 | 63.5 | Over-emphasizes final tokens |
| Linear (Ours) | 66.6 | 63.3 | 65.3 | Best match for progressive forgetting |
| Full-trajectory compensation | 53.0 | 54.2 | 53.5 | Hurts performance significantly |
| Late-trajectory compensation | 66.6 | 63.3 | 65.3 | Most effective by targeting late decay |

### Key Findings
- Text-dominant reasoning is a real phenomenon. Observations on Qwen2.5-VL-7B show that visual attention peaks briefly in early stages and then declines progressively as generation proceeds.
- The late/early visual accumulation ratio of correct samples is higher than that of incorrect samples (approx. 0.680 vs. 0.532), suggesting that sustained visual focus in later stages correlates with correctness.
- The improvements from VGPO are consistent across scales and datasets: it outperforms DAPO in 3B, 7B, 32B, and Geo3K/MMK12/ViRL39K settings.
- Visual compensation must be "late and accurate." Full-trajectory compensation reduced the overall score from 62.2 to 53.5, indicating that forcing visual focus too early hinders text-based problem parsing.

## Highlights & Insights
- The core highlight of VGPO is converting visual faithfulness from an external supervision task into an internal signal. It requires no extra GPT judges, no dual forward passes with noisy images, and no special visual look-back tokens.
- The dual-grained advantage design is intuitive. Token-level weighting addresses "which step should look at the image," while trajectory-level weighting addresses "which response is more visually grounded overall," providing a better fit for long-chain reasoning than a single regularization term.
- The late compensation ablation is insightful: visual grounding is not "more is better" but rather should be reinforced at stages where the model is most likely to forget visual evidence.
- This work serves as a reminder that RLVR based solely on final answers might reward incorrect reasoning paths. Correct answers can still be guessed based on language priors; visual process signals bring the training objective closer to the essence of multimodal tasks.

## Limitations & Future Work
- Visual Focus Score assumes that similarity between hidden states and image prototypes represents visual grounding, but this similarity may not always distinguish true visual evidence from language concepts semantically related to the image.
- The method requires access to internal hidden states and image tokens, making it difficult to apply to closed-source VLMs or API-only models.
- Hyperparameters $\beta$, $\gamma$, and $\kappa$ impact training stability. While the paper provides sensitivity analysis, recalibration may be needed for other architectures and tasks.
- Evaluation focuses primarily on verifiable math, geometry, and vision-dependent tasks; it is unclear if the method is equally effective for open-ended VQA, captioning, agent planning, or real-world long-term interactions.
- Increased visual attention ratios do not automatically equate to increased causal faithfulness. Future work could include counterfactual image editing, evidence attribution, or occlusion experiments to verify if the model truly depends on the correct visual regions.

## Related Work & Insights
- **vs GRPO/DAPO**: While GRPO/DAPO primarily optimize final verifiable rewards, VGPO incorporates visual attention into advantage allocation to solve visual forgetting in multimodal tasks.
- **vs PAPO/VPPO**: PAPO and VPPO highlight visual tokens through noisy images or KL divergence; VGPO utilizes hidden state similarity to avoid extra forward passes and external visual comparisons.
- **vs Look-Back / latent visual tokens**: These methods introduce special tokens to trigger re-observation. VGPO does not change the generation format but encourages sustained visual attention through the training objective.
- **Insights for Future Work**: Multimodal RL should not only design outcome rewards but also process-level modality-use rewards, teaching models to use the appropriate modality at the appropriate time.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Utilizing hidden-state visual focus to modulate RL advantages is clever, though it still builds on the GRPO/DAPO framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Includes main results, scaling analysis (model and data), component ablations, compensation strategy ablations, and hyperparameter sensitivity.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear; the method section is dense but the narrative is smooth.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable to multimodal RLVR, visually faithful reasoning, and reducing visual hallucinations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning](../../ICLR2026/reinforcement_learning/from_narrow_to_panoramic_vision_attention-guided_cold-start_reshapes_multimodal_.md)
- [\[ICLR 2026\] Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](../../ICLR2026/reinforcement_learning/unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)
- [\[ACL 2026\] Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning](bridging_sft_and_rl_dynamic_policy_optimization_for_robust_reasoning.md)
- [\[ICLR 2026\] FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning](../../ICLR2026/reinforcement_learning/fapo_flawed-aware_policy_optimization_for_efficient_and_reliable_reasoning.md)
- [\[ICML 2026\] Perceptual Flow Network for Visually Grounded Reasoning](../../ICML2026/reinforcement_learning/perceptual_flow_network_for_visually_grounded_reasoning.md)

</div>

<!-- RELATED:END -->
