---
title: >-
  [Paper Note] DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation
description: >-
  [CVPR2026][Robotics][Vision-and-Language Navigation] This paper proposes DecoVLN, a framework that decouples observation, reasoning, and correction in VLN tasks. By introducing an adaptive memory refinement (AMR) mechani…
tags:
  - "CVPR2026"
  - "Robotics"
  - "Vision-and-Language Navigation"
  - "Adaptive Memory Refinement"
  - "Correction Fine-tuning"
  - "POMDP"
  - "Long-horizon Navigation"
date: 2026-05-08
content_hash: cc1b0d4a64cdbbec
---

# DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation

**Conference**: CVPR2026  
**arXiv**: [2603.13133](https://arxiv.org/abs/2603.13133)  
**Code**: [Project Page](https://arxiv.org/abs/2603.13133) (Project Page and Code links provided)  
**Area**: Human Understanding / Embodied Navigation  
**Keywords**: Vision-and-Language Navigation, Adaptive Memory Refinement, Correction Fine-tuning, POMDP, Long-horizon Navigation

## TL;DR

This paper proposes DecoVLN, a framework that decouples observation, reasoning, and correction in VLN tasks. By introducing an adaptive memory refinement (AMR) mechanism and a state-action-pair-based correction fine-tuning strategy, DecoVLN achieves state-of-the-art performance on R2R-CE and RxR-CE using only egocentric RGB input.

## Background & Motivation

1. **Vision-and-Language Navigation (VLN)** requires an agent to navigate unseen 3D environments following natural language instructions, and is a core task in embodied AI.
2. **Perceptual limitations of existing methods**: The Stop-and-Think paradigm introduces perceptual blind spots (missing critical visual cues during movement); the Full-History Streaming paradigm dilutes key information density via fixed sampling or heuristic context retrieval.
3. **Compounding error problem**: As a sequential decision-making task, VLN is highly susceptible to compounding errors, where early minor mistakes accumulate over time and cause the agent to deviate significantly from the target path.
4. **Inadequate augmentation methods**: Most approaches focus on multimodal trajectory augmentation to improve open-loop action prediction, while lacking effective closed-loop reflection and online error correction capabilities.
5. **Difficulty in long-horizon memory construction**: Uniform sampling discards cues at critical navigation nodes and disrupts temporal coherence; StreamVLN introduces a Slow-Fast memory mechanism but relies on depth sensors.
6. **Spatial reasoning limitations of VLMs**: Existing vision-language models exhibit limited 3D spatial understanding and cannot establish correspondences between local egocentric observations and globally consistent spatial structures.

## Method

### Overall Architecture

DecoVLN models VLN as a POMDP tuple $M=(S,A,T,R,\Omega,O)$, where the agent learns a policy $\pi(H_t,I)$ to maximize expected cumulative reward. The framework decouples three core processes:

- **Observation stream**: The agent continuously perceives the environment during movement; an Adaptive Memory Refinement (AMR) module filters and stores high-information-density state representations into a memory bank.
- **Reasoning stream**: An LLM conditions on the instruction, current frame, and memory bank to output an action chunk comprising multiple consecutive actions.
- **Correction stream**: State-action-pair-based correction fine-tuning endows the model with introspective and self-corrective capabilities.

### Adaptive Memory Refinement (AMR)

Long-horizon memory construction is formulated as an optimization problem. At each timestep, $K$ frames are selected from a candidate pool $\mathcal{C}$ to form a refined memory $\mathcal{M}$ by maximizing a composite score:

$$f^* = \arg\max_{f \in \mathcal{C} \setminus \mathcal{M}} \left[ \lambda_R \cdot \text{Sim}_{\text{Sem}}(f, I) - (1-\lambda_R) \cdot \left( w_V \cdot \text{Sim}_{\text{Vis}}(f, \mathcal{M}) + w_T \cdot \text{Sim}_{\text{Temp}}(f, \mathcal{M}) \right) \right]$$

**Three scoring dimensions**:

- **Semantic relevance** $\text{Sim}_{\text{Sem}}$: cosine similarity between the candidate frame and the instruction, computed via a VLM encoder.
- **Visual diversity** $\text{Sim}_{\text{Vis}}$: maximum visual cosine similarity between a candidate frame and the already-selected memory frames (applied as a penalty).
- **Temporal coverage** $\text{Sim}_{\text{Temp}}$: $\frac{1}{\min_{m \in \mathcal{M}} |t_f - t_m| + \epsilon}$, penalizing candidates that are temporally too close to already-selected frames.

### Correction Fine-tuning Strategy

Error correction is performed at the step level rather than the episode level. State deviation is quantified via geodesic distance:

$$DM(s_t) = \min_{s^* \in P_{exp}} d_g(s_t, s^*)$$

- A deviation threshold $\tau$ is set; when $0 < DM(s_t) \leq \tau$, the agent is in the trustworthy region, and expert corrective actions are queried and stored in the correction dataset $\mathcal{D}_c$.
- When $DM(s_t) > \tau$, the current episode is terminated to filter out low-quality data.
- LLaVA-Video-178K data is incorporated to prevent catastrophic forgetting.

### Loss & Training

- Base model: LLaVA-Video-7B (SigLIP visual encoder + Qwen2-7B language model)
- Optimizer: AdamW; LLM learning rate $2 \times 10^{-5}$, visual encoder learning rate $5 \times 10^{-6}$
- Training data: approximately 360K navigation samples + 180K correction samples
- Inference input: instruction + memory bank ($K=8$) + 4 most recent frames → output: 4-step action chunk

## Key Experimental Results

### Main Results on R2R-CE & RxR-CE Val-Unseen

| Method | Input | R2R NE↓ | R2R SR↑ | R2R SPL↑ | RxR SR↑ | RxR SPL↑ | RxR nDTW↑ |
|--------|-------|---------|---------|----------|---------|----------|-----------|
| StreamVLN | RGB | 5.43 | 52.8 | 47.2 | 48.6 | 42.5 | 60.2 |
| NaVILA* | RGB | 5.22 | 54.0 | 49.0 | 49.3 | 44.0 | 58.8 |
| ETPNav | RGB+Pano+Depth | 4.71 | 57.0 | 49.0 | 54.7 | 44.8 | 61.9 |
| **DecoVLN** | **RGB** | **5.01** | **56.3** | **50.5** | **54.2** | **46.3** | **63.5** |

- On R2R, DecoVLN improves over StreamVLN by SR +3.5% and SPL +3.3%.
- On RxR, gains are SR +5.6% and SPL +3.8%.
- Using only RGB input, DecoVLN surpasses most multi-sensor methods that rely on panoramic views, depth, or odometry.

### Ablation Study

| AMR | CF | NE↓ | SR↑ | SPL↑ |
|-----|----|------|------|------|
| ✗ | ✗ | 5.89 | 47.3 | 43.9 |
| ✓ | ✗ | 5.50 | 50.9 | 46.1 |
| ✓ | ✓ | **5.01** | **56.3** | **50.5** |

- AMR contributes SR +3.6% and SPL +2.2%.
- Correction fine-tuning further improves SR +5.4% on top of AMR, for a total gain of SR +9.0%.
- Memory bank size $K=8$ achieves the best balance between performance and efficiency.
- Trustworthy region threshold $\tau=3$ yields optimal results; with fewer data (180K vs. 240K), DecoVLN outperforms DAgger.

## Highlights & Insights

- **Decoupled design**: Separating observation, reasoning, and correction avoids the high-latency bottleneck of VLM autoregressive generation.
- **Theoretically grounded memory optimization**: Memory construction is formalized as an optimization problem jointly considering semantic relevance, visual diversity, and temporal coverage.
- **Efficient correction strategy**: Step-level correction combined with trustworthy-region filtering achieves better data efficiency than DAgger.
- **RGB-only input surpasses multi-sensor methods**: No depth, panorama, or odometry required; the architecture is clean and practical.
- **Real-world deployment**: Successfully deployed on a Unitree GO2 quadruped robot, demonstrating strong sim-to-real transfer.
- **Emergent behavior**: The robot actively performs lateral micro-adjustments during movement to keep key navigation landmarks within the field of view.

## Limitations & Future Work

- The three weights in AMR ($\lambda_R$, $w_V$, $w_T$) require manual tuning and lack an adaptive learning mechanism.
- Evaluation is limited to Matterport3D environments, without coverage of larger-scale or outdoor scenes.
- Correction fine-tuning relies on Habitat's shortest-path follower as the expert policy, which is difficult to obtain in real environments.
- The action chunk length is fixed at 4 steps; dynamic-length action chunks are not explored.
- Real-world experiments are conducted only in office environments, with limited scene diversity.
- No comparison is made with other correction methods, such as RL-based approaches.

## Related Work & Insights

- **vs. NaVid**: NaVid is among the first to directly fine-tune a VLM for end-to-end navigation, but lacks long-horizon memory management; DecoVLN improves R2R SR by +19.3%.
- **vs. StreamVLN**: StreamVLN proposes a Slow-Fast memory mechanism but relies on depth sensors for voxel construction; DecoVLN surpasses it using only RGB.
- **vs. NaVILA**: NaVILA uses additional large-scale datasets for training; DecoVLN remains competitive without such supplementary data.
- **vs. DAgger**: Conventional DAgger collects correction data at the episode level; DecoVLN precisely filters trustworthy-region data at the step level, achieving better performance with 180K samples versus DAgger's 240K.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of observation/reasoning/correction decoupling, formalized memory optimization, and state-action-pair correction strategy is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-benchmark comparisons, detailed ablations, long-horizon navigation validation, and real-robot deployment.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rigorous problem formulation, and well-organized method description.
- Value: ⭐⭐⭐⭐ — Substantially outperforms multi-sensor methods using only RGB input, with practical deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation](profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)
- [\[ICLR 2026\] JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation](../../ICLR2026/robotics/janusvln_decoupling_semantics_and_spatiality_with_dual_implicit_memory_for_visio.md)
- [\[CVPR 2026\] Towards Open Environments and Instructions: General Vision-Language Navigation via Fast-Slow Interactive Reasoning](towards_open_environments_and_instructions_general_vision-language_navigation_vi.md)
- [\[CVPR 2026\] Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning](fast-thinkact_efficient_vision-language-action_reasoning_via_verbalizable_latent.md)
- [\[ICCV 2025\] COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation](../../ICCV2025/robotics/cosmo_combination_of_selective_memorization_for_low-cost_vision-and-language_nav.md)

</div>

<!-- RELATED:END -->
