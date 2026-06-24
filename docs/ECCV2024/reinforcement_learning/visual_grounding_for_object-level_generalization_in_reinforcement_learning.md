---
title: >-
  [Paper Note] Visual Grounding for Object-Level Generalization in Reinforcement Learning
description: >-
  [ECCV2024][Reinforcement Learning][visual grounding] This paper leverages the visual grounding capability of a vision-language model (MineCLIP) to generate confidence maps of target objects. VLM knowledge is transferred to reinforcement learning through two pathways—reward design and task representation—enabling zero-shot generalization to unseen objects and instructions.
tags:
  - "ECCV2024"
  - "Reinforcement Learning"
  - "visual grounding"
  - "zero-shot generalization"
  - "VLM"
  - "Minecraft"
  - "CLIP"
  - "intrinsic reward"
date: 2026-05-08
content_hash: 206abaa5a1c8f2d0
---

# Visual Grounding for Object-Level Generalization in Reinforcement Learning

**Conference**: ECCV2024  
**arXiv**: [2408.01942](https://arxiv.org/abs/2408.01942)  
**Code**: [PKU-RL/COPL](https://github.com/PKU-RL/COPL)  
**Area**: Reinforcement Learning  
**Keywords**: visual grounding, zero-shot generalization, VLM, Minecraft, CLIP, intrinsic reward

## TL;DR
This paper leverages the visual grounding capability of a vision-language model (MineCLIP) to generate confidence maps of target objects. VLM knowledge is transferred to reinforcement learning through two pathways—reward design and task representation—enabling zero-shot generalization to unseen objects and instructions.

## Background & Motivation
In open-world environments such as Minecraft, agents must interact with a variety of objects based on natural language instructions. However, the coverage of training data is always limited, and agents routinely encounter unseen object names during evaluation. Existing approaches suffer from two key limitations:

1. **MineCLIP rewards are distance-insensitive**: MineCLIP utilizes the similarity between image sequences and text as an intrinsic reward, but this similarity does not correlate with the agent's actual distance to the target object. Consequently, agents tend to stare at the target from afar without approaching, failing to complete difficult skills that require close interaction (such as hunting).
2. **Language embeddings lack generalization as task representations**: Traditional methods directly feed language embeddings as policy inputs. When encountering object names outside the training set, the policy network fails to comprehend their semantics.

## Core Problem
How to transfer vision-language knowledge from VLMs into RL at minimal cost, enabling the agent to efficiently learn basic skills while achieving zero-shot generalization to unseen target objects during training?

## Method
The proposed method, **COPL (CLIP-guided Object-grounded Policy Learning)**, consists of three core modules:

### 1. Visual Grounding: Generating Confidence Maps
- First, GPT-4 is used to extract target object names from natural language instructions (e.g., extracting "cow" from "hunt a cow in plains with a diamond sword").
- A MaskCLIP-style modification is applied to the image encoder of MineCLIP: **the scaled dot-product attention of the multi-head attention in the last ViT block is removed, keeping only the value-embedding transformation**, allowing the features at each patch position to be utilized individually.
- Each patch embedding is individually passed through MineCLIP's temporal transformer (with sequence length set to 1) to ensure alignment with MineCLIP's embedding space.
- The text side remains unmodified. It encodes the target name and a set of negative sample vocabularies. The cosine similarity between each patch and the text embedding is calculated, and a softmax is applied to obtain the probability of the target's presence at each patch.
- Ultimately, a 2D confidence map is output, with dimensions equal to the number of patches.

### 2. Transfer via Reward: Focal Reward
The designed **focal reward** is calculated at each timestep $t$ as:

$$r_t^f = \text{mean}(m_t^c \circ m^k)$$

where $m_t^c$ is the target confidence map, $m^k$ is a 2D Gaussian kernel centered at the field of view ($\sigma_1 = H/3, \sigma_2 = W/3$), and $\circ$ denotes the Hadamard product.

- **Area as a proxy for distance**: The closer the target is $\rightarrow$ the more pixels it occupies $\rightarrow$ the larger the reward.
- **Gaussian kernel encourages centering**: The closer the target is to the center of the field of view $\rightarrow$ the larger the reward, addressing the issue of the agent not knowing which target to pursue when multiple targets are present.
- **Denoising**: (1) Patches with the highest probability for negative sample words are set to zero; (2) Patches below the threshold $\tau=0.2$ are set to zero, and those above the threshold are binarized to one.

The final training reward is defined as $r_t = r_t^{env} + \lambda r_t^f$, where $\lambda=5$.

### 3. Transfer via Representation: Confidence Map as Task Representation
- Instead of using language embeddings as policy inputs, the confidence map is employed as a unified 2D task representation.
- The policy network adopts the MineAgent architecture with an additional branch to encode the confidence map, fusing multimodal features via concatenation.
- Key advantage: For unseen objects, the open-vocabulary property of MineCLIP can still generate reasonable confidence maps, which the policy network can comprehend and act upon using this vision-based 2D representation.
- Multi-task RL training is conducted using PPO.

## Key Experimental Results

### Single-Task Experiments (4 Hunting Skills)

| Task | Focal | MineCLIP | NDCLIP | Sparse |
|------|-------|----------|--------|--------|
| hunt a cow | **71.3±9.7** | 3.8±4.8 | 3.5±3.0 | 0.0±0.0 |
| hunt a sheep | **68.8±25.3** | 5.3±2.9 | 28.8±23.0 | 2.5±3.0 |
| hunt a pig | **58.3±7.8** | 2.3±1.7 | 0.3±0.5 | 0.5±0.6 |
| hunt a chicken | **29.5±10.9** | 0.0±0.0 | 4.8±1.5 | 0.5±0.6 |

Focal reward is the only method capable of mastering all four difficult skills.

### Multi-Task Zero-Shot Generalization (Hunting Domain, Unseen Targets)

| Unseen Target | COPL | LCRL[t] | LCRL[i] |
|----------|------|---------|---------|
| llama | **48.8±6.5** | 14.5±10.4 | 24.5±12.7 |
| horse | **49.0±5.5** | 2.5±1.3 | 5.5±4.7 |
| spider | **54.5±12.7** | 9.8±3.5 | 18.3±12.0 |
| mushroom cow | **40.3±11.2** | 19.3±20.5 | 0.0±0.0 |
| **Average** | **48.1** | 11.5 | 12.1 |

On unseen targets, the average success rate of COPL is approximately **4 times** that of language-conditioned methods (hunting domain) and **2 times** (harvest domain).

## Highlights & Insights
1. **Simple and effective idea**: Light modifications to MineCLIP (without fine-tuning) yield visual grounding capability with minimal computational overhead.
2. **Exquisitely designed focal reward**: The Gaussian kernel simultaneously addresses distance guidance and multi-object focusing, aligning better with task requirements than the original MineCLIP reward.
3. **Complementary transfer pathways**: The reward pathway enhances skill learning efficiency, while the representation pathway boosts generalization ability.
4. **Comprehensive experimental evaluation**: Single-task, multi-task, and generalization experiments progress systematically, accompanied by comparative evaluations against imitation learning baselines (VPT family).

## Limitations & Future Work
1. **Applicable only to object-centric tasks**: For non-object-centric tasks like "dig a hole" or "build a house", it is difficult to define explicit target objects for grounding.
2. **Lack of action generalization**: The method only supports generalization at the target-object level, and cannot generalize to novel, unseen behavior patterns.
3. **Dependency on LLMs for target extraction**: The pipeline relies on GPT-4 to extract target object names from instructions, which increases system dependencies.
4. **Noisy confidence maps**: Although denoising steps are applied, the quality of the initial confidence map is bounded by the MineCLIP vision encoder.

## Related Work & Insights
- **vs MineCLIP reward**: MineCLIP reward is distance-insensitive, whereas focal reward addresses this issue by using pixel area as a proxy for distance.
- **vs Imitation Learning Methods (VPT, STEVE-1)**: Imitation learning relies heavily on large-scale annotated datasets, and its generalization is limited by training data coverage; COPL bypasses this limitation by leveraging the open-vocabulary capability of the VLM.
- **vs Language-Conditioned RL (LCRL)**: LCRL directly uses language embeddings as policy inputs, failing when facing unseen vocabularies; COPL maps language into visual confidence maps, providing a more unified and interpretable representation.
- **vs MaskCLIP / CLIPSurgery**: In the Minecraft domain, using domain-specific MineCLIP outperforms general CLIP models and can be seamlessly integrated with existing computational pipelines.

## Insights & Connections
- **VLM $\rightarrow$ RL Knowledge Transfer Paradigm**: Utilizing both reward and representation pathways is a general framework that can be extended to robotic manipulation and other domains.
- **Visual Grounding as an Intermediate Representation**: Translating language instructions into 2D visual probability maps as policy inputs is more interpretable and generalizable than directly using language embeddings.
- **Value of Domain-Specific VLMs**: General CLIP performs poorly in Minecraft, while MineCLIP, fine-tuned on domain data, improves performance significantly. This suggests prioritizing domain-adapted VLMs for specific fields.
- **Generality of Denoising Strategies**: The thresholding and negative-sample filtering of confidence maps can be extended to other CLIP-based dense prediction tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ — Both focal reward and utilizing the confidence map as task representation are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Single-task, multi-task, and generalization experiments are progressively evaluated, with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation and detailed methodology.
- Value: ⭐⭐⭐⭐ — Provides a practical and inspiring paradigm for integrating VLMs with RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Talk2Move: Reinforcement Learning for Text-Instructed Object-Level Geometric Transformation in Scenes](../../CVPR2026/reinforcement_learning/talk2move_reinforcement_learning_for_text-instructed_object-level_geometric_tran.md)
- [\[AAAI 2026\] Start Small, Think Big: Curriculum-based Relative Policy Optimization for Visual Grounding](../../AAAI2026/reinforcement_learning/start_small_think_big_curriculum-based_relative_policy_optimization_for_visual_g.md)
- [\[CVPR 2026\] TSTM: Temporal Segmentation for Task-relevant Mask in Visual Reinforcement Learning Generalization](../../CVPR2026/reinforcement_learning/tstm_temporal_segmentation_for_task-relevant_mask_in_visual_reinforcement_learni.md)
- [\[AAAI 2026\] Object-Centric World Models for Causality-Aware Reinforcement Learning](../../AAAI2026/reinforcement_learning/object-centric_world_models_for_causality-aware_reinforcement_learning.md)
- [\[ECCV 2024\] AdaGlimpse: Active Visual Exploration with Arbitrary Glimpse Position and Scale](adaglimpse_active_visual_exploration_with_arbitrary_glimpse_position_and_scale.md)

</div>

<!-- RELATED:END -->
