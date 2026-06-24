---
title: >-
  [Paper Note] Adapt2Reward: Adapting Video-Language Models to Generalizable Robotic Rewards via Failure Prompts
description: >-
  [ECCV 2024][Video Understanding][Video-Language Models] Proposes Adapt2Reward, which adapts pre-trained video-language models into generalizable language-conditioned reward functions using learnable failure prompts. Requiring only a small amount of robot data from a single environment, it generalizes to new environments and tasks, outperforming prior methods by approximately 28% on MetaWorld.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Video-Language Models"
  - "Robotic Reward Functions"
  - "Failure Prompt Learning"
  - "Cross-Domain Contrastive Learning"
  - "Task Generalization"
date: 2026-05-08
content_hash: 08013e33e4723ce8
---

# Adapt2Reward: Adapting Video-Language Models to Generalizable Robotic Rewards via Failure Prompts

**Conference**: ECCV 2024  
**arXiv**: [2407.14872](https://arxiv.org/abs/2407.14872)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Video-Language Models, Robotic Reward Functions, Failure Prompt Learning, Cross-Domain Contrastive Learning, Task Generalization  

## TL;DR
Proposes Adapt2Reward, which adapts pre-trained video-language models into generalizable language-conditioned reward functions using learnable failure prompts. Requiring only a small amount of robot data from a single environment, it generalizes to new environments and tasks, outperforming prior methods by approximately 28% on MetaWorld.

## Background & Motivation
General-purpose robots require reward functions that can generalize across different environments and instructions for reinforcement learning or planning. Directly using video-language models like CLIP as reward functions faces two core challenges: (1) robot data is scarce, and direct fine-tuning leads to overfitting and catastrophic forgetting; (2) human video datasets rarely contain "failure" videos—while the model can distinguish between two different tasks like "opening a drawer" and "closing a drawer", it cannot distinguish between "successfully closing a drawer" and "failing to close a drawer (e.g., not fully closed or bouncing back open)". Existing methods like DVD only train video similarity discriminators, ignoring the binary classification requirement of success versus failure; simply adding a BCE loss leads to overconfidence on the training set and poor generalization.

## Core Problem
**How to convert human video-language models into reward functions that can distinguish "success vs. failure" and generalize to new environments and tasks under the constraint of extremely limited robot data?** The main difficulties are that human data lacks failure samples, the amount of robot data is too small, and there is a significant domain gap between human and robot domains (including camera views, appearances, and motion styles).

## Method

### Overall Architecture
Input: Human video-text pairs (large-scale, from the SS-V2 dataset) and robot success/failure videos with text and labels (small-scale, from only a few tasks in the training environment). Using Singularity (a pre-trained video-language model) as the backbone, the model is jointly trained with three learning objectives: cross-domain contrastive learning (aligning human-robot features of the same task), video-text contrastive learning (semantic alignment), and failure prompt learning (modeling failure codes as negative samples). Finally, the video-text similarity score is output as the reward, and action selection is executed using VMPC (Visual Model Predictive Control) or DDPG+CEM.

### Key Designs
1. **Cross-Domain Contrastive Learning (CDC)**: Pairs of human and robot videos executing the "same task" are treated as positive pairs, while different tasks are treated as negative pairs, learning domain-invariant representations in the feature space. Bi-directional contrastive losses are computed using human and robot samples as anchors respectively to mitigate the visual domain gap between human and robot.

2. **Learnable Failure Prompts**: Core innovation. For each training task, failure videos are clustered into $K$ failure modes using spherical K-means, where each mode corresponds to a learnable prompt vector $P_{\mathcal{T},k}^f \in \mathbb{R}^{L_p^f \times D}$. The failure prompt is concatenated with the task text embedding and fed into the text encoder to obtain the "failure context" feature $t_{\mathcal{T},k}^f$. During training: (a) in contrastive learning for success videos, all failure texts are treated as additional negative samples (preventing success videos from being misclassified as failures); (b) failure videos are matched with their corresponding failure prompt based on clustering labels for contrastive learning, allowing different causes of failure to be distinguished. Clustering is re-run and pseudo-labels are updated at the end of each epoch to maintain label stability.

3. **Video-Text Contrastive Learning (VLC)**: Bi-directional video-text contrastive alignment is performed simultaneously in both human and robot domains to enhance cross-domain semantic consistency. After incorporating failure prompts, the denominator for success videos is augmented with $K$ failure text items, enabling the model to better distinguish between "executing the task" and "failing to complete the task."

### Loss & Training
- **Total Loss**: $\mathcal{L} = \mathcal{L}_{CDC} + \mathcal{L}_{VLC} + \mathcal{L}_{fVLC} + \mathcal{L}_{BCE}$, with all four weights set to 1
- $\mathcal{L}_{CDC}$: Cross-domain contrastive loss, aligning human-robot features of the same task
- $\mathcal{L}_{VLC}$: Video-text contrastive loss (including failure prompt negative sample expansion)
- $\mathcal{L}_{fVLC}$: Contrastive loss for failure videos, aligning each failure video with its cluster-corresponding failure prompt
- $\mathcal{L}_{BCE}$: Binary cross-entropy, distinguishing success/failure
- Batch sampling: $B_h = B_r = B_f = 8$ (8 samples each for human, success robot, and failure robot)
- Number of failure clusters: $K = 3$ (default)
- Backbone: Singularity model (17M pre-training + SS-V2 fine-tuning)

## Key Experimental Results

| Experimental Setup | Metric | Adapt2Reward | DVD | LIV | Concept2Robot |
|----------|------|-------------|-----|-----|---------------|
| Env Generalization (Avg of 4 Envs) | Success Rate | **Best, outperforming by 28%+** | Baseline | Baseline | Baseline |
| Task Generalization Robot+6Human | Success Rate | **69.33%** | 31.92% | 38.83% | - |
| Task Generalization Robot+9Human | Success Rate | **67.00%** | 40.08% | 39.00% | - |
| Task Generalization Average | Success Rate | **52.73%** | 29.81% | 35.44% | 33.75% |
| C2R-Envs Viewpoint Variation | Success Rate Drop | **<12%** | - | - | 5%-37% |
| C2R-Envs Distractor Objects | Success Rate | **Outperforming by 20-46%** | - | - | Baseline |

### Ablation Study
- **Failure data is crucial**: Moving from no failure data -> using BCE loss with failure data -> using failure prompts leads to step-by-step performance improvements. Failure prompts show a significant advantage over BCE, as BCE tends to disrupt the knowledge learned from human data.
- **K=3 is optimal**: Ablations for K from 1 to 5 show that the model is robust to the choice of K, with K=3 yielding the best performance.
- **Multi-source failure data is better**: A mixture of random exploration and near-success failure scenarios performs better than a single source of failure data.
- **More reasonable reward distribution**: Adapt2Reward's rewards exhibit clear separation between success and failure trajectories, whereas the BCE-trained model incorrectly assigns high scores to failed trajectories.

## Highlights & Insights
- **Failure is also knowledge**: By clustering failure videos and learning failure prompts, "failure causes" are encoded into the model as structured knowledge. This is much more sophisticated than simple BCE classification, as failure modes can transfer across different tasks.
- **Extremely low data requirements**: Requiring only 560 robot videos (including success and failure) per task, combined with human video data, the model generalizes to new environments and tasks.
- **Transferable prompt pool design**: The paradigm of a learnable prompt pool combined with clustering pseudo-labels can be extended to other scenarios requiring negative pattern modeling (e.g., quality inspection, anomaly classification).
- **Advantage of dense rewards**: Consistently outperforms hand-crafted binary rewards in C2R environments because the model provides continuous similarity scores, partially rewarding "near-success" behaviors.

## Limitations & Future Work
- **Simulation-only validation**: All experiments were conducted in MetaWorld and PyBullet, leaving sim-to-real transfer performance unverified.
- **Limited task complexity**: The evaluated tasks are relatively simple tabletop manipulations (e.g., pushing cups, opening/closing drawers), while long-horizon complex tasks have not been verified.
- **Fixed number of clusters K**: The number of failure modes can vary significantly across different tasks; a fixed K is somewhat rigid, and adaptive clustering could be considered.
- **Manual selection of human data still required**: Deciding which human tasks in SS-V2 to match with robot tasks relies on manual selection; automated task matching remains a potential area for improvement.
- **Lack of stronger video foundation models**: Leveraging more advanced video encoders released after 2024 (e.g., InternVideo2, VideoMAE v2) could provide better initial representations.

## Related Work & Insights
- **vs DVD**: DVD trains a domain-agnostic video discriminator to measure robot-human video similarity but ignores failure signals entirely. Adapt2Reward explicitly models failure modes via failure prompts, outperforming DVD by around 23% in task generalization.
- **vs Concept2Robot**: Concept2Robot uses a 174-class classifier pre-trained on SS-V2 as a reward, representing objects uniformly as "[Something]", which discards specific object interaction details. Adapt2Reward preserves language-conditioned characteristics and exhibits significantly stronger robustness to viewpoints and distracting objects.
- **vs LIV**: LIV learns rewards via cross-modal embeddings and temporal consistency but does not exploit the structure of failure data. Adapt2Reward's failure prompt mechanism provides more fine-grained success/failure discrimination.

## Insights & Connections
- **Cross-domain contrastive learning + domain-specific prompt paradigm**: Can be applied to VLM zero-shot transfer scenarios, such as medical imaging (which lacks data but has extensive general-domain pre-training).
- **Learnable prompt pool + clustering**: This pattern also shows potential in continual learning and domain adaptation.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of failure prompt + clustering is very clever, treating failure modes as transferable knowledge.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes environment generalization, task generalization, and ablations, but lacks real-world robot experiments.
- Writing Quality: ⭐⭐⭐⭐ Clearly motivated, systematically described methodology, and well-designed figures.
- Value: ⭐⭐⭐⭐ Provides an effective solution for the VLM-to-robotic-reward path, and the concept of failure prompts is highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PAVE: Patching and Adapting Video Large Language Models](../../CVPR2025/video_understanding/pave_patching_and_adapting_video_large_language_models.md)
- [\[ECCV 2024\] Vamos: Versatile Action Models for Video Understanding](vamos_versatile_action_models_for_video_understanding.md)
- [\[ECCV 2024\] PiTe: Pixel-Temporal Alignment for Large Video-Language Model](pite_pixel-temporal_alignment_for_large_video-language_model.md)
- [\[ICML 2025\] MoMa: Modulating Mamba for Adapting Image Foundation Models to Video Recognition](../../ICML2025/video_understanding/moma_modulating_mamba_for_adapting_image_foundation_models_to_video_recognition.md)
- [\[ECCV 2024\] Towards Model-Agnostic Dataset Condensation by Heterogeneous Models](towards_model-agnostic_dataset_condensation_by_heterogeneous_models.md)

</div>

<!-- RELATED:END -->
