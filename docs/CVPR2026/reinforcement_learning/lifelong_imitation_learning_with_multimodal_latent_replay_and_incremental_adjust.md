---
title: >-
  [Paper Note] Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment
description: >-
  [CVPR 2026][Reinforcement Learning][Lifelong Imitation Learning] This paper proposes a lifelong imitation learning framework that combines Multimodal Latent Replay (storing and replaying compact multimodal features in the latent space of a frozen encoder) with Incremental Feature Adjustment (an adaptive margin constraint based on angular distance to prevent inter-task representation drift), achieving 10–17 point AUC gains and 65% reduction in forgetting on the LIBERO benchmark.
tags:
  - CVPR 2026
  - Reinforcement Learning
  - Lifelong Imitation Learning
  - Multimodal Latent Replay
  - Incremental Feature Adjustment
  - Catastrophic Forgetting
  - Robot Manipulation
date: 2026-05-08
content_hash: 1c3ad7bff36f860f
---

# Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment

**Conference**: CVPR 2026
**arXiv**: [2603.10929](https://arxiv.org/abs/2603.10929)
**Code**: [https://github.com/yfqi/lifelong_mlr_ifa](https://github.com/yfqi/lifelong_mlr_ifa)
**Area**: Reinforcement Learning
**Keywords**: Lifelong Imitation Learning, Multimodal Latent Replay, Incremental Feature Adjustment, Catastrophic Forgetting, Robot Manipulation

## TL;DR
This paper proposes a lifelong imitation learning framework that combines Multimodal Latent Replay (storing and replaying compact multimodal features in the latent space of a frozen encoder) with Incremental Feature Adjustment (an adaptive margin constraint based on angular distance to prevent inter-task representation drift), achieving 10–17 point AUC gains and 65% reduction in forgetting on the LIBERO benchmark.

## Background & Motivation
Imitation learning (IL) enables robots to acquire behaviors by observing human demonstrations, yet real-world environments are dynamic, with new objects, goals, and scenes continuously emerging. Standard IL assumes a fixed task set and cannot handle a stream of new tasks. Lifelong imitation learning (LIL) requires continually acquiring new skills while retaining previously learned behaviors, with catastrophic forgetting as the central challenge.

Limitations of existing LIL methods: (1) experience replay methods (e.g., ER) store raw trajectory data, incurring large memory costs and sensitivity to the similarity between old and new tasks; (2) progressive model expansion methods (e.g., TAIL) require task identifiers at test time, limiting practical applicability; (3) distillation methods (e.g., M2Distill) involve complex architectures. More critically, when new and old tasks share overlapping representations in latent space, interference within the shared embedding space exacerbates forgetting.

**Core Idea**: (1) Extract multimodal features using a frozen pretrained encoder and perform replay in latent space rather than raw data space, substantially reducing storage overhead; (2) actively push new-task representations away from old-task reference embeddings via angular-distance margin constraints, preserving inter-task discriminability.

## Method

### Overall Architecture
Two-stage training: (1) **Multi-task pretraining** — jointly trains all modules (vision/language/state encoders, FiLM modulation layers, temporal decoder, policy head); (2) **Lifelong learning** — freezes the encoders and updates only the temporal decoder and policy head, using MLR and IFA for continual learning on new tasks.

### Key Designs

1. **Multimodal Latent Replay (MLR)**:

    - **Function**: During the lifelong learning stage, stores multimodal latent features output by the frozen encoder (rather than raw images and trajectories), and mixes these stored features with new data during training on new tasks.
    - **Mechanism**: A replay buffer $\mathcal{B} = \{(\mathbf{H}_n, a_n)\}_{n=1}^{N_B}$, where $\mathbf{H}_n \in \mathbb{R}^{M \times L \times E}$ is a multimodal latent feature sequence concatenating visual, language, and state modalities. When learning new task $T_k$, the model minimizes $\mathcal{L}_{BC}$ over $\mathcal{D}_k \cup \mathcal{B}$.
    - **Design Motivation**: Storing raw trajectories (high-dimensional image sequences) is prohibitively expensive. By freezing the encoder, latent features become stable and compact, requiring only ~37–188 MB per task (depending on sampling probability), far less than storing raw images. Freezing the encoder also eliminates the need for PEFT techniques such as LoRA.

2. **Incremental Feature Adjustment (IFA)**:

    - **Function**: Imposes angular-distance margin constraints on the representations of new and old tasks in latent space, preventing interference caused by representation drift.
    - **Mechanism**: $\mathcal{L}_{IFA} = \frac{1}{|\mathcal{P}|}\sum_{(j,k) \in \mathcal{P}} \max(0, d(g_t(T_k), h^{(r)}(T_k)) - d(g_t(T_k), h^{(r)}(T_j)) + \delta)$, where $d(a,b) = \arccos(\frac{a^\top b}{\|a\|_2 \|b\|_2})$ is the angular distance, and the margin $\delta = \alpha \cdot d(h^{(r)}(T_k), h^{(r)}(T_j))$ scales adaptively with the reference distance between tasks.
    - **Design Motivation**: When the global latent representation $g_t(T_k)$ of a new task is too close to the reference $h^{(r)}(T_j)$ of an old task, the shared temporal decoder confuses the two. IFA introduces a repulsive force pushing the new task away from old tasks while maintaining attraction toward its own reference. The adaptive margin avoids the poor fit of fixed margins to tasks of varying similarity.

3. **Task Pair Selection and Reference Choice**:

    - **Function**: Determines which task pairs require IFA constraints and what serves as the reference embedding for each task.
    - **Mechanism**: Cosine similarities are computed for all task pairs across the agent-view and language modalities; IFA is applied only to pairs that simultaneously rank in the top 50% of similarity in both modalities and include both a new and an old task. Language embeddings $h^{(l)}$ are used as references, as language descriptions provide stable, fixed task representations.
    - **Design Motivation**: Not all task pairs are susceptible to representation interference — only similar task pairs are prone to confusion. Targeted constraints are therefore more efficient.

### Loss & Training
Total loss during the lifelong learning stage: $\mathcal{L} = \mathcal{L}_{BC} + \lambda_{IFA} \mathcal{L}_{IFA}$, with $\lambda_{IFA} = 0.1$. The AdamW optimizer is used with a learning rate of $10^{-4}$, batch size of 10, and 100 training epochs. Buffer size is equivalent to 5 demonstrations per task. The backbone encoder is frozen CLIP.

## Key Experimental Results

### Main Results

| Method | LIBERO-OBJECT AUC↑ | LIBERO-GOAL AUC↑ | LIBERO-50 AUC↑ | LIBERO-GOAL NBT↓ |
|------|------|------|------|------|
| Sequential | 30.0 | 23.0 | 14.0 | 70.0 |
| ER | 49.0 | 47.0 | 36.0 | 36.0 |
| LOTUS | 65.0 | 56.0 | 45.0 | 30.0 |
| M2Distill | 69.0 | 57.0 | NA | 20.0 |
| ISCIL | 66.3 | 60.5 | 37.7 | 19.4 |
| **MLR + IFA** | **79.4** | **77.2** | **56.1** | **6.9** |

### Ablation Study

| Configuration | LIBERO-OBJ AUC | LIBERO-GOAL AUC | Notes |
|------|---------------|----------------|------|
| MLR only | 77.6 | 74.6 | Latent replay alone already substantially surpasses SOTA |
| MLR + IFA | 79.4 | 77.2 | IFA yields further improvement |
| Modality pair: Lan+AV | 79.4 | 77.2 | Best combination |
| Language only | 77.3 | 71.8 | Inferior to Lan+AV |
| Mean global as reference | 75.7 | 70.5 | Inferior to language reference |

### Key Findings
- MLR alone already substantially outperforms all prior SOTA: LIBERO-OBJECT AUC 77.6 vs. ISCIL 66.3; LIBERO-GOAL AUC 74.6 vs. 60.5.
- IFA yields consistent further improvements across all datasets; on LIBERO-GOAL, NBT decreases from 10.0 to 6.9 (31% forgetting reduction).
- On the most challenging LIBERO-50, MLR+IFA achieves an NBT of 8.6, far below LOTUS's 43.0, demonstrating particular effectiveness on long task sequences.
- Language + agent-view is the best modality combination for IFA task pair selection.

## Highlights & Insights
- The "frozen encoder + latent replay" design is remarkably simple and effective — it requires no PEFT, no distillation, and no open-vocabulary encoders, yet surpasses all methods that employ these complex techniques.
- The adaptive margin $\delta = \alpha \cdot d(h^{(r)}(T_k), h^{(r)}(T_j))$ naturally adapts to task structure: similar tasks receive smaller margins, while dissimilar tasks receive larger ones.
- Using language embeddings as reference anchors is an elegant design choice — language descriptions are stable and invariant, serving as natural task identifiers.

## Limitations & Future Work
- The approach depends on the quality of the pretrained CLIP encoder; fine-tuning may be necessary if the encoder representations are ill-suited to the target domain.
- The uniform random sampling strategy for the buffer may be suboptimal; priority sampling based on informativeness warrants exploration.
- The IFA hyperparameter $\alpha$ requires manual tuning across datasets (0.1–0.7); an automatic search mechanism is worth investigating.
- Evaluation is limited to the LIBERO benchmark; validation on additional real-world robot platforms is necessary.

## Related Work & Insights
- LOTUS proposes an open-vocabulary lifelong learning solution; this work demonstrates that frozen CLIP + latent replay is a simpler and more effective alternative.
- M2Distill preserves representational stability via multimodal distillation; IFA achieves a similar objective through angular-distance constraints with greater simplicity.
- The latent replay concept is generalizable to other multimodal systems requiring continual learning.
- TAIL's per-task LoRA scheme requires task IDs at test time; this work achieves task-agnostic LIL.
- ER stores raw trajectories, whereas MLR stores latent features, reducing storage overhead by over 90%.

## Supplementary Details
- The buffer employs a random sampling strategy, maintaining balanced allocation across all previously encountered tasks.
- FiLM layers modulate visual and state embeddings using language features, enabling task-conditioned representations.
- GPT-2 serves as the temporal decoder, mapping multimodal latent feature sequences to global representations.
- Experiments are conducted on a single A100 GPU, requiring minimal computational resources.
- In LIBERO-50, 5 new tasks are introduced per stage over 5 stages (25 new tasks total), thoroughly testing scalability over long task sequences.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The MLR+IFA design is concise and effective; the adaptive margin constraint is a noteworthy contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three LIBERO datasets, multiple baselines, and detailed ablations (modality, reference, ratio, buffer size).
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear, ablation analysis is systematic, and the IFA illustration is intuitive.
- **Value**: ⭐⭐⭐⭐⭐ AUC gains of 10–17 points and 65% NBT reduction represent substantial improvements; the method is simple and reproducible.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion](graspldp_towards_generalizable_grasping_policy_via_latent_diffusion.md)
- [\[CVPR 2026\] MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning](msrl_scaling_generative_multimodal_reward_modeling.md)
- [\[CVPR 2026\] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)
- [\[CVPR 2026\] Anticipatory Planning for Multimodal AI Agents](anticipatory_planning_for_multimodal_ai_agents.md)
- [\[CVPR 2026\] See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs](see_it_say_it_sorted_an_iterative_training-free_framework_for_visually-grounded_.md)

<!-- RELATED:END -->
