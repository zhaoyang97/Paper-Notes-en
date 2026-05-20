---
title: >-
  [Paper Note] Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment
description: >-
  [CVPR 2026][Reinforcement Learning][lifelong imitation learning] This paper proposes a lifelong imitation learning framework that stores and replays compact representations in the feature space of frozen encoders via Mul…
tags:
  - "CVPR 2026"
  - "Reinforcement Learning"
  - "lifelong imitation learning"
  - "multimodal latent replay"
  - "incremental feature adjustment"
  - "catastrophic forgetting"
  - "LIBERO"
date: 2026-05-08
content_hash: 36e2f69b5fbc0559
---

# Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment

**Conference**: CVPR 2026
**arXiv**: [2603.10929](https://arxiv.org/abs/2603.10929)  
**Code**: [https://github.com/yfqi/lifelong_mlr_ifa](https://github.com/yfqi/lifelong_mlr_ifa)  
**Area**: Reinforcement Learning
**Keywords**: lifelong imitation learning, multimodal latent replay, incremental feature adjustment, catastrophic forgetting, LIBERO

## TL;DR
This paper proposes a lifelong imitation learning framework that stores and replays compact representations in the feature space of frozen encoders via Multimodal Latent Replay (MLR), and introduces an Incremental Feature Adjustment (IFA) mechanism that employs angular distance constraints to maintain inter-task separability. The method achieves AUC improvements of 10–17 points and reduces forgetting by up to 65% on the LIBERO benchmark.

## Background & Motivation
Imitation learning (IL) enables robots to acquire behaviors by observing human demonstrations, yet real-world environments are inherently dynamic—new objects, goals, and contexts emerge continuously. Standard IL assumes a fixed task set and does not support dynamic expansion. Lifelong imitation learning (LIL) aims to enable agents to continually acquire new skills while retaining previously learned ones, with catastrophic forgetting as the central challenge. Existing LIL methods fall into three categories: (1) experience replay methods (e.g., LOTUS stores raw trajectories; CRIL generates replay data via GANs), which incur large memory overhead and are sensitive to inter-task similarity; (2) progressive model expansion methods (e.g., TAIL trains a dedicated adapter per task), which require task IDs at test time; and (3) distillation-based methods (e.g., M2Distill), which involve complex pipelines. These approaches suffer from low storage efficiency, dependence on PEFT or task IDs, or costly distillation procedures.

## Core Problem
Lifelong imitation learning presents two core challenges: (1) **Storage efficiency** — conventional experience replay stores raw trajectories (high-dimensional image and state sequences), incurring substantial memory overhead; and (2) **Representation interference** — latent representations of new tasks may overlap with those of old tasks, causing inter-task interference in the shared embedding space. The key question is: how can efficient lifelong learning be achieved with a simple pipeline, without PEFT, task IDs, or knowledge distillation?

## Method

### Overall Architecture
A two-stage paradigm is adopted: multi-task pretraining first establishes a shared representation, followed by a lifelong learning stage in which new tasks are learned incrementally. The policy network consists of three modality encoders (CLIP visual encoder, CLIP text encoder, and MLP state encoder), FiLM modulation layers, a GPT-2 temporal decoder, and a GMM policy head. During pretraining, all modules are trainable (CLIP is fine-tuned with LoRA rank-8). During lifelong learning, all encoders and FiLM layers are frozen; only the temporal decoder and policy head are updated. Inputs are agent-view images, eye-in-hand images, language instructions, and robot states; outputs are 5-component GMM action distributions.

### Key Designs
1. **Multimodal Latent Replay (MLR)**: Rather than storing raw trajectories, the method stores multimodal latent features $\mathbf{H} \in \mathbb{R}^{M \times L \times E}$ (where $M$ = number of modalities, $L$ = timesteps, $E$ = embedding dimension) produced by the frozen encoders and FiLM layers, along with corresponding actions. Upon arrival of a new task, latent representations of previous tasks are sampled from the buffer and trained jointly with current data. This bypasses encoder forward passes and incurs far less memory overhead than storing raw images. The buffer is allocated uniformly across tasks; approximately 5 demonstrations per task are stored (with sampling probability 0.5).
2. **Incremental Feature Adjustment (IFA)**: To address inter-task representation drift, a reference embedding (derived from language embeddings, which are fixed and stable) is maintained per task. The IFA loss penalizes cases where the global representation of the current task $g_t(T_k)$ is farther from its own reference than from that of an old task:
$$\mathcal{L}_{IFA} = \max(0, d(g_t(T_k), h^{(r)}(T_k)) - d(g_t(T_k), h^{(r)}(T_j)) + \delta)$$
This is essentially a triplet loss that pulls the current task's representation closer to its own anchor while pushing it away from old task anchors.
3. **Adaptive Angular Distance Margin**: A fixed margin $\delta$ cannot adapt to varying inter-task similarity. The paper defines $\delta = \alpha \cdot \text{arccos}(\text{cos\_sim})$ as a proportion of the angular distance between task reference embeddings. Angular distance (arccos) offers better resolution than cosine distance in high-similarity regions—when two representations are very close, cosine distance saturates, whereas angular distance retains discriminative power. The scaling factor $\alpha$ ranges from 0.1 to 0.7 across datasets.
4. **Task Pair Selection Strategy**: Rather than applying IFA constraints to all task pairs, the method computes average cosine similarity between tasks across the agent-view and language modalities, selecting only pairs that rank in the top 50% most similar on both modalities simultaneously. Each selected pair must include one new task and one old task.

### Loss & Training
The overall training objective is $\mathcal{L} = \mathcal{L}_{BC} + \lambda_{IFA} \mathcal{L}_{IFA}$, where $\lambda_{IFA}=0.1$. $\mathcal{L}_{BC}$ is the behavior cloning loss (negative log-likelihood based on GMM policy head outputs). AdamW optimizer is used with learning rate $10^{-4}$, linear scheduler, batch size 10, and 100 training epochs. Configurations are identical across pretraining and lifelong learning stages.

## Key Experimental Results

| Dataset | Metric | MLR+IFA (Ours) | LOTUS (Prev. SOTA) | ISCIL | Gain |
|--------|------|------|----------|------|------|
| LIBERO-OBJECT | FWT↑ | 84.6 | 74.0 | 71.7 | +10.6 vs LOTUS |
| LIBERO-OBJECT | NBT↓ | 11.4 | 11.0 | 11.9 | Comparable |
| LIBERO-OBJECT | AUC↑ | 79.4 | 65.0 | 66.3 | +14.4 vs LOTUS |
| LIBERO-GOAL | FWT↑ | 80.0 | 61.0 | 70.4 | +19.0 vs LOTUS |
| LIBERO-GOAL | NBT↓ | 6.9 | 30.0 | 19.4 | −64% vs ISCIL |
| LIBERO-GOAL | AUC↑ | 77.2 | 56.0 | 60.5 | +16.7 vs ISCIL |
| LIBERO-50 | FWT↑ | 60.8 | 39.0 | 47.8 | +13.0 vs ISCIL |
| LIBERO-50 | NBT↓ | 8.6 | 43.0 | 15.0 | −43% vs ISCIL |
| LIBERO-50 | AUC↑ | 56.1 | 45.0 | 37.7 | +11.1 vs LOTUS |

### Ablation Study
- **MLR alone** already substantially surpasses prior SOTA (AUC 77.6 vs. LOTUS 65 on OBJECT); adding IFA yields further improvement (79.4).
- **Modality similarity selection**: The combination of language + agent-view performs best (AUC 79.4); single-modality or other combinations are consistently inferior.
- **Task pair ratio**: Top 50% is optimal; 33.3% is insufficient, while 66.6% over-regularizes and increases NBT.
- **Reference selection**: Language embeddings as references outperform global mean embeddings, as language embeddings remain fixed while global means drift during training.
- **Buffer size**: Reducing storage probability from 0.5 to 0.1 drops AUC from 79.4 to 76.6, demonstrating the importance of sufficient storage.
- **Angular distance vs. cosine distance**: Angular distance consistently outperforms cosine distance with lower variance.
- **Full fine-tuning vs. LoRA for temporal decoder**: Full fine-tuning is far superior to LoRA (AUC 79.4 vs. ≤54.2), indicating that the temporal decoder requires sufficient capacity.
- **FiLM layers**: Removing FiLM causes a dramatic performance drop (AUC from 79.4 to 41.6), confirming that task-conditional modulation is critical.

## Highlights & Insights
- **Minimal pipeline**: Frozen pretrained encoders + training only the temporal decoder + latent replay, requiring no distillation, PEFT, or task IDs—simple yet highly effective.
- **Elegant IFA formulation**: The use of arccos's amplification properties in high-similarity regions, combined with an adaptive margin, accommodates task pairs of varying difficulty.
- **Language embeddings as stable anchors**: The approach leverages the stability of language descriptions under frozen encoders as reference points, avoiding anchor drift during training.
- **Storage efficiency**: Memory consumption of latent replay is approximately 188 MB (OBJECT) / 121 MB (GOAL), far below the cost of storing raw images.

## Limitations & Future Work
- Validation is limited to the LIBERO simulation benchmark; real-robot experiments have not been conducted.
- LoRA fine-tuning of CLIP during pretraining may constrain generalization to out-of-domain scenarios.
- $\alpha$ requires per-dataset tuning (0.1 / 0.3 / 0.7) with substantial variation across optimal values; automatic selection of $\alpha$ remains an open problem.
- Task sequences are relatively short (only 4 new tasks in OBJECT/GOAL); scalability to longer sequences has yet to be verified.
- Cross-domain (simulator → real) transfer capability is unexplored.
- Latent replay depends on the representational quality of the frozen encoders, which may become a bottleneck when encoders are insufficiently expressive for certain tasks.

## Related Work & Insights
- **vs. LOTUS**: LOTUS stores raw trajectories and employs open-vocabulary visual encoders for skill discovery, resulting in a complex pipeline. The proposed method uses frozen CLIP with latent replay, achieving a simpler design and comprehensively surpassing LOTUS (AUC improvements of 10–17 points).
- **vs. M2Distill**: M2Distill maintains a consistent latent space via multimodal distillation, requiring an additional teacher model and GMM alignment. The proposed method requires no distillation and directly regularizes the representation space via IFA, yielding a cleaner design and superior metrics on LIBERO-50.
- **vs. TAIL**: TAIL requires task IDs to select the appropriate adapter, making it unsuitable for task-agnostic settings. The proposed method substantially outperforms TAIL under the same evaluation protocol.

The latent replay paradigm is transferable to other multimodal continual learning settings (e.g., continual fine-tuning of VLMs). The adaptive angular margin design of IFA is applicable to any continual learning method that requires maintaining inter-class separability. The use of language embeddings as stable anchors is a concept worth borrowing in other cross-modal learning tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of MLR and IFA is novel in the LIL domain; the angle-based adaptive margin design demonstrates meaningful insight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Ablation studies are highly comprehensive, covering nearly all design choices, with UMAP visualizations and computational efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, derivations are complete, and figures are highly informative.
- Value: ⭐⭐⭐⭐ Achieves state-of-the-art results across all LIBERO benchmarks, releases code, and provides practical reference value for lifelong robot learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Latent Wasserstein Adversarial Imitation Learning](../../ICLR2026/reinforcement_learning/latent_wasserstein_adversarial_imitation_learning.md)
- [\[ACL 2026\] Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions](../../ACL2026/reinforcement_learning/controlling_multimodal_conversational_agents_with_coverage-enhanced_latent_actio.md)
- [\[CVPR 2026\] GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion](graspldp_towards_generalizable_grasping_policy_via_latent_diffusion.md)
- [\[CVPR 2026\] MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning](msrl_scaling_generative_multimodal_reward_modeling.md)
- [\[CVPR 2026\] Anticipatory Planning for Multimodal AI Agents](anticipatory_planning_for_multimodal_ai_agents.md)

</div>

<!-- RELATED:END -->
