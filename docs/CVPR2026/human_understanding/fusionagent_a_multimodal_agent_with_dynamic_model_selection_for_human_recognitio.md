---
title: >-
  [Paper Note] FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition
description: >-
  [CVPR 2026][Human Understanding][model fusion] This paper proposes FusionAgent, an intelligent agent framework based on a multimodal large language model (MLLM) for dynamic sample-level model selection in whole-body biometric recognition. Each expert model (face recognition / gait recognition / person re-identification) is encapsulated as a callable tool. Through reinforcement fine-tuning (RFT), the agent learns to adaptively select the optimal model combination for each test sample based on its characteristics. Combined with the newly proposed ACT score fusion strategy, FusionAgent significantly outperforms existing state-of-the-art fusion methods.
tags:
  - CVPR 2026
  - Human Understanding
  - model fusion
  - multimodal large language model
  - dynamic model selection
  - biometric recognition
  - reinforcement fine-tuning
date: 2026-05-08
content_hash: 1a77e3d6d4ec6904
---

# FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition

**Conference**: CVPR 2026
**arXiv**: [2603.26908](https://arxiv.org/abs/2603.26908)
**Code**: [https://github.com/FusionAgent](https://github.com/FusionAgent) (project page)
**Area**: Human Body Understanding
**Keywords**: model fusion, multimodal large language model, dynamic model selection, biometric recognition, reinforcement fine-tuning

## TL;DR
This paper proposes FusionAgent, an intelligent agent framework based on a multimodal large language model (MLLM) for dynamic sample-level model selection in whole-body biometric recognition. Each expert model (face recognition / gait recognition / person re-identification) is encapsulated as a callable tool. Through reinforcement fine-tuning (RFT), the agent learns to adaptively select the optimal model combination for each test sample based on its characteristics. Combined with the newly proposed ACT score fusion strategy, FusionAgent significantly outperforms existing state-of-the-art fusion methods.

## Background & Motivation

1. **Background**: Whole-body human recognition requires fusing multiple biometric modalities, including face, gait, and body shape. Different expert models (FR/GR/ReID) excel in different scenarios and are typically integrated via score fusion. Existing fusion methods include rule-based approaches (Z-score, Min-max, etc.) and learning-based approaches (QME, etc.); however, all of them adopt a fixed model combination strategy—applying the same full set of models to every test sample.

2. **Limitations of Prior Work**: (1) Static fusion assumes all models contribute meaningfully to every sample, yet face recognition models provide no useful information for subjects facing away from the camera. (2) Scores from low-quality inputs contaminate the fusion result; even quality-aware methods such as QME cannot fully prevent low-quality model scores from affecting the final output. (3) Invoking all models for every sample is computationally wasteful and unnecessary.

3. **Key Challenge**: The optimal model combination is sample-dependent—inputs of different quality, viewing angle, and resolution require different model subsets. Applying the full model ensemble uniformly to all samples wastes computational resources and degrades fusion quality by introducing low-quality scores.

4. **Goal**: (1) How to adaptively select the optimal model subset for each sample? (2) How to effectively fuse scores from heterogeneous models that are dynamically selected?

5. **Key Insight**: Model selection is formulated as a tool-invocation decision problem for an MLLM agent. The agent observes the characteristics of an input sample, performs reasoning, decides which models to invoke, and learns an optimal policy from outcome feedback via reinforcement learning.

6. **Core Idea**: An MLLM agent performs sample-level dynamic model selection, transforming the decision of "which models to use" from a hand-crafted rule into a learnable reasoning process. This is coupled with an anchor-based top-$k$ score fusion strategy to achieve robust selective ensemble.

## Method

### Overall Architecture
The FusionAgent pipeline proceeds as follows. Each biometric model is encapsulated as a tool that provides a score vector and predicted labels. The MLLM agent (based on Qwen2.5-VL-3B) receives multimodal input and performs multi-turn ReAct-style reasoning (reasoning → action → observation → …), iteratively selecting models, observing intermediate results, and deciding whether to invoke additional models or emit a final answer. The agent is reinforcement fine-tuned via GRPO (Group Relative Policy Optimization) using a composite reward function comprising format reward, tool-success reward, accuracy reward, and metric reward. Final scores are integrated through the ACT (Anchor-based Confidence Top-$k$) fusion strategy.

### Key Designs

1. **ReAct-Style Multi-Turn Tool Invocation**

    - **Function**: Enables the agent to select models incrementally, adjusting its strategy at each step based on prior results.
    - **Mechanism**: A ReAct (reason-then-act) style multi-turn controller is adopted rather than generating a complete plan in a single pass. At each turn, the agent reasons over the current sample characteristics and accumulated results, then selects one model to invoke and receives its score vector and predicted labels. The agent then decides whether to call additional models or output the final answer directly. The maximum number of turns is limited to 4.
    - **Design Motivation**: (1) Decomposing the exponential model-combination search space into single-model selections per step greatly reduces learning difficulty. (2) The agent can dynamically adjust its strategy based on intermediate results (e.g., invoking a second model when the first prediction is uncertain). (3) This design supports effective credit assignment.

2. **Metric-Based Reward**

    - **Function**: Guides the agent to learn effective model selection policies that maximize overall performance metrics.
    - **Mechanism**: $N=6$ rollouts are sampled, each yielding a model combination $M_{o_i}$. For these combinations, a fraction $\gamma=0.8$ of sample combinations are kept unchanged, while the remaining 20% are randomly resampled to encourage exploration. The ACT strategy then fuses the score matrices under these selections, and a composite metric $R_{\mathrm{mat}} = \mathrm{Rank} + \mathrm{mAP} + \mathrm{TAR} - \mathrm{FNIR}$ is computed over the entire training set. This dataset-level reward enables the agent to understand the impact of different model combinations on global performance.
    - **Design Motivation**: Unlike per-sample accuracy rewards, the metric reward accounts for threshold-dependent indicators such as TAR@FAR and FNIR@FPIR, which can only be computed at the dataset level, thereby better reflecting real-world deployment requirements.

3. **ACT (Anchor-based Confidence Top-$k$) Score Fusion**

    - **Function**: Achieves robust fusion of scores from heterogeneous models that are dynamically selected.
    - **Mechanism**: The first model selected by the agent serves as the "anchor model" $m_a$, and its score vector is retained in full. For all other selected models, Z-score normalization is first applied to address scale inconsistency; then only the top-$k$ highest-scoring entries of each model contribute to the fusion (i.e., $c_{m,q,g} = z_{m,q,g} \cdot s_{m,q,g}$ if $g \in \mathcal{T}_{m,q}$, otherwise $0$). The final fused score is $\mathbf{s}_q' = \frac{1}{1 + |\mathbf{M}_q|}(\mathbf{s}_{m_a,q} + \sum_{m \in \mathbf{M}_q} \mathbf{c}_{m,q})$.
    - **Design Motivation**: The anchor model provides a global ranking foundation, while top-$k$ filtering prevents low-confidence impostor scores from inflating non-match scores. This "anchor + sparse contribution" strategy effectively addresses score scale misalignment and heterogeneity under dynamic model selection.

### Loss & Training
- GRPO optimization with composite reward $R = R_f + R_{\mathrm{tool}} + R_{\mathrm{acc}} + R_{\mathrm{mat}}$
- Base model: Qwen2.5-VL-3B with LoRA (rank=64, α=128)
- Learning rate $2 \times 10^{-5}$ (linear decay), KL coefficient $\beta = 0.04$
- Training: 200 steps on 4 H100 GPUs, approximately 4 hours
- All biometric model weights are frozen throughout training

## Key Experimental Results

### Main Results — CCVID Dataset

| Method | Rank1↑ | mAP↑ | TAR↑ | FNIR↓ |
|--------|--------|------|------|-------|
| AdaFace (single model) | 94.0 | 87.9 | 75.7 | 13.0±3.5 |
| Z-score | 92.2 | 90.6 | 73.9 | 15.1±1.5 |
| QME (Prev. SOTA) | **94.1** | 90.8 | 76.2 | 12.3±1.4 |
| **FusionAgent (CoT)** | 93.4 | **92.6** | **85.9** | **10.1±1.5** |

TAR improves from 76.2% to 85.9% (+9.7%), and FNIR decreases from 12.3% to 10.1%.

### LTCC Dataset

| Method | Rank1↑ | mAP↑ | TAR↑ | FNIR↓ |
|--------|--------|------|------|-------|
| QME | 73.8 | 39.6 | 35.0 | 64.3±8.0 |
| **FusionAgent (CoT)** | **75.5** | **41.0** | **37.0** | **50.0±8.5** |

FNIR decreases from 64.3% to 50.0% (−14.3%), representing a substantial improvement in open-set retrieval performance.

### Ablation Study

| Configuration | Rank1 | mAP | TAR | FNIR |
|---------------|-------|-----|-----|------|
| QME (baseline) | 73.8 | 39.6 | 35.0 | 64.3 |
| Agent + Z-score | 74.8 | **41.7** | **37.1** | 63.7 |
| Agent + FarSight | 74.8 | **41.7** | **37.2** | 62.5 |
| Agent + ACT (Ours) | **75.5** | 41.4 | 36.5 | **51.0** |

- Agent-based selection combined with any fusion method outperforms QME, demonstrating that dynamic selection is the key factor.
- ACT achieves the largest advantage on FNIR (−11.5%), as top-$k$ filtering effectively suppresses impostor scores.

### Key Findings
- **Dynamic model selection is the primary driver of performance gains**: Even with simple Z-score fusion, agent-based selection outperforms QME.
- **Hard selection (using all models) is inferior to agent-based dynamic selection**: This confirms that "more models ≠ better performance" and that selective fusion is essential.
- **FNIR benefits the most**: In open-set retrieval scenarios, impostor score noise is effectively controlled by top-$k$ filtering.
- **Cross-domain generalization**: Training on MEVID and testing on LTCC in a zero-shot setting still yields performance close to in-domain results.
- **Model selection statistics reveal dataset characteristics**: FusionAgent predominantly selects AdaFace on CCVID (where faces are clearly visible), and ReID models on LTCC/MEVID (low-quality surveillance footage).

## Highlights & Insights
- **Reformulating model fusion as an agent tool-selection problem**: This framework elevates years of score fusion research to a new paradigm—rather than designing better fusion formulas, the AI itself decides which models to use and how to combine them.
- **ReAct multi-turn design**: Decomposing the $2^Z$ model combination search space into single selections per step makes RL training feasible, while also allowing the agent to dynamically adjust its strategy based on intermediate results.
- **Metric reward design**: Dataset-level evaluation indicators (TAR@FAR, FNIR@FPIR) are elegantly encoded as RL reward signals, enabling the agent to learn a globally optimized strategy that transcends per-sample accuracy.
- **CoT reasoning provides interpretability**: The agent's reasoning chain explains why a particular model combination is selected (e.g., "a clear frontal face is detected; selecting the face recognition model as anchor"), enhancing system trustworthiness.

## Limitations & Future Work
- Based on Qwen2.5-VL-3B, inference is relatively slow (2.81 s/sample in CoT mode), which may be prohibitive in real-time applications.
- The reasoning capacity of the 3B model is limited; larger MLLMs may yield better selection policies.
- The current tool set is a fixed collection of biometric models; generalizing to new models requires retraining the agent.
- The top-$k$ hyperparameter in ACT requires tuning on the training set, and different values of $k$ are needed for different datasets.
- Exhaustive search for the sample-level optimal model combination is computationally infeasible, making it impossible to quantify the gap between the agent's decisions and the theoretical optimum.

## Related Work & Insights
- **vs. QME**: QME performs quality-aware weighted fusion but still employs all models. FusionAgent surpasses QME solely through dynamic subset selection, indicating that "which models to select" is more critical than "how to weight each model."
- **vs. traditional score fusion (Z-score, FarSight)**: With agent-based dynamic selection, even simple fusion methods outperform complex learned approaches, suggesting that fusion strategy complexity may not be the bottleneck—model selection is.
- **vs. SapiensID**: End-to-end multimodal recognition models lack modularity and interpretability. FusionAgent achieves interpretable, modular fusion through the agent framework.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing MLLM agents into model fusion/selection is novel, and the metric reward design is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, multiple baselines, comprehensive ablation, cross-domain evaluation, statistical analysis, and qualitative case studies.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and the framework is well-structured, though some equations could be presented more concisely.
- Value: ⭐⭐⭐⭐ The agent + tool-use paradigm offers instructive insights for multi-model fusion and is extensible to other scenarios requiring model selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] A Two-Stage Dual-Modality Model for Facial Expression Recognition](a_two_stage_dual_modality_model_for_facial_expression_recognition.md)
- [\[CVPR 2026\] Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](team_leya_in_10th_abaw_competition_multimodal_ambi.md)
- [\[ICCV 2025\] EgoAgent: A Joint Predictive Agent Model in Egocentric Worlds](../../ICCV2025/human_understanding/egoagent_a_joint_predictive_agent_model_in_egocentric_worlds.md)
- [\[CVPR 2026\] ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body](vibes_a_conversational_agent_with_behaviorally_intelligent_3d_virtual_body.md)
- [\[CVPR 2026\] 4DSurf: High-Fidelity Dynamic Scene Surface Reconstruction](textit4dsurf_high-fidelity_dynamic_scene_surface_reconstruction.md)

</div>

<!-- RELATED:END -->
