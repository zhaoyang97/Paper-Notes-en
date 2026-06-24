---
title: >-
  [Paper Note] PreLAR: World Model Pre-training with Learnable Action Representation
description: >-
  [ECCV 2024][LLM Pretraining][World Model Pre-training] This paper proposes PreLAR to bridge the gap between action-free pre-training and action-conditioned fine-tuning for world models. By encoding implicit action representations from adjacent frames and designing an action-state consistency loss during unsupervised pre-training on action-free videos, PreLAR significantly improves the sample efficiency of downstream visual control tasks.
tags:
  - "ECCV 2024"
  - "LLM Pretraining"
  - "World Model Pre-training"
  - "Learnable Action Representation"
  - "Unsupervised Pre-training"
  - "Model-Based Reinforcement Learning"
  - "Sample Efficiency"
date: 2026-05-08
content_hash: 38b8c10df1f98581
---

# PreLAR: World Model Pre-training with Learnable Action Representation

**Conference**: ECCV 2024  
**Code**: [https://github.com/zhanglixuan0720/PreLAR](https://github.com/zhanglixuan0720/PreLAR)  
**Area**: LLM Pre-training  
**Keywords**: World Model Pre-training, Learnable Action Representation, Unsupervised Pre-training, Model-Based Reinforcement Learning, Sample Efficiency

## TL;DR
This paper proposes PreLAR to bridge the gap between action-free pre-training and action-conditioned fine-tuning for world models. By encoding implicit action representations from adjacent frames and designing an action-state consistency loss during unsupervised pre-training on action-free videos, PreLAR significantly improves the sample efficiency of downstream visual control tasks.

## Background & Motivation

**Background**: Model-Based Reinforcement Learning (MBRL) makes decisions by constructing a world model of the environment. World models learn the dynamics of the environment, which typically requires a large volume of interactions with the real environment. Recent methods like APV propose unsupervised pre-training of world models from large-scale unlabeled videos, allowing high-quality world models to be fine-tuned with fewer environmental interactions.

**Limitations of Prior Work**: Existing unsupervised pre-training methods (such as APV) only pre-train the world model as a video prediction model—predicting the next frame given the current frame. However, the world model used in downstream tasks is action-conditional—predicting the next state given the current state and action. The lack of action conditioning in the pre-training stage, combined with its requirement in the fine-tuning stage, creates a gap that limits the effect of pre-training on enhancing world model capabilities.

**Key Challenge**: Unlabeled videos lack action information, preventing the pre-trained world model from learning the core knowledge of "how actions affect state changes." Consequently, when fine-tuning begins, the model must learn the relationship between actions and state changes from scratch, making it difficult to fully transfer the video prediction capabilities accumulated during pre-training.

**Goal**: (1) How to perform action-conditional world model pre-training on videos without action labels; (2) How to align the implicit action representations from pre-training with real actions during fine-tuning.

**Key Insight**: The transition between two adjacent frames implicitly contains "action" information—i.e., what caused the change from the current frame to the next frame. An implicit action representation can be constructed by encoding this inter-frame change, allowing the pre-training to also proceed in an action-conditional manner.

**Core Idea**: Extract implicit action representations from adjacent frames in action-free videos for action-conditional world model pre-training, and ensure alignment between implicit and real actions in the representation space via an action-state consistency loss.

## Method

### Overall Architecture
PreLAR extends the Recurrent State Space Model (RSSM) world model architecture. During the pre-training stage, an implicit action representation $\hat{a}_t$ is encoded from adjacent frame pairs $(o_t, o_{t+1})$ of unlabeled videos, and the world model is trained to predict $o_{t+1}$ conditioned on $\hat{a}_t$. During the fine-tuning stage, the real action $a_t$ is mapped to the same representation space as the implicit action using an action encoder, which is then used to train the world model on downstream control tasks.

### Key Designs

1. **Implicit Action Encoder**:

    - **Function**: Extract implicit action representations from observations of two adjacent timesteps.
    - **Mechanism**: An encoder network $E_{ia}$ is designed to take the hidden states of two adjacent timesteps $(h_t, h_{t+1})$ as input and output the implicit action representation $\hat{a}_t = E_{ia}(h_t, h_{t+1})$. Hidden states are obtained from observations $o_t$ and $o_{t+1}$ via the world model encoder. The encoder is implemented using an MLP that maps the concatenation of the two states into a low-dimensional action representation space. This implicit action representation captures information about "what changes occurred from $o_t$ to $o_{t+1}$."
    - **Design Motivation**: In the absence of action labels, inter-frame changes are the most natural proxies for "actions." Inferring actions from state transitions allows the pre-training to be action-conditioned, matching the format of fine-tuning.

2. **Action-State Consistency Loss**:

    - **Function**: Optimize implicit action representations via self-supervision to better align with the semantics of real actions.
    - **Mechanism**: A consistency constraint is designed to require that the implicit action representation can be correctly mapped back to the state change by another decoder. Specifically, given $\hat{a}_t$ and the current state $h_t$, the next state $\hat{h}_{t+1}$ is predicted via a transition model, and $\hat{h}_{t+1}$ is constrained to be consistent with the true $h_{t+1}$. The loss function is formulated as $\mathcal{L}_{asc} = \|h_{t+1} - T(h_t, \hat{a}_t)\|^2$. This requires the implicit action representation to contain sufficient information to describe the state transition.
    - **Design Motivation**: Without constraints, the implicit action encoder might learn degenerate solutions (e.g., ignoring action information). The consistency loss forces the implicit action to contain key information about the state transition, which is aligned with the essence of real actions—driving state changes.

3. **Action Space Alignment Fine-tuning**:

    - **Function**: Map real actions to the established implicit action representation space from pre-training during the fine-tuning stage.
    - **Mechanism**: An action encoder $E_a$ is designed to map the low-dimensional real action $a_t$ (such as joint angles, torques, etc.) to a representation $\tilde{a}_t = E_a(a_t)$ with the same dimension as the implicit action. This allows the transition module of the world model to directly use $\tilde{a}_t$ for state prediction without rebuilding the action-state mapping from scratch. The pre-trained transition model weights are frozen or updated with a small learning rate in the early stages of fine-tuning.
    - **Design Motivation**: By aligning the action spaces, the knowledge of "how actions affect states" learned during pre-training can seamlessly transfer to the fine-tuning stage, avoiding the overhead of learning action-conditioning from scratch.

### Loss & Training
The total pre-training loss is formulated as: $\mathcal{L} = \mathcal{L}_{recon} + \mathcal{L}_{kl} + \lambda \mathcal{L}_{asc}$, where $\mathcal{L}_{recon}$ is the reconstruction loss (predicting the next frame), $\mathcal{L}_{kl}$ is the KL divergence regularization, $\mathcal{L}_{asc}$ is the action-state consistency loss, and $\lambda$ is a trade-off coefficient. During fine-tuning, the standard training pipeline of Dreamer is followed.

## Key Experimental Results

### Main Results

| Meta-World Task | Metric (Success Rate) | PreLAR | APV | Dreamer(scratch) |
|---------------|-------------|--------|-----|------------------|
| Drawer-Open | Success Rate | **0.95** | 0.82 | 0.65 |
| Button-Press | Success Rate | **0.91** | 0.79 | 0.58 |
| Window-Open | Success Rate | **0.88** | 0.74 | 0.52 |
| Hammer | Success Rate | **0.72** | 0.56 | 0.38 |
| Average (10 tasks) | Success Rate | **0.83** | 0.69 | 0.51 |

### Ablation Study

| Configuration | Average Success Rate | Description |
|------|----------|------|
| Full PreLAR | **0.83** | Full method |
| w/o Implicit Action (APV-style) | 0.69 | Degenerates to action-free conditional pre-training |
| w/o Consistency Loss | 0.75 | Implicit action degenerates, drops by 8% |
| w/o Action Alignment | 0.71 | Insufficient transfer of pre-trained knowledge |
| Random Action instead of Implicit Action | 0.68 | Validates the effectiveness of implicit actions |

### Key Findings
- The introduction of implicit action representation is the largest contributor to performance improvement, validating the necessity of introducing action conditioning during pre-training.
- The action-state consistency loss is crucial to prevent representation degeneracy; without it, the implicit action encoder tends to ignore inter-frame differences.
- PreLAR significantly outperforms APV and Dreamer trained from scratch in terms of sample efficiency, especially when the number of interactions is limited (<100k steps).
- In more complex tasks (such as Hammer), the advantage of PreLAR is more pronounced, indicating that action-conditioned pre-training is more helpful for learning complex dynamics.

## Highlights & Insights
- **Extracting "implicit actions" from "action-free" videos** is an elegant approach. Inter-frame change serves as the most natural action encoding, making this insight both intuitive and highly effective.
- **Consistency loss as a self-supervised signal** elegantly ensures the semantic quality of implicit actions, learning meaningful action representations without any action labels. This technique can be transferred to other tasks requiring inference of hidden variables from observations.
- PreLAR demonstrates that "alignment of structure" (matching the conditioning forms between pre-training and fine-tuning) is more critical than "data quantity" in world model pre-training.

## Limitations & Future Work
- Currently validated only in the Meta-World simulation environment, without testing in real robotic environments.
- The dimension selection for the implicit action representation requires manual tuning and may need to match the dimension of the real action space.
- The source and quality of pre-training videos impact performance, but the discussion on this in the paper is limited.
- Combining implicit action representations with language instructions could be explored to achieve cross-modal world model pre-training.
- Multi-step prediction (rather than single-step adjacent frames) could potentially yield more hierarchical action representations.

## Related Work & Insights
- **vs APV**: APV pre-trains the world model as pure video prediction without action conditioning. PreLAR bridges the gap between pre-training and fine-tuning by introducing implicit actions, serving as a direct improvement over APV.
- **vs Dreamer**: Dreamer trains the world model from scratch, which requires extensive environment interaction. PreLAR significantly improves sample efficiency through pre-training, representing a "Dreamer + efficient pre-training" approach.
- **vs VPT (Video PreTraining)**: VPT uses an inverse dynamics model to predict action labels from videos in Minecraft. Its approach is similar to PreLAR but operates at the policy model level rather than the world model level.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using implicit action representation for world model pre-training is novel and intuitive.
- Experimental Thoroughness: ⭐⭐⭐ Only validated in the Meta-World environment; the scenarios are limited.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear, and the methodology is highly logical.
- Value: ⭐⭐⭐⭐ Provides important inspiration for the world model pre-training field, outlining a direction for narrowing the pre-training-to-fine-tuning gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MR-PLIP: Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](../../CVPR2025/llm_pretraining/mr_plip_multi_resolution_pathology.md)
- [\[ECCV 2024\] Scaling Backwards: Minimal Synthetic Pre-training?](scaling_backwards_minimal_synthetic_pre-training.md)
- [\[ICML 2025\] Metadata Conditioning Accelerates Language Model Pre-training](../../ICML2025/llm_pretraining/metadata_conditioning_accelerates_language_model_pre-training.md)
- [\[ACL 2026\] KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates](../../ACL2026/llm_pretraining/koco_conditioning_language_model_pre-training_on_knowledge_coordinates.md)
- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](../../NeurIPS2025/llm_pretraining/nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)

</div>

<!-- RELATED:END -->
