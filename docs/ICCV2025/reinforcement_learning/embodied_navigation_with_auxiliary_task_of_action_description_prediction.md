---
title: >-
  [Paper Note] Embodied Navigation with Auxiliary Task of Action Description Prediction
description: >-
  [ICCV 2025][Reinforcement Learning][Embodied Navigation] DescRL introduces action description generation as an auxiliary task for reinforcement learning-based navigation. By distilling knowledge from pretrained vision-la…
tags:
  - "ICCV 2025"
  - "Reinforcement Learning"
  - "Embodied Navigation"
  - "Action Description"
  - "Auxiliary Task"
  - "Knowledge Distillation"
  - "Semantic Audio-Visual Navigation"
date: 2026-05-08
content_hash: ac8889a07afc8cdc
---

# Embodied Navigation with Auxiliary Task of Action Description Prediction

**Conference**: ICCV 2025
**arXiv**: [2510.21809](https://arxiv.org/abs/2510.21809)
**Code**: None
**Area**: Reinforcement Learning / Embodied Navigation
**Keywords**: Embodied Navigation, Action Description, Auxiliary Task, Knowledge Distillation, Semantic Audio-Visual Navigation

## TL;DR

DescRL introduces action description generation as an auxiliary task for reinforcement learning-based navigation. By distilling knowledge from pretrained vision-language models to train an ADPredictor, the navigation agent simultaneously produces interpretable action descriptions and achieves improved navigation performance, attaining state-of-the-art results on Semantic Audio-Visual Navigation (SAVNav) and several other tasks.

## Background & Motivation

Multimodal robot navigation faces two key challenges:

**Interpretability vs. Performance Trade-off**: As navigation models grow increasingly complex, systems become black boxes. Interpretable systems typically fail to match the performance of non-interpretable counterparts.

**Absence of Ground-Truth Action Descriptions in RL**: In imitation learning (IL), instruction text can serve as a prediction target; however, RL lacks human-provided trajectory–instruction paired data, making direct extension of existing methods infeasible.

**Core Insight**: Through **knowledge distillation** from pretrained description generation models (VLMs), pseudo-labels can be obtained to introduce action description prediction as an auxiliary task in RL. Remarkably, this auxiliary task not only preserves navigation performance but consistently enhances it.

## Method

### Overall Architecture

DescRL proceeds in two phases:
- **Phase 1**: Pretrain an ADGenerator (Action Description Generator) to translate navigation observation sequences into natural language descriptions.
- **Phase 2**: During RL training, an ADPredictor (Action Description Predictor) serves as an auxiliary task, learning to predict the output of the ADGenerator.

Three types of action descriptions are defined:
- **P-AD (Past Action Description)**: Describes what the agent has done — aids in identifying object/spatial recognition errors.
- **F-AD (Future Action Description)**: Describes what the agent should do next — aids in improving planning capability.
- **PF-AD**: A combination of P-AD and F-AD.

### Key Designs

1. **ADGenerator Pretraining**: Trained on the R2R (Vision-and-Language Navigation) dataset. Given a visual observation sequence $V_0, \dots, V_T$ and action sequence $\mathbb{a}_1, \dots, \mathbb{a}_T$, visual features are extracted via CNN and concatenated with actions, then fed into a Transformer encoder-decoder to produce description tokens $w_1, \dots, w_l$. Training uses teacher-forcing and cross-entropy loss.

2. **ADPredictor Auxiliary Task**: The ADPredictor shares its Transformer encoder and decoder with the RL policy network. Task embeddings $E_{\text{RL}}^T, E_{\text{AD}}^T$ distinguish inputs for the two tasks. Training proceeds in two steps:

    - **Step 1 — Pretraining**: Only the action description prediction objective is optimized (using a pre-constructed trajectory–description dataset: ~100k samples for ObjNav, ~500k for SAVNav).
    - **Step 2 — Joint Training**: Navigation and action description prediction are learned simultaneously, with total loss $\mathcal{L}_{\text{RL}} + \lambda \mathcal{L}_{\text{CE}}$.

   Key implementation details:
    - The ADPredictor shares encoder/decoder weights with the policy, yielding more semantically meaningful observation encodings.
    - The ADGenerator is used only during training; at test time only the ADPredictor is required, introducing no additional inference overhead.
    - In F-DescRL, the ADGenerator receives future observations (along the shortest path) during training, while the ADPredictor predicts future action descriptions from past observations only.

3. **VLM as ADGenerator (Eliminating Dependence on Manual Annotations)**: VideoLLaMA2 or Qwen2.5-VL is used as the ADGenerator in a zero-shot manner, performing knowledge distillation from VLM outputs (soft targets) without any manually annotated data. Experiments demonstrate that DescRL improves navigation performance even without R2R data.

### Loss & Training

- RL algorithm: DD-PPO (ObjNav/SAVNav), DAgger (VLN).
- DescRL loss coefficient $\lambda = 0.1$.
- ADGenerator input history length $k+1 = 20$.
- Shared decoder layers: 2 (ObjNav/SAVNav); non-shared layers: 1 each.
- In SAVNav, target location/category prediction is used as the BOS token for the ADPredictor, aligning descriptions with goal awareness.

## Key Experimental Results

### Main Results

**Semantic Audio-Visual Navigation (SAVNav, Heard Setting)**:

| Method | SR↑ | SPL↑ | SNA↑ | DTG↓ | SWS↑ |
|--------|-----|------|------|------|------|
| AV-Nav | 19.3 | 15.9 | 15.0 | 12.6 | 5.6 |
| SAVi | 31.6 | 28.5 | 24.6 | 11.8 | 12.5 |
| KSAVEN | 25.1 | 18.1 | 13.5 | 10.3 | 15.8 |
| **SAVi + P-DescRL** | **37.4** | **32.4** | **28.0** | **8.4** | **19.1** |

**SAVNav Unheard Setting**:

| Method | SR↑ | SPL↑ | SNA↑ | DTG↓ | SWS↑ |
|--------|-----|------|------|------|------|
| SAVi | 24.7 | 22.4 | 18.9 | 11.8 | 10.2 |
| **SAVi + P-DescRL** | **31.4** | **26.9** | **22.5** | **8.7** | **15.1** |

P-DescRL comprehensively surpasses the prior SOTA on SAVi: SR +5.8, SPL +3.9, SWS +6.6 (Heard).

**VLN (Val Unseen)**:

| Method | NE↓ | SR↑ | SPL↑ |
|--------|-----|-----|------|
| DUET | 3.21 | 71.65 | 60.44 |
| DUET + P-DescRL | **3.09** | **72.33** | **61.37** |
| ScaleVLN | 2.40 | 78.63 | 69.15 |
| ScaleVLN + P-DescRL | **2.37** | **78.84** | 68.96 |

### Ablation Study

**Comparison with Other Auxiliary Tasks (SAVi baseline, Heard Setting)**:

| Auxiliary Task | SR↑ | SPL↑ | SNA↑ | SWS↑ |
|----------------|-----|------|------|------|
| No auxiliary task | 31.6 | 28.5 | 24.6 | 12.5 |
| Predict next action | 33.2 | 30.6 | 27.3 | 12.7 |
| Predict progress | 35.0 | 31.6 | 28.3 | 15.7 |
| Predict next frame | 35.4 | 31.6 | 27.0 | 15.8 |
| Predict goal category | 35.4 | 31.9 | 27.9 | 15.2 |
| **P-DescRL** | **37.4** | **32.4** | **28.0** | **19.1** |

P-DescRL outperforms all conventional auxiliary tasks across every metric, with particularly notable gains in SWS (success weighted by stopping at sound).

**VLM as ADGenerator (SAVNav Heard)**:

| ADGenerator | Fine-tuned | SR↑ | SPL↑ | SWS↑ |
|-------------|-----------|-----|------|------|
| None | — | 31.6 | 28.5 | 12.5 |
| CNN+TF (R2R) | ✓ | **37.4** | **32.4** | **19.1** |
| VideoLLaMA2 (zero-shot) | × | 33.7 | 29.8 | 16.0 |
| VideoLLaMA2 (fine-tuned) | ✓ | 28.9 | 25.6 | 11.6 |
| Qwen2.5-VL (zero-shot) | × | 33.4 | 28.6 | 15.2 |

### Key Findings

- **P-AD > F-AD**: Past action descriptions are more effective as an auxiliary task than future action descriptions. F-AD is inherently too difficult and acts as a detrimental auxiliary task — in the Unheard setting it even degrades the baseline performance.
- **VLM Fine-Tuning is Counterproductive**: Fine-tuning VideoLLaMA2 on R2R leads to overfitting and yields lower performance than zero-shot usage.
- **Stronger VLMs Are Not Necessarily Better**: Qwen2.5-VL, a more capable model, does not outperform VideoLLaMA2 in this setting.
- **SAVNav Benefits Most**: Because sound may cease mid-episode, RL lacks reward signals in the latter half of trajectories; the auxiliary task provides a continuous learning signal to compensate.

## Highlights & Insights

1. **Breaking the Interpretability–Performance Trade-off**: Conventional wisdom holds that interpretability degrades performance; by treating description generation as an auxiliary training signal rather than a standalone objective, DescRL achieves simultaneous improvements in both dimensions.
2. **Knowledge Distillation Resolves the Absence of Ground Truth in RL**: Pretrained models are leveraged as pseudo-label generators, elegantly circumventing the core difficulty of lacking human annotations in RL.
3. **Dual Value of Shared Encoders**: Sharing the Transformer encoder/decoder not only reduces parameter count but also yields more semantically meaningful observation encodings through multi-task learning.
4. **Failure Diagnosis Capability**: Description generation enables analysis of navigation failure modes (e.g., "approached the target but did not stop at the correct position"), offering practical debugging value.

## Limitations & Future Work

- ADGenerator training relies on the R2R dataset (VLN-specific data); domain discrepancy exists (VLN success threshold 3 m vs. SAVNav 1 m), leading to a failure pattern of "approaching but not stopping."
- Zero-shot VLM descriptions are viable but notably inferior to the R2R-trained ADGenerator, indicating that general-purpose VLMs still have limited capacity for navigation-specific descriptions.
- Evaluation is conducted exclusively in the Habitat simulator; real-robot validation remains absent.
- Quality assessment of generated descriptions is primarily qualitative, lacking systematic quantitative linguistic metrics.

## Related Work & Insights

- **Distinction from XRL (Explainable RL)**: Conventional XRL focuses on post-hoc explanation, whereas DescRL integrates description generation into policy learning.
- **Distinction from Instruction Prediction in VLN** (Zhu et al., Hejna et al.): Prior work relies solely on IL; ground-truth instructions are unavailable in RL. DescRL addresses this via distillation.
- **Distinction from LLM-Based Navigation** (Yang et al.): LLM-based approaches suffer from slow inference and poor SPL metrics; DescRL is lightweight and operates in real time.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Reframing action descriptions from "interpretable outputs" to "auxiliary training signals" is a genuinely novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three navigation tasks (ObjNav/VLN/SAVNav), multiple baselines, auxiliary task comparisons, and VLM ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Method motivation is clear and experimental design is thorough.
- **Value**: ⭐⭐⭐⭐ — Provides a generalizable paradigm for auxiliary task design in RL, extending beyond navigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation](navq_learning_a_q-model_for_foresighted_vision-and-language_navigation.md)
- [\[CVPR 2026\] RoboAgent: Chaining Basic Capabilities for Embodied Task Planning](../../CVPR2026/reinforcement_learning/roboagent_chaining_basic_capabilities_for_embodied_task_planning.md)
- [\[ICCV 2025\] RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints](robofactory_exploring_embodied_agent_collaboration_with_compositional_constraint.md)
- [\[NeurIPS 2025\] Bandit and Delayed Feedback in Online Structured Prediction](../../NeurIPS2025/reinforcement_learning/bandit_and_delayed_feedback_in_online_structured_prediction.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_with_action_chunking.md)

</div>

<!-- RELATED:END -->
