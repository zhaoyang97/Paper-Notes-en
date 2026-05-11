---
title: >-
  [Paper Note] MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning
description: >-
  [CVPR 2026][Reinforcement Learning][multimodal reward model] This paper proposes Multi-Stage Reinforcement Learning (MSRL), which first learns reward reasoning capabilities on large-scale text preference data and then pr…
tags:
  - "CVPR 2026"
  - "Reinforcement Learning"
  - "multimodal reward model"
  - "cross-modal transfer"
  - "knowledge distillation"
  - "preference alignment"
date: 2026-05-08
content_hash: 6689958b422cd852
---

# MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning

**Conference**: CVPR 2026
**arXiv**: [2603.25108](https://arxiv.org/abs/2603.25108)
**Code**: [GitHub](https://github.com/wangclnlp/MSRL)
**Area**: Reinforcement Learning / Multimodal Reward Modeling
**Keywords**: multimodal reward model, reinforcement learning, cross-modal transfer, knowledge distillation, preference alignment

## TL;DR

This paper proposes Multi-Stage Reinforcement Learning (MSRL), which first learns reward reasoning capabilities on large-scale text preference data and then progressively transfers them to multimodal tasks, addressing the bottleneck of scarce annotated data in multimodal reward model training. MSRL improves accuracy on VL-RewardBench from 66.6% to 75.9%.

## Background & Motivation

Multimodal reward models (MRMs) are a core component for aligning multimodal large language models (MLLMs) with human preferences. Recent research has shifted from discriminative to generative reward modeling (producing preference predictions via CoT reasoning), and has begun adopting RLVR (Reinforcement Learning from Verifiable Rewards) to further enhance MRM capabilities.

However, RLVR faces a fundamental bottleneck: **high-quality multimodal preference annotation data is extremely scarce**. Annotation costs are prohibitive, making it infeasible to scale RL training as extensively as in the text domain. Existing workarounds such as confidence estimation and self-verification are prone to error accumulation and rapid performance saturation.

The core insight of this paper is that **the fundamental capability of preference reasoning can be learned from abundant text-only data and effectively transferred to multimodal settings**. This challenges the prevailing assumption that multimodal data scarcity must be addressed with more multimodal data.

## Method

### Overall Architecture

MSRL adopts a three-stage curriculum training strategy:
1. **Stage 1**: RL on large-scale text preference data to establish general reward reasoning capabilities.
2. **Stage 2**: RL on caption-based data combined with cross-modal knowledge distillation to transfer preference reasoning.
3. **Stage 3**: RL on a small amount of real multimodal data for final adaptation.

### Key Designs

1. **Large-Scale RL on Text Data (Stage 1)**:
    - SFT on 40k HelpSteer3 data to learn the CoT output format.
    - GRPO optimization on 400k GRAM-R2 text preference data.
    - Visual encoder and projection layer parameters are frozen; only the language component is trained.
    - Design Motivation: text preference data is abundant and cheap to obtain, enabling full exploitation of RL's scaling properties.

2. **Caption-Based RL + Preference Generalization (Stage 2)**:
    - Images/videos in multimodal preference data are replaced with their corresponding textual captions, constructing text-only training data that retains multimodal semantics.
    - A **task recognition reward** $r_{\text{task}}$ is introduced: the model must first output a task type label (e.g., `<type>Image Understanding</type>`), receiving a reward of 0.2 for correct identification.
    - An **experience replay strategy** is adopted to prevent catastrophic forgetting: high-quality text samples from Stage 1 are mixed into training batches at a new-to-old ratio of 5:1.

3. **Cross-Modal Knowledge Distillation (CMKD)**:
    - To bridge the modality gap, given a preference sample and its caption, the caption-trained MRM generates $n$ candidate reasoning chains.
    - A three-step selection procedure identifies the optimal teacher signal $o^*$: (1) majority voting to determine a pseudo-label, (2) format filtering, and (3) selection of the highest-confidence candidate.
    - SFT is performed on $[c, o^*]$ pairs so that the model can reproduce the distilled reasoning process even from visual inputs alone.
    - In subsequent RL stages, the model is required to first generate a `<caption>` before performing reward reasoning.

4. **Multimodal RL Fine-Tuning (Stage 3)**:
    - Final adaptation uses only 20k multimodal samples.
    - Because the preceding stages have already established strong reward reasoning capabilities, only a small amount of data is required at this stage.
    - The task recognition reward is also applied here.

### Loss & Training

- All three stages are based on GRPO optimization, with the core objective: $\mathcal{L}_{\text{RLVR}} = -\mathbb{E}[r_v(s,o)] - \beta \mathbb{D}_{\text{KL}}(\pi_\theta || \pi_{\theta_{\text{old}}})$
- Verifiable reward: $r_v = r_{\text{format}} + r_{\text{accuracy}}$ (plus $r_{\text{task}}$ in Stages 2 and 3).
- Sampling size 8, learning rate 1e-6, batch size 128.

## Key Experimental Results

### Main Results

| Benchmark | Metric | MSRL (8B) | Generative MRM | Gain |
|---|---|---|---|---|
| VL-RewardBench | Avg Acc | 75.9% | 66.6% | +9.3% |
| Multimodal RewardBench | Avg Acc | 80.5% | 76.2% | +4.3% |
| GenAI-Bench (Image Gen.) | Acc | 75.7% | 70.2% | +5.5% |
| ShareGPT (Video Under.) | Acc | 85.5% | 80.6% | +4.9% |
| GenAI-Bench (Video Gen.) | Acc | 81.4% | 68.3% | +13.1% |

MSRL 8B with voting@16 achieves 77.5% on VL-RewardBench, surpassing Claude-3.7-Sonnet (66.5%) and GPT-4o (62.4%).

### Ablation Study

| Configuration | VL-RewardBench Avg | Note |
|---|---|---|
| Generative Baseline | 66.6% | Trained on multimodal data only |
| w/o Stage 1 | 68.8% | Removing text RL → largest drop (−7.1%) |
| w/o Stage 2 (Caption) | 74.3% | Removing caption RL → −1.6% |
| w/o Stage 2 (CMKD) | 73.4% | Removing cross-modal distillation → −2.5% |
| w/o Stage 3 | 72.6% | Removing multimodal RL → −3.3% |
| **Full MSRL** | **75.9%** | Complete method |

### Key Findings

- **Text RL is the most critical stage**: Stage 1 contributes the largest performance gain (+6.9%), demonstrating that reward reasoning capability can be learned from pure text data.
- **Consistent scaling behavior**: Across model sizes from 1B to 14B, MSRL's improvements are consistent and larger models benefit more.
- **High data efficiency**: MSRL trained with only 5k multimodal samples already substantially outperforms the multimodal-data-only baseline, suggesting that capabilities established via text RL diminish the marginal returns of multimodal signals.
- **Largest gains on video tasks**: Video generation tasks see the highest improvement (+13.1%), indicating that temporal visual data benefits most from strong reasoning capabilities.

## Highlights & Insights

1. **An elegant solution to the data bottleneck**: Rather than seeking more multimodal data, MSRL leverages cross-modal transfer — a reductive approach that sidesteps the core constraint.
2. **Captions as a modality bridge**: Replacing images with captions enables a smooth "text → multimodal" transition that is both simple and effective.
3. **Task recognition reward**: Requiring the model to identify the task type before reasoning improves the unified MRM's ability to discriminate across heterogeneous tasks.
4. **Engineering-friendly scalability**: The framework highlights a practical scaling axis — increasing text data volume alone can continuously improve multimodal performance, without expensive multimodal annotation.

## Limitations & Future Work

- Validation is limited to the InternVL3.5 model family; generalizability to other architectures (e.g., Qwen-VL, LLaVA) remains to be verified.
- Captions in CMKD are generated by GPT-5, introducing a dependency on caption quality.
- The experience replay ratio in Stage 2 (5:1) lacks sufficient discussion regarding its optimality.
- The downstream effect of MSRL-trained MRMs in practical MLLM alignment pipelines (e.g., for rejection sampling or PPO) is not explored.

## Related Work & Insights

- **Distinction from UnifiedReward**: The latter applies RLVR directly on multimodal data and is limited by data volume; MSRL circumvents this constraint through its multi-stage strategy.
- Inspired by LLaVA and VILA, which demonstrate that caption-based data can effectively transfer textual knowledge to visual tasks.
- Consistent with findings in text-based LLMs that RL can scale reasoning capabilities, this work extends that insight to the multimodal domain.

## Rating

- Novelty: ⭐⭐⭐⭐ — The multi-stage RL curriculum design is novel, though the individual components (GRPO, caption bridging, knowledge distillation) are not new in themselves.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-scale (1B–14B), multi-task (understanding + generation), multi-benchmark evaluation with complete ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear logic with well-articulated motivation.
- Value: ⭐⭐⭐⭐⭐ — Provides a practical and scalable training pathway for multimodal reward models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning](cccaption_dual-reward_reinforcement_learning_for_complete_and_correct_image_capt.md)
- [\[CVPR 2026\] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)
- [\[CVPR 2026\] Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](lifelong_imitation_learning_multimodal_latent_rep.md)
- [\[CVPR 2026\] Anticipatory Planning for Multimodal AI Agents](anticipatory_planning_for_multimodal_ai_agents.md)
- [\[CVPR 2026\] Cross-modal Identity Mapping: Minimizing Information Loss in Modality Conversion via Reinforcement Learning](cross-modal_identity_mapping_minimizing_information_loss_in_modality_conversion_.md)

</div>

<!-- RELATED:END -->
