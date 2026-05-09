---
title: >-
  [Paper Note] MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning
description: >-
  [CVPR 2026][Reinforcement Learning][Multimodal reward model] This paper proposes MSRL (Multi-Stage Reinforcement Learning), which scales generative multimodal reward modeling through a multi-stage RL curriculum: first learning general reward reasoning on large-scale text preference data (400K) via RL, then transferring to the multimodal domain via caption-based RL and cross-modal knowledge distillation, and finally fine-tuning with a small amount of multimodal preference data. Without additional multimodal annotations, MSRL improves performance from 66.6% to 75.9% on VL-RewardBench and from 70.2% to 75.7% on GenAI-Bench.
tags:
  - CVPR 2026
  - Reinforcement Learning
  - Multimodal reward model
  - generative reward
  - multi-stage RL
  - cross-modal knowledge distillation
  - preference alignment
date: 2026-05-08
content_hash: 689ebc36da35c729
---

# MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning

**Conference**: CVPR 2026  
**arXiv**: [2603.25108](https://arxiv.org/abs/2603.25108)  
**Code**: [https://github.com/wangclnlp/MSRL](https://github.com/wangclnlp/MSRL)  
**Area**: Reinforcement Learning / Multimodal Reward Modeling  
**Keywords**: Multimodal reward model, generative reward, multi-stage RL, cross-modal knowledge distillation, preference alignment

## TL;DR
This paper proposes MSRL (Multi-Stage Reinforcement Learning), which scales generative multimodal reward modeling through a multi-stage RL curriculum: first learning general reward reasoning on large-scale text preference data (400K) via RL, then transferring to the multimodal domain via caption-based RL and cross-modal knowledge distillation, and finally fine-tuning with a small amount of multimodal preference data. Without additional multimodal annotations, MSRL improves performance from 66.6% to 75.9% on VL-RewardBench and from 70.2% to 75.7% on GenAI-Bench.

## Background & Motivation

**Background**: Multimodal reward models (MRMs) are critical for MLLM preference alignment. Recent work has shifted from discriminative MRMs toward generative MRMs (CoT reasoning with textual reward scores), further enhanced by RLVR to strengthen reasoning.

**Limitations of Prior Work**: RLVR relies on manually annotated multimodal preference data for verifiable rewards, which is costly to obtain and difficult to scale. In contrast, text preference data is abundant (~1M samples), raising the question of whether text data can be leveraged to scale multimodal reward modeling.

**Core Assumption**: The core capability of reward reasoning can be learned from text data and transferred across modalities — the "reasoning process" underlying preference judgment is largely modality-agnostic.

**Core Idea**: A multi-stage curriculum — Stage 1: text RL (learning general reward reasoning) → Stage 2: caption-based RL + CMKD (transferring to multimodal) → Stage 3: multimodal RL (fine-tuning for adaptation).

## Method

### Overall Architecture
SFT (learning CoT format and reasoning structure) → **Stage 1: Text RL** (400K text preference data, GRPO optimization, frozen visual encoder) → **Stage 2: Caption-based RL + CMKD** (RL on multimodal preference data with images replaced by captions + cross-modal knowledge distillation to align text/visual representations) → **Stage 3: Multimodal RL** (fine-tuning adaptation with a small amount of genuine multimodal preference data).

### Key Designs

1. **Stage 1: Text RL**:

    - **Function**: Learn general reward reasoning on large-scale text preference data.
    - The visual encoder and projection layer are frozen (no visual input at this stage).
    - Verifiable reward: format reward ($r_{format}$) + accuracy reward ($r_{accuracy}$) → $r_v = r_{format} + r_{accuracy}$
    - **Design Motivation**: Text preference data is abundant (~1M), enabling thorough training of reasoning capabilities and providing a strong foundation for subsequent multimodal stages.

2. **Stage 2: Caption-based RL + CMKD**:

    - **Caption-based RL**: Images/videos in multimodal preference data are replaced by captions, maintaining text-only training while preserving multimodal task structure, thereby bridging the gap between text and multimodal domains.
    - A task identification reward $r_{task}$ is introduced: the model must first output the task type (Image Understanding/Generation, etc.), encouraging task-aware reasoning.
    - Experience replay: high-quality text samples from Stage 1 are mixed in to prevent forgetting.
    - **CMKD (Cross-Modal Knowledge Distillation)**: Given a preference sample $s$ and its caption $c$, $n$ reasoning chains are sampled from the caption-trained model $\pi_{\theta_{text}}$; the best reasoning chain $o^*$ is selected via voting, format filtering, and confidence scoring; SFT is then performed on multimodal inputs to teach the model to reproduce text-derived reasoning from visual inputs, bridging the modality gap.

3. **Stage 3: Multimodal RL**:

    - Fine-tuning on a small amount of multimodal preference data. Prior stages already endow strong reasoning capabilities, substantially reducing the required multimodal data.
    - Supports both visual understanding (image/video QA preference judgment) and visual generation (image/video generation quality assessment).

### Loss & Training
GRPO optimization: $\mathcal{L}_{RLVR} = -\mathbb{E}[r_v(s,o)] - \beta\mathbb{D}_{KL}(\pi_\theta || \pi_{\theta_{old}})$

## Key Experimental Results

### Main Results

| Method | VL-RewardBench Avg | Multimodal RewardBench Avg | GenAI-Bench |
|------|:---:|:---:|:---:|
| Baseline (SFT only) | 66.6 | 76.2 | 70.2 |
| + Single-stage multimodal RL | ~70 | ~78 | ~72 |
| **MSRL (multi-stage)** | **75.9** | **80.5** | **75.7** |

VL-RewardBench: +9.3, Multimodal RewardBench: +4.3, GenAI-Bench: +5.5.

### Cross-Scale Validation

| Model Scale | Gain from Text RL |
|----------|:---:|
| 1B | Effective |
| 7B | Better |
| 14B | **Largest gain** |

→ Larger models benefit more from text RL, validating the robustness of the scaling behavior.

### Ablation Study

| Configuration | VL-RewardBench |
|------|:---:|
| w/o Stage 1 (Text RL) | Significant drop |
| w/o Stage 2 (Caption RL + CMKD) | Moderate drop |
| w/o CMKD (Caption RL only) | Slight drop |
| **Full MSRL** | **75.9** |

### Key Findings
- **Text RL is the core engine**: Stage 1 contributes the most — the importance of general reward reasoning capability exceeds that of multimodal data adaptation.
- **CMKD effectively bridges the modality gap**: Caption-trained reasoning can be reproduced under visual inputs, demonstrating modality-agnostic reasoning transfer.
- **Task identification reward is beneficial**: Having the model identify the task type before reasoning improves performance, as different tasks require different evaluation logic.
- **Consistent across scales**: All model sizes from 1B to 14B benefit, with larger models gaining more, indicating strong scaling properties.

## Highlights & Insights
- **The core hypothesis of "scaling multimodal rewards via text preference data" is validated**: The core capability of reward reasoning is modality-agnostic and can be learned from abundant text data and transferred. This challenges the intuition that multimodal capabilities must be learned from multimodal data.
- **The multi-stage curriculum is elegantly designed**: The gradual transition from text → caption (bridging) → multimodal avoids the large gap of direct cross-modal transfer.
- **CMKD's "voting + filtering + selection" strategy**: Extracting the best reasoning chain from sampled outputs is more robust than direct distillation.
- **Unified visual understanding + generation reward model**: A single model evaluates both VQA answer quality and generated image quality, demonstrating high generality.

## Limitations & Future Work
- The quality and coverage of text preference data constrain the foundational capabilities acquired in Stage 1.
- Captions as proxies for visual inputs still incur information loss — richer textual descriptions may further close this gap.
- The voting mechanism in CMKD requires $n$ sampling passes, increasing Stage 2 training cost.
- The optimal proportion of multimodal data used in Stage 3 still requires manual tuning.

## Related Work & Insights
- **vs. Single-stage RLVR**: Performing RL directly on limited multimodal data yields insufficient reasoning capability. MSRL first trains thoroughly on text data before transferring.
- **vs. Discriminative MRM**: Discriminative models output scalar scores, lacking interpretability. Generative models output CoT + answers, enabling auditable reasoning.
- **vs. GRAM-R2**: A pioneer in text RL; MSRL further extends this paradigm to multimodal settings and introduces CMKD.
- **Insight**: The progressive text-to-multimodal transfer strategy is applicable to any scenario requiring scalable multimodal RL training.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The multi-stage RL approach to scaling multimodal reward modeling via text data is novel and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on VL-RewardBench / MM-RewardBench / GenAI-Bench with cross-scale validation and thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ The multi-stage design is clearly presented with intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ Addresses the data bottleneck in multimodal reward modeling.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning](cccaption_dual-reward_reinforcement_learning_for_complete_and_correct_image_capt.md)
- [\[CVPR 2026\] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)
- [\[CVPR 2026\] Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](lifelong_imitation_learning_multimodal_latent_rep.md)
- [\[CVPR 2026\] Anticipatory Planning for Multimodal AI Agents](anticipatory_planning_for_multimodal_ai_agents.md)
- [\[ICLR 2026\] RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning](../../ICLR2026/reinforcement_learning/rewardmap_tackling_sparse_rewards_in_fine-grained_visual_reasoning_via_multi-sta.md)

<!-- RELATED:END -->
