---
title: >-
  [Paper Note] From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons
description: >-
  [CVPR 2025][Multimodal VLM][Generalist Embodied Agent] GEA adapts a pretrained multimodal LLM (LLaVA-OneVision) to five major domains (manipulation, navigation, gaming, UI control, and planning) via a learned multi-embodiment action tokenizer. It first undergoes SFT using 2.2 million cross-domain expert trajectories, followed by fine-tuning with online PPO reinforcement learning, enabling a single model to outperform or match domain-specific models across multiple benchmarks.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Generalist Embodied Agent"
  - "Multimodal LLM Fine-tuning"
  - "Reinforcement Learning"
  - "Cross-domain Transfer"
  - "Action Tokenizer"
date: 2026-05-08
content_hash: 42eb63eabc113710
---

# From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons

**Conference**: CVPR 2025  
**arXiv**: [2412.08442](https://arxiv.org/abs/2412.08442)  
**Code**: To be open-sourced (Apple)  
**Area**: Multimodal VLM / Agent / Embodied AI  
**Keywords**: Generalist Embodied Agent, Multimodal LLM Fine-tuning, Reinforcement Learning, Cross-domain Transfer, Action Tokenizer

## TL;DR
GEA adapts a pretrained multimodal LLM (LLaVA-OneVision) to five major domains (manipulation, navigation, gaming, UI control, and planning) via a learned multi-embodiment action tokenizer. It first undergoes SFT using 2.2 million cross-domain expert trajectories, followed by fine-tuning with online PPO reinforcement learning, enabling a single model to outperform or match domain-specific models across multiple benchmarks.

## Background & Motivation

**Background**: Current MLLM-based embodied agents are typically trained for a single domain—e.g., OpenVLA for robot manipulation, SeeClick/CogAgent for UI navigation—rendering individual models unable to generalize across domains. Although these domains share significant similarities (such as visual reasoning, long-horizon decision making, and partial observability), no single model can handle all scenarios simultaneously.

**Limitations of Prior Work**: (1) The action spaces across different domains differ significantly (e.g., end-effector 7-DoF vs. discrete navigation commands vs. screen click coordinates), making direct unification extremely challenging; (2) SFT only learns from expert demonstrations, lacking error-recovery capabilities, which leads to error propagation during interactive tasks due to covariate shift; (3) Pioneering models like Gato do not leverage the knowledge of pretrained MLLMs, nor do they incorporate online RL, leading to limited generalization capabilities.

**Key Challenge**: The differences in action spaces and environments across domains are so vast that simple mixing leads to mutual interference, yet training them separately loses the benefits of cross-domain transfer.

**Goal**: How to design a unified action representation and training strategy that transforms a single MLLM into a generalist embodied agent across both digital and physical worlds?

**Key Insight**: Use a Residual VQ-VAE to learn a cross-embodiment continuous action tokenizer, encoding actions from various robots into tokens in the LLM vocabulary, while representing discrete actions directly via natural language. Consequently, action prediction in all domains is framed as next-token prediction. Online RL is also introduced to address the covariate shift problem inherent in SFT.

**Core Idea**: Unifying heterogeneous action spaces with a learned action tokenizer, performing cross-domain SFT fine-tuning, and applying online PPO reinforcement learning to transform MLLMs into generalist embodied agents.

## Method

### Overall Architecture
GEA is built upon LLaVA-OneVision 7B. Its input consists of environment prompts (describing the embodiment type), task instructions, the most recent $c=3$ frames of visual observations, and historical actions, with the output being action token sequences. Training consists of two stages: Stage 1 involves standard autoregressive SFT on 2.2 million cross-domain trajectories to obtain GEA-Base; Stage 2 applies PPO online RL alongside continued SFT in interactive simulators to yield the final GEA.

### Key Designs

1. **Multi-Embodiment Action Tokenizer**:

    - **Function**: Encode continuous and discrete actions from different embodiments into unified LLM token sequences.
    - **Mechanism**: For discrete actions (e.g., navigation command "right", UI action "tap"), default natural language tokenization is used. For continuous actions (e.g., end-effector displacements, joint velocities), a Residual VQ-VAE (RVQ) is trained with two codebooks of 512 tokens each to encode any arbitrary-dimensional continuous action vector into 2 discrete tokens. Action vectors from different embodiments are padded to the maximum dimension and fed into RVQ, being truncated back to the target embodiment's dimension during inference. The least frequently used tokens in the LLM vocabulary are replaced with the RVQ tokens.
    - **Design Motivation**: Prior uniform discretization methods (such as 256 bins in OpenVLA) suffer from limited precision and fail to generalize across embodiments. RVQ leverages hierarchical residual quantization to accurately represent continuous actions with very few tokens, and a single tokenizer adapting to all embodiments eliminates cross-domain conflicts.

2. **Two-Stage Training: SFT + Online PPO**:

    - **Function**: Train basic policies first, then mitigate covariate shift and learn error-recovery behaviors via online interaction.
    - **Mechanism**: Stage 1 conducts full-parameter fine-tuning of the MLLM on 2.2 million cross-domain trajectories using a standard cross-entropy loss (75k updates, batch size = 256). Stage 2 freezes the non-LLM components and fine-tunes the LLM using LoRA. The loss function is $\mathcal{L}_{GEA} = \sum_{i \in \mathcal{E}_{PPO}} \mathcal{L}_{PPO}(\mathcal{M}_i) + \lambda \sum_{i \in \mathcal{E}} \mathcal{L}_{SFT}(\mathcal{D}_i)$, where $\lambda=0.1$. The RL environments encompass three domains: Habitat Pick, LangR, and Procgen.
    - **Design Motivation**: Pure SFT can only mimic experts and tends to fail when encountering OOD states resulting from its own errors (covariate shift). Experiments demonstrate that neither Success SFT nor Offline RL is as effective as online PPO. Retaining the SFT loss also prevents performance degradation in non-RL domains.

3. **Cross-Domain Data Mixing and PopArt Normalization**:

    - **Function**: Enable the model to benefit simultaneously from data across diverse domains, including manipulation, navigation, gaming, and UI control.
    - **Mechanism**: The training data comprises 12 datasets (1.2 million trajectories from OpenX, 45k from Meta-World, 18k from CALVIN, 320k from Procgen, etc.), totaling over 2.2 million trajectories. During the RL stage, PopArt is utilized to normalize reward distributions across different environments, constrained decoding restricts each environment to output only valid action tokens, and normalized entropy coefficients unify PPO hyperparameters.
    - **Design Motivation**: Ablation studies demonstrate that cross-domain training outperforms single-domain training across all domains (+3-6% on average), indicating positive transfer across different embodied tasks. PopArt and constrained decoding are crucial for stable RL training.

### Loss & Training
Stage 1: Standard autoregressive cross-entropy loss on actions, learning rate of 1e-5, AdamW + cosine decay, trained on 8 nodes with 64 H100 GPUs for 2 days. Stage 2: PPO loss (clipped surrogate objective) + SFT loss with a weight of 0.1, learning rate of 3e-4 (LoRA), entropy coefficient of 1e-4, trained on 8 nodes with 64 GPUs for 1 day (100M cumulative steps).

## Key Experimental Results

### Main Results

| Task | Metric | GEA | Prev. SOTA | Notes |
|--------|------|------|----------|------|
| Meta-World (45 tasks) | Success Rate | **94.7%** | 87.0% (Gato) | Outperforms Gato by +7.7% |
| CALVIN (ABC→D) | Success Rate | **90.0%** | 82.4% (MLLM+IL) | Close to domain-specific method of 92.2% |
| Habitat Pick (20 scenes) | Success Rate | **82.5%** | 81.0% (RL+state) | Outperforms RL expert without using state |
| Procgen (16 games) | Expert Score Ratio | **44.0%** | 25% (Domain-specific) | Outperforms domain-specific method by +19% |
| AndroidControl (35 tasks) | Success Rate | **57.3%** | 45% (GPT-4o+SoM) | Outperforms GPT-4o by +12.3% |
| BabyAI (17 tasks) | Success Rate | 91.1% | 93.2% (Gato) | Close to SOTA with less data |

### Ablation Study

| Configuration | Habitat Pick | CALVIN | Procgen | AndroidControl | BabyAI |
|------|---------|---------|---------|---------|---------|
| GEA-Base (all domains) | 57.0 | 48.0 | 24.5 | 50.5 | 84.7 |
| Domain Specific only | 54.5 | 35.5 | 23.7 | 48.9 | 82.1 |
| Only LLM init | 9.5 | 0.0 | 7.6 | 26.4 | 49.4 |
| Only ViEncoder init | 34.5 | 13.0 | 24.5 | 28.3 | 70.6 |
| No pretrained weights | 9.0 | 0.0 | 7.4 | 14.1 | 44.4 |

### Key Findings
- **Cross-domain data yields consistent improvements**: Multi-domain training outperforms single-domain training across all domains, with the most notable gain on CALVIN (+12.5%), suggesting that manipulation data exhibits the strongest mutual transfer capabilities.
- **Online RL far outperforms other methods**: On Habitat Pick, SFT reaches 60.5% while online PPO achieves 82.5%, whereas Success SFT and Offline IQL actually degrade performance. This validates the importance of online interaction in overcoming covariate shift.
- **Visual encoder initialization is more critical than LLM initialization**: Performance with only visual encoder pretraining is substantially higher than with only LLM pretraining, indicating that visual generalization is the primary bottleneck.
- **Model scaling effects are consistent**: Performance across all domains scales continuously from 0.5B to 7B, with minor differences between different MLLM backbones (LLaVA-OV vs. MM1.5), demonstrating that pretraining data quality is more critical than architectural choices.

## Highlights & Insights
- **The RVQ action tokenizer is a more elegant solution than uniform discretization**: It accurately represents continuous actions of arbitrary dimensions using only 2 tokens while naturally supporting cross-embodied setups, addressing a core engineering hurdle for VLA models.
- **Online RL is indispensable for embodied agents**: This paper demonstrates through rigorous experiments the performance ceiling of SFT-only approaches, alongside the ineffectiveness of Success SFT and Offline RL. This finding provides valuable guidance for any work seeking to build agents using MLLMs.
- **Empirical evidence of positive cross-domain transfer**: Improving performance does not require manually designed auxiliary tasks; simply mixing data from different domains suffices. This implies that different embodied tasks share underlying cognitive capabilities (e.g., spatial reasoning, object relation understanding).

## Limitations & Future Work
- Using only 3 frames of historical observations as context limits performance in partially observable tasks (e.g., navigation), necessitating longer context windows or explicit memory mechanisms.
- RL is only applied in 3 domains (Habitat Pick, LangR, and Procgen). Performance in other domains like ManiSkill and AndroidControl is still subpar, and scaling RL to more environments could lead to further improvements.
- The model cannot achieve zero-shot control over completely novel embodiments, as training the action tokenizer still requires data from that specific embodiment type.
- Apple has not yet open-sourced the specific training data or established a final timeline for model weights.

## Related Work & Insights
- **vs. Gato**: Gato does not use pretrained MLLMs, RL, or learned tokenization (utilizing uniform discretization instead). GEA outperforms Gato by 7.7% on Meta-World and 1.7% on Atari, proving the value of MLLM pretraining, learned tokenization, and RL.
- **vs. OpenVLA**: OpenVLA focuses solely on robot manipulation, uses uniform discretization, and lacks RL. GEA's learned tokenizer is more precise, and its cross-domain training provides additional gains.
- **vs. Magma**: Magma leverages SoM/ToM to bridge different domains (without a learned tokenizer) but lacks RL constraint. GEA's RL stage exhibits clear advantages in interactive tasks. The design philosophies of the two are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ While individual components are not entirely novel, the overall system design and scale of cross-domain RL are pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, utilizing 12 datasets, 10 evaluation benchmarks, and extensive ablation analyses (covering data, models, and RL strategies).
- Writing Quality: ⭐⭐⭐⭐ Well-structured, with clearly synthesized lessons extracted from the methodology and experiments.
- Value: ⭐⭐⭐⭐ Provides a clear recipe and crucial empirical findings for building generalist embodied agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices](multi-layer_visual_feature_fusion_in_multimodal_llms_methods_analysis_and_best_p.md)
- [\[CVPR 2025\] HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios](homesafe-bench_evaluating_vision-language_models_on_unsafe_action_detection_for_.md)
- [\[CVPR 2025\] Embodied Scene Understanding for Vision Language Models via MetaVQA](embodied_scene_understanding_for_vision_language_models_via_metavqa.md)
- [\[CVPR 2025\] Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy](playing_the_fool_jailbreaking_llms_and_multimodal_llms_with_out-of-distribution_.md)
- [\[CVPR 2025\] V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents](v-stylist_video_stylization_via_collaboration_and_reflection_of_mllm_agents.md)

</div>

<!-- RELATED:END -->
