---
title: >-
  [Paper Note] Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions
description: >-
  [ACL 2026][Reinforcement Learning][Latent Actions] This paper proposes constructing a compact latent action space for multimodal conversational agents (MCAs) to replace the prohibitively large token action space in RL fine-tuning. A cross-modal projector and a cycle-consistency loss are employed to jointly leverage paired image-text data and text-only data for codebook construction, compressing the action space from 152K (vocabulary size) to 128 (codebook size). The proposed method consistently outperforms token-level RL baselines on two dialogue tasks.
tags:
  - ACL 2026
  - Reinforcement Learning
  - Latent Actions
  - Multimodal Dialogue
  - Vision-Language Models
  - Cross-Modal Projection
date: 2026-05-08
content_hash: b69c53e49ad6da33
---

# Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions

**Conference**: ACL 2026
**arXiv**: [2601.07516](https://arxiv.org/abs/2601.07516)
**Code**: [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/MMLatentAction)
**Area**: Reinforcement Learning / Multimodal Dialogue
**Keywords**: Latent Actions, Reinforcement Learning, Multimodal Dialogue, Vision-Language Models, Cross-Modal Projection

## TL;DR

This paper proposes constructing a compact latent action space for multimodal conversational agents (MCAs) to replace the prohibitively large token action space in RL fine-tuning. A cross-modal projector and a cycle-consistency loss are employed to jointly leverage paired image-text data and text-only data for codebook construction, compressing the action space from 152K (vocabulary size) to 128 (codebook size). The proposed method consistently outperforms token-level RL baselines on two dialogue tasks.

## Background & Motivation

**Background**: Vision-language models (VLMs) such as Qwen-VL and GPT-4o are increasingly deployed as multimodal conversational agents (MCAs) to support emotionally rich and contextually relevant dialogues grounded in both images and text. RL has been widely explored for adapting MCAs to diverse human-computer interaction scenarios.

**Limitations of Prior Work**: Token-level RL faces a severe exploration challenge: with a vocabulary size of $|\mathcal{V}|=152K$ (Qwen2.5-VL) and a maximum response length of $m$ steps, the sampling space grows exponentially as $|\mathcal{V}|^m$, leading to inefficient exploration and insufficient diversity.

**Key Challenge**: Constructing a latent action space requires diverse data with sufficient coverage, yet paired image-text data for VLMs is costly to annotate and limited in scale. Training a codebook on limited paired data results in poor coverage and weak generalization, while incorporating large-scale text-only data risks introducing unimodal bias, causing the model to over-rely on textual cues while neglecting visual information.

**Goal**: To design a coverage-enhanced latent action space construction method for MCAs that leverages both paired image-text data and large-scale text-only data while avoiding unimodal bias.

**Key Insight**: The authors draw on the *learning from observation* paradigm to construct a latent action codebook—inferring the current latent action from future observations, and then reconstructing future observations from the latent action.

**Core Idea**: A cross-modal projector $P$ is trained to map text embeddings into the joint image-text embedding space. Initialized on paired data and further regularized with a cycle-consistency loss on text-only data, the projector safely exploits 627B tokens of text-only data to broaden codebook coverage.

## Method

### Overall Architecture

Three new modules are introduced on top of a base VLM: (1) a **language world model** $f_{\text{world}}$ that receives the current observation and a latent action and autoregressively generates the next token; (2) an **inverse dynamics model** $f_{\text{inverse}}$ that infers the current latent action index from current and future observations; and (3) a **policy model** $\pi_\theta$ that predicts latent actions from the current observation alone. Training proceeds in two stages: latent action space construction (inverse dynamics learning + policy behavioral cloning), followed by latent-action RL on downstream tasks.

### Key Designs

1. **Cross-Modal Projector and Cycle-Consistency Loss**

    - **Function**: Reliably maps text-only embeddings into the joint image-text embedding space to safely exploit large-scale text-only data.
    - **Mechanism**: The forward projector $P$ maps a text embedding $e^T$ to the parameters $(\mu, \sigma) = P(e^T)$ of a diagonal Gaussian distribution. A backward projector $P'$ performs the inverse mapping. Both projectors are first initialized on paired data using Gaussian regression losses $\mathcal{L}_{\text{t2vt}} + \mathcal{L}_{\text{vt2t}}$, then jointly trained on text-only data with a cycle-consistency loss $\mathcal{L}_{\text{cycle}}$ enforcing $P'(P(e^T)) \approx e^T$. This enables the generation of plausible pseudo image-text embeddings even in the absence of real images.
    - **Design Motivation**: Learning the codebook directly in the text embedding space would introduce unimodal bias. The cycle-consistency constraint ensures that the projector maintains consistency on unpaired data, preventing the generation of embeddings that deviate from the true image-text space.

2. **Codebook Construction via Inverse Dynamics**

    - **Function**: Constructs a learnable latent action codebook $\mathcal{C} \in \mathbb{R}^{|\mathcal{C}| \times d}$ in an unsupervised manner.
    - **Mechanism**: The inverse dynamics model $f_{\text{inverse}}$ observes the current and future states and outputs a discrete action index $a_t \in \{1, \ldots, |\mathcal{C}|\}$, from which the corresponding embedding $c_{a_t}$ is retrieved from the codebook. The world model $f_{\text{world}}$ then reconstructs the next token using this embedding and the current observation. All three components are trained jointly with the loss $\mathcal{L}_{\text{inverse}} = -\sum_t \log f_{\text{world}}(x^T_{t+1} | e^{V,T}_t, a_t)$.
    - **Design Motivation**: The bidirectional constraint of inverse dynamics and reconstruction encourages the codebook to naturally encode high-level semantic information that governs generation. With $|\mathcal{C}|=128$, the action space is dramatically smaller than the 152K vocabulary, substantially compressing the exploration space.

3. **Latent-Action Reinforcement Learning**

    - **Function**: Optimizes the policy in the compact latent space to improve RL exploration diversity.
    - **Mechanism**: During the RL phase, the world model is frozen and only the policy model $\pi_\theta$ is optimized. At each step, the policy samples a latent action $a_t \sim \pi_\theta(\cdot | x^V, x^T_{1:t})$; the world model generates the token $x^T_{t+1} = f_{\text{world}}(x^V, x^T_{1:t}, a_t)$; and the objective is to maximize the expected reward $\mathcal{J}(\theta) = \mathbb{E}[R(x^T_{p+1:m})]$. The framework is compatible with multiple RL algorithms, including GRPO, Dr.GRPO, DAPO, and BNPO.
    - **Design Motivation**: Optimizing over latent action distributions rather than token distributions accelerates policy updates (0.86× of baseline time) and significantly improves rollout diversity (semantic diversity increases from ~1.07 to ~1.25).

### Loss & Training

Training proceeds in three stages with the following losses: (1) projector initialization: $\mathcal{L}_{\text{proj}_1} = \mathcal{L}_{\text{t2vt}} + \mathcal{L}_{\text{vt2t}}$; (2) joint training of inverse dynamics and projector: $\mathcal{L}_{\text{inverse}} + \mathcal{L}_{\text{proj}_2}$; (3) policy behavioral cloning: $\mathcal{L}_{\text{bc}}$. Data scale: 14M images + 1B text tokens (paired) + 627B text tokens (text-only).

## Key Experimental Results

### Main Results

Qwen2.5-VL-3B-Instruct; LLM-as-a-Judge score ratios:

| Method | MMRole-ID | MMRole-OOD | PCogAlign-LS1 | PCogAlign-LS2 | Avg. |
|--------|-----------|------------|---------------|---------------|------|
| SFT | 0.843 | 0.809 | 0.808 | 0.810 | 0.817 |
| GRPO (Token) | 0.838 | 0.796 | 0.845 | 0.845 | 0.831 |
| **GRPO (Latent)** | **0.949** | **0.915** | **0.871** | 0.837 | **0.893** |
| Dr.GRPO (Token) | 0.867 | 0.823 | 0.835 | 0.834 | 0.840 |
| **Dr.GRPO (Latent)** | **0.953** | **0.916** | **0.874** | **0.840** | **0.896** |

Rollout semantic diversity comparison:

| Method | MMRole | PCogAlignBench |
|--------|--------|----------------|
| GRPO (Token) | 1.079 | 1.042 |
| GRPO (Latent) | **1.248** | **1.191** |
| DAPO (Token) | 1.073 | 1.038 |
| DAPO (Latent) | **1.253** | **1.127** |

### Ablation Study

Based on GRPO + Qwen2.5-VL-3B-Instruct:

| Setting | MMRole-ID | MMRole-OOD | PCogAlign-LS1 | Avg. |
|---------|-----------|------------|---------------|------|
| Full method | 0.949 | 0.915 | 0.871 | 0.893 |
| w/o cycle-consistency | 0.921 | 0.878 | 0.858 | 0.870 |
| w/o cross-modal projector | 0.944 | 0.901 | 0.858 | 0.880 |
| w/o text-only data | 0.932 | 0.861 | 0.851 | 0.865 |

### Key Findings

- Latent-action RL outperforms token-level RL by an average of 4% and is consistently effective across all four RL algorithms.
- Semantic diversity improves substantially: GRPO increases from 1.079 to 1.248 on MMRole, confirming improved exploration efficiency.
- Text-only data is the most critical component—its removal causes the largest OOD performance drop (0.915→0.861), underscoring the importance of coverage for generalization.
- Total training overhead increases by only 1.08×, while policy update time is actually reduced to 0.86×, keeping computational costs manageable.

## Highlights & Insights

- This work is the first to introduce latent actions into RL fine-tuning of multimodal conversational agents; the compression ratio from 152K to 128 is remarkably significant.
- The cycle-consistency loss elegantly bridges limited paired data with large-scale text-only data by exploiting cross-modal redundancy assumptions.
- Algorithm-agnostic compatibility (GRPO / Dr.GRPO / DAPO / BNPO) indicates that latent actions constitute a general underlying paradigm.

## Limitations & Future Work

- The latent actions lack interpretability—it remains unclear what semantic concepts each of the 128 codewords encodes.
- Evaluation is limited to dialogue tasks; broader tasks such as visual mathematical reasoning and larger VLMs are left for future work.
- Inference latency increases by 1.13×, which may warrant optimization for real-time dialogue scenarios.

## Related Work & Insights

- CoLA (Jia et al., 2025) first introduced latent actions for text-only LLMs; this work extends the paradigm to multimodal settings and addresses the problem of scarce paired data.
- The *learning from observation* framework from robotics provides the theoretical basis for constructing the latent space.
- Key insight: the core bottleneck of RL fine-tuning for VLMs lies not in the algorithm itself but in the representation of the action space—abstracting from the token level to the latent level may be a critical path toward more efficient RL alignment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing latent actions into multimodal dialogue RL is a novel combinatorial contribution; the cycle-consistency loss design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two tasks × two model scales × four RL algorithms, with complete ablation and diversity analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are clear and the pipeline figures are informative, though the notation system is somewhat complex.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Anticipatory Planning for Multimodal AI Agents](../../CVPR2026/reinforcement_learning/anticipatory_planning_for_multimodal_ai_agents.md)
- [\[ACL 2026\] SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving](spiralthinker_latent_reasoning_through_an_iterative_process_with_text-latent_int.md)
- [\[CVPR 2026\] Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](../../CVPR2026/reinforcement_learning/lifelong_imitation_learning_multimodal_latent_rep.md)
- [\[ICLR 2026\] A Unifying View of Coverage in Linear Off-Policy Evaluation](../../ICLR2026/reinforcement_learning/a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)
- [\[ICLR 2026\] Retaining Suboptimal Actions to Follow Shifting Optima in Multi-Agent RL](../../ICLR2026/reinforcement_learning/retaining_suboptimal_actions_to_follow_shifting_optima_in_multi-agent_reinforcem.md)

<!-- RELATED:END -->
