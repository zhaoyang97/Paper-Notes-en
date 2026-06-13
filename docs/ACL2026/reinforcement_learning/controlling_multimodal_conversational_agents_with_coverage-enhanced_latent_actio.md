---
title: >-
  [Paper Note] Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions
description: >-
  [ACL 2026][Reinforcement Learning][Latent actions] This paper proposes constructing a compact latent action space for Multimodal Conversational Agents (MCAs) to replace the massive token action space during RL fine-tunin…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Latent actions"
  - "Multimodal dialogue"
  - "Vision-Language Models"
  - "Cross-modal projection"
date: 2026-05-08
content_hash: 17f61d1d77bd2d75
---

# Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions

**Conference**: ACL 2026  
**arXiv**: [2601.07516](https://arxiv.org/abs/2601.07516)  
**Code**: [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/MMLatentAction)  
**Area**: Reinforcement Learning / Multimodal Dialogue  
**Keywords**: Latent actions, Reinforcement Learning, Multimodal dialogue, Vision-Language Models, Cross-modal projection

## TL;DR

This paper proposes constructing a compact latent action space for Multimodal Conversational Agents (MCAs) to replace the massive token action space during RL fine-tuning. By utilizing cross-modal projectors and cycle-consistency loss, paired image-text data and text-only data are jointly leveraged to construct a codebook. This approach compresses the action space from 152K (vocabulary size) to 128 (codebook size) and consistently outperforms token-level RL baselines across two dialogue tasks.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) such as Qwen-VL and GPT-4o are increasingly employed as Multimodal Conversational Agents (MCAs), supporting emotionally rich and context-aware dialogues based on images and text. RL has been widely explored to adapt MCAs to diverse human-computer interaction scenarios.

**Limitations of Prior Work**: Token-level RL faces massive exploration space challenges—with a vocabulary size $|\mathcal{V}|=152K$ (Qwen2.5-VL) and a maximum response length of $m$ steps, the sampling space grows exponentially as $|\mathcal{V}|^m$. This leads to inefficient RL exploration and insufficient diversity.

**Key Challenge**: Constructing a latent action space requires diverse data with sufficient coverage, but the paired image-text data required by VLMs is expensive to annotate and limited in scale. Training a codebook with only limited paired data leads to insufficient coverage and poor generalization; introducing large-scale unpaired text data may introduce unimodal bias (where the model over-relies on text and ignores visual information).

**Goal**: Design a coverage-enhanced latent action space construction method for MCAs that utilizes both paired image-text data and large-scale text-only data while avoiding unimodal bias.

**Key Insight**: The authors draw inspiration from the "learning from observation" mechanism to construct the latent action codebook—inferring current latent actions from future observations and then using those latent actions to reconstruct future observations.

**Core Idea**: A cross-modal projector $P$ is trained to map text embeddings into the image-text embedding space. It is initialized with paired data and enhanced with text-only data plus a cycle-consistency loss for robustness, enabling the safe utilization of 627B tokens of text-only data to expand codebook coverage.

## Method

### Overall Architecture

Three new modules are introduced on top of the base VLM: (1) a language world model $f_{\text{world}}$ that receives current observations and latent actions to autoregressively generate the next token; (2) an inverse dynamics model $f_{\text{inverse}}$ that infers the current latent action index based on future observations; (3) a policy model $\pi_\theta$ that predicts latent actions based only on current observations. Training is conducted in two stages: first, constructing the latent action space (inverse dynamics learning + policy behavior cloning), followed by latent action RL on downstream tasks.

### Key Designs

1.  **Cross-modal Projector and Cycle-Consistency Loss**:
    - **Function**: Reliably maps text-only embeddings to the image-text embedding space to safely leverage large-scale text-only data.
    - **Mechanism**: The forward projector $P$ maps text embedding $e^T$ to parameters $(\mu, \sigma) = P(e^T)$ of a diagonal Gaussian distribution. The reverse projector $P'$ performs the inverse mapping. Both are initialized on paired data using Gaussian regression losses $\mathcal{L}_{\text{t2vt}} + \mathcal{L}_{\text{vt2t}}$, then jointly trained on text-only data using a cycle-consistency loss $\mathcal{L}_{\text{cycle}}$, constraining $P'(P(e^T)) \approx e^T$. This allows the generation of plausible pseudo-image-text embeddings even in the absence of real images.
    - **Design Motivation**: Directly learning a codebook on text embeddings introduces unimodal bias; the cycle-consistency constraint ensures the projector maintains consistency even on unpaired data, preventing generated embeddings from deviating from the real image-text space.

2.  **Inverse Dynamics-based Codebook Construction**:
    - **Function**: Constructs a learnable latent action codebook $\mathcal{C} \in \mathbb{R}^{|\mathcal{C}| \times d}$ in an unsupervised manner.
    - **Mechanism**: The inverse dynamics model $f_{\text{inverse}}$ observes current and future context to output a discrete action index $a_t \in \{1, \ldots, |\mathcal{C}|\}$, retrieving the corresponding embedding $c_{a_t}$ from the codebook. The world model $f_{\text{world}}$ then uses this embedding and current observations to reconstruct the next token. The components are trained jointly with the loss $\mathcal{L}_{\text{inverse}} = -\sum_t \log f_{\text{world}}(x^T_{t+1} | e^{V,T}_t, a_t)$.
    - **Design Motivation**: Through the bidirectional constraint of "inverse dynamics + reconstruction," the codebook naturally encodes high-level semantic information capable of controlling generation. The codebook size $|\mathcal{C}|=128$ is significantly smaller than the 152K vocabulary, drastically compressing the exploration space.

3.  **Latent Action Reinforcement Learning**:
    - **Function**: Optimizes the policy within a compact latent space to enhance RL exploration diversity.
    - **Mechanism**: During the RL phase, the world model is frozen, and only the latent action prediction distribution of the policy model $\pi_\theta$ is optimized. At each step, the policy samples a latent action $a_t \sim \pi_\theta(\cdot | x^V, x^T_{1:t})$, and the world model generates the token $x^T_{t+1} = f_{\text{world}}(x^V, x^T_{1:t}, a_t)$ to maximize the expected reward $\mathcal{J}(\theta) = \mathbb{E}[R(x^T_{p+1:m})]$. It supports various RL algorithms including GRPO, Dr.GRPO, DAPO, and BNPO.
    - **Design Motivation**: By optimizing the latent action distribution rather than the token distribution, policy updates are faster (0.86× baseline time), and rollout diversity is significantly improved (semantic diversity increases from approx. 1.07 to 1.25).

### Loss & Training

Losses across three stages: (1) Projector initialization $\mathcal{L}_{\text{proj}_1} = \mathcal{L}_{\text{t2vt}} + \mathcal{L}_{\text{vt2t}}$; (2) Joint training of inverse dynamics and projector $\mathcal{L}_{\text{inverse}} + \mathcal{L}_{\text{proj}_2}$; (3) Policy behavior cloning $\mathcal{L}_{\text{bc}}$. Data scale: 14M images + 1B text tokens (paired) + 627B text tokens (text-only).

## Key Experimental Results

### Main Results

Qwen2.5-VL-3B-Instruct, LLM-as-a-Judge score ratios:

| Method | MMRole-ID | MMRole-OOD | PCogAlign-LS1 | PCogAlign-LS2 | Average |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SFT | 0.843 | 0.809 | 0.808 | 0.810 | 0.817 |
| GRPO (Token) | 0.838 | 0.796 | 0.845 | 0.845 | 0.831 |
| **GRPO (Latent)** | **0.949** | **0.915** | **0.871** | 0.837 | **0.893** |
| Dr.GRPO (Token) | 0.867 | 0.823 | 0.835 | 0.834 | 0.840 |
| **Dr.GRPO (Latent)** | **0.953** | **0.916** | **0.874** | **0.840** | **0.896** |

Rollout semantic diversity comparison:

| Method | MMRole | PCogAlignBench |
| :--- | :--- | :--- |
| GRPO (Token) | 1.079 | 1.042 |
| GRPO (Latent) | **1.248** | **1.191** |
| DAPO (Token) | 1.073 | 1.038 |
| DAPO (Latent) | **1.253** | **1.127** |

### Ablation Study

Based on GRPO + Qwen2.5-VL-3B-Instruct:

| Setting | MMRole-ID | MMRole-OOD | PCogAlign-LS1 | Average |
| :--- | :--- | :--- | :--- | :--- |
| Full Method | 0.949 | 0.915 | 0.871 | 0.893 |
| w/o Cycle-Consistency | 0.921 | 0.878 | 0.858 | 0.870 |
| w/o Cross-modal Projector| 0.944 | 0.901 | 0.858 | 0.880 |
| w/o Text-only Data | 0.932 | 0.861 | 0.851 | 0.865 |

### Key Findings

- Latent action RL improves by an average of 4% over token-level RL and is consistently effective across all four RL algorithms.
- Semantic diversity significantly improves: GRPO increased from 1.079 to 1.248 (MMRole), confirming enhanced exploration efficiency.
- Text-only data is the most critical component—removing it resulted in the largest drop in OOD performance (0.915→0.861), indicating that coverage is vital for generalization.
- Total training time increased by only 1.08×, while policy updates were actually faster (0.86×), keeping overall overhead manageable.

## Highlights & Insights

- This work is the first to introduce latent actions into RL fine-tuning for multimodal conversational agents, with a highly significant compression ratio (152K→128).
- The cycle-consistency loss cleverly leverages cross-modal redundancy assumptions to bridge limited paired data with massive text-only data.
- RL algorithm agnosticism (compatibility with GRPO/Dr.GRPO/DAPO/BNPO) suggests that latent actions are a fundamental and general paradigm.

## Limitations & Future Work

- Latent actions lack interpretability—it remains unclear what semantic concepts the 128 codewords represent.
- The method was validated only on dialogue tasks; broader tasks like visual mathematical reasoning and application to larger VLMs are left for future work.
- Inference latency increased by 1.13×, which might require optimization for real-time dialogue scenarios.

## Related Work & Insights

- CoLA (Jia et al., 2025) first introduced latent actions to text-only LLMs; this paper extends them to multimodal scenarios and addresses the scarcity of paired data.
- The "learning from observation" concept from robotics provides a theoretical foundation for constructing the latent space.
- Insight: The core bottleneck of RL fine-tuning for VLMs lies not in the algorithms themselves, but in the representation of the action space—abstracting from the token level to the latent level may be the key path toward more efficient RL alignment.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing latent actions to multimodal dialogue RL is a novel combination innovation; the cycle-consistency loss is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covering two tasks, two model scales, and four RL algorithms, with complete ablation and diversity analyses.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and pipeline diagrams are highly informative, though the notation system is somewhat complex.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Anticipatory Planning for Multimodal AI Agents](../../CVPR2026/reinforcement_learning/anticipatory_planning_for_multimodal_ai_agents.md)
- [\[ACL 2026\] SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving](spiralthinker_latent_reasoning_through_an_iterative_process_with_text-latent_int.md)
- [\[ACL 2026\] DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents](dpepo_diverse_parallel_exploration_policy_optimization_for_llm-based_agents.md)
- [\[ACL 2026\] Breaking the Impasse: Dual-Scale Evolutionary Policy Training for Social Language Agents](breaking_the_impasse_dual-scale_evolutionary_policy_training_for_social_language.md)
- [\[ICLR 2026\] A Unifying View of Coverage in Linear Off-Policy Evaluation](../../ICLR2026/reinforcement_learning/a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)

</div>

<!-- RELATED:END -->
