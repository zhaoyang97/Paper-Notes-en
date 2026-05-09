---
title: >-
  [Paper Note] Hybrid Latent Reasoning via Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Latent reasoning] HRPO proposes a hybrid latent reasoning policy optimization framework: a learnable gating mechanism progressively blends the hidden state representation from the previous step into the sampled token embeddings, enabling LLMs to leverage both discrete tokens and continuous latent representations during inference. Without requiring CoT annotations, HRPO is trained entirely via RL and outperforms baselines such as PPO and GRPO on both knowledge-intensive and STEM reasoning tasks.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Latent reasoning
  - hybrid reasoning
  - gating mechanism
  - continuous representation
date: 2026-05-08
content_hash: 09362a1c242b171d
---

# Hybrid Latent Reasoning via Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.18454](https://arxiv.org/abs/2505.18454)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Latent reasoning, hybrid reasoning, reinforcement learning, gating mechanism, continuous representation

## TL;DR
HRPO proposes a hybrid latent reasoning policy optimization framework: a learnable gating mechanism progressively blends the hidden state representation from the previous step into the sampled token embeddings, enabling LLMs to leverage both discrete tokens and continuous latent representations during inference. Without requiring CoT annotations, HRPO is trained entirely via RL and outperforms baselines such as PPO and GRPO on both knowledge-intensive and STEM reasoning tasks.

## Background & Motivation

**Background**: Latent reasoning has attracted growing attention as an alternative to Chain-of-Thought (CoT). Methods such as Coconut feed the last-layer hidden state back as a "continuous thought" for the next step, achieving promising results on reasoning tasks. However, these approaches generally rely on CoT trajectories for training.

**Limitations of Prior Work**: (1) Existing latent reasoning methods (e.g., Coconut, CODI) require large amounts of CoT-annotated data and multi-stage training, incurring high cost while failing to exploit the LLM's intrinsic reasoning capacity. (2) Directly using hidden states as inputs for the next step disrupts generation quality (repetition, incoherence), since hidden states and token embeddings reside in different representation spaces.

**Key Challenge**: Latent reasoning demands continuous representations for richer information, yet autoregressive generation in LLMs is inherently discrete. Directly bridging the two induces distribution mismatch and degrades the model's generative capability.

**Goal**: How can existing LLMs leverage both discrete tokens and continuous hidden states for reasoning, without requiring CoT annotations?

**Key Insight**: Design a gating mechanism that progressively mixes hidden state information into token embeddings—starting with nearly pure token embeddings (preserving generation quality)—and let RL automatically learn when and how much latent representation to incorporate during training.

**Core Idea**: Progressively fuse discrete tokens and continuous hidden states via a gating mechanism, and train the LLM to autonomously learn a hybrid reasoning strategy through RL rather than CoT distillation.

## Method

### Overall Architecture
During inference, HRPO introduces hybrid inputs within the `<think>`…`</think>` reasoning span: at each timestep, the input is no longer the embedding of the sampled token alone, but a gated mixture of the token embedding and the hidden state from the previous step. The answer segment still uses standard autoregressive decoding. Training follows a REINFORCE-style online RL procedure with simple outcome rewards (correct = 1, incorrect = 0).

### Key Designs

1. **Hidden State Projection**:

    - Function: Projects the model's output hidden state $\hat{h}_t$ back into the embedding space.
    - Mechanism: Computes softmax output probabilities $p_{t+1} = \text{softmax}(\text{Head}(\hat{h}_t) / \tau)$ and forms a weighted sum over all token embeddings: $h_{t+1} = W_e^T \frac{p_{t+1}}{\|p_{t+1}\|}$. The temperature $\tau$ controls the sharpness of the distribution.
    - Design Motivation: Directly using hidden states causes distribution mismatch and generation degradation. Probability-weighted interpolation aligns the projected representation with the model's native embedding space while preserving differentiability.

2. **Gating Mechanism**:

    - Function: Controls the mixing ratio between the discrete token embedding $\hat{e}_{t+1}$ and the continuous hidden representation $h_{t+1}$.
    - Mechanism: Defines a reset gate $r_t = \sigma(W_a \hat{e}_{t+1} + b_a)$, an input gate $i_t = \sigma(W_x \hat{e}_{t+1} + b_x)$, and a decay coefficient $a_t = \exp(-c \cdot \text{softplus}(\Lambda) \odot r_t)$. The final input is $e_{t+1} = a_t \odot \hat{e}_{t+1} + \sqrt{1 - a_t^2} \odot (i_t \odot h_{t+1})$, where $\Lambda$ is a learnable parameter.
    - Design Motivation: Initializing $a_t \to 1$ ensures that training begins with nearly pure token embeddings (preserving LLM generation capability). As training proceeds, the gate learns to incorporate progressively more hidden state information. This gradual design prevents the generation collapse that arises from directly substituting hidden states.

3. **Hybrid Reasoning Policy Optimization (HRPO)**:

    - Function: Online policy optimization via REINFORCE-style RL.
    - Mechanism: For each question, $g$ hybrid rollouts (discrete tokens + hidden representations) are sampled. Outcome rewards (correct/incorrect) are used to compute group-normalized advantages $\hat{A}_i = \frac{r_i - \text{mean}([r_1,...,r_g])}{\text{std}([r_1,...,r_g])}$. The policy gradient is $\nabla_\theta \mathcal{J} = \mathbb{E}[\frac{1}{g}\sum_i \frac{1}{|y_i|}\sum_t \nabla_\theta \log \pi_\theta(y_{i,t}|...) \hat{A}_{i,t}] - \beta \nabla_\theta D_{KL}[\pi_\theta \| \pi_{ref}]$.
    - Design Motivation: The sampling operation preserves stochasticity, enabling RL rollouts. A strictly on-policy regime (each trajectory used only once) is required because the hidden representations directly depend on the current parameters $\theta$. The lightweight design requires no additional value network.

### Loss & Training
REINFORCE with KL regularization is used, with $\beta = 0.005$. PPO-style clipping ratios are omitted; raw log-probabilities are used directly, as conservative learning rates render ratio clipping rarely active. The method is runnable on a single GPU. LoRA (rank=32, $\alpha$=64) is applied for parameter-efficient fine-tuning.

## Key Experimental Results

### Main Results
Base models: Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct.

**STEM Benchmarks (Accuracy)**

| Method | GSM8k | MATH | MATH500 | MMLU-ST | ARC-C | Avg. |
|--------|-------|------|---------|---------|-------|------|
| SFT (1.5B) | 0.560 | 0.300 | 0.302 | 0.403 | 0.602 | 0.433 |
| PPO (1.5B) | 0.694 | 0.507 | 0.518 | 0.566 | 0.715 | 0.600 |
| GRPO (1.5B) | 0.711 | 0.502 | 0.524 | 0.562 | 0.737 | 0.607 |
| **HRPO (1.5B)** | **0.720** | **0.518** | **0.536** | **0.569** | **0.742** | **0.617** |
| PPO (3B) | 0.819 | 0.597 | 0.604 | 0.582 | 0.811 | 0.682 |
| GRPO (3B) | 0.834 | 0.602 | 0.604 | 0.601 | 0.814 | 0.691 |
| **HRPO (3B)** | **0.845** | **0.613** | **0.630** | 0.590 | **0.820** | **0.700** |

### Ablation Study

| Configuration | MATH (1.5B) | Note |
|---------------|-------------|------|
| Direct hidden states | ~0 (collapse) | Mismatch between hidden states and embedding space |
| Interpolation (no gating) | Normal then collapse | Excess noise causes training instability |
| **HRPO (gating)** | **0.518** | Progressive fusion, stable training |
| Coconut | 0.315 (GSM8k) | Relies on CoT compression; underperforms |
| CODI | 0.658 (GSM8k) | CoT self-distillation; still weaker than HRPO |
| **HRPO** | **0.720 (GSM8k)** | No CoT required; RL-driven |

### Key Findings
- HRPO achieves an average accuracy of 0.700 on the 3B model, matching or surpassing 7B-scale models (e.g., Qwen2.5-7B at 0.635 average).
- Directly using hidden states causes complete reward collapse (≈0), while pure interpolation eventually collapses as well—only gated mixing yields stable training.
- HRPO-trained models exhibit an interesting cross-lingual reasoning pattern (e.g., English–Chinese mixed reasoning), suggesting that latent representations can transcend language boundaries.
- The hidden ratio grows steadily during training, indicating that the model actively learns to exploit more latent information.
- Smaller $r_{min}$ (larger initial hidden ratio) benefits knowledge-intensive tasks, while STEM tasks perform best at both extremes.
- HRPO-trained models generate shorter completions, as hidden states effectively encode contextual information.

## Highlights & Insights
- **The progressive gating design is particularly elegant**: starting from nearly pure token embeddings and gradually incorporating hidden states—this "preserve capability first, then progressively enhance" strategy avoids catastrophic degradation in early training and represents a generalizable design principle.
- **Replacing CoT distillation with RL for latent reasoning training marks an important paradigm shift**: it demonstrates that LLMs can autonomously develop latent reasoning capabilities without CoT annotations, substantially reducing training cost.
- **Emergent cross-lingual reasoning behavior**: HRPO-trained models spontaneously switch languages during inference, indicating that latent representations capture reasoning patterns that transcend specific languages—a phenomenon of considerable theoretical interest.

## Limitations & Future Work
- Additional computational overhead: hybrid reasoning introduces gating computation and embedding projection, increasing forward-pass cost.
- The strictly on-policy regime limits large-scale training efficiency, as each trajectory can only be used once and cannot be reused.
- Validation is currently limited to 1.5B and 3B models; behavior at larger scales remains unknown.
- Generated sequences, though shorter, may exhibit format violations or repetitive loops.
- Promising directions include off-policy extensions, larger model validation, format rewards, and hybrid training with CoT.

## Related Work & Insights
- **vs. Coconut**: Coconut compresses tokens into continuous thoughts via multi-stage CoT training; HRPO requires no CoT and is trained entirely via RL. The gap on GSM8k is substantial: HRPO (0.720) vs. Coconut (0.315).
- **vs. CODI**: CODI aligns explicit and implicit reasoning tokens via self-distillation, still requiring CoT data. HRPO consistently outperforms CODI on both datasets.
- **vs. GRPO/PPO**: HRPO consistently surpasses pure RL baselines on all benchmarks, demonstrating that hybrid continuous representations provide genuine additional information gain.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The first method to train hybrid latent reasoning via RL; the gating design is elegant and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 10 benchmarks, multiple model sizes, detailed ablations, and significance tests; limited to small models.
- Writing Quality: ⭐⭐⭐⭐ — Generally clear, though some notation definitions could be made more concise.
- Value: ⭐⭐⭐⭐ — Opens a new direction for RL-based latent reasoning; practical value requires validation on larger models.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation](decoderhybriddecoder_architecture_for_efficient_reasoning_wi.md)
- [\[ACL 2026\] SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving](../../ACL2026/reinforcement_learning/spiralthinker_latent_reasoning_through_an_iterative_process_with_text-latent_int.md)
- [\[NeurIPS 2025\] Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](reasoning_gym_reasoning_environments_for_reinforcement_learning_with_verifiable_.md)
- [\[NeurIPS 2025\] Interactive and Hybrid Imitation Learning: Provably Beating Behavior Cloning](interactive_and_hybrid_imitation_learning_provably_beating_behavior_cloning.md)
- [\[NeurIPS 2025\] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)

<!-- RELATED:END -->
