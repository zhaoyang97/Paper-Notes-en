---
title: >-
  [Paper Note] Translate Policy to Language: Flow Matching Generated Rewards for LLM Explanations
description: >-
  [ICLR 2026][Image Generation][Policy Explanation] This paper proposes a general framework that leverages Rectified Flow to generate distributional rewards for training explanation-generating LLMs. By employing continuous…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Policy Explanation"
  - "Rectified Flow"
  - "Distributional Rewards"
  - "RLAIF"
  - "LLM Interpretability"
date: 2026-05-08
content_hash: fd4a053c59dde9fa
---

# Translate Policy to Language: Flow Matching Generated Rewards for LLM Explanations

**Conference**: ICLR 2026
**arXiv**: [2502.12530](https://arxiv.org/abs/2502.12530)  
**Code**: None  
**Area**: Diffusion Models / LLM Alignment
**Keywords**: Policy Explanation, Rectified Flow, Distributional Rewards, RLAIF, LLM Interpretability

## TL;DR
This paper proposes a general framework that leverages Rectified Flow to generate distributional rewards for training explanation-generating LLMs. By employing continuous normalizing flows (CNF) to capture the pluralistic and probabilistic nature of human judgments on explanations, the framework provides theoretical guarantees that CNF can effectively recover the true human reward distribution. It significantly outperforms RLHF/RLAIF baselines on SMAC, MMLU, MathQA, and other tasks.

## Background & Motivation

**Background**: As RL agents, LLMs, and other intelligent systems become deeply integrated into everyday life, explaining agent policies in natural language has become increasingly important. RLHF/RLAIF has emerged as the dominant paradigm for aligning LLM behavior, but faces unique challenges in explanation tasks.

**Limitations of Prior Work**: (1) Human judgments of explanations are inherently pluralistic and probabilistic, making the collection of diverse human feedback costly; (2) Existing RLAIF methods use rewards from surrogate LLMs that introduce noisy bias, and the problem of generating distributional rewards that account for surrogate error has not been rigorously studied; (3) Existing distributional reward modeling methods (QRM, DPRM, URM) require discretization or assumptions about specific distributional forms.

**Key Challenge**: An unavoidable bias exists between surrogate LLM rewards and the true human reward distribution, $W_2(\hat{p}, p) = \sqrt{|\mathcal{A}|}|\sigma_r|$, and directly training with surrogate rewards leads to suboptimal explanations.

**Goal**: How can distributional rewards that accurately reflect the plurality of human judgments be generated to train explanation-generating LLMs, without requiring large amounts of human feedback?

**Key Insight**: Rectified Flow is embedded within the LLM architecture as a reward model, exploiting the denoising properties of CNF to recover a distribution close to the true human reward distribution from noisy surrogate LLM rewards.

**Core Idea**: Rectified Flow treats the noise in surrogate LLM rewards as Gaussian noise injected during the forward process, and learns the reverse process to recover the true human reward distribution, while providing theoretical error bounds.

## Method

### Overall Architecture
The system comprises three key components: (1) Explanation LLM $\pi_e(\theta_e)$: given a decision context (with the true decision hidden), generates natural language explanations; (2) Proxy LLMs: $K=3$ independent LLMs providing reward samples; (3) Rectified Flow reward model $\varphi(\theta_\varphi)$: learns distributional rewards from proxy LLM reward samples. The training procedure alternates between updating the flow model and training the Explanation LLM with PPO.

### Key Designs

1. **Rectified Flow Reward Model Architecture**:

    - Function: Embeds Rectified Flow within an LLM, enabling it to generate reward distributions conditioned on linguistic context.
    - Mechanism: Designs flow tokens (projected from $\mathbf{z}_t$ and $PE(t)$ via MLP) that interact with the LLM hidden states of the decision context and explanation through cross-attention. The last-layer weight matrices $(W_Q, W_K, W_V)$ of the Explanation LLM are used to compute cross-attention.
    - Training Loss: $\mathcal{L}_{\text{Flow}}(\theta_\varphi) = \mathbb{E}[\|(\mathbf{z}_1 - \mathbf{z}_0) - \varphi(t, \mathbf{z}_t | c_j, y_j^e; \theta_\varphi)\|^2]$
    - Design Motivation: Standard fully connected networks or U-Nets cannot interpret linguistic cues; embedding the flow model within an LLM exploits its language understanding capabilities.

2. **Theoretical Error Bound (Theorem 1)**:

    - Core Result: $W_2(p_{\text{flow}}, p) \leq \varepsilon + L\sqrt{|\mathcal{A}|}|\sigma - \sigma_r|$
    - Interpretation: When the initial distribution of the flow and the surrogate LLM noise share the same functional form (e.g., both Gaussian), CNF transforms the unavoidable bias term $\sqrt{|\mathcal{A}|}|\sigma_r|$ into a controllable term $L\sqrt{|\mathcal{A}|}|\sigma - \sigma_r|$.
    - When $\sigma \approx \sigma_r$, the error can be made arbitrarily small.

3. **Sentence-Level Dense Rewards**:

    - Function: Provides a reward signal for each sentence of the explanation, rather than only a sparse reward at the end.
    - Mechanism: Explanation content is added sentence by sentence, and the change in the true decision logit is used as the reward for each sentence.
    - Design Motivation: Dense rewards accelerate PPO convergence and guide explanation quality at a finer granularity.

### Loss & Training
- The flow model applies rejection sampling: only samples for which the proxy LLM assigns the highest probability to the true decision are retained.
- Softmax activation is applied to logits to mitigate the influence of large values.
- The Explanation LLM is fine-tuned using PPO + LoRA.
- The flow model consists of a frozen LLM backbone plus two trainable MLPs ($\varphi_{\text{Emb}}$ and $\varphi_{\text{Proj}}$).
- $K=3$ independent Proxy LLMs are used: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and Gemma-2-2B-It.

## Key Experimental Results

### Main Results

| Method | SMAC ACC | MMLU ACC | MathQA ACC | AI2-THOR SR |
|--------|----------|----------|------------|-------------|
| **Ours** | **0.764** | **0.772** | **0.804** | **0.702** |
| Proxy LLM | 0.640 | 0.703 | 0.694 | — |
| KTO | 0.721 | 0.753 | 0.758 | 0.628 |
| ReFT | 0.722 | 0.743 | 0.763 | 0.642 |
| Skywork | 0.692 | 0.737 | 0.729 | 0.483 |
| o3-mini | 0.455 | 0.707 | 0.739 | 0.677 |

### Ablation Study

| Configuration | SMAC | MMLU | MathQA |
|---------------|------|------|--------|
| Full Model | 0.764 | 0.772 | 0.804 |
| w/o Attn (cross-attention removed) | 0.731 | 0.749 | 0.775 |
| Sparse Reward | 0.738 | 0.755 | 0.781 |
| w/o Flow (direct proxy LLM reward) | 0.640 | 0.703 | 0.694 |

### Human Evaluation (MathQA)

| Method | ACC | Logic | Actionable | Cognitive |
|--------|-----|-------|------------|-----------|
| **Ours** | **0.892** | **0.60** | **0.46** | **0.60** |
| DPO | 0.591 | 0.17 | 0.28 | 0.18 |
| ReFT | 0.635 | 0.23 | 0.26 | 0.22 |

### Key Findings
- Removing Rectified Flow and using proxy LLM rewards directly results in a performance drop of 6.9%–12.4%, validating the effectiveness of flow-based denoising.
- The cross-attention mechanism contributes a 3–5% performance improvement, highlighting the importance of language-conditioned reward generation.
- Sentence-level dense rewards outperform sparse rewards by 2–3%.
- In human evaluation, 89.2% of generated explanations enabled human subjects to correctly infer the agent's decision, surpassing DPO by 25.7%.

## Highlights & Insights
- **Tight Integration of Theory and Practice**: Theorem 1 provides a rigorous error bound for CNF's ability to manage surrogate noise, which is thoroughly validated experimentally. This represents a significant theoretical contribution to the foundations of RLAIF.
- **Strong Generality**: The framework spans both RL tasks (SMAC, AI2-THOR) and LLM tasks (MMLU, MathQA) without task-specific engineering.
- **Architectural Innovation**: The approach of embedding Rectified Flow within an LLM is noteworthy; the flow token + cross-attention design is an elegant bridge between generative models and language models.
- **Explanation Rather Than Decision-Making**: The focus on "explaining policies" rather than "making decisions" fills an important gap in explainable AI research.

## Limitations & Future Work
- Relying on three proxy LLMs for reward samples incurs non-trivial computational cost.
- The Gaussian noise assumption may not hold in all settings, although the paper discusses alternative functional forms in the appendix.
- Validation is limited to multiple-choice and discrete-action settings; performance on open-ended generation tasks remains unknown.
- Applying this framework to preference modeling in RLHF is a promising direction for future exploration.

## Related Work & Insights
- **vs. QRM/DPRM/URM**: These methods require discretization or restrict the distributional form; the proposed method directly models continuous distributions via CNF.
- **vs. Skywork RLAIF**: Skywork ranks highly on RewardBench but performs poorly on explanation tasks (0.692 vs. 0.764 on SMAC), underscoring the unique challenges of the explanation setting.
- **vs. o3-mini**: Even a strong reasoning model underperforms the proposed method on explanation tasks, emphasizing the value of task-specific training.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Applying Rectified Flow to distributional reward modeling is a genuinely novel perspective, supported by rigorous theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers both RL and LLM domains, four benchmarks, extensive ablations, and human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, though the overall structure is somewhat complex.
- Value: ⭐⭐⭐⭐ Makes important contributions to both the theoretical foundations of RLAIF and the practice of explainable AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Principled RL for Flow Matching Emerges from the Chunk-level Policy Optimization](../../ICML2026/image_generation/principled_rl_for_flow_matching_emerges_from_the_chunk-level_policy_optimization.md)
- [\[ICLR 2026\] SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)
- [\[ICLR 2026\] FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching](flowcast_advancing_precipitation_nowcasting_with_conditional_flow_matching.md)
- [\[ICLR 2026\] Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [\[ICLR 2026\] Purrception: Variational Flow Matching for Vector-Quantized Image Generation](purrception_variational_flow_matching_for_vector-quantized_image_generation.md)

</div>

<!-- RELATED:END -->
