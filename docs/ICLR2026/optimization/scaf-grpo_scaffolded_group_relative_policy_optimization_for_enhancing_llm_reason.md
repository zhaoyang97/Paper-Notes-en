---
title: >-
  [Paper Note] Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning
description: >-
  [ICLR 2026][Optimization][GRPO] This paper proposes the Scaf-GRPO framework, which injects hierarchical in-prompt hints (Knowledge → Planning → Solution) to overcome the "learning cliff" (zero-reward) problem in GRPO tra…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "GRPO"
  - "reinforcement learning"
  - "learning cliff"
  - "progressive guidance"
  - "scaffolding pedagogy"
date: 2026-05-08
content_hash: abdce92afada35e6
---

# Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning

**Conference**: ICLR 2026
**arXiv**: [2510.19807](https://arxiv.org/abs/2510.19807)  
**Code**: None  
**Area**: Optimization / LLM Reasoning Enhancement
**Keywords**: GRPO, reinforcement learning, learning cliff, progressive guidance, scaffolding pedagogy

## TL;DR

This paper proposes the Scaf-GRPO framework, which injects hierarchical in-prompt hints (Knowledge → Planning → Solution) to overcome the "learning cliff" (zero-reward) problem in GRPO training. On Qwen2.5-Math-7B, it achieves a 44.3% relative improvement in pass@1 on AIME24 while preserving on-policy training consistency.

## Background & Motivation

**Background**: Reinforcement learning from verifiable rewards (RLVR) has become the dominant paradigm for enhancing LLM reasoning. Algorithms such as GRPO update policies using advantage signals derived from group-relative rewards.

**Limitations of Prior Work**: When a model encounters problems far beyond its current capability, all exploratory attempts fail, yielding persistent zero-reward signals. In GRPO, all-zero rewards within a group cause the advantage $\hat{A}_i = \frac{R(o_i) - \mu_\mathcal{G}}{\sigma_\mathcal{G}} = 0$, leading to vanishing gradients and forming a "learning cliff."

**Key Challenge**: Existing solutions such as LUFFY adopt a prefix-continuation strategy—supplying the model with a correct solution prefix—which introduces a distribution mismatch between the teacher and student policies and forces the model along a predetermined path, suppressing exploration.

**Goal**: To help the model overcome the learning cliff and acquire reasoning capabilities from otherwise unsolvable problems, without introducing off-policy distribution mismatch.

**Key Insight**: Inspired by the pedagogical theory of scaffolding, the approach provides minimal, progressive in-prompt hints rather than imposing a forced solution-path prefix.

**Core Idea**: Rather than providing "rails" (prefixes), the method provides "signposts" (hints)—injecting hierarchical prompts so that the model generates correct solutions using its own policy, thereby avoiding off-policy issues while retaining exploratory freedom.

## Method

### Overall Architecture

Training proceeds in two phases. **Phase 1** (guidance exemption period, first 15% of steps) allows the model to explore autonomously and distinguish "pseudo-hard" from "truly hard" problems. **Phase 2** activates hierarchical hint-guided exploration for truly hard problems. When all rollouts in a batch yield zero reward, Scaf-GRPO injects hints in the order Knowledge → Planning → Solution until the model produces a correct solution. The successful trajectory replaces one failed trajectory, advantages are recomputed, and the policy is updated with the standard GRPO loss.

### Key Designs

1. **Guidance Exemption Period and Truly Hard Problem Diagnosis**:

    - **Function**: No hints are provided during the initial training phase (first 15% of steps), allowing fully autonomous exploration.
    - **Mechanism**: The resolution rate of zero-reward problems is monitored; once the rate stagnates, remaining unsolved problems are labeled as "truly hard." The rapid early decline corresponds to "pseudo-hard" problems (unfamiliar formatting, elementary reasoning skills).
    - **Design Motivation**: Prevents premature hint dependency and ensures hints are reserved for genuine capability gaps. Ablation experiments show that removing the exemption period causes a 9.2% performance drop.

2. **Hierarchical Hint-Guided Exploration (K→P→S)**:

    - **Function**: Three levels of progressive in-prompt hints, from abstract to concrete, are injected for truly hard problems.
    - **Mechanism**: $H_{\text{knowledge}}$ (key concepts/formulas) → $H_{\text{planning}}$ (high-level strategy framework) → $H_{\text{solution}}$ (concrete computational steps). Hints are provided incrementally within each level; the process halts as soon as the model succeeds, and the minimum effective hint level is recorded.
    - **Design Motivation**: Minimal intervention preserves model autonomy—rewarding successful problem-solving with the most abstract hint encourages internalization of reasoning skills rather than memorization. Removing any single level degrades performance (removing the Solution level causes a 5.7% drop).

3. **On-Policy Batch Augmentation and Unified Loss**:

    - **Function**: A successful hint-guided trajectory replaces one failed trajectory, restoring the advantage signal.
    - **Mechanism**: $\mathcal{G}_{\text{final}} = (\mathcal{G} \setminus \{o_j\}) \cup \{o_h^*\}$, where $o_h^* \sim \pi_\theta(\cdot | q \oplus h^*)$. The probability ratio $r_{i,t}'(\theta) = \frac{\pi_\theta(o_{i,t}'|o_{i,<t}', q \oplus h^*)}{\pi_{\theta_{\text{old}}}(o_{i,t}'|o_{i,<t}', q \oplus h^*)}$ is a standard on-policy ratio.
    - **Design Motivation**: Unlike prefix-based methods that use the off-policy ratio $\frac{\pi_\theta(\cdot|q)}{\pi_{\theta_{\text{old}}}(\cdot|q \oplus h^*)}$, this approach conditions both policies on the same hint-augmented prompt, guaranteeing on-policy consistency.

### Loss & Training

The loss function is identical to standard GRPO (clipped surrogate objective), with differences confined to the data level: $J_{\text{Scaf-GRPO}}(\theta) = \hat{\mathbb{E}}_{i,t}[\min(r_{i,t}'(\theta)\hat{A}_i', \text{clip}(r_{i,t}'(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i')]$. The KL divergence penalty is set to 0 to maximize exploration. Training runs for 10 epochs with a maximum response length of 2048 tokens.

## Key Experimental Results

### Main Results

| Model / Benchmark | Metric | Scaf-GRPO | Vanilla GRPO | LUFFY | Gain |
|---|---|---|---|---|---|
| Qwen2.5-Math-7B / AIME24 | pass@1 | 43.3 | 30.0 | 33.3 | +44.3% vs GRPO |
| Qwen2.5-Math-7B / AIME25 | pass@1 | 20.0 | 13.3 | 16.7 | +50.4% vs GRPO |
| Qwen2.5-Math-7B / AMC | pass@1 | 70.0 | 60.0 | 62.5 | +16.7% vs GRPO |
| Qwen2.5-Math-7B / Avg. 7 benchmarks | pass@1 | 50.9 | 45.2 | 46.6 | +12.6% vs GRPO |
| Qwen2.5-Math-1.5B / Average | pass@1 | 41.5 | 37.6 | — | +10.4% |
| DeepSeek-R1-Distill-1.5B / Average | pass@1 | 53.6 | 50.6 | — | +5.9% |

### Ablation Study

| Configuration | Avg. 7 Benchmarks | Note |
|---|---|---|
| Full K→P→S | 50.9 | Complete three-level hierarchy |
| w/o Progressive (Solution-Only) | 48.4 | Directly provide most concrete hint |
| w/o Knowledge Hint | 49.2 | Remove concept level |
| w/o Solution Hint | 48.0 | Remove concrete step level; largest drop |
| w/o Incremental Chunking | 47.7 | Full hint provided at once |
| No Guidance (Vanilla GRPO) | 45.2 | Unguidanced baseline |

### Key Findings

- Progressive guidance outperforms directly providing Solution hints by 2.5 points—abstract hints compel the model to reason autonomously, cultivating more generalizable skills.
- Removing any single hint level degrades performance, indicating the three levels are complementary rather than redundant.
- Incremental delivery (revealing hint content step by step) outperforms one-shot delivery by 3.2 points, validating the principle of minimal intervention.
- Models exhibit an observable evolution from "imitating hints" to "solving problems autonomously."

## Highlights & Insights

- The "signposts vs. rails" analogy is apt: in-prompt hints allow the model to freely choose its reasoning path, whereas prefix-continuation forces a predetermined route.
- The guidance exemption period is an elegant design choice—allowing the model to struggle independently before intervening, akin to effective pedagogical practice.
- The GRPO loss function is preserved in its entirety; intervention occurs only at the data level, resulting in an engineering solution that is both clean and elegant.

## Limitations & Future Work

- The three-level hints must be pre-generated by a strong external model (DeepSeek-R1), increasing data preparation costs.
- Validation is currently limited to mathematical reasoning tasks; transferability to domains such as code generation or logical reasoning remains unexplored.
- Hint quality has a significant impact (a 4% gap between DeepSeek-R1 and Qwen-72B), making dependence on hint generation a potential bottleneck.
- While the guidance exemption period percentage (15%) is stable in the 10%–40% range, the optimal value may vary across models.

## Related Work & Insights

- **vs. LUFFY**: LUFFY's prefix-continuation approach introduces distribution mismatch requiring policy shaping corrections; Scaf-GRPO uses in-prompt hints to maintain on-policy training, outperforming LUFFY by an average of 4.3 points on 7B models.
- **vs. Vanilla GRPO**: GRPO learning stalls due to zero gradients under zero reward; Scaf-GRPO restores the signal through batch augmentation.
- **vs. DAPO/DeepScaleR**: These methods modify the GRPO algorithm itself, whereas Scaf-GRPO modifies the data and guidance strategy; the two approaches are orthogonal and can be combined.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The application of scaffolding pedagogy to reinforcement learning is novel; in-prompt hints as a distinct alternative to prefix-continuation represent a key contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple models (Qwen/Llama/DeepSeek) and scales (1.5B–7B) with carefully designed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated; the training dynamics visualization in Figure 2 is particularly intuitive.
- **Value**: ⭐⭐⭐⭐ Provides a practical and theoretically grounded solution to the learning cliff problem in RLVR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Provable and Practical In-Context Policy Optimization for Self-Improvement](provable_and_practical_in-context_policy_optimization_for_self-improvement.md)
- [\[ICLR 2026\] ∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space](nabla-reasoner_llm_reasoning_via_test-time_gradient_descent_in_latent_space.md)
- [\[ICML 2026\] Enhancing LLM Training via Spectral Clipping](../../ICML2026/optimization/enhancing_llm_training_via_spectral_clipping.md)
- [\[NeurIPS 2025\] MeCeFO: Enhancing LLM Training Robustness via Fault-Tolerant Optimization](../../NeurIPS2025/optimization/mecefo_enhancing_llm_training_robustness_via_fault-tolerant_optimization.md)
- [\[ICLR 2026\] CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving](cogflow_bridging_perception_and_reasoning_through_knowledge_internalization_for_.md)

</div>

<!-- RELATED:END -->
