---
title: >-
  [Paper Note] Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models
description: >-
  [AAAI 2026][Recommender Systems][Inference-Aware Optimization] This paper reveals a non-trivial interaction between prompt selection and inference strategies (Best-of-N, Majority Voting), proposes the IAPO framework that jointly optimizes prompt design and inference scaling as a contextual best-arm identification problem, and introduces PSST—a fixed-budget training algorithm—achieving up to 50% improvement over inference-agnostic methods across 6 tasks.
tags:
  - AAAI 2026
  - Recommender Systems
  - Inference-Aware Optimization
  - Prompt Optimization
  - Best-of-N Sampling
  - Contextual Bandits
  - Black-Box Alignment
date: 2026-05-08
content_hash: c67983507e0bcac1
---

# Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models

**Conference**: AAAI 2026  
**arXiv**: [2508.10030](https://arxiv.org/abs/2508.10030)  
**Code**: [https://iapo-aaai25.github.io/](https://iapo-aaai25.github.io/)  
**Area**: Recommender Systems  
**Keywords**: Inference-Aware Optimization, Prompt Optimization, Best-of-N Sampling, Contextual Bandits, Black-Box Alignment  

## TL;DR
This paper reveals a non-trivial interaction between prompt selection and inference strategies (Best-of-N, Majority Voting), proposes the IAPO framework that jointly optimizes prompt design and inference scaling as a contextual best-arm identification problem, and introduces PSST—a fixed-budget training algorithm—achieving up to 50% improvement over inference-agnostic methods across 6 tasks.

## Background & Motivation

**Background**: Alignment of black-box LLMs primarily relies on two families of methods—prompt optimization (steering outputs via rewriting or appending instructions) and inference scaling strategies (Best-of-N sampling, Majority Voting to generate multiple candidates and select the best). Both have achieved notable success independently.

**Limitations of Prior Work**: Existing prompt optimization methods entirely ignore the inference strategy used at deployment time—they optimize prompts under single-generation ($N=1$) and then directly apply BoN or MV at deployment. This decoupling leads to suboptimal or even incorrect prompt selection.

**Key Challenge**: The optimal prompt varies with the inference strategy and budget. The authors find that on MATH, Prompt A achieves 65% accuracy at $N=1$, outperforming Prompt B at 62%; however, under MV at $N=10$, Prompt B rises to ~77% while Prompt A drops to ~63%. This occurs because MV amplifies the nonlinear effects of the per-query correctness distribution.

**Goal**: How can one jointly optimize prompts and inference scale under a limited computational budget, while accounting for user preference trade-offs across multiple objectives?

**Key Insight**: The problem is formulated as contextual best-arm identification, where each "arm" is a (prompt, inference scale $N$) combination and the "context" encodes user preferences and budget constraints.

**Core Idea**: Inference-aware prompt optimization—the training phase simulates the nonlinear aggregation effects of inference strategies (BoN/MV), thereby selecting the prompt–scale combination that is optimal for the actual deployment configuration.

## Method

### Overall Architecture
The IAPO (Inference-Aware Prompt Optimization) framework defines arms $a = (p, N) \in \mathcal{A} = \mathcal{P} \times [N_{\max}]$, with context $c = (w_1, \dots, w_{K+1})$ encoding multi-objective weights and budget preferences. A policy $\pi: \mathcal{C} \to \mathcal{A}$ selects the optimal arm upon observing the context. The optimization objective is to maximize the Average Contextual Return $\text{ACR}(\pi) = \mathbb{E}_{c}[Q^\alpha(c, \pi(c))]$, where $Q^\alpha$ denotes the expected return under inference strategy $\alpha \in \{\text{BoN}, \text{MV}\}$.

### Key Designs
1. **Inference-Strategy-Aware Utility Functions**

    - **Function**: Explicitly models the aggregation logic of BoN and MV into the utility functions.
    - **Mechanism**: The BoN utility is $R_x^{\text{BoN}} = \max_{i \leq N} \sum_k w_k O_k + w_{K+1} \sum_i O_{K+1}$ (maximizing weighted task reward minus inference cost); the MV utility is based on vote counts and correct-answer probability. Both are non-affine transformations of the inference-agnostic utility.
    - **Design Motivation**: Proposition 2 proves that an inference-agnostic policy is optimal only under affine transformations; since BoN/MV do not satisfy this condition, inference-aware optimization is necessary.

2. **PSST Algorithm (Prompt Scaling via Sequential Trimming)**

    - **Function**: An arm elimination algorithm under a fixed budget for learning the optimal IAPO policy.
    - **Mechanism**: The algorithm proceeds for $R = \lceil \log_2 |\mathcal{A}| \rceil$ rounds, allocating budget uniformly per round. Three structural properties are exploited to accelerate optimization: (1) *cross-context information sharing*—a single arm pull can be used to estimate $Q$ values for all contexts; (2) *cross-scale nested reuse*—pulling $(p, N_i)$ automatically yields $\lfloor N_i/N_j \rfloor$ samples for $(p, N_j)$; (3) *asymmetric cost awareness*—budget is allocated only to the largest surviving scale for each prompt. Each round ranks arms by estimated $Q$ value and eliminates the worst half per context.
    - **Design Motivation**: Naively applying Sequential Halving ignores IAPO's structure, incurring $O(|\mathcal{C}| N_{\max})$ times higher sample complexity. PSST's structure-aware allocation causes error probability to decay at an exponential rate.

3. **Top-K Screening Heuristic**

    - **Function**: Prior to PSST, uses a small fraction of the budget ($\rho T$) to rapidly screen prompts at $N=1$, retaining the top $K$ prompts for full PSST.
    - **Mechanism**: Allocates $T_0 = \lfloor \rho T \rfloor$ (with $\rho=0.2$) to uniformly evaluate all prompts under single-generation, retains the top $K$ per context, and runs PSST on the reduced arm space with the remaining budget $T' = T - T_0$.
    - **Design Motivation**: Full-space search is inefficient under low budgets; however, aggressive pruning (small $K$) risks discarding prompts that excel only at large $N$ (constructible counterexamples exist), making this heuristic suitable for budget-constrained scenarios but not for high-stakes, high-frequency deployments.

### Loss & Training
No explicit loss function is used—PSST is a hyperparameter-free, sampling-based exploration algorithm. During training, the LLM API is queried in batches to collect $(x, a, \mathbf{o}_{1:N})$ data, and $Q^\alpha(c, a)$ is estimated via Monte Carlo. Stockpiling (accumulating data across outer rounds) further reduces the outer $R$ factor. All training uses Llama-3.3-70B-Instruct as the black-box LLM, with generation via vLLM on 8×A100 GPUs (~2,000 GPU hours). Once the environment is constructed, all experiments can be run quickly on CPU.

## Key Experimental Results

### Main Results

| Environment | Strategy $\alpha$ | $|\mathcal{P}|$ | $N_{\max}$ | $|\mathcal{C}|$ | PSST vs. Uniform | PSST vs. UCB |
|------|---------|------|------|------|------|------|
| Synth-Bernoulli | MV | 32 | 32 | 3 | Significant | Significant ($p<0.05$) |
| MATH | MV | 25 | 32 | 3 | Significant | Significant |
| CommonsenseQA | MV | 48 | 32 | 3 | Significant | Significant |
| Synth-Categorical | BoN | 32 | 32 | 27 | Significant | Significant |
| Helpful-Harmless | BoN | 20 | 32 | 27 | Significant | Significant |
| Summarization | BoN | 20 | 32 | 27 | Significant | Significant |

Across all 6 environments, PSST and Top-K screening significantly outperform all baselines under the Wilcoxon test ($p < 0.05$), identifying strong policies with as few as 5K inference calls.

### Ablation Study

| Configuration | Relative ACR | Notes |
|------|---------|------|
| IAPO + PSST (Full) | Best | Joint optimization of prompt + inference scale |
| TRIPLE (N=1) | −50% | Prompt-only optimization, no inference scaling |
| TRIPLE (N=Random) | −30% | Prompt optimization with random $N$ assignment |
| PSST+K1 (approx. decoupled) | −25% | Select prompt first, then tune $N$; especially poor on Summarization |
| PSST+K4 | Near-optimal | Moderate pruning; performs well on most tasks |
| PSST+K8 | Near-optimal | Light pruning |

### Key Findings
- **Inference-aware vs. inference-agnostic**: IAPO achieves up to 50% improvement over inference-agnostic methods (prompt-only) and 25% over decoupled optimization, validating the necessity of joint optimization.
- **Failure mode of PSST+K1**: It gets trapped by "deceptive prompts"—prompts that perform well at $N=1$ but do not scale under large $N$.
- Top-K screening converges faster under low budgets, while full PSST is superior under high budgets.
- **Theoretical guarantee**: Theorem 1 provides a finite-budget error probability upper bound for PSST, with complexity reduced by $O(|\mathcal{C}| N_{\max})$ compared to naive Sequential Halving.

## Highlights & Insights
- **Theoretical characterization of the prompt–inference interaction**: Proposition 2 precisely establishes the necessary and sufficient conditions under which an inference-agnostic policy is optimal (affine transformation), revealing that BoN/MV fundamentally fail to satisfy this condition—providing the theoretical foundation for inference-aware optimization that generalizes to any nonlinear aggregation strategy.
- **Cross-scale nested reuse**: A single pull of $(p, N_{\max})$ yields samples for all $N < N_{\max}$ at no additional cost, substantially reducing budget requirements. This trick transfers naturally to other bandit problems requiring evaluation across different configurations.
- **Hyperparameter-free design**: PSST requires no tuning, and batch queries can exploit API discounts, making deployment highly practical.
- **Context-aware policy learning**: IAPO outputs not a single optimal configuration, but a context-specific optimal policy for each (user preference, budget) pair, enabling personalized alignment.

## Limitations & Future Work
- Only BoN and MV are considered as inference strategies; more complex approaches such as tree search or parallel thinking are not addressed.
- The prompt set $\mathcal{P}$ must be provided a priori; integration with prompt generation/search methods (e.g., OPRO) is left to future work.
- The context space $\mathcal{C}$ is discrete and finite; continuous preference spaces would require function approximation.
- The impact of distribution shift at deployment time is not considered.

## Related Work & Insights
- **vs. TRIPLE-SH (Shi et al. 2024)**: TRIPLE formulates prompt optimization as BAI but is inference-agnostic and single-objective; IAPO introduces contextualized multi-objective optimization and inference awareness, with PSST exploiting IAPO's structure for greater efficiency.
- **vs. BonBon/BOND (Gui 2024, Sessa 2025)**: These methods distill BoN policies into single-pass decoding via fine-tuning, but require white-box access to model weights; IAPO operates in a fully black-box manner.
- **vs. Inference-Aware Fine-Tuning (Chow et al. 2025)**: This white-box fine-tuning method optimizes the BoN exploration–exploitation trade-off during training; it is complementary to IAPO—one may first use IAPO to select prompts and then apply white-box fine-tuning.
- **vs. MORL-Prompt (Jafari et al. 2024)**: Multi-objective prompt optimization that is inference-agnostic and does not account for the nonlinear aggregation effects of BoN/MV.
- **vs. GenARM/DEAL (Xu et al. 2025)**: These inference-time alignment methods require logit access and are inapplicable to purely black-box settings; IAPO requires only final output text and an external scorer.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Inference-aware prompt optimization is an entirely new problem formulation with compelling theoretical characterization
- Experimental Thoroughness: ⭐⭐⭐⭐ Six environments (2 synthetic + 4 real tasks), 200 independent runs, rigorous statistical significance testing
- Writing Quality: ⭐⭐⭐⭐ Motivating examples are intuitive and compelling; theory and experiments are tightly integrated
- Value: ⭐⭐⭐⭐⭐ Provides important practical guidance for joint prompt and inference scale selection in real-world LLM deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Inference-Time Reward Hacking in Large Language Models](../../NeurIPS2025/recommender/inference-time_reward_hacking_in_large_language_models.md)
- [\[ICLR 2026\] From Evaluation to Defense: Advancing Safety in Video Large Language Models](../../ICLR2026/recommender/from_evaluation_to_defense_advancing_safety_in_video_large_language_models.md)
- [\[AAAI 2026\] Moral Change or Noise? On Problems of Aligning AI With Temporally Unstable Human Feedback](moral_change_or_noise_on_problems_of_aligning_ai_with_temporally_unstable_human_.md)
- [\[AAAI 2026\] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)
- [\[ICLR 2026\] GoalRank: Group-Relative Optimization for a Large Ranking Model](../../ICLR2026/recommender/goalrank_group-relative_optimization_for_a_large_ranking_model.md)

</div>

<!-- RELATED:END -->
