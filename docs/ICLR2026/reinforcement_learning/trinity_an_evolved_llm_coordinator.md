---
title: >-
  [Paper Note] Trinity: An Evolved LLM Coordinator
description: >-
  [ICLR 2026][Reinforcement Learning][LLM coordination] Trinity introduces a lightweight coordinator (0.6B SLM + ~10K trainable parameters in a linear head) optimized via sep-CMA-ES. In multi-turn dialogues…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "LLM coordination"
  - "model composition"
  - "evolutionary strategy"
  - "CMA-ES"
  - "multi-role collaboration"
  - "test-time composition"
date: 2026-05-08
content_hash: 9c72a69ba109882c
---

# Trinity: An Evolved LLM Coordinator

**Conference**: ICLR 2026
**arXiv**: [2512.04695](https://arxiv.org/abs/2512.04695)  
**Code**: None (Sakana AI)  
**Area**: Reinforcement Learning / LLM Coordination
**Keywords**: LLM coordination, model composition, evolutionary strategy, CMA-ES, multi-role collaboration, test-time composition

## TL;DR
Trinity introduces a lightweight coordinator (0.6B SLM + ~10K trainable parameters in a linear head) optimized via sep-CMA-ES. In multi-turn dialogues, the coordinator routes queries to different LLMs and assigns one of three roles—Thinker, Worker, or Verifier—achieving 86.2% pass@1 SOTA on LiveCodeBench and consistently outperforming all single-model and multi-agent baselines across 4 in-distribution and 4 out-of-distribution tasks.

## Background & Motivation

**Background**: LLM scaling laws remain effective but are increasingly costly with diminishing returns. Model merging is constrained by architectural incompatibility and closed-source APIs. Macro-level test-time model composition (coordination) represents a promising alternative.

**Limitations of Prior Work**: (1) Existing routing/coordination methods (MasRouter, RouterDC, Smoothie, etc.) fail to effectively exploit the complementary strengths of diverse models; some even degrade performance below random selection. (2) These methods lack rich contextual understanding of input queries to make effective delegation decisions.

**Key Challenge**: A coordinator must possess sufficient semantic understanding to correctly assign tasks, yet need not—and should not—be as powerful as the underlying agents. The core challenge is learning the most effective coordination strategy with minimal parameters.

**Goal**: (1) How to extract sufficient semantic signals from a small model's internal representations for coordination? (2) How to optimize coordination strategies under an extreme parameter budget (~10K)? (3) How to design effective multi-turn collaboration patterns?

**Key Insight**: Leveraging SLM hidden states (rather than generated text) as contextual representations, using an ultra-lightweight head for routing decisions, and optimizing via evolutionary strategies rather than RL.

**Core Idea**: The hidden states of a small model contain sufficient semantic signals such that a head with fewer than 20K parameters can coordinate multiple top-tier LLMs to surpass any single model.

## Method

### Overall Architecture
The coordinator consists of a Qwen3-0.6B SLM and a linear head (~10K parameters). At each turn, the full dialogue transcript is fed into the coordinator; the head produces two sets of logits from the hidden state—one for LLM selection and one for role assignment (T/W/V). A message processing module injects role-specific prompts before dispatching to the selected LLM.

### Key Designs

1. **Efficient Parameterization**:

    - Head: a single linear mapping from hidden state $h \in \mathbb{R}^d$ to $\mathbb{R}^{L+3}$ logits ($L$ LLMs + 3 roles).
    - SVD fine-tuning: SVD decomposition is applied to selected SLM weight matrices, learning only singular value scaling while keeping orthogonal matrices fixed.
    - Total parameters < 20K, orders of magnitude smaller than typical fine-tuning.
    - Key insight: the coordinator's **generated text is discarded**; only the hidden state logits are used—enabling fast decisions from early-token hidden states.

2. **Tri-role Coordination**:

    - **Thinker**: Strategic planning—analyzes state and returns high-level guidance (plans, decompositions, critiques).
    - **Worker**: Concrete execution—produces code, derivations, numerical results, and other actionable content.
    - **Verifier**: Quality assessment—outputs ACCEPT/REVISE with optional diagnostic information.
    - Termination: triggered when the Verifier is selected and outputs ACCEPT, or when the fixed turn limit $K$ is reached.
    - Design Motivation: complex skill acquisition is offloaded to the underlying LLMs; the coordinator only performs lightweight assignment decisions.

3. **sep-CMA-ES Optimization**:

    - Problem characteristics: high-dimensional (~10K parameters), weak inter-parameter coupling, high per-step cost (each step requires running full coordinated agent inference), binary terminal reward.
    - Why not RL: REINFORCE yields extremely low SNR per-parameter gradients in this setting—weak inter-block coupling leads to ill-conditioned gradients and poor credit assignment.
    - Why sep-CMA-ES: maintains a diagonal covariance matrix, particularly well-suited to block-diagonal loss landscapes; theoretically superior to RL and random search under high-dimensional, strict-budget constraints.
    - Theoretical guarantee: Proposition 1 proves that in the small-$T$ regime, sep-CMA-ES improvement grows linearly with iterations, whereas random search grows only logarithmically.

### Objective Function
$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$, where $R(\tau) \in \{0,1\}$ is the terminal reward (correct/incorrect answer).

## Key Experimental Results

### In-Distribution Evaluation (4 Benchmarks)

| Method | MATH500 | MMLU | RLPR | LiveCodeBench v6 |
|--------|---------|------|------|------------------|
| GPT-5 (4K) | 0.91 | 0.92 | 0.34 | 0.56 |
| Gemini-2.5-pro (4K) | 0.92 | 0.91 | 0.41 | 0.47 |
| Claude-4-Sonnet (4K) | 0.90 | 0.89 | 0.37 | 0.51 |
| MoA | 0.83 | - | 0.38 | 0.39 |
| **Trinity** | **0.95** | **0.94** | **0.44** | **0.61** |

Trinity consistently leads across all 4 tasks. Relative error reduction on MATH500: 11.76% (vs. Gemini-2.5-pro at 5× context). LiveCodeBench SOTA: **86.2% pass@1** (V1 train → V6 test).

### Zero-Shot Transfer (4 Unseen Tasks)

| Model | AIME | BigCodeBench | MT-Bench | GPQA-D | Average |
|-------|------|-------------|----------|--------|---------|
| Gemini Pro 2.5 | 46.67 | 35.10 | 9.37 | 75.25 | 52.34 |
| GPT-5 | 46.67 | 33.80 | 9.35 | 72.73 | 51.07 |
| **Trinity** | **50.00** | **35.80** | **9.60** | **76.82** | **54.21** |

Trinity surpasses every individual model on all 4 unseen tasks, demonstrating strong generalization.

### Key Findings
- Average relative error reduction of 21.9% versus the second-best method.
- Certain baselines degrade performance below random (e.g., RouterDC achieves 0.28 on RLPR vs. random 0.32), highlighting the difficulty of effective coordination.
- Trinity approaches the "Per-Question-Best" upper bound on 3 of 4 tasks.
- Emergent task-aware strategies: distinct T/W/V selection patterns are observed across different task types.

## Ablation Study
- Head architecture: block-diagonal-10 (extremely few parameters) retains most performance, confirming block-$\varepsilon$-separability.
- SVD fine-tuning vs. no fine-tuning: fine-tuning provides additional representational improvement.
- sep-CMA-ES vs. REINFORCE vs. random search vs. imitation learning: CMA-ES substantially outperforms all alternatives in this regime.

## Highlights & Insights
- **Extreme parameter efficiency**: fewer than 20K trainable parameters coordinate 7 top-tier LLMs (including GPT-5 and Claude-4-Sonnet)—a remarkable scale contrast.
- **Semantic density of hidden states**: demonstrates that even the internal representations of a 0.6B SLM provide rich contextual signals sufficient for effective coordination.
- **The niche of evolutionary strategies over RL**: in the specific regime of high dimensionality, weak coupling, sparse rewards, and high per-step cost, CMA-ES outperforms policy gradient methods both theoretically and empirically—challenging the assumption that RL is universally superior.
- **Elegance of the tri-role design**: the T/W/V division of labor frees the coordinator from complex skill acquisition, reducing its task to pure assignment.

## Limitations & Future Work
- Dependence on closed-source API LLM pools makes cost and latency practical bottlenecks for deployment.
- The SLM coordinator must process the full transcript each turn, which may be inefficient for very long dialogues.
- The three-role prompt design is hand-crafted; automated role discovery warrants exploration.
- Training set size is relatively small (400 LiveCodeBench samples); the effect of larger-scale training remains to be validated.

## Related Work & Insights
- **vs. MoA/LLM-Blender**: simple mixture/fusion approaches are insufficient—effective coordination requires query-level contextual understanding.
- **vs. RouterDC/MasRouter**: existing routing methods lack the capacity for multi-turn reasoning and role assignment.
- **vs. Model merging**: Trinity does not modify any underlying model weights, making it compatible with closed-source and heterogeneous models.
- **vs. Self-reflection**: single-model self-reflection (5× SR) still underperforms Trinity, as it cannot leverage inter-model complementarity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of SLM hidden states, an ultra-lightweight head, and CMA-ES is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 benchmarks (4 in-distribution + 4 zero-shot), comprehensive ablations, and theoretical analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Precise problem formulation, rigorous theoretical analysis, and clear experimental presentation.
- Value: ⭐⭐⭐⭐⭐ Achieves LiveCodeBench SOTA and establishes a new paradigm for ultra-lightweight coordination.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] References Improve LLM Alignment in Non-Verifiable Domains](references_improve_llm_alignment_in_non-verifiable_domains.md)
- [\[ICLR 2026\] ReMix: Reinforcement Routing for Mixtures of LoRAs in LLM Finetuning](remix_reinforcement_routing_lora.md)
- [\[ICLR 2026\] How Far Can Unsupervised RLVR Scale LLM Training?](how_far_can_unsupervised_rlvr_scale_llm_training.md)
- [\[ICLR 2026\] $\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving](textbfre2_unlocking_llm_reasoning_via_reinforcement_learning_with_re-solving.md)
- [\[ACL 2026\] Efficient Hyperparameter Optimization for LLM Reinforcement Learning](../../ACL2026/reinforcement_learning/efficient_hyperparameter_optimization_for_llm_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
