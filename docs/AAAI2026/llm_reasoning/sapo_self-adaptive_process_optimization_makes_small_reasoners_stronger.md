---
title: >-
  [Paper Note] SAPO: Self-Adaptive Process Optimization Makes Small Reasoners Stronger
description: >-
  [AAAI 2026][Reasoning][process supervision] Inspired by Error-Related Negativity (ERN) in neuroscience, this paper proposes SAPO, a self-adaptive process optimization method that replaces costly step-wise Monte Carlo rollouts with first error detection and local posterior estimation. SAPO reduces computational cost by 2–3× while enabling joint optimization of the reasoner and verifier, allowing small language models (≤2B) to outperform most self-evolution methods on mathemati…
tags:
  - "AAAI 2026"
  - "Reasoning"
  - "process supervision"
  - "self-evolution"
  - "first error detection"
  - "small language models"
  - "reasoner-verifier gap"
date: 2026-05-08
content_hash: b26ee9aebd7a1379
---

# SAPO: Self-Adaptive Process Optimization Makes Small Reasoners Stronger

**Conference**: AAAI 2026
**arXiv**: [2601.20312](https://arxiv.org/abs/2601.20312)  
**Code**: N/A  
**Area**: LLM Reasoning
**Keywords**: process supervision, self-evolution, first error detection, small language models, reasoner-verifier gap

## TL;DR
Inspired by Error-Related Negativity (ERN) in neuroscience, this paper proposes SAPO, a self-adaptive process optimization method that replaces costly step-wise Monte Carlo rollouts with first error detection and local posterior estimation. SAPO reduces computational cost by 2–3× while enabling joint optimization of the reasoner and verifier, allowing small language models (≤2B) to outperform most self-evolution methods on mathematical and code reasoning tasks.

## Background & Motivation
**Background**: Large language models exhibit strong reasoning capabilities but incur high computational costs. Small language models (SLMs, ≤2B) are viable candidates for deployment on mobile devices. Self-evolution methods enhance SLM reasoning through a reasoner–verifier interaction framework.

**Limitations of Prior Work**:
- Most self-evolution methods rely solely on outcome rewards, neglecting fine-grained step-level feedback, which leads to reward hacking.
- Monte Carlo process supervision is more precise but computationally prohibitive (10k problems × 8 steps × 10 trajectories = 800k rollouts).
- Across multiple iterations, the gap between the reasoner and verifier widens (Figure 1), degrading the quality of verifier evaluations.

**Key Challenge**: A fundamental trade-off exists between the granularity of process supervision and computational efficiency—step-level feedback is necessary to reduce the reasoner-verifier gap, yet full MC estimation is too costly.

**Goal**: To achieve joint optimization of the reasoner and verifier without significantly increasing supervision cost.

**Key Insight**: Inspired by Error-Related Negativity (ERN)—humans can rapidly localize the point of error after making a mistake and adjust their behavior accordingly—only the first error needs to be located to provide effective process supervision.

**Core Idea**: Replace step-wise MC rollouts with verifier-based first error localization followed by local posterior verification by the reasoner, introducing process supervision signals at minimal cost.

## Method

### Overall Architecture
SAPO adopts an iterative explore-exploit paradigm. In each iteration: the verifier pre-scores trajectories → first error detection → reasoner posterior verification → verifier update → preference data construction → reasoner alignment. The reasoner and verifier co-evolve throughout this process.

### Key Designs
1. **First Error Detection**

    - Function: The verifier $V$ scores each step in a trajectory, and the position with the largest score drop is identified as the candidate first error location.
    - Mechanism: Computes score differences between adjacent steps $\Delta_j = \hat{c}_{j-1} - \hat{c}_j$, and selects $\hat{t} = \arg\max_j \Delta_j$.
    - Design Motivation: The first error position alone is sufficient to provide effective process supervision (Uesato et al. 2022), eliminating the need to annotate every step.

2. **Self-Verification**

    - Function: The reasoner performs posterior estimation at the verifier-predicted first error location to correct pre-annotated labels.
    - Mechanism: Rollout verification is conducted only at positions $\hat{t}-1$ and $\hat{t}$, covering three cases:
     - Case (a): $c_{\hat{t}-1}=1, c_{\hat{t}}=0$ → prediction is correct.
     - Case (b): $c_{\hat{t}-1}=1, c_{\hat{t}}=1$ → first error lies further ahead ($\hat{t}<t$); extend correct labels.
     - Case (c): $c_{\hat{t}-1}=0, c_{\hat{t}}=0$ → first error occurred earlier ($\hat{t}>t$); shift error label backward.
    - Design Motivation: Only 2 rollouts are required for verification, compared to full step-wise rollouts under MC estimation.

3. **Expansion**

    - Function: Leverages trajectories generated during rollout to improve the generalization of verification.
    - Mechanism: If $c_{\hat{t}}=1$, all steps of the correct trajectory prefixed by $s_{(0:\hat{t})}$ are labeled correct; if $c_{\hat{t}}=0$, all suffixes starting from $s_{\hat{t}}$ are labeled incorrect.
    - Design Motivation: Rollouts naturally yield diverse samples that can be reused to augment PRM training data.

### Loss & Training
- **Verifier Training**: A classification PRM trained with MSE loss — $\mathcal{L}_{PRM} = \frac{1}{n}\sum_i\sum_j(f(s_{(0:j)}^i;q) - c^k)^2$
- **Reasoner Alignment**: Preference pairs are constructed using the ORPO algorithm — $\mathcal{L}_{ORPO} = \mathbb{E}[\mathcal{L}_{SFT}(q,\tau^w) - \beta\log\sigma(\frac{\text{odds}(\tau^w|q)}{\text{odds}(\tau^l|q)})]$
- **Preference Data Construction**: A threshold $\eta$ is set; a positive–negative sample pair is constructed when $r(\tau_i^w) - r(\tau_i^l) \geq \eta$.
- **Iteration Strategy**: 3 iterations in total; the verifier is updated before the reasoner is optimized in each round.

## Key Experimental Results

### Main Results — Mathematical Reasoning and Code Generation

| Method | Qwen-2.5-0.5B GSM8K | Qwen-2.5-0.5B MATH (OOD) | Llama-3.2-1B GSM8K | Gemma-2-2B GSM8K |
|--------|----------------------|---------------------------|---------------------|-------------------|
| CoT | 28.51 | 25.97 | 5.31 | 19.86 |
| SFT | 34.19 | 24.84 | 22.14 | 39.12 |
| RFT | 37.83 | 27.35 | 26.31 | 45.34 |
| Online-RFT | 40.79 | 28.85 | 29.03 | 48.67 |
| SFT+GRPO | **46.24** | **34.53** | 26.46 | 44.65 |
| SAPO-iter3 | 41.62 | 31.72 | **34.19** | **49.73** |

| Method | Qwen-2.5-0.5B MBPP | Llama-3.2-1B MBPP | Gemma-2-2B MBPP |
|--------|---------------------|--------------------|--------------------|
| SFT+GRPO | 35.20 | 25.55 | 33.40 |
| SAPO-iter3 | **36.67** | **28.92** | **35.43** |

### Ablation Study — GSM8K

| Method | Qwen-2.5-0.5B | Llama-3.2-1B |
|--------|---------------|--------------|
| SAPO (Full) | **41.62** | **34.19** |
| w/o Process Feedback (PF) | 39.65 | 32.37 |
| w/o Detection & Verification (DV) | 40.86 | 31.69 |
| w/o Reward Model (RM) | 40.71 | 32.75 |
| w/o Expansion (EP) | 40.33 | 33.73 |

### Key Findings
- **Model Dependency of GRPO**: GRPO performs best on Qwen-2.5-0.5B but underperforms SAPO on Llama and Gemma, indicating that GRPO's effectiveness is highly dependent on the base model's capacity.
- **Advantage on Code Tasks**: SAPO consistently outperforms GRPO on code generation, suggesting that process supervision is more effective for structured reasoning.
- **Efficiency Gains**: Compared to Shepherd (step-wise rollout), SAPO reduces FLOPs and wall-clock time by 2–3×.
- **Verifier Performance Improvement**: SAPRM continually improves process verification performance across iterations and outperforms Online ORM on code verification in particular.
- **Importance of Synchronized Alignment**: Reasoner–verifier pairs from the same iteration (e.g., V3-R3) exhibit the lowest error rates; cross-iteration pairings yield higher error rates.

## Highlights & Insights
- **Neuroscience Inspiration**: The ERN analogy provides an elegant intuition—humans do not need to examine every step, but can rapidly localize the first error and adjust accordingly.
- **Adaptive Efficiency**: The first error localization cleverly reduces O(n) MC estimation to O(1) local verification, striking an effective balance between accuracy and efficiency.
- **Two New Benchmarks**: GSM_Process (3,786 instances) and MBPP_Process (1,499 instances) are introduced to evaluate process verifiers, addressing the absence of step-level verification benchmarks.
- **Iterative Convergence Analysis**: Self-verification error rate experiments (Table 2) directly quantify changes in the reasoner–verifier gap across iterations.

## Limitations & Future Work
- SAPO still underperforms SFT+GRPO on mathematical tasks with Qwen-2.5-0.5B (41.62 vs. 46.24), indicating that GRPO remains more effective in certain settings.
- Experiments are limited to models ≤2B; applicability to larger models has not been validated.
- First error detection depends on verifier quality; a weak initial verifier may introduce cascading errors.
- The expansion strategy assumes all steps before the first error are correct and all subsequent steps are incorrect, whereas in practice some later steps may be partially correct.

## Related Work & Insights
- **GRPO** (Shao et al. 2024): Group Relative Policy Optimization is effective for large models but not necessarily optimal for SLMs → SAPO offers a more stable alternative for SLMs.
- **V-STaR** (Hosseini et al. 2024): Uses verifiers to internalize alignment signals → SAPO further advances reasoner–verifier co-optimization.
- **OmegaPRM** (Luo et al. 2024): Employs binary search to localize the first error → SAPO's detect-and-verify strategy provides a more comprehensive solution.

## Rating
- Novelty: ⭐⭐⭐⭐ The ERN-inspired first error detection combined with self-verification is novel, and the efficiency optimization approach is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three backbone models × two task types × multiple baselines, with comprehensive ablation and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; the logic from problem to method to experiment is coherent.
- Value: ⭐⭐⭐⭐ Offers clear practical contributions to SLM self-evolution and efficient process supervision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RSAT: Structured Attribution Makes Small Language Models Faithful Table Reasoners](../../ACL2026/llm_reasoning/rsat_structured_attribution_makes_small_language_models_faithful_table_reasoners.md)
- [\[AAAI 2026\] Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning](small_language_models_for_efficient_agentic_tool_calling_outperforming_large_mod.md)
- [\[ICLR 2026\] Thinking-Free Policy Initialization Makes Distilled Reasoning Models More Effective and Efficient Reasoners](../../ICLR2026/llm_reasoning/thinking-free_policy_initialization_makes_distilled_reasoning_models_more_effect.md)
- [\[AAAI 2026\] In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback](in-token_rationality_optimization_towards_accurate_and_concise_llm_reasoning_via.md)
- [\[ICLR 2026\] StepORLM: A Self-Evolving Framework with Generative Process Supervision for Operations Research Language Models](../../ICLR2026/llm_reasoning/steporlm_a_self-evolving_framework_with_generative_process_supervision_for_opera.md)

</div>

<!-- RELATED:END -->
