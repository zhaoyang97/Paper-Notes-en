---
title: >-
  [Paper Note] Alignment of Large Language Models with Constrained Learning
description: >-
  [NeurIPS 2025][LLM Alignment][Constrained Alignment] This paper proposes CAID (Constrained Alignment via Iterative Dualization), an iterative dualization method that alternately updates the LLM policy and dual variables. It theoretically establishes that the dual approach can identify the optimal constrained LLM policy (up to a parametrization gap), and empirically demonstrates significant improvements in constraint satisfaction and the helpfulness–safety trade-off on the PKU…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "Constrained Alignment"
  - "Lagrangian Duality"
  - "Multi-objective Optimization"
  - "Safe RLHF"
  - "DPO"
date: 2026-05-08
content_hash: f29e47a01ea31e77
---

# Alignment of Large Language Models with Constrained Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.19387](https://arxiv.org/abs/2505.19387)  
**Code**: None  
**Area**: Alignment / RLHF
**Keywords**: Constrained Alignment, Lagrangian Duality, Multi-objective Optimization, Safe RLHF, DPO

## TL;DR

This paper proposes CAID (Constrained Alignment via Iterative Dualization), an iterative dualization method that alternately updates the LLM policy and dual variables. It theoretically establishes that the dual approach can identify the optimal constrained LLM policy (up to a parametrization gap), and empirically demonstrates significant improvements in constraint satisfaction and the helpfulness–safety trade-off on the PKU-SafeRLHF dataset.

## Background & Motivation

**Background**: RLHF is the dominant paradigm for LLM alignment, yet a single reward model is insufficient to capture the full dimensionality of human preferences. Existing approaches fall into two categories: multi-objective alignment (aggregating rewards via weighted combinations) and constrained alignment (maximizing a primary reward subject to secondary constraints). Constrained alignment is more natural in safety-critical settings—for example, requiring that safety improvements exceed a specified threshold while preserving helpfulness.

**Limitations of Prior Work**: Lagrangian-based policy search for LLMs suffers from two key issues: (1) iterative primal-dual methods (e.g., Safe RLHF) may fail to converge in the worst case; (2) non-iterative dual methods (e.g., One-shot Safety Alignment) can find optimal solutions in distribution space but provide no guarantee of optimality in LLM parameter space.

**Key Challenge**: The convexity properties of distribution space do not transfer directly to LLM parameter space. Strong duality in distribution space does not imply optimality in parameter space, and prior work lacks theoretical analysis of whether dual methods can identify optimal constrained policies in LLM parameter space.

**Goal**: (1) Design a practical iterative dual alignment method; (2) establish theoretical optimality guarantees in LLM parameter space; (3) empirically validate effectiveness on safety alignment tasks.

**Key Insight**: The authors draw on constrained learning theory, leveraging Lagrangian duality to connect the non-convex parameter space problem to the convex distribution space problem, and analyze the *parametrization gap* to bridge the optimality difference between the two.

**Core Idea**: Through multi-shot iterative alternation between LLM policy updates and dual variable descent, initialized with the one-shot solution as a warm start, the method achieves both theoretical optimality and strong empirical performance.

## Method

### Overall Architecture

Given a pretrained reference model $\pi_{\text{ref}}$, a reward model $r$ (helpfulness), and a utility model $g$ (safety), CAID alternates between two steps: (1) fixing the dual variable $\lambda$ and updating the LLM policy by maximizing the Lagrangian via DPO; (2) fixing the policy and updating $\lambda$ via dual subgradient descent. The output is an aligned LLM satisfying the safety constraint.

### Key Designs

1. **Iterative Dual Alignment Algorithm (CAID)**:

    - **Function**: Find the optimal constrained policy in LLM parameter space.
    - **Mechanism**: Decomposes the constrained alignment problem into a dual problem and solves it via alternating updates. At each iteration $t$, the dual variable is first updated along the subgradient direction $u(\lambda^{(t)}) = \mathbb{E}[\mathbb{E}_{y \sim \pi}[g(x,y)] - \mathbb{E}_{y \sim \pi_{\text{ref}}}[g(x,y)]] - b$, yielding $\lambda^{(t+1)} = [\lambda^{(t)} - \eta u(\lambda^{(t)})]_+$; the policy is then updated by maximizing the composite reward $r_\lambda = r + \lambda^\top g$ via DPO. The one-shot solution serves as the warm-start initialization.
    - **Design Motivation**: The one-shot method obtains the optimal dual variable in distribution space but incurs error in parameter space; multi-shot iteration corrects this error over successive rounds, and warm-start initialization substantially accelerates convergence.

2. **Two Practical Implementations (MoCAID and PeCAID)**:

    - **Function**: Realize CAID in settings with and without explicit reward models, respectively.
    - **Mechanism**: MoCAID (model-based) directly scores responses using reward and utility models, constructs pseudo-preference pairs via the Bradley-Terry model, and feeds them to DPO. PeCAID (preference-based) operates when only human preference annotations are available: it first pre-aligns separately on $r$ and $g$ via DPO to obtain implicit reward models $\beta \log(\pi_{\theta_r}/\pi_{\text{ref}})$, then constructs composite preferences.
    - **Design Motivation**: The two variants cover different practical scenarios—some applications provide explicit scorers, while others only have preference data.

3. **Theoretical Optimality Analysis**:

    - **Function**: Prove that CAID can approximate the optimal constrained LLM policy.
    - **Mechanism**: The parametrization gap $\nu$ is defined to measure how well LLM parameter space covers distribution space. The analysis proves that the primal-dual gap satisfies $|D_p^* - P^*| \lesssim \nu$ (Theorem 2), and that the optimality gaps of the learned policy in both reward and constraint functions are $O(\nu)$ (Theorems 3 and 4).
    - **Design Motivation**: This constitutes the first complete optimality guarantee for constrained LLM alignment, filling a significant theoretical gap in the literature.

### Loss & Training

Training employs the DPO loss with composite reward $r_\lambda = r + \lambda^\top g$. Dual variables are updated via projected subgradient descent. The online dataset is constructed by sampling 600 prompts × 64 responses from the current policy. Training uses LoRA (rank=8, alpha=16) with a cosine learning rate schedule (lr=5e-4) over 4 iterations.

## Key Experimental Results

### Main Results

| Dataset / Setting | Metric | Multi-shot (CAID) | One-shot | Notes |
|---|---|---|---|---|
| PKU-SafeRLHF (b=5) | Safety Improvement | 5.758 | 2.285 | Multi-shot more accurately satisfies constraint |
| PKU-SafeRLHF (b=5) | Helpfulness Improvement | 9.769 | 7.248 | Multi-shot also improves helpfulness |
| PKU-SafeRLHF (b=9) | Safety Improvement | 11.420 | 9.574 | Gap narrows under high constraint threshold |
| PKU-SafeRLHF (b=9) | Helpfulness Improvement | 6.879 | 4.271 | Multi-shot retains clear advantage |
| GPT-4o-mini Eval | Helpfulness Win Rate | ~55–60% | baseline | Multi-shot wins at most values of $b$ |
| GPT-4o-mini Eval | Safety Win Rate | ~55–65% | baseline | Multi-shot achieves better safety |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| DPO (helpfulness only) | H.I.=high, S.I.=low | Single-objective cannot satisfy safety constraint |
| DPO (safety only) | H.I.=low, S.I.=high | Excessive sacrifice of helpfulness |
| One-shot (small $b$) | Safety < threshold | Insufficient at low constraint levels |
| One-shot (large $b$) | Safety > threshold | Over-satisfies at high constraint levels |
| Multi-shot (warm-start) | Converges in 4 iterations | Warm-start enables rapid dual variable convergence |
| Multi-shot (AdvBench) | Safety Win >60% | Safety advantage maintained under adversarial evaluation |

### Key Findings

- The multi-shot method more closely tracks the target safety constraint across all thresholds $b \in \{3,\ldots,9\}$, whereas the one-shot method under-satisfies at small $b$ and over-satisfies at large $b$.
- Multi-shot strictly dominates one-shot on the Pareto frontier of the helpfulness–safety trade-off.
- Adversarial evaluation on AdvBench confirms that multi-shot models achieve higher safety scores at all constraint levels.
- Red-teaming case studies show that multi-shot models effectively refuse harmful requests, whereas one-shot models may still produce partially harmful content.

## Highlights & Insights

- This work is the first to provide complete optimality guarantees for constrained LLM alignment, proving that dual methods can identify the optimal policy up to the parametrization gap $\nu$.
- The warm-start strategy is practically elegant—initializing dual variables with the one-shot solution means multi-shot only requires fine-grained adjustment within a small neighborhood, adding only approximately 170 minutes of training time.
- Theory and experiment align well: Theorem 2 predicts that the parametrization gap governs optimization quality, and the multi-shot method empirically corrects the parameter-space errors of the one-shot approach through iterative refinement.

## Limitations & Future Work

- Experiments are conducted only on a 7B-scale model (Alpaca-7b); the effectiveness of multi-shot at larger scales remains to be validated.
- Only a single constraint (safety) is considered; theoretical and empirical investigation of multi-constraint settings (e.g., simultaneously constraining safety, toxicity, and bias) is needed.
- The theory relies on the parametrization gap assumption, whose quantification in practical LLMs remains unclear.
- The convergence rate of dual variables depends on the number of prompts and sampled responses; the computational overhead in deployment warrants careful consideration.

## Related Work & Insights

- **vs. Safe RLHF (Dai et al., 2024)**: Safe RLHF employs an iterative primal-dual method that simultaneously updates the policy and dual variables, which may fail to converge in the worst case. CAID avoids this by adopting an alternating strategy that updates the dual variable before the policy.
- **vs. One-shot Safety Alignment (Huang et al., 2024)**: The one-shot method computes the optimal dual variable in distribution space and performs a single alignment pass, but provides no optimality guarantee in parameter space. CAID generalizes the one-shot approach to a multi-shot framework, preserving its advantages while correcting parametrization error.
- **vs. SimPO / DPO variants**: The CAID framework is orthogonal to the specific alignment algorithm and DPO can be replaced with variants such as SimPO.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First complete optimality theory for constrained LLM alignment; the multi-shot warm-start design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Includes scorer-based evaluation, GPT-4o-based evaluation, adversarial evaluation, and red-teaming case studies, providing multi-faceted validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — 51 pages including appendix; theoretical derivations are complete and rigorous; experimental details are thorough.
- **Value**: ⭐⭐⭐⭐ — Provides both theoretical foundations and practical methods for constrained alignment, though the 7B scale limits immediate applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](reinforcement_learning_finetunes_small_subnetworks_in_large_language_models.md)
- [\[NeurIPS 2025\] Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency](adjacent_words_divergent_intents_jailbreaking_large_language_models_via_task_con.md)
- [\[ACL 2025\] Safety Alignment via Constrained Knowledge Unlearning](../../ACL2025/llm_alignment/safety_alignment_via_constrained_knowledge_unlearning.md)
- [\[NeurIPS 2025\] LASeR: Learning to Adaptively Select Reward Models with Multi-Armed Bandits](laser_learning_to_adaptively_select_reward_models_with_multi-armed_bandits.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)

</div>

<!-- RELATED:END -->
