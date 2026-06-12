---
title: >-
  [Paper Note] Optimizing Token Choice for Code Watermarking: An RL Approach
description: >-
  [ICML 2026][LLM Safety][Code Watermarking] CodeTracer attaches a small watermark policy network alongside a frozen code LLM. Using GRPO combined with dual rewards (execution pass + z-score) and Gumbel-Top-k straight-thro…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Code Watermarking"
  - "GRPO"
  - "Gumbel-Top-k"
  - "Straight-Through Estimator"
  - "z-score"
date: 2026-05-08
content_hash: 3f70e3224ffcf4d4
---

# Optimizing Token Choice for Code Watermarking: An RL Approach

**Conference**: ICML 2026  
**arXiv**: [2508.11925](https://arxiv.org/abs/2508.11925)  
**Code**: https://github.com/TimeLovercc/CodeTracer (Available)  
**Area**: LLM Security / Code Watermarking / Reinforcement Learning  
**Keywords**: Code Watermarking, GRPO, Gumbel-Top-k, Straight-Through Estimator, z-score

## TL;DR
CodeTracer attaches a small watermark policy network alongside a frozen code LLM. Using GRPO combined with dual rewards (execution pass + z-score) and Gumbel-Top-k straight-through estimation, it jointly learns "where to watermark and which green tokens to select." It improves detection AUROC from ~70% to ~78% with almost no drop in Pass@1.

## Background & Motivation

**Background**: Dominant LLM watermarking (the green-red scheme by Kirchenbauer 2023) randomly splits the vocabulary into green/red sets during generation, applying a fixed logit bias $\delta$ to green tokens. Detection utilizes a z-test based on green token frequency. This performs well in natural language where multiple semantically equivalent tokens often exist at most positions.

**Limitations of Prior Work**: In code generation (1) many positions are syntactically mandatory (`def`, brackets, keywords), where changes cause compilation failure; (2) transparency to perturbations is heterogeneous across positions (variable names are flexible, API names are not); (3) low-entropy distributions make indiscriminate biasing prone to breaking code. Early methods like SWEET or CodeIP either require the original LLM's logits/prompt entropy during detection or rely on manually written transformation rules for each language, resulting in high deployment barriers.

**Key Challenge**: The "statistical detectability" of watermarks and the "functional correctness" of code are fundamentally adversarial under low entropy and strong syntactic constraints—weak biases are undetectable, while strong biases break the code.

**Goal**: (i) Automatically identify safe watermarking positions, (ii) select a function-preserving green token set $G$ at those positions, and (iii) ensure the detection process does not depend on the original LLM.

**Key Insight**: Modeling "whether to watermark $w$" and the "green set $G$" as a context-dependent policy $\pi_\phi(a\mid\mathbf{c})$, which combines with the frozen LLM $\pi_\theta$ to form $\pi_{\theta\oplus\phi}$. Reinforcement learning is used to learn syntax/semantic constraints because code provides two natural verifiable rewards: unit test passage and z-score significance.

**Core Idea**: Formulate code watermarking as an RL problem where a small policy network biases the next-token distribution of the LLM, utilizing GRPO, STE, and Gumbel-Top-k to integrate discrete decisions into end-to-end gradients.

## Method

### Overall Architecture
Upon receiving prompt $\mathbf{x}$, the frozen LLM $\pi_\theta$ computes logits $\mathbf{l}\in\mathbb{R}^{|\mathcal{V}|}$. Simultaneously, the trainable watermark policy $\pi_\phi$ observes a fixed-window context $\mathbf{c}$ and outputs $(w, G)$: $w\in\{0,1\}$ indicates whether to watermark the position, and $G\subset\mathcal{V}$ is the green token set of size $k=\lfloor\gamma|\mathcal{V}|\rfloor$. The combined watermarked logits are $\tilde{l}_j = l_j + w\cdot\delta\cdot\mathbb{1}_{v_j\in G}$, followed by softmax sampling to obtain $\tilde y_t$. During detection, only $\pi_\phi$ is needed to replay $(w, G)$ for each position, and a z-test is performed on the subset where $w=1$: $z = (N_G - T\gamma)/\sqrt{T\gamma(1-\gamma)}$. The original LLM is not required.

### Key Designs

1. **Policy-driven watermarking with frozen LLM**:
    - **Function**: Upgrades the fixed green-red partitioning to a context-aware, learnable policy and ensures the trained $\pi_\phi$ is a plug-and-play module.
    - **Mechanism**: LLM parameters $\theta$ remain frozen during training; only the watermark model $\phi$ is learned (approx. 118M parameters, <10% of a 1.5B base LLM). $\pi_\phi$ is a small Transformer outputting a $(|\mathcal{V}|+1)$-dimensional vector $(w_\phi, \mathbf{l}_\phi)$, where $w_\phi$ determines $w$ and $\mathbf{l}_\phi$ determines the ranking of $G$.
    - **Design Motivation**: Freezing the LLM avoids side effects on coding capabilities caused by fine-tuning (as seen in Xu et al. 2024); meanwhile, $\pi_\phi$ can serve as a plug-in for larger, unseen LLMs.

2. **GRPO + Three-part rewards (execution + z-score + process token-level)**:
    - **Function**: Enables the policy to learn "where to watermark & what to select" without pre-labeled "watermarked code" data.
    - **Mechanism**: Utilizing the GRPO framework, rewards comprise: $R_1$ (execution reward: 1 if all tests pass, 0 otherwise); $R_2$ (saturated z-score reward: 1 if $z\geq 4$, linear between $0<z<4$, 0 if $z\leq 0$); $R_3$ (token-level process reward: $+1$ if $w_t=1$ and $s_t\in G_t$, $-1$ if in red set, 0 if no watermark). The advantage function $\hat A(s_t, a_t) = (A_1 + A_2)\cdot\mathbb{1}_{\text{is\_code}}(s_t)$ combines outcome-level $A_1$ and token-level $A_2$ while masking non-code tokens.
    - **Design Motivation**: Pure outcome rewards provide identical signals to every token, which is too coarse for identifying "where to add." Process-level $R_3$ significantly accelerates training. $\mathbb{1}_{\text{is\_code}}$ prevents wasting the watermark budget on CoT passages.

3. **STE + Gumbel-Top-k making $(w, G)$ differentiable**:
    - **Function**: Bypasses the non-differentiability of $w\in\{0,1\}$ and "selecting top-$k$ from $|\mathcal{V}|$," enabling end-to-end training.
    - **Mechanism**: For $w$, a Straight-Through Estimator is used: $w = \mathbb{1}_{w_\phi>0} + \sigma(w_\phi) - \text{sg}(\sigma(w_\phi))$. For $G$, Gumbel-Top-$k$ is applied: perturbed logits $\mathbf{g} = \mathbf{l}_\phi + (-\log(-\log \mathbf{u}))$ where $\mathbf{u}\sim\text{Uniform}(0,1)^{|\mathcal{V}|}$ are used to select $G$. An indicator $\mathbf{l}_G = \mathbb{1}_{v\in G} + \mathcal{S}(\mathbf{g}) - \text{sg}(\mathcal{S}(\mathbf{g}))$ allows hard selection in the forward pass and Gumbel-Softmax relaxation in the backward pass.
    - **Design Motivation**: Unlike Categorical reparameterization, Gumbel-Top-$k$ (Xie & Ermon 2019) natively handles fixed-size sets. The discrete property preserves statistical verifiability, while continuity feeds the policy gradient.

### Loss & Training
The final objective is the GRPO clipped objective with KL regularization:

$$\max_\phi \mathbb{E}_{s\sim\mathcal{D}}\left[\frac{1}{|s|}\sum_t \min\left(r_t(\phi)\hat A_t, \text{clip}(r_t(\phi), 1-\varepsilon, 1+\varepsilon)\hat A_t\right)\right] - \beta D_{\text{KL}}(\pi_{\theta\oplus\phi}\|\pi_{\text{ref}})$$

where $r_t(\phi) = \pi_{\theta\oplus\phi}(s_t|s_{<t})/\pi_{\text{ref}}(s_t|s_{<t})$. The reference policy $\pi_{\text{ref}}$ is an old copy of $\pi_{\theta\oplus\phi}$. Training involves an initial SFT phase for code token distribution followed by GRPO, completing in ~1 day on a single A100. Base LLM is OpenCoder-1.5B-Instruct.

## Key Experimental Results

### Main Results
Comparison on HumanEval / MBPP against post-hoc detection and active watermarking:

| Dataset | Method | Pass@1 (%) | AUROC (%) | TPR@5%FPR (%) |
|--------|------|-----------|-----------|---------------|
| HumanEval | Base (No Watermark) | 65.42 | – | – |
| HumanEval | WLLM | 58.05 | 70.17 | 20.73 |
| HumanEval | EXP-edit | 59.29 | 66.50 | 25.61 |
| HumanEval | SWEET† | 60.46 | 76.24 | 27.44 |
| HumanEval | **Ours (CodeTracer)** | **62.65** | **77.71** | **32.32** |
| MBPP | Base | 43.35 | – | – |
| MBPP | WLLM | 39.66 | 76.44 | 27.80 |
| MBPP | SWEET† | 39.64 | 77.24 | 24.80 |
| MBPP | **Ours (CodeTracer)** | **42.10** | **78.42** | **31.60** |

Post-hoc methods (LogRank, DetectGPT) show AUROC around 47–52% (near random). Ours shows the smallest drop in Pass@1 (HumanEval -2.77pp vs WLLM -7.37pp) and higher TPR. The $\pi_\phi$ trained on the 1.5B model transfers well to the 8B model, maintaining a Pass@1 of 71.77% (vs Base 72.04%) and AUROC of 78.69%.

### Ablation Study
| Configuration | Pass@1 (%) | AUROC (%) | TPR (%) | Notes |
|------|-----------|-----------|---------|------|
| CodeTracer (Full) | 60.82 | 82.95 | 46.34 | Full three rewards |
| w/o $A_2$ (No token-level $R_3$) | 61.15 | 75.11 | 30.29 | Detection collapses |
| w/o $A_1$ (No outcome reward) | 60.34 | 79.52 | 34.91 | Performance drop in both |
| CodeTracer-1 (Pure RL, no SFT) | 62.65 | 77.71 | 32.32 | High functionality mode |
| CodeTracer-2 (SFT + RL) | 60.82 | 82.95 | 46.34 | High detection mode |

### Key Findings
- Process-level reward $R_3$ is the most critical: omitting it drops AUROC by 7.84pp and TPR by 16pp, indicating that token-level feedback contributes significantly more to convergence than sequence-level z-score feedback.
- SFT initialization provides a "knob" between detectability and functionality.
- Inference overhead is negligible: $\pi_\phi$ runs in parallel with the LLM, adding <100μs latency and <0.5GB VRAM.
- Performance is consistent across Java / C++ (HumanEvalPack), suggesting the RL learns general "safe-to-modify" priors.

## Highlights & Insights
- **Automating watermarking position discovery**: CodeTracer bypasses manual AST rules or entropy dependence by using RL to learn the policy, removing language-specific engineering hurdles.
- **Suitability of Gumbel-Top-k**: The green set problem is inherently a "fixed-size subset sampling" problem, making Gumbel-Top-k a more natural fit than standard categorical reparameterization.
- **Process rewards dominate**: In contrast to the intuition that end-to-end outcome-only RL is superior, dense signals (when cheaply available) prove much more effective for sequence-based tasks.
- **Zero LLM dependency at detection**: Distilling the watermarking logic into $\pi_\phi$ allows third-party verification without exposing the base LLM.

## Limitations & Future Work
- Vulnerable to strong semantic rewriting (AUROC falls to 58.42% under DIPPER). 
- Training depends on a runtime sandbox for rollouts, which limits scalability across diverse libraries.
- Hyperparameters like $\gamma$ and $\delta$ are still globally fixed rather than adaptive per position.
- White-box attacks (where the adversary possesses $\pi_\phi$) were not explored.

## Related Work & Insights
- **Vs. WLLM (Kirchenbauer 2023a)**: WLLM uses fixed green/red splits and $\delta$; Ours makes these learnable. CodeTracer halves the Pass@1 drop compared to WLLM in code scenarios.
- **Vs. SWEET (Lee 2023)**: SWEET uses entropy thresholds but requires the original LLM for detection; CodeTracer internalizes this "where-to-add" logic into $\pi_\phi$.
- **Vs. CodeIP (Guan 2024)**: CodeIP relies on manual token type predictors; CodeTracer automates this via RL.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach](../../AAAI2026/llm_safety/uncovering_pretraining_code_in_llms_a_syntax-aware_attribution_approach.md)
- [\[AAAI 2026\] WaterMod: Modular Token-Rank Partitioning for Probability-Balanced LLM Watermarking](../../AAAI2026/llm_safety/watermod_modular_token-rank_partitioning_for_probability-balanced_llm_watermarki.md)
- [\[ICML 2026\] ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)
- [\[ICML 2026\] Watermarking LLM Agent Trajectories (ACTHOOK)](watermarking_llm_agent_trajectories.md)
- [\[ICML 2026\] Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping](memory_as_a_markov_matrix_sample_efficient_knowledge_expansion_via_token-to-dict.md)

</div>

<!-- RELATED:END -->
