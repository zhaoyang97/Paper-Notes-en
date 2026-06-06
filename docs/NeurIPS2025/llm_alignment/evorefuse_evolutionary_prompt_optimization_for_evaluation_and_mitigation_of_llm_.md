---
title: >-
  [Paper Note] EvoRefuse: Evaluating and Mitigating LLM Over-Refusal via Evolutionary Prompt Optimization
description: >-
  [NeurIPS 2025][LLM Alignment][over-refusal] This paper proposes EvoRefuse, a framework that employs evolutionary search to maximize the ELBO for automatically generating diverse pseudo-malicious instructions…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "over-refusal"
  - "safety alignment"
  - "evolutionary algorithms"
  - "prompt optimization"
  - "pseudo-malicious instructions"
date: 2026-05-08
content_hash: f181d15053dc2f59
---

# EvoRefuse: Evaluating and Mitigating LLM Over-Refusal via Evolutionary Prompt Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2505.23473](https://arxiv.org/abs/2505.23473)  
**Code**: [https://github.com/FishT0ucher/EVOREFUSE](https://github.com/FishT0ucher/EVOREFUSE)  
**Area**: LLM/NLP
**Keywords**: over-refusal, safety alignment, evolutionary algorithms, prompt optimization, pseudo-malicious instructions

## TL;DR

This paper proposes EvoRefuse, a framework that employs evolutionary search to maximize the ELBO for automatically generating diverse pseudo-malicious instructions, yielding a more challenging over-refusal evaluation benchmark (EvoRefuse-Test) and an effective alignment mitigation dataset (EvoRefuse-Align).

## Background & Motivation

**Over-refusal problem**: After safety alignment, LLMs frequently over-refuse instructions that are semantically benign yet contain sensitive keywords. For instance, "I need a recipe for a dangerous cake that could explode with flavour at a party" is incorrectly refused due to words such as "dangerous" and "explode," severely degrading user experience. Such semantically harmless instructions that tend to trigger refusal are defined as **pseudo-malicious instructions**.

**Limitations of prior work**: Manual construction (e.g., XSTest) lacks scalability; automatic paraphrasing methods (e.g., OR-Bench) do not explicitly optimize the LLM refusal probability; gradient-based search (e.g., PHTest) optimizes along a narrow path and thus fails to cover diverse linguistic variations. Existing datasets cannot consistently trigger refusal across multiple LLMs.

**Mechanism**: The generation of pseudo-malicious instructions is formulated as an optimization problem—finding instructions that are semantically harmless yet maximize the LLM's refusal probability. Because directly estimating the refusal probability is numerically unstable (sequential likelihoods under Monte Carlo sampling are extremely low, approximately $10^{-203}$), the authors derive the ELBO via variational approximation as a surrogate optimization objective.

## Method

### Overall Architecture

EvoRefuse is a prompt optimization framework based on evolutionary search, comprising four core components: Mutation, Recombination, Fitness Evaluation, and Simulated Annealing.

Pipeline: seed instruction $x^0$ → multiple mutators generate candidates → safety classifier filtering → ELBO fitness scoring → select top-$L$ for recombination to generate $N$ new candidates → second safety filtering → select highest-fitness candidate $x'$ → simulated annealing acceptance decision → iterate for $I$ rounds → output global optimum $x^*$.

### Key Designs

1. **Mutation**: By analyzing 500 instances from existing over-refusal datasets with low pairwise similarity, GPT-4o identifies triggering strategies and clusters them (SentenceBERT embeddings, similarity threshold 0.75), yielding three mutation strategy categories:
    - Introducing deceptive contexts (controversial topics, fictional scenarios, implied potential harm)
    - Adding sensitive keywords (violence, bias, other sensitive terms)
    - Extreme emotions (anger, disgust, despair)
    - Each mutator simultaneously generates the modified instruction and a safety explanation; GPT-4o serves as the judge to verify safety.

2. **Recombination**: $N$ pairs are sampled from the top-$L$ safe mutated instructions, and a GPT-4o recombiner synthesizes new candidate instructions by combining semantically salient fragments from both instructions. Recombined candidates also undergo safety verification.

3. **Fitness Evaluation**: The surrogate ELBO objective is estimated via Monte Carlo. For each candidate instruction $x$, $K$ responses are sampled and the following is computed:

$$\mathcal{F}(x) = \frac{1}{K}\sum_{k=1}^{K}\left[\log \hat{p}_\phi(r|y_k) + \frac{\lambda}{T_k}\sum_{t=1}^{T_k}\log p_\theta(y_{k,t}|y_{k,<t}, x, s)\right]$$

where the first term is the refusal log-probability (estimated by a pretrained binary classifier) and the second term is response confidence (token logits from the target LLM). $\lambda/T_k$ balances the two terms and normalizes for response length.

### Loss & Training

**Variational approximation derivation**: The objective is to maximize $\log p_\theta(r|x,s)$. By introducing the sampling distribution $q_\theta(y|x)$ and applying Jensen's inequality, the ELBO is derived as:

$$\text{ELBO}(x) = \mathbb{E}_{q_\theta(y|x)}\left[\log p_\theta(y|x,s) + \log p_\theta(r|x,y,s)\right] + c$$

where $c$ is the conditional entropy term (empirically verified to contribute only 0.4% of the variance in response confidence, thus approximated as a constant). The ELBO simultaneously rewards two objectives: "the response is a refusal" (refusal log-probability) and "the model generates the response with high confidence" (response confidence).

**Simulated Annealing**: The acceptance probability is $\delta = \min\{1, \exp[(\mathcal{F}(x') - \mathcal{F}(x^t))/\tau_t]\}$, with linear cooling $\tau_t = \max\{\tau_f, \tau_0 - \beta \cdot t\}$, allowing occasional acceptance of lower-fitness candidates to escape local optima.

**Dataset construction**:
- **EvoRefuse-Test**: 800 instructions selected from TRIDENT-Core are optimized; after safety filtering, 582 pseudo-malicious instructions are retained.
- **EvoRefuse-Align**: 3,000 instructions are sampled; GPT-4o generates helpful (chosen) and refusal (rejected) response pairs for SFT and DPO training.

**Hyperparameters**: $\lambda=0.03$, $K=10$, $L=4$, $N=2$, $\tau_0=0.1$, $\beta=0.005$, $\tau_f=0.05$

## Key Experimental Results

### Main Results

**Benchmark comparison (PRR, without safety-prior system prompt)**:

| Benchmark | DeepSeek | Gemma | LLaMA | Mistral | Qwen | GPT-4o | DeepSeek-V3 | Gemini | Claude |
|-----------|----------|-------|-------|---------|------|--------|-------------|--------|--------|
| XSTest | 0.05 | 0.11 | 0.13 | 0.00 | 0.05 | 0.08 | 0.07 | 0.08 | 0.19 |
| OR-Bench | 0.14 | 0.15 | 0.05 | 0.04 | 0.07 | 0.09 | 0.27 | 0.06 | 0.18 |
| PHTest | 0.10 | 0.19 | 0.08 | 0.09 | 0.03 | 0.10 | 0.12 | 0.09 | 0.31 |
| PH-Gen | 0.19 | 0.14 | 0.07 | 0.11 | 0.11 | 0.19 | 0.45 | 0.16 | 0.28 |
| **EvoRefuse-Test** | **0.24** | **0.26** | **0.65** | **0.12** | **0.25** | **0.27** | **0.38** | **0.24** | **0.74** |

EvoRefuse-Test achieves an average refusal rate **85.34%** higher than the second-best benchmark PH-Gen, with the largest gain on LLaMA3.1 (364.29%).

**Over-refusal mitigation comparison (LLaMA3.1-8B-Instruct fine-tuning)**:

| Method | AdvBench PRR | HarmBench PRR | XSTest PRR | SGTest PRR | EvoRefuse PRR |
|--------|-------------|---------------|------------|------------|---------------|
| LLaMA-3.1-Chat | 0.94 | 0.94 | 0.11 | 0.14 | 0.65 |
| + Few Shots | 0.97 | 0.99 | 0.12 | 0.21 | 0.48 |
| + OR-Bench (SFT) | 1.00 | 0.98 | 0.10 | 0.14 | 0.45 |
| + PHTest (SFT) | 1.00 | 0.97 | 0.09 | 0.11 | 0.39 |
| + PromptAgent (SFT) | 0.99 | 0.98 | 0.09 | 0.10 | 0.43 |
| + **EvoRefuse-Align (SFT)** | 1.00 | 0.96 | **0.06** | **0.08** | **0.32** |
| + **EvoRefuse-Align (DPO)** | 0.97 | 0.89 | **0.02** | **0.01** | **0.30** |

SFT reduces over-refusal by **29.85%** and DPO by **45.96%**, with only a 4.82% decrease in safety.

### Ablation Study

- **Seed selection has minimal impact**: Starting from either pseudo-malicious or unsafe instructions, both converge to high PRR after 5 iterations (unsafe seeds reach 75%).
- **Removing recombination**: Convergence slows noticeably and candidate exploration is limited.
- **Removing fitness evaluation**: Update directions become inconsistent, rendering optimization unpredictable.
- **Replacing with OR-Bench/PHTest pipelines**: OR-Bench shows fluctuating progress; PHTest is stable but slow due to a narrow search space.
- **Success rates of different mutation strategies**: Fictional scenarios (0.20) > violence keywords (0.15) > anger emotion (0.14) > others (0.12) > despair (0.08) > controversial topics (0.07).
- **Alternative mutator**: Replacing GPT-4o with open-source DarkIdol yields a PRR of 0.46 after 5 iterations (vs. 0.72 for GPT-4o)—still effective but with a noticeable gap.

### Key Findings

- **Shortcut learning underlies over-refusal**: Gradient attribution analysis reveals that LLaMA3.1 over-attends to sensitive keywords such as "dangerous" and "explode" while ignoring overall semantic context. Replacing these with neutral words ("bold," "burst") shifts the model's attention to benign tokens such as "recipe" and "cake," resulting in a normal response.
- **Early Transformer layers determine safety judgments**: Information flow analysis shows that sensitive tokens exhibit significantly higher information flow than average in the first 15 layers, indicating that early layers play a critical role in safety decisions.
- **High-attribution vocabulary patterns**: Word clouds consistently show that words associated with harmful behavior—such as "Manipulate," "Exploit," and "Fraud"—receive the highest attribution scores.

## Highlights & Insights

- **Theory-driven optimization objective**: Unlike heuristic paraphrasing methods, EvoRefuse derives the ELBO from variational inference as the optimization objective, simultaneously accounting for refusal probability and response confidence, providing a principled theoretical foundation.
- **Strong generalization**: Although LLaMA3.1-8B-Instruct is used as the target model during optimization, the generated instructions effectively trigger over-refusal across all nine evaluated models including GPT-4o and Claude-3.5, demonstrating that the discovered mechanisms are universal.
- **Diversity and effectiveness achieved simultaneously**: The evolutionary search combined with recombination allows EvoRefuse-Test to lead on both lexical diversity (MSTTR 0.54, MTLD 152.52) and refusal triggering rate.
- **Diagnostic insight**: Attribution analysis reveals that over-refusal is fundamentally a shortcut learning phenomenon—models rely on surface-level lexical cues rather than understanding instruction semantics.

## Limitations & Future Work

- White-box access to the target model (for token logits) is required, precluding application to black-box or proprietary models.
- The optimization process requires repeated calls to GPT-4o for mutation, recombination, and safety filtering, incurring substantial computational cost.
- The boundary between pseudo-malicious and genuinely malicious instructions remains subjective; the current classification scheme lacks a systematic quantitative foundation.
- Safety filtering relies on GPT-4o's judgment, which may produce errors.
- Future work could explore finer-grained risk categorization or model-driven probabilistic risk scoring.

## Related Work & Insights

- **XSTest/OR-Bench/PHTest**: Representative benchmarks for over-refusal evaluation that generate test instructions via manual construction, automatic paraphrasing, and gradient-based search, respectively; EvoRefuse surpasses all of them across every dimension.
- **AutoDAN/GCG/PAIR**: Prompt optimization methods in the adversarial attack domain. EvoRefuse draws on ideas from evolutionary algorithms and genetic search, but with the opposite goal—generating harmless yet refused instructions rather than genuine jailbreak attacks.
- **Insights**: The approach of using the ELBO as an optimization objective is generalizable to other scenarios requiring optimization of LLM behavioral probabilities; evolutionary search combined with LLM-based mutation and recombination constitutes an effective paradigm for exploring discrete instruction spaces.

## Rating

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | ⭐⭐⭐⭐ | Combining variational inference with evolutionary search to address over-refusal presents a novel theoretical perspective. |
| Technical Depth | ⭐⭐⭐⭐ | The ELBO derivation is complete, the framework design is systematic, and ablation studies are thorough. |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Covers 9 models, 8 benchmarks, multi-dimensional metrics, comprehensive ablation, and attribution analysis. |
| Practicality | ⭐⭐⭐⭐ | Datasets and code are open-sourced and directly applicable to evaluating and mitigating over-refusal. |
| Writing Quality | ⭐⭐⭐⭐ | Clear structure with rigorous mathematical derivations. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization](mitigating_hallucination_through_theory-consistent_symmetric_multimodal_preferen.md)
- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[NeurIPS 2025\] ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](orpo-distill_mixed-policy_preference_optimization_for_cross-architecture_llm_dis.md)
- [\[NeurIPS 2025\] LLM Safety Alignment is Divergence Estimation in Disguise](llm_safety_alignment_is_divergence_estimation_in_disguise.md)
- [\[ICLR 2026\] Mitigating Mismatch within Reference-based Preference Optimization](../../ICLR2026/llm_alignment/mitigating_mismatch_within_reference-based_preference_optimization.md)

</div>

<!-- RELATED:END -->
