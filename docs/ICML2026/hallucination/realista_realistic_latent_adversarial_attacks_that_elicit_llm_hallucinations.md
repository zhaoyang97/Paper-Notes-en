---
title: >-
  [Paper Note] REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations
description: >-
  [ICML 2026][Hallucination Detection][Hallucination induction] REALISTA constructs an "input-dependent dictionary of editing directions" in the LLM latent space to transform adversarial prompt optimization into a continuo…
tags:
  - "ICML 2026"
  - "Hallucination Detection"
  - "Hallucination induction"
  - "latent space attack"
  - "semantic preservation"
  - "simplex constraint"
  - "concept editing"
date: 2026-05-08
content_hash: 20a59bc4b1a03671
---

# REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations

**Conference**: ICML 2026  
**arXiv**: [2605.12813](https://arxiv.org/abs/2605.12813)  
**Code**: https://github.com/Buyun-Liang/REALISTA  
**Area**: Hallucination Detection  
**Keywords**: Hallucination induction, latent space attack, semantic preservation, simplex constraint, concept editing

## TL;DR
REALISTA constructs an "input-dependent dictionary of editing directions" in the LLM latent space to transform adversarial prompt optimization into a continuous problem under a simplex constraint. This approach maintains the semantic equivalence and coherence of discrete methods like SECA while achieving the search flexibility of continuous methods like LARGO. It represents the first successful induction of hallucinations in the free-form outputs of closed-source reasoning models such as GPT-5.

## Background & Motivation

**Background**: LLMs hallucinate even on benign user queries. A typical example provided in the paper: the model correctly calculates "Simplify $(2+5)^2-42$" as 7, but mistakenly answers 16 for the synonymous "Compute the result after squaring the sum of 2 and 5, then subtracting 42". To systematically expose such failures, "realistic adversarial attacks" are needed—attacks that induce hallucinations while appearing as natural prompts written by real users.

**Limitations of Prior Work**: Existing attacks are divided into two categories, both with deficiencies. One category consists of discrete prompt attacks (e.g., SECA), which rely on LLMs to rephrase candidate prompts and pick the worst-performing one. These strictly guarantee semantic equivalence and coherence, but their search space is limited by the finite candidates sampled by the rephrasing model, resulting in low diversity. The other category consists of continuous latent attacks (e.g., LARGO, Sheshadri), which directly add arbitrary perturbations to LLM hidden states. While the search space is large, the semantic meaning often deviates significantly after decoding back to prompts—LARGO's SEE (Semantic Equivalence Error) is nearly 100%. Other methods modify numbers in fictional stories ("$-42$" changed to "$-33$") to induce 16, but this is not a hallucination on the original prompt; it is simply a change to the problem itself.

**Key Challenge**: Search flexibility vs. Semantic authenticity. Discrete methods preserve authenticity but have restricted search; continuous methods have strong search capabilities but often produce "false successes" that are no longer the same problem.

**Goal**: To search within the LLM latent space but restrict the search to a "semantically equivalent" subspace—specifically, a linear combination of a set of "interpretable editing directions," where each direction corresponds to a semantically equivalent rephrasing of the original prompt.

**Key Insight**: Work by Park, Zou, et al. found that high-level semantic concepts in LLM latent space are approximately linearly additive. If an "input-dependent dictionary" can be prepared for each original prompt $x_0$, where each column is a direction $z^{(i)} = \phi(x_{\text{SE}}^{(i)}) - \phi(x_0)$ corresponding to a synonymous rewrite, then continuous optimization of the editing coefficients $\delta$ is both flexible and safe.

**Core Idea**: Formulate the hallucination induction problem using an "editing direction dictionary + scaled simplex constraint + LLM as encoder/decoder" as:
$$\min_\delta \mathcal{L}_{\text{hall}}(f_T(\psi(z_0 + D\delta)), y^*) \quad \text{s.t.} \quad \delta \in \Delta_\varepsilon$$

## Method

### Overall Architecture

Input: Original prompt $x_0$, target LLM $f_T$, and target erroneous response $y^*$.
Two-stage process: (1) Dictionary Construction—Generate $n$ semantically equivalent rephrasings $x_{\text{SE}}^{(i)}$ for $x_0$ via WordNet and concept optimization, then encode them into the latent space to obtain directions $z^{(i)} = \phi(x_{\text{SE}}^{(i)}) - \phi(x_0)$, forming $D^{(z_0)} \in \mathbb{R}^{L \times d \times n}$; (2) Attack Optimization—Optimize $\delta$ over a scaled simplex $\Delta_\varepsilon = \{\delta \succeq 0 : \|\delta\|_1 \leq \varepsilon\}$ using Projected Langevin Dynamics (noisy PGD). At each step, decode $z_0 + D\delta$ back to prompt $x$ using an LLM-based decoder $\psi$, compute the loss via the target model, and accept only after an LLM judge verifies semantic equivalence.

### Key Designs

1.  **Input-dependent Edit Dictionary**:
    *   **Function**: Constructs a compact, diverse, and effective concept basis $\{c^{(1)}, \dots, c^{(n)}\}$ for each original prompt $x_0$, where each $c^{(i)} = \phi(x_{\text{SE}}^{(i)})$ corresponds to the latent position of a semantically equivalent rewrite.
    *   **Mechanism**: Extracts concept words related to $x_0$ from WordNet and performs constrained concept optimization (Eq. 12 in the appendix) to ensure orthogonal directions, appropriate distance from $x_0$, and that decoded rephrasings are valid. Finally, $z^{(i)} = c^{(i)} - z_0$ are concatenated as columns into the dictionary $D^{(z_0)}$.
    *   **Design Motivation**: This is the core difference between REALISTA and LARGO. LARGO perturbs in arbitrary latent directions without guaranteeing the results fall within the "valid prompt" region. REALISTA constrains the search domain to a "subspace spanned by linear interpolations of semantically equivalent rewrites," naturally staying on the valid prompt manifold.

2.  **Scaled Simplex Constraint + Single-Concept Initialization**:
    *   **Function**: Restricts $\delta$ within a non-negative and $\ell_1$-bounded "scaled simplex" $\Delta_\varepsilon$, ensuring only a few concept directions are activated per attack to avoid semantic collapse.
    *   **Mechanism**: The $\ell_1$ norm acts as both an entry bound and a sparsity proxy. Non-negativity is used because negative directions often lack clear semantic interpretation and tend to produce gibberish. During initialization, $\delta = \varepsilon \cdot e_i$ is evaluated for each concept $i$, and the top-$N$ indices by loss are used as starting points for optimization.
    *   **Design Motivation**: The simplex geometricizes the "semantic equivalence" hard constraint—empirically, smaller $\delta$ and fewer concepts lead to decoded prompts closer to the original semantics. Single-concept initialization addresses the non-convexity of $\mathcal{L}_\mathcal{T}$, as starting from zero often leads to local optima. Trying $n$ seeds with "only one concept active" improves the success rate. Table 3 shows an average of 1-2 concepts activated for open-source LLMs and $<1$ for closed-source reasoning models.

3.  **Projected Langevin Dynamics + Semantic Equivalence Safeguard**:
    *   **Function**: Explores the piece-wise flat optimization landscape using noisy PGD and performs real-time checks to ensure decoded prompts remain semantically equivalent to $x_0$.
    *   **Mechanism**: Update rule: $\delta_{k+1} \leftarrow \text{Proj}_{\Delta_\varepsilon}[\delta_k - \eta \tilde{\nabla}_\delta \mathcal{L}_\mathcal{T} + \sqrt{2\eta T} \xi_k]$, where $T = T_0 \cdot \gamma^k$ is the annealing temperature and $\xi_k \sim \mathcal{N}(0, I)$. Gradients are obtained via Gumbel-Softmax reparameterization. If the decoded prompt $x$ is judged "not equivalent to $x_0$", the gradient signal for that step is zeroed out.
    *   **Design Motivation**: The landscape is piece-wise flat because multiple adjacent $z$ values may decode to the same prompt $x$. Vanilla GD is difficult to tune; Langevin noise allows the algorithm to jump between regions. The safeguard is the final gate for "REALISTic" attacks—hard-guaranteeing that outputs pass semantic equivalence checks.

### Loss & Training

Two attack objectives: (i) In open-ended MCQA, $\mathcal{L}_\mathcal{T}(\cdot) = -\log P_T(y^* | \cdot)$, where $y^*$ is the token for the incorrect answer option; (ii) In free-form response, $\mathcal{L}_\mathcal{T}(\cdot) = -J(R_T(\cdot))$, where $J$ is an LLM-based hallucination evaluator. The latter is used for closed-source reasoning models like GPT-5 where logits are unavailable; gradients are transferred from open-source surrogate models.

## Key Experimental Results

### Main Results

Dataset: MMLU subset (347 questions, 16 subjects). ASR@30 indicates success rate in 30 independent trials.

| Target LLM | Raw | SECA | LARGO | ICD | **REALISTA** |
|------------|-----|------|-------|-----|--------------|
| Llama-3-3B ASR↑ | 45.48 | 79.61 | 84.71 | 90.77 | **97.11** |
| Llama-3-3B SEE↓ | 0.00 | 0.87 | 97.42 | 100.00 | 0.86 |
| Llama-3-8B ASR↑ | 54.40 | 82.97 | 57.92 | 87.32 | **93.60** |
| Llama-3-8B SEE↓ | 0.00 | 2.59 | 96.45 | 100.00 | 3.48 |
| Qwen-2.5-7B ASR↑ | 6.40 | 32.47 | 23.89 | 11.50 | **41.61** |

While LARGO/ICD show high ASR, their SEE is near 100% (meaning they change the semantics entirely). REALISTA is the only method achieving high ASR with low SEE, outperforming SECA on Llama-3 by 10-20%.

| Reasoning LLM (free-form) | Raw | ICD | **REALISTA** | SECA/LARGO |
|---------------------------|-----|-----|--------------|------------|
| GPT-5-Nano ASR↑ | 4.02 | 6.32 | **23.61** | N/A |
| GPT-5-Mini ASR↑ | 2.01 | 2.57 | **20.72** | N/A |
| GPT-5-Mini SEE↓ | 1.58 | 100.00 | 0.72 | – |

SECA and LARGO are incompatible with GPT-5 (one requires token-level logits, the other requires target latents). REALISTA successfully transfers gradients from open-source surrogates.

### Ablation Study

| Configuration | Key Findings |
|------|----------|
| Top-20 active concepts | "Polarity reversal" (counterfactual/inverted/opposite) is most frequent; "Logical structure modification" (conditional/disjunctive) follows; "Instructional framing" (imperative/elaborate) is also common. |
| Active concepts per attack | 1-2 on open-source LLMs, <1 on closed-source reasoning models (many attacks succeed even with 0 concepts/original prompt). |
| Human Evaluation (100 samples) | REALISTA's SEE ≈ 5% according to human annotators, consistent with the LLM judge (5.27%); LARGO is near 100%. |

### Key Findings
*   Successful "realistic attacks" primarily work by changing framing and logical structures rather than modifying factual content. This suggests that LLM defense should focus on robustness against these structural transformations.
*   Attack patterns: Polarity reversal is most common as it preserves entities and correctness criteria while subtly inverting the framing. Logical modifications create ambiguity by expanding the reasoning space.
*   Gradient transfer to GPT-5 is effective, implying significant overlap in the score landscapes of open-source surrogates and closed-source reasoning models.
*   Convergence: Approx. 100 steps for open-source LLMs; more steps required for closed-source models due to larger free-form output spaces.

## Highlights & Insights
*   **Geometric embedding of semantic constraints**: Unlike previous latent attacks that use semantic checks as post-hoc filters, REALISTA makes "semantic equivalence" a prior by design through dictionary parameterization. This approach can be extended to jailbreaking or controllable generation.
*   **Input-dependent dictionary** is superior to input-agnostic universal directions (like those in representation engineering for "happiness" or "honesty") for attacks, as the latent implementation of an abstract direction (e.g., "reversal") varies significantly across prompts.
*   **PLD for piece-wise flat landscapes**: Using Gumbel-Softmax with annealed Gaussian noise addresses the vanishing gradient problem caused by the discrete nature of decoders.
*   The effectiveness of **gradient transfer** to closed-source models suggests that red-teaming tools do not necessarily need internal access to a target model, which has major implications for security audits.

## Limitations & Future Work
*   Dictionary construction depends on WordNet and LLM collaboration; coverage for non-English or specialized domains is unverified.
*   ASR@30 is "best-of-30," meaning 30 trials are needed for the reported success rate.
*   On Qwen-2.5-14B, REALISTA is slightly weaker than SECA (27.24 vs 27.51), indicating some robustness in larger models against paraphrase-based attacks.
*   The linear combination assumption is strong; future work should explore non-convex constraints (similar to perceptual constraints in vision) to represent complex semantic transformations.
*   SEE is not 0% but 1-3%, showing that the LLM judge is imperfect and some "successful" prompts may not be strictly equivalent to human eyes.

## Related Work & Insights
*   **vs. SECA (Liang 2025b)**: Both enforce semantic equivalence and coherence, but SECA is limited to discrete candidates. REALISTA explores a continuous simplex, yielding 10-20% higher ASR and the ability to attack reasoning models.
*   **vs. LARGO (Li 2025a)**: Both use continuous optimization in latent space. LARGO lacks semantic constraints (leading to ~100% SEE), whereas REALISTA’s dictionary provides the necessary guardrails.
*   **vs. ICD (Zhang 2024)**: ICD uses template-based prompts to force hallucinations (modifying the problem); REALISTA tests inconsistency on equivalent prompts.
*   **vs. Representation Engineering (Zou 2025)**: Both utilize linear combinations of latent concepts. While RE is used for safety steering, REALISTA adapts the same toolbox for attacks.

## Rating
*   Novelty: ⭐⭐⭐⭐ The "geometric subspace for semantic constraints" is clean, though individual components (latent linear concepts, PLD, Gumbel) are known.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes 4 open-source and 2 closed-source LLMs, 5 baselines, ASR@K, human evaluation, and visualization.
*   Writing Quality: ⭐⭐⭐⭐⭐ Formulas and diagrams are clear; the motivating example is consistent throughout.
*   Value: ⭐⭐⭐⭐ Provides a truly realistic red-teaming tool that transfers to closed-source models, holding significant value for LLM safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations](../../NeurIPS2025/hallucination/seca_semantically_equivalent_and_coherent_attacks_for_eliciting_llm_hallucinatio.md)
- [\[ICML 2026\] Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models](revis_sparse_latent_steering_to_mitigate_object_hallucination_in_large_vision-la.md)
- [\[ICML 2026\] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)](when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)
- [\[ACL 2026\] Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations](../../ACL2026/hallucination/two_pathways_to_truthfulness_on_the_intrinsic_encoding_of_llm_hallucinations.md)
- [\[ACL 2026\] Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate](../../ACL2026/hallucination/dialectic-med_mitigating_diagnostic_hallucinations_via_counterfactual_adversaria.md)

</div>

<!-- RELATED:END -->
