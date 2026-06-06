---
title: >-
  [Paper Note] REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations
description: >-
  [ICML 2026][LLM Safety][Hallucination Induction] REALISTA constructs an "input-dependent edit direction dictionary" in the LLM latent space…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Hallucination Induction"
  - "latent space attack"
  - "semantic preservation"
  - "simplex constraint"
  - "concept editing"
date: 2026-05-08
content_hash: b9b6a1a42d8f1190
---

# REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations

**Conference**: ICML 2026  
**arXiv**: [2605.12813](https://arxiv.org/abs/2605.12813)  
**Code**: https://github.com/Buyun-Liang/REALISTA  
**Area**: LLM Security / Adversarial Attacks / Hallucination Induction  
**Keywords**: Hallucination Induction, latent space attack, semantic preservation, simplex constraint, concept editing

## TL;DR
REALISTA constructs an "input-dependent edit direction dictionary" in the LLM latent space, turning adversarial prompt optimization into a continuous problem under a simplex constraint. This approach preserves the semantic equivalence/coherence of discrete methods like SECA, while achieving the search flexibility of continuous methods like LARGO. It is the first to successfully induce hallucinations in free-form outputs of closed-source inference models such as GPT-5.

## Background & Motivation

**Background**: LLMs can hallucinate even on benign user queries. A typical example from the paper: the model correctly computes "Simplify $(2+5)^2-42$" as 7, but for the semantically equivalent "Compute the result after squaring the sum of 2 and 5, then subtracting 42," it answers 16. To systematically expose such failures, "realistic adversarial attacks" are needed—ones that can induce hallucinations while appearing as prompts written by real users.

**Limitations of Prior Work**: Existing attacks fall into two categories, each lacking something. One is discrete prompt attacks (e.g., SECA), which rely on LLM rephrasing to generate candidate prompts and select the worst-performing one. These strictly guarantee semantic equivalence and coherence, but the search space is limited by the finite candidates sampled by the rephrase model, resulting in low diversity. The other is continuous latent attacks (e.g., LARGO, Sheshadri), which directly perturb the LLM hidden state, offering a large search space but often drifting semantically when decoding back to prompts—LARGO's measured SEE (semantic equivalence error rate) approaches 100%. Some methods simply change numbers in the prompt (e.g., changing "$-42$" to "$-33$" to induce 16), which is not a hallucination on the original prompt but a change of the question.

**Key Challenge**: Search flexibility vs. semantic authenticity. Discrete methods preserve authenticity but are search-limited; continuous methods have strong search power but often result in "hallucinations that are not about the same question," i.e., false positives.

**Goal**: To search in the LLM latent space, but restrict the search to the "semantic equivalence" subspace—specifically, to the linear span of a set of "interpretable editing directions," each corresponding to a semantically equivalent rephrase of the original prompt.

**Key Insight**: Prior work (Park, Zou, etc.) found that high-level semantic concepts in LLM latent space are approximately linearly additive. If, for each original prompt $x_0$, an "input-dependent dictionary" is prepared, with each column representing the direction of a certain synonymous rephrase $z^{(i)} = \phi(x_{\text{SE}}^{(i)}) - \phi(x_0)$, then continuous optimization over the edit coefficients $\delta$ can be both flexible and safe.

**Core Idea**: By using an "edit direction dictionary + scaled simplex constraint + LLM as encoder/decoder," the hallucination induction problem is formulated as $\min_\delta \mathcal{L}_{\text{hall}}(f_T(\psi(z_0 + D\delta)), y^*)$ s.t. $\delta \in \Delta_\varepsilon$.

## Method

### Overall Architecture

Input: original prompt $x_0$, target LLM $f_T$, target incorrect response $y^*$.
Two-stage process: (1) Dictionary construction—generate $n$ semantically equivalent rephrases $x_{\text{SE}}^{(i)}$ for $x_0$ using WordNet + concept optimization, encode to latent to obtain directions $z^{(i)} = \phi(x_{\text{SE}}^{(i)}) - \phi(x_0)$, forming $D^{(z_0)} \in \mathbb{R}^{L \times d \times n}$; (2) Attack optimization—optimize $\delta$ over the scaled simplex $\Delta_\varepsilon = \{\delta \succeq 0 : \|\delta\|_1 \leq \varepsilon\}$ using Projected Langevin Dynamics (noisy PGD). At each step, $z_0 + D\delta$ is decoded back to a prompt $x$ via the LLM-based decoder $\psi$, the target model computes the loss, and only solutions passing an LLM judge semantic equivalence check are accepted.

### Key Designs

1. **Input-dependent Edit Dictionary**:

    - **Function**: For each original prompt $x_0$, construct a compact, diverse, and effective conceptual basis $\{c^{(1)}, \dots, c^{(n)}\}$, where each $c^{(i)} = \phi(x_{\text{SE}}^{(i)})$ corresponds to the latent position of a semantically equivalent rephrase.
    - **Mechanism**: Extract concept words related to $x_0$ from WordNet, run constrained concept optimization (see Appendix Eq. 12) to ensure directions are orthogonal, at moderate distance from $x_0$, and that decoded rephrases are valid. Finally, $z^{(i)} = c^{(i)} - z_0$ are concatenated as columns to form the dictionary $D^{(z_0)}$.
    - **Design Motivation**: This is the core distinction from LARGO—LARGO perturbs in arbitrary latent directions without ensuring the result remains in the "valid prompt" region; REALISTA hard-constrains the search domain to the subspace spanned by linear interpolations of semantically equivalent rephrases, thus geometrically staying on the valid prompt manifold.

2. **Scaled Simplex Constraint + Single-Concept Initialization**:

    - **Function**: Restrict $\delta$ to the non-negative, $\ell_1$-bounded "scaled simplex" $\Delta_\varepsilon$, so each attack activates only a few concept directions, avoiding semantic collapse from activating all at once.
    - **Mechanism**: The $\ell_1$ norm both bounds entry magnitude and promotes sparsity; non-negativity is used because negative directions lack clear semantic interpretation and tend to produce gibberish. For initialization, each concept $i$ is individually tested with $\delta = \varepsilon \cdot e_i$, decoded, and the top-$N$ by loss are used as seeds for further optimization.
    - **Design Motivation**: The simplex geometrically hard-constrains "semantic equivalence"—empirically, smaller $\delta$ and fewer active concepts yield decoded prompts closer in meaning to the original. Single-concept initialization addresses the non-convexity of $\mathcal{L}_\mathcal{T}$; starting from zero often gets stuck in local minima, but trying all $n$ single-concept seeds greatly increases the chance of finding good solutions. Table 3 shows that on open-source LLMs, 1-2 concepts are activated on average, and <1 on closed-source inference models.

3. **Projected Langevin Dynamics + Semantic Equivalence Safeguard**:

    - **Function**: Use noisy PGD to explore the piece-wise flat optimization landscape, and check in real time whether the decoded prompt remains semantically equivalent to $x_0$, accepting only those that pass.
    - **Mechanism**: Update rule $\delta_{k+1} \leftarrow \text{Proj}_{\Delta_\varepsilon}[\delta_k - \eta \tilde{\nabla}_\delta \mathcal{L}_\mathcal{T} + \sqrt{2\eta T} \xi_k]$, where $T = T_0 \cdot \gamma^k$ is the annealing temperature, $\xi_k \sim \mathcal{N}(0, I)$. Gradients are obtained via Gumbel-Softmax reparameterization (since decoding is discrete); if the decoded prompt $x$ is judged by the LLM as not semantically equivalent to $x_0$, the gradient signal for that step is zeroed to prevent further movement in non-equivalent directions.
    - **Design Motivation**: The landscape is piece-wise flat because multiple adjacent $z$ may decode to the same prompt $x$, making GD step size hard to tune (too small: no movement; too large: overshoot). Injecting annealed Gaussian noise allows the algorithm to jump between regions. The safeguard is the final barrier making REALISTA "REALISTic"—it strictly ensures the final adversarial prompt passes the semantic equivalence check.

### Loss & Training

Two attack objectives: (i) For open-ended MCQA, $\mathcal{L}_\mathcal{T}(\cdot) = -\log P_T(y^* | \cdot)$, where $y^*$ is the token for the incorrect answer; (ii) For free-form response, $\mathcal{L}_\mathcal{T}(\cdot) = -J(R_T(\cdot))$, where $J$ is an LLM-based hallucination evaluator. The latter is used to attack closed-source inference models like GPT-5, which do not provide logits or fixed-format outputs; gradients are borrowed from open-source surrogate models (gradient transfer).

## Key Experimental Results

### Main Results

Dataset: 347-question MMLU subset (16 subjects), ASR@30 = success rate over 30 independent trials.

| Target LLM | Raw | SECA | LARGO | ICD | **REALISTA** |
|------------|-----|------|-------|-----|--------------|
| Llama-3-3B ASR↑ | 45.48 | 79.61 | 84.71 | 90.77 | **97.11** |
| Llama-3-3B SEE↓ | 0.00 | 0.87 | 97.42 | 100.00 | 0.86 |
| Llama-3-8B ASR↑ | 54.40 | 82.97 | 57.92 | 87.32 | **93.60** |
| Llama-3-8B SEE↓ | 0.00 | 2.59 | 96.45 | 100.00 | 3.48 |
| Qwen-2.5-7B ASR↑ | 6.40 | 32.47 | 23.89 | 11.50 | **41.61** |

LARGO / ICD show high ASR, but SEE is nearly 100% (i.e., semantics are completely changed, not realistic attacks). REALISTA is the only method achieving both high ASR and low SEE, outperforming SECA by 10–20% on Llama-3.

| Reasoning LLM (free-form) | Raw | ICD | **REALISTA** | SECA/LARGO |
|---------------------------|-----|-----|--------------|------------|
| GPT-5-Nano ASR↑ | 4.02 | 6.32 | **23.61** | N/A |
| GPT-5-Mini ASR↑ | 2.01 | 2.57 | **20.72** | N/A |
| GPT-5-Mini SEE↓ | 1.58 | 100.00 | 0.72 | – |

SECA and LARGO cannot run on GPT-5 (one requires token-level logits and fixed format, the other needs target latent); REALISTA successfully transfers gradients from open-source surrogates.

### Ablation Study

| Configuration | Key Findings |
|---------------|-------------|
| Top-20 active concepts | "Polarity reversal" types (counterfactual/inverted/opposite) are most frequently activated; "logical structure modification" (conditional/disjunctive) are next; "imperative/elaborate" (instructional framing) are also common |
| Active concepts per attack | 1–2 on open-source LLMs, <1 on closed-source reasoning models (many attacks succeed by keeping the original prompt, i.e., using 0 concepts) |
| Human evaluation (100 samples) | REALISTA's SEE is ≈5% by two human annotators, consistent with LLM judge (5.27%); LARGO is near 100%, SECA is 5–11% |

### Key Findings
- Successful "realistic attacks" mainly work by changing framing and logical structure, not factual content—this suggests that LLM defenses must be robust to paraphrase attacks involving such structural changes.
- Attack success patterns: polarity reversal is most common, as it preserves entities and correctness criteria while subtly reversing framing; logical structure modification (adding conditionals/disjunctives) creates ambiguity by expanding the reasoning space.
- Gradient transfer is surprisingly effective on GPT-5, indicating substantial overlap in the score landscape between open-source surrogates and closed-source reasoning models.
- Convergence: about 100 steps on open-source LLMs; more steps needed for closed-source reasoning models due to larger free-form output space.

## Highlights & Insights
- **"Geometrically hard-coding semantic equivalence constraints into the search space"**—previous latent attacks treated semantic checks as post-hoc filters (generate then filter), but REALISTA's dictionary parameterization makes "any feasible $\delta$ corresponds to a semantically equivalent prompt" a prior, greatly improving efficiency. This idea can be transferred to jailbreak, controllable generation, and any problem requiring latent exploration under constraints.
- **Input-dependent dictionary** is more suitable for attack scenarios than input-agnostic universal directions (e.g., "happy/honesty" in representation engineering), since the latent realization of the same abstract direction (e.g., "reversal") varies greatly across prompts.
- **PLD for piece-wise flat landscapes** is an engineering highlight—the discreteness of the decoder often yields zero gradients; pure PGD stagnates, but annealed Gaussian noise enables escape.
- Transferring gradients from open-source surrogates to closed-source inference models demonstrates that future red-teaming tools need not "access the target internals," which is significant for deployed model security evaluation.

## Limitations & Future Work
- Dictionary construction relies on WordNet + LLM collaboration; coverage for non-English or domain-specific prompts is untested.
- ASR@30 is best-of-30, meaning real-world deployment would require 30 runs to achieve this success rate; single-run success rate is lower.
- On Qwen-2.5-14B, REALISTA is slightly weaker than SECA (27.24 vs 27.51), indicating that larger models have some robustness to paraphrase-based attacks.
- The simplex + linear combination assumption is strong; the authors note that future work should explore richer non-convex constraints (similar to perceptual constraints in vision) to express more complex semantic transformations.
- Successful attacks have SEE of 1–3%, not zero, indicating that the LLM judge is imperfect; some prompts passing the check may still not be fully equivalent to humans.
- Evaluation metrics SEE/SCE rely on LLM judges, introducing self-bias risk.

## Related Work & Insights
- **vs SECA (Liang 2025b)**: Both enforce semantic equivalence and coherence, but SECA can only select from discrete LLM rephrase candidates, limiting search; REALISTA operates on a continuous simplex, achieving 10–20% higher ASR and can attack reasoning models.
- **vs LARGO (Li 2025a)**: Both perform continuous optimization in latent space and invert to prompt space, but LARGO lacks semantic equivalence constraints, so SEE approaches 100%; REALISTA's dictionary is the key difference.
- **vs ICD (Zhang 2024)**: ICD uses template-based attacks to directly prompt the model to "hallucinate," essentially changing the question; REALISTA truly tests model inconsistency on equivalent prompts.
- **vs Zou 2025 and representation engineering**: Both use latent concept linear combinations, but RE is mainly for alignment/safety steering, while REALISTA uses it for attack—showing the dual-use nature of these tools.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of "turning semantic constraints into a geometric subspace" is very clean, though individual components (latent linear concept, PLD, Gumbel reparam) are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four open-source and two closed-source LLMs, five baselines, ASR@K for multiple K, human evaluation, convergence analysis, and concept visualization are all included.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear correspondence between formulas and figures, motivating example (2+5)² is consistent throughout, Algorithm 1 + Figure 2 clearly explain the process.
- Value: ⭐⭐⭐⭐ Provides a truly realistic attack tool for red-teaming, and can transfer to closed-source models, which is significant for LLM safety evaluation.

## Related Papers

- [\[NeurIPS 2025\] SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations](../../NeurIPS2025/llm_safety/seca_semantically_equivalent_and_coherent_attacks_for_eliciting_llm_hallucinatio.md)
- [\[ICML 2026\] Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection](tracing_the_dynamics_of_refusal_exploiting_latent_refusal_trajectories_for_robus.md)
- [\[AAAI 2026\] GraphTextack: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs](../../AAAI2026/llm_safety/graphtextack_a_realistic_black-box_node_injection_attack_on_llm-enhanced_gnns.md)
- [\[ICML 2025\] X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP](../../ICML2025/llm_safety/x-transfer_attacks_towards_super_transferable_adversarial_attacks_on_clip.md)
- [\[ACL 2026\] Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations](../../ACL2026/llm_safety/two_pathways_to_truthfulness_on_the_intrinsic_encoding_of_llm_hallucinations.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations](../../NeurIPS2025/llm_safety/seca_semantically_equivalent_and_coherent_attacks_for_eliciting_llm_hallucinatio.md)
- [\[ICML 2026\] Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection](tracing_the_dynamics_of_refusal_exploiting_latent_refusal_trajectories_for_robus.md)
- [\[ICLR 2026\] Efficient Adversarial Attacks on High-dimensional Offline Bandits](../../ICLR2026/llm_safety/efficient_adversarial_attacks_on_high-dimensional_offline_bandits.md)
- [\[ICML 2026\] From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity](from_flat_facts_to_sharp_hallucinations_detecting_stubborn_errors_via_gradient_s.md)
- [\[ACL 2026\] Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations](../../ACL2026/llm_safety/two_pathways_to_truthfulness_on_the_intrinsic_encoding_of_llm_hallucinations.md)

</div>

<!-- RELATED:END -->
