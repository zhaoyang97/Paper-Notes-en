---
title: >-
  [Paper Note] Top-W: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for LLMs
description: >-
  [ICML 2026][LLM/NLP][truncation decoding] Top-W formulates next-token truncation as a minimization problem with three terms—Wasserstein (geometry-aware), entropy…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "truncation decoding"
  - "Wasserstein distance"
  - "token embedding geometry"
  - "entropy constraint"
  - "high-temperature robustness"
date: 2026-05-08
content_hash: 9a13c9b35f04feec
---

# Top-W: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for LLMs

**Conference**: ICML 2026  
**arXiv**: [2602.10346](https://arxiv.org/abs/2602.10346)  
**Code**: https://github.com/arashgholami/top-w-decoding (available)  
**Area**: LLM decoding / evaluation / inference-time control  
**Keywords**: truncation decoding, Wasserstein distance, token embedding geometry, entropy constraint, high-temperature robustness

## TL;DR
Top-W formulates next-token truncation as a minimization problem with three terms—Wasserstein (geometry-aware), entropy, and mass—explicitly considering token embedding geometry. Theoretically, the optimal solution is either a singleton token or a prefix sorted by $f(i)+\lambda\log p_i$. The engineering implementation is just an $O(n\log n)$ scan. On GSM8K, GPQA, AlpacaEval, and MT-Bench, Top-W outperforms in the majority of 15 (T, model) combinations, and at high temperatures, improves GSM8K by up to 33.7% over Top-H.

## Background & Motivation

**Background**: Truncated sampling for LLM decoding is foundational—Top-$k$, Top-$p$ (nucleus), Min-$p$, and locally typical sampling all prune the low-probability tail from a "probability ranking" perspective. Recently, Top-$H$ explicitly constrains the entropy of the truncated sub-distribution, representing one of the first works to approach decoding from a "distribution shaping" viewpoint.

**Limitations of Prior Work**: All these rules treat tokens as unstructured categories—only considering probabilities, ignoring semantic distances in embedding space. As a result: (i) At high temperatures ($T\geq 1.5$), Top-$p$/Min-$p$ often include nearly the entire vocabulary, leading to output collapse; (ii) Even with entropy control (Top-$H$), probability may concentrate on synonymous/near-synonymous tokens, resulting in "pseudo-diversity" and loss of true creativity.

**Key Challenge**: The decoder must balance (i) faithfulness (not deviating too far from the original distribution), (ii) creativity (sufficient diversity), and (iii) coherence (not pruning too much mass). The first two inherently require measurement in token geometric space, but all existing samplers ignore geometric information.

**Goal**: Explicitly incorporate token embedding geometry into the truncation objective, providing a sampler with a closed-form theoretical solution, deployable via logits-processor interface, and robust to temperature.

**Key Insight**: The authors view truncation from the optimal transport (OT) perspective—treating "truncate + renormalize" as transporting the original distribution $p$ to a $q_S$ supported on $S$, naturally introducing the Wasserstein-1 distance $W_1(p,q_S)$ as the faithfulness term, with $W_1$ computed using Mahalanobis distance over token embeddings.

**Core Idea**: Define the optimal truncation set via the minimization of "$W_1$ (geometry) + $\lambda H(q_S)$ (creativity) − $\beta\log\Gamma_S$ (mass)", and prove that the solution has a "prefix/single-token" structural form.

## Method

### Overall Architecture
Given the next-token distribution $p\in\Delta^{|V|}$ and token embedding geometry (Mahalanobis distance of whitened embeddings), Top-W at each step: (a) computes the objective $F_{\lambda,\beta}(S)=W_1(p,q_S)+\lambda H(q_S)-\beta\log\Gamma_S$ for each candidate set $S$; (b) since $W_1$ is intractable at vocabulary scale, replaces it with the Kantorovich-Rubinstein (KR) dual: $W_1=\sup_{f\in\mathcal{F}}(\mathbb{E}_p[f]-\mathbb{E}_{q_S}[f])$, and selects the anchored potential $f_S(i)=-\mathrm{dist}(i,S)$ as the most adversarial feasible potential; (c) alternately updates $f$ (using current $S$ to compute distance-to-set) and $S$ (using Theorem 3.4, a 1D prefix scan), and after 3-4 rounds of convergence, sets logits outside $S$ to $-\infty$ and proceeds with standard sampling.

### Key Designs

1. **Wasserstein-Entropy-Mass Objective + Exact Decomposition**:

    - **Function**: Unifies "geometric faithfulness + creativity control + mass preservation" into a single optimizable objective.
    - **Mechanism**: Defines $F_{\lambda,\beta}(S)=W_1(p,q_S)+\lambda H(q_S)-\beta\log\Gamma_S$. The paper proves $W_1(p,q_S)=(1-\Gamma_S)W_1(p(\cdot|S^c),p(\cdot|S))$, separating "deleted mass" from "geometric distance between deleted and retained parts"; thus, high-probability tokens far from the retained set are strongly penalized (avoiding pruning synonyms while retaining noise), while low-probability tokens close to the retained set may be included in $S$.
    - **Design Motivation**: Pure probability truncation ignores semantic similarity between tokens, treating "clusters of synonyms" and "semantic outliers" alike; introducing $W_1$ makes truncation "aware" of semantic structure.

2. **Geometric Anchored Potential + Closed-Form S-step**:

    - **Function**: Replaces the intractable KR dual LP with near-zero-cost distance queries, ensuring a closed-form solution for the truncation subproblem.
    - **Mechanism**: Fixing a feasible potential $f$, the paper proves $\arg\min_S F$ is equivalent to $\arg\max_S G_f(S)=\frac{1}{\Gamma_S}\sum_{i\in S}p_i\phi_i(f)+(\beta-\lambda)\log\Gamma_S$, where $\phi_i(f)=f_i+\lambda\log p_i$; further, (i) if $\beta\geq\lambda$, the optimal $S$ is a prefix sorted by $\phi_i$; (ii) if $\beta\leq\lambda$, the optimal $S$ degenerates to a singleton token. By fixing the potential as $f_S(i)=-\mathrm{dist}(i,S)$—the "most attractive" among all anchored 1-Lipschitz functions—the combined score becomes $\phi_i=-\mathrm{dist}(i,S)+\lambda\log p_i$.
    - **Design Motivation**: Solving the original $W_1$ LP at vocabulary scale ($10^5$) is impractical; using distance-to-set as the potential ensures 1-Lipschitz feasibility and explicitly penalizes tokens far from the current set. The closed-form S-step means the sampler only requires one sort and one prefix scan.

3. **Alternating f-step / S-step + Candidate Pool Pruning**:

    - **Function**: Approximates the joint optimum without explicitly solving OT.
    - **Mechanism**: Alternates (i) computing $f^{(t)}_i=-\mathrm{dist}(i,S^{(t)})$ with current $S^{(t)}$; (ii) sorting $\phi_i^{(t)}$ in descending order, prefix scanning $J_k=\Phi_k/\Gamma_k+(\beta-\lambda)\log\Gamma_k$ to select $k^\star$; (iii) stopping upon convergence. To avoid computing distances over the full vocabulary, a nucleus warm-start restricts to a candidate pool of top_m=1200; the appendix provides sufficient conditions for "exactness under pruning".
    - **Design Motivation**: Three iterations suffice for convergence; the candidate pool ensures per-token overhead is only milliseconds, empirically only 5.4% slower than Top-$H$/Top-$p$/Min-$p$.

### Loss & Training
This is an inference-time method with no training; the only hyperparameters $(\lambda,\beta)$ default to $(2.2,2.8)$. When $\beta>\lambda$ (prefix regime), adjusting $\beta$ allows trade-off between sharpness and creativity.

## Key Experimental Results

### Main Results
Three LLMs (Qwen2.5-3B, LLaMA-3.1-8B-Inst, Phi-3-Mini) × five temperatures $T\in\{0.5,0.7,1.0,1.5,2.0\}$, totaling 15 combinations:

| Benchmark | Top-W Wins | Max Relative Gain vs Top-H | Notes |
|-----------|-----------:|---------------------------:|-------|
| GSM8K     | 13/15      | **+33.7%** ($T=2.0$)      | Baselines collapse at high T |
| GPQA      | 12/15      | Typically 1-3 points      | Wins for all 3 models at $T\in\{1.5,2.0\}$ |
| AlpacaEval| 12/15      | Judge scores always win   | Length-controlled win-rate |
| MT-Bench  | 8/15       | Better multi-turn consistency | No drift at high T |

GSM8K at $T=2.0$: Top-W 75.13% / 73.09% / 84.63%, while Top-$p$ drops to 9.10% / 2.65% / 7.73%.

### Ablation Study

| Configuration | GSM8K@T=2.0 (LLaMA) | Description |
|---------------|--------------------:|-------------|
| $\beta>\lambda$ (prefix regime) | 73.09 | Default setting |
| $\beta\leq\lambda$ (singleton)  | Significant drop | Degenerates to single token |
| Excessively large $\beta$       | Creativity ↑ but GSM8K ↓ | Too much mass retained |
| Top-W (creative rubric $\beta=2.8$) | 27 settings win 12 | Outperforms Top-$p$/Top-$H$/Min-$p$ on average |

### Key Findings
- **Geometry + Entropy + Mass are all essential**: Mass-only → Top-$k$; entropy-only → Top-$H$; adding geometry fundamentally improves high-temperature robustness.
- **$\beta$ is a "creativity ↔ accuracy" dial**: Rubric evaluation (Diversity/Originality/Narrative/Emotion/Imagery) shows increasing $\beta$ boosts creativity but lowers strict answer scores; can be tuned per task.
- **Unified perspective**: The paper proves that under 0-1 uniform metric, Top-W reduces to Top-$k$ ($\lambda=\beta=0$) or Top-$H$ (Lagrangian relaxation with $\beta=0$), subsuming existing samplers in a single framework.
- **Controllable runtime overhead**: 3 alt rounds × top_m=1200, per-token cost is milliseconds, only 5.4% slower than Top-$p$—geometry-awareness is not a throughput killer.

## Highlights & Insights
- **Discovery of structurally optimal solutions**: Theorem 3.4 reduces combinatorial search over $2^{|V|}$ to a 1D scan; this general technique applies to any truncation objective with "weighted mean + concave/convex mass term".
- **OT perspective unifies truncation samplers**: Top-$k$/$p$/$H$ are all special cases of $W_1$+entropy+mass, providing a unified coordinate system for future decoding research.
- **"Whitelist" approach of anchored potentials**: Using 1-Lipschitz envelopes as surrogates avoids LP solving, a key engineering trick for OT deployment; this "distance-to-set as potential" approach is also instructive for other OT-on-discrete problems.
- **"Temperature robustness" as a new evaluation dimension**: Previous sampler papers rarely report at $T=2.0$; this work systematically demonstrates the high-temperature anti-collapse ability of geometry-aware truncation, advancing evaluation paradigms.

## Limitations & Future Work
- $W_1$ uses token embedding Mahalanobis distance as ground cost, but LLM embeddings do not strictly reflect "semantic distance"—polysemy, compound tokens, and rare tokens may mislead geometry.
- Candidate pool top_m=1200 is empirical; for very large vocabularies (>200k) or code tokens, reasonable but distant tokens may be missed.
- $(\lambda,\beta)$ require task-specific tuning; while the paper provides sensitivity analysis, there is no automated scheme, so industrial deployment still needs manual tuning.
- Experiments focus on instruction-tuned models and QA/chat scenarios; effects on code generation, long-context summarization, etc., remain untested.

## Related Work & Insights
- **vs Top-$k$ / nucleus / Min-$p$**: All rely solely on probability ranking; Top-W adds geometric correction, theoretically subsuming the former as special cases, and is significantly more robust at high temperatures.
- **vs Top-$H$ (bounded-entropy)**: Also a "distribution shaping" approach, but Top-$H$ ignores geometry; Top-W uses $W_1$ to treat synonymous neighbors as redundant, avoiding "pseudo-diversity".
- **vs Contrastive decoding / DoLa**: The latter modulate distributions via model/layer contrast; Top-W requires no reference model, using only single-model embedding geometry, with lower overhead.
- **Transferable insights**: Viewing "truncation" as "distribution-to-distribution transport" is a cross-domain idea, applicable to constrained generation (COMET-based MT, RAG re-ranking) and safety filtering; the proof of closed-form prefix solutions also inspires more combinatorial sampling problems to try "sort by mixed score then scan" structures.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce token embedding geometry + OT perspective into truncation samplers, with proof of structurally optimal solutions.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 benchmarks × 3 models × 5 temperatures (60 combinations) + rubric creative evaluation + runtime analysis; coverage is comprehensive, but lacks code generation scenarios.
- Writing Quality: ⭐⭐⭐⭐ Proofs and algorithm pseudocode are clear; a few symbols ($\phi,c,\beta-\lambda$) switch back and forth, which can be mentally taxing.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, training-free decoding improvement with high-temperature robustness, directly deployable for production use.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Gradient-Regularized Out-of-Distribution Detection](../../ECCV2024/llm_evaluation/gradient-regularized_out-of-distribution_detection.md)
- [\[ICLR 2026\] Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](../../ICLR2026/llm_evaluation/unpacking_human_preference_for_llms_demographically_aware_evaluation_of_long-fo.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/llm_evaluation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)
- [\[ACL 2026\] Min-k Sampling: Decoupling Truncation from Temperature Scaling via Relative Logit Dynamics](../../ACL2026/llm_evaluation/min-k_sampling_decoupling_truncation_from_temperature_scaling_via_relative_logit.md)
- [\[CVPR 2026\] Reframing Long-Tailed Learning via Loss Landscape Geometry](../../CVPR2026/llm_evaluation/reframing_long-tailed_learning_via_loss_landscape_geometry.md)

</div>

<!-- RELATED:END -->
