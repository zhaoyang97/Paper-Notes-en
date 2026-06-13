---
title: >-
  [Paper Note] Top-W: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for LLMs
description: >-
  [ICML 2026][LLM Evaluation][Truncated Decoding] Top-W formulates next-token truncation as a minimization problem involving three terms: "Wasserstein-Entropy-Mass…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Truncated Decoding"
  - "Wasserstein Distance"
  - "Token Embedding Geometry"
  - "Entropy Constraint"
  - "High-Temperature Robustness"
date: 2026-05-08
content_hash: c0354a3604671a88
---

# Top-W: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for LLMs

**Conference**: ICML 2026  
**arXiv**: [2602.10346](https://arxiv.org/abs/2602.10346)  
**Code**: https://github.com/arashgholami/top-w-decoding (Yes)  
**Area**: LLM Decoding / Evaluation / Runtime Control  
**Keywords**: Truncated Decoding, Wasserstein Distance, Token Embedding Geometry, Entropy Constraint, High-Temperature Robustness

## TL;DR
Top-W formulates next-token truncation as a minimization problem involving three terms: "Wasserstein-Entropy-Mass," which accounts for token embedding geometry. Theoretical proofs show that the optimal solution is either a single token or a prefix sorted by $f(i)+\lambda\log p_i$. The implementation requires only an $O(n\log n)$ scan; it outperforms competitors in the majority of 15 (T, model) combinations across GSM8K, GPQA, AlpacaEval, and MT-Bench, with up to a 33.7% improvement over Top-$H$ on GSM8K at high temperatures.

## Background & Motivation

**Background**: Truncated sampling for LLM decoding has long been standard infrastructure—Top-$k$, Top-$p$ (nucleus), Min-$p$, and locally typical sampling all prune low-probability tails from a "probability ranking" perspective. Recently, Top-$H$ explicitly introduced an "entropy of the truncated sub-distribution not exceeding a threshold" as a constraint, marking one of the first works to approach the problem from a "distribution shaping" perspective.

**Limitations of Prior Work**: All existing rules treat tokens as structureless categories—considering only probability while ignoring the semantic distance of tokens in the embedding space. Consequently: (i) at high temperatures ($T\geq 1.5$), Top-$p$ / Min-$p$ frequently expand to nearly the entire vocabulary, leading to collapsed outputs; (ii) even with entropy control (Top-$H$), probability may still concentrate on synonymous or near-synonymous neighbor tokens, resulting in "pseudo-diversity" that lacks true creativity.

**Key Challenge**: Decoders must balance (i) faithfulness (not straying too far from the original distribution), (ii) creativity (sufficient diversity), and (iii) coherence (retaining enough probability mass). The first two essentially require measurement within the token geometric space, yet all current samplers bypass geometric information.

**Goal**: To "explicitly" incorporate token embedding geometry into the truncation objective, providing a geometry-aware sampler that has a theoretical closed-form solution, can be deployed via the logits-processor interface, and is robust to temperature variations.

**Key Insight**: The authors view truncation through the lens of Optimal Transport (OT)—treating "truncation + renormalization" as transporting the original distribution $p$ to a distribution $q_S$ supported on $S$. This naturally introduces the Wasserstein-1 distance $W_1(p,q_S)$ as a faithfulness term, where $W_1$ can use the Mahalanobis distance on token embeddings as the ground cost.

**Core Idea**: Define the optimal truncation set through the optimization of "$W_1$ (Geometry) + $\lambda H(q_S)$ (Creativity) − $\beta\log\Gamma_S$ (Quality)" and prove that this problem possesses a structural solution consisting of a "prefix" or a "single token."

## Method

### Overall Architecture
Given a next-token distribution $p\in\Delta^{|V|}$ and token embedding geometry (Mahalanobis distance of whitened embeddings), Top-W performs the following at each step: (a) computes the objective $F_{\lambda,\beta}(S)=W_1(p,q_S)+\lambda H(q_S)-\beta\log\Gamma_S$ for each candidate set $S$; (b) since $W_1$ is intractable at vocabulary scale, it employs the Kantorovich-Rubinstein (KR) dual: $W_1=\sup_{f\in\mathcal{F}}(\mathbb{E}_p[f]-\mathbb{E}_{q_S}[f])$ and selects an anchored potential $f_S(i)=-\mathrm{dist}(i,S)$ as the most aggressive feasible potential; (c) iteratively updates $f$ (calculating distance-to-set using the current $S$) and $S$ (using a 1D prefix scan according to Theorem 3.4). After 3-4 rounds of convergence, logits not in $S$ are set to $-\infty$ for standard sampling.

### Key Designs

1. **Wasserstein-Entropy-Mass Triangular Objective + Precise Decomposition**:
    - **Function**: Unifies "geometric faithfulness, creativity control, and quality retention" into a single optimizable objective.
    - **Mechanism**: Defines $F_{\lambda,\beta}(S)=W_1(p,q_S)+\lambda H(q_S)-\beta\log\Gamma_S$. The paper proves $W_1(p,q_S)=(1-\Gamma_S)W_1(p(\cdot|S^c),p(\cdot|S))$, separating "deleted mass" from the "geometric distance between deleted and retained portions." Consequently, tokens with high probability but far geometric distance from the retained set are heavily penalized (preventing the accidental retention of noise over synonyms), while low-probability tokens close to the retained set may be included in $S$.
    - **Design Motivation**: Pure probability truncation ignores semantic similarity, treating "synonym clusters" and "semantic outliers" identically. Introducing $W_1$ allows truncation to "perceive" semantic structure.

2. **Geometric Anchored Potential + Closed-form S-step**:
    - **Function**: Replaces the intractable KR dual LP with near-zero-cost distance queries and ensures the truncation sub-problem has a closed-form optimum.
    - **Mechanism**: Fixing a feasible potential $f$, the paper proves $\arg\min_S F$ is equivalent to $\arg\max_S G_f(S)=\frac{1}{\Gamma_S}\sum_{i\in S}p_i\phi_i(f)+(\beta-\lambda)\log\Gamma_S$, where $\phi_i(f)=f_i+\lambda\log p_i$. It further proves: (i) if $\beta\geq\lambda$, the optimal $S$ must be a prefix sorted by $\phi_i$; (ii) if $\beta\leq\lambda$, the optimal $S$ collapses to a single token. By fixing the potential as $f_S(i)=-\mathrm{dist}(i,S)$—the "most attractive" among all anchored 1-Lipschitz functions—the combined score becomes $\phi_i=-\mathrm{dist}(i,S)+\lambda\log p_i$.
    - **Design Motivation**: Solving an LP for $W_1$ on a vocabulary of size $10^5$ is impractical. Using distance-to-set as a potential ensures 1-Lipschitz feasibility while explicitly penalizing tokens far from the current set. The closed-form S-step ensures the sampler costs only one sorting operation plus one prefix scan.

3. **Alternating f-step / S-step + Candidate Pool Pruning**:
    - **Function**: Approximates the joint optimal solution without explicitly solving the OT problem.
    - **Mechanism**: An alternating loop: (i) calculate $f^{(t)}_i=-\mathrm{dist}(i,S^{(t)})$ based on $S^{(t)}$; (ii) sort by $\phi_i^{(t)}$ in descending order and perform a prefix scan $J_k=\Phi_k/\Gamma_k+(\beta-\lambda)\log\Gamma_k$ to find $k^\star$; (iii) stop upon convergence. To avoid calculating distances across the entire vocabulary, a nucleus warm-start limits the pool to $top\_m=1200$ candidates. The appendix provides sufficient conditions for "exactness under pruning."
    - **Design Motivation**: Three iterations are sufficient for convergence. The candidate pool ensures per-token overhead is in milliseconds, empirically only 5.4% slower than Top-$H$/Top-$p$/Min-$p$ on average.

### Loss & Training
This is an inference-time method with no training. The only hyperparameters $(\lambda,\beta)$ default to $(2.2, 2.8)$. When $\beta>\lambda$ enters the prefix range, one can slide between sharpness and creativity by adjusting $\beta$.

## Key Experimental Results

### Main Results
Testing across 3 LLMs (Qwen2.5-3B, LLaMA-3.1-8B-Inst, Phi-3-Mini) $\times$ 5 temperatures $T\in\{0.5, 0.7, 1.0, 1.5, 2.0\}$, totaling 15 combinations:

| Benchmark | Top-W Wins | Max Relative Gain vs Top-H | Remarks |
|-----------|-----------:|----------------------:|------|
| GSM8K     | 13/15 | **+33.7%** ($T=2.0$) | Baselines almost collapse at high $T$ |
| GPQA      | 12/15 | Generally 1-3 points | Wins across all 3 models at $T\in\{1.5, 2.0\}$ |
| AlpacaEval| 12/15 | Consistent judge wins | Length-controlled win-rate |
| MT-Bench  | 8/15  | Better multi-turn consistency | Prevents drift at high temperatures |

On GSM8K at $T=2.0$: Top-W achieves 75.13% / 73.09% / 84.63%, while Top-$p$ drops to 9.10% / 2.65% / 7.73%.

### Ablation Study

| Configuration | GSM8K@T=2.0 (LLaMA) | Description |
|------|--------------------:|------|
| $\beta>\lambda$ (Prefix range) | 73.09 | Default setting |
| $\beta\leq\lambda$ (Singleton) | Significant Drop | Degenerates to single token |
| Excessively large $\beta$ | Creativity ↑ but GSM8K ↓ | Retains too much mass |
| Top-W (creative rubric $\beta=2.8$) | Wins 27 configurations | Higher avg than Top-$p$/Top-$H$/Min-$p$ |

### Key Findings
- **Geometry + Entropy + Mass are all indispensable**: Mass alone leads to Top-$k$; Entropy alone leads to Top-$H$; adding Geometry creates a qualitative leap in high-temperature robustness.
- **$\beta$ acts as a "Creativity $\leftrightarrow$ Accuracy" regulator**: Rubric evaluations (Diversity/Originality/Narrative/Emotion/Imagery) show that increasing $\beta$ raises creativity scores but lowers exact-answer scores, allowing for task-specific tuning.
- **Unified Perspective**: The paper proves that under a 0-1 uniform metric, Top-W degenerates into Top-$k$ (with $\lambda=\beta=0$) or Top-$H$ (Lagrangian relaxation with $\beta=0$), placing existing samplers within a single framework.
- **Controllable Overhead**: 3 alt rounds $\times$ $top\_m=1200$ takes roughly milliseconds per token, only 5.4% slower than Top-$p$—meaning geometry-awareness is not a throughput killer.

## Highlights & Insights
- **Discovery of Structural Optimal Solutions**: Theorem 3.4 reduces a $2^{|V|}$ combinatorial search to a 1D scan, a general-purpose trick reusable for any truncation objective involving "weighted average + concave/convex mass terms."
- **Unifying Truncated Samplers via OT**: Viewing Top-$k$/$p$/$H$ as special cases of $W_1$+Entropy+Mass provides a unified coordinate system for future decoding research.
- **"White-list" Approach via Anchored Potentials**: Using a 1-Lipschitz envelope as a surrogate to avoid solving LPs is a key engineering masterstroke for landing OT; this "using distance-to-set as potential" tactic is valuable for other OT-on-discrete problems.
- **Temperature Robustness as a New Evaluation Dimension**: Previous sampler papers rarely reported results at $T=2.0$. This work systematically demonstrates the anti-collapse capability of geometry-aware truncation, representing progress in evaluation paradigms.

## Limitations & Future Work
- $W_1$ uses Mahalanobis distance on token embeddings as ground cost, but LLM embeddings do not strictly reflect "semantic distance"—polysemy, subword tokens, and rare tokens might mislead the geometry.
- The candidate pool $top\_m=1200$ is empirical; for ultra-large vocabularies (>200k) or code tokens, distant but reasonable tokens might still be missed.
- $(\lambda, \beta)$ require task-specific tuning; the paper provides sensitivity analysis but no automated scheme, meaning industrial deployment still requires parameter search.
- Experiments focused on instruction-tuned models + QA / chat scenarios; effectiveness on tasks like code generation or long-context summarization remains unverified.

## Related Work & Insights
- **vs Top-$k$ / Nucleus / Min-$p$**: These only consider probability ranking. Top-W adds a geometric correction, theoretically including the former as special cases and showing significantly more stability at high temperatures.
- **vs Top-$H$ (Bounded-Entropy)**: Also uses a "distribution shaping" perspective, but Top-$H$ ignores geometry. Top-W uses $W_1$ to treat synonymous neighbors as redundant, avoiding "pseudo-diversity."
- **vs Contrastive Decoding / DoLa**: The latter adjust distributions by contrasting different models/layers. Top-W requires no reference model, using only a single model + embedding geometry, resulting in lower overhead.
- **Transferable Insights**: Treating "truncation" as "distribution-to-distribution transport" is a cross-disciplinary idea applicable to constrained generation (COMET-based MT, RAG re-ranking) and safety filtering; the proof of a closed-form prefix solution inspires similar "sort by hybrid score then scan" approaches for other combinatorial sampling problems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to bring token embedding geometry + OT perspective into truncated samplers and prove a structural optimal solution.
- Experimental Thoroughness: ⭐⭐⭐⭐ 60 combinations (4 benchmarks × 3 models × 5 temperatures) + rubric creativity evaluation + overhead analysis; well-covered but lacks code generation.
- Writing Quality: ⭐⭐⭐⭐ Clear proof formatting and complete pseudocode; minor friction with switching symbols ($\phi, c, \beta-\lambda$).
- Value: ⭐⭐⭐⭐⭐ Training-free, plug-and-play at the decoding side with high-temperature robustness—a deployment-ready improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Spherical Steering: Geometry-Aware Activation Rotation for Language Models](spherical_steering_geometry-aware_activation_rotation_for_language_models.md)
- [\[ACL 2026\] Pressure-Testing Deception Probes in LLMs: Scaling, Robustness, and the Geometry of Deceptive Representations](../../ACL2026/llm_evaluation/pressure-testing_deception_probes_in_llms_scaling_robustness_and_the_geometry_of.md)
- [\[ICLR 2026\] Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](../../ICLR2026/llm_evaluation/unpacking_human_preference_for_llms_demographically_aware_evaluation_of_long-fo.md)
- [\[ICML 2026\] REAL: Integrating Regression-Aware Rewards into RL for Fine-Grained Human-Centric LLM Evaluation](real_regression-aware_reinforcement_learning_for_llm-as-a-judge.md)
- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](../../ACL2026/llm_evaluation/contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)

</div>

<!-- RELATED:END -->
