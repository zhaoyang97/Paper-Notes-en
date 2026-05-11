---
title: >-
  [Paper Note] AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution
description: >-
  [ICLR 2026][Model Compression][Knowledge Distillation] This paper proposes the α-mixture assistant distribution and a unified distillation framework…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Assistant Distribution"
  - "α-mixture"
  - "f-divergence"
  - "LLM Compression"
date: 2026-05-08
content_hash: b745c16362b5c78e
---

# AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution

**Conference**: ICLR 2026
**arXiv**: [2510.15982](https://arxiv.org/abs/2510.15982)
**Code**: [https://github.com/aailab-kaist/AMiD](https://github.com/aailab-kaist/AMiD)
**Area**: Model Compression / Knowledge Distillation
**Keywords**: Knowledge Distillation, Assistant Distribution, α-mixture, f-divergence, LLM Compression

## TL;DR
This paper proposes the α-mixture assistant distribution and a unified distillation framework, AMiD. By introducing a new design variable α that controls the geometric shape of the interpolation path between teacher and student distributions, AMiD generalizes existing assistant distribution methods (m-mixture and e-mixture are special cases at α=±1), proves optimality guarantees under arbitrary divergences and α values, and achieves state-of-the-art performance on multiple LLM distillation benchmarks.

## Background & Motivation

**Background**: LLM knowledge distillation compresses models by aligning token-level distributions between teacher and student. Recent work has introduced "assistant distributions"—mixtures of teacher and student distributions—to mitigate training instability caused by capacity gaps and near-zero probabilities.

**Limitations of Prior Work**: (a) Existing assistant distribution methods (GKD/DistiLLM use arithmetic mean, i.e., m-mixture; TAID uses geometric mean, i.e., e-mixture) were proposed independently without a unified framework; (b) the design of the assistant distribution is coupled with the choice of divergence, artificially constraining the search space; (c) α, which controls the geometry of the interpolation path, is fixed at ±1 and has not been explored.

**Key Challenge**: In the high-dimensional output space of LLMs, a large number of probabilities are near zero, making density ratio estimation unstable. Furthermore, the teacher–student capacity gap makes direct alignment difficult. The assistant distribution is key to addressing both issues, yet existing designs lack generality.

**Goal**: To establish a unified theoretical framework for assistant distributions and divergences, and to discover novel, more effective assistant distribution forms.

**Key Insight**: Existing assistant distributions are unified via generalized $f_\alpha$-means from information geometry—m-mixture and e-mixture correspond to the arithmetic mean (α=−1) and geometric mean (α=1), respectively, while α can take any real value.

**Core Idea**: The generalized $f_\alpha$-mean extends the assistant distribution from two discrete choices to a continuously parameterized family of distributions, with optimality guarantees proven for arbitrary α and divergence.

## Method

### Overall Architecture
AMiD introduces two control variables: α (controlling the geometry of the interpolation path) and λ (controlling the interpolation position). The assistant distribution is defined as:
$$\tilde{r}_\theta^{(\alpha,\lambda)}(z) = \left(\lambda p(z)^{\frac{1-\alpha}{2}} + (1-\lambda) q_\theta(z)^{\frac{1-\alpha}{2}}\right)^{\frac{2}{1-\alpha}} \quad (\alpha \neq 1)$$
which is normalized to yield a valid probability distribution. The distillation objective minimizes $D(p, r_\theta^{(\alpha,\lambda)})$ or $D(q_\theta, r_\theta^{(\alpha,\lambda)})$, where D can be any divergence.

### Key Designs

1. **α-mixture Assistant Distribution Family**

   - **Function**: Controls the geometric shape of the interpolation path between teacher and student distributions via the parameter α.
   - **Mechanism**: At α=−1, the distribution reduces to the arithmetic mean (m-mixture, a linear path); at α=1, it reduces to the geometric mean (e-mixture, a linear path in log-space); at α=3, it yields the harmonic mean; other values of α define novel interpolation paths. Theorem 3.2 proves that $r^{(\alpha,\lambda)}$ is an interior point (a point on the geodesic) between p and q in the sense of α-divergence.
   - **Design Motivation**: When α<1, the support is the union of those of p and q (mode-covering); when α≥1, the support is the intersection (mode-seeking)—a distinction that directly governs distillation behavior.

2. **Optimality Guarantee (Theorem 3.4)**

   - **Function**: Proves that for any regular divergence D and any α, the optimal solution of AMiD is equivalent to p=q_θ.
   - **Mechanism**: If the assistant distribution perfectly matches the teacher, then q_θ must equal p—when the interpolation point coincides with one endpoint, it must coincide with the other.
   - **Design Motivation**: Ensures that introducing an assistant distribution does not alter the ultimate objective of distillation.

3. **Gradient Analysis and Mode-Covering/Seeking Control (Proposition 3.5)**

   - **Function**: Analyzes how α influences gradient behavior under f-divergence.
   - **Mechanism**: The gradient contains a weighting term $w = \frac{(1-\lambda)q_\theta^{\frac{1-\alpha}{2}}}{\lambda p^{\frac{1-\alpha}{2}} + (1-\lambda)q_\theta^{\frac{1-\alpha}{2}}}$; larger α amplifies w in regions where p>q_θ (mode-covering), while smaller α amplifies w where p<q_θ (mode-seeking).
   - **Design Motivation**: Even with a fixed divergence D, the quality–diversity trade-off can still be controlled through α.

### Loss & Training
- Compatible with arbitrary divergences and data strategies; α-β divergence with λ=0.1 is recommended.
- α<1 is suited for mode-covering; α≥1 for mode-seeking.
- Adaptive α scheduling is supported.

## Key Experimental Results

### Main Results — GPT-2 XL → GPT-2 Distillation (Instruction-Following ROUGE-L)

| Method | Dolly | Self-Inst | Vicuna | Super NI | Avg |
|--------|-------|-----------|--------|----------|-----|
| GKD (α=−1) | 24.58 | 11.78 | 14.60 | 22.84 | ~18 |
| DistiLLM (α=−1) | ~25 | ~12 | ~15 | ~23 | ~19 |
| TAID (α=1) | ~25 | ~12 | ~15 | ~23 | ~19 |
| **AMiD** | **Best** | **Best** | **Best** | **Best** | **Best** |

### Ablation Study — Effect of α
Intermediate values of α between −1 and 1 (e.g., α=0) achieve the best performance on most tasks, demonstrating that the endpoint values used by existing methods miss the optimal region. Toy experiments validate the theoretical prediction that α controls mode-covering/seeking behavior.

### Key Findings
- α and λ are orthogonal design dimensions: λ controls "how far" to interpolate, while α controls "along which path."
- The optimal α varies across tasks, but intermediate values consistently outperform endpoint values.
- AMiD exhibits more stable training, as the assistant distribution alleviates near-zero probability issues.

## Highlights & Insights
- The **unification from an information-geometric perspective** is highly elegant—generalized means combined with the α-divergence geodesic theorem consolidate disparate methods into a continuous parametric family.
- The **orthogonality of α and λ** is the central insight: all prior work tuned only λ and not α, missing an important design dimension.
- The gradient analysis in Proposition 3.5 formalizes the intuition behind mode-covering/seeking behavior.

## Limitations & Future Work
- The optimal choice of α still requires empirical tuning; an automated selection mechanism is lacking.
- Experiments are conducted primarily at the GPT-2 scale (0.1B–1.5B); validation on larger LLMs is insufficient.
- The normalization constant $Z_r$ introduces additional computational overhead.
- No comparison with non-KD compression methods is provided.

## Related Work & Insights
- **vs. GKD**: GKD employs GJS divergence with an implicit m-mixture (α=−1); AMiD generalizes this to arbitrary α.
- **vs. TAID**: TAID uses e-mixture (α=1); AMiD reveals this to be merely an endpoint value.
- **vs. DistiLLM**: DistiLLM uses skew KL divergence (α=−1); AMiD demonstrates that intermediate α values are superior.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — An information-geometry-driven unified framework; α as a new design dimension is a profound contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-task evaluation, ablations, and toy experiments are comprehensive, but model scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous; figures are highly intuitive.
- **Value**: ⭐⭐⭐⭐ — Provides a unified theoretical foundation for assistant distribution design in LLM knowledge distillation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[AAAI 2026\] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](../../AAAI2026/model_compression/tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)
- [\[ICLR 2026\] Distillation of Large Language Models via Concrete Score Matching](distillation_of_large_language_models_via_concrete_score_matching.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](../../ICCV2025/model_compression/knowledge_distillation_with_refined_logits.md)
- [\[AAAI 2026\] Condensed Data Expansion Using Model Inversion for Knowledge Distillation](../../AAAI2026/model_compression/condensed_data_expansion_using_model_inversion_for_knowledge_distillation.md)

</div>

<!-- RELATED:END -->
