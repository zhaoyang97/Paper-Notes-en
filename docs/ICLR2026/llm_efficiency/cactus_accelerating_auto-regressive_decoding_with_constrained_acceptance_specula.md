---
title: >-
  [Paper Note] Cactus: Accelerating Auto-Regressive Decoding with Constrained Acceptance Speculative Sampling
description: >-
  [ICLR 2026][LLM Efficiency][Paper Note] This paper reformulates speculative sampling as a constrained optimization problem—"maximizing acceptance rate under controlled divergence constraints"—and proposes Cactus. By reading only one probability value of a candidate token and adding a "bonus" determined by $q$ and a divergence budget $\delta$, it significantl
tags:
  - ICLR 2026
  - LLM Efficiency
date: 2026-05-08
content_hash: 0363e662981ef181
---
# Cactus: Accelerating Auto-Regressive Decoding with Constrained Acceptance Speculative Sampling

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=lpUIkCAy9p](https://openreview.net/forum?id=lpUIkCAy9p)  
**Code**: [https://github.com/MANGA-UOFA/Cactus](https://github.com/MANGA-UOFA/Cactus)  
**Area**: LLM Inference Acceleration / Speculative Sampling  
**Keywords**: Speculative sampling, constrained optimization, KL divergence, acceptance rate, lossless decoding, LLM inference  

## TL;DR
This paper reformulates speculative sampling as a constrained optimization problem—"maximizing acceptance rate under controlled divergence constraints"—and proposes Cactus. By reading only one probability value of a candidate token and adding a "bonus" determined by $q$ and a divergence budget $\delta$, it significantly improves acceptance rate and throughput while keeping the deviation from the verifier distribution controllable.

## Background & Motivation
**Background**: Speculative Sampling (SpS) employs a small draft model to autoregressively propose $m$ tokens, which are then verified in parallel by a large verifier model. This allows a single forward pass of the large model to produce multiple tokens, mitigating memory bottlenecks and increasing decoding throughput. The primary selling point of SpS is its "lossless" nature—guaranteeing that the generation distribution strictly matches the verifier distribution $q$.

**Limitations of Prior Work**: Such strict distributional equivalence is often "overly conservative." In practice, verifier distributions are already modified by sampling techniques like top-k or temperature; thus, slight deviations from $q$ are acceptable. SpS rejects "correct but slightly lower probability" tokens, unnecessarily sacrificing acceptance rates. Typical Acceptance Sampling (TAS) attempts to mitigate this using entropy-based heuristics, but this paper points out that TAS **distorts the verifier distribution**. When $q$ carries critical fine-grained signals, it causes semantic drift and performance degradation (e.g., TAS significantly drops below the $-1\sigma$ line on GPQA).

**Key Challenge**: A tension exists between acceptance rate (throughput) and fidelity to the verifier distribution (quality). SpS sits at the "fidelity" extreme, sacrificing throughput; TAS sits at the "aggressive acceptance" extreme, sacrificing quality without **explicit control over the deviation**.

**Goal**: To provide a **hard-constrained, adjustable, and guaranteed** knob for "how much to deviate from $q$," allowing users to safely trade-off quality and throughput rather than relying on heuristics.

**Core Idea**: **Modeling speculative sampling as constrained optimization**. The objective is to maximize the acceptance rate of candidate tokens, subject to the constraint that the $f$-divergence between the dynamically constructed target distribution $h$ and the verifier $q$ does not exceed a budget $\delta$. Solving this problem yields a simple closed-form correction: Cactus.

## Method

### Overall Architecture
The authors first abstract the draft-and-verify process into a unified algorithm (comprising two pluggable components: an acceptance rate function $\phi$ and a recovery distribution $g$). They prove that as long as the target distribution is set to any $h$ and $\phi, g$ are defined accordingly, the algorithm "precisely produces $h$ with optimal acceptance." SpS is a special case where $h=q$. The key insight of Cactus is: **one does not need to stick to $h=q$; instead, one can dynamically pick an $h$ that makes the candidate token easier to accept under the constraint that it remains "near $q$."** The pipeline is: Constrained optimization modeling → Closed-form solution $h$ (Theorem 2) → Minimal scoring via KL + second-order Taylor approximation (Corollary 5) → Reversing $\phi, g$ back into the unified algorithm.

```mermaid
flowchart LR
    A[Draft model p<br/>Propose candidate token n] --> B[Constrained optimization<br/>max acceptance rate<br/>s.t. D_f h‖q ≤ δ]
    B --> C[Closed-form solution h<br/>Add bonus to token n]
    C --> D[KL + 2nd order Taylor<br/>γ* = q + √2δq1-q]
    D --> E[Reverse φ, g<br/>Plug into draft-verify]
    E --> F[Higher acceptance rate<br/>+ controlled divergence]
```

### Key Designs

**1. Modeling speculative sampling as constrained optimization: Explicitly buying acceptance rate with a divergence budget $\delta$.** This is the theoretical cornerstone. At each timestep, given the token index $n$ sampled by the draft model, the authors treat the target distribution $h\in\mathbb{R}^{|V|}$ as an optimization variable, solving $\max_h \min\{h_n/p(n|x_{<t}),1\}$ subject to $h$ being a valid distribution and $D_f(h\|q(\cdot|x_{<t}))\le\delta$. Here, $\delta\ge0$ is the knob: $\delta=0$ recovers lossless SpS, while larger $\delta$ allows more aggressive acceptance. This formulation transforms heuristic-based "accept more" logic into a convex optimization with explicit goals and guarantees.

**2. Closed-form optimal solution: Boosting only the candidate token and scaling others proportionally.** For the aforementioned optimization, Theorem 2 provides the closed-form $h$: the candidate token receives $h_n=\gamma^\*$, and other tokens $i$ are set to $h_i=\frac{1-\gamma^\*}{1-q(n|x_{<t})}q(i|x_{<t})$, where $\gamma^\*$ is the root of the divergence constraint equation in the interval $[q(n),+\infty)$. Intuitively, it "lifts the probability of the candidate from $q(n)$ to $\gamma^\*\ge q(n)$ and subtracts the difference proportionally from other tokens." Theorem 3 provides a safeguard: $D_f(h_{\text{alg}}\|q)\le\min\{\Gamma(\delta),\,D_f(p\|q)\}$, meaning the **total deviation remains implicitly locked by $\delta$**.

**3. KL Divergence + Second-order Taylor yields the minimal scoring $\gamma^\*$.** While any $f$-divergence can be used, the authors argue for KL divergence ($f(t)=t\log t$) over the cross-entropy implicitly used by TAS. The reason lies in the decomposition $H(h,q)=D_{\mathrm{KL}}(h\|q)+H(h)$: the entropy term $H(h)$ in cross-entropy encourages $h$ to collapse into a deterministic distribution. TAS tends to force $h$ toward zero entropy, increasing divergence by at least $H(q)$—causing severe deviation when $q$ is high-entropy. Pure KL avoids this collapse. Using a second-order Taylor expansion at $\gamma_0=q(n)$, the final Cactus score is derived:

$$\gamma^\* = \min\left\{\, q(n|x_{<t}) + \sqrt{2\delta\, q(n|x_{<t})\big(1-q(n|x_{<t})\big)},\ 1 \,\right\}$$

This adds a bonus to the candidate token's probability. Corollary 6 proves that when the verifier is less confident ($\gamma^\*\le0.5$), this approximation **strictly satisfies** $D_{\mathrm{KL}}(h\|q)\le\delta$.

**4. Engineering efficiency: Reading a single probability to reduce memory access.** Unlike TAS, which requires iterating over the entire vocabulary to calculate entropy, Cactus only requires reading the probability $q(n)$ at the candidate token's index. In modern LLMs with large vocabularies, this further reduces memory overhead, aligning perfectly with the goal of speculative sampling.

## Key Experimental Results

### Main Results
Evaluated on GSM8K, IFEval, and GPQA using the Qwen3 series (0.6B draft + 8B/14B verifiers), reporting strict-match accuracy, expected acceptance length $AL_m$, and rejected tokens relative to SpS (Rej).

| Setting (m=20, Qwen3-14B Verifier + 0.6B Draft) | GSM8K Score↑ / AL↑ / Rej↓ | GPQA Score↑ / AL↑ / Rej↓ |
|---|---|---|
| Verifier (Upper bound) | 91.71 / – / – | 40.07 / – / – |
| SpS (Lossless baseline) | 91.89 / 5.11 / Ref | 40.91 / 3.84 / Ref |
| TAS | 92.87 / 6.78 / −32% | 40.40 / 6.41 / −46% |
| Cactus 0.75 (**Ours**) | 92.15 / 7.15 / −36% | **45.46** / 6.46 / −46% |
| Cactus 1.0 (**Ours**) | 92.87 / 7.00 / −34% | **45.46** / 6.74 / −50% |

Cactus achieves the highest acceptance rates across all benchmarks while maintaining or improving accuracy. On the difficult GPQA benchmark, TAS degrades compared to SpS, whereas Cactus improves accuracy to 45.46, excelling in both throughput and quality.

### Ablation Study

| Analysis | Key Findings |
|---|---|
| Necessity of Divergence Control | At ~90% acceptance, Cactus scores >86, while simple linear interpolation ($\alpha=0.9$) falls to <72. Cactus maintains >80 even at 96.3% acceptance. |
| Accuracy-Acceptance Trade-off | TAS falls below the $-1\sigma$ significance line on GPQA; Cactus remains within or above the verifier's confidence interval. |
| Wall-clock Throughput (A100 + vLLM) | Cactus 1.0 achieves nearly **1.9× speedup** over a single verifier (0.6B+14B, m=10) while achieving the highest GPQA score. |
| Generalization | Validated across Gemma, DeepSeek-R1, LLaMA, and Qwen3 series, outperforming top-k acceptance baselines. |

## Highlights & Insights
- **Theoretical Unification**: Reformulates speculative sampling as constrained optimization, covering both SpS ($\delta=0$) and providing a new interpretation of TAS as a cross-entropy variant.
- **Precise Diagnosis**: Uses the $H(h,q)=D_{\mathrm{KL}}+H(h)$ decomposition to mathematically identify that the performance drop in TAS stems from distributional collapse caused by the entropy term.
- **Simplicity and Practicality**: The final method merely adds an analytical bonus to the candidate token and reads only one probability value, making it easy to integrate into existing SpS/vLLM pipelines with near-zero overhead.

## Limitations & Future Work
- $\Gamma(\delta)$ lacks a closed-form expression; the mapping between $\delta$ and the actual overall deviation still relies on empirical tuning.
- The second-order Taylor approximation is strictly valid only for small $\delta$; guarantees for more aggressive $\delta$ intervals are less robust.
- Experiments focus on reasoning/math/instruction-following with smaller draft models; quality drift in open-ended long-text generation or massive draft-verifier ratios requires further systematic study.

## Related Work & Insights
- **Speculative Sampling Taxonomy**: From SpS (Chen et al., 2023) to Medusa/TAS (Cai et al., 2024), this work unifies "lossless vs. lossy acceptance" into constrained optimization, providing a principled template for future acceptance criteria (changing $f$-divergence equals changing the strategy).
- **Alignment with Sampling Research**: Resonates with nucleus/typical sampling (Holtzman et al., 2020), acknowledging that "exact $q$ is not required" while elevating deviation from a heuristic to a controllable constraint.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Reformulating speculative sampling as constrained optimization is a clean and profound theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers three benchmarks, four model series, and various ratios; includes wall-clock throughput and interpolation ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Progresses logically from motivation to theorems and approximations; the analysis of TAS's flaws is particularly clear.
- **Value**: ⭐⭐⭐⭐ — A near-zero overhead, easily applicable improvement for lossy speculative sampling with a reusable design paradigm.

<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICLR 2026\] NI Sampling: Accelerating Discrete Diffusion Sampling by Token Order Optimization](ni_sampling_accelerating_discrete_diffusion_sampling_by_token_order_optimization.md)
- [\[ICLR 2026\] Global Resolution: Optimal Multi-Draft Speculative Sampling via Convex Optimization](global_resolution_optimal_multi-draft_speculative_sampling_via_convex_optimizati.md)
- [\[ICLR 2026\] CONCUR: A Framework for Continual Constrained and Unconstrained Routing](concur_a_framework_for_continual_constrained_and_unconstrained_routing.md)
- [\[ICLR 2026\] vAttention: Verified Sparse Attention via Sampling](vattention_verified_sparse_attention_via_sampling.md)
- [\[ICLR 2026\] Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding](overcoming_joint_intractability_with_lossless_hierarchical_speculative_decoding.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Speculative Speculative Decoding](speculative_speculative_decoding.md)
- [\[ICLR 2026\] Global Resolution: Optimal Multi-Draft Speculative Sampling via Convex Optimization](global_resolution_optimal_multi-draft_speculative_sampling_via_convex_optimizati.md)
- [\[ICLR 2026\] CONCUR: A Framework for Continual Constrained and Unconstrained Routing](concur_a_framework_for_continual_constrained_and_unconstrained_routing.md)
- [\[ICLR 2026\] LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)
- [\[ICLR 2026\] Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs](not-a-bandit_provably_no-regret_drafter_selection_in_speculative_decoding_for_ll.md)

</div>

<!-- RELATED:END -->
