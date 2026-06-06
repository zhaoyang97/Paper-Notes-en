---
title: >-
  [Paper Note] FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA
description: >-
  [ICML 2026][Model Compression][Federated Learning] This paper identifies that the true "enemy" of naive factor-wise averaging in federated LoRA is the latent subspace misalignment caused by rotational invariance. It prop…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Federated Learning"
  - "LoRA"
  - "Procrustes Alignment"
  - "Rotational Invariance"
  - "Subspace Alignment"
date: 2026-05-08
content_hash: 6ae5e0c8a44399c0
---

# FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA

**Conference**: ICML 2026  
**arXiv**: [2602.23638](https://arxiv.org/abs/2602.23638)  
**Code**: https://github.com/haoran-zh/FedRot-LoRA (available)  
**Area**: Federated Learning / Parameter-Efficient Fine-Tuning / LoRA  
**Keywords**: Federated Learning, LoRA, Procrustes Alignment, Rotational Invariance, Subspace Alignment

## TL;DR
This paper identifies that the true "enemy" of naive factor-wise averaging in federated LoRA is the latent subspace misalignment caused by rotational invariance. It proposes that each client solves for a rotation matrix $R_i^t$ via orthogonal Procrustes to align $A,B$ factors before aggregation. Both theoretical and experimental results demonstrate significant reduction in aggregation error without increasing communication overhead.

## Background & Motivation

**Background**: LoRA represents weight updates as $\Delta W = BA$, with $B\in\mathbb{R}^{d\times r}, A\in\mathbb{R}^{r\times d}$, greatly reducing parameter count, making it a natural vehicle for LLM federated fine-tuning (FedIT, FFA-LoRA, FlexLoRA, etc. are all based on this).

**Limitations of Prior Work**: Ideally, aggregation should be $\Delta W_{ideal}=\tfrac1N\sum B_i A_i$, but this rank is generally $>r$, making it impossible to maintain low rank. The fallback, factor-wise averaging $\Delta W_{naive}=\bar B \bar A$, introduces cross terms $B_i A_j$, causing instability. Existing solutions take three paths: parameter freezing (FFA-LoRA, RoLoRA) sacrifices expressiveness; SVD projection (FlexLoRA) is computationally expensive; high-communication residual transmission (FedEx-LoRA) defeats LoRA's purpose.

**Key Challenge**: Existing analyses focus only on the algebraic reason of "operator non-commutativity," overlooking the **rotational invariance** inherent in LoRA factorization—i.e., for any orthogonal $R\in\mathbb{R}^{r\times r}$, $(B_i R)(R^\top A_i)=B_i A_i$. This means semantically equivalent updates may be represented in different latent subspaces, and naive averaging causes these misaligned subspaces to "destructively interfere," leading to errors greater than those from algebraic terms.

**Goal**: Explicitly eliminate the underestimated "rotational error" without increasing communication, freezing parameters, or performing high-dimensional SVD.

**Key Insight**: Since rotation $R$ does not change semantics, actively choose one to align all client factors to a common reference $R_i^t$. This $R$ has $r(r-1)/2$ degrees of freedom on the orthogonal group, sufficient for subspace alignment, and can be efficiently solved in closed form via Procrustes.

**Core Idea**: Use the previous round's global $\bar A^{t-1}/\bar B^{t-1}$ as reference; each client solves a Procrustes problem to obtain rotation $R_i^t$, alternately aligning $A$ or $B$, and finally interpolates a "soft rotation" to avoid excessive noise from early references.

## Method

### Overall Architecture
For each communication round $t$: ① Server broadcasts $(\bar A^{t-1}, \bar B^{t-1})$; ② Clients locally train to obtain $(A_i^t, B_i^t)$; ③ If $t$ is odd, align $A$, if even, align $B$, solving Procrustes for $R_i^{t,*}$; ④ Use $\lambda$ to interpolate between the identity and $R_i^{t,*}$ for a soft rotation $R_{i,\text{soft}}^t$; ⑤ Apply $\tilde A_i^t=(R_{i,\text{soft}}^t)^\top A_i^t,\;\tilde B_i^t=B_i^t R_{i,\text{soft}}^t$ before uploading for aggregation.

### Key Designs

1. **Orthogonal Procrustes Alignment (Alternating Factor Alignment)**:

    - **Function**: Rotates each client's LoRA factors into the global reference subspace, eliminating rotational ambiguity in the factorization.
    - **Mechanism**: For odd rounds, solve $\min_{R}\|R^\top A_i^t - A_{ref}\|_F^2,\;\text{s.t.}\;R^\top R=I,\det R>0$, a classic Procrustes problem. For correlation matrix $M=A_{ref}(A_i^t)^\top$, perform SVD $M=U\Sigma V^\top$ and obtain closed-form $R_i^{t,*}=V\cdot\text{diag}(1,\dots,1,\det(UV^\top))\cdot U^\top$. For even rounds, align $B$ with $M=(B_{ref})^\top B_i^t$. Computational complexity is $\mathcal{O}(d r^2+r^3)$, much less than FlexLoRA's $\mathcal{O}(d^3)$.
    - **Design Motivation**: The authors prove (Theorem 4.1) that scalar scaling has only 1 degree of freedom and cannot eliminate subspace misalignment; unconstrained invertible matrices are ill-conditioned. Orthogonal matrices optimally balance flexibility ($r(r-1)/2$ degrees) and well-conditioning.

2. **Alternating Alignment (Alternating $A$ and $B$)**:

    - **Function**: Prevents uncontrolled drift of the other factor when repeatedly aligning the same one.
    - **Mechanism**: Odd rounds fix $A$'s semantics to $A_{ref}$, $B$ compensates; even rounds vice versa. Only one SVD per round, but both factors are alternately "calibrated" globally.
    - **Design Motivation**: Ablation shows aligning only $B$ significantly drops performance (SST-2: 0.879 vs 0.954), because $B$'s initial norm is small and early alignment signals are weak; alternating ensures both factors' subspaces are regularly calibrated and mutually constrained.

3. **Soft Rotation Interpolation (Soft Rotation)**:

    - **Function**: Early in training, reference noise is high; hard rotation causes excessive correction.
    - **Mechanism**: Construct $R'=(1-\lambda)I+\lambda R_i^{t,*}$, then project back to the orthogonal group to get $R_{i,\text{soft}}^t$; $\lambda=0$ degenerates to FedIT, $\lambda=1$ is hard Procrustes. Lemma A.1 proves $\|R_{\text{soft}}-I\|_F\le 2\lambda\|R-I\|_F$, i.e., the correction magnitude is linearly bounded by $\lambda$.
    - **Design Motivation**: Early global models are not yet converged; forcefully pulling clients may disrupt personalized convergence. $\lambda\in[0.2,0.8]$ consistently outperforms hard alignment in experiments, with optimal values typically in 0.4-0.6.

### Loss & Training
The standard FedIT training process is retained, with the rotation step inserted before client upload. The paper provides convergence analysis under non-convexity (Theorem 4.4), decomposing error into initial gap + accumulated aggregation error $\|E^t\|_F^2$ + $\mathcal{O}(\eta)$; Theorem 4.8 further proves the post-alignment error bound is strictly tighter than naive, with tightness gain $\Gamma(\lambda)=(c_0-\tfrac{4\sqrt\tau\kappa\eta G_B}{\delta_A})\lambda - 4\kappa^2\lambda^2\tau$, giving a feasible range for $\lambda$.

## Key Experimental Results

### Main Results
Experiments on RoBERTa-Large with five GLUE tasks, rank=4, and three client scales $N\in\{3,10,50\}$.

| Task/Scale | FedIT | FFA-LoRA | RoLoRA | FedRot-LoRA |
|--------|------|------|------|------|
| MNLI ($N=3$) | 0.866 | 0.862 | 0.868 | **0.876** |
| RTE ($N=3$) | 0.840 | 0.830 | 0.854 | **0.868** |
| GLUE Avg ($N=50$) | 0.768 | 0.772 | 0.824 | **0.873** |
| GSM8K (Llama 3-8B) | 0.429 | 0.436 | 0.344 | **0.444** |
| HumanEval pass@1 | 0.288 | 0.385 | 0.295 | **0.409** |

Aggregation error reduction is especially significant: on MNLI, FedIT error is $3.98\times10^{-3}$, while FedRot-LoRA is only $1.48\times10^{-4}$, an order of magnitude lower.

### Ablation Study

| Configuration | MNLI Acc |
|------|---------|
| No Alignment (FedIT) | 0.866 |
| Random Rotation | 0.318 |
| Scalar Scaling Alignment | 0.865 |
| Align $A$ Only | 0.861 |
| Align $B$ Only | 0.862 |
| Alternate $A/B$ (Full) | **0.876** |
| reference = $W^{t-2}$ | 0.866 |
| reference = $W^{t-1}$ (default) | **0.876** |

### Key Findings
- Scalar scaling (1D rotation) is almost ineffective in high-dimensional LoRA—proof shows that for rank>1, subspace-level (orthogonal) alignment is necessary, not just norm adjustment.
- Random rotation drops performance to 0.318, indicating that only meaningful directional alignment is effective; this rules out the trivial explanation that "FedRot-LoRA's improvement is just due to added randomness."
- The more heterogeneous (Dirichlet $h=0.5$, client number $N=50$), the greater the advantage; for near IID ($h=100$), baselines also perform well, but FedRot-LoRA still leads by 1-2 points.

## Highlights & Insights
- **Reframing the root cause from "algebraic trick" to "geometric invariance"**: Previous solutions remedied the algebraic term $(B_i-B_j)(A_i-A_j)$, while this paper identifies the true disruptor as relative rotation of latent subspaces. This reframing shifts solutions from "freezing/projection" to "alignment," an elegant conceptual advance.
- **Procrustes is a perfect tool for federated LoRA aggregation**—closed-form solution, $\mathcal{O}(r^3)$ complexity, preserves orthogonality and semantics, no extra communication, meeting almost all desiderata at once.
- **Soft rotation $\lambda$ is a crucial detail**: Hard alignment can hurt in some settings; the authors use $\lambda\in[0,1]$ interpolation to control alignment strength, engineering the theoretically "optimal orthogonal matrix" for practical use. This "conservative early, confident later" approach can transfer to other federated/distributed schemes.

## Limitations & Future Work
- The paper assumes the reference uses the previous global model; if client selection is uneven or communication drops occur, $W^{t-1}$ may be outdated. Experiments show $W^{t-2}$ leads to performance drop.
- $\lambda$ is a hyperparameter; the feasible range depends on constants $c_0,\delta_A,\kappa,\tau$ that are hard to estimate, so grid search is needed in practice.
- Extreme scales with rank>24 or $N>50$ are not validated—at very high rank, SVD's $\mathcal{O}(r^3)$ cost becomes significant.
- No direct comparison with FedSA-LoRA or FedEx-LoRA on generative tasks; Appendix Table 7 only provides a summary comparison.

## Related Work & Insights
- **vs FedIT**: FedIT is the baseline, directly averaging factors; adding rotational alignment reduces aggregation error by an order of magnitude, improving nearly all tasks.
- **vs FFA-LoRA/RoLoRA**: These freeze one factor to achieve linear aggregation, essentially "bypassing" rotational ambiguity; this paper "actively resolves" it, preserving parameter space. FedRot-LoRA leads RoLoRA by 5 points at $N=50$.
- **vs FlexLoRA**: FlexLoRA aggregates in full-parameter space then projects back to low rank via SVD, costing $\mathcal{O}(d^3)$ and being numerically unstable; this paper's SVD is only on $r\times r$, much cheaper.
- **Insights**: Any low-rank/decomposition-based distributed learning (e.g., federated PCA, federated matrix factorization, federated diffusion adapter) may suffer from "decomposition invariance causing destructive interference," for which orthogonal Procrustes is a recommended first tool.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The observation that "rotational invariance is the true root cause" is sharp, unifying previous scattered attempts under a clear geometric framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five GLUE tasks × three scales × five ranks + GSM8K + HumanEval + multiple ablations + scalar scaling/random rotation controls, very comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Figures 1-2 illustrate the motivation intuitively; theoretical section Theorem 4.4 + 4.8 with Corollary 4.9 gives feasible domain for $\lambda$, structure is clear.
- Value: ⭐⭐⭐⭐ Can be directly used in existing federated LoRA frameworks by simply replacing the aggregation step; very friendly for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA](../../NeurIPS2025/model_compression/robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)
- [\[ICML 2026\] Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning](task-driven_subspace_decomposition_for_knowledge_sharing_and_isolation_in_lora-b.md)
- [\[ACL 2026\] LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging](../../ACL2026/model_compression/lora_on_the_go_instance-level_dynamic_lora_selection_and_merging.md)
- [\[CVPR 2026\] Bilevel Layer-Positioning LoRA for Real Image Dehazing](../../CVPR2026/model_compression/bilevel_layer-positioning_lora_for_real_image_dehazing.md)
- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](../../ICLR2026/model_compression/ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)

</div>

<!-- RELATED:END -->
