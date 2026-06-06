---
title: >-
  [Paper Note] Efficient DP-SGD for LLMs with Randomized Clipping
description: >-
  [ICML 2026][LLM Safety][DP-SGD] This paper proposes DP-SGD-RC, which replaces the exact per-sample gradient norm calculation in DP-SGD with Hutchinson / Hutch++ randomized trace estimation. This reduces the clipping memo…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "DP-SGD"
  - "Randomized Clipping"
  - "Randomized Trace Estimation"
  - "Hutchinson"
  - "Long-Context LLM"
date: 2026-05-08
content_hash: 7bef936052cd1ec3
---

# Efficient DP-SGD for LLMs with Randomized Clipping

**Conference**: ICML 2026  
**arXiv**: [2605.24879](https://arxiv.org/abs/2605.24879)  
**Code**: None  
**Area**: LLM Security / Differential Privacy / Efficient Training  
**Keywords**: DP-SGD, Randomized Clipping, Randomized Trace Estimation, Hutchinson, Long-Context LLM

## TL;DR
This paper proposes DP-SGD-RC, which replaces the exact per-sample gradient norm calculation in DP-SGD with Hutchinson / Hutch++ randomized trace estimation. This reduces the clipping memory overhead for training long-context LLMs from $O(B\min\{T^2,d^2\})$ to $O(BkT+kp)$. A tight $f$-DP analysis based on a chi-square mixture envelope CDF is provided. Fine-tuning Llama-3.2-1B on long contexts maintains accuracy while reducing peak VRAM in the largest linear layers by approximately 40% and saving about 2× FLOPs.

## Background & Motivation

**Background**: DP-SGD is the de facto standard for providing provable privacy protection in LLM training. It adds two steps to standard SGD: per-sample gradient clipping and Gaussian noise addition. To avoid the astronomical memory overhead of $O(BLd^2)$ in naive implementations, the community mainly relies on Fast Gradient Clipping (FGC, $O(Bd^2)$) and Ghost Clipping (GC, $O(B)$ for linear-like layers) to keep the computational cost of DP-SGD close to non-private training.

**Limitations of Prior Work**: The aforementioned memory savings only hold for "non-sequential" inputs. For text inputs with sequence length $T$, the memory complexity of the best performers among FGC and GC degrades to $O(B\min\{d^2,T^2\})$. Front-line LLMs often have contexts of $O(100\text{K})$, and this "quadratic" overhead hits the limits of GPUs; even fine-tuning a 1B model with a 4K context becomes strained.

**Key Challenge**: The bottleneck of DP-SGD lies not in noise addition but in "calculating the exact per-sample gradient norm $\|A_i^\top G_i\|_F^2$." This step requires either explicit storage of gradients ($d^2$) or explicit storage of $AA^\top$ or $GG^\top$ ($T^2$). As long as "exact norms + strict clipping" are pursued, the quadratic term in sequence length cannot be eliminated.

**Goal**: To reduce the memory and computational overhead of long-context DP-SGD from $O(T^2/d^2)$ to the same order as non-private training, $O(BkT)$, without incurring significant privacy or utility penalties. This requires solving three problems simultaneously: (i) finding a mathematically equivalent but memory-efficient way to compute the norm; (ii) performing a new privacy analysis for "randomized" clipping; and (iii) integrating the analysis into a functional numerical accountant.

**Key Insight**: The authors rewrite the norm calculation as a trace estimation: $\|A^\top G\|_F^2 = \mathrm{trace}(G^\top AA^\top G)$. Once viewed as a trace, classical randomized trace estimators (Hutchinson, Hutch++) can be used to approximate it using only $k$ "matrix-vector products," compressing the storage to $O(k(T+d+p))$, where $k\approx 32$ is sufficient.

**Core Idea**: Replace the "exact norm + deterministic clipping" in DP-SGD with "randomized trace-estimated norm + same clipping rule," analyze the privacy loss of this "clipping scale randomization" in the form of $f$-DP, and finally output standard $(\varepsilon, \delta)$-DP using a PRV-based numerical accountant.

## Method

### Overall Architecture
DP-SGD-RC maintains the overall three-stage process of DP-SGD: **first backward pass to estimate norms → rescale per-sample losses by $\min(C/n_i, 1)$ → second backward pass + noise addition + optimizer step**—modifying only the "norm calculation" step. For each linear-like layer, a forward hook retrieves the activation tensor $A \in \mathbb{R}^{B \times T \times d}$, and a backward hook retrieves the output gradient $G \in \mathbb{R}^{B \times T \times p}$. A "norm estimation routine" is then called to return $\hat n_i^{(l)}$. The squared whole-model gradient norm for sample $i$, $n_i = \sum_l \hat n_i^{(l)}$, is obtained by summing across layers, and the loss is rewritten based on $\min(C/\sqrt{n_i}, 1)$. The rewritten loss undergoes a second backward pass to yield the aggregated gradient $\nabla$. After adding noise $\nabla + \sigma C \cdot \mathcal{N}(0, I)$, it is fed into Adam / SGD.

The implementation utilizes forward/backward hooks per layer, avoiding the need for $k+1$ forward passes required by methods like DP-SGD-JL. Total cost is 1 forward + 2 backward passes, matching FGC, but with significantly lower peak per-layer VRAM.

### Key Designs

1.  **Hutchinson Randomized Projection for Norm Estimation**:
    - **Function**: Replaces the exact per-layer norm $\|A^\top G\|_F^2$ with a randomized estimate requiring only $k$ projections from $\mathbb{R}^{p \to k}$, compressing intermediate storage per layer from $O(\min\{d^2, T^2\})$ to $O(k(T+d+p))$.
    - **Mechanism**: Noting that $\|A^\top G\|_F^2 = \mathrm{trace}(G^\top AA^\top G) = \mathrm{trace}(O)$, the Hutchinson estimator $\widehat{n} = \mathrm{trace}(P^\top OP) = \|(P^\top G^\top)^\top A\|_F^2$ is used, where $P \in \mathbb{R}^{p \times k}, P_{uv} \sim \mathcal{N}(0, 1/k)$. The implementation computes $Y = A^\top(GP)$ and then $\|Y\|_F^2$, without explicitly constructing $d \times p$ per-sample gradients or $T \times T$ intermediate matrices. Classical analysis shows $k = O(\log(1/\beta)/\alpha^2)$ provides $(1\pm\alpha)$ relative accuracy; $k=32$ is sufficient in experiments.
    - **Design Motivation**: DP-SGD requires the "norm" scalar, not the per-sample gradient itself. Since the scalar does not need to be exact, JL-type randomized projections can amortize the cost of "exact norm calculation." This eliminates quadratic terms for sequence length $T$ and layer widths $p, d$, making DP training of long-context LLMs comparable to non-private training for the first time.

2.  **Hutch++ Low-Error Variant with Head-Tail Decomposition**:
    - **Function**: Significantly reduces norm estimation variance for a fixed $k$, maintaining utility in the small $\varepsilon$ regime (where noise is high and sensitivity to norm error is greater).
    - **Mechanism**: Splits the spectrum of $O = (A^\top G)(A^\top G)^\top$ into "head + tail." A random $S$ estimates an orthogonal basis $Q$ for $\mathrm{Col}(OS)$ for a low-rank approximation $\mathrm{trace}(Q^\top OQ)$ to accurately calculate the "head." The Hutchinson estimator is then applied to the "tail" on the residual $(I - QQ^\top)$. The final estimate is $\|(QG^\top)A\|_F^2 + \|(PG^\top)A - (((PG^\top)A)Q)Q^\top\|_F^2$. Theoretically, this improves Hutchinson's $O(\log(1/\beta)/\alpha^2)$ to $O(\sqrt{\log(1/\beta)}/\alpha)$, approaching the lower bound for matrix-vector query models.
    - **Design Motivation**: At larger $d$, the privacy envelope functions of Hutch and Hutch++ nearly overlap (noise multipliers are similar). However, in extreme scenarios like the small BBC dataset at $\varepsilon=0.7$, Hutch++ acts as a "gentle regularizer" due to smaller variance, raising accuracy from 64.3% to 70.6%. The trade-off is 3× higher matrix-vector multiplication and QR decomposition costs.

3.  **$f$-DP Privacy Accounting via Chi-Square Mixture Envelope**:
    - **Function**: Provides a tight, numerically computable $(\varepsilon, \delta)$ guarantee for DP-SGD-RC with randomized clipping, allowing direct integration with PRV-style accountants and the Opacus ecosystem.
    - **Mechanism**: The single-step privacy analysis reduces to $T(Z, \mathcal{N}(Z/\sigma, 1) \| Z, \mathcal{N}(0, 1))$, where $Z = \|Q_0\|/R(Q_0)$ is the "true norm divided by the estimated norm." For Hutch, $Z^2(\lambda) \sim \|\lambda\|_1 / \sum_i \lambda_i \chi^2(k)$ depends on the eigenvalue spectrum $\lambda$ of the difference sample gradient. Using stochastic ordering and majorization tools, authors prove the envelope CDF on the simplex $\lambda \in \Delta^{d-1}$ has a 3-segment structure (equal-weight $\chi^2$ mixture at the left, two-element mixture $\frac{\lambda}{ik}\chi^2(ik) + \frac{1-\lambda}{jk}\chi^2(jk)$ in a narrow middle interval, and single $\chi^2$ at the right). An explicit binary search algorithm for $x_+ \in [1, 2]$ is provided. The accounting stage performs a Riemann–Stieltjes weighted integral of the Gaussian kernel over the envelope CDF to find $\alpha(t), \beta(t)$, followed by PRV/subsampling amplification/composition for the final $(\varepsilon, \delta)$.
    - **Design Motivation**: Randomized clipping is no longer a standard Gaussian mechanism with a fixed noise scale; traditional RDP/PRV accountants cannot be applied directly. By making the distribution of $Z$ explicit (envelope CDF), the problem reduces to calculating weighted $\Phi$ integrals over a known CDF, enabling stable integration into numerical accounting pipelines. The paper also identifies and corrects a bug in the 2003 Székely–Bakirov theorem regarding chi-square mixture envelopes.

### Loss & Training
The training objective is identical to DP-SGD: original task loss (categorical cross-entropy / NLL) plus $\ell_2$ clipping of per-sample gradients to threshold $C$, followed by isotropic Gaussian noise $\sigma C \cdot \mathcal{N}(0, I)$. Optimizers used are SGD or Adam. Experiments use $\delta \in \{10^{-5}, 10^{-6}\}, \varepsilon \in \{0.7, 2, 9\}, k=32$, with context lengths fixed at 4096, covering both full fine-tuning and LoRA modes.

## Key Experimental Results

### Main Results

Llama-3.2-1B Full Fine-Tuning (mean and std across 3 random seeds):

| Dataset (Task) | Metric | Non-private | DP-SGD (FGC) $\varepsilon=9$ | DP-SGD-RC ($k=32$) $\varepsilon=9$ | DP-SGD $\varepsilon=2$ | DP-SGD-RC $\varepsilon=2$ |
|---|---|---|---|---|---|---|
| BBC (Cls) | Acc ↑ | 95.20±0.51% | 96.33±0.59% | 96.40±0.22% | 94.06±0.12% | 95.60±0.37% |
| BillSum (Sum) | ROUGE-1 ↑ | 0.4928±0.0027 | 0.4882±0.0011 | 0.4864±0.0013 | 0.4831±0.0005 | 0.4796±0.0018 |
| HotpotQA (QA) | EM ↑ | 61.06±0.39% | 61.44±0.05% | 61.42±0.08% | 61.35±0.03% | 61.31±0.09% |

Results are similarly consistent for LoRA fine-tuning: BBC drops < 0.4%, BillSum ROUGE drops 0.005, matching the scale of full fine-tuning.

### Ablation Study

Hutch vs Hutch++ under low budget $\varepsilon=0.7, \delta=10^{-5}$ (BBC Full Fine-Tuning):

| Method | Proj. Dim $k$ | Noise Multiplier | Accuracy (%) |
|---|---|---|---|
| DP-SGD (FGC) | N/A | 4.073 | 67.07±5.26 |
| DP-SGD-RC w/ Hutch | 32 | 4.354 | 64.29±6.77 |
| DP-SGD-RC w/ Hutch++ | 32 | 4.354 | **70.59±3.49** |

Efficiency Ablation (Llama-3.2-1B max linear layer $8192\times 2048, T=4096, k=32$, relative to DP-SGD baseline):

| Metric | Hutch Savings | Hutch++ Savings |
|---|---|---|
| Peak VRAM (incl. input) | 39.18% | 38.57% |
| Peak VRAM (excl. input) | 99.22% | 97.65% |
| FLOPs (Max Layer / Min Layer) | 98.05% / 92.19% | 92.17% / 68.69% |
| Latency (A100 80GB) | ≈3× faster than Hutch++ | — |

### Key Findings
- **Norm estimation accuracy is not always better**: In the small $\varepsilon$ regime, Hutch++ is an order of magnitude more accurate than Hutch (Fig. 11). However, the authors argue that noisy norms act as a form of regularization, leading Hutch++ to outperform Hutch by 6 percentage points at $\varepsilon=0.7$. For $\varepsilon \ge 2, d \gtrsim 128$, their envelopes nearly overlap, making the faster Hutch more efficient.
- **The ceiling for memory savings is defined by the projection matrix itself**: When $k$ increases to 4096, the randomized projection matrix $P \in \mathbb{R}^{p \times k}$ consumes significant VRAM, nullifying savings. The authors note that $\{-1, +1\}$ 1-bit projection matrices (Achlioptas style) could reduce this further, but require re-proving privacy.
- **Larger layers benefit more**: The $8192 \times 2048$ layer achieves approx 40% peak VRAM savings, while a small $2048 \times 512$ layer sees only approx 15%. This implies that the method's advantages will be more pronounced in frontier-scale LLMs (larger $p, d, T$).

## Highlights & Insights
- **Repurposing "Per-sample Norm" as "Trace Estimation"**: Translating the engineering bottleneck of "per-sample gradient norm" into trace estimation allows the application of 30 years of randomized linear algebra tools. This perspective of mapping private ML sub-problems to linear algebra problems is highly reusable—e.g., for per-sample curvature, Fisher information, or influence functions.
- **Tight $f$-DP Analysis Template for Randomized Clipping**: The "envelope CDF + PRV numerical integration" workflow is generalizable. By replacing the distribution of $Z$, any mechanism where the noise is Gaussian but the scale is randomized can be integrated into existing accountants. This serves as a direct scaffold for future work like sketched-gradient or low-rank DP-SGD.
- **Correcting the 2003 Székely–Bakirov Bug**: In Appendix B.6, the authors use simulation counter-examples to show the old theorem for middle-interval envelope characterization is incorrect and provide the correct three-segment form. This is a byproduct for pure random variable theory, but will benefit any future work requiring "extremal distributions of chi-square convex combinations."

## Limitations & Future Work
- **Hutch++ Privacy Analysis Compromises**: To keep it tractable, the authors assume the attacker knows intermediate states of the head low-rank estimation. Thus, in practice (large $d$), the noise multiplier is conservative. Relaxing this might allow Hutch++ to win even at medium $\varepsilon$.
-  **Memory Gains Specific to Linear-like Layers**: While attention, convolution, and linear layers enjoy $O(k(T+d+p))$, RMSNorm, element-wise, or custom operators still follow the "explicit per-sample gradient" path. Future non-typical architectures (Mamba, MoE) may require reassessment.
- **Projection Matrix as Achilles' Heel**: At $k \ge 4096$, the random matrix itself saturates memory. The authors leave $\{\pm1\}$ 1-bit or sparse hashing projections to future work, as these require recalculating the envelope CDF.
- **Scale of Experiments**: Evaluation was limited to Llama-3.2-1B with 4096 context, not reaching "true long context" of 70B+ models or 100K context. The method's advantages should maximize when $T \gg d$, which remains to be verified at frontier scale.

## Related Work & Insights
- **vs DP-SGD-JL (Bu et al., 2021)**: DP-SGD-JL applies a JL projection $\mathbb{R}^{dp} \to \mathbb{R}^k$ to flattened gradients at $T=1$, requiring $k+1$ forward passes (JVP mode). This paper targets $T > 1$ sequential inputs, projecting on the factored form $A^\top G$ via $\mathbb{R}^{p \times d} \to \mathbb{R}^{k \times d}$ in only 1 forward + 2 backward passes. It can be seen as a strict generalization of DP-SGD-JL for sequential data and multiple output dimensions.
- **vs Fast Gradient Clipping (FGC, Lee & Kifer 2021)**: FGC avoids explicit per-sample gradients by rescaling loss but still materializes gradients per layer, resulting in $O(BTd^2)$ memory for long contexts. DP-SGD-RC eliminates the "per-layer materialization," removing both sequential and width quadratic terms.
- **vs Ghost Clipping (GC, Li et al. 2021) / Mixed-Ghost + Book-keeping (Bu et al. 2023)**: GC computes norms for linear-like layers at $O(BT^2)$, which is disadvantaged in long contexts. Mixed-Ghost chooses the cheaper of FGC/GC, remaining in the "exact norm" paradigm. DP-SGD-RC is an orthogonal direction—trading accuracy for memory—which can be stacked with engineering optimizations like Book-keeping to further reduce computation.
- **vs Low-rank DP-SGD (Yu et al. 2022, etc.)**: That line of work restricts private updates to low-rank subspaces to save noise budget and parameters. This paper leaves the rank of updates unchanged, optimizing only the "norm calculation" step. These can be combined (e.g., low-rank parameterization + randomized trace estimation for LoRA weight norms).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First method to remove $T^2/d^2$ terms in long-context DP-LLM training by mapping per-sample norm calculation to randomized trace estimation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Uses Llama-3.2-1B across 3 long-context tasks, multiple $\varepsilon$ levels, LoRA/full-tuning, and full VRAM/FLOPs/latency ablations. Only lacks frontier-scale verification.
- Writing Quality: ⭐⭐⭐⭐ Algorithm/Analysis/Accounting/Experiments are independent yet well-connected. Appendix corrects an old theorem bug. Privacy formulas are dense.
- Value: ⭐⭐⭐⭐⭐ Directly eases the engineering bottleneck for long-context DP-LLMs. The envelope-CDF + PRV template is highly reusable for future randomized DP mechanisms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Fast-MIA: Efficient and Scalable Membership Inference for LLMs](../../ACL2026/llm_safety/fast-mia_efficient_and_scalable_membership_inference_for_llms.md)
- [\[ICML 2026\] Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping](memory_as_a_markov_matrix_sample_efficient_knowledge_expansion_via_token-to-dict.md)
- [\[ICML 2026\] PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](pipesd_an_efficient_cloud-edge_collaborative_pipeline_inference_framework_with_s.md)
- [\[ICML 2026\] Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility](multilingual_unlearning_in_llms_transfer_dynamics_and_reversibility.md)
- [\[ICML 2026\] Gradient Transformer: Learning to Generate Updates for LLMs](gradient_transformer_learning_to_generate_updates_for_llms.md)

</div>

<!-- RELATED:END -->
