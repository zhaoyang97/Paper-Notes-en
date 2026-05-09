---
title: >-
  [Paper Note] Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix
description: >-
  [NeurIPS 2025][LLM Safety][Differential Privacy] This paper proposes FedASK, a framework that employs a **two-stage sketching pipeline** (inspired by randomized SVD) to, for the first time under differential privacy, enable **simultaneous effective updates of both low-rank matrices A and B** in federated LoRA, achieving up to 11.5% improvement on MMLU and 46% on GSM8K over baselines on Llama-2 7B/13B.
tags:
  - NeurIPS 2025
  - LLM Safety
  - Differential Privacy
  - Federated Learning
  - LoRA
  - Low-Rank Adaptation
  - LLM Fine-Tuning
date: 2026-05-08
content_hash: 5eff369fc766dcde
---

# Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix

**Conference**: NeurIPS 2025
**arXiv**: [2507.09990](https://arxiv.org/abs/2507.09990)
**Code**: [GitHub](https://github.com/FLEECERmw/PrivacyFedLLM)
**Area**: AI Security
**Keywords**: Differential Privacy, Federated Learning, LoRA, Low-Rank Adaptation, LLM Fine-Tuning

## TL;DR

This paper proposes FedASK, a framework that employs a **two-stage sketching pipeline** (inspired by randomized SVD) to, for the first time under differential privacy, enable **simultaneous effective updates of both low-rank matrices A and B** in federated LoRA, achieving up to 11.5% improvement on MMLU and 46% on GSM8K over baselines on Llama-2 7B/13B.

## Background & Motivation

**Background**: Federated Learning (FL) combined with LoRA has become the dominant paradigm for distributed fine-tuning of large language models (LLMs). LoRA efficiently adapts models by training low-rank matrices $A \in \mathbb{R}^{r \times n}$, $B \in \mathbb{R}^{m \times r}$ (where $r \ll \min(m,n)$), with the update $\Delta W = BA$.

**Limitations of Prior Work**: Applying differential privacy (DP) to federated LoRA faces a **fundamental dilemma**:
   - **Noising both matrices → noise amplification**: When DP noise is independently added to the gradients of A and B, the noise undergoes **quadratic amplification** in the product $\Delta W = BA$—producing a dominant term $\sigma^4 C^4 d_l^2 r$ in the expected noise power.
   - **Fixing one matrix → reduced learning capacity**: Existing methods (e.g., FFA-LoRA) fix A and train only B, avoiding quadratic noise but confining updates to a fixed subspace.

**Key Challenge**: A fundamental tension between privacy protection and model learning capacity—adding noise protects privacy but amplifies it, while fixing a matrix eliminates noise but sacrifices expressiveness.

**Goal**: Design a federated LoRA framework that effectively updates both A and B simultaneously under strong DP guarantees, balancing privacy, learning capacity, and communication efficiency.

**Key Insight**: Inspired by randomized SVD, the paper designs a two-stage projection pipeline in which clients transmit compressed representations rather than full matrices; the server then exactly reconstructs the global update from the privatized compressed representations via SVD and distributes it back to A and B.

**Core Idea**: DP-SGD is applied locally only to B (avoiding quadratic noise), while server-side SVD decomposition redistributes the learned knowledge to both global A and B—achieving privacy and dual-matrix updates simultaneously.

## Method

### Overall Architecture

The core of FedASK is a **two-stage sketching pipeline**, where each communication round involves two client–server interactions:

**Stage 1 (Random Subspace Sketching)**:
1. The client locally trains $B_k^t, A_k^t$.
2. Using a shared random projection matrix $\Omega \in \mathbb{R}^{n \times (r+p)}$, compute $Y_k^{proj} = B_k^t(A_k^t \Omega)$.
3. Upload $Y_k^{proj}$ to the server.
4. The server aggregates and applies QR decomposition to obtain an orthonormal basis $Q$.

**Stage 2 (Global Alignment Projection)**:
1. The client receives $Q$ and computes $\tilde{Y}_k^{proj} = (A_k^t)^\top((B_k^t)^\top Q)$.
2. Upload $\tilde{Y}_k^{proj}$ to the server.
3. The server aggregates and applies SVD decomposition.
4. Update global parameters: $B^t = QU\Sigma^{1/2}$, $A^t = \Sigma^{1/2}V^\top$.

### Key Designs

#### 1. Two-Stage Sketching (Algorithm 1)

- **Function**: Exactly recovers the aggregated LoRA product $\frac{1}{K}\sum_k B_k A_k$ via two rounds of compressed projection.
- **Core Insight**: Although directly averaging $A, B$ introduces cross-term error ($\frac{1}{K}\sum B_k A_k \neq \frac{1}{K}\sum B_k \cdot \frac{1}{K}\sum A_k$), projecting first to a low-dimensional space and then recovering via SVD enables **exact aggregation**.
- **Design Motivation**: Borrowing from randomized SVD—first capture the column space via random projection, then perform exact decomposition within that space.

#### 2. DP Integration Strategy

- **Function**: In DP mode, only B undergoes DP-SGD locally (A remains fixed at the value synchronized from the previous global round).
- **Core Formula**:
$$B_k^{\tau+1} = B_k^\tau - \frac{\gamma\alpha}{r}\left(\frac{\partial l}{\partial W_k^\tau} / \max\left(1, \frac{\|\partial l / \partial W_k^\tau\|_2}{C}\right) + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I})\right)(A^{t-1})^T$$
- **Key**: Although only B is trained locally, the global SVD **redistributes** the privatized information to both matrices A and B—realizing a global update to A via $A^t = \Sigma^{1/2}V^\top$.

#### 3. Noise Analysis (Lemma 1)

- **Noise of standard DP-LoRA**: $\mathbb{E}[\|\Delta W_{\text{noise}}\|_F^2] \approx \underbrace{\eta^2 \sigma^2 C^2 d_l r(\|A\|_F^2 + \|B\|_F^2)}_{\text{linear term}} + \underbrace{\eta^4 \sigma^4 C^4 d_l^2 r}_{\text{quadratic term (dominant)}}$
- **FedASK**: Contains only the linear noise term (the catastrophic quadratic term is eliminated).
- **SNR degradation**: Standard method $1/\sigma^4$; FedASK $1/\sigma^2$.

### Theoretical Guarantees

- **Theorem 1 (Privacy Guarantee)**: FedASK satisfies $(\epsilon, \delta)$-DP with noise variance $\sigma^2 = \mathcal{O}\left(\frac{q_D^2 \cdot m \cdot q_K \cdot T \cdot \ln(2/\delta) \cdot \ln(2Tq_K/\delta)}{\epsilon^2 \cdot K}\right)$.
- **Theorem 2 (Exact Aggregation)**: When the over-sketching parameter satisfies $p \geq d_B - r + 2$, $\|\Delta W^t - \frac{1}{K}\sum_k B_k A_k\|_F = 0$.

## Key Experimental Results

### Main Results: Llama-2-7B (MMLU/DROP/HumanEval)

| Task | Privacy Budget | FedASK | FedAvg | FFA-LoRA | FedSA-LoRA | FedProx | Scaffold |
|------|---------------|--------|--------|----------|------------|---------|----------|
| MMLU | Non-Private | **46.15** | 45.13 | 45.98 | 45.19 | 44.98 | 45.65 |
| MMLU | $\epsilon=1$ | **45.80** | 42.07 | 42.76 | 42.90 | 41.99 | 43.41 |
| MMLU | $\epsilon=3$ | **46.25** | 41.49 | 42.72 | 41.13 | 43.17 | 42.47 |
| DROP | $\epsilon=1$ | **31.23** | 29.55 | 29.10 | 31.04 | 29.51 | 29.66 |
| HumanEval | $\epsilon=1$ | **15.24** | 12.80 | 12.20 | 13.41 | 12.20 | 9.76 |

### Llama-2-13B (GSM8K/MATH)

| Task | Privacy Budget | FedASK | FedAvg | FFA-LoRA | FedSA-LoRA |
|------|---------------|--------|--------|----------|------------|
| GSM8K | Non-Private | **50.0** | 48.5 | 48.4 | 47.2 |
| GSM8K | $\epsilon=1$ | **22.7** | 15.5 | 14.2 | 12.2 |
| GSM8K | $\epsilon=3$ | **24.8** | 16.5 | 20.0 | 20.2 |
| GSM8K | $\epsilon=6$ | **27.7** | 19.3 | 20.2 | 17.3 |
| MATH | $\epsilon=1$ | **6.9** | 5.2 | 5.8 | 5.6 |

**On GSM8K at $\epsilon=1$: FedASK (22.7) vs. FFA-LoRA (14.2) → 46% gain!**

### Data Heterogeneity Experiments (Llama-2-7B, $\epsilon=3$)

| Task | Data Distribution | FedASK | FedAvg | FFA-LoRA |
|------|------------------|--------|--------|----------|
| MMLU | IID | **46.25** | 41.49 | 42.72 |
| MMLU | Dir(0.1) | **46.04** | 42.69 | 42.54 |
| MMLU | Dir(0.5) | **45.95** | 42.11 | 41.46 |

### Ablation Study

| Variable | Key Findings |
|----------|-------------|
| Over-sketching $p$ | $p=2$–$4$ suffices to achieve near-exact aggregation |
| Communication cost | Same order as FFA-LoRA: $O(Kd_lr)$ |
| Server memory | $O(d_l r)$, on par with baselines |
| DP noise can improve performance | Acts as implicit regularization under certain conditions |

### Key Findings

1. **Large advantage under DP**: The tighter the privacy budget (smaller $\epsilon$), the greater FedASK's relative advantage—46% lead on GSM8K at $\epsilon=1$.
2. **Also optimal in the non-private setting**: Even without DP, FedASK outperforms FedAvg due to exact aggregation.
3. **Strong robustness**: Consistently leads under both IID and non-IID (Dir(0.1)–Dir(1.0)) distributions.
4. **Communication-efficient**: The two-stage design introduces no additional communication overhead.

## Highlights & Insights

1. **Elegantly resolves the fundamental DP-LoRA dilemma**: Perturbing only one matrix locally avoids quadratic noise, while global SVD restores dual-matrix updates.
2. **Federated application of randomized SVD**: A creative adaptation of a classical numerical linear algebra tool for federated aggregation.
3. **Exact aggregation guarantee (Theorem 2)**: This is zero-error aggregation, not an approximation—unique in the federated LoRA literature.
4. **Validated on 13B models**: One of the few works to conduct DP federated fine-tuning on genuinely large models.
5. **Unexpected regularization effect of DP noise**: An interesting observation—under certain conditions, adding noise actually improves performance.

## Limitations & Future Work

1. **Two communication rounds per round**: Each round requires two client–server interactions, doubling latency.
2. **Stage 1 projection $Y_k^{proj}$ is not DP-protected**: Potential privacy leakage risk (though the paper argues safety via post-processing).
3. **Local A is frozen**: Although global A is updated via SVD, A remains fixed during local training—potentially limiting local adaptation capacity.
4. **Integration with other PEFT methods unexplored**: E.g., AdaLoRA, DoRA.
5. **Computational overhead**: Additional server-side SVD and QR decomposition, as well as two projection computations on the client side.

## Related Work & Insights

- **Relationship to FFA-LoRA**: FFA-LoRA fixes A and trains only B; FedASK also trains only B locally but updates A globally—the key distinction is SVD-based redistribution.
- **Relationship to FLoRA**: FLoRA achieves exact aggregation via stacking but incurs $O(K^2 d_l r)$ communication; FedASK maintains $O(K d_l r)$.
- **Inspiration**: The combination of randomized SVD and federated learning may be broadly applicable in other settings (e.g., matrix factorization in federated recommender systems).

## Rating

⭐⭐⭐⭐ (4/5)
- The method is elegantly designed with complete theoretical guarantees, validated on genuinely large models.
- The practical latency of two communication rounds per round and the constraint of frozen local A are the primary weaknesses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](on_the_sample_complexity_of_differentially_private_policy_optimization.md)
- [\[NeurIPS 2025\] Demystifying Language Model Forgetting with Low-Rank Example Associations](demystifying_language_model_forgetting_with_low-rank_example_associations.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](../../ICLR2026/llm_safety/heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)
- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)

</div>

<!-- RELATED:END -->
