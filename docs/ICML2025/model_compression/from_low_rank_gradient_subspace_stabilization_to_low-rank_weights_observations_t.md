---
title: >-
  [Paper Note] From Low Rank Gradient Subspace Stabilization to Low-Rank Weights: Observations, Theories, and Applications
description: >-
  [ICML2025][Model Compression][low-rank compression] By revealing the low-rank convergence differences across various LLM weight matrices through Hessian spectral analysis, this work proposes WeLore—a non-uniform low-rank decomposition method that unifiedly addresses both model compression and parameter-efficient fine-tuning.
tags:
  - "ICML2025"
  - "Model Compression"
  - "low-rank compression"
  - "gradient subspace"
  - "Hessian analysis"
  - "parameter-efficient fine-tuning"
  - "LLM compression"
  - "SVD decomposition"
date: 2026-05-08
content_hash: 0e8981fb32e09aa9
---

# From Low Rank Gradient Subspace Stabilization to Low-Rank Weights: Observations, Theories, and Applications

**Conference**: ICML2025  
**arXiv**: [2407.11239](https://arxiv.org/abs/2407.11239)  
**Code**: [VITA-Group/WeLore](https://github.com/VITA-Group/WeLore)  
**Area**: Model Compression  
**Keywords**: low-rank compression, gradient subspace, Hessian analysis, parameter-efficient fine-tuning, LLM compression, SVD decomposition  
**Authors**: Ajay Jaiswal, Yifan Wang, Lu Yin, Shiwei Liu, Runjin Chen, Jiawei Zhao, Ananth Grama, Yuandong Tian, Zhangyang Wang

## TL;DR

By revealing the low-rank convergence differences across various LLM weight matrices through Hessian spectral analysis, this work proposes WeLore—a non-uniform low-rank decomposition method that unifiedly addresses both model compression and parameter-efficient fine-tuning.

## Background & Motivation

LLM weight matrices often exhibit low-rank structures after pre-training, which opens opportunities for compression and efficient inference. However, existing low-rank compression methods almost exclusively apply **uniform rank reduction** across all layers, ignoring the fact that the degree of low-rankness varies significantly across different components and depths.

The core starting point of this paper is: **Why do weights become low-rank? Why do different components differ in their degree of low-rankness?** Instead of explaining this from the perspective of data manifolds or regularization, the authors approach it from the perspective of **gradient subspace stabilization**. Utilizing Hessian eigenspectrum analysis, they establish a theoretical chain of "gradient dynamics $\rightarrow$ emergence of low-rank structures", which in turn guides non-uniform compression and selective fine-tuning.

## Method

### 1. Theoretical Framework: Hessian Eigenspace and Gradient Alignment

**Theorem 2.1 (Hessian Stabilization)**: Under standard assumptions (Hessian Lipschitz continuity, KŁ conditions, and uniform spectral gap), as SGD training progresses, both the eigenvalues and eigenspaces of the Hessian $H_t = \nabla^2 L(W_t)$ converge. The upper bound of the step-to-step variation is:

$$\|\Delta H_t\| \le \eta L_H \|\nabla L(W_t)\| \le \frac{\eta L_H C}{t^{\theta/(1-\theta)}}$$

where $\theta > 1/2$ guarantees series convergence.

**Theorem 2.2 (Gradient-Hessian Alignment)**: The gradient vector asymptotically aligns with the dominant eigenspace $U_t$ of the Hessian:

$$\lim_{t\to\infty} \frac{\|(I - U_t U_t^\top) G_t\|}{\|G_t\|} = 0$$

Intuitive understanding: Component vectors in the non-dominant subspace shrink exponentially at a rate of $(1 - \eta\gamma/2)$ (where $\gamma$ is the spectral gap); consequently, the gradient eventually concentrates on a small number of dominant directions.

### 2. Key Empirical Findings

Through analysis of Hessian spectra and gradient subspace similarity on LLaMA-2 7B and LLaMA-130M, two core observations are derived:

**By Component Type**:

- **Low-Rank Comfortable (LRC)**: `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.o_proj`, `mlp.gate_proj` $\rightarrow$ clear Hessian spectral gaps, fast stabilizing gradient subspaces, and heavy-tailed singular value distributions.
- **Non-Low-Rank Comfortable (N-LRC)**: `mlp.up_proj`, `mlp.down_proj`, `self_attn.v_proj` $\rightarrow$ flat Hessian spectra, diffused gradients, and uniform singular value distributions.

**By Layer Depth**:

- Early and late layers $\rightarrow$ high degree of low-rankness (simple input representations / gradients dominated by the loss function).
- Middle layers $\rightarrow$ lower degree of low-rankness (multi-layer feature mixing, softmax/LayerNorm attenuation).

### 3. WeLore-COMP: Non-Uniform Low-Rank Compression

Given a global normalized singular value threshold $k$, components with normalized singular values $\ge k$ are retained for each weight matrix $W_l$, yielding a compressed rank of $r_l = \text{sum}(\mathcal{S}_{W_l} \ge k)$. The threshold is determined via linear search to satisfy the target Effective Rank Reduction (ERR):

$$\frac{\sum_l \text{sum}(\mathcal{S}_{W_l} < k)}{\sum_l \text{len}(\mathcal{S}_{W_l})} \approx \text{ERR}$$

Accordingly, all layers are categorized into LRC ($r_l < 0.5 \times \text{rank}(W_l)$, which can be heavily compressed) and N-LRC (retained at full rank or minimally reduced). The weights of the LRC layers are replaced by the product form of $A_l \in \mathbb{R}^{m \times r}$ and $B_l \in \mathbb{R}^{r \times n}$.

Method characteristics: **data-agnostic** (requires no calibration set), **one-shot** (requires no iterative optimization), and employs only one global hyperparameter $k$.

### 4. WeLore-PEFT: Selective Fine-Tuning

Core strategy: In the compressed model, only the LRC layers are fine-tuned via backpropagation, while the N-LRC layers are frozen. The theoretical rationale is that LRC layers possess rich gradient signals and stabilized subspaces, carrying the primary learning capacity.

Since LRC layers are stored in low-rank format $(A, B)$, both the gradients and optimizer states are automatically low-rank, significantly reducing GPU memory. Experiments show that fine-tuning only the LRC layers (approx. 35% trainable parameters) matches or even exceeds full fine-tuning performance.

## Training & Experimental Settings

- **Models**: LLaMA-2 7B/13B, Mistral-7B; LLaMA-130M used for pre-training analysis (C4 dataset, 25K steps, Adam)
- **Compression Evaluation**: C4 validation set perplexity, with ERR ranging from 10% to 50%
- **Fine-Tuning Settings**: C4 dataset, sequence length 1024, 0.7M tokens, all methods sharing identical hyperparameters
- **Downstream Tasks**: CommonsenseQA, SVAMP, BoolQ, CoinFlip, BigBench (Object Tracking), StrategyQA
- **Real-world Task Evaluation**: Factoid-QA, multi-turn dialogue, context summarization
- **Baselines**: Uniform rank reduction, OWL reduction, SVD-LLM, ASVD; fine-tuning baselines include LoRA, GaLore

## Main Results

### Compression Performance (WeLore-COMP)

| Model | ERR | Uniform PPL | WeLore PPL |
|------|-----|------------|------------|
| LLaMA-2 7B | 10% | 10.58 | **7.13** |
| LLaMA-2 7B | 30% | 91.99 | **14.41** |
| LLaMA-2 7B | 50% | NaN | **1836.62** |
| LLaMA-2 13B | 30% | 13.99 | **8.66** |
| LLaMA-2 13B | 40% | 1178.03 | **24.92** |

- At 30% ERR, WeLore-COMP outperforms Uniform by ~6.4× (LLaMA-2 7B).
- Performance further improves when combined with ASVD: at 50% ERR, PPL drops from 1836 to 14.76.

### Fine-Tuning Recovery (WeLore-PEFT)

| Model | ERR | Post-Compression PPL | WeLore-PEFT PPL |
|------|-----|-----------|----------------|
| LLaMA-2 7B | 30% | 14.41 | **8.18** |
| LLaMA-2 7B | 50% | 1836.62 | **11.87** |
| LLaMA-2 13B | 50% | 1142.53 | **11.40** |
| Mistral-7B | 30% | 30.69 | **9.71** |

### Downstream Tasks (50% Compressed LLaMA-2 7B)

| Method | CommonsenseQA | BoolQ | CoinFlip | BigBench |
|------|--------------|-------|----------|----------|
| Dense Full FT | 77.05 | 88.19 | 75.00 | 83.74 |
| WeLore-COMP + LoRA | 35.38 | 75.48 | 50.67 | 54.02 |
| WeLore-COMP + GaLore | 35.12 | 71.55 | 47.67 | 58.98 |
| **WeLore-COMP + WeLore-PEFT** | **70.52** | **80.38** | **94.67** | **87.80** |

WeLore-PEFT even outperforms Dense Full FT on CoinFlip and BigBench, and significantly surpasses LoRA/GaLore at the same compression rate.

### Efficiency Comparison (50% Compressed LLaMA-2 7B)

- Trainable parameters: Only ~35% (vs. Full Fine-Tuning)
- Throughput: ~3× speedup
- GPU memory: ~40% savings

## Limitations & Future Work

1. **Performance Drop under High Compression Rates**: At 50% ERR, the compressed PPL is extremely high (>1000). Although WeLore-PEFT can restore it to ~12, a gap remains compared to the original PPL (~7); rates exceeding 50% are basically unusable.
2. **Strong Theoretical Assumptions**: Assumptions such as the KŁ conditions, uniform spectral gap, and architecture invertibility may not strictly hold in actual Transformers.
3. **Validation Limited to Language Models**: All experiments are restricted to the LLaMA/Mistral series, without involving visual or multimodal architectures.
4. **Combination with Quantization Not Fully Explored**: Low-rank compression is highly complementary to quantization (e.g., in combination with GPTQ), but this paper does not investigate it deeply.
5. **Hard Threshold for LRC/N-LRC Classification**: Defining the boundary purely at 0.5 × rank might be too coarse; a progressive approach could be superior.
6. **SVD Computational Overhead**: Although it is a one-shot process, the up-front cost of performing full SVD on all layers is non-negligible (especially for 13B+ parameters).

## Key Points for Reproducibility

- Code is open-sourced: [VITA-Group/WeLore](https://github.com/VITA-Group/WeLore)
- The core hyperparameter involves only one global threshold $k$, which is determined via a linear search over `np.linspace(0, 1, 0.005)`.
- The compression workflow is data-agnostic, requiring no calibration set.
- Fine-tuning uses the standard C4 dataset, 0.7M tokens, with all baselines sharing identical hyperparameters.
- Model weights are obtained directly from public pretrained HuggingFace checkpoints.

## Personal Review

**Strengths**: The greatest contribution of this paper is the establishment of the theoretical chain of "Hessian spectral gap $\rightarrow$ gradient subspace stabilization $\rightarrow$ emergence of low-rank structures", elevating low-rank compression from being purely empirical to theoretically guided. The distinction between LRC and N-LRC is not only utilized for compression but also yields a novel PEFT strategy—selectively fine-tuning layers with high learning capacity, instead of adding adapters to all layers as in LoRA. The experimental design is solid, with rigorous control variables (identical hyperparameters, identical token budget) and broad downstream task coverage.

**Weaknesses**: A gap still exists between theory and empirical findings—the theory relies on continuous optimization assumptions, whereas actual LLMs are trained with Adam and various engineering tricks. Furthermore, the practical value of compression rates above 50% is limited, and within the truly competitive range (10%-30%), the improvement over simpler methods is not overwhelmingly large. The astonishing performance of PEFT on CoinFlip (94.67 vs. Dense 75.00) warrants a deeper investigation into whether benchmark-specific factors are at play.

Overall, this is a work with clear theoretical motivation, simple methods, and thorough experiments. Although its limitations are obvious under extreme compression rates, its "layer selection based on gradients" approach offers valuable inspiration for future designs of low-rank compression and PEFT.

## Rating
- Novelty: ⭐⭐⭐⭐ (Explains the emergence of low-rankness from a gradient-Hessian perspective to guide compression, offering a unique angle)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Evaluates across multiple models, tasks, and baselines with strict control variables)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with logical transitions from theory to empirics and applications)
- Value: ⭐⭐⭐⭐ (The concept of unifying compression with PEFT holds practical significance; open-sourced code)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GoRA: Gradient-Driven Adaptive Low Rank Adaptation](../../NeurIPS2025/model_compression/gora_gradient-driven_adaptive_low_rank_adaptation.md)
- [\[ACL 2025\] CoLA: Collaborative Low-Rank Adaptation](../../ACL2025/model_compression/cola_collaborative_low-rank_adaptation.md)
- [\[ACL 2025\] BeamLoRA: Beam-Constraint Low-Rank Adaptation](../../ACL2025/model_compression/beamlora_beam_constraint_lora.md)
- [\[ACL 2025\] Low-Rank Interconnected Adaptation across Layers](../../ACL2025/model_compression/low-rank_interconnected_adaptation_across_layers.md)
- [\[ICCV 2025\] Beyond Low-Rank Tuning: Model Prior-Guided Rank Allocation for Effective Transfer in Low-Data and Large-Gap Regimes](../../ICCV2025/model_compression/beyond_low-rank_tuning_model_prior-guided_rank_allocation_for_effective_transfer.md)

</div>

<!-- RELATED:END -->
