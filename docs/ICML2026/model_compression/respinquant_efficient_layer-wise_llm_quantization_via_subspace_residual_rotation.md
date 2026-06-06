---
title: >-
  [Paper Note] ReSpinQuant: Efficient Layer-Wise LLM Quantization via Subspace Residual Rotation Approximation
description: >-
  [ICML 2026][Model Compression][LLM Quantization] ReSpinQuant simultaneously preserves the advantages of "global rotations being fusible with weights" and "inter-layer rotations being adaptable to per-layer outliers" in l…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "LLM Quantization"
  - "Rotation Quantization"
  - "Inter-layer Rotation"
  - "Subspace Approximation"
  - "Residual Alignment"
date: 2026-05-08
content_hash: 65322482686e3e9c
---

# ReSpinQuant: Efficient Layer-Wise LLM Quantization via Subspace Residual Rotation Approximation

**Conference**: ICML 2026  
**arXiv**: [2604.11080](https://arxiv.org/abs/2604.11080)  
**Code**: To be confirmed  
**Area**: Model Compression  
**Keywords**: LLM Quantization, Rotation Quantization, Inter-layer Rotation, Subspace Approximation, Residual Alignment  

## TL;DR
ReSpinQuant simultaneously preserves the advantages of "global rotations being fusible with weights" and "inter-layer rotations being adaptable to per-layer outliers" in low-bit LLM PTQ. This is achieved by replacing the non-eliminable rotation transition matrix $\mathbf{T}=\mathbf{R}_{out}\mathbf{R}_{in}^{\top}$ at residual connections with a rank-$r\!\approx\!32$ subspace orthogonal approximation. The online overhead increases by only $\sim0.2\%$, while outperforming SpinQuant and FlatQuant on W4A4/W3A3 settings.

## Background & Motivation

**Background**: The mainstream path for low-bit LLM PTQ has evolved from weight-only quantization (GPTQ, AWQ) to "weight + activation" joint quantization (W4A4 or even W3A3). Tools based on orthogonal rotations are critical for handling activation outliers. QuaRot uses random Hadamard matrices to distribute outlier energy across all dimensions, while SpinQuant further makes rotation matrices learnable, constrained to the orthogonal manifold using a Cayley optimizer.

**Limitations of Prior Work**: Current rotation strategies are divided into two categories, both with significant drawbacks. **Global Rotation** uses a shared $\mathbf{R}$ for the entire model, allowing the activation rotation $\mathbf{X}\mathbf{R}$ to be pre-fused into the weights $\mathbf{W}\mathbf{R}$ with zero inference overhead; however, using a single basis cannot accommodate heterogeneous outlier distributions across different layers. **Inter-layer Rotation** (FlatQuant, OSTQuant, ButterflyQuant, ParoQuant) assigns independent $\mathbf{R}^i$ to each layer, offering higher precision, but the inconsistent bases between adjacent layers prevent pre-fusion of activation rotations, requiring online computation. FlatQuant's Kronecker form still requires $\mathcal{O}(D^{1.5})$ MACs, and ButterflyQuant requires $\mathcal{O}(D\log D)$. To suppress online overhead, these methods must use structured matrices (scaling, Butterfly, Kronecker), sacrificing expressiveness.

**Key Challenge**: At residual connections, the transition matrix $\mathbf{T}=\mathbf{R}_{out}\mathbf{R}_{in}^{\top}$ in $\tilde{x}_{out}=\mathbf{R}_{out}\mathbf{R}_{in}^{\top}\tilde{x}_{in}+\mathbf{R}_{out}\,\text{Block}(\mathbf{R}_{in}^{\top}\tilde{x}_{in})$ only reduces to the identity matrix and disappears if $\mathbf{R}_{in}=\mathbf{R}_{out}$. This is the root cause of the trade-off between "expressiveness" and "online overhead."

**Goal**: (1) Retain dense, layer-independent rotation matrices to maximize expressiveness; (2) Fuse all rotations within attention/FFN blocks into weights offline; (3) Pay a negligible online cost for basis transition at residual connections.

**Key Insight**: The authors observe that rotations $\mathbf{R}$ trained with a Cayley optimizer starting from Hadamard initialization remain very close to the initial values—the Frobenius norm shift is small, and the cosine similarity with the initial value remains near 1. A direct implication is $\mathbf{T}=\mathbf{R}_{out}\mathbf{R}_{in}^{\top}\approx \mathbf{H}\mathbf{H}^{\top}=\mathbf{I}$, meaning $\Delta\mathbf{T}=\mathbf{T}-\mathbf{I}$ is a low-rank, diagonally dominant "small perturbation."

**Core Idea**: Given that the energy of $\Delta\mathbf{T}$ is concentrated in a tiny subspace, the method performs a dense orthogonal rotation correction within that subspace while applying an identity mapping to the orthogonal complement, reducing the complexity of $\mathcal{O}(D^2)$ dense alignment to $\mathcal{O}(rD)$ low-rank alignment.

## Method

### Overall Architecture
ReSpinQuant independently learns four dense $D\times D$ orthogonal matrices for each transformer layer: $\mathbf{R}_1^i$ rotates MHSA inputs and FFN outputs, $\mathbf{R}_2^i$ rotates FFN inputs and MHSA outputs, $\mathbf{R}_3^i$ acts inside the attention (e.g., after V projection), and $\mathbf{R}_4, \mathbf{R}_5$ are implemented via Fast Hadamard Transform to handle specific activation paths in the SpinQuant protocol. All $\mathbf{R}$ matrices are learnable during the optimization phase, constrained to the Stiefel manifold using a Cayley optimizer to minimize $\|\mathbf{Y}-Q(\tilde{\mathbf{X}})Q(\tilde{\mathbf{W}})^{\top}\|_F^2$. During inference, $\mathbf{R}^i$ within MHSA/FFN are absorbed into adjacent weights (e.g., $\tilde{\mathbf{W}}_v=\mathbf{R}_1^{i\top}\mathbf{W}_v\mathbf{R}_3$, $\tilde{\mathbf{W}}_o=\mathbf{R}_3^{i\top}\mathbf{W}_o\mathbf{R}_2$), while basis mismatches on cross-layer residual paths are processed online by a low-rank subspace residual correction module. The framework can be summarized as: "huge parameter space $\mathcal{O}(L\cdot D^2)$ during training, only $\mathcal{O}(L\cdot rD)$ online parameters during inference."

### Key Designs

1.  **Dense Inter-layer Rotation + Full Offline Weight Fusion**:
    - **Function**: Allows each layer to use independent dense $\mathbf{R}^i\in\mathbb{R}^{D\times D}$ to handle its specific outlier patterns without explicitly invoking these matrices during inference.
    - **Mechanism**: On every attention/FFN path, $\mathbf{R}^i$ is used to rotate activations, and its transpose is used for the input side of the next weight or the output side of the current weight. These rotations mathematically cancel out ($\mathbf{R}\mathbf{R}^{\top}=\mathbf{I}$), allowing all $\mathbf{R}$ coefficients to be pre-multiplied into $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o, \mathbf{W}_{up}, \mathbf{W}_{down}$ before quantization. The learnable parameters during training reach $1091.0\text{M}$ (approx. 63× more than SpinQuant), while 100% of these parameters are hidden in weights during inference, leaving only $8.4\text{M}$ online parameters.
    - **Design Motivation**: Previous inter-layer solutions were forced to use structured matrices (e.g., $\mathcal{O}(D)$ scaling in OSTQuant, Kronecker in FlatQuant) because dense matrices could not be absorbed. This "large training, small inference" design breaks that trade-off.

2.  **Low-Rank Residual Approximation via Empirical Observation**:
    - **Function**: Replaces the $D\times D$ dense matrix $\mathbf{T}=\mathbf{R}_{out}\mathbf{R}_{in}^{\top}$ at residual connections with a rank-$r$ subspace rotation plus an identity complement, reducing online complexity from $\mathcal{O}(D^2)$ to $\mathcal{O}(rD)$.
    - **Mechanism**: SVD is performed on $\Delta\mathbf{T}=\mathbf{T}-\mathbf{I}$ to obtain $\mathbf{Q}\in\mathbb{R}^{D\times r}$ from the top $r$ left singular vectors. The full transition matrix is projected onto this subspace to get $\mathbf{T}_{\text{sub}}=\mathbf{Q}^{\top}\mathbf{T}\mathbf{Q}$. For strict orthogonality, Polar decomposition $\hat{\mathbf{R}}_{\text{sub}}=\mathbf{U}_{sub}\mathbf{V}_{sub}^{\top}$ is applied to ensure $\hat{\mathbf{R}}_{\text{sub}}\in SO(r)$. The final approximation is $\hat{\mathbf{T}}=\mathbf{I}+\mathbf{Q}(\hat{\mathbf{R}}_{\text{sub}}-\mathbf{I}_r)\mathbf{Q}^{\top}$, which remains in $SO(D)$.
    - **Design Motivation**: Empirical results show that the learned dense $\mathbf{R}$ hardly deviates from the Hadamard initialization. Visualizing $\mathbf{R}_1^{\top}\mathbf{R}_2$ sub-blocks shows they are diagonally dominant and sparse, implying basis mismatch occurs only in a few principal directions; cutting to lower rank preserves accuracy.

3.  **Lightweight Online Residual Path**:
    - **Function**: Completes residual alignment in three steps without explicitly constructing the $D\times D$ $\hat{\mathbf{T}}$ matrix.
    - **Mechanism**: The update formula is executed as a three-step pipeline: projection $y=\mathbf{Q}^{\top}\tilde{x}_{in}\in\mathbb{R}^r$, subspace transformation $z=\mathbf{M}y$ (where $\mathbf{M}=\hat{\mathbf{R}}_{\text{sub}}-\mathbf{I}_r$ merges the identity term and addition), and back-projection with residual addition $\tilde{x}_{out}=\tilde{x}_{in}+\mathbf{Q}z$. No $D\times D$ matrix multiplications occur on this path.
    - **Design Motivation**: Aligns with the LoRA-style paradigm of "projecting to a principal subspace—performing non-trivial transformations in small space—then projecting back," but applied for inference-time alignment rather than training-time updates.

### Loss & Training
Rotation matrices maintain strict orthogonality under Cayley optimizer constraints, ensuring the learning process stays on the Stiefel manifold. Calibration uses 800 segments from WikiText-2, with the loss being standard cross-entropy (consistent with SpinQuant, not introducing KL divergence or layer-wise loss, unlike OSTQuant/FlatQuant), decoupling "architectural innovation" from "training objective innovation." After rotation optimization, weights are quantized via GPTQ with fixed clipping. The default subspace rank is $r=32$, chosen as the Pareto elbow from rank-PPL curves in W3A3 settings. The full pipeline takes approximately 42 minutes for LLaMA-3 8B on a single H100.

## Key Experimental Results

### Main Results
Quantization performed on LLaMA-2 7B/13B, LLaMA-3 8B, and LLaMA-3.2 1B/3B for W4A4 and W3A3.

| Model / Setting | Metric | RTN | QuaRot | SpinQuant | FlatQuant | ReSpinQuant |
|-------------|------|-----|--------|-----------|-----------|-------------|
| LLaMA-3 8B / W4A4 | PPL ↓ | 219.82 | 7.82 | 7.50 | 7.73 | **7.24** |
| LLaMA-3 8B / W4A4 | 0-shot Avg ↑ | 36.74 | 62.90 | 64.53 | 62.72 | **64.65** |
| LLaMA-3 8B / W3A3 | PPL ↓ | 77055 | 98.04 | 15.07 | 133.52 | **13.09** |
| LLaMA-3.2 1B / W3A3 | PPL ↓ | 115358 | 812.46 | 69.70 | 543.66 | **49.90** |
| LLaMA-3.2 3B / W4A4 | PPL ↓ (FP16=7.81) | 266.80 | 9.99 | 9.46 | 9.57 | **9.06** |

### Ablation Study (LLaMA-3 8B, W3A3, varying rank $r$)

| Configuration | PPL ↓ | 0-shot Avg ↑ | Description |
|------|-------|--------------|------|
| $r=0$ (No correction) | 20.03 | 46.94 | Residual path uses identity mapping |
| $r=8$ | 14.20 | 49.77 | Significant improvement with only 8 principal directions |
| $r=32$ (Default) | 13.09 | 50.74 | Online MAC only 32.3M (0.2% of total) |
| $r=128$ | 12.80 | 50.80 | Diminishing returns with larger rank |
| $r=4096$ (Full rank) | 12.52 | 51.22 | Upper bound reference |

### Key Findings
- Information in the residual transition matrix $\mathbf{T}$ is highly concentrated: moving from $r=0\to 8$ cuts PPL by 30%, whereas $r=32\to 4096$ only shifts PPL by 0.57, validating the "low-rank inter-layer difference" hypothesis.
- Training costs are manageable: LLaMA-3 8B calibration takes 42 minutes (compared to 17m for SpinQuant, 45m for FlatQuant). The 63× parameter space expansion yields substantial accuracy gains with good cost-performance.
- End-to-end latency nearly matches SpinQuant: On H100 for LLaMA-3 8B with batch=16, TTIT increases only from 160.95 ms to 163.81 ms (+1.7%), confirming negligible online overhead.
- Quantized Large Models > Full Precision Small Models: ReSpinQuant's W4A4 LLaMA-3.2 3B achieves 9.06 PPL, outperforming FP16 LLaMA-3.2 1B (9.76 PPL) with lower memory footprint.

## Highlights & Insights
- "Train-Large, Infer-Small" is a paradigm contribution: No parsimony in the training stage ($\mathcal{O}(L\cdot D^2)$ dense), while using mathematical fusion and low-rank approximation to cut online costs to 0.2%. This can be ported to other "complex training, constrained inference" scenarios like LoRA-style fusion or dynamic sparse MoE routing.
- The use of SVD + Polar decomposition for "low-rank + orthogonality" is a versatile tool: SVD alone destroys the orthogonality required for quantization error bounds; Polar decomposition $\mathbf{U}\mathbf{V}^{\top}$ pulls it back to $SO(r)$, and thus $SO(D)$.
- The narrative is elegant: starts from the observation that the Cayley optimizer keeps $\mathbf{R}$ near Hadamard, deduces $\mathbf{T}\approx\mathbf{I}$, proves $\Delta\mathbf{T}$ is low-rank, and quantifies principal directions.
- Decoupling "training parameters vs. online parameters" provides a new design degree of freedom: the 63× expansion in training parameters is "free," suggesting future quantization research might adopt more aggressive Split-Architectures.

## Limitations & Future Work
- The authors explicitly optimized the "structure" but not the training objective; KL divergence or layer-wise losses used in OSTQuant/FlatQuant remain unintegrated.
- Experiments are limited by H100 availability; the largest model tested is 13B. Evidence for whether $\mathbf{R}$ stays close to Hadamard in 70B+ models is lacking.
- Lack of specialized low-bit hardware kernel testing. While TTIT matches SpinQuant, throughput might differ with optimized W4A4/W3A3 kernels.
- The default $r=32$ is tuned for LLaMA; cross-architecture (Mistral, Qwen, MoE) optimality may vary; adaptive $r$ is suggested for production.
- The core assumption is Cayley optimization maintaining $\mathbf{R}\approx\mathbf{H}$. If the initialization is random orthogonal or Householder parameterization is used, $\Delta\mathbf{T}$ may no longer be low-rank.

## Related Work & Insights
- **vs SpinQuant**: Both use Cayley optimizer for $\mathbf{R}$, but SpinQuant uses a global $\mathbf{R}$ for zero overhead. ReSpinQuant uses dense layer-wise $\mathbf{R}$ with subspace approximation to reclaim fusibility, winning on both accuracy and expression.
- **vs FlatQuant**: FlatQuant's inter-layer affine uses Kronecker to compress parameters to 5.8M but has 198.1M online MACs ($\mathcal{O}(D^{1.5})$). ReSpinQuant expands to 1091M training parameters but requires only 8.4M parameters / 32.3M MACs for inference.
- **vs OSTQuant**: OSTQuant's inter-layer component is restricted to diagonal scaling ($\mathcal{O}(D)$), limiting expressiveness. ReSpinQuant retains full dense rotation and relies on fusion rather than structuring for efficiency.
- **vs ButterflyQuant / ParoQuant**: These structure matrices into $\mathcal{O}(D\log D)$. ReSpinQuant imposes no structural constraints, only a low-rank constraint on the "non-trivial components."
- **vs QuaRot**: QuaRot uses fixed random Hadamard with zero training cost but low accuracy potential. ReSpinQuant learns from the Hadamard starting point, significantly leading in precision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](../../ICLR2026/model_compression/paroquant_pairwise_rotation_quantization_for_efficient_reasoning_llm_inference.md)
- [\[ICML 2026\] RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression](rq-moe_residual_quantization_via_mixture_of_experts_for_efficient_input-dependen.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](../../ACL2026/model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[ICML 2026\] RaBiT: Residual-Aware Binarization Training for Accurate and Efficient LLMs](rabit_residual-aware_binarization_training_for_accurate_and_efficient_llms.md)
- [\[ICML 2026\] WUSH: Near-Optimal Adaptive Transforms for LLM Quantization](wush_near-optimal_adaptive_transforms_for_llm_quantization.md)

</div>

<!-- RELATED:END -->
