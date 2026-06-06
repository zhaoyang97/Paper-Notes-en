---
title: >-
  [Paper Note] RaBiT: Residual-Aware Binarization Training for Accurate and Efficient LLMs
description: >-
  [ICML 2026][Model Compression][Residual Binarization] This paper addresses a failure mode in residual binarized LLMs where parallel binary paths learn redundant features—a phenomenon the authors name "inter-path adaptati…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Residual Binarization"
  - "Quantization-Aware Training (QAT)"
  - "LLM"
  - "Inter-path Adaptation"
  - "Matmul-free Inference"
date: 2026-05-08
content_hash: a5592af1421c4288
---

# RaBiT: Residual-Aware Binarization Training for Accurate and Efficient LLMs

**Conference**: ICML 2026  
**arXiv**: [2602.05367](https://arxiv.org/abs/2602.05367)  
**Code**: To be confirmed  
**Area**: Model Compression / LLM Quantization / Binarization  
**Keywords**: Residual Binarization, Quantization-Aware Training (QAT), LLM, Inter-path Adaptation, Matmul-free Inference  

## TL;DR
This paper addresses a failure mode in residual binarized LLMs where parallel binary paths learn redundant features—a phenomenon the authors name "inter-path adaptation." It proposes RaBiT, which derives all binary paths online from a single shared full-precision weight coupled with function-aware initialization. This structurally enforces a residual hierarchy, allowing 2-bit Llama2-7B under a matmul-free architecture to surpass strong VQ baselines for the first time (Wiki2 PPL 5.78 vs QTIP 5.86) while achieving a 4.49× inference speedup.

## Background & Motivation

**Background**: For deploying LLMs at extreme compression ratios, 4-bit quantization (GPTQ, AWQ) has become the industry standard, but research is pushing toward 2-bit. The 2-bit landscape has two main routes: (i) Vector Quantization (VQ) (AQLM, QuIP#, QTIP), which preserves higher accuracy through lookup tables or complex rotations at the cost of high hardware overhead; (ii) Residual Binarization, which stacks multiple $\{\pm1\}$ binary layers to naturally support extremely efficient matmul-free execution (using only additions/subtractions). The core promise of residual binarization is that subsequent paths compensate for errors in preceding paths, achieving expressiveness near multi-bit levels at binarized costs.

**Limitations of Prior Work**: Although residual structures appear promising, they are consistently unstable in QAT. The authors identify a failure mode named "inter-path adaptation," a specific manifestation of Hinton’s 2012 "feature co-adaptation" in residual binarization. Standard QAT applies the same global gradient to all parallel paths simultaneously, driving each path to learn almost identical features while competing to lower the same global loss. Consequently, the error compensation hierarchy is destroyed, severely weakening the model's expressibility.

**Key Challenge**: MSE decomposition dictates that paths must be negatively correlated, and the second path must actively align with the residual of the first to effectively utilize multi-path capacity. However, the symmetric structure and shared gradients in standard QAT leave paths nearly independent with near-zero correlation. Stacking paths merely increases parameters without providing error compensation. Prior works (DB-LLM, MBOK) rely on heuristic constraints (path freezing, mechanical splitting) to break this symmetry, which either sacrifice the joint optimization space or produce negative correlation without proper residual alignment.

**Goal**: (i) Provide formal diagnostic metrics for inter-path adaptation; (ii) Embed residual hierarchy into the training loop algorithmically rather than heuristically; (iii) Solve the high sensitivity of 2-bit QAT to initialization.

**Key Insight**: Since the root cause is that paths maintain independent latent weights while sharing global gradients, the approach should be reversed: maintain only one full-precision weight $\mathbf{W}_{\mathrm{FP}}$ as an anchor. In each step, derive the first path and its residual online, then derive the second path from that residual. This makes the "second path compensating the first" a hard structural constraint rather than a soft loss-based encouragement.

**Core Idea**: Use a shared full-precision weight to derive all binary paths in sequence online (coupled forward), ensuring the residual hierarchy is automatically reconstructed at every step. Use Iterative Residual SVID + I/O channel importance preconditioning to provide a stable "function-preserving" instead of "weight-preserving" initialization.

## Method

### Overall Architecture
RaBiT addresses the training pathology of 2-bit residual binarization through two categories of design: (a) Instead of maintaining independent latent weights for each path, it keeps a single shared $\mathbf{W}_{\mathrm{FP}}$, deriving $\mathbf{B}_1=\text{sign}(\mathbf{W}_{\mathrm{FP}})$, $\mathbf{R}_1=\mathbf{W}_{\mathrm{FP}}-\hat{\mathbf{W}}_1$, and $\mathbf{B}_2=\text{sign}(\mathbf{R}_1)$ at each forward step; (b) Each path retains independent learnable per-channel scales $\{\mathbf{g}_i,\mathbf{h}_i\}$ to preserve capacity; (c) Iterative Residual SVID + I/O importance preconditioning is used for function-aware initialization; (d) At inference, the trained $\mathbf{B}_i$ are frozen and $\mathbf{W}_{\mathrm{FP}}$ is discarded, returning to the original parallel matmul-free architecture.

The basic binary block is written as $\hat{\mathbf{W}}=\mathbf{g}\odot\mathbf{B}\odot\mathbf{h}$, where $\mathbf{B}\in\{-1,+1\}^{d_{\text{out}}\times d_{\text{in}}}$, $\mathbf{g}\in\mathbb{R}^{d_{\text{out}}}$, and $\mathbf{h}\in\mathbb{R}^{d_{\text{in}}}$. The matrix-vector product $\mathbf{y}=\mathbf{g}\odot(\mathbf{B}(\mathbf{h}\odot\mathbf{x}))$ is implemented using only additions and subtractions. For 2-bit, $k=2$ such blocks are stacked.

### Key Designs

1.  **Shared FP Weights + Coupled Forward Pass**:
    - **Function**: Converts "residual compensation" from a loss preference to a hard structural constraint, fundamentally eliminating inter-path adaptation.
    - **Mechanism**: Only $\mathbf{W}_{\mathrm{FP}}$ is stored during training. Each step follows three stages: (1) $\mathbf{B}_1=\text{sign}(\mathbf{W}_{\mathrm{FP}})$, forming $\hat{\mathbf{W}}_1=\mathbf{g}_1\odot\mathbf{B}_1\odot\mathbf{h}_1$; (2) Residual $\mathbf{R}_1=\mathbf{W}_{\mathrm{FP}}-\hat{\mathbf{W}}_1$; (3) $\mathbf{B}_2=\text{sign}(\mathbf{R}_1)$. The final effective weight is $\hat{\mathbf{W}}^{(2)}=\hat{\mathbf{W}}_1+\hat{\mathbf{W}}_2$. Backpropagation uses a single STE to flow $\nabla_{\hat{\mathbf{W}}^{(2)}}\mathcal{L}=(\partial\mathcal{L}/\partial\mathbf{Y})\mathbf{X}^{\top}$ directly to $\mathbf{W}_{\mathrm{FP}}$. Scale vectors $\{\mathbf{g}_i,\mathbf{h}_i\}$ are updated via standard chain rule, treating dynamically derived $\mathbf{B}_i$ as constants.
    - **Design Motivation**: Previous methods used independent latent weights, failing to distinguish between "main path" and "compensation path" structurally. This led to redundant features when updated by shared gradients. By making the second path a function of the first path's residual, $\mathbf{B}_2$ is structurally forced to follow $\mathbf{R}_1$. Additionally, using one set of FP weights halves the optimizer state memory (e.g., Adam's momentum/variance), saving scarce VRAM during LLM fine-tuning.

2.  **Function-Aware Initialization (Iterative Residual SVID + I/O Importance Preconditioning)**:
    - **Function**: Addresses the sensitivity of 2-bit QAT to the starting point by prioritizing functional reconstruction over weight reconstruction.
    - **Mechanism**: First, the original weights are preconditioned using input activation magnitudes $\mathbf{s}_{\text{in}}$ and output gradient magnitudes $\mathbf{s}_{\text{out}}$ from a calibration set: $\mathbf{W}'=\mathbf{s}_{\text{out}}^{\alpha_{\text{out}}}\odot\mathbf{W}_{\mathrm{FP}}\odot\mathbf{s}_{\text{in}}^{\alpha_{\text{in}}}$. This focuses decomposition on functionally sensitive channels. Then, Iterative Residual SVID refreshes $(\mathbf{B}_i,\mathbf{g}_i,\mathbf{h}_i)$ for each path over $T$ rounds in a Gauss-Seidel style: in each round, the contribution of other paths is subtracted from $\mathbf{W}'$, and SVID (magnitude decomposition based on rank-1 SVD) fits the remaining residual. Scales are finally mapped back to the original domain.
    - **Design Motivation**: Standard SVID is greedy—the first path monopolizes the optimal fit, pushing the residual structure into a poor local minimum. Iterative residuals and channel importance preconditioning combine to solve greedy coupling and the "uniform fitting" problem where functionally critical channels are ignored.

3.  **Inter-Path Adaptation Diagnostic**:
    - **Function**: Provides a quantitative diagnostic for "path co-adaptation."
    - **Mechanism**: For a 2-bit residual network $y_s=y_1+y_2$, MSE is expanded as $\text{MSE}(y_t,y_s)=C'+2\sigma_1\sigma_2\cdot\text{Corr}(y_1,y_2)$, with an equivalent perspective $\text{MSE}\approx\sigma_{R_1}^2+\sigma_{y_2}^2-2\sigma_{R_1}\sigma_{y_2}\cdot\text{Corr}(R_1,y_2)$, where $R_1=y_t-y_1$ is the functional residual of the first path. These yield two criteria: Corr$(y_1,y_2)$ should be sufficiently negative, and Corr$(R_1,y_2)$ should be sufficiently positive.
    - **Design Motivation**: This diagnostic reveals why prior works failed: standard QAT correlation is near 0 (no compensation); DB-LLM achieves mechanical negative correlation (-0.49) but poor residual alignment (0.26). Only RaBiT achieves both high negative correlation ($\approx$-0.35 to -0.50) and high residual alignment (0.58–0.65).

### Loss & Training
Total loss $\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{kl}}+\gamma\sum_i\mathcal{L}_{\text{inter},i}$, combining KL divergence distillation and intermediate layer MSE distillation. Training uses the Muon optimizer for 6 epochs on a 200M token calibration set (WikiText-2 + C4) with a 4096 context length.

## Key Experimental Results

### Main Results
Comparison with SOTA 2-bit methods on Llama2/3 and Gemma3.

| Model / Data | Metric | RaBiT (2-bit) | Prev. SOTA (2-bit) | FP16 Baseline |
| :--- | :--- | :--- | :--- | :--- |
| Llama2-7B Wiki2 | PPL ↓ | **5.78** | QTIP 5.86 / DBF 6.10 / MBOK 6.99 | 5.12 |
| Llama2-7B QA Avg | Acc ↑ | **61.51** | QTIP 58.97 / DBF 58.42 | 62.26 |
| Llama2-13B Wiki2 | PPL ↓ | 5.15 | QTIP 5.11 (Second best) | 4.57 |
| Llama3-8B Wiki2 | PPL ↓ | **7.34** | QTIP 7.52 / QuIP# 8.70 | 5.75 |
| Llama3-8B QA Avg | Acc ↑ | **64.13** | AQLM 64.12 / QTIP 63.88 | 68.66 |
| Gemma3-1B Wiki2 | PPL ↓ | **11.27** | QTIP 13.14 / DBF 13.28 | 9.80 |
| Llama2-13B Hard Task Avg | Acc ↑ | **27.14** | QTIP 25.38 | 29.27 |
| Llama2-7B Dec. Speedup | Speedup ↑ | **4.49×** | — | 1.00× |

### Ablation Study

| Configuration | Llama2-7B Wiki2 PPL ↓ | Description |
| :--- | :--- | :--- |
| Standard QAT (Indep. weights) | 6.55 | Baseline with severe inter-path adaptation |
| Standard QAT + Iterative SVID Init | 6.21 | Gains from initialization alone |
| Standard QAT + I/O Preconditioning | 6.31 | Function-aware preconditioning alone |
| Standard QAT + Combined Init | 6.18 | Synergy in initialization |
| Coupled QAT (Forward only) | 5.84 | Dominant gain from eliminating adaptation |
| **RaBiT (Full)** | **5.78** | Full proposed solution |

### Key Findings
- **Coupled training provides the largest gain**: Moving from Standard QAT (6.55) to Coupled QAT (5.84) by changing only the forward structure yields 0.71 PPL improvement, proving that inter-path adaptation is the primary bottleneck.
- **Synergy of two components**: Coupling and function-aware initialization contribute $\approx$0.7 and 0.4 PPL respectively; combined they reach 5.78, showing that structural optimization and starting point optimization are complementary.
- **Surpassing VQ**: On Llama2-7B, RaBiT (5.78 PPL) outperforms QTIP (5.86) while remaining matmul-free, with 4.49× speedup on RTX 4090.
- **Memory Reduction**: Maintaining a single $\mathbf{W}_{\mathrm{FP}}$ halves optimizer states, significantly reducing QAT memory pressure.

## Highlights & Insights
- Shifting "residual compensation" from a loss preference to a structural constraint is an elegant paradigm shift. This "chained derivation + shared anchor" approach could be transferred to MoE routing, multi-branch distillation, or low-rank adapters.
- The dual metrics (Corr$(y_1,y_2)$ and Corr$(R_1,y_2)$) provide high interpretability, revealing that mechanical negative correlation is "fake compensation." True compensation must involve residual alignment.
- Keeping scales $\{\mathbf{g}_i, \mathbf{h}_i\}$ learnable while deriving $\mathbf{B}_i$ online is a clever compromise between structural rigor and optimization flexibility.

## Limitations & Future Work
- **Gap in hard tasks**: A significant accuracy gap remains on complex reasoning tasks (e.g., Llama3-8B averages 25.12 vs 31.03), suggesting 2-bit quantization is not yet fully optimized for complex logic.
- **Instruction following (IFEval)**: Performance on IFEval (Llama3-8B: 15.42 vs baseline 32.51) shows that "format sensitivity" is severely damaged under binarization.
- **Scale limitations**: RaBiT slightly trailed QTIP on Llama2-13B. Higher weight-to-activation ratios in larger models may favor VQ.
- **Stability for $k \ge 3$**: The framework was validated for $k=2$ (2-bit); numerical stability and gradient clarity for $k \ge 3$ remain unexplored.

## Related Work & Insights
- **vs DB-LLM**: Heuristic splitting forces negative correlation but lacks residual alignment. This paper proves negative correlation $\neq$ compensation.
- **vs MBOK**: Uses path freezing to avoid co-adaptation, which limits optimization. RaBiT allows joint optimization through structural constraints.
- **vs VQ (QTIP/AQLM)**: These rely on hardware-intensive LUTs. RaBiT proves that matmul-free residual binarization can match or exceed VQ accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ReSpinQuant: Efficient Layer-Wise LLM Quantization via Subspace Residual Rotation Approximation](respinquant_efficient_layer-wise_llm_quantization_via_subspace_residual_rotation.md)
- [\[ICML 2026\] RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression](rq-moe_residual_quantization_via_mixture_of_experts_for_efficient_input-dependen.md)
- [\[ACL 2026\] CadLLM: Improving the Throughput of Diffusion-based LLMs via Training-Free Confidence-Aware Calibration](../../ACL2026/model_compression/improving_the_throughput_of_diffusion-based_large_language_models_via_a_training.md)
- [\[ICML 2026\] WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points](winq_accelerating_quantization-aware_training_of_language_models_around_saddle_p.md)
- [\[ACL 2026\] TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination](../../ACL2026/model_compression/tell-tale_task_efficient_llms_with_task_aware_layer_elimination.md)

</div>

<!-- RELATED:END -->
