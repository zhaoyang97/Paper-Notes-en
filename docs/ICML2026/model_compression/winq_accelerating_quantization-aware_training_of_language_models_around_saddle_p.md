---
title: >-
  [Paper Note] WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points
description: >-
  [ICML2026][Model Compression][Quantization-Aware Training (QAT)] WinQ explains the slow convergence of low-bit LLM QAT as weight trajectories getting stuck near low-curvature saddle points. It utilizes periodic weight-qu…
tags:
  - "ICML2026"
  - "Model Compression"
  - "Quantization-Aware Training (QAT)"
  - "Low-bit LLMs"
  - "Hessian Spectrum"
  - "Saddle Point Optimization"
  - "Noise Injection"
date: 2026-05-08
content_hash: 59a973f79558c22a
---

# WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points

**Conference**: ICML2026  
**arXiv**: [2605.17471](https://arxiv.org/abs/2605.17471)  
**Code**: https://github.com/facebookresearch/WinQ  
**Area**: Model Compression / Low-bit Quantization / LLM Efficiency  
**Keywords**: Quantization-Aware Training (QAT), Low-bit LLMs, Hessian Spectrum, Saddle Point Optimization, Noise Injection  

## TL;DR
WinQ explains the slow convergence of low-bit LLM QAT as weight trajectories getting stuck near low-curvature saddle points. It utilizes periodic weight-quantization interpolation re-initialization coupled with gradient noise perturbations to accelerate 1-2 bit QAT by 1.5–4$\times$ with minimal computational overhead, improving perplexity and zero-shot accuracy across various LLaMA/Qwen configurations under equivalent training budgets.

## Background & Motivation
**Background**: Large Language Model (LLM) deployment increasingly relies on low-bit quantization. While post-training quantization (PTQ) maintains performance above 4 bits, it fails significantly at extreme precisions like 1-2 bits or 1.58 bits. Consequently, the mainstream approach shifts to quantization-aware training (QAT), which maintains full-precision latent weights during training while performing forward passes and gradient estimation based on quantized weights.

**Limitations of Prior Work**: Although QAT yields superior results, its cost is prohibitive. The paper notes that even 4-bit QAT training costs can approach 10% of full-precision pre-training; 1-bit QAT is even slower, often requiring training on billions of tokens to achieve usability. Prior methods such as ParetoQ and QuEST primarily modify quantization functions, Hadamard transforms, or gradient estimators, but fail to explain why low-bit QAT rapidly enters a plateau after the early stages of training.

**Key Challenge**: Low-bit quantization requires latent weights to be close to a discrete quantization grid, while optimization continues in a continuous weight space. The authors observe that the relative gradient norm decreases quickly despite insufficient loss reduction, suggesting the model is not simply lacking a high learning rate but is likely trapped in regions with weak local curvature. Hessian spectral analysis reveals that the eigenvalues of low-bit QAT are concentrated near 0, with both positive and negative values present—a classic sign of stagnation near flat saddle points.

**Goal**: The paper seeks to answer two questions: first, what is the underlying optimization cause of slow convergence in low-bit QAT; and second, can a training technique be designed that is quantizer-agnostic and computationally efficient to extract the model from these low-curvature zones?

**Key Insight**: Instead of designing complex quantizers to minimize quantization error, the authors treat QAT as a non-convex optimization problem and measure the spectral distribution of the loss Hessian. This perspective transforms the issue of "lower bit-width difficulty" into a measurable curvature problem: lower bit-widths result in smaller maximum Hessian eigenvalue magnitudes and a higher proportion of near-zero eigenvalues, leading to slower convergence.

**Core Idea**: Use periodic $W \leftarrow (1-\alpha)W+\alpha Q(W)$ resets to pull latent weights closer to the quantized weights and increase local curvature, while injecting noise $Q(W+U)$ at each step to perturb gradients and assist in escaping saddle points.

## Method
The WinQ method consists of two layers: "diagnosis" and "intervention." The diagnosis layer uses the Hessian spectrum to prove that slow convergence in low-bit QAT is a structural phenomenon, while the intervention layer translates this into two lightweight training operations: periodic weight interpolation re-initialization and step-wise noise injection.

### Overall Architecture
The input is an existing QAT pipeline: given full-precision latent weights $W$, a quantization function $Q(\cdot)$, a language model $f_W$, and training data. Standard QAT updates $W$ using the quantized $Q(W)$ for forward passes and gradient estimation (e.g., STE). WinQ does not replace the quantization function but wraps the training loop with two additions.

First, at each training step, Gaussian noise $U \sim \mathcal{N}(0, \sigma^2 I)$ is sampled. Gradients are calculated using the noisy quantized weights $Q(W+U)$ to update the original $W$. Second, every $K$ steps, the latent weights are reset via linear interpolation between the current latent weights and the quantized weights: $W \leftarrow (1-\alpha)W+\alpha Q(W)$. After training, final latent weights are quantized for inference as in standard QAT.

The authors also provide a version for Hadamard transforms. If a method quantizes $HW$, interpolation occurs in the Hadamard space and is mapped back: $W \leftarrow H^\top((1-\alpha)HW+\alpha Q(HW))$. This allows WinQ to be stacked on methods like QuEST that utilize rotations or transforms.

### Key Designs
1.  **Hessian Spectrum Diagnosis for Saddle-Point Stagnation**:
    *   **Function**: Explains why training loss and test performance stagnate quickly in low-bit QAT, rather than just providing an empirical acceleration trick.
    *   **Mechanism**: The authors estimate the eigenvalue distribution of the loss Hessian using stochastic Lanczos quadrature and Hessian-vector products. In 1-4 bit QAT, a massive proportion of eigenvalues cluster near 0 late in training, with both positive and negative values present. For 3-bit QAT, the probability mass near 0 increases from 7% at start to 41% at 80K steps; for 1-bit, this reaches 63%.
    *   **Design Motivation**: If gradient norms are small, convergence speed is dominated by local curvature. Smaller max eigenvalues at lower bits show the model resides in flatter regions, directly explaining the "lower-bit, slower-convergence" phenomenon.

2.  **Weight Interpolation Re-initialization**:
    *   **Function**: Periodically moves latent weights from flat regions to areas of higher curvature while reducing the distance between latent weights and the quantization grid.
    *   **Mechanism**: Executed every $K$ steps as $W \leftarrow (1-\alpha)W+\alpha Q(W)$. If local quantization points remain constant, this approximates a proximal update on $L_Q(W)+\frac{\gamma}{2}\|W-q\|^2$ with $q=Q(W)$ and $\alpha=\eta\gamma/(1+\eta\gamma)$. Experimentally, the max Hessian eigenvalue magnitude increases significantly after interpolation; in 2-bit QAT with $\alpha=0.4$, it increases from 1.64 to 3.09.
    *   **Design Motivation**: There is a natural disparity between latent weights and final quantized weights in QAT, which is exacerbated at low bits. Interpolation pulls weights toward the grid and alters the optimization trajectory without significantly changing the current loss, representing a cheap step with a direct impact on optimization geometry.

3.  **Noise-Injected Gradient Estimation**:
    *   **Function**: Introduces lightweight perturbations at each training step to facilitate escape from low-curvature stagnation zones near saddle points.
    *   **Mechanism**: Instead of using $Q(W)$, gradients are calculated on $Q(W+U)$ where $U \sim \mathcal{N}(0,\sigma^2I)$. Hessian analysis shows that appropriate noise leads to larger negative curvature and slightly larger gradient norms. In 2-bit QAT, $\sigma=0.001$ with $\alpha=0.6$ results in a max eigenvalue of 3.96, compared to 2.65 without noise.
    *   **Design Motivation**: In non-convex optimization, noisy SGD is frequently used to escape saddle points. WinQ applies this to quantized weights without increasing forward/backward passes, merely adding a single noise vector sampling.

### Loss & Training
WinQ does not modify the original language modeling objective, continuing with auto-regressive modeling on corpora like FineWebEdu. Experiments typically run for 20B tokens (~240K steps). Hyperparameters include re-initialization interval $K \in \{40K, 60K, 80K\}$, interpolation coefficient $\alpha \in [0.1, 0.6]$, and noise standard deviation $\sigma \in [0.0002, 0.002]$. The AdamW optimizer is used with learning rates between $1\times10^{-5}$ and $4\times10^{-5}$. The authors emphasize that the extra wall-clock overhead is $<1\%$ of the base QAT process.

## Key Experimental Results

### Main Results
Evaluation was conducted on LLaMA-3-1B/3B and Qwen-3-0.6B/1.7B for 1, 1.58, 2, 3, and 4-bit weights with 16/8/4-bit activations. Metrics include WikiText2 perplexity and average zero-shot accuracy across 8 QA datasets.

| Model & Quant Config | Baseline | PPL ↓ | Avg QA Acc ↑ | WinQ PPL ↓ | WinQ Acc ↑ | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LLaMA-1B W1A16 | ParetoQ | 16.9 | 51.9 | 15.3 | 52.6 | PPL drops significantly, Acc +0.7 |
| LLaMA-1B W1.58A16 | ParetoQ | 14.0 | 54.7 | 12.9 | 55.6 | PPL -1.1, Acc +0.9 in ternary setting |
| LLaMA-1B W2A16 | ParetoQ | 12.5 | 56.7 | 11.9 | 56.6 | Lower PPL, stable Acc |
| LLaMA-1B W1A8 | ParetoQ | 23.3 | 48.2 | 21.9 | 49.0 | Gains persist with 8-bit activations |
| LLaMA-3B W1.58A8 | ParetoQ | 13.1 | 55.9 | 12.2 | 58.6 | Acc +2.7 at larger scale |

Compared to PTQ methods (RTN, GPTQ, AWQ, SpinQuant), which often yield astronomical PPL at 1-2 bits (e.g., $10^8$ for LLaMA-1B W1A16), QAT is necessary, and WinQ further improves QAT efficiency. The paper reports 1.5–4$\times$ acceleration relative to SoTA QAT for sub-4-bit weights, with performance improvements up to 8.8% under the same compute budget.

### Ablation Study
| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| $\alpha=0.0$, no interpolation | W1A16 LLaMA-1B PPL 16.5 | Standard training plateaus at poor PPL |
| $\alpha=0.2$, $K=60K$ | PPL 15.5 | Moderate interpolation yields significant gain |
| $\alpha=0.4$, $K=60K$ | PPL 15.3 | Optimal performance near the main setting |
| $\alpha=0.8$, $K=60K$ | PPL 16.0 | Excessive interpolation disrupts training state |
| $\sigma=0$ | PPL 16.0 | Weaker effect without noise injection |
| $\sigma=0.001$ | PPL 15.3 | Optimal noise helps escape saddle points |
| $\sigma=0.004$ | PPL 18.5 | Excessive noise destabilizes optimization |

### Key Findings
- The most significant finding is that slow convergence in low-bit QAT can be quantified via the Hessian spectrum: fewer bits lead to more near-zero eigenvalues and smaller curvature, causing model stagnation.
- Both weight interpolation and noise injection are essential. Interpolation alters the curvature and position of latent weights relative to the grid, while noise injection enhances local perturbation; excessive use of either harms training.
- WinQ demonstrates high generalizability across ParetoQ and Hadamard Transform base methods, LLaMA/Qwen architectures, varying model scales, and different bit-widths.

## Highlights & Insights
- The value of this paper lies in transforming the engineering difficulty of QAT into a measurable optimization geometry problem. The Hessian spectrum serves as a direct derivation for the operations rather than just an ex-post explanation.
- The design of weight interpolation is restrained: it requires no changes to the quantization function, optimizer states, or model architecture, making it a plug-and-play QAT optimizer trick.
- The proximal update interpretation is insightful. Incorporating the distance between latent and quantized weights into the geometric explanation clarifies why interpolation boosts curvature more effectively than simple error penalties.
- For other fields, this suggests that when training plateaus due to constraints (discretization, pruning, sparsity), one should inspect for low-curvature saddle points in the constrained landscape rather than just adjusting surrogate gradients.

## Limitations & Future Work
- Validation was primarily on 0.6B–3B models. While covering LLaMA and Qwen, there is a scale gap compared to commonly deployed 7B, 13B, or 70B models. Whether the Hessian phenomena and hyperparameters remain stable at larger scales requires verification.
- The method involves tuning $K$, $\alpha$, and $\sigma$. Ablations show sensitivity, suggesting that automated tuning or curvature-based adaptive strategies would be more practical.
- Hessian analysis is computationally expensive. While WinQ training is cheap, the diagnostic pipeline might not be suitable for routine engineering monitoring. Future work could explore using gradient norms or loss plateaus as cheaper signals for re-initialization timing.
- The focus is on language model QAT. Transferring these ideas to vision models, MoEs, KV cache quantization, or training-time activation quantization remains an open question.

## Related Work & Insights
- **vs ParetoQ**: ParetoQ focuses on stretched elastic quantization and learnable step sizes to reduce quantization error; WinQ addresses convergence speed directly and can be superimposed on ParetoQ.
- **vs QuEST**: QuEST improves extreme low-bit quantization via Hadamard transforms and trust gradient estimators; WinQ's compatibility when defined in Hadamard space indicates they solve different bottlenecks.
- **vs GPTQ/AWQ/SpinQuant (PTQ)**: PTQ fails catastrophically at 1-2 bits, proving the necessity of training. WinQ's contribution is reducing the training cost given this necessity.
- **vs ProxQuant/LOTION/CAGE**: These frame QAT as regularization or smoothing. WinQ differs by identifying saddle-point stagnation through the Hessian spectrum and interpreting interpolation as a proximal-like update.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Clear perspective explaining QAT convergence through Hessian saddle points; utilizes established interpolation and noise concepts effectively.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Broad coverage of models, bit-widths, and ablations; however, larger model scales and real-world deployment performance could be further explored.
- Writing Quality: ⭐⭐⭐⭐☆ Complete logical loop from motivation and spectral analysis to algorithm and results.
- Value: ⭐⭐⭐⭐⭐ Highly practical for low-bit LLM QAT, especially as a low-cost acceleration plugin for existing methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Compute-Optimal Quantization-Aware Training](../../ICLR2026/model_compression/compute-optimal_quantization-aware_training.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](../../CVPR2026/model_compression/quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ICML 2026\] RaBiT: Residual-Aware Binarization Training for Accurate and Efficient LLMs](rabit_residual-aware_binarization_training_for_accurate_and_efficient_llms.md)
- [\[ICML 2026\] NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](nanoquant_efficient_sub-1-bit_quantization_of_large_language_models.md)

</div>

<!-- RELATED:END -->
