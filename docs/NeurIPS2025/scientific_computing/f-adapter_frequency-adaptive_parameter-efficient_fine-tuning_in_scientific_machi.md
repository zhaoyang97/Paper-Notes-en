---
title: >-
  [Paper Note] F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning
description: >-
  [NeurIPS 2025][Scientific Computing][Parameter-efficient fine-tuning] This paper presents the first systematic study of parameter-efficient fine-tuning (PEFT) for pretrained large operator models (LOMs) in scientific mac…
tags:
  - "NeurIPS 2025"
  - "Scientific Computing"
  - "Parameter-efficient fine-tuning"
  - "Fourier neural operator"
  - "frequency-adaptive"
  - "scientific machine learning"
  - "large operator model"
date: 2026-05-08
content_hash: ae97a906f70c4011
---

# F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.23173](https://arxiv.org/abs/2509.23173)  
**Code**: [Available](https://github.com/)  
**Area**: Scientific Computing
**Keywords**: Parameter-efficient fine-tuning, Fourier neural operator, frequency-adaptive, scientific machine learning, large operator model

## TL;DR
This paper presents the first systematic study of parameter-efficient fine-tuning (PEFT) for pretrained large operator models (LOMs) in scientific machine learning. It demonstrates that LoRA exhibits a depth-amplified approximation error lower bound in Fourier layers, whereas Adapter preserves universal approximation capacity. Building on this analysis, the paper proposes the Frequency-Adaptive Adapter (F-Adapter), which allocates adapter capacity according to spectral energy distribution. On 3D Navier-Stokes prediction tasks, F-Adapter achieves state-of-the-art performance while tuning fewer than 2% of parameters.

## Background & Motivation
**Background**: PEFT methods (LoRA, Adapter, Prompt Tuning, etc.) have been extensively validated for adapting large models in NLP and CV, yet remain largely unexplored in scientific machine learning (SciML). Recent large operator models (LOMs) such as DPOT (1B parameters) have acquired strong generalization ability through multi-PDE pretraining.

**Limitations of Prior Work**: LOMs are parameter-heavy (DPOT-H reaches 1B parameters), making full fine-tuning prohibitively expensive (~25–37 GB GPU memory, 100% parameter update). Directly transferring LoRA from NLP/CV to LOMs yields poor results: L2RE consistently hovers around 0.63–0.64 across different rank settings, far below full fine-tuning's 0.54.

**Key Challenge**: PDE solution manifolds exhibit broadband, cascade-coupled spectral characteristics that differ fundamentally from the low-rank structure of natural language or images. The linear low-rank constraint of LoRA introduces a depth-amplified approximation error lower bound in Fourier layers, preventing effective matching of PDE spectral properties.

**Goal**: Can a PEFT method be designed that respects the frequency-adaptive structure and physical priors of PDE solutions, achieving both efficiency and spectral fidelity?

**Key Insight**: The paper begins with theoretical analysis—proving the inherent limitation of LoRA (depth-amplified lower bound in Proposition 3.2) and the advantage of Adapter (exponentially decaying error in Proposition 3.2)—and then leverages the energy-concentration property of PDE solution spectra (low-frequency dominance) to guide the design of a frequency-aware Adapter.

**Core Idea**: Low-frequency bands contain the majority of PDE solution energy and should be assigned larger adapter bottleneck dimensions, while high-frequency bands are sparse and noise-sensitive, requiring only lightweight adapters.

## Method

### Overall Architecture
F-Adapter is a plug-and-play PEFT module inserted into every Fourier-domain mixing layer of a LOM (e.g., DPOT). The overall pipeline is:
1. The input tensor is transformed to the frequency domain via 3D rFFT.
2. The spectrum is partitioned into $B$ non-overlapping frequency bands by radial shells (typically $B=4$).
3. The channel dimension is divided into $K$ blocks.
4. Each (block, band) combination is equipped with three F-Adapters (input/mid/output), with bottleneck width determined by band position.
5. The output is transformed back to physical space via iRFFT and added via residual connection.

### Key Designs
1. **Frequency-Adaptive Bottleneck Allocation**: Bottleneck width is assigned by $r_b = \lfloor r_{min} + (r_{max} - r_{min})(1 - f_b/M)^p \rfloor$, where $f_b$ is the center frequency of band $b$. Low-frequency bands receive larger $r_b$ (close to $r_{max}$), while high-frequency bands shrink to $r_{min}$. This design is directly motivated by the spectral energy concentration theory for PDE solutions (Proposition 3).
2. **Bottleneck MLP Micro-Architecture**: Each Adapter follows a standard down-activation-up bottleneck residual structure: $\tilde{z} = z + s_b \cdot W^{up}_b \cdot \sigma(W^{down}_b \cdot z + b^{down}_b) + b^{up}_b$, using GELU activation, with $s_b$ as a learnable scalar.
3. **Zero-Initialization Strategy**: $W^{up}_b$ and $b^{up}_b$ are zero-initialized so that the Adapter acts as an identity mapping at the start of training, leaving pretrained weights undisturbed. Down-projections are initialized with Kaiming-uniform.
4. **Separate Real/Imaginary Processing**: The real and imaginary parts of complex frequency-domain tensors are processed by separate Adapters, avoiding the additional complexity of complex-valued arithmetic.

### Theoretical Foundations
- **Lower Bound for LoRA** (Proposition 3.2): The worst-case operator-norm approximation error of block-wise LoRA is lower-bounded by the $(Kr+1)$-th singular value of the global matrix, with errors accumulating as depth $K$ increases.
- **Exponential Convergence for Adapter** (Proposition 3.2): The approximation error of Fourier-domain Adapter is $O(K^{d/2-\alpha}) + O(K^{d/2} e^{-cm})$, decaying exponentially with bottleneck width $m$.
- **Spectral Sparsity of PDE Solutions** (Proposition 3): The cumulative energy of high-frequency modes decays polynomially as $O(K^{d-2s})$, confirming the dominance of low-frequency modes.

### Loss & Training
- AdamW optimizer, trained for 500 epochs.
- Only F-Adapter parameters (fewer than 2% of total) are updated; the pretrained backbone is frozen.
- Evaluation metric: L2 relative error (L2RE).

## Key Experimental Results

### Main Results

| Method | % Params | L2RE (Rand M=1.0) | L2RE (Rand M=0.1) | L2RE (Turb) |
|---|---|---|---|---|
| LoRA (r=32) | 1.37% | 0.6395 | 0.6211 | 0.6842 |
| AdaLoRA | 0.69% | 0.6726 | 0.6275 | 0.6795 |
| HydraLoRA | 0.85% | 0.6333 | 0.6164 | 0.6888 |
| Vanilla Adapter (d=8) | 1.16% | 0.5496 | 0.4893 | 0.4696 |
| FiLM Adapter | 1.30% | 0.5655 | 0.5054 | 0.4987 |
| **F-Adapter (Ours)** | **1.91%** | **0.5329** | **0.4639** | **0.4523** |
| Full Fine-Tuning | 100% | 0.5391 | 0.4002 | 0.2382 |

On SWE-2D, F-Adapter achieves an L2RE of 0.0116 (vs. 0.0902 for Vanilla Adapter and 0.1081 for LoRA). Under the data-scarce MHD-3D setting, it achieves 0.6341 (vs. 0.7226 for Vanilla Adapter).

### Ablation Study

| Method | Rand M=1.0 | Rand M=0.1 | Turb M=1.0 |
|---|---|---|---|
| F-Inverse-Adapter (reversed allocation) | 0.5664 | 0.4983 | 0.4747 |
| Vanilla Adapter | 0.5496 | 0.4893 | 0.4696 |
| **F-Adapter** | **0.5329** | **0.4639** | **0.4523** |

Among frequency-domain Adapter variants: Chebyshev Adapter achieves slightly lower accuracy with 3× higher latency; Fourier Adapter incurs 29% more memory, 10× slower inference, and substantially worse accuracy; WaveAct Adapter has comparable speed and memory but inferior accuracy to F-Adapter.

### Key Findings
1. LoRA and all its variants (AdaLoRA, HydraLoRA, RandLoRA, SVFT) fail comprehensively on FNO-based architectures, with L2RE consistently above 0.60.
2. Adapter-based methods significantly outperform LoRA, with performance improving as bottleneck width increases.
3. Frequency-adaptive allocation (larger dimensions for low frequencies, smaller for high frequencies) consistently outperforms both uniform and reversed allocation.
4. On the Transformer-based Poseidon architecture (non-FNO), F-LoRA (frequency-adaptive + LoRA) achieves state-of-the-art performance (L2RE = 0.2746), demonstrating the cross-architecture generality of the frequency-aware principle.

## Highlights & Insights
1. **First systematic study of PEFT in SciML**: Fills a gap in efficient fine-tuning of large models for scientific machine learning, and reveals that PEFT methods from NLP/CV cannot be directly transferred to SciML.
2. **Tight theory–practice coupling**: The paper first proves the inherent limitation of LoRA and the superiority of Adapter theoretically, then uses PDE spectral theory to guide architectural design, and finally validates the approach experimentally.
3. **Spectral drop-high diagnostic experiment**: A diagnostic experiment that progressively removes high-frequency components intuitively demonstrates the low-frequency energy dominance phenomenon.
4. **Cross-architecture generalization**: The F-LoRA variant is also effective on Transformer architectures, demonstrating the universality of the frequency-aware idea.

## Limitations & Future Work
1. F-Adapter's parameter count (~1.91%) is slightly higher than LoRA (~1.37%), though GPU memory overhead is comparable.
2. A gap remains relative to full fine-tuning (particularly on the Turbulence dataset: 0.4523 vs. 0.2382), indicating that recovery of high-frequency details remains imperfect.
3. The number of bands $B$ and the allocation curve exponent $p$ require manual tuning; automatic determination of these hyperparameters warrants further investigation.
4. Validation is currently limited to FNO-based architectures and Poseidon; extension to other architectures (e.g., U-Net variants, graph neural networks) remains unexplored.
5. Non-FNO architectures require an additional frequency estimation step (e.g., local FFT), increasing implementation complexity.

## Related Work & Insights
- **LoRA family**: LoRA, AdaLoRA, HydraLoRA, RandLoRA, and SVFT all rely on low-rank linear updates. Their success on LLMs but failure on FNOs suggests that the nonlinear nature of frequency-domain operations demands nonlinear adaptation.
- **DPOT**: Currently the largest public LOM (1B parameters), featuring a Fourier-Attention architecture and denoising-based pretraining.
- **Poseidon**: A purely Transformer-based operator model; the success of F-LoRA on this architecture confirms that frequency-aware PEFT generalizes to non-FFT architectures.
- **Insight**: Domain-specific inductive biases (e.g., spectral sparsity) should be explicitly encoded into PEFT architectures, rather than naively applying general-purpose methods.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of SciML PEFT + frequency-adaptive design + rigorous theoretical analysis
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-baseline comparison + extensive ablations + cross-architecture validation
- Writing Quality: ⭐⭐⭐⭐ Clear theory–experiment–design logic with natural motivation
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for efficient adaptation of large SciML models, with notable theoretical and methodological contributions

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study](../../AAAI2026/scientific_computing/scientific_knowledge-guided_machine_learning_for_vessel_power_prediction_a_compa.md)
- [\[NeurIPS 2025\] Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data](neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)
- [\[NeurIPS 2025\] Physics-Guided Machine Learning for Uncertainty Quantification in Turbulence Models](physics-guided_machine_learning_for_uncertainty_quantification_in_turbulence_mod.md)
- [\[NeurIPS 2025\] Integration Matters for Learning PDEs with Backward SDEs](integration_matters_for_learning_pdes_with_backward_sdes.md)
- [\[NeurIPS 2025\] One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs](oneshot_transfer_learning_nonlinear_pdes_perturbative_pinns.md)

</div>

<!-- RELATED:END -->
