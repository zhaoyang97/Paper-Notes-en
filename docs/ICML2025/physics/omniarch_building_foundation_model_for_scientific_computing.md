---
title: >-
  [Paper Note] OmniArch: Building Foundation Model For Scientific Computing
description: >-
  [ICML 2025][Physics & Scientific Computing][foundation model] OmniArch is the first scientific computing foundation model uniformly pre-trained on 1D-2D-3D PDEs. It addresses multi-scale challenges via a Fourier encoder-decoder, handles multi-physical quantity couplings using a Temporal Mask mechanism, and aligns physical priors with a PDE-Aligner, achieving SOTA performance on 11 families of PDEs in PDEBench.
tags:
  - "ICML 2025"
  - "Physics & Scientific Computing"
  - "foundation model"
  - "PDE Solver"
  - "Fourier Neural Operator"
  - "Multi-scale"
  - "Physics-Informed"
date: 2026-05-08
content_hash: 4fb3e4737dae104c
---

# OmniArch: Building Foundation Model For Scientific Computing

**Conference**: ICML 2025  
**arXiv**: [2402.16014](https://arxiv.org/abs/2402.16014)  
**Code**: [https://openi.pcl.ac.cn/cty315/OmniArch](https://openi.pcl.ac.cn/cty315/OmniArch)  
**Area**: Scientific Computing  
**Keywords**: foundation model, PDE Solver, Fourier Neural Operator, Multi-scale, Physics-Informed

## TL;DR
OmniArch is the first scientific computing foundation model uniformly pre-trained on 1D-2D-3D PDEs. It addresses multi-scale challenges via a Fourier encoder-decoder, handles multi-physical quantity couplings using a Temporal Mask mechanism, and aligns physical priors with a PDE-Aligner, achieving SOTA performance on 11 families of PDEs in PDEBench.

## Background & Motivation
Solving partial differential equations (PDEs) is the core foundation of numerous scientific and engineering applications (e.g., aircraft design, weather forecasting, semiconductor manufacturing). Conventional methods (such as the finite element method and finite volume method) require extensive manual programming and incur extremely high computational overhead, consuming vast amounts of time even on high-performance computing clusters. While neural operator methods (such as FNO and DeepONet) can learn mappings between function spaces, each model is limited to a specific type of PDE and cannot transfer across physical systems.

Among existing works, MPP, Poseidon, and DPOT have attempted unified pre-training, but three key challenges remain: (1) **Multi-scale**—different PDEs involve 1D/2D/3D data, various grid resolutions, and shapes, whereas existing methods are mostly limited to fixed mapping grids; (2) **Multi-physical quantities**—different systems contain varying numbers of physical quantities (velocity, density, pressure, etc.), requiring concurrent modeling of their coupling relationships; (3) **Physics alignment**—predictions must conform to known physical laws (conservation laws, boundary conditions, etc.) rather than merely fitting the data.

The key insight of this paper is: **Can a unified foundation model, akin to Large Language Models, be utilized to solve various 1D, 2D, and 3D PDEs simultaneously**? The core idea is to eliminate dimensional discrepancies using Fourier domain encoding, model temporal evolution via autoregressive Transformers, and align physical priors through contrastive learning.

## Method

### Overall Architecture
OmniArch adopts a "pre-training + fine-tuning" paradigm. In the pre-training phase, physical field data of different dimensions (1D/2D/3D) are transformed to the frequency domain via a Fourier encoder, where representations are unified in length through TopK mode truncation. Then, a shared Transformer backbone models the temporal dynamics, and finally, a Fourier decoder restores the spatial-domain predictions. In the fine-tuning phase, a PDE-Aligner is introduced to perform physics-aligned contrastive learning utilizing textual descriptions of the equations.

### Key Designs

1. **Fourier Encoder-Decoder (Addressing Multi-scale)**:

    - **Function**: Uniformly encode physical fields of different dimensions and resolutions into the frequency domain.
    - **Mechanism**: For the physical field $u(x^{(d)}, t)$, a linear projection $\Psi$ is first applied to align dimensions, followed by FFT. Then, TopK selection is used to extract the $K$ most significant frequency components: $\hat{u}_K(k,t) = \text{TopK}(\text{FFT}(\Psi[u(x^{(1)},t), \ldots, u(x^{(D)},t)]^\top))$. During decoding, zero-padding is performed on the predicted $K$ modes to restore the target shape, followed by an IFFT back to the spatial domain. Because data from different grids possess frequency-domain representations of the same length after truncation, unified cross-scale input is achieved.
    - **Design Motivation**: The complexity of FFT is $O(N \log N)$, which is lower than the $O(N^2)$ complexity of convolution. High frequencies (detailed variations) and low frequencies (overall trends) are naturally separated in the frequency domain, and global information is inherently weighted, making it suitable for handling complex boundary conditions and heterogeneous grids.

2. **Temporal Mask + Transformer Backbone (Addressing Multi-physical Quantities)**:

    - **Function**: Model the temporal evolution of multi-physical quantities using the Transformer autoregressive mechanism.
    - **Mechanism**: The embedding of all physical quantities at each time step is grouped as $\mathbf{Z}_t = \{\mathbf{U}_t, \mathbf{V}_t\}$, and a Temporal Mask $\mathbf{M}$ is designed so that the token at each time step can attend to all physical quantities at the current and all preceding time steps, but cannot see the future. Specifically, for $C$ physical quantities, the masking rule is: $\mathbf{M}(i,j) = 0$ when $\lfloor j/C \rfloor \le \lfloor i/C \rfloor$, and $-\infty$ otherwise. This differs from standard causal masking, as physical quantities within the same time step can fully attend to each other.
    - **Design Motivation**: In Navier-Stokes equations, velocity and pressure are coupled and must be processed simultaneously (to satisfy constraints such as the continuity equation). Sequential token processing cannot correctly model such synchronized constraints. This design bridges Transformer autoregression with an analogy to traditional multi-step solver methods.

3. **PDE-Aligner (Physics-Aligned Fine-tuning)**:

    - **Function**: Apply physical constraints to predictions using textual descriptions of PDE equations during the fine-tuning phase.
    - **Mechanism**: A pre-trained BERT is used to encode the PDE equation text $E_{\text{text}}(\mathcal{P})$, while physical evolution features are extracted from the frequency-domain representations of the initial and current states (phase difference $\Delta\phi$ captures wave propagation and dispersion properties, and amplitude ratio $R$ quantifies cross-scale energy transfer). The alignment loss is $L_{\text{Align}} = L_{\text{eq}} + \lambda L_E$, where $L_{\text{eq}}$ is the contrastive loss between textual and physical features, and $L_E = |\sum_K R - 1|$ ensures Parseval's theorem (frequency-domain energy conservation). The total fine-tuning loss is $L_{\text{ft}} = L_{\text{sim}} - L_{\text{eq}}$.
    - **Design Motivation**: PDE equations are the most natural "supervision signals" for physical phenomena. Conducting alignment in the frequency domain is more effective because conservation laws constrain the distribution of energy across modes, and different PDEs exhibit characteristic spectral fingerprints.

### Loss & Training
- **Pre-training Loss**: Normalized nRMSE loss $L_{\text{sim}}^u = \frac{1}{|B|}\sqrt{\sum_{(x,t)\in B}\left(\frac{u^{\text{pred}}(x,t)-u(x,t)}{\sigma_u}\right)^2}$, averaged across physical quantities.
- **Fine-tuning Loss**: $L_{\text{ft}} = L_{\text{sim}} - L_{\text{eq}}$, optimizing both prediction accuracy and physical consistency simultaneously.
- **Backbone Architecture**: LLaMA-style Transformer (trained from scratch), available in Base and Large versions.
- **PDE-Aligner Text Encoding**: Uses the pre-trained BERT-base-cased model.

## Key Experimental Results

### Main Results

| PDE Type | FNO | MPP-AVIT-L | DPOT-L | OmniArch-L + Aligner | Gain |
|----------|-----|-----------|--------|----------------------|------|
| 1D CFD | 1.4100 | – | – | **0.0200** | 98.7% |
| 1D Advection | 0.0091 | – | – | **0.0041** | 4.65% |
| 1D Burgers | 0.0174 | – | – | **0.0032** | 66.3% |
| 2D CFD | 0.2060 | 0.0178 | 0.0112 | **0.0125** | – |
| 2D Reaction | 0.1203 | 0.0098 | 0.0263 | **0.0084** | 14.3% |
| 2D SWE | 0.0044 | 0.0022 | 0.0451 | **0.0012** | 45.5% |
| 2D Incom. | 0.2574 | – | – | **0.0827** | 67.9% |
| 3D Maxwell | 0.1906 | – | – | **0.1671** | 12.3% |

### Ablation Study

| Configuration | 2D Incom. | 2D CFD | 3D CFD |
|------|-----------|--------|--------|
| Causal Mask | 0.0277 | 0.0198 | 0.1842 |
| No Mask | 0.0285 | 0.0205 | 0.1923 |
| **Temporal Mask** | **0.0227** | **0.0148** | **0.1494** |

| Configuration | 1D PDEs | 2D PDEs | 3D PDEs |
|------|---------|---------|---------|
| Pre-training Only | 0.0103 | 0.0440 | 0.3399 |
| Fine-tuning w/o Aligner | 0.0073 | 0.0345 | 0.3432 |
| **Fine-tuning w/ Aligner** | **0.0056** | **0.0262** | **0.2697** |
| Gain | 23.3% | 24.1% | 21.4% |

### Key Findings
- **1D-2D-3D unified pre-training is effective**: OmniArch is the first model to perform unified pre-training across three dimensions, overall surpassing all expert models and pre-trained models on 11 families of PDEs.
- **Temporal Mask significantly outperforms causal mask**: Showing an improvement of 18-20%, with the most pronounced advantage in 3D CFD (where 5 physical quantities are coupled).
- **PDE-Aligner consistently yields approximately 22% improvement**: With similar improvement ratios across different dimensions (23.3% for 1D, 24.1% for 2D, 21.4% for 3D), indicating that physics alignment is dimension-independent.
- **Zero-shot generalization**: On unseen PDEs (Shock, KH, OTVortex), the error is 4-7 times lower than that of MPP.
- **Multi-scale inference**: Due to the Fourier truncation mechanism, it can handle inputs of different resolutions without retraining, with 128-256 resolutions performing optimally.
- **In-context learning**: Demonstrating emerging capabilities similar to LLMs, it can learn new neural operators given observations from only a few time steps.

## Highlights & Insights
- Transferring the successful paradigms of foundation models in the NLP field (pre-training + fine-tuning + alignment) to the domain of PDE solving is conceptually simple yet powerful.
- Fourier-domain encoding is an elegant solution to multi-scale problems—frequency truncation naturally achieves a unified representation across resolutions.
- The design of the Temporal Mask captures the essence of multi-physical systems—coupled variables must be processed synchronously.
- PDE-Aligner utilizes equation texts for physics alignment, cleverly drawing inspiration from CLIP-style contrastive learning. The design of using frequency-domain features (phase difference + amplitude ratio) as physical fingerprints is highly novel.
- The emergent capabilities of zero-shot and in-context learning are impressive, implying that the model has learned transferable physical operators rather than simple data patterns.

## Limitations & Future Work
- **Room for improvement in 3D performance**: The nRMSE for 3D CFD and Maxwell remains relatively high (0.37, 0.17), and the authors acknowledge that 3D systems pose a challenge to the model.
- **Insufficient interpretability**: Although the PDE-Aligner enhances physics alignment, the model fundamentally remains a data-driven black box.
- **Computational and data bottlenecks**: Scalability is limited by computational resources and available training data, especially in systems with complex transient behaviors.
- **Unverified on practical engineering problems**: All experiments were conducted on standard benchmarks; the performance on real engineering applications (complex geometries, unstructured grids) remains unknown.
- **PDE-Aligner requires equation text**: Physics alignment cannot be directly applied to systems with unknown underlying equations.

## Related Work & Insights
- **vs FNO**: While retaining FNO's advantages in frequency-domain processing, OmniArch gains transferability across different PDEs via pre-training.
- **vs MPP/DPOT**: These methods only support 2D pre-training, whereas OmniArch achieves 1D-2D-3D unification for the first time, with far superior zero-shot generalization.
- **vs Poseidon**: Poseidon supports single-step inference at arbitrary time steps but lacks satisfactory accuracy; OmniArch uses autoregressive multi-step inference to achieve higher accuracy.
- **PDE-Aligner Inspiration**: Describing physical laws in natural language and aligning them via contrastive learning is a promising direction worthy of further study—it could potentially be extended in the future to leveraging LLMs for understanding and generating PDE constraints.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First 1D-2D-3D unified pre-training; both Temporal Mask and PDE-Aligner designs demonstrate strong originality.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on 11 families of PDEs, including extensive experiments on zero-shot, in-context learning, multi-scale, inverse problems, etc.
- Writing Quality: ⭐⭐⭐⭐ Method text is clearly described, but some equations are highly dense, and the analysis of 3D experiments lacks depth.
- Value: ⭐⭐⭐⭐⭐ Establishes an important milestone for the direction of foundation models in PDE solving, with the unified architecture concept possessing profound impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FM4NPP: A Scaling Foundation Model for Nuclear and Particle Physics](../../ICLR2026/physics/fm4npp_a_scaling_foundation_model_for_nuclear_and_particle_physics.md)
- [\[AAAI 2026\] Towards a Foundation Model for Partial Differential Equations Across Physics Domains](../../AAAI2026/physics/towards_a_foundation_model_for_partial_differential_equations_across_physics_dom.md)
- [\[ICML 2025\] Finetuning Stellar Spectra Foundation Models with LoRA](finetuning_stellar_spectra_foundation_models_with_lora.md)
- [\[NeurIPS 2025\] F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](../../NeurIPS2025/physics/f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)
- [\[AAAI 2026\] Data Verification is the Future of Quantum Computing Copilots](../../AAAI2026/physics/data_verification_is_the_future_of_quantum_computing_copilots.md)

</div>

<!-- RELATED:END -->
