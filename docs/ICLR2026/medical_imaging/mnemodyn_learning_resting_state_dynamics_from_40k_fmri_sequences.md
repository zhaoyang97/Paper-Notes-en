---
title: >-
  [Paper Note] MnemoDyn: Learning Resting State Dynamics from 40K fMRI Sequences
description: >-
  [ICLR 2026][Medical Imaging][rs-fMRI] MnemoDyn conceptualizes resting-state fMRI (rs-fMRI) as a trajectory in a latent space driven by a "learnable evolution operator." By replacing Transformer self-attention with wavelet-parameterized pseudo-differential operators, the authors pre-train a lightweight, long-sequence-friendly, and cross-dataset generalizabl
tags:
  - ICLR 2026
  - Medical Imaging
  - rs-fMRI
date: 2026-05-08
content_hash: cf90ed05cffe78a1
---
# MnemoDyn: Learning Resting State Dynamics from 40K fMRI Sequences

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=zexMILcQOV](https://openreview.net/forum?id=zexMILcQOV)  
**Code**: [https://github.com/vsingh-group/mnemodyn](https://github.com/vsingh-group/mnemodyn)  
**Area**: Medical Imaging / Computational Neuroscience / Foundation Models  
**Keywords**: rs-fMRI, brain dynamics, operator learning, wavelets, pseudo-differential operators, controlled differential equations, foundation models  

## TL;DR
MnemoDyn conceptualizes resting-state fMRI (rs-fMRI) as a trajectory in a latent space driven by a "learnable evolution operator." By replacing Transformer self-attention with wavelet-parameterized pseudo-differential operators, the authors pre-train a lightweight, long-sequence-friendly, and cross-dataset generalizable brain imaging foundation model on approximately 40K rs-fMRI sequences.

## Background & Motivation
**Background**: rs-fMRI records spontaneous Blood-Oxygen-Level-Dependent (BOLD) signals, serving as a critical modality for surgical planning, epilepsy localization, and cognitive/disease research. Recent rs-fMRI foundation models (e.g., BrainLM, Brain-JEPA) predominantly adopt Transformer backbones from NLP, using self-attention to model temporal dependencies, which performs well on standard 5–7 minute acquisition protocols.

**Limitations of Prior Work**: Directly applying attention to brain signals faces three major challenges. First is the **cost of long sequences**—clinical and sleep scenarios are shifting toward continuous recordings lasting hours, where the quadratic complexity of attention leads to computational explosion. Second is **data efficiency**—downstream cohorts in real-world scenarios often contain only a few hundred samples, making attention-based models difficult to fine-tune. Third is **deployment cost**—large models are hard to deploy in resource-constrained clinical environments. More fundamentally, the inductive biases of attention (tokenization, position encoding, global interaction) do not align well with the nature of brain signals, which are discrete samplings of continuous processes with strong local temporal correlations and hierarchical, multi-scale structures.

**Key Challenge**: The foundation model paradigm requires "large-scale pre-training + strong generalization," yet the "continuous, multi-scale, noisy, and expensive" nature of brain signals conflicts with the "discrete tokens + global dense interaction" assumptions of attention. There is a need for the transferability of foundation models without the computational overhead and mismatched inductive biases of attention.

**Goal**: Construct an rs-fMRI foundation model that is independent of attention, computationally and parameter efficient, capable of handling long sequences, and able to transfer stably to small-sample downstream tasks.

**Core Idea**: **[Operator Learning instead of Auto-regression]** Instead of learning auto-regressive mappings of raw signals or latent states, the model treats the brain as a dynamical system generating trajectories in a latent space and directly learns the **evolution operator** governing these trajectories. **[Wavelets × Pseudo-differential Operators]** The operator kernel is parameterized using multi-resolution wavelet bases. Leveraging the natural sparse (block-diagonal) representation arising from the interaction between wavelets and pseudo-differential operators, multi-scale modeling becomes both expressive and computationally efficient.

## Method

### Overall Architecture
MnemoDyn views the observed signal $x(t)\in\mathbb{R}^n$ (for $n$ brain regions/parcels) as a measurement of a latent neural state $z(t)\in\mathbb{R}^d$, which evolves according to a continuous-time dynamical system $\frac{dz(t)}{dt}=F(z(t),u(t);\theta)$. Solving this ODE is equivalent to learning a non-linear integral operator that maps the "initial state + input path" to the "entire latent trajectory." The operator kernel is parameterized using multi-resolution wavelet bases and CP tensor low-rank decomposition, enabling the model to be efficiently implemented via convolutional kernels with linear scaling for long sequences. During pre-training, the model uses (masked/denoising) auto-encoding for signal reconstruction. During fine-tuning, the backbone is frozen, and only lightweight adapters or MLP heads are trained to predict clinical variables.

```mermaid
flowchart LR
    A[rs-fMRI Preprocessing<br/>NIfTI→CIFTI→Parcellation<br/>450 ROI Time Series] --> B[Projection to Latent Space<br/>Low-rank Bottleneck]
    B --> C[Wavelet-parameterized Evolution Operator<br/>Multi-resolution + Pseudo-differential<br/>CP Tensor Decomposition]
    C --> D[Latent Dynamical Trajectory z(t)]
    D --> E[Pre-training: Masked/Denoising Reconstruction]
    D --> F[Fine-tuning: Frozen Backbone<br/>+ Adapter/MLP Head]
    F --> G[Clinical Variable Prediction<br/>Diagnosis/Age/Sex/Cognition]
```

### Key Designs
**1. From State-Space to Operators: Capturing non-Markovian dependencies via integral operators.** The authors start with a standard state-space model—latent state transition $z_{t+1}=f(z_t,u_t;\theta)+w_t$ and observation $x_t=h(z_t;\phi)+v_t$—and argue that the continuous-time form $\frac{dz}{dt}=F(z,u;\theta)$ better suits the nature of brain signals. The discrete mapping $f$ is essentially an integration step of the ODE flow. Writing the ODE in integral form yields a Volterra-type equation $z(t)=z_0+\int_0^t F(z(\tau),u(\tau);\theta)\,d\tau$, where the integral from 0 to $t$ allows the operator to access the **entire history** of the input at each time point. By decomposing the vector field into an autonomous drift term $P$ and a controlled modulation term $K$, the latter forms a non-linear integral operator acting on the input: $(K_\theta u)(t)=\int_0^t K(z(\tau);\theta)u(\tau)\,d\tau$. The authors further formulate this as a Controlled Differential Equation (CDE) $z(t)=z_0+\int_0^t P\,d\tau+\int_0^t K\,du_W(\tau)$, where the wavelet-transformed path $u_W$ acts as a "rough path," enabling the model to encode history beyond point-wise values—a key advantage over pure ODEs for modeling non-Markovian dependencies.

**2. Multi-resolution Wavelet Kernels: Embedding "multi-scale" into the operator.** Neuroscience suggests that brain signals possess hierarchical, multi-scale organization. Accordingly, the authors expand the integral kernel into a linear combination of separable wavelet bases $K(z(\tau);\theta)=\sum_{j=0}^{J}\sum_k \phi_{j,k}(\tau)A_{j,k}(z(\tau);\theta)$, where $\phi_{j,k}$ represent wavelet bases at scale $j$ and translation $k$, and $A_{j,k}$ are matrix-valued functions modulated by the current state $z(\tau)$. This allows the operator to perform local filtering at multiple scales and positions before applying state-conditional transformations, preserving temporal locality while achieving scale adaptability. Each layer corresponds to a wavelet scale, coupled via residual connections to integrate fine-grained fluctuations and long-range structures.

**3. Pseudo-differential Operators + CP Low-rank: Achieving computational feasibility.** A naive implementation faces two issues: the operator kernel requires massive matrices as sequences lengthen, and high-dimensional rs-fMRI requires large latent dimensions. The authors address this by noting that since signals are represented in the wavelet domain, **the interaction between wavelets and pseudo-differential operators naturally results in highly sparse, block-diagonal representations**. Consequently, the interaction between parameters and latent dynamics can be computed compactly in the wavelet domain using parallel convolutional filters. To handle parameter explosion in high-dimensional latent spaces, Canonical Polyadic (CP) tensor decomposition $X\approx\sum_{r=1}^{R}\lambda_r a_r^{(1)}\otimes\cdots\otimes a_r^{(N)}$ is applied to compress operator parameter tensors into sums of rank-one outer products, drastically reducing free parameters while maintaining expressivity. This allows MnemoDyn (92M parameters) to finish pre-training in ~3 hours on a single A100-40GB.

**4. Pre-training Goal and Fine-tuning: Masked reconstruction + Frozen transfer.** Three self-supervised variants are explored: MnemoDyn-Denoise (denoising auto-encoding), MnemoDyn-Mask (reconstructing from 70% random spatio-temporal masks), and MnemoDyn-Mask-JEPA (following the Brain-JEPA masking scheme). During fine-tuning, the pre-trained backbone is frozen, and an MLP head (with LayerNorm/GELU/Dropout) is attached to the pooled temporal and ROI features. Experiments show consistent performance across pre-training objectives, suggesting that the **operator parameterization itself** is the primary driver of performance.

## Key Experimental Results
Pre-training was conducted on UK Biobank (~65K samples, TR 0.735s, length ~490) and HCP (~1000 samples, TR 0.72s, length 1200). All data were preprocessed to NIfTI→CIFTI→450 ROIs (Schaefer-400 cortical + Tian subcortical). Downstream evaluation utilized six additional datasets: HCP-Aging, ADNI, ADHD-200, ABIDE, and NKIR.

### Main Results
Performance on ADNI (diagnosis/biomarkers) and UK Biobank (demographics) (test set, mean):

| Method | NC/MCI ACC↑ | NC/MCI F1↑ | Amyloid ACC↑ | Amyloid F1↑ | Age MSE↓ | Sex ACC↑ |
|---|---|---|---|---|---|---|
| BrainNetCNN | 60.00 | 64.72 | 59.00 | 59.43 | 0.99 | 77.86 |
| BrainGNN | 67.40 | 71.42 | 57.00 | 62.61 | 0.93 | 77.31 |
| BNT | 78.90 | 83.14 | 62.00 | 59.53 | 0.86 | 80.78 |
| BrainLM | 75.79 | 85.66 | 67.00 | 68.82 | 0.61 | 86.47 |
| Brain-JEPA | 76.84 | 86.32 | 71.00 | 75.97 | 0.50 | 88.17 |
| **MnemoDyn-Mask** | **96.12** | **95.98** | **95.27** | **95.61** | 0.44 | **88.40** |
| MnemoDyn-Mask-JEPA | 93.67 | 93.32 | 94.89 | 94.60 | **0.42** | 88.30 |

Significant gains were observed in diagnosis/biomarker tasks (NC/MCI accuracy increased from ~77% to 96%; Amyloid from ~71% to 95%).

Performance on HCP-Aging (demographics + cognitive traits) (test set, mean):

| Method | Age MSE↓ | Sex ACC↑ | Sex F1↑ | Neuroticism MSE↓ | Flanker MSE↓ |
|---|---|---|---|---|---|
| BrainLM | 1.14 | 75.27 | 73.19 | 1.05 | 0.77 |
| Brain-JEPA | 1.02 | 79.17 | 76.29 | 0.99 | 1.28 |
| MnemoDyn-Denoise | 0.91 | 80.20 | 80.11 | 0.91 | 0.61 |
| **MnemoDyn-Mask** | **0.90** | **83.10** | **82.77** | **0.90** | **0.60** |
| MnemoDyn-Mask-JEPA | 0.90 | 82.57 | 82.23 | 0.90 | 0.60 |

### Ablation Study
Reconstruction generalization across foundation models (Validation set MSE / R²):

| Model | UK-Biobank (MSE, R²) | HCP (MSE, R²) |
|---|---|---|
| MnemoDyn-UKB | 2.36e-5, 0.985 | 4.52e-8, 0.934 |
| MnemoDyn-HCP | 1.86e-9, 0.969 | 3.94e-6, 0.987 |

Models trained on UKB can reconstruct HCP data with high quality and vice versa (R² > 0.93), indicating that the operator representation generalizes well across datasets.

### Key Findings
- **Efficiency**: A 92M parameter model completes pre-training in ~3 hours on a single A100-40GB, significantly lower than the 4-GPU setup often required by baselines.
- **Structural Emergence**: After pre-training, the Frobenius norm concentrates on the wavelet operator kernels, and the output projection matrix reaches ~95% sparsity, suggesting that dynamics are driven by structured, multi-scale filters.
- **Small-sample Friendliness**: Functional foundation models can be trained even with moderate-sized data like HCP (~1000 samples), which is typically insufficient for attention-based models.

## Highlights & Insights
- **Paradigm Shift**: Reformulating fMRI modeling from "learning auto-regressive sequence mappings" to "identifying operators governing dynamical systems" bypasses complex components like tokenization and position encoding.
- **Domain Priors in Architecture**: Multi-resolution wavelets directly correspond to the hierarchical multi-scale organization of brain signals, aligning inductive biases with the data's nature rather than adopting NLP assumptions.
- **Mathematical Structure for Efficiency**: The combination of pseudo-differential operators (sparsity) and CP low-rank decomposition is the engineering key to making multi-scale integral operators trainable on a single GPU.
- **Huge Jump in Diagnostic Tasks**: The ~20% improvement on ADNI suggests that operator representations may be exceptionally sensitive to disease-related dynamical differences.

## Limitations & Future Work
- The experiments are limited to parcellated (region-level) rs-fMRI and have not yet extended to voxel-level or multimodal inputs (e.g., EEG/PET).
- The authors emphasize that latent dynamical system modeling **does not equal physiological validation**—while operator spectra are interpretable, they should not be taken as direct evidence of neurophysiological mechanisms.
- The magnitude of improvement on ADNI is unusually large and warrants further verification across more cohorts to ensure statistical robustness and rule out data leakage.
- Future Work: Extending to long sequences (e.g., hours-long sleep recordings) and multimodal fusion, where the linear scaling of operators is particularly advantageous.

## Related Work & Insights
- **Operator Learning / State Space**: DeepONet and FNO pioneered learning mappings between function spaces. SSMs like S4 and Mamba decompose latent evolution and observation, but few target neurophysiological data or utilize multi-scale bases like wavelets.
- **Attention Models in Brain Imaging**: BNT, BrainLM, and Brain-JEPA introduced Transformers to fMRI but often struggle with long-range noise or irregular sampling. This work provides an alternative, domain-aligned route without global attention.
- **Lightweight Domain Models**: Recent time-series benchmarks suggest that structurally aligned CNNs/RNNs can outperform Transformers on low-data tasks. This paper provides strong evidence for this trend in brain imaging foundation models.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Combining operator learning, wavelet pseudo-differential operators, and CP decomposition for rs-fMRI is a coherent new paradigm orthogonal to mainstream attention-based routes.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers various tasks (reconstruction, classification, regression), eight datasets, cross-dataset generalization, and ablation studies; however, the massive gain on ADNI needs more rigorous cohort validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear progression from state-space to ODEs to integral operators. Motivation is well-defined, though engineering details are somewhat condensed.
- **Value**: ⭐⭐⭐⭐⭐ Singe-GPU training capability and small-sample utility make this model highly practical for real-world neuroimaging research with limited resources.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

1. **BrainLM**: A Foundation Model for fMRI Analysis via Generative Pre-training.
2. **Brain-JEPA**: Joint-Embedding Predictive Architecture for Self-Supervised Learning on rs-fMRI.
3. **Mamba**: Linear-Time Sequence Modeling with Selective State Spaces.
4. **Neural Operators**: Fourier Neural Operator for Parametric Partial Differential Equations.

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation](learning_patient-specific_disease_dynamics_with_latent_flow_matching_for_longitu.md)
- [\[NeurIPS 2025\] GeoDynamics: A Geometric State-Space Neural Network for Understanding Brain Dynamics on Riemannian Manifolds](../../NeurIPS2025/medical_imaging/geodynamics_a_geometric_state-space_neural_network_for_understanding_brain_dynam.md)
- [\[ICLR 2026\] Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer](brain-it_image_reconstruction_from_fmri_via_brain-interaction_transformer.md)
- [\[ICLR 2026\] Moving Beyond Diffusion: Hierarchy-to-Hierarchy Autoregression for fMRI-to-Image Reconstruction](moving_beyond_diffusion_hierarchy-to-hierarchy_autoregression_for_fmri-to-image_.md)
- [\[CVPR 2026\] fMRI-LM: Towards a Universal Foundation Model for Language-Aligned fMRI Understanding](../../CVPR2026/medical_imaging/fmri-lm_towards_a_universal_foundation_model_for_language-aligned_fmri_understan.md)

</div>

<!-- RELATED:END -->
