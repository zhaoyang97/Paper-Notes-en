---
title: >-
  [Paper Note] Latent Diffusion Pretraining for Crystal Property Prediction
description: >-
  [ICML2026][Image Generation][Crystal Property Prediction] CrysLDNet transfers "diffusion pretraining" from the raw crystal feature space to a smooth latent space learned by a VAE. This allows the PDDFormer encoder to lea…
tags:
  - "ICML2026"
  - "Image Generation"
  - "Crystal Property Prediction"
  - "Latent Diffusion"
  - "Variational Autoencoder"
  - "GNoME Pretraining"
  - "Materials Foundation Model"
date: 2026-05-08
content_hash: 00f27930007c2339
---

# Latent Diffusion Pretraining for Crystal Property Prediction

**Conference**: ICML2026  
**arXiv**: [2606.00776](https://arxiv.org/abs/2606.00776)  
**Code**: https://github.com/shrimonmuke0202/CrysLDNet.git  
**Area**: Scientific Computing / Materials Science / Crystal Property Prediction / Latent Diffusion Pretraining  
**Keywords**: Crystal Property Prediction, Latent Diffusion, Variational Autoencoder, GNoME Pretraining, Materials Foundation Model

## TL;DR
CrysLDNet transfers "diffusion pretraining" from the raw crystal feature space to a smooth latent space learned by a VAE. This allows the PDDFormer encoder to learn more compact and symmetry-aware structural semantics on 380,000 unlabeled GNoME crystals. Downstream JARVIS / MP property predictions achieve average MAE reductions of 4.26% / 4.90% over strong supervised SOTA models, with even greater advantages in low-data and experimental data correction scenarios.

## Background & Motivation
**Background**: Using GNNs (CGCNN, ALIGNN) and equivariant Transformers (Matformer, PDDFormer) to predict properties like formation energy and bandgap from 3D crystal structures has approached DFT accuracy on labeled datasets, serving as a primary surrogate for materials screening.

**Limitations of Prior Work**: DFT-labeled data are extremely scarce and unevenly distributed (some properties have only thousands of samples), causing supervised models to overfit severely in low-data regimes. While massive unlabeled crystal structures are available (GNoME collected 380,000), current self-supervised solutions (CrysXPP, Crystal Twins, CrysGNN) still struggle to capture structural semantics effectively. Recent diffusion-based pretraining methods like CrysDiff and DPF perform diffusion directly in the raw feature space, requiring the simultaneous handling of three heterogeneous variables: **discrete atomic types** (using D3PM discrete diffusion), **continuous lattice parameters** (using DDPM), and **periodic fractional coordinates** (requiring score matching based on wrapped normal distributions). This forces architectural complexity, increases diffusion steps, and constrains the final representation to a non-smooth input space.

**Key Challenge**: Crystal properties are essentially determined by atomic arrangement and lattice geometry. However, the raw feature space is a "fragmented structure" composed of discrete, continuous, and periodic components. Performing diffusion directly on this space is neither elegant nor conducive to learning smooth, transferable representations.

**Goal**: Construct a diffusion-based pretraining framework that **uniformly processes the three heterogeneous variables** and is non-intrusive to the encoder architecture. The learned representations should be capable of fully reconstructing crystal features ($A / X / L$) while transferring well to downstream few-shot scenarios.

**Key Insight**: Drawing from the paradigm of Stable Diffusion—"compress to latent space via VAE, then perform diffusion in latent space"—the VAE encodes the three heterogeneous variables into a unified continuous, smooth, and low-dimensional latent space $\mathbf{Z}\in\mathbb{R}^{N\times d}$. All diffusion occurs within this continuous space, while equivariant constraints (rotation/periodic translation) are naturally handled by the PDDFormer encoder.

**Core Idea**: Jointly pretrain using a "VAE encoder (PDDFormer) + Latent Flow Matching (DiT denoising)." The heavy lifting of diffusion is offloaded to the latent space, and downstream tasks only fine-tune this doubly refined encoder.

## Method

### Overall Architecture
CrysLDNet is a pipeline consisting of two-stage pretraining and one-stage finetuning:

- **Input**: Crystal material $\mathcal{M}=(\mathbf{A},\mathbf{X},\mathbf{L})$, where $\mathbf{A}\in\mathbb{R}^{N\times k}$ is the atomic type one-hot matrix, $\mathbf{X}\in\mathbb{R}^{N\times 3}$ represents 3D coordinates, and $\mathbf{L}\in\mathbb{R}^{3\times 3}$ is the lattice basis.
- **Stage 1 (VAE Pretraining)**: The PDDFormer encoder $\mathcal{E}_\phi$ maps $\mathcal{M}$ to node-level latent representations $\mathbf{Z}\in\mathbb{R}^{N\times d}$. Three independent MLP decoders reconstruct $\tilde{\mathbf{A}},\tilde{\mathbf{X}},\tilde{\mathbf{L}}$, minimizing reconstruction loss and KL regularization end-to-end.
- **Stage 2 (Latent Diffusion)**: Partially fixing semantics, $\mathcal{E}_\phi$ and the DiT denoising network $\mathcal{F}_\theta$ are jointly trained. Flow matching is used in the latent space for linear interpolation $\mathbf{Z}^t=(1-t)\mathbf{Z}^0+t\mathbf{Z}^1$, with the goal of predicting the clean latent variable $\bar{\mathbf{Z}}^1$.
- **Finetune**: The pretrained $\mathcal{E}_\phi$ is connected to a READOUT + MLP for end-to-end finetuning on each property using MSE.
- **Output**: Property value $\hat{y}=\text{MLP}_\lambda(\text{READOUT}(\mathcal{E}_\phi(\mathcal{M})))$.

The entire framework is backbone-agnostic—replacing PDDFormer with Matformer works directly (see Table 2).

### Key Designs

1.  **Symmetry-aware VAE Encoder**:
    - **Function**: Compress heterogeneous $(\mathbf{A},\mathbf{X},\mathbf{L})$ into a unified continuous smooth latent space $\mathbf{Z}\in\mathbb{R}^{N\times d}$ while strictly preserving rotational and periodic translational invariance.
    - **Mechanism**: The encoder utilizes PDDFormer—one of the strongest equivariant Transformers for periodic crystals—which naturally satisfies $\mathcal{E}_\phi(\mathbf{A},\mathbf{QX},\mathbf{QL})=\mathcal{E}_\phi(\mathbf{A},\mathbf{X},\mathbf{L})$. Three independent MLP decoders output atomic types (cross-entropy), coordinates ($\ell_2$), and lattice ($\ell_2$). The total loss $\mathcal{L}_{\text{VAE}}=\mathcal{L}^{\mathbf{A}}_{\text{recon}}+\mathcal{L}^{\mathbf{X}}_{\text{recon}}+\mathcal{L}^{\mathbf{L}}_{\text{recon}}+\alpha\mathcal{L}_{\text{reg}}$, where $\mathcal{L}_{\text{reg}}=d_{\text{KL}}(q_\phi(\mathbf{Z}|\mathcal{M})\,\|\,p(\mathbf{Z}))$ pushes the latent distribution toward a standard Gaussian to stabilize variance for subsequent diffusion.
    - **Design Motivation**: Unlike CrysDiff/DPF, which run separate diffusions for different variable types (D3PM + DDPM + wrapped normal), this approach "flattens" heterogeneous variables into a unified continuous latent space via VAE. This ensures stage-2 diffusion only handles a simple distribution, while PDDFormer "encapsulates" equivariance within the encoder, removing the need for explicit symmetry constraints in the latent space.

2.  **Latent Flow Matching Diffusion**:
    - **Function**: Perform diffusion pretraining on the latent space learned in stage-1 to further refine $\mathcal{E}_\phi$, ensuring the latent representation is both reconstructible and follows a clean target distribution, thereby enhancing downstream transferability.
    - **Mechanism**: Define clean samples $\mathbf{Z}^1=\mathcal{E}_\phi(\mathcal{M})$ and Gaussian noise $\mathbf{Z}^0\sim\mathcal{N}(0,1)^{N\times d}$. After sampling $t\sim\mathcal{U}(0,1)$, linear interpolation is performed: $\mathbf{Z}^t=(1-t)\mathbf{Z}^0+t\mathbf{Z}^1$. The conditional vector field is $u_t(\mathbf{Z}^t|\mathbf{Z}^1)=(\mathbf{Z}^1-\mathbf{Z}^t)/(1-t)$. The DiT denoising network $\mathcal{F}_\theta$ predicts $\bar{\mathbf{Z}}^1=\mathcal{F}_\theta(\mathbf{Z}^t,t)$, and the loss simplifies to $\mathcal{L}_{\text{LDM}}=\frac{1}{(1-t)^2}\frac{1}{N}\sum_i\|\mathbf{z}^1_i-\bar{\mathbf{z}}^1_i\|^2$. Crucially, $\mathcal{E}_\phi$ and $\mathcal{F}_\theta$ are updated **jointly**, allowing diffusion signals to backpropagate to the encoder, making $\mathbf{Z}$ satisfy both "reconstructibility" and "denoisability."
    - **Design Motivation**: Latent space diffusion offers three benefits: (1) A single continuous Gaussian target avoids heterogeneous diffusions like D3PM/wrapped normal; (2) The low-dimensional, smooth nature of $\mathbf{Z}$ reduces DiT denoising steps and parameters; (3) The encoder captures finer structural and chemical information after being "refined" by the diffusion objective—Figure 3 shows that CrysLDNet's reconstruction accuracy for $A/X/L$ comprehensively outperforms CrysDiff and DPF, validating the benefit of latent diffusion on representational power.

3.  **Backbone-Agnostic Plug-and-Play**:
    - **Function**: Decouple the pretrain-finetune pipeline from the VAE encoder architecture, allowing future equivariant crystal Transformers to be substituted without rewriting losses, diffusion, or decoders.
    - **Mechanism**: The VAE decoders, DiT, loss, and objectives only depend on the latent representation shape $(N,d)$, independent of the encoder's internal neighborhood aggregation logic. Tests show that upgrading $\mathcal{E}_\phi$ from Matformer to PDDFormer yields average gains of 10.46% / 12.39% on JARVIS / MP (Table 2). Conversely, even with a weaker encoder like Matformer, CrysLDNet reduces MAE by 7.53% / 7.87% over basic Matformer, indicating that gains stem primarily from the "latent diffusion" paradigm.
    - **Design Motivation**: Architectures in crystal representation learning iterate rapidly (CGCNN → ALIGNN → Matformer → PDDFormer). Deeply coupling a pretraining framework to a specific encoder necessitates total redesigns for every upgrade. This backbone-agnostic design decouples the "pretraining paradigm" from the "backbone network," favoring long-term evolution.

### Loss & Training
- Stage 1: $\mathcal{L}_{\text{VAE}}=\mathcal{L}^{\mathbf{A}}_{\text{recon}}+\mathcal{L}^{\mathbf{X}}_{\text{recon}}+\mathcal{L}^{\mathbf{L}}_{\text{recon}}+\alpha\mathcal{L}_{\text{reg}}$ until convergence.
- Stage 2: $\mathcal{L}_{\text{LDM}}=\frac{1}{(1-t)^2}\frac{1}{N}\sum_i\|\mathbf{z}^1_i-\bar{\mathbf{z}}^1_i\|^2$, updating $\mathcal{E}_\phi$ and $\mathcal{F}_\theta$ **jointly**.
- Pretrain Data: 380,740 unlabeled crystal structures filtered from GNoME (duplicates and physically ambiguous entries removed).
- Finetune: $\mathcal{L}_{\text{MSE}}=\|\hat{y}-y\|^2$, with independent encoder copies fine-tuned for each property.

## Key Experimental Results

### Main Results: MAE Comparison on JARVIS-DFT and MP

The following table presents MAE for representative properties (lower is better), covering the strongest supervised baseline PDDFormer, diffusion pretraining DPF / CrysDiff, and the proposed CrysLDNet:

| Dataset | Property | PDDFormer | DPF | CrysDiff | CrysLDNet | Gain vs PDDFormer |
|--------|------|-----------|-----|----------|-----------|------|
| JARVIS | Formation Energy (eV/atom) | 0.027 | 0.029 | 0.029 | **0.026** | -3.7% |
| JARVIS | Bandgap OPT (eV) | 0.120 | 0.122 | 0.131 | **0.118** | -1.7% |
| JARVIS | Bandgap MBJ (eV) | 0.251 | 0.311 | 0.287 | **0.238** | -5.2% |
| JARVIS | Ehull (eV/atom) | 0.033 | 0.059 | 0.062 | **0.032** | -3.0% |
| JARVIS | Bulk Modulus (GPa) | 9.546 | 10.43 | 9.875 | **8.817** | -7.6% |
| JARVIS | Shear Modulus (GPa) | 8.808 | 9.596 | 9.191 | **8.428** | -4.3% |
| JARVIS | SLME (%) | 4.300 | 5.129 | 5.030 | **4.120** | -4.2% |
| MP | Formation Energy | 0.016 | 0.020 | – | **0.015** | -6.3% |
| MP | Bulk Modulus | 0.034 | 0.042 | – | **0.032** | -5.9% |
| MP | Shear Modulus | 0.062 | 0.073 | – | **0.059** | -4.8% |

Overall Average: CrysLDNet vs PDDFormer = **-4.26% (JARVIS) / -4.90% (MP)**; CrysLDNet vs DPF = **-16.76% / -19.34%**.

### Ablation Study

| Configuration | Formation | Bandgap OPT | Ehull | Bulk | Spillage | Description |
|------|-----------|------|------|------|------|------|
| VAE only | 0.031 | 0.126 | 0.059 | 10.61 | 0.374 | No LDM, Stage-1 reconstruction only |
| LDM only | 0.030 | 0.123 | 0.052 | 10.37 | 0.370 | No VAE, diffusion on raw space |
| Only A | 0.032 | 0.125 | 0.058 | 10.49 | 0.355 | Reconstruct atomic types only |
| Only X | 0.031 | 0.122 | 0.060 | 10.21 | 0.352 | Reconstruct coordinates only |
| Only L | 0.032 | 0.136 | 0.055 | 10.46 | 0.351 | Reconstruct lattice only |
| A + X | 0.034 | 0.125 | 0.052 | 10.25 | 0.358 | Reconstruct A and X |
| L + X | 0.033 | 0.124 | 0.046 | 10.51 | 0.354 | Reconstruct L and X |
| **CrysLDNet (Full)** | **0.026** | **0.118** | **0.032** | **8.817** | **0.340** | All reconstructions + LDM |

### Key Findings
- **VAE and LDM are complementary**: VAE-only or LDM-only reached only 10.61 / 10.37 on Bulk Modulus, far worse than the full model's 8.817. This proves that the "VAE flattens heterogeneous space, then LDM refines semantics" approach is essential.
- **Greater gains in low-data regimes**: Figure 2 shows that with only 20% / 40% finetune data, CrysLDNet (Matformer) outperforms PDDFormer trained on full data. With 40% data, CrysLDNet reduces MAE by 12.83% and 22.49% over PDDFormer and Matformer, respectively, demonstrating the "leverage effect" of pretraining.
- **Backbone-agnostic validity**: Upgrading the encoder from Matformer to PDDFormer resulted in an additional 10.46% / 12.39% improvement on JARVIS / MP, nearly proportional to the backbone's own improvement, proving the framework's future-proofing.
- **Correction of DFT systematic bias**: On OQMD-EXP experimental data, zero-shot MAE dropped from CrysGNN's 0.253 to CrysLDNet's 0.205. Finetuning with 20% experimental data further reduced it to 0.097 (vs CrysGNN's 0.135), proving latent pretraining helps bridge the DFT-to-experiment gap.
- **Reconstruction quality correlates with downstream performance**: Figure 3 shows CrysLDNet outperforms CrysDiff/DPF in reconstructing A/X/L on GNoME. This "representation power gap" directly reflects in downstream MAE, providing a clear causal chain.

## Highlights & Insights
- **Adapting Stable Diffusion's "Latent Space Diffusion" to Crystals**: The heterogeneity of crystals (discrete atoms + continuous lattice + periodic coordinates) mirrors the complexity of high-resolution RGB space where raw diffusion is inefficient. By applying the same solution—compressing to a VAE latent space—this work yields similar dividends (simplified modeling, enhanced expression).
- **Joint Training of $\mathcal{E}_\phi$ + $\mathcal{F}_\theta$ is the Essence**: While many might fix the VAE before training the LDM, stage-2 here allows diffusion gradients to backpropagate to the encoder. This "reshapes" the latent space with a diffusion objective, which is key to surpassing PDDFormer—the VAE-only version had a 10.61 Bulk MAE, which dropped to 8.817 after joint training.
- **Backbone-agnostic is an Honest Selling Point**: Many self-supervised frameworks actually derive improvements from backbone upgrades. This paper shows a 7.87% gain using the same backbone (Matformer), proving the paradigm's inherent value.
- **Transferability to Other "Heterogeneous 3D Structures"**: Molecules, proteins, and catalytic interfaces all combine discrete atoms with continuous coordinates and topological/periodic constraints. This template—"equivariant encoder → smooth latent space → latent flow matching"—can likely be applied to these domains as a lightweight yet effective alternative for SE(3)/E(3) equivariant diffusion research.

## Limitations & Future Work
- **Acknowledged Limitations**: Experiments primarily focus on JARVIS and MP benchmarks. Evaluation on more complex crystal types like alloys or glasses is missing. The OQMD-EXP dataset size is limited (~1500 samples), restricting the scale of DFT bias correction experiments.
- **Potential Methodological Issues**: (1) There is no sensitivity analysis for the KL regularization weight $\alpha$; excessive strength might hurt reconstruction precision, while weakness could lead to a non-smooth latent space—both impacting LDM stability. (2) DiT uses self-attention on $N$ atom tokens, which may become computationally expensive for very large cells (e.g., moiré/supercells). (3) All experiments use independent per-property finetuning without exploring multi-task or zero-shot prompt settings.
- **Future Directions**: (a) Introduce conditional LDM (conditional flow matching) to inject property labels during pretraining for semi-supervised training. (b) Use LoRA or adapters to allow one pretrained encoder to serve multiple properties, reducing deployment costs. (c) Unify latent diffusion with generation tasks—enabling both property prediction and inverse generation of new crystals satisfying specific constraints.

## Related Work & Insights
- **vs CrysDiff (Song et al. 2024)**: CrysDiff performs diffusion on A/X/L simultaneously in raw space (D3PM + DDPM + wrapped normal), leading to complex architectures and many diffusion steps. This work uses VAE to flatten the space, resulting in a simpler, more efficient structure that beats CrysDiff on JARVIS Bulk Modulus by 10.7%.
- **vs DPF (Shen et al. 2025a)**: DPF also uses diffusion pretraining but performs it in the feature space using Matformer. CrysLDNet uses PDDFormer + latent diffusion, outperforming DPF by 16.76% overall on JARVIS. Table 2 confirms that even when using Matformer to match DPF, the latent approach is superior.
- **vs CrysGNN / Crystal Twins (2022-2023)**: These early reconstruction-based or contrastive self-supervised methods lack diffusion. Their downstream MAE is significantly higher (e.g., 13.41 Bulk MAE vs 8.817), proving that generative SSL > contrastive SSL trends hold for materials GNNs.
- **vs Latent Diffusion in Stable Diffusion / DALL-E 2**: The methodology is homologous—compressing high-dimensional heterogeneous data into a continuous low-dimensional VAE latent space for diffusion. However, the purpose differs: CV seeks high-resolution generation, whereas this work seeks robust representation pretraining.

## Rating
- Novelty: ⭐⭐⭐⭐ Cleanly transfers the mature latent diffusion paradigm to crystal pretraining; methods are clear but not fundamentally disruptive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 13 properties across two major datasets + backbone-agnostic tests + low-data analysis + bias correction + full ablation; highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative, well-defined formulas and algorithm steps; sensitivity analysis for some hyperparameters could be expanded.
- Value: ⭐⭐⭐⭐⭐ Provides a reusable pretraining paradigm in a domain where labeling is expensive and backbones iterate quickly; high engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MIRO: Multi-Reward Conditioned Pretraining Simultaneously Enhances T2I Quality and Efficiency](miro_multi-reward_conditioned_pretraining_improves_t2i_quality_and_efficiency.md)
- [\[NeurIPS 2025\] LLM Meets Diffusion: A Hybrid Framework for Crystal Material Generation](../../NeurIPS2025/image_generation/llm_meets_diffusion_a_hybrid_framework_for_crystal_material_generation.md)
- [\[ICML 2026\] Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces](discrete_diffusion_samplers_and_bridges_off-policy_algorithms_and_applications_i.md)
- [\[ICML 2026\] Transferable Multi-Bit Watermarking Across Frozen Diffusion Models via Latent Consistency Bridges](transferable_multi-bit_watermarking_across_frozen_diffusion_models_via_latent_co.md)
- [\[CVPR 2026\] Diffusion Probe: Generated Image Result Prediction Using CNN Probes](../../CVPR2026/image_generation/diffusion_probe_generated_image_result_prediction_using_cnn_probes.md)

</div>

<!-- RELATED:END -->
