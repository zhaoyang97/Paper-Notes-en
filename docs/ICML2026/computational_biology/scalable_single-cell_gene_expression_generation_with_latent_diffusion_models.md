---
title: >-
  [Paper Note] Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models
description: >-
  [ICML 2026][Computational Biology][Single-cell RNA-seq] scLDM utilizes a unified Multi-head Cross-Attention Block (MCAB) to encode exchangeable gene expression data into fixed-length…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Single-cell RNA-seq"
  - "exchangeability"
  - "multi-head cross-attention"
  - "latent diffusion"
  - "flow matching"
  - "multi-condition CFG"
date: 2026-05-08
content_hash: a105a9b30135b17f
---

# Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models

**Conference**: ICML 2026  
**arXiv**: [2511.02986](https://arxiv.org/abs/2511.02986)  
**Code**: https://github.com/czi-ai/scldm (Available)  
**Area**: Computational Biology / Single-cell Transcriptomics / Latent Diffusion Models / Transformer VAE  
**Keywords**: Single-cell RNA-seq, exchangeability, multi-head cross-attention, latent diffusion, flow matching, multi-condition CFG

## TL;DR
scLDM utilizes a unified Multi-head Cross-Attention Block (MCAB) to encode exchangeable gene expression data into fixed-length, permutation-invariant latent token sets. By replacing the Gaussian prior with DiT, flow matching, and joint multi-attribute classifier-free guidance, it significantly outperforms scVI, scDiffusion, and CFGen across reconstruction, conditional/unconditional generation, and perturbation response prediction tasks on multiple scRNA-seq datasets.

## Background & Motivation

**Background**: Single-cell RNA-seq (scRNA-seq) enables the simultaneous measurement of expressions for millions of cells across tens of thousands of genes, facilitating studies on cell differentiation, disease progression, and drug perturbations. Prevailing generative modeling approaches include: (i) VAE-based (scVI / scVAEDer), (ii) GAN-based (scGAN), (iii) diffusion-based (scDiffusion), and recent latent diffusion with flow matching (CFGen).

**Limitations of Prior Work**:

- Almost all methods treat gene expression as a "fixed-order vector," strictly binding the $i$-th dimension to gene $g_i$. This forces the input dimension to match the gene vocabulary size, requiring the selection of a "Highly Variable Gene" (HVG) subset and retraining for different tissues or species.
- This "position-based encoding" contradicts biological reality: gene expression is an **exchangeable set**, where the ordering is arbitrary.
- GAN-based methods suffer from training instability and mode collapse; MLP-based autoencoders have limited capacity and show diminishing returns when scaling.
- Over 70% of scRNA-seq data consists of zeros (dropout). Feeding all zeros into a Transformer is computationally wasteful and dilutes signals.

**Key Challenge**: Simultaneously achieving (a) a truly exchangeable probabilistic model, (b) a Transformer architecture scalable to large vocabularies and contexts, (c) precise modeling of count data (NB distribution), and (d) support for multi-attribute controllable generation. Existing methods typically satisfy only one or two of these requirements.

**Goal**:

1. Design a **permutation-invariant (encoding) and permutation-equivariant (decoding) Transformer-based VAE** where the number of latent variables is fixed and decoupled from the number of input genes.
2. Train an LDM using **DiT, linear interpolation, and flow matching** in the latent space to replace the Gaussian prior, supporting joint multi-attribute CFG.
3. Validate performance across five tasks: reconstruction, unconditional generation, conditional generation, perturbation prediction, and downstream classification.

**Key Insight**: Tools like SetTransformer and Perceiver IO already use learnable pseudo-inputs to pool variable-length inputs into a fixed-length token set. By replacing generic pseudo-inputs with gene embeddings, one can achieve (i) permutation-invariant pooling during encoding and (ii) permutation-equivariant unpooling during decoding—**using a single block for both**, simplifying the separate pool/unpool architectures of SetTransformer.

**Core Idea**: Utilize a **unified Multi-head Cross-Attention Block (MCAB)** for both permutation-invariant pooling in the encoder and permutation-equivariant unpooling in the decoder. Run DiT-based latent diffusion on a fixed-size latent space to ensure the model is naturally invariant to gene order, scalable to vocabulary size, precise in count modeling, and capable of multi-conditional generation.

## Method

### Overall Architecture

The scLDM pipeline consists of two-stage training and one-stage sampling:

- **Input**: Each cell is represented as $(\mathbf{x}_{\mathcal{I}}, \mathcal{I})$, where $\mathcal{I}$ is the set of gene IDs (not position indices) and $\mathbf{x}_{\mathcal{I}}$ is the corresponding integer count vector.
- **Stage 1 (VAE)**: Sparse filtering $G$ retains only genes with expression $>0$ (padded with PAD if count $<d$). Embeddings combine counts and gene IDs. $\mathrm{MCAB}_{\mathbf{S}}$ uses $m$ learnable pseudo-inputs $\mathbf{S}$ to pool the input into a fixed $m \times D$ token matrix $\mathbf{Z}$. Transformer blocks then output the Gaussian posterior $\mu, \sigma^2$. The decoder reverses this: $\mathbf{Z}$ passes through Transformer blocks followed by $\mathrm{MCAB}_{\mathbf{E}_{\mathcal{I}}}$ (pseudo-inputs replaced by query gene embeddings $\mathbf{E}_{\mathcal{I}}$), outputting params $(r, p)$ for the Negative Binomial distribution.
- **Stage 2 (LDM)**: The VAE is frozen. Latent tokens $\mathbf{Z}$ are treated as a sequence for DiT. A score model is trained using linear interpolation and flow matching loss to replace the standard Gaussian prior. Conditional signals $\mathbf{y} \in \{0,1\}^J$ (cell type, perturbation, batch, etc.) are injected via classifier-free guidance.
- **Sampling**: $\mathbf{Z}$ is sampled from the LDM, then passed through the VAE decoder to sample NB counts.

### Key Designs

1. **Unified MCAB (Invariant Encoding + Equivariant Decoding)**:

    - **Function**: A single block performs "set to fixed tokens" pooling and "fixed tokens to arbitrary gene subset" unpooling.
    - **Mechanism**: Define $\mathrm{MCAB}_{\mathbf{S}}(\mathbf{X}) = F(\mathbf{X},\mathbf{S}) + \mathrm{MLP}(\mathrm{LN}(F(\mathbf{X},\mathbf{S})))$, where $F$ uses $\mathbf{S}$ as queries and $\mathbf{X}$ as keys/values for multi-head cross-attention. In the Encoder, $\mathbf{S}$ consists of $m$ **ID-independent** learnable vectors. Permuting $\mathbf{X}$ leaves $\mathbf{S}$ and the attention output unchanged, ensuring $\mathbf{Z}$ is invariant. In the Decoder, $\mathbf{S} = \mathbf{E}_{\mathcal{I}}$ (embeddings of query genes). Permuting $\mathcal{I}$ permutes the rows of $\mathbf{S}$, resulting in an equivalent permutation of the output, ensuring equivariance.
    - **Design Motivation**: Traditional solutions (SetTransformer) require separate modules (PMA/ISAB) with different biases. MCAB unifies these properties in one block, enabling parameter sharing and stability. It decouples the latent size $m$ from the gene vocabulary size; the latter only interacts via the embedding matrix $\mathbf{E}$, allowing cross-species transfer by extending $\mathbf{E}$ without modifying the core network.

2. **Sparsity-aware Input Processing $G(\mathbf{x},\mathcal{I})$**:

    - **Function**: Compresses high-dimensional sparse count vectors into dense token sequences of length $d$.
    - **Mechanism**: Select $\mathcal{J} = \{i : x_i > 0\}$ (expressed genes). If $|\mathcal{J}| < d$, fill with a PAD token (count 0, index PAD): $\mathrm{Out} = \{(x_i, i)\}_{i \in \mathcal{J}} \cup \{(0, \mathrm{PAD})\}^{d - |\mathcal{J}|}$. The embedding layer concatenates the count and gene embedding: $\mathrm{Emb}(\bar{\mathbf{x}}_{\mathcal{J}}, \mathcal{J}) = \mathrm{Linear}(\mathrm{repeat}_d(\bar{\mathbf{x}}_{\mathcal{J}}) \,\Vert\, \mathbf{E}_{\mathcal{J}})$.
    - **Design Motivation**: Over 70% of scRNA-seq values are dropout. Removing zeros saves $O(D^2)$ computation and avoids signal dilution. This is only **encoder-side context cropping**; the decoder still outputs NB parameters for all $\mathcal{I}$. Since NB naturally concentrates probability mass at zero, the model's ability to represent structural zeros is preserved.

3. **DiT Latent Diffusion + Joint Multi-attribute CFG**:

    - **Function**: Replaces the basic Gaussian prior with a controllable generative model supporting simultaneous conditioning on multiple attributes.
    - **Mechanism**: $m$ latent tokens serve as the DiT input sequence. A velocity field $v_{t,\epsilon}(\mathbf{Z}; y)$ is trained using linear interpolation and flow matching loss. Multi-attribute conditions use **Joint CFG**: $\tilde{v}_{t,\epsilon}(\mathbf{Z}, y) = v_{t,\epsilon}(\mathbf{Z}; \mathrm{Null}) + \omega [v_{t,\epsilon}(\mathbf{Z}; y) - v_{t,\epsilon}(\mathbf{Z}; \mathrm{Null})]$. The attribute vector $\mathbf{y} \in \{0,1\}^J$ is treated as a **singular condition** rather than the additive decomposition approach used in CFGen.
    - **Design Motivation**: (i) The aggregated posterior of LDM is more complex than $\mathcal{N}(0, I)$; a strong prior prevents the "prior-posterior mismatch" that collapses generation quality. (ii) Additive CFG assumes one-hot attributes and fails to represent combinations (e.g., "perturbation A + cell type B"). Joint CFG encodes these combinations directly into conditional embeddings, crucial for perturbation benchmarks.

### Loss & Training

- **Stage 1**: $\beta$-VAE ELBO: $\mathcal{L} = \mathbb{E}_q[\ln p(\mathbf{x}_{\mathcal{I}} | \eta(\mathbf{Z}, \mathcal{I}))] - \beta \cdot \mathrm{KL}(q(\mathbf{Z}|\mathbf{x}_{\mathcal{I}}) \,\Vert\, p(\mathbf{Z}))$. Count likelihood uses NB. Setting $\beta = 0$ reduces it to a deterministic autoencoder.
- **Stage 2**: Freeze VAE. DiT is trained with flow matching loss $\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t, \mathbf{Z}_0, \mathbf{Z}_1, \mathbf{y}} \| v_{t,\epsilon}(\mathbf{Z}_t; \mathbf{y}) - (\mathbf{Z}_1 - \mathbf{Z}_0) \|^2$ for linear interpolation $\mathbf{Z}_t = (1-t)\mathbf{Z}_0 + t \mathbf{Z}_1$. CFG drop-out probability $\rho$ determines the Null conditioning frequency. Sampling uses the SiT (Scalable Interpolant Transformers) library.

## Key Experimental Results

### Main Results

**Table 1: Cell Reconstruction (NB Likelihood + Pearson + MSE)**

| Dataset | Model | RE ↓ | PCC ↑ | MSE ↓ |
|--------|------|------|-------|-------|
| Dentate Gyrus | scVI | 5193.2 | 0.058 | 0.378 |
| Dentate Gyrus | CFGen | 5468.8 | 0.076 | 0.253 |
| Dentate Gyrus | **Ours (NB)** | **4571.6** | **0.273** | **0.206** |
| Tabula Muris | scVI | 5588.2 | 0.221 | 0.132 |
| Tabula Muris | CFGen | 5547.6 | 0.136 | 0.127 |
| Tabula Muris | **Ours (NB)** | **4993.6** | **0.376** | **0.106** |
| HLCA | scVI | 5659.2 | 0.125 | 0.238 |
| HLCA | CFGen | 5428.7 | 0.146 | 0.117 |
| HLCA | **Ours (NB)** | **4898.9** | **0.310** | **0.095** |

PCC on Tabula Muris is 0.376 vs CFGen 0.136 (**nearly 3x improvement**), demonstrating that Transformer-VAEs reconstruct complex cell populations significantly better than MLP-based VAEs.

**Table 2: (Un)conditional Generation (HVG, Wasserstein-2 / MMD / 1-NN accuracy → 0.5 / Precision / Recall)**

| Dataset | Setting | Model | W2 ↓ | MMD² RBF ↓ | 1-NN →0.5 | Prec ↑ | Rec ↑ |
|--------|------|------|------|-----------|----------|--------|-------|
| Dentate Gyrus | Uncond | CFGen | 12.617 | 0.022 | 0.856 | 0.278 | **0.385** |
| Dentate Gyrus | Uncond | **Ours (NB)** | **10.710** | **0.017** | **0.709** | **0.664** | 0.291 |
| Tabula Muris | Uncond | CFGen | 11.658 | 0.008 | 0.773 | 0.255 | 0.591 |
| Tabula Muris | Uncond | **Ours (NB)** | **7.267** | **0.002** | **0.596** | **0.539** | **0.608** |
| HLCA | Uncond | CFGen | 12.433 | 0.007 | 0.760 | 0.272 | 0.583 |
| HLCA | Uncond | **Ours (NB)** | **9.272** | **0.004** | **0.605** | — | — |

W2 is nearly halved on Tabula Muris (7.27 vs 11.66). The 1-NN classifier accuracy drops from 0.77 towards 0.60 (closer to 0.5 is better), indicating generated samples are harder to distinguish from real ones.

### Ablation Study

| Configuration | Key Finding |
|------|---------|
| Ours (NB) | Optimal performance across all metrics (W2, PCC). |
| Ours (Gauss) | Metric collapse when using Gaussian likelihood; proves NB is essential for counts. |
| w/o LDM (Gauss Prior) | Significant drop in generation quality; confirms mismatch bottleneck in VAE priors. |
| Additive CFG vs Joint CFG | Joint CFG is superior for multi-attribute perturbation benchmarks. |
| Input Filtering | Reconstruction metrics remain consistent or improve; optimizes compute without loss. |
| MCAB vs SetTransformer | MCAB outperforms separate pooling/unpooling operators. |

### Key Findings

- **NB Likelihood is Mandatory**: Replacing NB with Gaussian causes metrics to collapse to or below scDiffusion levels. Discrete distribution modeling (NB) is necessary for capturing zero-inflation.
- **PCC Gains Exceed W2 Gains**: The jump from 0.14 to 0.38 in PCC shows MCAB Transformers are qualitatively better at preserving inter-cellular variations, directly benefiting downstream classification.
- **Joint CFG > Additive CFG**: For non-mutually exclusive attribute combinations (e.g., cell type + drug), joint encoding captures interactions essential for perturbation prediction.
- **Sparse Filtering is a Free Lunch**: Excluding 70% of zeros actually improves reconstruction because the encoder focuses on signal, while the decoder recovers structural zeros via NB modeling.

## Highlights & Insights

- **Dual-use MCAB**: The same attention mechanism switches between invariant and equivariant semantics based on pseudo-input properties. This elegant parameterization of dual symmetries could apply to any set-to-set task (e.g., point clouds or atomic force fields).
- **Decoupling Genes from Positions**: By moving genes from the "position dimension" to the "embedding dimension," scLDM follows the maturation of NLP where tokenizers decouple vocabulary from architecture. This allows for zero-shot species transfer by extending the lookup table.
- **The Success of the Stable Diffusion Paradigm**: VAE dimension reduction followed by DiT latent diffusion and CFG works natively on biological count data, outperforming domain-specific baselines.
- **Engineering Value of Joint CFG**: Identifying the failure of additive CFG for multi-attribute combinations and providing a joint-encoding solution is a critical refinement for controllable generation research.

## Limitations & Future Work

- **High Training Cost**: Two-stage training (VAE followed by DiT) is computationally heavy; end-to-end training remains unexplored.
- **Heuristic Token Count $m$**: A fixed $m$ is used for datasets with vastly different numbers of genes, which may be suboptimal for extremely large-scale data.
- **Cross-species Zero-shot**: While the architecture supports it, the current work does not demonstrate transfer from mouse pre-training to human inference.
- **PAD Token Bias**: Mapping all non-expressed genes to one PAD token collapses the distinction between structural zeros and dropout; an explicit mask might be more robust.

## Related Work & Insights

- **vs CFGen**: Both use latent flow matching, but CFGen relies on MLP-based VAEs and additive CFG. scLDM’s upgrade to Transformer-based VAEs and Joint CFG results in a comprehensive performance win.
- **vs scDiffusion**: scDiffusion operates in raw expression space without latent compression, which is expensive and poor at modeling count discreteness.
- **vs SetTransformer / SetVAE**: scLDM simplifies the separate pooling and unpooling blocks into a single unified MCAB, enhancing parameter sharing.
- **vs Perceiver IO**: While the encoder is similar, scLDM’s novelty lies in reusing the same block for the decoder by swapping queries with gene embeddings.
- **Insight**: The victory of exchangeability as an architectural inductive bias over data augmentation reinforces the core tenets of Geometric Deep Learning in the omics domain.

## Rating
- Novelty: ⭐⭐⭐⭐ MCAB dual-usage and Joint CFG are genuine architectural contributions within the bio-domain.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 3 datasets and 5 distinct tasks plus ablations.
- Writing Quality: ⭐⭐⭐⭐ Rigorous math, though MCAB patterns could benefit from more visual diagrams.
- Value: ⭐⭐⭐⭐⭐ Provides a SOTA, scalable, and open-source foundation for set-structured scientific data generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Universal Gene Regulatory Network Inference: Unlocking Generalizable Regulatory Knowledge in Single-cell Foundation Models](towards_universal_gene_regulatory_network_inference_unlocking_generalizable_regu.md)
- [\[CVPR 2026\] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](../../CVPR2026/computational_biology/adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)
- [\[AAAI 2026\] Gene Incremental Learning for Single-Cell Transcriptomics](../../AAAI2026/computational_biology/gene_incremental_learning_for_single-cell_transcriptomics.md)
- [\[ICML 2026\] Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models](temporal_score_rescaling_for_temperature_sampling_in_diffusion_and_flow_models.md)
- [\[CVPR 2026\] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](../../CVPR2026/computational_biology/cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)

</div>

<!-- RELATED:END -->
