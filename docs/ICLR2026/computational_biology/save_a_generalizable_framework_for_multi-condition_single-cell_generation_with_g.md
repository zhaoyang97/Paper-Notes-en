---
title: >-
  [Paper Note] SAVE: A Generalizable Framework for Multi-Condition Single-Cell Generation with Gene Block Attention
description: >-
  [ICLR 2026][Computational Biology][Flow Matching] SAVE aggregates thousands of genes into several "gene blocks" based on LLM semantic similarity. It performs Transformer attention at the block level, combined with a Variational Autoencoder (VAE) for compression and Latent Flow Matching for generation. By using AdaLN to inject conditions and condition masking to unify
tags:
  - ICLR 2026
  - Computational Biology
  - Flow Matching
date: 2026-05-08
content_hash: 900419ca5f9b55c0
---
# SAVE: A Generalizable Framework for Multi-Condition Single-Cell Generation with Gene Block Attention

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=l7QEoK4uDP](https://openreview.net/forum?id=l7QEoK4uDP)  
**Code**: https://github.com/fdu-wangfeilab/sc-save  
**Area**: Computational Biology / Single-cell Generation / Conditional Generation / Flow Matching  
**Keywords**: Single-cell RNA sequencing, Gene Block Attention, Conditional Generation, Flow Matching, Batch Correction

## TL;DR
SAVE aggregates thousands of genes into several "gene blocks" based on LLM semantic similarity. It performs Transformer attention at the block level, combined with a Variational Autoencoder (VAE) for compression and Latent Flow Matching for generation. By using AdaLN to inject conditions and condition masking to unify generation and transfer tasks, SAVE significantly outperforms existing methods across conditional generation, batch correction, and perturbation prediction, particularly in low-resource and unseen condition configurations.

## Background & Motivation

**Background**: Single-cell RNA sequencing (scRNA-seq) enables the observation of gene expression at cellular resolution. Researchers aim to use generative models to simulate cellular states under various combinations of conditions (cell types, disease states, drug perturbations, sequencing batches, etc.) to reduce the need for expensive wet-lab experiments. Current mainstream approaches include VAEs like scVI (modeling zero-inflated negative binomial likelihood and encoding covariates into latent space) and Transformer "foundation models" like scGPT, scBERT, and Geneformer (treating each gene as a token and training via masked modeling).

**Limitations of Prior Work**: Both approaches have structural flaws. VAE architectures are often too simple to model complex interactions between multiple external conditions. Transformer foundation models treat genes as **flat, token-level** independent units, ignoring high-order biological structures such as gene modules or pathways. Furthermore, they often focus only on non-zero expression values, losing the information-rich zero inflation in scRNA-seq. Crucially, most are organized as encoders and lack a generative framework capable of **sampling from learned conditional distributions**.

**Key Challenge**: Gene expression data is high-dimensional, sparse, and has **no natural spatial order**. To model high-order dependencies, a natural approach is to create "coarse-grained" representations similar to patches in ViT. However, unlike pixels, genes lack spatial proximity for grouping. Determining a biologically meaningful way to "block" unordered genes is the core obstacle to applying coarse-grained modeling to single cells.

**Goal**: (1) Identify a semantically reasonable block representation for genes to capture high-order relationships. (2) Integrate it into a sampling-capable conditional generative framework. (3) Enable the model to generalize to condition combinations not seen during training.

**Key Insight**: The authors draw inspiration from masked generative modeling in vision (e.g., MaskGIT), where unordered data can be effectively modeled using coarse-grained representations like image patches. Since genes are unordered but can be clustered, they use LLMs pre-trained on massive text corpora to read NCBI gene functional descriptions and extract semantic features. Genes with similar semantics are aggregated into "gene blocks" for block-level attention.

**Core Idea**: Replace "single-gene tokens" with "LLM-semantically clustered gene blocks" in a coarse-grained Transformer. This is integrated into a framework featuring VAE compression, Latent Flow Matching, AdaLN condition injection, and condition masking to unify multi-condition single-cell generation and transfer tasks.

## Method

### Overall Architecture
SAVE (Single-cell gene block Attention-based Variational gEnerative framework) is a Latent Flow Matching (LFM) framework consisting of three modules in series: **restructuring flat expression profiles into "gene block" tensors**, **compressing cells into latents using a VAE with Gene Block Attention**, and **training a conditional Flow Matching network on the latents to generate new cells from noise**. Generated latents are then restored to gene expression profiles via the VAE decoder.

Specifically: The input is a scRNA-seq matrix $Y \in \mathbb{R}^{N \times G}$ ($N$ cells, $G$ genes). Step 1, Gene Block Construction, clusters $G$ genes into $L$ equal-sized blocks (each with $K$ genes) based on semantics, rearranging $Y$ into a structured tensor $X \in \mathbb{R}^{N \times L \times K}$. Step 2, the VAE encoder, performs Transformer attention at the block level to output latent distribution parameters $\mu, \sigma^2$ and sample latents; the decoder mirrors this process. Step 3, the conditional Flow Matching network learns a time-dependent velocity field $v_\theta$ to transport prior noise $x_0$ to data latent $x_1$ along a straight path. Conditions (cell type, disease, batch, etc.) are injected via AdaLN. During training, **condition masking** is used: conditions are randomly replaced with `[MASK]` to unify generation and transfer tasks and facilitate generalization to unseen combinations.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input scRNA-seq<br/>Y (N×G)"] --> B["Gene Block Construction<br/>LLM Semantic Embeddings<br/>+ Optimal Transport Clustering"]
    B --> C["Block Tensor X (N×L×K)"]
    C --> D["Gene Block Attention VAE<br/>Block-level Transformer Compression"]
    D --> E["Latent Representation"]
    E --> F["Conditional Flow Matching<br/>Noise → Latent Velocity Field"]
    G["AdaLN Condition Injection<br/>+ Condition Masking"] -->|Condition s, [MASK]| F
    F --> H["Generated Latent"]
    H --> I["VAE Decoder<br/>Reconstruct scRNA-seq"]
```

### Key Designs

**1. Gene Block Construction: Semantic Clustering of Unordered Genes via LLM**

This directly addresses the challenge that genes lack natural local structure. SAVE constructs blocks in two stages. First, LLM Embedding Construction: functional "Summary" descriptions for each gene are retrieved from NCBI Gene, cleaned of uninformative content, and encoded into 1536-dimensional embedding vectors $g_i$ using an LLM (`text-embedding-ada-002`). Second, Iterative Clustering via Optimal Transport (OT): the task of partitioning $G$ embeddings into $L$ equal-sized, non-overlapping blocks is modeled as a transport problem with uniform marginal constraints. The Euclidean cost matrix $C_{ij} = \|g_i - c_j^{(t)}\|_2^2$ is solved under constraints $T\mathbf{1}_L = a, T^\top \mathbf{1}_G = b$ to update block centroids $c_j^{(t)}$ until convergence. OT is preferred over K-means as it **enforces equal block sizes**, ensuring regular tensor shapes and balanced computation.

**2. Gene Block Attention: Shifting Attention to the Block Level**

Applying attention directly to genes is computationally prohibitive due to sequence lengths in the tens of thousands. SAVE performs standard Transformer attention on $L$ blocks. Each block is first projected into an $e$-dimensional space via an MLP $W^{in}$, followed by layers with pre-LayerNorm:

$$h_0 = X W^{in},\quad h_{t'} = h_t + \mathrm{Attention}(\mathrm{LayerNorm}(h_t)),\quad h_{t+1} = h_{t'} + \mathrm{FeedForward}(\mathrm{LayerNorm}(h_{t'}))$$

Reducing sequence length from the number of genes $G$ to the number of blocks $L$ significantly compresses the quadratic complexity of self-attention. Ablation studies show a **191×** training speedup compared to naive gene-level attention ($K=1$). Furthermore, block representations naturally aggregate semantically related genes, allowing the model to learn high-order transcriptional dependencies between **modules** rather than noisy gene-to-gene relationships.

**3. AdaLN Condition Injection + Condition Masking: Unified Generation and Generalization**

Dealing with multi-condition single cells requires decoupling and modeling multiple covariates. SAVE encodes all conditions into a matrix $S \in \mathbb{R}^{N \times d_s}$. Categorical conditions are mapped to learnable embeddings $S_E$. Adaptive Layer Normalization (AdaLN) modulates the representation by deriving scaling and shifting parameters $\alpha, \beta, \gamma$ from $S_E$:

$$h_{t'} = h_t + \alpha_1 \cdot \mathrm{Attention}(\mathrm{AdaLN}(h_t, \gamma_1, \beta_1)),\quad \mathrm{AdaLN}(h,\gamma,\beta) = \frac{h - \mathbb{E}[h]}{\sqrt{\mathrm{Var}[h]+\epsilon}}\cdot\gamma + \beta$$

**Condition Masking** is applied during training: each element of $S$ is independently replaced by `[MASK]` with probability $p=0.6$. This allows the model to generate robustly when some conditions are unknown and unifies "generation" (unconditional Flow Matching) and "transfer/batch correction" (masked VAE) into a single training objective.

**4. Latent Flow Matching + VAE: A Sampleable Generative Core**

The VAE applies a Gaussian prior to the latent space. The encoder outputs $\mu, \sigma^2$ with reparameterized sampling and KL regularization $L_p = D_{KL}(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,1))$. Reconstruction $L_{recon} = -\log L(\hat X | X)$ is handled by the decoder. The generative core is conditional Flow Matching: a linear interpolation (affine probability path) $x_t = (1-t)x_0 + t x_1$ connects noise $x_0$ and data latent $x_1$. The target velocity field is the derivative $u_t = x_1 - x_0$, which the network $v_\theta(x_t, t, s)$ regresses via MSE: $L_{FM} = \mathbb{E}\,\|v_\theta(x_t,t,s) - u_t\|^2$. Inference involves solving the ODE $\mathrm{d}x_t/\mathrm{d}t = v_\theta(x_t,t,s)$ using Classifier-Free Guidance.

### Loss & Training
The total loss consists of: VAE KL regularization $L_p$, reconstruction loss $L_{recon}$, and Flow Matching MSE $L_{FM}$. Data is normalized to $10^4$ counts, log-transformed, and max-abs scaled to $[0,1]$. Default settings include block size $K=3200$, masking ratio 0.6, and AdamW optimizer.

## Key Experimental Results

### Main Results

For conditional generation, Wasserstein Distance (WD↓) and MMD↓ measure distributional similarity between generated and real cells.

| Task / Dataset | Metric | SAVE | Next Best | Note |
|--------------|------|------|----------|------|
| Single · Dentate gyrus | WD / MMD | **9.16 / 0.17** | 21.55 / 1.12 (CFGen) | WD/MMD reduced by >50% |
| Single · Tabula Muris | MMD | **0.04** | 0.19 (CFGen) | High alignment with global mean |
| Dual · Heart | WD / MMD | **8.30 / 0.63** | 12.57 / 0.66 (CFGen) | Batch + Cell Type covariates |
| Dual · PBMC | WD / MMD | **5.37 / 0.29** | 11.38 / 0.48 (scDiff) | — |
| Dual · Lung Atlas | WD / MMD | **4.37 / 1.14** | 13.89 / 1.71 (scDiff) | — |
| Multi · Lung Cancer (Unseen) | WD | **4.63** | 5.29 (scDiff) | 13/24 unseen combinations |

In batch correction (Bio. conservation↑ / Batch correction↑ / scIB↑), SAVE achieved the highest Bio. scores across three datasets and lead in overall scIB scores (e.g., Lung Atlas 0.81), indicating superior decoupling of biological variation from technical noise.

For perturbation prediction (PBMC-IFN), SAVE achieved average PCC > 0.95 and $R^2$ of 0.86, outperforming all baselines including CellOT and scGEN.

### Ablation Study

| Configuration | WD ↓ | MMD ↓ | Note |
|------|------|-------|------|
| SAVE (Full, Heart) | 8.30 | 0.63 | Complete model |
| SAVE w/o Gene Block Attention | 8.89 | 0.65 | WD degrades significantly |

Impact of gene block size $K$ on efficiency and quality:

| $K$ | Blocks $L$ | WD ↓ | MMD ↓ | Training Time (min) |
|-----|---------|------|-------|----------------|
| 1 (Per-gene) | 19112 | — | — | 2391.2 |
| 1600 | 12 | 9.64 | 0.65 | 16.4 |
| 3200 | 6 | **8.30** | 0.63 | 12.5 |
| 5600 | 4 | 8.41 | 0.63 | 9.0 |

### Key Findings
- **Gene Block Attention is the core contribution**: Removing it increases WD from 8.30 to 8.89. Moving from per-gene ($K=1$) to block-level ($K=3200$) provides a **191×** speedup with better generation quality.
- **Optimal Block Size**: $K=3200$ is the "sweet spot" for the Heart dataset.
- **Superiority in Complex/Unseen Scenarios**: While simpler baselines compete on basic datasets, SAVE's robustness becomes evident as complexity and the number of unseen condition combinations increase.

## Highlights & Insights
- **Semantic LLM Embeddings as Prior Knowledge**: Using LLMs to measure gene similarity based on NCBI functional descriptions effectively injects human biological knowledge into the model's structural design.
- **OT for Balanced Clustering**: Unlike K-means, the use of Optimal Transport ensures equal block sizes, making tensorization and attention calculation highly efficient.
- **Unified Task Handling**: The same masking mechanism enables both sample generation (masking Flow Matching) and batch correction/transfer (masking VAE).
- **Transferable "Coarse-grained Token" Concept**: This paradigm of "semantic grouping + block attention" could be applied to any high-dimensional, unordered tabular data (proteomics, metabolomics, etc.) to reduce compute while introducing structural priors.

## Limitations & Future Work
- **Dependency on Annotation Quality**: The LLM-based block construction relies on the completeness of NCBI functional descriptions; poorly annotated genes may result in noisy embeddings.
- **Historical Bias**: Purely text-driven unsupervised grouping may inherit historical biases present in biological literature.
- **Future Directions**: Integrating structured biological Knowledge Graphs (KGs) or curated gene regulatory networks (GRNs) could provide more reliable supervision for block partitioning.

## Related Work & Insights
- **vs scVI**: While scVI is excellent for batch correction, it struggles with multiple condition interactions. SAVE extends this by adding block-level Transformers and a Flow Matching generative core.
- **vs scGPT/Geneformer**: These treat genes as flat tokens and primarily focus on encoding. SAVE introduces semantic blocks and a proper sampling-based generative framework.
- **vs scDiffusion/CFGen**: These rely on fine-grained diffusion, which scale poorly. SAVE's affine path Flow Matching combined with block-level latents offers better scalability and fidelity.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Universal Gene Regulatory Network Inference: Unlocking Generalizable Regulatory Knowledge in Single-cell Foundation Models](../../ICML2026/computational_biology/towards_universal_gene_regulatory_network_inference_unlocking_generalizable_regu.md)
- [\[ICML 2026\] Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models](../../ICML2026/computational_biology/scalable_single-cell_gene_expression_generation_with_latent_diffusion_models.md)
- [\[AAAI 2026\] Gene Incremental Learning for Single-Cell Transcriptomics](../../AAAI2026/computational_biology/gene_incremental_learning_for_single-cell_transcriptomics.md)
- [\[ICLR 2026\] A Foundation Model with Multi-Variate Parallel Attention to Generate Neuronal Activity](a_foundation_model_with_multi-variate_parallel_attention_to_generate_neuronal_ac.md)
- [\[NeurIPS 2025\] scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration](../../NeurIPS2025/computational_biology/scmrdr_a_scalable_and_flexible_framework_for_unpaired_single-cell_multi-omics_da.md)

</div>

<!-- RELATED:END -->
