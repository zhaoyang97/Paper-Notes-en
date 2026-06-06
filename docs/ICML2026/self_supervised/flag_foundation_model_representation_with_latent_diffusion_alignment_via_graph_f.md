---
title: >-
  [Paper Note] FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction
description: >-
  [ICML 2026][Self-Supervised Learning][Spatial Transcriptomics] FLAG reformulates the prediction of spatial gene expression from H&E pathology images as a structured distribution generation problem. It utilizes a fixed sp…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Spatial Transcriptomics"
  - "Pathological H&E"
  - "Latent Diffusion"
  - "Graph Encoder"
  - "Gene Foundation Model Alignment"
date: 2026-05-08
content_hash: 30a7d4de2cf8ae67
---

# FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction

**Conference**: ICML 2026  
**arXiv**: [2605.18055](https://arxiv.org/abs/2605.18055)  
**Code**: https://github.com/darkflash03/FLAG  
**Area**: Medical Imaging / Spatial Transcriptomics / Diffusion Models  
**Keywords**: Spatial Transcriptomics, Pathological H&E, Latent Diffusion, Graph Encoder, Gene Foundation Model Alignment

## TL;DR
FLAG reformulates the prediction of spatial gene expression from H&E pathology images as a structured distribution generation problem. It utilizes a fixed spatial graph encoder to compress tissue topology into condition vectors and employs a DiT (Diffusion Transformer) for denoising in the gene dimension. By injecting gene-gene regulatory priors through representation alignment with intermediate layers of a Gene Foundation Model (GFM), FLAG elevates Gene Structural Correlation (GSC) and Spatial Structural Correlation (SSC) to new heights while maintaining competitive PCC/MSE performance.

## Background & Motivation

**Background**: Spatial Transcriptomics (ST) sequencing is expensive and has low throughput, whereas H&E Whole Slide Images (WSI) are readily available in clinical practice. Consequently, predicting spot-level gene expression from H&E images has become a popular research direction. Prevailing methods typically treat this as a gene-wise scalar regression task: HisToGene, BLEEP, and TRIPLEX directly minimize MSE, while Stem and STFlow utilize diffusion or flow-matching along the gene dimension.

**Limitations of Prior Work**: All these methods rely on point-wise metrics such as PCC/MSE for evaluation, completely ignoring two types of structural properties critical for downstream pathway analysis and spatial domain identification: gene-gene regulatory networks and gene-spatial distributions (Moran's I). As a result, while point-wise metrics may appear satisfactory, the generated expression maps lack coherent internal structures, being either over-smoothed or characterized by fragmented synergistic patterns between genes.

**Key Challenge**: There is a fundamental conflict between modeling the task as "independent scalar regression" and the goal of "recovering the full multivariate distribution." The mapping from tissue to expression is inherently a one-to-many stochastic mapping, which regression tends to average out. A natural solution is graph-diffusion, where spots are nodes and correlations are edges in a joint diffusion process. However, the authors empirically discovered a fatal **Gene Dimension Curse**: as the number of genes $G$ increases from 50 to 800, the PCC of joint node-edge diffusion collapses from $>0.8$ to nearly 0, failing much faster than node-only diffusion.

**Goal**: (1) Explain why joint diffusion inevitably collapses under high-dimensional gene settings; (2) Design a generative framework that respects both spot-spot topology and gene-gene regulation while scaling to 200/800 genes; (3) Propose evaluation metrics that reflect biological structures rather than just point-wise accuracy.

**Key Insight**: The authors observe that as $G$ increases, empirical estimates of correlations between spots rapidly concentrate around population values. This causes the "node-edge consistency manifold" $\{(\mathbf{X}, \mathbf{A}) : \mathbf{A} = \mathrm{corr}(\mathbf{X})\}$ to become extremely thin. Fitting the score field on this manifold requires near-singular gradient magnitudes, which exceeds the expressive capacity of finite-width networks. Thus, the issue lies not in the architecture but in the modeling choice of treating high-dimensional correlation matrices as diffusion targets.

**Core Idea**: Instead of treating the graph as a generation target, use it as a spatial encoder. A graph encoder with fixed topology compresses spot-spot relationships into a spatial context $\mathbf{H}_{\text{spatial}}$, while DiT performs denoising exclusively in the gene dimension. Furthermore, representation alignment with gene embeddings from pretrained GFMs is used to inject gene-gene priors from large-scale external single-cell data.

## Method

### Overall Architecture
Input: $S$ spots on an H&E WSI, where each spot has 2D coordinates $u_s$, visual features $v_s$ extracted by a pathology foundation model, and a target gene expression vector $x_s \in \mathbb{R}^G$. The pipeline consists of two relatively independent branches:

- **Left Branch (Deterministic, One-time Encoding)**: Connects all spots into a fully connected graph. Node conditions $\mathbf{C}_v$ are visual features, and edge conditions $\mathbf{C}_e = [d_{ij}, s_{ij}]$ combine physical distance and visual similarity. The graph encoder outputs a spatial context vector for each spot: $\mathbf{H}_{\text{spatial}} = \mathrm{GraphEncoder}(\mathbf{C}_v, \mathbf{C}_e)$.
- **Right Branch (Generative, Iterative Denoising)**: Conditioned on $\mathbf{H}_{\text{spatial}}$, DiT diffusion is performed in the gene dimension: $\hat{\epsilon} = \epsilon_\theta(\mathbf{X}_t \mid \mathbf{H}_{\text{spatial}}, t)$. Hidden states $\mathbf{H}^{(k)} \in \mathbb{R}^{B \times G \times d_h}$ are extracted at specific DiT blocks and aligned via cosine similarity with per-gene embeddings $\mathbf{F} \in \mathbb{R}^{G \times d_e}$ pre-extracted offline from Geneformer or scGPT.

Upon completion of denoising, the predicted $S \times G$ expression matrix is obtained, enabling simultaneous calculation of PCC/MSE and GSC/SSC.

### Key Designs

1. **From Joint Node-Edge Diffusion to Spatial-Conditional Decomposition (Core Architectural Decision)**:
    - **Function**: Completely circumvents the Gene Dimension Curse by reducing the generation target dimension from $G+N$ back to $G$.
    - **Mechanism**: The authors initially experimented with a motivating scheme that put nodes $\mathbf{X}$ and latent edges $\mathbf{A} = \mathrm{corr}(\mathbf{X})$ into graph diffusion, using Edge-Modulated Attention where edge states modulate QK attention scores via "structural gating" ($1 + \mathrm{Linear}(\mathbf{A}_{t,ij}) + \alpha\mathrm{Linear}(\mathbf{C}_{e,ij})$) and "structural bias." This was supervised by a consistency loss $\mathcal{L}_{\text{cons}} = \mathbb{E}_t\|\hat{\mathbf{A}}_0 - \mathrm{Corr}(\hat{\mathbf{X}}_0)\|_1$. While this improved PCC at $G=50$, formal analysis showed a lower bound $\mathcal{L}^*_{\text{joint}}(G) - \mathcal{L}^*_{\text{node}} \ge \Omega(G)$, indicating an unavoidable optimization penalty as $G$ grows. FLAG's solution uses the graph purely for encoding, allowing the DiT to focus on gene-to-gene attention.
    - **Design Motivation**: This shift from "graph as target" to "graph as condition" decomposes the high-dimensional joint distribution into $p(\mathbf{X} \mid \mathbf{H}_{\text{spatial}})$. Spot-spot structures are absorbed by the graph encoder, while the diffusion model focuses on the gene-gene joint distribution, preserving spatial regularization without the curvature explosion of correlation matrices.

2. **Gene Foundation Model (GFM) Representation Alignment**:
    - **Function**: Compensates for the limited data scale and narrow gene coverage in ST slides by transferring gene-gene relationships learned by scGPT/Geneformer from millions of single cells to the intermediate layers of the DiT.
    - **Mechanism**: Per-gene embeddings $\mathbf{F}$ are extracted from a pretrained GFM and frozen. During training, a hidden state $\mathbf{H}^{(k)}$ from an intermediate DiT block is mapped via a lightweight MLP to the GFM embedding space. An alignment loss $\mathcal{L}_{\text{align}} = -\langle\mathrm{MLP}(\mathbf{H}^{(k)}), \mathbf{F}\rangle / (\|\cdot\|\|\cdot\| + \epsilon)$ is applied.
    - **Design Motivation**: ST data only covers a few thousand spots and genes, making it difficult to reliably estimate gene-gene covariance. GFMs, trained on massive cell counts, provide embeddings that encode pathways and regulatory structures. Aligning at the intermediate level rather than concatenating at the input preserves denoising flexibility while providing a "soft constraint" through a global biological prior.

3. **Structured Evaluation Metrics (GSC / SSC) and Total Loss**:
    - **Function**: Transforms the subjective impression of "whether the generated expression map preserves biological structure" into quantifiable metrics.
    - **Mechanism**: GSC (Gene Structural Correlation) measures the integrity of gene-gene regulation by comparing the correlation matrices of predicted and ground truth data in the gene dimension. SSC (Spatial Structural Correlation) uses Moran's I of each gene to measure whether spatial autocorrelation is preserved, directly corresponding to downstream spatial domain identification and marker discovery. The total training objective is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{score}} + \lambda_{\text{align}}\mathcal{L}_{\text{align}}$.
    - **Design Motivation**: Previous ST papers focused solely on PCC/MSE, often resulting in maps that are numerically accurate but biologically "blurry." GSC and SSC are what pathologists truly care about for pathway analysis and spatial clustering.

### Loss & Training
- **Primary Loss**: Standard $\epsilon$-prediction score matching $\mathcal{L}_{\text{score}}$.
- **Auxiliary Loss**: GFM cosine alignment $\mathcal{L}_{\text{align}}$ with a small weight $\lambda_{\text{align}}$ (order of $10^{-1}$ to $10^{0}$).
- **Data**: HER2ST, KIDNEY, and PRAD cohorts from HEST-1k, split at the slide level (7:2:1). Top-200 High-Mean & High-Variance Genes (HMHVG) are selected.
- **Hardware**: Single NVIDIA H800 GPU.

## Key Experimental Results

### Main Results
Evaluation of Top-200 HMHVG across three HEST-1k datasets (slide-level mean ± std):

| Dataset | Metric | Prev. SOTA (Generative) | Prev. SOTA (Discriminative) | FLAG | Gain |
|---|---|---|---|---|---|
| HER2ST | PCC ↑ | STFlow 0.706 | TRIPLEX 0.691 | 0.684 | Comparable to strongest baseline |
| HER2ST | GSC ↑ | Stem 0.832 | TRIPLEX 0.559 | **0.893** | +6 pt in structural correlation |
| HER2ST | SSC ↑ | Stem 0.381 | TRIPLEX 0.071 | **0.639** | +26 pt in Moran's I consistency |
| KIDNEY | PCC ↑ | STFlow 0.315 | TRIPLEX 0.374 | **0.392** | Higher than best discriminative |
| KIDNEY | GSC ↑ | Stem 0.845 | BLEEP 0.533 | **0.871** | Optimal regulatory structure |
| PRAD | SSC ↑ | STFlow 0.564 | TRIPLEX 0.634 | **0.751** | Most faithful spatial distribution |

Major signals: FLAG's point-wise accuracy (PCC/MSE) is competitive with state-of-the-art discriminative and generative baselines. However, GSC and SSC are the highest across almost all datasets by significant margins, proving that structural fidelity is the primary dimension of improvement.

### Ablation Study
Ablation of key components on HER2ST:

| Configuration | PCC ↑ | MSE ↓ | GSC ↑ | SSC ↑ | Description |
|---|---|---|---|---|---|
| Full FLAG | 0.684 | 0.734 | 0.893 | 0.639 | Full Model |
| w/o GFM Alignment | 0.668 | 0.794 | 0.871 | 0.589 | Without biological prior; PCC/SSC drop |
| w/o Spatial Graph | 0.630 | 0.850 | 0.903 | 0.340 | Without graph encoder; SSC drops significantly |
| w/o Diffusion (Supervised) | 0.675 | 0.786 | 0.322 | 0.569 | Deterministic regression; GSC collapses |

### Key Findings
- **Diffusion is Key to Anti-Over-smoothing**: Replacing diffusion with supervised regression caused GSC to plummet from 0.89 to 0.32. This provides direct evidence that generative modeling prevents the collapse of gene regulatory structures that PCC fails to detect.
- **Graph and GFM are Orthogonal Priors**: The graph primarily manages SSC (spatial), while the GFM manages GSC (gene). Removing either leads to drops in different directions, indicating clean factorization.
- **Empirical Evidence of Gene Dimension Curse**: As $G \in \{10, 50, 100, 200, 400, 800\}$ increases, joint node-edge diffusion PCC drops toward 0. FLAG maintains significantly higher PCC at $G=800$, showing a qualitative difference in robustness to dimensionality.
- **Downstream Tasks are the True Test**: On HER2ST, FLAG achieved a DEG (Differentially Expressed Gene) Top-50 overlap of 0.500 and spatial domain clustering ARI of 0.845 / NMI 0.914, significantly outperforming all baselines (e.g., STFlow ARI 0.600).

## Highlights & Insights
- Comparing "graph as target" vs. "graph as condition" at a methodological level and providing an $\Omega(G)$ lower bound to prove why the former fails is a rare and honest form of academic writing—analyzing a failed attempt to justify the design of version 2.
- The GFM intermediate layer alignment trick translates the idea of "diffusion state alignment with frozen encoder features" (from REPA/SVG in visual generation) to the biological domain.
- The introduction of GSC and SSC effectively changes the evaluation language for the WSI→ST subfield. If adopted, SOTA leaderboards previously ranked by PCC might be completely reshuffled.
- The framework follows a clear three-stage structure: graph encoder + conditional diffusion + foundation model alignment. This can likely be generalized to other "high-dimensional multivariate + weak annotation + adjacent FM" scenarios like multi-omics fusion.

## Limitations & Future Work
- Evaluations were conducted on three cohorts in HEST-1k with intra-tissue splits; zero-shot cross-tissue generalization remains an open question.
- The iterative cost of diffusion models is high; a fair latency comparison with STFlow (flow matching is typically faster) was not provided. Distillation or consistency models may be needed for clinical deployment.
- GFM embeddings are offline and frozen per-gene vectors, which do not explicitly model context-dependent gene behavior across different tissues.
- Testing on Top-200 HMHVG targets "easier" genes. Whether the immunity to the Gene Dimension Curse holds for the full genome (~20K genes) is yet to be determined, as the paper only scales to $G=800$.

## Related Work & Insights
- **vs Stem (Conditional Diffusion on H&E)**: Stem uses gene-gene attention within spots but ignores spot-spot relationships. FLAG uses a graph encoder to capture spatial structure, leading to much higher SSC (0.64 vs. 0.38).
- **vs STFlow (Flow Matching)**: STFlow uses a graph attention backbone for holistic generation, but spots tend to be "over-correlated." FLAG's decoupled approach allows the SSC scatter plot to align more closely with the diagonal.
- **vs Discriminative TRIPLEX**: TRIPLEX is the strongest discriminative baseline for PCC but has a low GSC (0.56) due to over-smoothing. FLAG proves that generative models with structural priors have a structural advantage in fidelity.
- **vs REPA / SVG (Representation Alignment in Visual Diffusion)**: FLAG adapts the idea of using a frozen encoder as a semantic judge for diffusion models from the visual domain to ST tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Systematically combines "graph as condition" and "GFM alignment" for WSI→ST while introducing the Gene Dimension Curse and GSC/SSC metrics.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes three datasets, full ablation, gene dimension scanning, and two downstream tasks, although it lacks cross-tissue or full-genome scaling.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The narrative chain from motivating attempts to failure analysis to the V2 design is exceptionally clear.
- **Value**: ⭐⭐⭐⭐⭐ Advances both methodology and evaluation (GSC/SSC), with the potential to reshape standards in the computational pathology community.

## Related Papers

- [\[ICML 2026\] Learning Graph Foundation Models on Riemannian Graph-of-Graphs](learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)
- [\[ICLR 2026\] Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models](../../ICLR2026/self_supervised/regularized_latent_dynamics_prediction_is_a_strong_baseline_for_behavioral_found.md)
- [\[ICML 2026\] NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation Models](numleak_public_numeric_benchmarks_as_latent_labels_in_foundation_models.md)
- [\[ICML 2026\] How 'Neural' is a Neural Foundation Model?](how_neural_is_a_neural_foundation_model.md)
- [\[ICML 2025\] Griffin: Towards a Graph-Centric Relational Database Foundation Model](../../ICML2025/self_supervised/griffin_towards_a_graph-centric_relational_database_foundation_model.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Graph Foundation Models on Riemannian Graph-of-Graphs](learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)
- [\[ICLR 2026\] Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models](../../ICLR2026/self_supervised/regularized_latent_dynamics_prediction_is_a_strong_baseline_for_behavioral_found.md)
- [\[ICML 2026\] NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation Models](numleak_public_numeric_benchmarks_as_latent_labels_in_foundation_models.md)
- [\[ICML 2026\] How 'Neural' is a Neural Foundation Model?](how_neural_is_a_neural_foundation_model.md)
- [\[ICML 2026\] The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence](the_geometric_mechanics_of_contrastive_representation_learning_alignment_potenti.md)

</div>

<!-- RELATED:END -->
