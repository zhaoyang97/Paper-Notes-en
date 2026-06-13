---
title: >-
  [Paper Note] Towards Universal Gene Regulatory Network Inference: Unlocking Generalizable Regulatory Knowledge in Single-cell Foundation Models
description: >-
  [ICML 2026][Computational Biology][Gene Regulatory Networks] This paper demonstrates that single-cell foundation models (scFMs) contain rich gene regulatory knowledge often obscured by "reconstructive pre-training." It p…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Gene Regulatory Networks"
  - "scFM"
  - "Counterfactual Perturbation"
  - "Gradient Trajectory"
  - "Zero-shot Generalization"
date: 2026-05-08
content_hash: ccd6940dd20849bb
---

# Towards Universal Gene Regulatory Network Inference: Unlocking Generalizable Regulatory Knowledge in Single-cell Foundation Models

**Conference**: ICML 2026  
**arXiv**: [2605.08128](https://arxiv.org/abs/2605.08128)  
**Code**: Undisclosed  
**Area**: Foundation Models / Single-cell Bioinformatics / Representation Distillation  
**Keywords**: Gene Regulatory Networks, scFM, Counterfactual Perturbation, Gradient Trajectory, Zero-shot Generalization

## TL;DR
This paper demonstrates that single-cell foundation models (scFMs) contain rich gene regulatory knowledge often obscured by "reconstructive pre-training." It proposes two probes, Virtual Value Perturbation and Gradient Trajectory, to distill pairwise gene features from frozen scFMs that generalize across genes and datasets. This approach pushes AUPRC on the BEELINE benchmark from ~0.5 to 0.8–0.97, initiating a new paradigm called "Universal GRN Inference (UGRN)."

## Background & Motivation

**Background**: Gene Regulatory Network (GRN) inference is a core task for understanding cellular mechanisms. Traditional methods (e.g., GENIE3, PIDC) rely on co-expression regression or mutual information within single datasets. Recently, scFMs (scGPT, Geneformer, scBERT) pre-trained on hundreds of millions of single cells via masked value reconstruction have been expected to perform zero-shot GRN inference. Two primary scFM use cases are "in-silico perturbation" (zeroing out source gene $g_i$ to observe changes in target gene $g_j$) and "attention extraction" (using cross-layer attention weights as regulatory strength).

**Limitations of Prior Work**: Several recent benchmarks (Jin et al. 2025, Ahlmann-Eltze et al. 2025) indicate that these scFM applications typically yield an AUPRC of only 0.49–0.55, nearly equivalent to random guessing. This has led the biological community to doubt whether scFMs truly learn regulatory knowledge. Meanwhile, traditional GRN methods are "closed-world": model dimensions are tied to the cell count $N$ of the training set, causing failure on new datasets with different $N'$ or unseen genes.

**Key Challenge**: The pre-training objective of scFMs is "expression value reconstruction," essentially learning "which genes can predict the expression of $g_j$," which is not causally equivalent to "whether $g_i$ regulates $g_j$." Simple zero-out perturbations only reflect the model's dependency strength on $g_i$. Furthermore, baseline expression levels vary significantly across genes, making the perturbation magnitudes incomparable. Attention weights also conflate semantic and positional signals. Thus, the issue is not a lack of knowledge in scFMs, but rather that the "probes are too crude."

**Goal**: (1) Design an evaluation protocol (UGRN benchmark) that forces models to generalize across datasets and genes; (2) Develop probing methods to extract "regulatorily interpretable" pairwise features $\mathbf{e}_{ij}$ from frozen scFMs.

**Key Insight**: scFMs can receive arbitrary virtual expression values as input, even those outside the training distribution. Thus, one can decouple the model from "real cells" by constructing virtual perturbation states, treating the scFM as a "counterfactual reasoning engine." This allows for systematic detection of the $g_i \to g_j$ response curve, followed by a lightweight "translator" $f_\phi$ learning the mapping from response features to regulatory labels.

**Core Idea**: Distill implicit pairwise regulatory knowledge from scFMs into dense feature vectors that generalize across genes and datasets using unified virtual baseline values with multi-target perturbation (VVP) and multi-baseline gradient trajectories (GDT).

## Method

### Overall Architecture
UGRN reformulates GRN inference into two stages: (1) **Feature Extraction**—Freeze the scFM $\mathcal{M}$ and extract a fixed-dimension pairwise feature $\mathbf{e}_{ij}$ for any gene pair $(g_i, g_j)$; (2) **Translator Training**—Train a shallow MLP $f_\phi$ using BCE loss on a source dataset $\mathcal{D}_b$ (e.g., hESC) to map $\mathbf{e}_{ij}$ to a regulatory probability $s_{ij}=f_\phi(\mathbf{e}_{ij})$. This is then transferred zero-shot to target datasets (mDC, mESC, mHSC-E/G/L, hHEP, etc.) containing unseen genes and cell types. The key is making $\mathbf{e}_{ij}$ comparable across datasets; while traditional perturbation is tied to cell count $N$, this work uses "virtual expression vectors" to break this dependency. The authors use two naive strategies as baselines: **Pert** (zero-out using real mean expression $\bar{\mathbf{x}}$, $e_{ij}=\mathcal{M}(\bar{\mathbf{x}})_j-\mathcal{M}(\bar{\mathbf{x}}_{\neg i})_j$) and **Emb** (using scFM vocabulary embeddings $\mathbf{E}_{\mathcal{M},i}+\mathbf{E}_{\mathcal{M},j}$), then introduce VVP and GDT, averaging their logits for an **Ensemble**.

### Key Designs

1.  **Virtual Value Perturbation (VVP)**:
    - **Function**: Upgrades zero-out perturbation from a single-point "on/off" to a multi-target response curve and eliminates incomparability due to gene expression scales via a unified virtual baseline.
    - **Mechanism**: A virtual baseline $v_b$ (a fixed scalar near zero mean) is selected to construct a virtual cell vector $\mathbf{v}_{g_i\leftarrow v}$ where $g_i$ is set to $v$ and all other genes to $v_b$. A set of perturbation target values $\{v_{p,1},\dots,v_{p,M}\}$ is defined to cover the dynamic range. For each target, the response is calculated as $e_{ij}^{v_p}=\mathcal{M}(\mathbf{v}_{g_i\leftarrow v_p})_j-\mathcal{M}(\mathbf{v}_{g_i\leftarrow v_b})_j$, forming $\mathbf{e}_{ij}=[e_{ij}^{v_{p,1}};\dots;e_{ij}^{v_{p,M}}]$. This essentially captures a discrete response curve of "$g_j$ response to $g_i$ change."
    - **Design Motivation**: In traditional zero-out, the perturbation magnitude equals the original expression $\mathbf{x}_{c,i}$. Highly expressed genes are "perturbed more" than lowly expressed ones, leading to misaligned scales across datasets. Using a unified reference $v_b$ and multiple $v_p$ allows all gene pairs to be queried in the same coordinate system, making features naturally alignable across datasets while capturing non-linear responses.

2.  **Gradient Trajectory (GDT)**:
    - **Function**: Leverages scFM differentiability to extract "instantaneous" regulatory strength signals—the magnitude of $\partial \mathcal{M}(\cdot)_j/\partial v_i$ at specific expression levels—strung into a gradient trajectory.
    - **Mechanism**: Defines an ordered set of baselines $\{v_{b,1},\dots,v_{b,T}\}$. Each $v_{b,t}$ corresponds to a virtual input $\mathbf{v}_{g_i\leftarrow v_{b,t}}$ (with other genes fixed at background values). Backpropagation yields $\nabla_{ij}^{(t)}=\partial \mathcal{M}(\mathbf{v}_{g_i\leftarrow v_{b,t}})_j / \partial v_i$, concatenated as $\mathbf{e}_{ij}=[\nabla_{ij}^{(1)};\dots;\nabla_{ij}^{(T)}]$. This replaces the "interval response" of VVP with "local slope evolution."
    - **Design Motivation**: VVP reflects the cumulative response over the interval $v_b\to v_p$, but lacks detail on instantaneous steepness at specific levels. Gradient trajectories inform the translator of details like "strong influence in low expression regions that saturates in high expression regions," providing a complementary perspective to VVP.

3.  **Ensemble + Translator Training**:
    - **Function**: Fuses VVP and GDT to obtain a comprehensive description of regulatory relationships.
    - **Mechanism**: Separate lightweight MLPs $f_\phi^{\text{VVP}}, f_\phi^{\text{GDT}}$ are trained on VVP and GDT features respectively (input dimensions $M$ and $T$, outputting sigmoid probabilities). The final prediction uses the logit average $s_{ij}=\sigma(\tfrac{1}{2}(\text{logit}_{\text{VVP}}+\text{logit}_{\text{GDT}}))$. scFM parameters remain frozen.
    - **Design Motivation**: As shown in Table 1, VVP and GDT excel on different datasets (e.g., GDT is stronger on mDC, VVP on mH-G). A simple logit average consistently outperforms single probes, indicating they capture distinct regulatory clues.

### Loss & Training
The only learnable parameter is the translator $f_\phi$, optimized with standard binary cross-entropy (BCE) loss: $\mathcal{L}_\phi = -\sum_{(i,j)\in\Omega_{tr}}[y_{ij}\log s_{ij}+(1-y_{ij})\log(1-s_{ij})]$. the scFM is frozen throughout. Evaluation follows a Leave-One/Some-Dataset-Out protocol: $f_\phi$ is trained on one dataset (e.g., hESC + STRING network) and evaluated zero-shot on all others. Source and target datasets share neither genes nor cell expression matrices, forcing the translator to learn a "truly generalized" mapping. VVP uses $M=8$ and GDT uses $T=8$.

## Key Experimental Results

### Main Results
Evaluations were conducted using scGPT and scBenchmark backbones across 7 scRNA-seq datasets in the BEELINE framework (hESC, hHEP, mDC, mESC, mHSC-E/G/L) with 4 ground-truth networks (STRING, Non-specific, Cell-type-specific, Lofgof). Below is a selection of AUPRC on scGPT for STRING (Str) and Non-specific (Nsp) networks:

| Dataset / Network | Pert (Origin) | Attn (Origin) | Pert (Baseline) | Emb (Baseline) | VVP | GDT | Ens |
|---|---|---|---|---|---|---|---|
| Str / hHEP | 0.496 | 0.507 | 0.586 | 0.732 | 0.609 | 0.906 | **0.909** |
| Str / mDC | 0.512 | 0.536 | 0.569 | 0.637 | 0.606 | 0.917 | **0.923** |
| Str / mESC | 0.542 | 0.531 | 0.493 | 0.699 | 0.600 | **0.969** | 0.966 |
| Str / mH-L | 0.622 | 0.534 | 0.624 | 0.815 | 0.656 | 0.895 | 0.873 |
| Nsp / hHEP | 0.516 | 0.512 | 0.546 | 0.586 | 0.549 | **0.716** | 0.711 |
| Nsp / mESC | 0.551 | 0.539 | 0.512 | 0.638 | 0.582 | 0.835 | **0.836** |

Original scFM usage (Pert/Attn) is nearly random. Converting to UGRN baseline format (Pert/Emb as translator features) improves performance to 0.6–0.8. GDT + Ensemble pushes AUPRC to 0.83–0.97, an improvement of 40%–80% over original Pert.

### Ablation Study

| Configuration | mESC (Str) AUPRC | Description |
|---|---|---|
| Pert (Origin, real $\bar{\mathbf{x}}$) | 0.542 | Original scFM usage |
| Pert (Baseline, Translated) | 0.493 | Direct perturbation difference as feature; worse due to incomparable scales |
| Emb (Baseline) | 0.699 | Only gene vocabulary embeddings |
| VVP (Single target $v_p$) | ~0.60 | No response curve; slightly better than original Pert |
| VVP (Multi-target $M=8$) | 0.600 | Full VVP, stable across datasets |
| GDT ($T=8$) | 0.969 | Gradient trajectory provides core gain |
| Ensemble (VVP+GDT) | 0.966 | Equal to GDT, more robust on other datasets |

### Key Findings
- **GDT is the primary source of gain**: Performance nearly doubles from Pert (0.49) to GDT (0.97), suggesting that the scFM's "gradient signal" is the true carrier of regulatory knowledge, rather than reconstruction residuals.
- **Unified virtual baselines are key to generalization**: The fact that Pert Baseline (0.49) is worse than Emb Baseline (0.70) reveals that incomparable perturbation magnitudes from real expression values are the root cause of cross-dataset failure.
- **scFMs do contain regulatory knowledge**: GDT/Ensemble consistently outperform random (0.5) and traditional scFM usage across all models, datasets, and ground-truth networks, reversing the pessimistic conclusion that scFMs cannot learn GRNs.
- **Predictions possible without real cell measurements**: Since VVP/GDT use virtual values, regulatory predictions can be made without target gene expression data, which is particularly useful for rare cell types and new genes.

## Highlights & Insights
- **Reshaping scFM interpretability**: Reinterprets the model from a "reconstructor" to a "counterfactual reasoning engine," allowing internal knowledge to be probed systematically regardless of training distribution—a concept transferable to causal attribution in LLMs or attribute disentanglement in generative models.
- **GDT as an undervalued probe**: The authors show that $\partial \mathcal{M}_j / \partial v_i$ obtained via backpropagation on a frozen scFM provides stable, alignable features across datasets, outperforming attention- or residual-based features and offering a new tool for mechanistic interpretability.
- **Unified virtual baseline + multi-target sampling**: This paradigm of "eliminating scales + sampling response curves" is highly generalizable and could be applied to RecSys, causal inference, or drug response for constructing cross-domain counterfactual features.
- **The UGRN benchmark is a significant contribution**: Unlike traditional in-distribution GRN evaluations, this benchmark enforces leave-dataset-out and unseen genes to reflect true "universality," a design philosophy that should be promoted in AI for biology.

## Limitations & Future Work
- **Dependency on scFM quality**: Assumes scFMs have already learned latent regulatory knowledge. If the backbone is weak (e.g., pre-trained on too few cells), VVP/GDT might not extract valid signals.
- **Computational cost of GDT**: Backpropagation for $T=8$ baselines across all gene pairs (tens of thousands) may hit memory or time bottlenecks; sparse engineering solutions were not provided.
- **Lack of explicit temporal modeling**: GRN regulation is time- and developmental stage-dependent. The static sampling of virtual values does not capture "activation evolution over time," suggesting potential integration with trajectory inference.
- **Black-box translator $f_\phi$**: While input features are more interpretable, the BCE-trained MLP does not directly reveal which pathways drive predictions, leaving a gap between model output and biologists' need for mechanistic explainability.

## Related Work & Insights
- **vs. original scFM in-silico perturbation (Theodoris et al. 2023; Cui et al. 2024)**: Previous works used a single zero-out difference as a regulatory score. This paper converts that operation into a feature vector with a "unified baseline + multi-target values" and a learnable translator, jumping from ~0.5 to 0.8+ AUPRC.
- **vs. attention extraction (Yang et al. 2022)**: Attention weights conflate semantic and positional signals. This work replaces them with GDT, proving gradients are "regulatorily purer."
- **vs. traditional GRN inference (GENIE3, PIDC)**: Those methods are in-distribution and closed-world. This work uses the scFM vocabulary $\mathcal{V}$ and virtual inputs to achieve true zero-shot generalization.
- **vs. Causal Tracing / Mechanistic Interpretability**: Similar to activation patching in LLMs, this can be viewed as "counterfactual intervention on input dimensions + gradient attribution," marking a practical application of mechanistic interpretability in biological foundation models.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Reinterprets scFM as a counterfactual engine, defines the UGRN evaluation paradigm, and provides complementary interpretable probes.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Dense ablation across 2 scFMs, 7 datasets, and 4 ground-truth networks, though lacks control experiments on scFM scale/pre-training data.
- **Writing Quality**: ⭐⭐⭐⭐ Clear formulas and diagrams with a natural reasoning chain; however, "baseline" and "origin" naming can be confusing.
- **Value**: ⭐⭐⭐⭐⭐ Reverses pessimism regarding scFMs and GRNs, providing a counterfactual probing template transferable to other foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models](scalable_single-cell_gene_expression_generation_with_latent_diffusion_models.md)
- [\[AAAI 2026\] Gene Incremental Learning for Single-Cell Transcriptomics](../../AAAI2026/computational_biology/gene_incremental_learning_for_single-cell_transcriptomics.md)
- [\[CVPR 2026\] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](../../CVPR2026/computational_biology/adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)
- [\[CVPR 2026\] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](../../CVPR2026/computational_biology/cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)
- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](../../CVPR2026/computational_biology/cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)

</div>

<!-- RELATED:END -->
