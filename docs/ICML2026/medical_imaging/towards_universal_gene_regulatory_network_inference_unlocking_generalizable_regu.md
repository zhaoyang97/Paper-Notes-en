---
title: >-
  [Paper Note] Towards Universal Gene Regulatory Network Inference: Unlocking Generalizable Regulatory Knowledge in Single-cell Foundation Models
description: >-
  [ICML 2026][Medical Imaging][Gene Regulatory Network] This work identifies that single-cell foundation models (scFM) contain rich gene regulatory knowledge that is obscured by "reconstruction-based pretraining." It intro…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Gene Regulatory Network"
  - "scFM"
  - "Counterfactual Perturbation"
  - "Gradient Trajectory"
  - "Zero-shot Generalization"
date: 2026-05-08
content_hash: 407c29cbf54d97d7
---

# Towards Universal Gene Regulatory Network Inference: Unlocking Generalizable Regulatory Knowledge in Single-cell Foundation Models

**Conference**: ICML 2026  
**arXiv**: [2605.08128](https://arxiv.org/abs/2605.08128)  
**Code**: Not released  
**Area**: Foundation Models / Single-cell Bioinformatics / Representation Distillation  
**Keywords**: Gene Regulatory Network, scFM, Counterfactual Perturbation, Gradient Trajectory, Zero-shot Generalization

## TL;DR
This work identifies that single-cell foundation models (scFM) contain rich gene regulatory knowledge that is obscured by "reconstruction-based pretraining." It introduces two probes—Virtual Value Perturbation and Gradient Trajectory—to distill pairwise gene features from frozen scFM that generalize across genes and datasets. On the BEELINE benchmark, AUPRC is improved from ~0.5 to 0.8–0.97, inaugurating a new paradigm of "Universal GRN Inference (UGRN)."

## Background & Motivation

**Background**: Gene regulatory network (GRN) inference is central to understanding cellular mechanisms. Traditional approaches (GENIE3, PIDC, etc.) rely on co-expression regression or mutual information within a single dataset. Recently, single-cell foundation models (scGPT, Geneformer, scBERT) have been pretrained on hundreds of millions of single-cell samples using masked value reconstruction, raising hopes for zero-shot GRN inference. The two mainstream scFM usages are "in-silico perturbation" (zeroing out source gene $g_i$ and observing changes in target gene $g_j$'s reconstruction) and "attention extraction" (using cross-layer attention weights as regulatory strength).

**Limitations of Prior Work**: Recent benchmarks (Jin et al. 2025, Ahlmann-Eltze et al. 2025) show that both scFM usages yield AUPRCs of only 0.49–0.55, nearly random, leading the biology community to question whether scFM truly learns regulatory knowledge. Traditional GRN methods are "closed-world": model dimensions are tied to the training set cell count $N$, failing on new datasets ($N'$ different), let alone unseen genes.

**Key Challenge**: The pretraining objective of scFM is "expression value reconstruction," essentially learning "which genes can predict $g_j$'s expression," which is not causally equivalent to "$g_i$ regulates $g_j$." Simple zero-out perturbation only reflects model dependence on $g_i$, and baseline expression differences across genes make perturbation magnitudes incomparable; attention weights mix semantic and positional signals. Thus, scFM does contain regulatory knowledge, but the "probes are too coarse."

**Goal**: (1) Design an evaluation protocol (UGRN benchmark) that enforces cross-dataset/gene generalization; (2) Propose probe methods to extract "regulation-explainable" pairwise features $\mathbf{e}_{ij}$ from frozen scFM.

**Key Insight**: scFM can accept arbitrary (even out-of-distribution) virtual expression values as input. Thus, one can bypass "real cells" and construct a series of virtual perturbation states, treating scFM as a "counterfactual inference engine" to systematically probe $g_i \to g_j$ response curves, and train a lightweight "translator" $f_\phi$ to map response features to regulatory labels.

**Core Idea**: Use unified virtual baseline values + multi-target perturbations (VVP) and multi-baseline gradient trajectories (GDT) to distill the implicit pairwise regulatory knowledge in scFM into dense feature vectors that generalize across genes and datasets.

## Method

### Overall Architecture
UGRN reformulates GRN inference as two stages: (1) **Feature Extraction**—freeze scFM $\mathcal{M}$, and for any gene pair $(g_i, g_j)$ extract a fixed-dimensional pairwise feature $\mathbf{e}_{ij}$; (2) **Translator Training**—on a source dataset $\mathcal{D}_b$ (e.g., hESC), train a shallow MLP $f_\phi$ with BCE loss to map $\mathbf{e}_{ij}$ to regulatory probability $s_{ij}=f_\phi(\mathbf{e}_{ij})$, then perform zero-shot transfer to target datasets (mDC, mESC, mHSC-E/G/L, hHEP, etc.) containing unseen genes and cell types. The key is to ensure $\mathbf{e}_{ij}$ is comparable across datasets—traditional perturbation is tied to cell count $N$ and cannot generalize, while this work uses "virtual expression vectors" to break this dependency. Two naive strategies are used as baselines: **Pert** (zero-out on real mean expression $\bar{\mathbf{x}}$, $e_{ij}=\mathcal{M}(\bar{\mathbf{x}})_j-\mathcal{M}(\bar{\mathbf{x}}_{\neg i})_j$) and **Emb** (directly sum scFM vocabulary embeddings $\mathbf{E}_{\mathcal{M},i}+\mathbf{E}_{\mathcal{M},j}$), followed by the introduction of the two main probes, VVP and GDT, whose logits are averaged for the Ensemble.

### Key Designs

1. **Virtual Value Perturbation (VVP)**:

    - **Function**: Upgrades zero-out perturbation from "single-point on/off" to "multi-target value response curve," and eliminates incomparability of gene expression scales via a unified virtual baseline.
    - **Mechanism**: Select a virtual baseline $v_b$ (a fixed scalar near zero mean), construct a virtual cell vector $\mathbf{v}_{g_i\leftarrow v}$—set $g_i$ to $v$, all other genes to $v_b$; define a set of perturbation targets $\{v_{p,1},\dots,v_{p,M}\}$ covering the dynamic range. For each target, compute response $e_{ij}^{v_p}=\mathcal{M}(\mathbf{v}_{g_i\leftarrow v_p})_j-\mathcal{M}(\mathbf{v}_{g_i\leftarrow v_b})_j$, concatenate as $\mathbf{e}_{ij}=[e_{ij}^{v_{p,1}};\dots;e_{ij}^{v_{p,M}}]$. This essentially draws a discrete response curve: "how $g_j$ changes as $g_i$ varies."
    - **Design Motivation**: Traditional zero-out perturbation magnitude equals $g_i$'s original expression $\mathbf{x}_{c,i}$, so high-expression genes are "perturbed harder," low-expression genes "lighter," causing scale mismatch across datasets. Introducing a unified reference $v_b$ and multiple $v_p$ aligns all gene pairs in the same coordinate system, making feature vectors naturally comparable; multiple $M$ targets also capture nonlinear responses.

2. **Gradient Trajectory (GDT)**:

    - **Function**: Uses scFM's differentiability to extract "instantaneous" regulatory signals—the magnitude of $\partial \mathcal{M}(\cdot)_j/\partial v_i$ at a given expression level, concatenated along a series of virtual baselines to form a gradient trajectory.
    - **Mechanism**: Define an ordered set of baselines $\{v_{b,1},\dots,v_{b,T}\}$, each $v_{b,t}$ corresponds to a virtual input $\mathbf{v}_{g_i\leftarrow v_{b,t}}$ (other genes fixed at background), backpropagate to obtain $\nabla_{ij}^{(t)}=\partial \mathcal{M}(\mathbf{v}_{g_i\leftarrow v_{b,t}})_j / \partial v_i$, concatenate as $\mathbf{e}_{ij}=[\nabla_{ij}^{(1)};\dots;\nabla_{ij}^{(T)}]$. This replaces VVP's "interval response" with "local slope evolution across expression levels."
    - **Design Motivation**: VVP reflects the "cumulative response over $v_b\to v_p$," but lacks detail on instantaneous steepness at specific expression levels. Gradient trajectory informs the translator, e.g., "in low-expression regions, $g_i$ strongly influences $g_j$, but saturates at high expression," providing a complementary perspective to VVP.

3. **Ensemble + Translator Training**:

    - **Function**: Integrates VVP (interval response) and GDT (instantaneous sensitivity) for a comprehensive description of regulatory relationships.
    - **Mechanism**: Train two lightweight MLPs $f_\phi^{\text{VVP}}, f_\phi^{\text{GDT}}$ on VVP and GDT features (input dimensions $M$ and $T$, outputting sigmoid probabilities), final prediction is logit average $s_{ij}=\sigma(\tfrac{1}{2}(\text{logit}_{\text{VVP}}+\text{logit}_{\text{GDT}}))$. During training, scFM parameters are frozen; only the source dataset $\mathcal{D}_b$'s GRN labels are used to optimize $\phi$ via BCE.
    - **Design Motivation**: Table 1 shows VVP and GDT excel on different datasets (e.g., GDT is stronger on mDC, VVP on mH-G); simple logit averaging consistently outperforms single probes, indicating the two perspectives capture distinct regulatory cues.

### Loss & Training

The only learnable parameters are in the translator $f_\phi$, with standard binary cross-entropy loss: $\mathcal{L}_\phi = -\sum_{(i,j)\in\Omega_{tr}}[y_{ij}\log s_{ij}+(1-y_{ij})\log(1-s_{ij})]$. scFM remains frozen throughout. Evaluation uses a Leave-One/Some-Dataset-Out protocol: train $f_\phi$ on one dataset (e.g., hESC + STRING network), then zero-shot evaluate AUPRC on all other datasets. Source and target datasets share neither gene sets nor cell expression matrices, forcing the translator to learn a truly generalizable mapping. VVP uses $M=8$ target values, GDT uses $T=8$ baselines (see appendix for ablation).

## Key Experimental Results

### Main Results
The authors evaluate on 7 scRNA-seq datasets (hESC, hHEP, mDC, mESC, mHSC-E/G/L) × 4 ground-truth networks (STRING, Non-specific, Cell-type-specific, Lofgof) within the BEELINE framework, using scGPT and scBenchmark as scFM backbones. The table below excerpts AUPRC for STRING (Str) and Non-specific (Nsp) networks on scGPT:

| Dataset / Network | Pert (Origin) | Attn (Origin) | Pert (Baseline) | Emb (Baseline) | VVP | GDT | Ens |
|---|---|---|---|---|---|---|---|
| Str / hHEP | 0.496 | 0.507 | 0.586 | 0.732 | 0.609 | 0.906 | **0.909** |
| Str / mDC | 0.512 | 0.536 | 0.569 | 0.637 | 0.606 | 0.917 | **0.923** |
| Str / mESC | 0.542 | 0.531 | 0.493 | 0.699 | 0.600 | **0.969** | 0.966 |
| Str / mH-L | 0.622 | 0.534 | 0.624 | 0.815 | 0.656 | 0.895 | 0.873 |
| Nsp / hHEP | 0.516 | 0.512 | 0.546 | 0.586 | 0.549 | **0.716** | 0.711 |
| Nsp / mESC | 0.551 | 0.539 | 0.512 | 0.638 | 0.582 | 0.835 | **0.836** |

Original scFM usages (Pert/Attn) are nearly random; switching to UGRN baseline (Pert/Emb as translator features) already raises AUPRC to 0.6–0.8; GDT + Ensemble further boosts AUPRC to 0.83–0.97, a 40%–80% improvement over original Pert.

### Ablation Study

| Configuration | mESC (Str) AUPRC | Notes |
|---|---|---|
| Pert (Origin, real $\bar{\mathbf{x}}$) | 0.542 | Original scFM usage |
| Pert (Baseline, translator) | 0.493 | Feeding perturbation difference as feature to translator, worse due to scale mismatch |
| Emb (Baseline) | 0.699 | Using only gene vocabulary embeddings |
| VVP (single target $v_p$) | ~0.60 | No response curve, only slightly better than original Pert |
| VVP (multi-target $M=8$) | 0.600 | Full VVP, stable across datasets |
| GDT ($T=8$) | 0.969 | Gradient trajectory is the main contributor |
| Ensemble (VVP+GDT) | 0.966 | Matches GDT, but more robust on most datasets |

### Key Findings
- **GDT is the main source of improvement**: From original Pert (0.49) to GDT (0.97) is nearly a twofold increase, indicating that the "gradient signal" in scFM truly carries regulatory knowledge, not the reconstruction residual.
- **Unified virtual baseline for scale elimination is key to generalization**: Pert Baseline (0.49) is worse than Emb Baseline (0.70), highlighting that "real expression values causing incomparable perturbation magnitudes" is the root cause of cross-dataset failure.
- **scFM indeed contains regulatory knowledge**: Across all scFM, datasets, and ground-truth networks, GDT/Ensemble consistently outperform random (0.5) and traditional scFM usages, overturning the pessimistic view that "scFM cannot learn GRN."
- **Prediction possible without real cell measurements**: VVP/GDT use only virtual values, enabling regulatory prediction even without target gene expression data, especially useful for rare cell types and novel genes.

## Highlights & Insights
- **Redefines scFM interpretability**: Reinterprets the model from a "reconstructor" to a "counterfactual inference engine," enabling systematic probing of internal knowledge with arbitrary virtual inputs—this approach can be transferred to causal attribution in LLMs or disentangling attributes in image generation models.
- **"Gradient as regulatory signal" is an underrated probe**: The authors show that direct backpropagation on frozen scFM yields $\partial \mathcal{M}_j / \partial v_i$ as a stable, cross-dataset-aligned feature, outperforming attention- or residual-based features, providing a new toolbox for mechanistic interpretability.
- **Unified virtual baseline + multi-target sampling**: This "scale elimination + response curve sampling" paradigm is highly generalizable and can be used to construct cross-domain counterfactual features in RecSys, causal inference, drug response, etc.
- **UGRN benchmark is itself a contribution**: Traditional GRN evaluation is in-distribution; this work enforces leave-dataset-out and unseen genes, making evaluation truly reflect "universality"—a benchmark design worth promoting in biological AI.

## Limitations & Future Work
- **Depends on scFM quality**: Assumes scFM has learned latent regulatory knowledge; if the backbone is weak (e.g., pretrained on only tens of thousands of cells), VVP/GDT may not yield effective signals; no comparison on smaller-scale scFM is provided.
- **GDT is computationally expensive**: Requires backpropagation for $T=8$ virtual baselines per gene pair; scaling to all gene pairs (tens of thousands × tens of thousands) may be bottlenecked by memory/time, and no engineering sparsification is proposed.
- **No explicit modeling of temporal dynamics**: GRN regulation is time/development-stage dependent; the virtual values here are static samples, unable to capture "activation evolution over time," which could be addressed by integrating trajectory inference.
- **Translator $f_\phi$ remains a black box**: Although input features are more interpretable, the MLP trained with BCE still cannot directly reveal "which pathways drive the prediction," falling short of biologists' needs for mechanistic interpretability.

## Related Work & Insights
- **vs Original scFM in-silico perturbation (Theodoris et al. 2023; Cui et al. 2024)**: They use single zero-out output difference as regulatory score; this work transforms the same perturbation into a "unified baseline + multi-target" feature vector with a learnable translator, boosting AUPRC from ~0.5 to 0.8+.
- **vs Attention extraction (Yang et al. 2022)**: Attention weights mix semantic and positional signals; this work replaces them with gradient-based GDT, demonstrating that gradients are more "regulation-pure."
- **vs Traditional GRN inference (GENIE3, PIDC)**: Those methods are in-distribution, closed-world, and cannot transfer to new genes/datasets; this work leverages scFM's unified vocabulary $\mathcal{V}$ and virtual input capability for true zero-shot generalization.
- **vs Causal Tracing / Mechanistic Interpretability**: Analogous to activation patching in LLMs, this work performs "counterfactual intervention + gradient attribution on input dimensions" in scFM, serving as a practical instance of mechanistic interpretability in biological foundation models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Recasts scFM as a counterfactual inference engine, defines the new UGRN evaluation paradigm, and proposes two complementary interpretable probes.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 2 scFM × 7 datasets × 4 ground-truth networks with dense ablations, but lacks comparison on scFM scale/pretraining data.
- Writing Quality: ⭐⭐⭐⭐ Clear formulas and illustrations, with a natural reasoning chain from problem to method; however, baseline and origin naming may be confusing.
- Value: ⭐⭐⭐⭐⭐ Reverses pessimism in the biological foundation model community and provides counterfactual probe routines directly transferable to other foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] KINDLE: Knowledge-Guided Distillation for Prior-Free Gene Regulatory Network Inference](../../NeurIPS2025/model_compression/kindle_knowledge-guided_distillation_for_prior-free_gene_regulatory_network_infe.md)
- [\[ICML 2026\] Stochastic Sparse Attention for Memory-Bound Inference](stochastic_sparse_attention_for_memory-bound_inference.md)
- [\[AAAI 2026\] StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion](../../AAAI2026/model_compression/stepfun-formalizer_unlocking_the_autoformalization_potential_of_llms_through_kno.md)
- [\[NeurIPS 2025\] Learning to Factorize and Adapt: A Versatile Approach Toward Universal Spatio-Temporal Foundation Models](../../NeurIPS2025/model_compression/learning_to_factorize_and_adapt_a_versatile_approach_toward_universal_spatio-tem.md)
- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)

</div>

<!-- RELATED:END -->
