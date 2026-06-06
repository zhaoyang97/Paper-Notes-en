---
title: >-
  [Paper Note] Protein Circuit Tracing via Cross-layer Transcoders
description: >-
  [ICML 2026][Computational Biology][pLM] The authors adapt cross-layer transcoders from NLP to the protein language model ESM2, proposing the ProtoMech framework. It identifies sparse latent circuits (< 1% of latents) tha…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "pLM"
  - "ESM2"
  - "cross-layer transcoder"
  - "circuit tracing"
  - "steering"
date: 2026-05-08
content_hash: 590a3a7f9abf4981
---

# Protein Circuit Tracing via Cross-layer Transcoders

**Conference**: ICML 2026  
**arXiv**: [2602.12026](https://arxiv.org/abs/2602.12026)  
**Code**: https://github.com/amirgroup-codes/ProtoMech  
**Area**: Protein Language Models / Mechanistic Interpretability / Circuit Discovery / Biological Foundation Models  
**Keywords**: pLM, ESM2, cross-layer transcoder, circuit tracing, steering

## TL;DR
The authors adapt cross-layer transcoders from NLP to the protein language model ESM2, proposing the ProtoMech framework. It identifies sparse latent circuits (< 1% of latents) that recover 79% of downstream performance and enables steering along these circuits to design high-fitness protein variants, outperforming baselines in over 70% of cases.

## Background & Motivation

**Background**: Protein language models (pLMs) such as ESM2, ESMFold, and Boltz have achieved strong baselines in structure prediction, function prediction, and sequence design, serving as "biological foundation models." Recently, sparse autoencoders (SAEs) have been used to decompose pLM hidden states into interpretable features, such as identifying binding sites or conserved motifs.

**Limitations of Prior Work**: SAEs perform sparse factorization of *single-layer representations* only, failing to capture the computational process of passing information between layers. Per-layer transcoders (PLTs) attempt to approximate the input-output mapping of each MLP layer but are trained independently, leading to error accumulation and the neglect of cross-layer dependencies, resulting in poor reconstruction quality and untrustworthy circuits.

**Key Challenge**: To identify the "computational circuits" of a pLM, a *replacement model* is required to replace the original MLP blocks while explicitly modeling information transfer between layers. SAEs do not provide "transfer," and PLTs do not provide "cross-layer" modeling.

**Goal**: (1) Construct a cross-layer model for pLMs that can replace the MLP components of ESM2; (2) identify sparse circuits within the latent space (<1% latents) that recover most performance; (3) verify that these circuits correspond to interpretable biological motifs and use them for steering the design of high-fitness sequences.

**Key Insight**: Drawing inspiration from Anthropic's Cross-Layer Transcoder (CLT), where the output of each MLP layer is reconstructed by the sum of sparse latents from all *previous* layers, thereby explicitly modeling cumulative computation in the depth direction.

**Core Idea**: Replace each MLP layer in ESM2 with a CLT, then use a greedy search based on gradient attribution to find the subset of latents most critical to each downstream task. These represent the "protein circuits," which, when visualized, map to known biological structures such as the HRD catalytic motif, the Rossmann fold, and the GB1 hydrophobic core.

## Method

### Overall Architecture
ProtoMech consists of four components: (i) the CLT replacement model—comprising a sparse TopK encoder and cross-layer decoder for each ESM2 MLP layer; (ii) circuit discovery—using gradient attribution and incremental greedy search until the circuit recovers ≥70% of the original performance; (iii) steering—performing activation clamping on specific latents in the CLT to push wildtype sequences toward high-fitness regions (within 5 mutations); (iv) visualizer—selecting the top-5 activated nodes per layer, calculating edge weights via activation $\times$ gradient to generate readable circuit diagrams for manual cross-referencing with Swiss-Prot.

### Key Designs

1.  **Cross-layer transcoder (CLT) as ESM2 replacement**:

    - **Function**: Reconstructs the output of each MLP layer using sparse latents from all preceding layers, preserving interpretability while explicitly expressing cross-layer computation.
    - **Mechanism**: The residual stream $\mathbf x^\ell$ at layer $\ell$ is encoded as $\mathbf a^\ell=\text{TopK}(\mathbf W_{\text{enc}}^\ell(\mathbf x^\ell-\mathbf b_{\text{pre}}^\ell)+\mathbf b_{\text{enc}}^\ell)$, and then $\hat{\mathbf y}^\ell=\sum_{\ell'=1}^{\ell}\mathbf W_{\text{dec}}^{\ell'\to\ell}\mathbf a^{\ell'}+\mathbf b_{\text{pre}}^\ell$. The training objective is $\mathcal L_{\text{MSE}}=\sum_\ell \|\mathbf y^\ell-\hat{\mathbf y}^\ell\|_2^2$ plus an auxiliary reconstruction residual loss $\mathcal L_{\text{aux}}$ to mitigate dead latents.
    - **Design Motivation**: Upgrading signals from "intra-layer reconstruction" to "cross-layer composition" faithfully reproduces the actual computational paths of ESM2 and provides compositionality for circuit discovery (where deeper latents interpret functional combinations of earlier ones).

2.  **Attribution-based sparse circuit discovery algorithm**:

    - **Function**: Identifies the minimal subset of latents in the CLT to recover target task performance.
    - **Mechanism**: Supervised probes are first trained on the final MLP output $\mathbf y^L$ of the original ESM2 (logistic regression for family, CNN for function). Hybrid replacement is then performed in ProtoMech (MLP replaced by CLT, attention fixed). Latents are ranked by gradient attribution to the probe output and incrementally added until performance recovers to ≥70% or full-set performance. F1 is used for family tasks, and Spearman $\rho$ for function tasks.
    - **Design Motivation**: Greedy search with attribution avoids a brute-force $2^{d_{\text{latent}}}$ search. Fixing attention isolates the computation path to the MLP, preventing error accumulation from attention reconstruction, consistent with Anthropic's LLM designs.

3.  **Activation clamping steering along circuits**:

    - **Function**: Transitions the circuit from an "explanatory tool" to a "generative tool" for designing high-fitness protein variants.
    - **Mechanism**: During the wildtype forward pass, activation values of latents in the target function circuit are "clamped"—setting the node's activation to its maximum observed magnitude across the sequence multiplied by a scalar. Reconstructed $\hat{\mathbf y}^L$ at $\ell=L$ is decoded to ESM2 logits to select mutations. Variants are limited to 5 mutations from the wildtype to ensure downstream CNN evaluator reliability.
    - **Design Motivation**: Applying attribution results ("which latents are important for function") back to generation represents mechanistic-guided protein design. Unlike CAA, which uses a global concept vector, this method targets specific sub-circuits, resulting in cleaner signals and less interference.

### Loss & Training
The CLT uses $\mathcal L_{\text{CLT}}=\mathcal L_{\text{MSE}}+\alpha\mathcal L_{\text{aux}}$, pre-trained on 5M sequences (≤1022 aa) from UniRef50. The CLT for ESM2-8M contains 28M parameters (3.5× original). To mitigate the $\mathcal O(L^2)$ scaling bottleneck of the decoding matrix, a "windowed CLT" is proposed—each layer only attends to the previous 4 layers. For ESM2-35M, this reduces parameters from 207M to 125M and speeds up training by 1.75×, with family recovery dropping only slightly from 85% to 82%.

## Key Experimental Results

### Main Results

Recovery performance of circuits on ESM2-8M for two downstream tasks:

| Task | Full Latents (PLT / ProtoMech) | Circuit (PLT / ProtoMech) | Latent Fraction in Circuit |
|---|---|---|---|
| Protein family F1 | 0.50 ± 0.34 / **0.82 ± 0.19** | 0.49 ± 0.33 / **0.73 ± 0.19** | ~0.8% |
| Function Spearman $\rho$ | 0.38 ± 0.18 / **0.41 ± 0.19** | 0.35 ± 0.19 / **0.38 ± 0.18** | ~0.9% |
| Original ESM2 baseline | 0.92 (family F1) / 0.50 ($\rho$) | – | – |

Steering Mean scores across seven DMS assays (selected):

| Method | SPG1 | HIS7 | GFP | CAPSD | RASK |
|---|---|---|---|---|---|
| ProtoMech | 1.67 | **1.28** | 4.17 | **1.68** | -0.12 |
| PLT | **1.97** | 1.27 | **4.40** | 0.81 | -0.19 |
| CAA | 0.70 | 0.52 | 2.93 | -0.26 | -0.35 |
| Random | -2.76 | 0.56 | 2.74 | -1.04 | -0.64 |

### Ablation Study

| Configuration | Observation | Implication |
|---|---|---|
| Recursive replacement (attn also via CLT) | Significant performance collapse | Cross-layer error accumulation; attention must be fixed |
| Windowed CLT (window=4) on ESM2-35M | family 82% vs vanilla 85% | Viable trade-off for larger models |
| Comparison at same sparsity as PLT | PLT family F1 only 0.50 | Cross-layer connections, not sparsity tuning, drive performance |

### Key Findings
- On families where the original ESM2 performs poorly (F1<0.5), ProtoMech circuits achieve a higher average F1 (0.43 vs 0.39). This represents a "sparse denoising regularization" effect—the circuit filters out task-irrelevant noise, making it more reliable than the original model and a potential mechanistic filter for protein screening.
- The circuit maintains 74% performance on GFP variants with mutation depth ≥5, suggesting ProtoMech captures *global* functional motifs rather than overfitting local statistics.
- Visualization confirmed: In the Kinase circuit, L1 identifies arginine (R) → L3 identifies the HRD catalytic loop → L5 splits into the ATP binding site and G-loop; in the NADP+ circuit, L4 identification of the Rossmann fold → L5 narrows to the NADP+/FAD pocket. Deep layers reactivate early residues, consistent with the "token reiteration" phenomenon in NLP.

## Highlights & Insights
- This work is the first to adapt Anthropic's CLT concepts to biological foundation models, completing the circuit tracing and steering pipeline and verifying that "mechanistic interpretability" is a universal paradigm rather than an LLM-specific phenomenon.
- It introduces "mechanistic-guided protein design": mutations are driven by "which latents are responsible for high fitness" rather than global concept vectors or time-consuming evolutionary algorithms, offering high efficiency and strong explainability.
- The denoising phenomenon, where the "circuit is more accurate than the original model," suggests that sparse latent space inherently acts as a learnable regularizer, valuable for small-sample or noisy protein prediction tasks.

## Limitations & Future Work
- Circuit interpretation still relies on manual Swiss-Prot cross-referencing, which lacks scalability; an automated motif annotation pipeline is urgently needed.
- Only validated on masked LMs (ESM2); whether CLT applies to autoregressive pLMs (ProGen) or diffusion pLMs (DPLM) remains an open challenge.
- The $\mathcal O(L^2)$ parameter scaling bottleneck, though partially mitigated by windowed CLT, may remain prohibitive for models like ESM2-650M and larger.

## Related Work & Insights
- **vs Adams 2025 (SAE on pLM)**: While they use SAEs to explain single-layer representations, this work focuses on CLT to explain cross-layer computation, upgrading from "features" to "circuits."
- **vs Ameisen 2025 (CLT on LLM)**: This work represents the first application of Anthropic's LLM circuit tracing to the biological domain, verifying cross-domain portability.
- **vs CAA (Huang 2025) protein steering**: CAA relies on extensive fitness labels and local mutations, prone to overfitting; ProtoMech enables sparse intervention through circuits, offering better data efficiency and extrapolation.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of cross-layer transcoders for pLM circuit tracing and introduction of a new protein steering paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers family, function, and steering tasks across two model scales, with quantitative and biological case studies.
- Writing Quality: ⭐⭐⭐⭐ Clear framework with intuitive layer-by-layer interpretation of biological cases (Kinase/NADP+/GB1).
- Value: ⭐⭐⭐⭐ Provides a cross-domain case for the mech-interp community and a low-cost mechanism-driven method for protein design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/computational_biology/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[ICML 2026\] Cross-Chirality Generalization by Axial Vectors for Hetero-Chiral Protein-Peptide Interaction Design](cross-chirality_generalization_by_axial_vectors_for_hetero-chiral_protein-peptid.md)
- [\[ICML 2026\] Learning the Interaction Prior for Protein-Protein Interaction Prediction: A Model-Agnostic Approach](learning_the_interaction_prior_for_protein-protein_interaction_prediction_a_mode.md)
- [\[ICML 2026\] Protein Fold Classification at Scale: Benchmarking and Pretraining](protein_fold_classification_at_scale_benchmarking_and_pretraining.md)
- [\[ICML 2026\] Towards A Generative Protein Evolution Machine with DPLM-Evo](towards_a_generative_protein_evolution_machine_with_dplm-evo.md)

</div>

<!-- RELATED:END -->
