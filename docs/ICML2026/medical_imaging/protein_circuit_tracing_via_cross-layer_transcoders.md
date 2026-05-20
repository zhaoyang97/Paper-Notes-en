---
title: >-
  [Paper Note] Protein Circuit Tracing via Cross-layer Transcoders
description: >-
  [ICML 2026][Medical Imaging][pLM] The authors adapt the cross-layer transcoder from NLP to the protein language model ESM2, proposing the ProtoMech framework…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "pLM"
  - "ESM2"
  - "cross-layer transcoder"
  - "circuit tracing"
  - "steering"
date: 2026-05-08
content_hash: 7cdafb15de0ae0e7
---

# Protein Circuit Tracing via Cross-layer Transcoders

**Conference**: ICML 2026  
**arXiv**: [2602.12026](https://arxiv.org/abs/2602.12026)  
**Code**: https://github.com/amirgroup-codes/ProtoMech  
**Area**: Protein Language Models / Mechanistic Interpretability / Circuit Discovery / Biological Foundation Models  
**Keywords**: pLM, ESM2, cross-layer transcoder, circuit tracing, steering

## TL;DR
The authors adapt the cross-layer transcoder from NLP to the protein language model ESM2, proposing the ProtoMech framework, which recovers 79% downstream performance with less than 1% sparse latent circuits, and enables circuit-based steering to design high-fitness protein variants, outperforming baselines in over 70% of cases.

## Background & Motivation

**Background**: Protein language models (pLMs) such as ESM2, ESMFold, and Boltz have achieved strong baselines in structure prediction, function prediction, and sequence design, and are regarded as "foundation models" in biology. Recently, sparse autoencoders (SAEs) have been used to decompose pLM hidden states into interpretable features, such as identifying binding sites and conserved motifs.

**Limitations of Prior Work**: SAE only provides *single-layer representation* sparse factorization and cannot express the computation process of "passing information from one layer to the next." Per-layer transcoder (PLT) attempts to approximate the input-output mapping of each layer's MLP, but each layer is trained independently, leading to error accumulation and a complete neglect of cross-layer dependencies, resulting in poor reconstruction quality and unreliable circuits.

**Key Challenge**: To identify the "computational circuits" of pLMs, a *replacement model* is needed that can fully substitute the MLP blocks of the original model, with explicit modeling of information flow between layers. SAE cannot provide "transmission," and PLT cannot provide "cross-layer" modeling.

**Goal**: (1) Build a cross-layer model on pLMs that can fully replace the ESM2 MLP components; (2) Identify sparse circuits in the latent space of this model that recover most performance with less than 1% of latents; (3) Verify that these circuits correspond to interpretable biological motifs and can be used for steering to design high-fitness sequences.

**Key Insight**: Inspired by Anthropic's Cross-Layer Transcoder (CLT)—each layer's MLP output is reconstructed by accumulating decoded sparse latents from all *preceding* layers, thus explicitly modeling cumulative computation along the depth.

**Core Idea**: Replace each ESM2 MLP with a CLT, then use gradient attribution-based greedy search to find the most critical subset of latents for each downstream task—these constitute the "protein circuits," which, when visualized, can be mapped to known biological structures such as HRD catalytic motifs, Rossmann folds, and GB1 hydrophobic cores.

## Method

### Overall Architecture
ProtoMech consists of four components: (i) CLT replacement model—implement a sparse TopK encoder and cross-layer decoder for each ESM2 MLP; (ii) circuit discovery—gradient attribution plus incremental greedy selection until the circuit recovers ≥70% of original performance; (iii) steering—activation clamping on specific latents in the CLT to push wildtype sequences toward high-fitness regions within five mutations; (iv) visualizer—select top-5 activated nodes per layer, compute edge weights as activation × gradient, render the entire circuit as a readable graph, and manually cross-reference Swiss-Prot for biological meaning.

### Key Designs

1. **Cross-layer transcoder (CLT) as ESM2 replacement**:

    - **Function**: Each layer's MLP output is reconstructed from all preceding layers' sparse latents, preserving interpretability and explicitly modeling cross-layer computation.
    - **Mechanism**: For the $\ell$-th layer residual stream $\mathbf x^\ell$, encode as $\mathbf a^\ell=\text{TopK}(\mathbf W_{\text{enc}}^\ell(\mathbf x^\ell-\mathbf b_{\text{pre}}^\ell)+\mathbf b_{\text{enc}}^\ell)$, then $\hat{\mathbf y}^\ell=\sum_{\ell'=1}^{\ell}\mathbf W_{\text{dec}}^{\ell'\to\ell}\mathbf a^{\ell'}+\mathbf b_{\text{pre}}^\ell$. The training objective is $\mathcal L_{\text{MSE}}=\sum_\ell \|\mathbf y^\ell-\hat{\mathbf y}^\ell\|_2^2$ plus an auxiliary reconstruction loss $\mathcal L_{\text{aux}}$ to mitigate dead latents.
    - **Design Motivation**: Upgrades the signal from "within-layer reconstruction" to "cross-layer composition," more faithfully reproducing ESM2's actual computation path and providing compositionality for circuit discovery (later-layer latents can be interpreted as "functional compositions of earlier-layer latents").

2. **Attribution-based sparse circuit discovery algorithm**:

    - **Function**: Identify the minimal subset of latents in CLT that can recover target task performance.
    - **Mechanism**: First, train a supervised probe (logistic regression for family, CNN for function) on the original ESM2 final-layer MLP output $\mathbf y^L$; then, perform hybrid replacement on ProtoMech—MLP uses CLT, attention remains unchanged; rank latents by gradient attribution to probe output, incrementally add candidates in batches until performance recovers ≥70% or matches the full set. F1 is used for family, Spearman $\rho$ for function.
    - **Design Motivation**: Greedy + attribution avoids brute-force $2^{d_{\text{latent}}}$ search; fixing attention prevents "error accumulation from reconstructed attention," ensuring the circuit only explains the MLP computation path, consistent with Anthropic's LLM design.

3. **Activation clamping steering along the circuit**:

    - **Function**: Elevate circuits from "interpretation tools" to "generation tools" for designing high-fitness protein variants.
    - **Mechanism**: During wildtype forward pass, "clamp" the activation values of selected latents in the target function circuit—set each node to the maximum observed activation across the sequence, scaled by a multiplier; then, use Eq. (2) to reconstruct $\hat{\mathbf y}^L$ at $\ell=L$ and decode to ESM2 logits, selecting mutations by maximum probability. Variants are restricted to within five mutations of wildtype to ensure reliability of the downstream CNN evaluator.
    - **Design Motivation**: Reverse the attribution results of "which latents are important for function" to guide generation—essentially mechanistic-guided protein design; compared to CAA's global concept vector, this method only perturbs the necessary sub-circuit, resulting in cleaner and more targeted signals.

### Loss & Training
CLT uses $\mathcal L_{\text{CLT}}=\mathcal L_{\text{MSE}}+\alpha\mathcal L_{\text{aux}}$, pre-trained on 5M randomly sampled UniRef50 sequences of ≤1022 amino acids. The CLT for ESM2-8M has 28M parameters, 3.5× the original model. To alleviate the $\mathcal O(L^2)$ decoder matrix scaling bottleneck, the authors propose "windowed CLT"—each layer only attends to the previous 4 layers, reducing parameters from 207M to 125M and speeding up training by 1.75× on ESM2-35M, with family recovery dropping only from 85% to 82%.

## Key Experimental Results

### Main Results

Circuit recovery performance on two downstream tasks for ESM2-8M:

| Task | Full latent (PLT / ProtoMech) | Circuit (PLT / ProtoMech) | Circuit latent ratio |
|---|---|---|---|
| Protein family F1 | 0.50 ± 0.34 / **0.82 ± 0.19** | 0.49 ± 0.33 / **0.73 ± 0.19** | ~0.8% |
| Function Spearman $\rho$ | 0.38 ± 0.18 / **0.41 ± 0.19** | 0.35 ± 0.19 / **0.38 ± 0.18** | ~0.9% |
| Original ESM2 baseline | 0.92 (family F1) / 0.50 ($\rho$) | – | – |

Steering mean scores (excerpt) on seven DMS assays:

| Method | SPG1 | HIS7 | GFP | CAPSD | RASK |
|---|---|---|---|---|---|
| ProtoMech | 1.67 | **1.28** | 4.17 | **1.68** | -0.12 |
| PLT | **1.97** | 1.27 | **4.40** | 0.81 | -0.19 |
| CAA | 0.70 | 0.52 | 2.93 | -0.26 | -0.35 |
| Random | -2.76 | 0.56 | 2.74 | -1.04 | -0.64 |

### Ablation Study

| Configuration | Phenomenon | Meaning |
|---|---|---|
| Recursive replacement (attn also via CLT) | Significant performance collapse | Cross-layer error accumulation, attention must be fixed |
| Windowed CLT (window=4) on ESM2-35M | family 82% vs vanilla 85% | Trade-off strategy, feasible for large models |
| PLT with same sparsity control | PLT family F1 only 0.50 | Cross-layer connections, not sparsity tuning, are the source of performance |

### Key Findings
- For families where original ESM2 performs poorly (F1<0.5), ProtoMech circuits actually achieve *higher* average F1 than the original model (0.43 vs 0.39), showing a "sparse denoising regularization" effect—the circuit filters out task-irrelevant noise, making it more reliable than the original model and promising as a mechanistic filter for protein screening.
- Circuits recover 74% performance even for GFP remote mutations with mutation depth ≥5, indicating ProtoMech captures *global* functional motifs rather than overfitting local statistics.
- Visualization confirmed: In the Kinase circuit, L1 recognizes arginine R → L3 recognizes the HRD catalytic loop → L5 splits into ATP binding site and G-loop; in the NADP+ circuit, L4 recognizes the Rossmann fold → L5 narrows to the NADP+/FAD pocket. Deep layers re-activate early residues, consistent with the token reiteration phenomenon in NLP.

## Highlights & Insights
- First to adapt Anthropic's CLT paradigm to biological foundation models and successfully implement the full circuit discovery + steering pipeline, demonstrating that "mechanistic interpretability" is a cross-domain general paradigm, not unique to LLMs.
- Proposes a new mechanism-guided protein design path: instead of relying on global concept vectors or time-consuming evolutionary algorithms, directly use "which latents are responsible for high fitness" to drive mutation selection—efficient and interpretable.
- The finding that "circuits are more accurate than the original model" suggests that the sparse latent space essentially acts as a learnable regularizer, valuable for protein prediction tasks with small samples or high noise.

## Limitations & Future Work
- Circuit interpretation still relies on manual cross-referencing with Swiss-Prot, limiting scalability; an automated motif annotation pipeline is urgently needed.
- Only validated on masked LM (ESM2); whether CLT can be applied to autoregressive pLMs (ProGen) or diffusion-based pLMs (DPLM) remains unknown and is explicitly left as an open challenge.
- The parameter scaling bottleneck of $\mathcal O(L^2)$, though partially alleviated by windowed CLT, may still be prohibitive for ESM2-650M and larger models.

## Related Work & Insights
- **vs Adams 2025 (SAE on pLM)**: They use SAE to interpret single-layer representations; this work shifts to using CLT for cross-layer computation, upgrading from "features" to "circuits."
- **vs Ameisen 2025 (CLT on LLM)**: Anthropic performed LLM circuit tracing on Claude; this work is the first application in biology, demonstrating cross-domain transferability.
- **vs CAA (Huang 2025) protein steering**: CAA relies on extensive fitness labels and local mutations, prone to overfitting; ProtoMech achieves sparse intervention via circuits, with higher data efficiency and extrapolation capability.

## Rating
- Novelty: ⭐⭐⭐⭐ First to use cross-layer transcoder for pLM circuit tracing and propose a new protein steering paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers family, function, and steering tasks, two model sizes, with complementary quantitative and biological case studies.
- Writing Quality: ⭐⭐⭐⭐ Clear framework, intuitive layer-by-layer biological case interpretations (Kinase/NADP+/GB1).
- Value: ⭐⭐⭐⭐ Provides a cross-domain case for the mech-interp community and a low-cost mechanism-driven method for protein design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/medical_imaging/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[ICML 2026\] Learning the Interaction Prior for Protein-Protein Interaction Prediction: A Model-Agnostic Approach](learning_the_interaction_prior_for_protein-protein_interaction_prediction_a_mode.md)
- [\[ICML 2026\] Towards A Generative Protein Evolution Machine with DPLM-Evo](towards_a_generative_protein_evolution_machine_with_dplm-evo.md)
- [\[NeurIPS 2025\] Scaling Laws and Pathologies of Single-Layer PINNs: Network Width and PDE Nonlinearity](../../NeurIPS2025/medical_imaging/scaling_laws_and_pathologies_of_single-layer_pinns_network_width_and_pde_nonline.md)
- [\[ICLR 2026\] Protein as a Second Language for LLMs](../../ICLR2026/medical_imaging/protein_as_a_second_language_for_llms.md)

</div>

<!-- RELATED:END -->
