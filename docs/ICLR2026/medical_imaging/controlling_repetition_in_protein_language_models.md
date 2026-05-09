---
title: >-
  [Paper Note] Controlling Repetition in Protein Language Models
description: >-
  [ICLR2026][Medical Imaging][protein language models] This work presents the first systematic study of pathological repetition in protein language models (PLMs), introducing a unified repetition metric $R(x)$ and a utility metric $U(x)$, and proposes UCCS (Utility-Controlled Contrastive Steering), a method that injects steering vectors decoupled from repetition into hidden layers at inference time to suppress repetition while preserving folding credibility without retraining the model.
tags:
  - ICLR2026
  - Medical Imaging
  - protein language models
  - repetition control
  - contrastive steering
  - representation engineering
  - sequence generation
date: 2026-05-08
content_hash: 6dac24ce2591e475
---

# Controlling Repetition in Protein Language Models

**Conference**: ICLR2026
**arXiv**: [2602.00782](https://arxiv.org/abs/2602.00782)
**Code**: To be confirmed
**Area**: Protein / AI4Science
**Keywords**: protein language models, repetition control, contrastive steering, representation engineering, sequence generation

## TL;DR
This work presents the first systematic study of pathological repetition in protein language models (PLMs), introducing a unified repetition metric $R(x)$ and a utility metric $U(x)$, and proposes UCCS (Utility-Controlled Contrastive Steering), a method that injects steering vectors decoupled from repetition into hidden layers at inference time to suppress repetition while preserving folding credibility without retraining the model.

## Background & Motivation
1. PLMs such as ESM-3 and ProtGPT2 have achieved significant progress in protein structure prediction and de novo design, yet frequently exhibit pathological repetition during generation — sequences collapsing into redundant motifs or long homopolymers.
2. Unlike repetition in natural language, which merely reduces readability, repetition in proteins directly disrupts structural diversity, leading to folding instability and functional loss (e.g., polyQ expansions in Huntington's disease).
3. Existing decoding strategies (temperature, top-p, n-gram penalty) are transferred directly from NLP without adaptation to protein design, and often degrade AlphaFold pLDDT scores.
4. Repetition and structural utility are highly entangled: naively reducing repetition tends to simultaneously impair folding reliability, necessitating methods that disentangle the two.
5. PLMs lack explicit mechanisms to separate repetition from other generative factors, and conventional text repetition metrics fail to capture protein-specific degeneration patterns.
6. Pathological repetition as a critical failure mode of PLMs has been entirely overlooked, with no formal definitions, evaluation metrics, or systematic investigation.

## Method

### Overall Architecture
The approach proceeds in three steps: (1) defining two canonical forms of pathological repetition — motif-level repetition (e.g., short segment cycling as in AGAGAG) and homopolymer repetition (e.g., single amino acid extension as in AAAAAA), and establishing a unified metric $R(x)$ and a utility metric $U(x)$; (2) constructing a utility-controlled contrastive dataset with $\mathcal{D}^+$ (low repetition) and $\mathcal{D}^-$ (high repetition) aligned on $U(x)$ but maximally separated on $R(x)$; (3) extracting steering vectors $v^L$ from hidden-layer activations and injecting them at inference time to suppress repetition. Repetition control is formalized as a constrained optimization: $\min_f R(f(M,p))$ s.t. $U(f(M,p)) \ge U(M,p) - \epsilon$.

### Key Design 1: Unified Repetition Metric
Three complementary metrics are proposed: (a) normalized token entropy $H_{\text{norm}}$ to capture global amino acid distributional imbalance; (b) Distinct-2/3 to capture local motif cycling; (c) a homopolymer diversity score $R_{\text{hpoly}} = 1 - \frac{1}{T}\sum_i \ell_i \cdot \mathbf{1}(\ell_i \ge 4)$ to capture long homopolymer collapse. These are aggregated into a unified score $R(x)$, while the utility score $U(x)$ is composed of the mean pLDDT and pTM from AlphaFold.

### Key Design 2: Utility-Controlled Contrastive Dataset
A candidate pool is collected from natural proteins (CATH/SCOP/UniRef50) and PLM-generated sequences, bucketed by length and filtered to remove sequences whose $U(x)$ deviates from the reference mean. The procedure then solves $\arg\max_{\mathcal{D}^+,\mathcal{D}^-} \Delta R$ s.t. $\Delta U \le \epsilon$, ensuring maximal separation in the repetition dimension with utility alignment across the contrastive set.

### Key Design 3: Steering Vector Extraction and Injection
Sequence-level representations $\phi^L(x)$ are obtained via mean pooling for MLMs and last-token embeddings for AR-LMs. The steering vector is computed as $v^L = \mathbb{E}_{\mathcal{D}^+}[\phi^L] - \mathbb{E}_{\mathcal{D}^-}[\phi^L]$. At inference time, the selected layer is intervened as $\tilde{h}_t^L = h_t^L + \alpha \cdot v^L$ (default $\alpha=1$), making UCCS plug-and-play without retraining.

### Loss & Training
UCCS is an inference-time intervention that **does not modify model parameters** and requires no additional training. It requires only a one-time construction of the contrastive dataset (100 sequences per length bucket) and extraction of the steering vector. The only hyperparameters are injection strength $\alpha$ (default 1) and layer selection $L$. The candidate pool comprises approximately 10k natural proteins and 10k PLM-generated sequences, refined into a subset via utility filtering and Pareto selection.

## Key Experimental Results

### Main Results — ProtGPT2 Unconditional Generation (Table 2a)

| Method | R↑ (CATH) | U↑ (CATH) | R↑ (SCOP) | U↑ (SCOP) |
|--------|-----------|-----------|-----------|-----------|
| Original | 0.728 | 0.621 ✓ | 0.728 | 0.621 ✓ |
| Repetition Penalty | 0.780 | 0.622 ✓ | 0.780 | 0.622 ✓ |
| **UCCS** | **0.845** | **0.711** ✓ | **0.835** | **0.722** ✓ |

### Ablation Study / Conditional Generation (Table 2b)

| Method | R↑ (CATH) | U↑ (CATH) |
|--------|-----------|-----------|
| Original | 0.836 | 0.704 ✓ |
| Temperature | 0.847 | 0.700 |
| **UCCS** | **0.877** | **0.743** ✓ |

### Key Findings
- On ESM-3, UCCS improves $R(x)$ by **+55%** relative to the original model under CATH unconditional generation.
- On ProtGPT2, UCCS achieves approximately **+20%** higher $R(x)$ than temperature sampling and is the only method that satisfies utility constraints across all datasets and generation settings.
- In conditional generation, UCCS achieves R=0.890 (highest) with U=0.737 satisfying the constraint on SCOP.
- Proteins generated by UCCS exhibit high structural confidence (dominated by pLDDT > 90 blue regions) and diverse folds.
- Neuron Deactivation and Probe Steering degrade $U$ below the constraint threshold in some settings, demonstrating less stability than UCCS.
- Ablation experiments show that $\alpha$ is stable over a wide range, and layer selection has a moderate but not extreme effect on results.
- Jensen-Shannon divergence heatmaps confirm that the proposed metrics cleanly separate PLM-generated sequences from the natural protein distribution.

## Highlights & Insights
- **Pioneering scope**: This is the first work to systematically identify, quantify, and address pathological repetition in PLMs, filling a critical gap in the field.
- **Elegant decoupling design**: The utility-controlled contrastive dataset construction ensures that the steering vector encodes only repetition signals rather than confounds related to folding capability.
- **Plug-and-play**: No model retraining is required; vectors are injected at inference time and applicable to both MLM and AR-LM paradigms.
- **Biologically grounded**: The threshold $k=4$ in the homopolymer diversity score $R_{\text{hpoly}}$ has explicit biological justification, as homopolymers of length ≥4 are almost invariably low-complexity, unstable regions.

## Limitations & Future Work
- Validation is limited to ESM-3 and ProtGPT2; larger-scale PLMs (e.g., ESM-2 15B or ProGen2) are not evaluated.
- $U(x)$ is based on AlphaFold confidence rather than experimental validation, leaving a gap in the assessment of folding reliability.
- The interpretability of the steering vector $v^L$ merits further investigation — it remains unclear precisely what features it encodes in representation space.
- Contrastive dataset construction relies on precomputed $R(x)$ and $U(x)$, and may require recalibration for proteins from new domains (e.g., membrane proteins).

## Related Work & Insights
- **vs. Repetition Penalty (decoding-based)**: Simple but coarse; achieves R=0.780 vs. UCCS R=0.845 on ProtGPT2, while UCCS simultaneously yields a substantial gain in $U$ (0.622→0.711).
- **vs. Probe Steering**: Uses probe-trained steering vectors but does not control for utility entanglement; achieves R=0.735 with U=0.619 (below original) on SCOP, compared to UCCS at R=0.835/U=0.722.
- **vs. Activation Steering (NLP)**: UCCS transfers the representation steering paradigm from NLP safety/sentiment domains to protein generation, with the key innovation being utility-controlled contrastive dataset construction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to identify and systematically address pathological repetition in PLMs, with pioneering problem formulation and methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two models × three datasets × two generation settings with comprehensive comparisons.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear, metric design is well-justified, and the overall structure is rigorous.
- Value: ⭐⭐⭐⭐ Provides a practical tool for reliable protein generation with PLMs; the method is generalizable to other generative degeneration problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Protein as a Second Language for LLMs](protein_as_a_second_language_for_llms.md)
- [\[AAAI 2026\] ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders](../../AAAI2026/medical_imaging/protsae_disentangling_and_interpreting_protein_language_models_via_semantically-.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[CVPR 2026\] Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation](../../CVPR2026/medical_imaging/multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_.md)
- [\[ICLR 2026\] Reverse Distillation: Consistently Scaling Protein Language Model Representations](reverse_distillation_consistently_scaling_protein_language_model_representations.md)

</div>

<!-- RELATED:END -->
