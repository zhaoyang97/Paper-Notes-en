---
title: >-
  [Paper Note] PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion
description: >-
  [ICML 2025][Computational Biology][Therapeutic peptide design] PepTune combines a Masked Discrete Language Model (MDLM) with a Monte Carlo Tree Search (MCTS) multi-objective-guided strategy within the discrete peptide SMILES space to optimize multiple therapeutic properties (binding affinity, solubility, membrane permeability, etc.) simultaneously, enabling the de novo design of peptide drugs containing non-canonical amino acids and cyclization modifications.
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Therapeutic peptide design"
  - "discrete diffusion"
  - "multi-objective optimization"
  - "MCTS"
  - "SMILES"
date: 2026-05-08
content_hash: 2eb3d577e5130406
---

# PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion

**Conference**: ICML 2025  
**arXiv**: [2412.17780](https://arxiv.org/abs/2412.17780)  
**Code**: [https://huggingface.co/ChatterjeeLab/PepTune](https://huggingface.co/ChatterjeeLab/PepTune)  
**Area**: Computational Biology  
**Keywords**: Therapeutic peptide design, discrete diffusion, multi-objective optimization, MCTS, SMILES

## TL;DR
PepTune combines a Masked Discrete Language Model (MDLM) with a Monte Carlo Tree Search (MCTS) multi-objective-guided strategy within the discrete peptide SMILES space to optimize multiple therapeutic properties (binding affinity, solubility, membrane permeability, etc.) simultaneously, enabling the de novo design of peptide drugs containing non-canonical amino acids and cyclization modifications.

## Background & Motivation

**Background**: Peptide therapeutics (such as GLP-1 receptor agonists semaglutide/liraglutide) have achieved milestone successes in treating diseases like diabetes and obesity. Peptides are capable of binding diverse protein surfaces and disrupting protein-protein interactions; since 2000, 33 therapeutic peptides have been approved by the FDA.

**Limitations of Prior Work**: Designing peptides that simultaneously satisfy multiple conflicting objectives (e.g., binding affinity, solubility, and membrane permeability) remains a major challenge. Existing methods are limited to (1) continuous spaces, (2) unconditional generation, or (3) single-objective guidance. Traditional methods rely on screening random combinatorial libraries of $10^{12}$ magnitude.

**Key Challenge**: Therapeutic peptides require non-canonical amino acids (nAAs) and cyclization modifications to improve stability and permeability, but existing deep learning models can only process 20 standard amino acids. Meanwhile, multi-objective guidance in discrete spaces is extremely difficult because gradients cannot be directly computed.

**Goal**: To build a diffusion model that can perform multi-objective conditional generation in the discrete peptide SMILES space.

**Key Insight**: (1) Representing peptides using SMILES instead of amino acid sequences to support nAAs and cyclization; (2) Using MCTS rather than gradient guidance to solve the challenge of discrete-space guidance.

**Core Idea**: MDLM is responsible for exploring valid structures within the discrete peptide space, while MCTS guides the generation process to evolve towards the Pareto-optimal direction across multiple therapeutic targets.

## Method

### Overall Architecture

- **Input**: Target protein sequence + list of therapeutic properties to be optimized
- **First Stage**: Pre-train PepMDLM (unconditional generator) on 11 million peptide SMILES
- **Second Stage**: Perform multi-objective conditional guidance on the generation process using an MCTS strategy guided by property classifiers
- **Output**: A set of Pareto-optimal peptide SMILES + corresponding property scores

### Key Designs

1. **State-dependent Masking Schedule**:

    - Key Insight: Peptide bonds are the foundational structure of all valid peptides.
    - Designing a polynomial masking schedule: $\alpha_t(\mathbf{x}_0) = 1 - t^w$ for peptide bond tokens, and $\alpha_t = 1 - t$ for non-peptide bond tokens.
    - Peptide bond tokens are masked later in the forward process and unmasked earlier in the reverse process.
    - The loss weight of peptide bond tokens in training is scaled up by a factor of $w$: $\frac{w}{t} \log \langle \mathbf{x}_0, \mathbf{x}_\theta \rangle$.
    - **Why**: The vast majority of arbitrary SMILES are not valid peptides—letting the model first "build the backbone" before filling in side chains is crucial.

2. **Global Sequence Invalidity Loss (Invalid Loss)**:

    - Take the argmax of predicted probabilities to obtain a discrete sequence, then check if it represents a valid peptide.
    - Invalid sequences backpropagate gradients using a penalty weighted by softmax probabilities.
    - $\mathcal{L}_{\text{invalid}} = \sum_\ell \text{SM}(x_{\theta,k}^{(\ell)}) \cdot \mathbf{1}[\tilde{\mathbf{x}}_0 \text{ is Invalid}]$
    - **Why**: Since argmax is non-differentiable, the softmax probability is used as a surrogate to bypass this bottleneck.

3. **MCTS Multi-Objective Guidance**:

    - **Selection**: Starting from the root node (fully masked), non-dominated child paths are selected based on cumulative rewards.
    - **Expansion**: Sample $M$ different unmasking schemes using Gumbel noise.
    - **Rollout**: Greedily unmask to a complete sequence and calculate scores for $K$ objectives using classifiers.
    - **Backpropagation**: The reward vector is backpropagated to update all nodes along the path.
    - Reward Definition: $r_k(\mathbf{x}) = \frac{1}{|\mathcal{P}^*|} \sum_n \mathbb{I}[s_k(\mathbf{x}) \geq s_k(\tilde{\mathbf{x}}_n)]$ (the proportion of designs beaten in the Pareto set).
    - Penalty for invalid peptides: Deduct scores proportional to the invalidity ratio across all dimensions.
    - **Why**: Since the discrete space lacks gradients, MCTS achieves gradient-free multi-objective guidance through search and reward feedback.

### Loss & Training

- Total Loss: $\mathcal{L} = \mathcal{L}_{\text{NELBO}}^\infty + \mathcal{L}_{\text{invalid}}$
- RoFormer Backbone: 8 layers, 768 hidden dimensions, 8 attention heads
- Training Data: 11 million peptide SMILES (synthetic data from SmProt + CycloPs)
- PeptideCLM SPE Tokenization: 581 tokens, averaging 4 characters/token
- 8×A6000 GPUs, 1600 GPU hours, AdamW, lr=3e-4
- MCTS: 128 iterations, 50 children/expansion

## Key Experimental Results

### Main Results

| Target Protein | Property | Best Docking Score | Baseline/Control |
|---|---|---|---|
| TfR (Blood-brain barrier) | Binding + Solubility + Non-hemolytic | -8.4 kcal/mol | T7 peptide: -8.4 (but PepTune is shorter) |
| GLP-1R (Diabetes) | Binding + Solubility + Non-hemolytic | -7.4 kcal/mol | Semaglutide: -5.7 (longer) |
| GFAP (Alexander disease) | Binding + Permeability + Solubility + Non-hemolytic| -8.5 kcal/mol | No known peptide binder |
| TfR+GLAST (Dual-target) | Dual-target binding + Solubility | TfR: -10.5, GLAST: -9.2 | First dual-target peptide design |

### Ablation Study

| Configuration | Validity↑ | Diversity | Description |
|---|---|---|---|
| PepMDLM (No Guidance) | 45% (len=100) | 0.705 | Baseline unconditional model |
| PepTune (MCTS-Guided) | **100%** (after 20 iter) | 0.677 | MCTS boosts validity to 100% |
| HELM-GPT (Control) | 83.9% | 0.595 | HELM representation, does not support nAAs |
| Without state-dependent mask | ~30% | - | Validity drops significantly |
| Without invalid loss | ~35% | - | Validity drops |

### Key Findings

- MCTS guidance achieves a 100% valid peptide generation rate after just 20 iterations.
- The generated peptides are rich in non-canonical amino acids (average of 2.94 per peptide) and cyclization structures.
- Multi-objective guidance does not significantly sacrifice diversity (dropping by only 0.03 compared to unconditional generation).
- Dual-target peptide experiments demonstrate that PepTune can optimize binding affinities for two proteins simultaneously.
- The docking score of the GLP-1R binder outperforms semaglutide with a shorter sequence.

## Highlights & Insights

1. **First Discrete-Space Multi-Objective Guided Diffusion**: The integration of MCTS and masked diffusion is a significant contribution to the field.
2. **Advantages of SMILES Representation**: Supports nAAs and cyclization, greatly expanding the designable space of peptides.
3. **Physical Intuition of State-Dependent Masking**: The generation order of "backbone first, side chains second" aligns well with chemical intuition.
4. **Clinical Relevance**: Case studies on multiple real-world therapeutic targets demonstrate promising prospects for downstream experimental validation.

## Limitations & Future Work

1. Reliance on synthetic peptide data (CycloPs); rare nAAs may increase the difficulty and cost of synthesis.
2. The quality of property classifiers is a bottleneck—properties like membrane permeability lack external validation.
3. MCTS sampling speed is relatively slow (128 iterations × 50 children × rollout).
4. Lack of wet-lab validation (only computational docking has been performed).

## Related Work & Insights

- MDLM (Sahoo et al.) provides the foundation for masked discrete diffusion.
- Structure-based models like RFpeptides require the target's 3D structure, whereas PepTune only needs the sequence.
- Inspiration: The MCTS guidance strategy can be extended to other discrete biological sequence generation tasks such as protein design and DNA sequence optimization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of MCTS + discrete diffusion + peptide SMILES is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 single-target + 2 dual-target case studies, with detailed ablation details.
- Writing Quality: ⭐⭐⭐⭐ Rich content but somewhat lengthy.
- Value: ⭐⭐⭐⭐⭐ Significant practical value for drug design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](../../NeurIPS2025/computational_biology/uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[ICML 2025\] Latent Imputation before Prediction: A New Computational Paradigm for De Novo Peptide Sequencing](latent_imputation_before_prediction_a_new_computational_paradigm_for_de_novo_pep.md)
- [\[NeurIPS 2025\] De novo generation of functional terpene synthases using TpsGPT](../../NeurIPS2025/computational_biology/de_novo_generation_of_functional_terpene_synthases_using_tpsgpt.md)
- [\[ICML 2025\] UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design](unimomo_unified_generative_modeling_of_3d_molecules_for_de_novo_binder_design.md)
- [\[ICML 2025\] Kinetic Langevin Diffusion for Crystalline Materials Generation](kinetic_langevin_diffusion_for_crystalline_materials_generation.md)

</div>

<!-- RELATED:END -->
