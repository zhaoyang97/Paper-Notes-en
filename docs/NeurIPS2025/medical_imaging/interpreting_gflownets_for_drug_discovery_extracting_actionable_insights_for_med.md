---
title: >-
  [Paper Note] Interpreting GFlowNets for Drug Discovery: Extracting Actionable Insights for Medicinal Chemistry
description: >-
  [NeurIPS 2025][Medical Imaging][GFlowNet] This work constructs a multi-level interpretability toolkit for SynFlowNet (a GFlowNet grounded in synthetic reaction templates), integrating gradient saliency, counterfactual perturbation, sparse autoencoders (SAE), and motif probes to reveal how internal representations encode physicochemical properties and functional group information relevant to medicinal chemistry.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - GFlowNet
  - Interpretability
  - Drug Discovery
  - Sparse Autoencoder
  - Counterfactual Analysis
date: 2026-05-08
content_hash: 7421faf995df59fa
---

# Interpreting GFlowNets for Drug Discovery: Extracting Actionable Insights for Medicinal Chemistry

**Conference**: NeurIPS 2025
**arXiv**: [2511.19264](https://arxiv.org/abs/2511.19264)
**Code**: [GitHub](https://github.com/amirtha-montai/synflownet_public/tree/main/src/interpretability)
**Area**: Drug Discovery / Explainable AI
**Keywords**: GFlowNet, Interpretability, Drug Discovery, Sparse Autoencoder, Counterfactual Analysis

## TL;DR

This work constructs a multi-level interpretability toolkit for SynFlowNet (a GFlowNet grounded in synthetic reaction templates), integrating gradient saliency, counterfactual perturbation, sparse autoencoders (SAE), and motif probes to reveal how internal representations encode physicochemical properties and functional group information relevant to medicinal chemistry.

## Background & Motivation

**Background**: GFlowNets are a powerful class of generative models that learn stochastic policies to sequentially construct molecules, distributing probability mass across multiple high-reward structures. SynFlowNet further incorporates synthetic reaction constraints to ensure the synthetic feasibility of generated molecules.

**Limitations of Prior Work**: Despite strong performance, the internal decision-making strategies of GFlowNets remain opaque. Medicinal chemists need to understand why a model favors certain scaffolds or synthetic routes; without such understanding, these models are difficult to trust and integrate into DMTA cycles.

**Key Challenge**: Existing attribution methods are primarily designed for supervised learning or diffusion models and are ill-suited for discrete, graph-based generative policies. GFlowNets must select among heterogeneous action types, and the rich structural information in intermediate graph states has never been systematically analyzed.

**Goal**: To develop the first systematic interpretability framework for GFlowNets, providing multi-scale explanations spanning the atomic, latent, and motif levels.

**Key Insight**: Adapting established methods from supervised learning and network dissection (integrated gradients, SAEs, probe classifiers) to structured sequential graph-generation policies.

**Core Idea**: The internal representations of SynFlowNet organize drug-likeness (QED) along interpretable axes such as polarity, lipophilicity, and molecular size, with functional group information encoded in embeddings in a linearly decodable manner.

## Method

### Overall Architecture

Three complementary interpretability methods are proposed: ① gradient saliency combined with counterfactual QED analysis for atom- and motif-level attribution; ② sparse autoencoders (SAE) to discover latent factors and correlate them with physicochemical properties; ③ motif probes to detect whether embeddings encode discrete functional groups.

### Key Designs

1. **Gradient Saliency and Counterfactual Analysis**:

    - **Function**: Estimate atom-level saliency and assess causal influence on QED through structural editing.
    - **Design Motivation**: Pure gradient attribution provides only correlational evidence; counterfactual editing yields interventional causal attribution.
    - **Mechanism**: Integrated gradients (IG) are applied to the log probability of the Stop action with $M=64$ interpolation steps:
    $\mathrm{IG}_i(\text{Stop}, s_t) \approx (x_i - \tilde{x}_i) \frac{1}{M} \sum_{m=1}^{M} \nabla_{x_i} \log \pi_\theta(\text{Stop} \mid s_t^{(m)})$
    High-saliency connected components and ring systems are extracted as candidate motifs; chemically driven RDKit transformation rules (e.g., ether→thioether, methyl→fluorine, chlorine→bromine) are applied to each motif, and $\Delta\text{QED} = \text{QED}(m') - \text{QED}(m)$ is computed.
    - **Novelty**: Focusing on the Stop action enables attribution to the complete molecular graph, though the analysis is currently limited to the final decision step.

2. **Sparse Autoencoder (SAE)**:

    - **Function**: Discover disentangled physicochemical factors in SynFlowNet embeddings.
    - **Design Motivation**: Assess whether the model organizes drug-relevant properties in an interpretable manner.
    - **Mechanism**: An SAE with architecture $256 \to 128 \to 256$ is trained on pooled embeddings from the frozen final graph Transformer layer of SynFlowNet, with the objective:
    $\mathcal{L} = \|h - \hat{h}\|_2^2 + \lambda \|z\|_1$
    where $\lambda = 0.01$, target sparsity 0.05, Adam optimizer with lr=$10^{-3}$, trained for 200 epochs.
    - **Novelty**: SAE analysis is applied to generative model embeddings rather than the conventional LLM setting.

3. **Motif Probes**:

    - **Function**: Test whether discrete chemical motifs are linearly encoded in the embeddings.
    - **Design Motivation**: Bridge abstract representations with recognizable medicinal chemistry concepts.
    - **Mechanism**: With the GFlowNet frozen, a 3-layer MLP classifier ($256 \to 256 \to 128 \to 64 \to 1$) is trained for each SMARTS-defined motif using BCEWithLogitsLoss, class-balanced sampling, and 50 epochs.
    - **Novelty**: The probing methodology, common in NLP, is systematically applied to molecular generative models for the first time.

### Loss & Training

- The SAE is trained on 32,054 molecular embeddings, with a train/test split of 28,848/3,206.
- A reward predictor is additionally trained for 100 epochs with dropout=0.2.
- All analyses are based on SynFlowNet trained with QED as the reward function.

## Key Experimental Results

### Main Results: SAE Reward Prediction

| Reward Signal | Train $R^2$ | Test $R^2$ |
|---------|------------|-----------|
| Drug-likeness (QED) | 0.289 | 0.251 |
| Complexity | 0.756 | 0.750 |
| Lipophilicity | 0.670 | 0.664 |
| Size | 0.735 | 0.711 |
| **Polarity** | **0.920** | **0.918** |
| Flexibility | 0.488 | 0.502 |

Polarity and molecular size alone achieve $R^2 > 0.7$, whereas composite QED reaches only 0.25, indicating that QED is a nonlinear combination of underlying physicochemical properties that are nonetheless linearly encoded.

### Key Factor Correlations

| Factor | Reward Signal | Correlation $r$ |
|------|---------|------------|
| Factor_11 | Size | 0.757 |
| Factor_75 | Size | -0.574 |
| Factor_86 | Polarity | -0.570 |
| Factor_118 | Polarity | 0.540 |

### Motif Probe Results

| Motif | Prevalence | AUROC | AP |
|------|-------|-------|-----|
| halogen_F | 0.322 | 1.000 | 1.000 |
| halogen_Cl | 0.147 | 1.000 | 1.000 |
| aromatic_ring | 0.906 | 1.000 | 1.000 |
| nitrile | 0.225 | 0.997 | 0.989 |
| methyl | 0.733 | 0.996 | 0.998 |
| amide | 0.528 | 0.923 | 0.927 |

Mean AUROC ≈ 0.95; halogens, aromatic rings, and ionizable groups achieve AUROC > 0.99.

### Ablation Study / Key Findings

- **Counterfactual saliency**: Modifications to high-saliency motifs produce the largest QED changes, validating the reliability of the attributions.
- **SAE sparsity**: Mean activation sparsity is 0.105; most factors are active in fewer than 10% of molecules.
- **Ground-truth validation**: The motif correlation matrix extracted by SynFlowNet is consistent with the physicochemical correlation matrix computed by RDKit.

## Highlights & Insights

- **First interpretability framework for GFlowNets**: Addresses an important gap in the transparency of generative policies.
- **Multi-scale design**: Hierarchical explanations ranging from the atomic level (IG) to the concept level (SAE) to the functional-group level (probes).
- **Interventional attribution**: Counterfactual editing provides actionable medicinal chemistry recommendations rather than merely post-hoc explanations.
- **Disentangled physicochemical axes**: Although QED is difficult to predict directly, its constituent properties (polarity $R^2$=0.92, size $R^2$=0.71) are clearly encoded.
- The framework is broadly generalizable and transferable to other GFlowNet variants.

## Limitations & Future Work

- Only a single reward function (QED) is analyzed; the framework has not been extended to multi-objective settings (synthetic feasibility, binding affinity, etc.).
- Gradient saliency is applied only to the Stop action (final step) and does not cover attribution for intermediate decision steps.
- The SAE and probing methods impose relatively simple structural assumptions; more complex reward landscapes may require nonlinear disentanglement.
- Future directions include conditioning GFlowNets directly on physicochemical properties so that latent representations naturally align with medicinal chemistry axes.

## Related Work & Insights

- **SynFlowNet**: A GFlowNet grounded in synthetic reaction templates that guarantees synthesizability of generated molecules.
- **SAE in LLM interpretability**: Popularized by Anthropic's work on monosemanticity; this paper constitutes the first transfer of this approach to molecular generation.
- **Counterfactual explanations in GNNs**: Explored by CF-GNNExplainer; this work extends the paradigm to generative policies.
- **Insight**: Interpretability tools should not serve as post-hoc decorations but should be integrated into model design (e.g., conditional generation).

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic interpretability framework for GFlowNets, though individual sub-methods are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three methods provide mutual corroboration with quantitative analysis, but evaluation is limited to a single reward function.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with effective bridging between medicinal chemistry and ML, though brevity leaves some details underspecified.
- **Value**: ⭐⭐⭐⭐ Substantially advances the practical deployment of drug discovery AI and fills an important gap.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] GFlowNets for Learning Better Drug-Drug Interaction Representations](gflownets_for_learning_better_drug-drug_interaction_representations.md)
- [\[NeurIPS 2025\] Compressing Biology: Evaluating the Stable Diffusion VAE for Phenotypic Drug Discovery](compressing_biology_evaluating_the_stable_diffusion_vae_for_phenotypic_drug_disc.md)
- [\[NeurIPS 2025\] Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules](pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)
- [\[NeurIPS 2025\] Online Feedback Efficient Active Target Discovery in Partially Observable Environments](online_feedback_efficient_active_target_discovery_in_partially_observable_enviro.md)
- [\[NeurIPS 2025\] Robust or Suggestible? Exploring Non-Clinical Induction in LLM Drug-Safety Decisions](robust_or_suggestible_exploring_non-clinical_induction_in_llm_drug-safety_decisi.md)

<!-- RELATED:END -->
