---
title: >-
  [Paper Note] Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining
description: >-
  [ICML 2026][Computational Biology][Multimodal Molecular Graphs] C-FREE decomposes molecules into fixed-radius $k$-EgoNet subgraphs. It combines 2D topology and multiple 3D conformations through GINE + PaiNN + Transformer…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Multimodal Molecular Graphs"
  - "Ego-Net"
  - "JEPA"
  - "Contrast-free"
  - "3D Conformations"
date: 2026-05-08
content_hash: 0db6fec0c7723842
---

# Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining

**Conference**: ICML 2026  
**arXiv**: [2509.22468](https://arxiv.org/abs/2509.22468)  
**Code**: https://github.com/ariguiba/C-FREE  
**Area**: Molecular Representation / Self-Supervised Pretraining / Graph Neural Networks  
**Keywords**: Multimodal Molecular Graphs, Ego-Net, JEPA, Contrast-free, 3D Conformations

## TL;DR
C-FREE decomposes molecules into fixed-radius $k$-EgoNet subgraphs. It combines 2D topology and multiple 3D conformations through GINE + PaiNN + Transformer encoding, followed by JEPA-style latent space prediction pretraining. The process is entirely free of negative samples, augmentations, and positional encodings. Using only 0.33M molecules (GEOM), it outperforms multimodal baselines like UniMol and MolFM (trained on 19M–77M molecules) across 8 MoleculeNet tasks.

## Background & Motivation
**Background**: Self-supervised learning (SSL) for molecular representation generally follows three paradigms: contrastive (GraphCL / GraphMVP / 3D InfoMax), generative (AttrMask / GROVER / MoleBlend), and latent space prediction (BGRL / LaGraph / GraphJEPA). Recently, multimodal fusion using 3D conformations (UniMol, GEM, MolFM) has been introduced.

**Limitations of Prior Work**: Contrastive methods rely on manual design of positive/negative samples; however, chiral isomers in molecules have near-identical structures but different properties, making augmentation-based positives problematic. Generative methods require reconstructing nodes/edges in discrete graph space, while autoregressive approaches must impose an artificial node ordering. GraphJEPA adapts JEPA to graphs but involves a heavy pipeline including METIS clustering, hyperbolic positional encodings, and hierarchical targets.

**Key Challenge**: Molecular "neighborhood structures" are the true carriers of chemical properties, yet existing SSL frameworks spend excessive computation on view generation, which dilutes the modeling of these neighborhoods. Furthermore, mainstream methods often use only 2D or only 3D, neglecting their complementarity.

**Goal**: (i) Eliminate negative samples and complex augmentations; (ii) Unify 2D topology and multiple 3D conformations into a single prediction target; (iii) Surpass UniMol (19M molecules) using the GEOM dataset (0.33M molecules).

**Key Insight**: Treat molecules like "image patches"—fixed-radius $k$-EgoNets serve as the molecular patches. Use a context patch to predict a complementary target patch in latent space, migrating the I-JEPA paradigm to graphs while removing unnecessary complexity.

**Core Idea**: Use "$k$-EgoNet subgraphs + their complementary subgraphs" as context-target pairs for $L_2$ prediction in latent space. The target encoder uses EMA, and 2D/3D modalities are concatenated into a multimodal token sequence for a Transformer. No negative samples, positional encodings, or graph reconstructions are required.

## Method

### Overall Architecture
Molecules are represented as 2D graphs $G=(V,E)$ where nodes possess 3D coordinates $r_v \in \mathbb{R}^3$ for multiple conformations. The pipeline consists of four steps: (1) View Construction—sample an anchor node $v$, take its $k$-hop neighborhood as the context subgraph, and define the remaining edges as the complementary target subgraph ($k \in \{3,4\}$; boundary nodes are shared while edges are assigned to one side to ensure no overlap); (2) Modal Encoding—2D uses GINE to output node embeddings $\{\mathbf{h}^{2D}_v\}$, 3D uses PaiNN/SchNet for each conformation $c$ to output $\{\mathbf{h}^{3D}_{v,c}\}$; (3) Multimodal Fusion—concatenate into a sequence $\mathbf{H}=[\mathbf{h}_{CLS}, \mathbf{h}_{SEP}, \{\mathbf{h}^{2D}_v\}, \mathbf{h}_{SEP}, \{\mathbf{h}^{3D}_{v,c}\}, \mathbf{h}_{SEP}]$ with learnable modal embeddings to distinguish 2D/3D, then process with a Transformer; (4) Prediction Alignment—context embeddings pass through a predictor (lightweight Transformer + MLP) to output a predicted CLS, which is aligned with the target encoder's CLS via $L_2$ distance. Context and target roles alternate during training, and multiple anchors are sampled per molecule to enrich signals.

### Key Designs

1.  **k-EgoNet complementary subgraphs as prediction units**:
    - **Function**: Replaces image patches with fixed-radius local blocks on graphs, allowing the JEPA paradigm to migrate to molecules without manual augmentations or clustering.
    - **Mechanism**: From node $v$, the $k$-hop induced subgraph serves as context and the remaining edges form the target; edges are strictly partitioned so they are **edge-disjoint** but jointly cover the graph. Multiple anchors per molecule generate an unsupervised "intra-molecular mini-batch."
    - **Design Motivation**: Despite varying molecular sizes, local chemical environments are finite. Fixed radius ensures each subgraph captures a consistent amount of information. Complementary construction creates natural structural pairs, avoiding manual definitions of positive samples. Compared to METIS, EgoNet is computed via a simple $O(|V|)$ BFS.

2.  **2D + Multi-conformation 3D multimodal token sequence**:
    - **Function**: Simultaneously models topology (covalent bonds) and geometry (atomic coordinates) within a single Transformer, explicitly incorporating multiple conformations to encode property dependence on conformational diversity.
    - **Mechanism**: GINE provides 2D atom embeddings; PaiNN/SchNet provides 3D embeddings for each conformation $c$. A BERT-style sequence `[CLS][SEP] {2D tokens} [SEP] {3D tokens (multi-conf)} [SEP]` with modal embeddings allows self-attention to aggregate information within and across modalities. The final output $\mathbf{h}_{CLS}^{out}$ is used as the subgraph representation.
    - **Design Motivation**: Many molecular properties rely on a **weighted average of multiple high-probability conformations** rather than a single 3D state. Positional encodings are omitted because GINE and PaiNN inductive biases already encode spatial/topological information; adding PE would break equivariance.

3.  **EMA target encoder + Transformer predictor for anti-collapse**:
    - **Function**: Prevents representation collapse in negative-free latent prediction and ensures the predictor refines representations rather than becoming an identity mapping.
    - **Mechanism**: The target encoder $f_{\bar{\theta}}$ is an exponential moving average of the context encoder $\bar{\theta}^{(t)} = \tau \bar{\theta}^{(t-1)} + (1-\tau)\theta^{(t)}$, with $\tau$ linearly increasing from 0.995 to 1. The loss is $\frac{1}{M}\sum_i \sum_j \|\hat{\mathbf{s}}_{y_j} - \mathbf{s}_{y_j}\|^2$. The predictor is a node-level Transformer + MLP that operates **before pooling** to preserve structural details.
    - **Design Motivation**: Following the BYOL/I-JEPA principle, EMA requires an asymmetric predictor to avoid collapse. Ablations show that removing the predictor leads to a zero SSL loss (total collapse), while a Transformer predictor achieves lower MAE than an MLP on Kraken.

### Loss & Training
The pretraining loss is the $L_2$ distance described above. Fine-tuning uses two heads: (i) C-FREE$_{\text{MOL}}$ uses whole-graph embeddings with a linear layer; (ii) C-FREE$_{\text{SUB}}$ aggregates multiple subgraph embeddings via DeepSets. Theoretically, C-FREE$_{\text{SUB}}$ + DeepSets is equivalent to ESAN and thus **strictly stronger than 1-WL** (Lemma 1). Pretraining is conducted on 330K GEOM molecules. 2D-only backbone has 4M parameters; multimodal has 9.1M. Missing conformations during fine-tuning are generated using RDKit.

## Key Experimental Results

### Main Results

**MoleculeNet 8 tasks, frozen backbone + linear probing, ROC-AUC ↑**

| Setting | Category | Representative Method | Avg |
|------|------|---------|-----|
| 2D Contrastive | CL | GraphCL | 65.04 |
| 2D Non-contrastive | Non-CL | ContextPred | 60.36 |
| **Ours 2D-MOL** | Non-CL | C-FREE$_{\text{2D-MOL}}$ | **66.63** |
| **Ours 2D-SUB** | Non-CL | C-FREE$_{\text{2D-SUB}}$ | **67.27** |
| **Ours MM-MOL** | Multi | C-FREE$_{\text{MM-MOL}}$ | **71.07** |
| **Ours MM-SUB** | Multi | C-FREE$_{\text{MM-SUB}}$ | 70.92 |

MM-MOL achieves first or second place in 6 out of 8 tasks. Even the 2D-only version outperforms all 2D baseline averages.

**MoleculeNet Full Fine-tuning (Comparison with Multimodal Models Pretrained on 19M+ Molecules)**

| Method | Pretraining Scale | MoleculeNet Avg ROC-AUC ↑ |
|------|-----------|--------------------------|
| MoleBlend | PCQM4Mv2 (3M) | 76.16 |
| GEM | ZINC-20M | 78.11 |
| UniMol | 19M molecules / 209M confs | 78.56 |
| **C-FREE$_{\text{PaiNN-3C}}$** | **GEOM 0.33M** | **79.81** |

Ours outperforms UniMol by 1 point using 1/60 of the pretraining data, achieving SOTA on BBBP, Tox21, ToxCast, and HIV.

### Ablation Study

**(a) Modal Ablation (Kraken, MAE ↓, FFT = fine-tune from pretrain, RND = random initialization)**

| Modality | Init | B5 | L | BurB5 | BurL |
|------|--------|-----|----|-------|------|
| 2D | RND | 0.297 | 0.396 | 0.205 | 0.152 |
| 2D | FFT | 0.276 | 0.340 | 0.176 | 0.146 |
| 3D | FFT | 0.194 | 0.329 | 0.134 | 0.131 |
| **MM** | **FFT** | **0.193** | **0.306** | **0.134** | **0.126** |

**(b) Predictor / EMA / k Ablation Summary**

| Configuration | Key Metric | Description |
|------|---------|------|
| Full Model (Transformer predictor + EMA τ₀=0.995 + k∈{3,4}) | Lowest Kraken MAE | Baseline |
| w/o predictor | SSL loss → 0, worst MAE | Total collapse; EMA alone is insufficient |
| MLP predictor | Intermediate | Predictor capacity matters |
| τ₀=1.0 (No EMA decay) | Kraken Avg 0.502, worse than RND | No momentum teacher, no learning |
| τ₀=0.5 (Aggressive) | Kraken Avg 0.428 (Best) | 0.995 selected for stability trade-off |
| k=1 (1-hop only) | Equal to RND | Too local, structural signal insufficient |
| k=5 | Best | Balanced context/target sizes enrich representations |

**(c) Drugs-75K Label Efficiency (1% label, MAE ↓)**

| Metric | RND | FFT | Gain |
|--------|------|------|---------|
| 1% IP | 0.638 | **0.608** | -4.7% |
| 1% EA | 0.613 | **0.583** | -4.9% |
| 1% χ | 0.334 | **0.317** | -5.1% |

Pretraining shows clear advantages in low-label scenarios, while results converge with full data.

### Key Findings
- **3D is more important than 2D**: Modal ablation reveals 3D-only almost catches up to multimodal, while 2D-only lags. Molecular properties are inherently geometry-sensitive.
- **Predictor is the true hero against collapse**: Removing it results in zero loss. The combination of "EMA + asymmetric predictor" is essential.
- **Small data + strong inductive bias beats 60x more data**: C-FREE trained on 0.33M GEOM molecules outperforms 19M UniMol, proving that "conformational diversity + subgraph prediction" is more sample-efficient than raw data volume.
- **Optimal k exists**: $k=1$ is equivalent to random initialization; $k=5$ provides optimal complexity for the JEPA task.

## Highlights & Insights
- **"Minimalist" JEPA-on-graph**: Compared to GraphJEPA, it removes METIS clustering, hyperbolic PE, and hierarchical targets, proving these graph-specific complexities are not necessary for JEPA.
- **Conformations as natural data augmentation**: Unlike contrastive learning which struggles with chirality, C-FREE incorporates multiple conformations into the encoder, turning diversity into signal rather than noise.
- **Unified multimodal tokenization**: The BERT-style sequence is an "out-of-the-box" template for feeding arbitrary geometric views into a Transformer without architectural changes.
- **Theory-experiment synergy**: Lemma 1 provides a formal guarantee (C-FREE$_{\text{SUB}}$ is equivalent to ESAN), validated by empirical results on the EXP dataset.

## Limitations & Future Work
- **Conformation generation bottleneck**: Poor performance on SIDER is linked to failed conformation generation for large molecules. Integration with stronger diffusion-based generators (e.g., Torsional Diffusion) is needed.
- **Pretraining scale not fully extended**: The sample efficiency is proven on 0.33M data, but scaling laws on 20M+ datasets (e.g., ZINC) remain unexplored.
- **Lack of SMILES/Text modality**: The authors omitted 1D representations, though text modalities can be highly beneficial in low-data scenarios.
- **Hand-picked EgoNet radius**: While $k=5$ is optimal in ablations, adaptive $k$ based on molecular size or multi-scale ego-nets could be investigated.

## Related Work & Insights
- **vs GraphJEPA (Skenderi 2025)**: Both adapt JEPA to graphs, but C-FREE outperforms it on ZINC while stripping away clustering and hyperbolic PE, proving "complexity $\neq$ necessity."
- **vs UniMol / GEM / MoleBlend (Multimodal Generative)**: These use mask reconstruction or cross-modal alignment loss. C-FREE performs latent $L_2$ prediction, showing that "prediction $\geq$ generation" for molecules.
- **vs GraphMVP / 3D InfoMax (2D-3D Contrastive)**: These rely on negative sampling, which is difficult for chiral isomers. C-FREE avoids negative samples by concatenating multi-conformations directly.
- **vs ESAN (Bevilacqua 2022)**: C-FREE extends ESAN's subgraph decomposition to SSL and recovers its expressive power upper bound via the DeepSets head.
- **vs I-JEPA / BYOL (Vision JEPA)**: Directly inherits the EMA + predictor duo but swaps image patches for $k$-EgoNets and discards PE as GNNs already capture topology.

## Rating
- Novelty: ⭐⭐⭐⭐ JEPA on graphs exists, but "minimalist complexity" + multi-conf tokenization is a solid engineering contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across MoleculeNet, QM9, Kraken, ZINC, and Drugs-75K, plus four-way ablations and theory.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and figures, though distinctions between MOL and SUB heads require detailed reading.
- Value: ⭐⭐⭐⭐⭐ Outperforming UniMol with 1/60 of the data strongly refutes the "data-only" scaling myth in molecular SSL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Protein Fold Classification at Scale: Benchmarking and Pretraining](protein_fold_classification_at_scale_benchmarking_and_pretraining.md)
- [\[AAAI 2026\] ConSurv: Multimodal Continual Learning for Survival Analysis](../../AAAI2026/computational_biology/consurv_multimodal_continual_learning_for_survival_analysis.md)
- [\[ICML 2026\] SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning](sigma_structure-invariant_generative_molecular_alignment_for_chemical_language_m.md)
- [\[NeurIPS 2025\] PROSPERO: Active Learning for Robust Protein Design Beyond Wild-Type Neighborhood](../../NeurIPS2025/computational_biology/prospero_active_learning_for_robust_protein_design_beyond_wild-type_neighborhood.md)
- [\[ICML 2026\] CARD: Coarse-to-fine Autoregressive Modeling with Radix-based Decomposition for Transferable Free Energy Estimation](card_coarse-to-fine_autoregressive_modeling_with_radix-based_decomposition_for_t.md)

</div>

<!-- RELATED:END -->
