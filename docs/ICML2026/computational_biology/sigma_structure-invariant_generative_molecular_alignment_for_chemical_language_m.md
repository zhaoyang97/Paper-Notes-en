---
title: >-
  [Paper Note] SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning
description: >-
  [ICML 2026][Computational Biology][SMILES] SIGMA uses token-level contrastive loss to force the latent states of different SMILES permutations of the same molecule to align onto the same trajectory. Complemented by IsoBe…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "SMILES"
  - "Contrastive Learning"
  - "Trajectory Alignment"
  - "Isomorphism Beam Search"
  - "Molecular Generation"
date: 2026-05-08
content_hash: e729f8d476975885
---

# SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning

**Conference**: ICML 2026  
**arXiv**: [2603.25062](https://arxiv.org/abs/2603.25062)  
**Code**: None  
**Area**: Graph Learning / Chemical Language Models / Autoregressive Generation  
**Keywords**: SMILES, Contrastive Learning, Trajectory Alignment, Isomorphism Beam Search, Molecular Generation

## TL;DR
SIGMA uses token-level contrastive loss to force the latent states of different SMILES permutations of the same molecule to align onto the same trajectory. Complemented by IsoBeam, which prunes isomorphic redundant paths during decoding, it enables sequence models to truly "think by graph rather than by string" in chemical space.

## Background & Motivation

**Background**: Current chemical language models (ChemLMs) serialize molecular graphs into SMILES strings for autoregressive generation using Transformers. This "linguistic modeling" leverages pre-training on hundreds of millions of unlabeled samples from databases like PubChem, ChEMBL, and ZINC, and is widely used for de novo drug design, property prediction, and activity modeling.

**Limitations of Prior Work**: A single molecular graph corresponds to **factorially many** valid SMILES representations (depending on traversal order), yet models treat these equivalent representations as completely distinct sequences. Consequently, different prefixes of the same molecule are mapped to orthogonal positions in the latent space, a phenomenon the authors call "Trajectory Divergence," which leads to "Manifold Fragmentation"—where chemical space is partitioned into isolated islands based on syntax rather than structure. This is particularly harmful for reinforcement learning-driven molecular optimization: agents may get trapped in a syntactic region, repeatedly sampling the same scaffold, leading to mode collapse.

**Key Challenge**: Graph models (MPNN/GraphAF) have built-in permutation invariance but sacrifice the scalability of Transformers; sequence models offer scalability but lack geometric inductive biases. Existing Randomized SMILES data augmentation only provides passive exposure, and models often memorize high-frequency permutations instead of learning structural equivalence. What is needed is a method that retains sequential efficiency while enforcing geometric invariance.

**Goal**: (1) Explicitly align structurally equivalent prefixes to the same latent state during training without abandoning SMILES representations; (2) Eliminate the waste caused by "multiple paths decoding to the same molecule" during beam search in the inference phase; (3) Maintain compatibility with existing Transformer training pipelines without introducing extra encoders.

**Key Insight**: The authors observe that if two different SMILES prefixes can be concatenated with the **exact same suffix** to produce the same molecule, they are chemically equivalent as they point to the same intermediate subgraph. This provides a rigorous criterion for "Functional Equivalence," avoiding pseudo-positives that "look similar but are chemically incompatible."

**Core Idea**: Use token-level contrastive loss to align prefixes that "share the same suffix" onto the same latent trajectory, while pushing away chemically different prefixes as structural negatives, allowing the autoregressive model to "behave like a graph model" in the latent space.

## Method

### Overall Architecture
SIGMA consists of three core modules: (1) **Functional Equivalence View Construction**—sampling two different traversals from the same molecular graph to obtain a positive pair $(p_u, p_v)$ and using an InChIKey hash oracle $\mathcal{H}$ to verify $\mathcal{H}(\text{Mol}(p_u \oplus s)) \equiv \mathcal{H}(\text{Mol}(p_v \oplus s))$; (2) **Decoupled Projection Head Contrastive Learning**—adding a non-linear projection head $g_\phi$ to the Transformer backbone to map latent states $\mathbf{h}_t$ to a contrastive metric space $\mathbf{z}_t$, preventing the contrastive objective from competing with the MLE task for syntactic details; (3) **IsoBeam Inference**—dynamically detecting whether prefixes correspond to the same subgraph during beam search and pruning low-probability isomorphic paths to redistribute the budget to structurally distinct branches. The training objective is MLE loss + token-level "suffix-align + prefix-repel" contrastive loss, replaced by IsoBeam during inference.

### Key Designs

1.  **Functional Equivalence View and Probe Suffix Protocol**:
    *   Function: Construct strictly positive pairs—two syntactically different but structurally equivalent prefixes.
    *   Mechanism: Two traversals $S^u, S^v$ are randomized from the original SMILES, and a common split point is found to divide the sequences into $(p, s)$, requiring $p_u \neq p_v$ (syntactic divergence) but $\mathcal{H}(\text{Mol}(p_u \oplus s)) \equiv \mathcal{H}(\text{Mol}(p_v \oplus s)) \equiv \mathcal{H}(\mathcal{G})$ (structural equivalence). Since incomplete SMILES prefixes are often chemically invalid (e.g., unclosed rings), the authors introduce the **Probe Suffix Protocol**: if the split point creates dangling bonds during training, a stable cap fragment $s_{probe}$ (such as a methyl group or ring closure) is temporarily appended for structural verification. Additionally, **Structural Negatives** are introduced: negative prefixes where $\mathcal{H}(\text{Mol}(p_{neg} \oplus s)) \neq \mathcal{H}(\mathcal{G})$ (e.g., stereoisomers or scaffold hops) are explicitly selected from the batch to force the model to distinguish "true isomorphism" from "apparent similarity."
    *   Design Motivation: To avoid syntactic false positives from random augmentation and ensure contrastive signals strictly reflect topological equivalence rather than string similarity.

2.  **Decoupled Projection Head + Dense Trajectory Alignment**:
    *   Function: Pull equivalent prefixes closer and push negative prefixes apart in the latent space without harming the MLE task's dependence on syntactic details.
    *   Mechanism: Applying contrastive loss directly to the backbone latent states $\mathbf{H}$ would conflict with MLE—MLE requires precise distinction of syntactic features like "ring index 1 vs 2," while contrastive learning aims to erase such differences. The authors introduce a 2-layer MLP projection head $\mathbf{z}_t = W^{(2)} \sigma(W^{(1)} \mathbf{h}_t + b^{(1)}) + b^{(2)}$ to move the contrastive loss to a projection space $\mathcal{Z}$. The contrastive objective acts on **each token position of the matching suffix** (suffix-align, positive signal) and performs prefix-repel (negative signal) at **mismatched prefix positions**, involving both token output distribution alignment and cross-attention weight alignment to form "dense trajectory alignment."
    *   Design Motivation: Global [CLS] alignment like SimCLR/MoCo is insufficient for autoregressive generation—each token decision step requires geometrically consistent latent states. Token-level alignment is needed to guide step-by-step decoding.

3.  **IsoBeam: Structure-Aware Beam Search**:
    *   Function: Dynamically eliminate redundant paths during the decoding phase where "different SMILES strings actually correspond to the same molecule."
    *   Mechanism: Standard beam search on large chemical molecules often results in multiple paths in the top-k decoding to different representations of the same molecule (e.g., multiple SMILES for acetophenone), wasting search budget. IsoBeam performs a **Partial Graph Check** on prefixes in the current beam at each step: if two prefixes correspond to isomorphic subgraphs with identical open connection points, only the higher-probability one is kept, and the budget for the other is recycled for other scaffolds (e.g., switching from a benzene scaffold to a pyridine scaffold).
    *   Design Motivation: Trajectory invariance from the training phase should also be recognized and utilized during inference—otherwise, even if the model learns equivalence, the output may still appear to have "low structural diversity" due to redundancy.

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_{\text{MLE}} + \lambda \mathcal{L}_{\text{contrast}}$, where the contrastive term includes suffix-align (InfoNCE-style positive alignment) and prefix-repel (structural negative pushing), controlled by a temperature parameter $\tau$. Randomized SMILES pairs are sampled online for each batch and verified for hash equivalence; pairs that fail verification are discarded. The projection head is trained jointly with the backbone.

## Key Experimental Results

### Main Results
The paper compares against strong baselines (standard ChemLM, Randomized SMILES augmentation, CONSMI global contrastive, SimCTG self-contrastive, LO-ARM graph generator) on standard multi-parameter molecular optimization (MPO) benchmarks. Evaluation metrics include sample efficiency, structural diversity, and property optimization score.

| Task Category | Metric | SIGMA | Prev. SOTA | Gain Description |
| :--- | :--- | :--- | :--- | :--- |
| MPO | Top-K Molecules | Significantly Leads | Randomized SMILES | Large improvement in sample efficiency |
| Structural Diversity | Unique Scaffolds | Significantly Leads | Standard beam search | IsoBeam recycles budget for distinct scaffolds |
| Latent Alignment | Cosine Sim of Isomorphs | Near 1 | < 0.5 | Verified that Manifold Fragmentation is fixed |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full SIGMA (suffix-align + prefix-repel + IsoBeam) | Optimal | — |
| w/o Structural Negatives | Diversity Drop | In-batch random negatives are insufficient for stereoisomers |
| w/o Projection Head | Higher Perplexity | Direct contrastive learning on backbone hurts generation quality |
| w/o IsoBeam (SIGMA training, standard beam) | Unique Scaffolds Drop | Training alignment alone cannot fully eliminate inference redundancy |
| w/o suffix-align (global CLS only) | Weaker Alignment | Verifies the necessity of token-level dense alignment |

### Key Findings
*   The projection head is essential: Direct contrastive learning on the backbone creates a tug-of-war with the MLE objective, degrading token prediction accuracy.
*   IsoBeam and training-stage alignment are complementary: Alignment addresses "whether the model knows equivalence," while IsoBeam addresses "whether the output avoids repetition."
*   Structural negatives (stereoisomers/scaffold hops) significantly improve fine-grained discrimination; regular in-batch negatives are insufficient.
*   The Probe Suffix ensures equivalence is determined based on stable topology rather than transient states, avoiding false judgments caused by chemically invalid intermediates.

## Highlights & Insights
*   **Geometric Consistency = Latent Constraint**: Explicitly encoding "graph symmetries that sequential models should respect" into the latent space via token-level contrast is an elegant way to inject "graph model inductive bias" into Transformers.
*   **Training-Inference Duality**: Suffix-align (learning equivalence during training) + IsoBeam (using equivalence during inference) forms a complete loop, avoiding the common pitfall where "training learns something the inference cannot utilize."
*   **InChIKey Hash Oracle** as an equivalence criterion is clean and rigorous, avoiding heuristic false positives based on edit distance or substring matching.
*   This "trajectory alignment" concept is transferable to any "one-to-many serialization" problem, such as different expression orderings for equivalent ASTs in code generation, point cloud sequences for 3D shapes, or different notations for chemical reaction paths.

## Limitations & Future Work
*   Hash verification relies on chemoinformatics tools like RDKit, which may fail for macrocycles or unconventional molecules, and calculating hashes for every batch introduces extra overhead.
*   The choice of Probe Suffix (methyl cap vs. ring closure) affects the boundary of equivalence determination—in extreme cases, different probes might yield different conclusions.
*   IsoBeam's Partial Graph Check has its own computational complexity, which may become a bottleneck for large beams or long sequences, requiring engineering optimization.
*   The paper focuses on SMILES; whether it is equally effective for more robust linear representations like SELFIES or DeepSMILES needs further verification.
*   Future work could extend this to reaction SMILES, 3D conformations, and multi-modal molecular representations.

## Related Work & Insights
*   **vs Randomized SMILES (Bjerrum 2017)**: They rely on data augmentation to **passively** expose equivalent permutations; SIGMA uses contrastive loss to **actively** enforce equivalence, achieving an order of magnitude higher sample efficiency.
*   **vs CONSMI / SimSon (Global Contrastive)**: They align [CLS] global embeddings; SIGMA performs dense alignment at the token level—every step of autoregressive decoding requires geometrically consistent latent states.
*   **vs SimCTG (Intra-sequence Contrastive)**: SimCTG focuses on the anisotropy of tokens within the same sequence; SIGMA focuses on structural equivalence **across sequences**; the two are orthogonal.
*   **vs LO-ARM / GraphAF (Graph Generators)**: Graph models have built-in permutation invariance but sacrifice Transformer scalability; SIGMA "simulates" the geometric properties of graph models within a sequence model, getting the best of both worlds.
*   **vs FineMolTex (Token-Motif Alignment)**: They require complex multi-modal architectures; SIGMA only requires the backbone + projection head, making it more lightweight.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ "Trajectory alignment" is a fresh and elegant path for injecting graph model geometric properties into sequence models.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers latent space analysis, property optimization, and structural diversity with complete ablations; however, it lacks coverage of alternative representations like SELFIES.
*   Writing Quality: ⭐⭐⭐⭐⭐ The "Manifold Fragmentation" concept is well-defined, illustrations are clear, and the methodology is rigorously derived.
*   Value: ⭐⭐⭐⭐⭐ Simultaneously improving training (alignment) and inference (IsoBeam) provides direct practical value to the chemical language model ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Protein Autoregressive Modeling via Multiscale Structure Generation](protein_autoregressive_modeling_via_multiscale_structure_generation.md)
- [\[AAAI 2026\] S2Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening](../../AAAI2026/computational_biology/s2drug_bridging_protein_sequence_and_3d_structure_in_contrastive_representation_.md)
- [\[AAAI 2026\] Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics](../../AAAI2026/computational_biology/dual-path_knowledge-augmented_contrastive_alignment_network_for_spatially_resolv.md)
- [\[ICML 2026\] Learning Protein Structure-Function Relationships through Knowledge-guided Representation Decomposition](learning_protein_structure-function_relationships_through_knowledge-guided_repre.md)
- [\[NeurIPS 2025\] Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations](../../NeurIPS2025/computational_biology/beyond_chemical_qa_evaluating_llms_chemical_reasoning_with_modular_chemical_oper.md)

</div>

<!-- RELATED:END -->
