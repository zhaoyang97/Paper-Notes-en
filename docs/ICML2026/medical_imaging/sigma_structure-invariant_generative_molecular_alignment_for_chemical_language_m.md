---
title: >-
  [Paper Note] SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning
description: >-
  [ICML 2026][Medical Imaging][SMILES] SIGMA enforces the alignment of hidden states for different SMILES permutations of the same molecule onto a unified trajectory using token-level contrastive loss…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "SMILES"
  - "Contrastive Learning"
  - "Trajectory Alignment"
  - "Isomorphic Beam Search"
  - "Molecular Generation"
date: 2026-05-08
content_hash: 491dd385b7a7a3f4
---

# SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning

**Conference**: ICML 2026  
**arXiv**: [2603.25062](https://arxiv.org/abs/2603.25062)  
**Code**: None  
**Area**: Graph Learning / Chemical Language Models / Autoregressive Generation  
**Keywords**: SMILES, Contrastive Learning, Trajectory Alignment, Isomorphic Beam Search, Molecular Generation

## TL;DR
SIGMA enforces the alignment of hidden states for different SMILES permutations of the same molecule onto a unified trajectory using token-level contrastive loss, and introduces IsoBeam to prune isomorphic redundant paths during decoding, enabling sequence models to "think in chemical space by structure, not by string."

## Background & Motivation

**Background**: Current chemical language models (ChemLM) serialize molecular graphs into SMILES strings and use Transformers for autoregressive generation. This "language-style modeling" leverages large-scale unlabeled corpora (PubChem/ChEMBL/ZINC) for pretraining and is widely applied in de novo drug design, property prediction, and activity modeling.

**Limitations of Prior Work**: A single molecular graph corresponds to a factorial number of valid SMILES representations (depending on traversal order), yet models treat these equivalent forms as entirely different sequences. As a result, different prefixes of the same molecule are mapped to orthogonal positions in hidden space, termed "Trajectory Divergence" by the authors, leading to "Manifold Fragmentation"—the chemical space is fragmented into islands based on syntax rather than structure. This is particularly detrimental for reinforcement learning-driven molecular optimization: agents may get trapped in a syntactic region, repeatedly sampling the same scaffold, causing mode collapse.

**Key Challenge**: Graph models (MPNN/GraphAF) are inherently permutation-invariant but sacrifice the scalability of Transformers; sequence models are scalable but lack geometric inductive bias. Existing Randomized SMILES data augmentation only passively exposes equivalence, with models often memorizing frequent permutations rather than learning structural equivalence. A method is needed that preserves sequence efficiency while enforcing geometric invariance.

**Goal**: (1) Without abandoning SMILES representation, explicitly align structurally equivalent prefixes to the same hidden state during training; (2) Eliminate redundant decoding paths mapping to the same molecule during inference; (3) Maintain compatibility with existing Transformer training pipelines without introducing extra encoders.

**Key Insight**: The authors observe that if two different SMILES prefixes can be concatenated with the exact same suffix to yield the same molecule, they correspond to the same intermediate subgraph in chemical terms. This provides a strict criterion for "Functional Equivalence," avoiding pseudo-positive samples that are syntactically similar but chemically incompatible.

**Core Idea**: Use token-level contrastive loss to align prefixes sharing the same suffix onto the same latent trajectory, while pushing apart chemically different prefixes as structural negatives, so that the autoregressive model "behaves like a graph model" in latent space.

## Method

### Overall Architecture
SIGMA consists of three core modules: (1) **Functional Equivalence View Construction**—sample two different traversals from the same molecular graph to obtain positive pairs $(p_u, p_v)$, and use an InChIKey hash oracle $\mathcal{H}$ to verify $\mathcal{H}(\text{Mol}(p_u \oplus s)) \equiv \mathcal{H}(\text{Mol}(p_v \oplus s))$; (2) **Decoupled Projection Head Contrastive Learning**—add a nonlinear projection head $g_\phi$ on top of the Transformer backbone to map hidden states $\mathbf{h}_t$ to the contrastive metric space $\mathbf{z}_t$, preventing the contrastive objective from competing with the MLE task for syntactic details; (3) **IsoBeam Inference**—dynamically detect whether prefixes correspond to the same subgraph during beam search, pruning low-probability isomorphic paths and reallocating budget to structurally distinct branches. The training objective is MLE loss plus token-level "suffix-align + prefix-repel" contrastive loss; IsoBeam is used during inference.

### Key Designs

1. **Functional Equivalence Views & Probe Suffix Protocol**:

    - **Function**: Construct strict positive pairs—syntactically different but structurally equivalent prefixes.
    - **Mechanism**: Randomize two traversals $S^u, S^v$ from the original SMILES, find a common split point to decompose into $(p, s)$, requiring $p_u \neq p_v$ (syntactic divergence) but $\mathcal{H}(\text{Mol}(p_u \oplus s)) \equiv \mathcal{H}(\text{Mol}(p_v \oplus s)) \equiv \mathcal{H}(\mathcal{G})$ (structural equivalence). Since incomplete SMILES prefixes are often chemically invalid (e.g., open rings), the authors introduce the **Probe Suffix Protocol**: if the split point creates dangling bonds during training, a stable cap fragment $s_{probe}$ (e.g., methyl or ring closure) is temporarily appended for structural validation. **Structural Negatives** are also introduced: explicitly select negative prefixes from the batch where $\mathcal{H}(\text{Mol}(p_{neg} \oplus s)) \neq \mathcal{H}(\mathcal{G})$ (e.g., stereoisomers or scaffold hops), forcing the model to distinguish "true isomorphism" from "seemingly similar but essentially different."
    - **Design Motivation**: Avoid syntactic false positives from random augmentation, ensuring the contrastive signal strictly reflects topological equivalence rather than string similarity.

2. **Decoupled Projection Head + Dense Trajectory Alignment**:

    - **Function**: Pull together equivalent prefixes and push apart negative prefixes in latent space, without harming the MLE task's reliance on syntactic details.
    - **Mechanism**: Applying contrastive loss directly on backbone hidden states $\mathbf{H}$ conflicts with MLE—MLE requires precise distinction of syntactic features like "ring index 1 vs 2," while contrastive learning seeks to erase such differences. The authors introduce a two-layer MLP projection head $\mathbf{z}_t = W^{(2)} \sigma(W^{(1)} \mathbf{h}_t + b^{(1)}) + b^{(2)}$, moving the contrastive loss to the projection space $\mathcal{Z}$. The contrastive objective is applied to **matched suffix token positions** (suffix-align, positive signal), and **unmatched prefix positions** for prefix-repel (negative signal), covering both token output distribution alignment and cross-attention weight alignment, forming "dense trajectory alignment."
    - **Design Motivation**: Global [CLS] alignment as in SimCLR/MoCo is insufficient for autoregressive generation—each token decision requires geometrically consistent hidden states. Token-level alignment is necessary to guide stepwise decoding.

3. **IsoBeam: Structure-Aware Beam Search**:

    - **Function**: Dynamically remove redundant paths during decoding where different SMILES strings correspond to the same molecule.
    - **Mechanism**: Standard beam search often yields multiple top-k paths that decode to different SMILES representations of the same molecule (e.g., various SMILES for acetophenone), wasting search budget. IsoBeam performs a **Partial Graph Check** at each step for some prefixes in the current beam: if two prefixes correspond to isomorphic subgraphs with identical open connection points, only the higher-probability path is retained, reallocating budget to other scaffolds (e.g., switching from benzene to pyridine scaffolds).
    - **Design Motivation**: Trajectory invariance learned during training should also be recognized and utilized during inference—otherwise, even if the model learns equivalence, outputs may still appear "structurally diverse" due to redundancy.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{\text{MLE}} + \lambda \mathcal{L}_{\text{contrast}}$, where the contrastive term includes suffix-align (InfoNCE-style positive alignment) and prefix-repel (pushing apart structural negatives), with a temperature parameter $\tau$ controlling sharpness. Each batch samples randomized SMILES pairs online and verifies hash equivalence, discarding failed pairs. The projection head and backbone are trained jointly.

## Key Experimental Results

### Main Results

The paper benchmarks against strong baselines (standard ChemLM, Randomized SMILES augmentation, CONSMI global contrast, SimCTG self-contrast, LO-ARM graph generator) on standard multi-parameter molecular optimization (MPO) tasks, evaluating sample efficiency, structural diversity, and property optimization score.

| Task Type | Metric | SIGMA | Prev. SOTA | Gain |
|-----------|--------|-------|------------|------|
| Multi-parameter optimization | Number of high-scoring molecules | Significantly better | Randomized SMILES | Large improvement in sample efficiency |
| Structural diversity | Number of unique scaffolds | Significantly better | Standard beam search | IsoBeam reallocates budget to different scaffolds |
| Latent space alignment | Cosine similarity of isomorphic prefixes | Near 1 | < 0.5 | Validates that Manifold Fragmentation is fixed |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Full SIGMA (suffix-align + prefix-repel + IsoBeam) | Optimal | — |
| w/o Structural Negatives | Structural diversity drops | In-batch random negatives insufficient for stereoisomer distinction |
| w/o Projection Head | MLE perplexity increases | Direct contrast on backbone harms generation quality |
| w/o IsoBeam (train as SIGMA, infer with standard beam) | Unique scaffold count drops | Training alignment alone can't fully eliminate inference redundancy |
| w/o suffix-align (global CLS contrast only) | Latent alignment weakens | Validates necessity of token-level dense alignment |

### Key Findings
- The projection head is essential: direct contrast on the backbone causes a tug-of-war between MLE and contrastive objectives, reducing token prediction accuracy.
- IsoBeam and training alignment are complementary: alignment ensures the model knows equivalence, IsoBeam ensures outputs avoid redundancy.
- Structural negatives (stereoisomers/scaffold hops) significantly improve fine-grained discrimination; ordinary in-batch negatives are insufficient.
- Probe Suffix ensures equivalence determination is based on stable topology rather than transient states, avoiding pseudo-judgments from chemically invalid intermediates.

## Highlights & Insights
- **Geometric Consistency = Latent Space Constraint**: Token-level contrastive learning explicitly encodes "graph symmetry that sequence models should respect" into the latent space, elegantly injecting "graph model inductive bias" into Transformers.
- **Training-Inference Duality**: suffix-align (learning equivalence during training) + IsoBeam (using equivalence during inference) forms a complete closed loop, avoiding the common pitfall of "learned in training but unused in inference."
- **InChIKey hash oracle** as an equivalence criterion is clean and strict: avoids heuristic false positives from edit distance or substring matching.
- The "trajectory alignment" approach is transferable to any "one-to-many serialization" problem: e.g., different expression orderings of equivalent ASTs in code generation, different point cloud orderings for 3D shapes, or alternative representations of chemical reaction paths.

## Limitations & Future Work
- Hash validation relies on cheminformatics tools like RDKit, which may fail for large or unconventional molecules, and incurs extra overhead per batch due to hash computation.
- Probe Suffix selection (methyl cap vs ring closure) affects the boundary of equivalence determination—different probes may yield different conclusions in extreme cases.
- IsoBeam's Partial Graph Check has inherent computational complexity and may become a bottleneck for large beams or long sequences, requiring engineering optimization.
- The paper focuses on SMILES; whether the approach is equally effective for more robust linear representations like SELFIES/DeepSMILES needs further validation.
- Future extensions could include reaction SMILES, 3D conformations, and multimodal molecular representations.

## Related Work & Insights
- **vs Randomized SMILES (Bjerrum 2017)**: They rely on data augmentation to **passively** expose equivalent permutations, while SIGMA uses contrastive loss to **actively** enforce equivalence, achieving an order-of-magnitude higher sample efficiency.
- **vs CONSMI / SimSon (global contrast)**: They align global [CLS] embeddings, while SIGMA performs dense token-level alignment—autoregressive decoding requires geometrically consistent hidden states at every step.
- **vs SimCTG (intra-sequence contrast)**: SimCTG focuses on anisotropy within a single sequence, while SIGMA targets **cross-sequence** structural equivalence; the two are orthogonal.
- **vs LO-ARM / GraphAF (graph generators)**: Graph models have built-in permutation invariance but sacrifice Transformer scalability; SIGMA "simulates" graph model geometric properties within sequence models, achieving the best of both worlds.
- **vs FineMolTex (token-motif alignment)**: They require complex multimodal architectures, while SIGMA only needs a backbone plus projection head, making it more lightweight.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Trajectory alignment" is a novel and elegant way to inject graph geometric properties into sequence models
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers latent space analysis, property optimization, structural diversity, and complete ablations; unfortunately does not cover alternative representations like SELFIES
- Writing Quality: ⭐⭐⭐⭐⭐ Manifold Fragmentation is well-articulated, illustrations are clear, and the method section is rigorously derived
- Value: ⭐⭐⭐⭐⭐ Simultaneously improves training (alignment) and inference (IsoBeam), with direct practical value for the chemical language model ecosystem

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MolLangBench: A Comprehensive Benchmark for Language-Prompted Molecular Structure Recognition, Editing, and Generation](../../ICLR2026/graph_learning/mollangbench_a_comprehensive_benchmark_for_language-prompted_molecular_structure.md)
- [\[ICLR 2026\] Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](../../ICLR2026/graph_learning/entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)
- [\[ICML 2026\] Structure-Centric Graph Foundation Model via Geometric Bases](structure-centric_graph_foundation_model_via_geometric_bases.md)
- [\[NeurIPS 2025\] Learning Repetition-Invariant Representations for Polymer Informatics](../../NeurIPS2025/graph_learning/learning_repetition-invariant_representations_for_polymer_informatics.md)
- [\[ICML 2025\] From RAG to Memory: Non-Parametric Continual Learning for Large Language Models](../../ICML2025/graph_learning/from_rag_to_memory_non-parametric_continual_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
