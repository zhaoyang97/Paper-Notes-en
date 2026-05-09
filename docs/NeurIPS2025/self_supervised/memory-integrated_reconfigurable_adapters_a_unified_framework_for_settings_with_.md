---
title: >-
  [Paper Note] Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks
description: >-
  [NeurIPS 2025][Self-Supervised Learning][Associative Memory] MIRA embeds Hopfield-style associative memory modules into each layer of a ViT, storing and retrieving LoRA adapter weights as key-value pairs. Through a two-stage training procedure (adaptation + consolidation), it simultaneously addresses domain generalization (DG), class-incremental learning (CIL), and domain-incremental learning (DIL) within a single unified architecture, significantly outperforming task-specific methods across multiple benchmarks.
tags:
  - NeurIPS 2025
  - Self-Supervised Learning
  - Associative Memory
  - Hopfield Network
  - Adapters
  - Continual Learning
  - Domain Generalization
date: 2026-05-08
content_hash: 5635ad1a4f848490
---

# Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks

**Conference**: NeurIPS 2025
**arXiv**: [2512.00940](https://arxiv.org/abs/2512.00940)
**Code**: [https://snimm.github.io/mira_web/](https://snimm.github.io/mira_web/)
**Area**: Continual Learning / Domain Generalization
**Keywords**: Associative Memory, Hopfield Network, Adapters, Continual Learning, Domain Generalization

## TL;DR

MIRA embeds Hopfield-style associative memory modules into each layer of a ViT, storing and retrieving LoRA adapter weights as key-value pairs. Through a two-stage training procedure (adaptation + consolidation), it simultaneously addresses domain generalization (DG), class-incremental learning (CIL), and domain-incremental learning (DIL) within a single unified architecture, significantly outperforming task-specific methods across multiple benchmarks.

## Background & Motivation

**Background**: Domain generalization (DG), class-incremental learning (CIL), and domain-incremental learning (DIL) are three important yet independently developed research directions in deep learning. DG requires models to generalize to unseen domains, while continual learning (CL) requires models to retain prior knowledge as new tasks arrive. Existing methods typically design specialized architectures and strategies for individual scenarios.

**Limitations of Prior Work**: Biological systems can switch between multiple behavioral modes within milliseconds (e.g., bats adjusting echolocation from 20 Hz to 200 Hz, jazz pianists improvising), while retaining previously acquired knowledge without forgetting. This capability relies on the dynamic reuse of shared neural circuits via neuromodulatory signals (dopamine, acetylcholine, etc.). Current deep learning methods lack such a unified "rapid multi-task switching + persistent memory" mechanism.

**Key Challenge**: Although DG, CIL, and DIL appear distinct, they fundamentally all require efficient cross-task/cross-domain adaptation while preserving knowledge. Existing work treats them as entirely separate problems and has not leveraged biological associative memory (AM) to address them in a unified manner.

**Goal**: (1) Construct a unified architecture that simultaneously handles DG, CIL, and DIL; (2) leverage associative memory mechanisms to enable dynamic per-sample adapter composition and retrieval; (3) learn effective retrieval keys for indexing stored adapter weights.

**Key Insight**: Inspired by neuroscience, associative memory can store and retrieve task-specific weight modulation signals. If LoRA adapters are treated as "values" stored in associative memory and retrieved per sample via learnable "keys," rapid task switching analogous to neural modulation in the brain becomes achievable.

**Core Idea**: Embed Hopfield associative memory into each ViT layer, storing task-specific LoRA adapters as values, and realize per-sample affine combination retrieval of adapters via post-hoc learned retrieval keys, thereby unifying DG, CIL, and DIL.

## Method

### Overall Architecture

MIRA is built upon a frozen ViT-B/16 backbone (CLIP-initialized), with a Universal Hopfield Network (UHN) memory unit attached to each layer. As an input image passes through the backbone, the memory unit at each layer generates a query vector from the previous layer's output, retrieves a weighted combination of stored adapter weights, and loads it into the current layer. Training proceeds in two stages: **Adaptation** (training independent LoRA adapters per task/domain and writing them into memory) and **Consolidation** (optimizing only the retrieval keys and query modules so that the retrieved combination is optimal). Only a forward pass is required at inference time.

### Key Designs

1. **Associative Memory for Adapter Storage and Retrieval**:

    - **Function**: Stores LoRA adapters trained for each task/domain as key-value pairs, and retrieves and combines them on demand at inference time.
    - **Mechanism**: A memory unit $\mathcal{M}_\ell$ is attached to each ViT layer $\ell$. The write operation stores the trained adapter $\theta_\ell^{(t)}$; the read operation computes a weighted combination via the similarity between a query vector $q$ and all keys $\mathbf{K}$: $\hat{\theta}_\ell = \Theta_\ell \cdot \text{sep}(\text{sim}(K_\ell^\top, q))$. Critically, an **affine function** is used as the separation function (rather than Softmax), permitting negative weights to actively suppress interfering information rather than merely masking it.
    - **Design Motivation**: Storing weight adapters rather than raw data allows dynamic composition of multi-task parametric knowledge at inference time without gradient computation. The affine function outperforms Softmax in CIL and DG, as confirmed by ablation studies.

2. **Two-Stage Training (Adaptation + Consolidation)**:

    - **Function**: Decouples adapter training from retrieval optimization.
    - **Mechanism**: In the Adaptation stage, LoRA adapters (rank=4) are trained per task using cross-entropy loss, then written into memory with randomly Gaussian-initialized keys. In the Consolidation stage, adapter values are frozen; only the query module $g_\ell$ and keys $\mathbf{K}_\ell$ at each layer are trained to minimize cross-entropy over the retrieved combination. In DG, all domain data are consolidated jointly; in CL, consolidation proceeds sequentially per task.
    - **Design Motivation**: Decoupling the two stages enables independent optimization of the retrieval space. The Consolidation stage essentially solves the optimal adapter combination problem via AM inner-product approximation, which is formally proven in Lemma 1 of the paper.

3. **Learnable Query Module**:

    - **Function**: Aligns the previous layer's output from representation space to key space.
    - **Mechanism**: Each layer is equipped with a lightweight module $g_\ell: \mathbb{R}^{d_h} \to \mathbb{R}^{d_k}$, which may be an identity map, a linear transformation, or a small network. The query $q_\ell = g_\ell(h_{\ell-1})$ is combined with keys via an inner product passed through the separation function to obtain adapter combination weights. Keys and query modules are jointly optimized via backpropagation.
    - **Design Motivation**: Layer outputs and keys may reside in different representation spaces; the query module bridges this gap. Post-hoc learning (rather than fixed keys) enables the retrieval to be adaptively optimized.

### Loss & Training

Cross-entropy loss is used in both stages. Only LoRA parameters are updated during Adaptation; only keys and query modules are updated during Consolidation. In the CL setting, forgetting mitigation techniques such as DualGPM can be integrated within Consolidation. The additional parameters introduced by Hopfield keys account for less than 0.4% of total parameters (~276K / 86M), and the inference latency overhead is only ~0.4%.

## Key Experimental Results

### Main Results

| Dataset | Setting | Metric | MIRA | Prev. SOTA | Gain |
|--------|------|------|------|--------|------|
| iDigits | CIL | Avg Acc↑ | **83.00%** | 71.53% (ICON) | +11.47% |
| CORe50 | CIL | Avg Acc↑ | **83.39%** | 80.85% (ICON) | +2.54% |
| DomainNet | CIL | Avg Acc↑ | **67.29%** | 65.43% (ICON) | +1.86% |
| CORe50 | DIL | Avg Acc↑ | **93.89%** | 89.01% (ICON) | +4.88% |
| DomainNet | DIL | Avg Acc↑ | **69.18%** | 54.44% (ICON) | +14.74% |
| PACS | DG | Acc | **97.01%** | 96.50% (PEGO) | +0.51% |
| OfficeHome | DG | Acc | **87.36%** | 84.20% (PEGO) | +3.16% |
| DomainNet | DG | Acc | **61.19%** | 59.80% (CoOp) | +1.39% |
| DN4IL | DIL | Last Acc | **78.40%** | 44.45% (DUCA) | +33.95% |
| ImageNet-R | CIL-5 | ACC5 | **78.06%** | 75.85% (C-LoRA) | +2.21% |

### Ablation Study

| Separation Function | CIL Acc | DIL Acc | DG Acc | Average |
|----------|---------|---------|--------|------|
| Affine (default) | **67.29** | 69.18 | **61.19** | **65.89** |
| Softmax | 66.87 | **69.21** | 60.82 | 65.63 |
| ReLU | 66.60 | 69.20 | 60.90 | 65.57 |
| Tanh | 66.73 | 68.96 | 60.94 | 65.54 |

| Adapters per Task | CIL Acc | DIL Acc | DG Acc |
|----------------|---------|---------|--------|
| 1 | 63.75 | 69.08 | 61.21 |
| 5 | 67.21 | 69.10 | 61.01 |
| 10 | **67.29** | **69.18** | 61.19 |

### Key Findings

- The affine separation function performs best in CIL and DG because negative weights actively suppress interfering information; Softmax/ReLU can only mask but not remove such interference.
- Increasing the number of adapters from 1 to 5 yields a substantial gain (CIL: +3.46%), while the improvement from 5 to 10 is marginal, indicating that 5 adapters are sufficient to capture most task-specific variation.
- On DN4IL, MIRA achieves 78.40%, significantly outperforming DARE++ (44.11%) which uses a replay buffer of 200 samples, demonstrating that associative memory can serve as a substitute for replay buffers.
- Inference latency increases by only ~0.4% (0.0241s vs. 0.0240s), and additional parameters account for less than 0.4% of the total, making deployment overhead negligible.

## Highlights & Insights

- **A new paradigm for associative memory — storing weights rather than data**: Traditional AM stores data or features for replay; MIRA stores adapter weights for direct network modulation. Storage scales with the number of tasks rather than data volume, making the approach highly efficient. This represents a paradigm shift in how AM is applied in deep learning.
- **The elegance of post-hoc key learning**: Training adapters first and then learning retrieval keys is analogous to first "writing the encyclopedia" and then "building the index system." This two-step decoupling simplifies the optimization of each component independently.
- **A unified architecture with negligible overhead**: The same architecture handles three learning paradigms simply by varying the loss function and data provision strategy, with near-zero inference overhead. This design philosophy of "change the loss, not the architecture" is broadly transferable to other multi-scenario learning settings.
- **Theoretical guarantee (Lemma 1)**: The paper formally proves that AM retrieval can express the optimal adapter combination problem, providing a theoretical foundation for the proposed approach.

## Limitations & Future Work

- The current formulation uses only affine (linear) combination; nonlinear combinations (e.g., MoE gating) may yield further improvements, particularly for OOD extrapolation scenarios.
- Experiments are limited to classification tasks and ViT architectures; applicability to detection, segmentation, generation, and NLP tasks has not been verified.
- Each task requires training an independent set of adapters, leading to linear growth in storage as the number of tasks increases.
- The retrieval fidelity of Hopfield networks in high-dimensional adapter spaces is insufficiently analyzed.
- All ablation studies are conducted on DomainNet; whether the same trends hold on other datasets remains unverified.

## Related Work & Insights

- **vs. ICON**: ICON unifies CIL and DIL but does not support DG, relying on a dedicated prompt pool mechanism. MIRA offers a more natural unification via associative memory, with DG capability that ICON lacks.
- **vs. L2P/DualPrompt/CODA-P**: Prompt-based CL methods append learnable prompts to each layer but lack explicit storage-retrieval semantics between prompts. MIRA's Hopfield memory provides a well-defined indexing and lookup mechanism.
- **vs. PEGO**: A DG-specific method that marginally outperforms MIRA on VLCS but cannot handle CL scenarios. MIRA significantly surpasses PEGO on OfficeHome and DomainNet.
- **vs. LoRA/VeRA and other PEFT methods**: Standard PEFT addresses single-task adaptation without considering cross-task knowledge consolidation. MIRA augments PEFT with a memory indexing layer.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of storing weight adapters in associative memory with post-hoc key learning is novel and theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers three learning paradigms across 7+ datasets with detailed ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — The biological motivation is vividly presented, and the chain from theory to experiments is complete.
- **Value**: ⭐⭐⭐⭐ — Provides an elegant framework for unified multi-task learning; the AM + adapter paradigm is likely to inspire follow-up work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings](hybrid_autoencoders_for_tabular_data_leveraging_model-based_augmentation_in_low-.md)
- [\[NeurIPS 2025\] BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)
- [\[CVPR 2026\] UniGeoCLIP: Unified Geospatial Contrastive Learning](../../CVPR2026/self_supervised/unigeoclip_geospatial_contrastive.md)
- [\[AAAI 2026\] HiLoMix: Robust High- and Low-Frequency Graph Learning Framework for Mixing Address Association](../../AAAI2026/self_supervised/hilomix_robust_high-_and_low-frequency_graph_learning_framework_for_mixing_addre.md)
- [\[ICLR 2026\] Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning](../../ICLR2026/self_supervised/fly-cl_a_fly-inspired_framework_for_enhancing_efficient_decorrelation_and_reduce.md)

</div>

<!-- RELATED:END -->
