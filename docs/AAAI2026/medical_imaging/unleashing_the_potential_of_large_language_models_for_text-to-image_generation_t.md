---
title: >-
  [Paper Note] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment
description: >-
  [AAAI 2026][Medical Imaging][Autoregressive image generation] This paper proposes ARRA (Autoregressive Representation Alignment), a training framework that distills global visual representations from an external vision foundation model into the hidden states of an autoregressive LLM via a hybrid token \<HYBNEXT\>, significantly improving text-to-image generation quality without any architectural modification.
tags:
  - AAAI 2026
  - Medical Imaging
  - Autoregressive image generation
  - representation alignment
  - large language models
  - text-to-image
  - global consistency
date: 2026-05-08
content_hash: 8ad68f3e975374e7
---

# Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment

**Conference**: AAAI 2026
**arXiv**: [2503.07334](https://arxiv.org/abs/2503.07334)
**Code**: [https://github.com/HKU-HealthAI/ARRA](https://github.com/HKU-HealthAI/ARRA)
**Area**: Medical Imaging / Image Generation
**Keywords**: Autoregressive image generation, representation alignment, large language models, text-to-image, global consistency

## TL;DR

This paper proposes ARRA (Autoregressive Representation Alignment), a training framework that distills global visual representations from an external vision foundation model into the hidden states of an autoregressive LLM via a hybrid token \<HYBNEXT\>, significantly improving text-to-image generation quality without any architectural modification.

## Background & Motivation

1. **Background**: Large language models (LLMs) based on the autoregressive (AR) next-token prediction paradigm have achieved remarkable success on language tasks. Researchers have attempted to extend this paradigm to text-to-image (T2I) generation, as exemplified by DALL·E and LlamaGen.

2. **Limitations of Prior Work**: While next-token prediction aligns naturally with the local sequential structure of language, applying it to image generation forces the model to focus on isolated token-level features, neglecting the global consistency required for spatially structured visual content. This leads to fragmented artifacts in generated images (e.g., misaligned ribs in chest X-rays) and semantic inconsistencies.

3. **Key Challenge**: Existing solutions such as Transfusion and Show-O inject global constraints by modifying the architecture (bidirectional attention, grafted diffusion modules), which deviates from the standard LLM framework, limits compatibility with pretrained LLMs, and forfeits the benefits of established scaling laws.

4. **Goal**: Can the full potential of LLMs for T2I generation be unleashed without altering the original architecture or inference mechanism?

5. **Key Insight**: Global consistency does not require architectural complexity; it can instead be achieved by redefining the training objective.

6. **Core Idea**: A hybrid token \<HYBNEXT\> is designed so that each token is jointly constrained during training by a local autoregressive loss and a global visual alignment loss, distilling semantic information from an external foundation model into the autoregressive sequence.

## Method

### Overall Architecture

ARRA augments the standard autoregressive training pipeline with a Global Visual Alignment (GVA) loss branch: (1) text and images are tokenized into token sequences and fed into a transformer for next-token prediction; (2) concurrently, an external pretrained visual encoder (e.g., BioMedCLIP) extracts global visual representations of the target image; (3) the hidden state of each \<HYBNEXT\> token in the autoregressive model is aligned with the global representation via a projection layer. This alignment is applied only during training and is entirely removed at inference.

### Key Designs

1. **Hybrid Token \<HYBNEXT\>**:
    - **Function**: Serves as a bidirectional anchor between local and global learning.
    - **Mechanism**: Each next token to be predicted in the autoregressive sequence is defined as \<HYBNEXT\>, subject to dual constraints — locally, standard next-token prediction via codebook matching; globally, alignment of its hidden state with compressed visual features from the external model. This ensures every token is effectively constrained by the external global representation.
    - **Design Motivation**: Compared to placing a fixed \<REP\> token at the beginning of the sequence, \<HYBNEXT\> traverses every token during training, avoiding the attention decay ("attention sink" effect) in long sequences that would render the constraint ineffective.

2. **Global Visual Alignment (GVA Loss)**:
    - **Function**: Distills rich semantic knowledge (spatial relationships, object consistency, etc.) from the external foundation model into the autoregressive model.
    - **Mechanism**: A pretrained encoder $\mathcal{E}_F$ extracts a global image representation $f_{GF} = \text{agg}(\mathcal{E}_F(I)) \in \mathbb{R}^{1\times D}$; a 2-layer MLP projection maps the hidden state $f_L^i$ of \<HYBNEXT\> to $f_A$; the cosine similarity loss $\mathcal{L}_{GVA} = \text{sim}(f_A, f_{GF})$ is minimized.
    - **Design Motivation**: Leverages knowledge from well-trained vision foundation models to provide global structural priors at no additional architectural cost.

3. **Three Flexible Variants**:
    - **Function**: A plug-and-play framework adaptable to different scenarios.
    - **Mechanism**: ARRA-Base (training from scratch), ARRA (extending a text-only LLM to T2I), and ARRA-Adapt (adapting a general-purpose generative model to a specific domain such as medical imaging).
    - **Design Motivation**: Demonstrates the generality of ARRA as a training framework that is agnostic to the specific LLM architecture.

### Loss & Training

The total loss function is:
$$\mathcal{L}_{ARRA}(\theta, \phi) = \mathcal{L}_{AR}(\theta) + \lambda \mathcal{L}_{GVA}(\theta, \phi)$$

- $\mathcal{L}_{AR}$: Standard autoregressive cross-entropy loss.
- $\mathcal{L}_{GVA}$: Global visual alignment cosine similarity loss.
- $\lambda = 1$, balancing the two objectives.

Key design decisions:
- **Aggregation strategy**: [CLS] token outperforms average pooling (more compact global information).
- **Encoder selection**: Cross-modal encoders (BioMedCLIP) are preferred for LLMs lacking image generation capability; domain-specific encoders are more effective for LLMs with existing generation capability undergoing domain adaptation.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Baseline | Gain |
|---------|--------|------|----------|------|
| ImageNet (111M) | FID ↓ | 5.08 | 5.47 (LlamaGen) | −7.1% |
| ImageNet (343M) | FID ↓ | 3.61 | 4.33 (LlamaGen) | −16.6% |
| LAION-COCO (3.1B) | FID ↓ | 10.45 | 11.88 (LlamaGen) | −12.0% |
| MIMIC-CXR | FID ↓ | 5.30 | 7.11 (Chameleon) | −25.5% |
| MIMIC-CXR (Adapt) | FID ↓ | 4.15 | 5.10 (direct fine-tuning) | −18.6% |
| DeepEyeNet | FID ↓ | 35.01 | 38.37 (Chameleon) | −8.8% |

### Ablation Study

| Configuration | FID (MIMIC-CXR) | Notes |
|---------------|-----------------|-------|
| w/o alignment | 5.10 | No-alignment baseline (ARRA-Adapt) |
| \<REP\> fixed token | 4.85 | Alignment at sequence start |
| \<HYBNEXT\> hybrid token | 4.15 | Per-token alignment, best |
| [CLS] aggregation | 5.30 | Best global feature aggregation (ARRA) |
| Average pooling | 6.56 | Mixed local features |
| BioMedCLIP encoder | 4.15 | Domain-specific cross-modal, best |
| CLIP-L general encoder | 4.63 | General cross-modal |
| Med-SAM visual encoder | 4.08 | Strong structural prior, weaker semantics |

### Key Findings

- ARRA exhibits consistent performance improvement as model scale increases from 343M to 3.1B (FID: 11.67 → 10.45), preserving favorable scaling properties.
- \<HYBNEXT\> consistently outperforms fixed-position alignment \<REP\> across all settings, confirming the necessity of per-token global constraints.
- For LLMs lacking image generation capability, cross-modal encoders (BioMedCLIP/CLIP) are critical; for LLMs with existing generation capability, domain-specific encoders are more effective.
- ARRA yields stable improvements across different resolutions (256×256 and 512×512).

## Highlights & Insights

- **Remarkably simple solution**: The global cross-modal consistency problem is addressed solely by redefining the training objective — no architectural modification is required and inference efficiency is unaffected.
- **Three core takeaways**: (1) Per-token alignment is superior to fixed-position alignment; (2) the [CLS] token is the optimal global aggregation strategy; (3) encoder selection should be matched to the capability of the LLM.
- **Plug-and-play**: The same framework supports training from scratch, extending text-only LLMs, and domain adaptation, demonstrating strong generality.
- The medical imaging experiments are particularly compelling — improvements in fine-grained details such as rib alignment in chest X-ray generation are visually evident.

## Limitations & Future Work

- Training requires additional forward passes through the external encoder, increasing training cost.
- The current approach aligns only the global [CLS] representation; finer-grained patch-level alignment strategies remain unexplored due to compatibility challenges with AR models.
- Experiments are primarily conducted on medical imaging datasets; validation on general natural image benchmarks is relatively limited in scale.
- The alignment weight $\lambda$ has not been subjected to extensive hyperparameter search.

## Related Work & Insights

- **vs. Transfusion/Show-O**: These methods require architectural modifications (bidirectional attention / masked token modeling) and are incompatible with standard LLMs; ARRA leaves the architecture intact.
- **vs. REPA (diffusion model alignment)**: REPA employs patch-wise alignment, which is incompatible with AR models (since AR training does not output all patch tokens simultaneously); ARRA bridges the alignment mechanism and the AR architecture via \<HYBNEXT\>.
- **vs. LlamaGen (pure autoregressive)**: Constrained only by local objectives, lacking global structural consistency; ARRA injects global knowledge without any architectural change.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of redesigning the training objective is novel and the \<HYBNEXT\> design is elegant; representation distillation itself is not an entirely new concept.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers natural images and medical imaging, multi-scale models, three variants, and systematic component analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is well-argued, experimental analysis yields practical takeaways, and the structure is clear.
- **Value**: ⭐⭐⭐⭐ The plug-and-play framework provides practical value to the AR T2I community, with strong application prospects in medical imaging.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](../../CVPR2026/medical_imaging/unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)
- [\[AAAI 2026\] Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models](coarse-to-fine_open-set_graph_node_classification_with_large_language_models.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/medical_imaging/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[AAAI 2026\] CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records](clicare_grounding_large_language_models_in_clinical_guidelines_for_decision_supp.md)
- [\[AAAI 2026\] Measuring Stability Beyond Accuracy in Small Open-Source Medical Large Language Models for Pediatric Endocrinology](measuring_stability_beyond_accuracy_in_small_open-source_medical_large_language_.md)

<!-- RELATED:END -->
