---
title: >-
  [Paper Note] PriVi: Towards a General-Purpose Video Model for Primate Behavior in the Wild
description: >-
  [CVPR 2026][Model Compression][primate behavior recognition] PriVi constructs a large-scale primate video pretraining dataset of 424 hours and performs **domain-level pretraining** (rather than dataset-level pretraining)…
tags:
  - "CVPR 2026"
  - "Model Compression"
  - "primate behavior recognition"
  - "self-supervised pretraining"
  - "V-JEPA"
  - "domain-level pretraining"
  - "data curation pipeline"
date: 2026-05-08
content_hash: 955e257d4e8285a6
---

# PriVi: Towards a General-Purpose Video Model for Primate Behavior in the Wild

**Conference**: CVPR 2026
**arXiv**: [2511.09675](https://arxiv.org/abs/2511.09675)
**Code**: [https://privi.eckerlab.org](https://privi.eckerlab.org) (data + models + code)
**Area**: Model Compression
**Keywords**: primate behavior recognition, self-supervised pretraining, V-JEPA, domain-level pretraining, data curation pipeline

## TL;DR
PriVi constructs a large-scale primate video pretraining dataset of 424 hours and performs **domain-level pretraining** (rather than dataset-level pretraining) on V-JEPA. This work is the first to demonstrate that domain-level pretraining of video models generalizes across datasets, surpassing fully fine-tuned specialized models on four primate behavior recognition benchmarks using a frozen classifier with only 220K parameters.

## Background & Motivation

1. **Background**: Primate behavior analysis is critical for cognitive science, evolutionary biology, and conservation ecology. Computer vision methods hold promise for assisting behavioral analysis, but existing approaches primarily rely on human-centric pretrained models (e.g., Kinetics) and train specialized models on individual datasets, resulting in limited generalization.

2. **Limitations of Prior Work**: (a) Pretraining data is human-centric, making it out-of-domain for non-human animals such as primates; (b) existing methods typically train separate models for each target dataset (dataset-level pretraining), precluding cross-dataset knowledge sharing; (c) annotations are scarce due to the high cost and expertise required for labeling, necessitating methods that work with few labels.

3. **Key Challenge**: Language models have demonstrated that domain-level pretraining—pretraining on data similar but not identical to the target—improves downstream performance, yet no analogous results exist for video models. The core challenge is the lack of large-scale, diverse primate video datasets and scalable curation methods.

4. **Goal**: (a) How to curate a large-scale primate video dataset without seed datasets or text annotations; (b) whether domain-level pretraining is effective for video models; (c) how to design a lightweight classifier for efficient behavior recognition on frozen features.

5. **Key Insight**: The work shifts from a model-centric to a data-centric perspective. The core hypothesis is that pretraining on sufficiently diverse in-domain data provides cross-dataset general representations that are more effective than separately pretraining on each small dataset.

6. **Core Idea**: A scalable data curation pipeline is used to construct PriVi, a large primate video dataset. Domain-level pretraining via V-JEPA yields general primate representations, achieving state-of-the-art performance on four behavior benchmarks under frozen evaluation.

## Method

### Overall Architecture
The method consists of three stages: (A) **domain-level pretraining**—continuing training of V-JEPA on the PriVi dataset; (B) optional **dataset-level pretraining**—further unsupervised pretraining on the target dataset; (C) **frozen evaluation**—freezing the encoder and training only a lightweight attentive classifier for behavior recognition.

### Key Designs

1. **PriVi Dataset Curation Pipeline**:

    - **Function**: Automatically curates 424 hours of high-quality primate video from YouTube and research recordings.
    - **Mechanism**: Data comes from two sources—**R&O (research data)**, comprising 174 hours from 11 distinct behavioral ecology research settings; and **YouTube data**, collected by searching primate-related playlists, yielding 458 hours of raw video. The curation pipeline includes: (1) **shot-cut detection** to remove segments with frequent jump cuts; (2) **relevance filtering**—a 2-layer MLP classifier trained on CLIP embeddings (recall 82.8%, precision 90.3%) to filter relevant clips; (3) **detection filtering**—zero-shot primate detection with GroundingDINO, discarding clips with no detections; (4) **subsampling**—adjusting per-dataset contribution within R&O proportionally. The final dataset contains 720K clips of 3 seconds each.
    - **Design Motivation**: Unlike existing pipelines that rely on high-quality seed datasets or text annotations, this pipeline requires neither, needing only a small amount of manual annotation (2,500 images) to train the relevance classifier, making it highly scalable.

2. **V-JEPA Domain-Level Pretraining**:

    - **Function**: Learns primate-specific video representations.
    - **Mechanism**: Starting from the pretrained V-JEPA ViT-L model (trained on VideoMix2M), training continues on PriVi for 75K steps (~8 epochs) with batch size 80 and a constant learning rate of $1.5 \times 10^{-5}$ (since the original weights already underwent cosine annealing). During training, random crops are applied with centers aligned to bounding boxes predicted by a zero-shot primate detector, directing model attention toward individual primates rather than background.
    - **Design Motivation**: Primate-centric cropping prevents the model from expending capacity on background representations. Total training time is approximately 11 hours on 4 A100 GPUs, representing minimal computational overhead.

3. **Attentive Classifier**:

    - **Function**: Lightweight behavior recognition on frozen encoder features.
    - **Mechanism**: The $N$ patch tokens output by the encoder are first projected from $D=1024$ to $D'=64$, then concatenated with $C$ learnable CLS tokens (one per class). These are processed by 3 layers of self-attention, after which each CLS token is passed through a linear projection followed by softmax/sigmoid to yield class probabilities. Total parameter count is only 220K.
    - **Design Motivation**: Existing frozen-evaluation classifiers for video models (12M parameters for V-JEPA, 49M for V-JEPA2) are over-parameterized for small animal behavior datasets. Projecting to 64 dimensions and assigning an independent CLS token per class avoids information bottlenecks, preventing overfitting with minimal parameters.

### Loss & Training

Pretraining uses V-JEPA's masked prediction loss $L_{JEPA} = \|P(E(\text{Mask}(X))) - \text{Mask}^C(\bar{E}(X))\|_1$. The classifier is trained with cross-entropy loss (EQL loss for BaboonLand). Classifier training runs for 1–40 epochs depending on the dataset.

## Key Experimental Results

### Main Results

| Method | ChimpACT mAP | PanAf500 B-Acc | BaboonLand B-Acc | ChimpBehave B-Acc |
|------|-------------|---------------|-----------------|------------------|
| X3D | 27.05 | 50.35 | 31.41 | 62.8 |
| VideoMAEv2 | - | - | - | 74.8 |
| VideoPrism-g | 31.5 | - | - | - |
| V-JEPA (human data) | 36.33 | 56.69 | 26.99 | 68.41 |
| **PriVi** | **39.25** | **62.75** | **33.99** | **71.30** |
| **PriVi + DaLP** | **40.00** | **62.96** | **38.57** | **75.14** |

PriVi outperforms all existing methods on all four datasets, including fully fine-tuned specialized models (ChimpVLM with 167M parameters).

### Ablation Study

| Configuration | ChimpACT mAP | PanAf500 B-Acc |
|------|-------------|---------------|
| V-JEPA baseline | 32.00 | 71.95 |
| + DaLP: ChimpACT | 35.86 | - |
| YT-Random (unfiltered) | 33.87 | 71.61 |
| YT-Filtered (filtered) | 37.88 | 76.33 |
| R&O only | 33.01 | 73.85 |
| **PriVi (YT-F + R&O)** | **38.75** | **79.95** |
| w/o primate-centric crop | 32.62 | 72.93 |
| w/o dimensionality reduction (37.84M params) | 30.15 | 62.52 |

### Key Findings
- **YouTube relevance filtering is critical**: YT-Filtered substantially outperforms YT-Random, validating the importance of data curation.
- **Domain-level pretraining > dataset-level pretraining**: PriVi outperforms dataset-level pretraining on both datasets without cross-dataset negative transfer.
- **Primate-centric cropping matters**: Removing it causes a 6.13-point drop in ChimpACT mAP.
- **Dimensionality reduction helps**: The 37.84M-parameter model without dimensionality reduction achieves the worst performance (30.15 mAP), confirming that fewer parameters are preferable on small datasets.
- **Strong performance with very few labels**: With only 10% of training data, PriVi loses only 4.4% accuracy (PanAf500) while still outperforming X3D trained on the full dataset.
- Both data sources (YT-Filtered and R&O) contribute independently, with further gains when combined.

## Highlights & Insights
- **First demonstration that domain-level pretraining is effective for video models**: Previously validated only for language models. This implies that similar pretraining datasets can be constructed for other specialized domains (marine biology, agriculture, etc.).
- **Extremely lightweight frozen classifier**: With only 220K parameters, it surpasses a fully fine-tuned 167M-parameter model, demonstrating that strong representations matter more than large classification heads.
- **Practical data curation pipeline**: Requires no seed datasets or text annotations—only 2,500 annotated images—and is readily transferable to other animals or domains.
- **Projection to 64 dimensions**: Counterintuitive yet effective—on small datasets, reducing classifier parameters is key to preventing overfitting.

## Limitations & Future Work
- PriVi covers only 11 research settings with limited primate diversity (primarily macaques and chimpanzees).
- Evaluation is limited to chimpanzee and baboon datasets; generalization to other primates is unverified.
- Three-second clips may be insufficient for long-horizon behaviors such as social interaction sequences.
- Vision-language models (e.g., fine-tuned CLIP) are not explored as alternatives.
- 14% of R&O data is not publicly released due to privacy constraints.

## Related Work & Insights
- **vs. VideoPrism**: VideoPrism-g achieves only 31.5 mAP on ChimpACT, while PriVi reaches 40.0, indicating that domain-specific data is more effective than large-scale general-purpose data.
- **vs. ChimpVLM**: A fully fine-tuned VLM with 167M parameters achieves 61.94 B-Acc on PanAf500; PriVi reaches 62.96 with a 220K frozen classifier, demonstrating that high-quality pretraining data outweighs large-model fine-tuning.
- **vs. AlphaChimp**: On ChimpACT without ground-truth detections, SAM3 + PriVi (30.76 mAP) outperforms AlphaChimp (25.35), highlighting the practical value of combining zero-shot detection with frozen features.

## Rating
- Novelty: ⭐⭐⭐⭐ First validation of domain-level pretraining for video models; practical data curation pipeline design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets, classifier ablations, data ablations, low-label experiments, and cross-shot generalization—comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and thorough experiments, though some notation and formulas could be more concise.
- Value: ⭐⭐⭐⭐ Significant impact on animal behavior analysis; methodology is generalizable to other specialized visual domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Generative Video Compression with One-Dimensional Latent Representation](generative_video_compression_with_one-dimensional_latent_representation.md)
- [\[CVPR 2026\] UniComp: Rethinking Video Compression Through Informational Uniqueness](unicomp_rethinking_video_compression_through_informational_uniqueness.md)
- [\[CVPR 2026\] F²HDR: Two-Stage HDR Video Reconstruction via Flow Adapter and Physical Motion Modeling](textf2texthdr_two-stage_hdr_video_reconstruction_via_flow_adapter_and_physical_m.md)
- [\[ICLR 2026\] FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning](../../ICLR2026/model_compression/flyprompt_brain-inspired_random-expanded_routing.md)
- [\[ICLR 2026\] SFT Doesn't Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs](../../ICLR2026/model_compression/sft_doesnt_always_hurt_general_capabilities_revisiting_domain-specific_fine-tuni.md)

</div>

<!-- RELATED:END -->
