---
title: >-
  [Paper Note] BUSSARD: Normalizing Flows for Bijective Universal Scene-Specific Anomalous Relationship Detection
description: >-
  [CVPR2026][Multimodal VLM][Scene graph anomaly detection] This paper proposes BUSSARD, the first learning-based scene-specific anomalous relationship detection method. It encodes scene graph triplets via pretrained langu…
tags:
  - "CVPR2026"
  - "Multimodal VLM"
  - "Scene graph anomaly detection"
  - "normalizing flows"
  - "semantic embeddings"
  - "relationship anomaly"
  - "multimodal"
date: 2026-05-08
content_hash: fde1e802f00ea203
---

# BUSSARD: Normalizing Flows for Bijective Universal Scene-Specific Anomalous Relationship Detection

**Conference**: CVPR2026  
**arXiv**: [2603.16645](https://arxiv.org/abs/2603.16645)  
**Code**: [github.com/mschween/BUSSARD](https://github.com/mschween/BUSSARD)  
**Area**: Multimodal VLM  
**Keywords**: Scene graph anomaly detection, normalizing flows, semantic embeddings, relationship anomaly, multimodal

## TL;DR
This paper proposes BUSSARD, the first learning-based scene-specific anomalous relationship detection method. It encodes scene graph triplets via pretrained language model embeddings, applies an autoencoder for dimensionality reduction, and employs normalizing flows for likelihood estimation. BUSSARD achieves approximately 10% AUROC improvement on the SARD dataset and demonstrates robustness to synonym variation.

## Background & Motivation
1. Image anomaly detection encompasses not only industrial defects but also scene context understanding—such as objects appearing in inappropriate locations or abnormal human–object relationships.
2. Existing methods focus predominantly on individual components such as human pose, neglecting broader contextual information and object relationships.
3. The SARD task and dataset address relational anomaly detection in scene graphs (e.g., "plate on chair"), yet existing approaches are counting-based and lack learning capability.
4. Counting-based methods are severely affected by long-tail distributions—a small number of high-frequency triplets dominate, causing many normal but infrequent triplets to be incorrectly flagged as anomalous.
5. Counting-based methods are not robust to lexical variation (synonyms)—"person" and "human" are treated as entirely distinct entities.
6. Learning-based methods capable of leveraging semantic knowledge to generalize to rare or unseen vocabulary are needed.

## Method

### Overall Architecture (4-Step Pipeline)
Image → Pretrained SGG extracts scene graph → GloVe word embeddings encode triplets → Autoencoder dimensionality reduction → Normalizing flow anomaly scoring

### Key Designs

**Word Embeddings**: GloVe ($d=300$) is used to encode each token of a triplet $(o_i, p_{i,j}, o_j)$ into a vector, which are concatenated to form $\mathbf{t} \in \mathbb{R}^{900}$. Semantically similar words (e.g., "person" and "human") are close in embedding space, naturally resolving the synonym problem.

**Autoencoder**: A 4-layer fully connected network with ReLU activations compresses the 900-dimensional input to a latent vector of dimension $d_z=512$. Trained exclusively on normal data, it addresses the tension between the bijectivity constraint of normalizing flows (requiring matched input/output dimensionality) and the training instability associated with high-dimensional inputs.

**Normalizing Flow (RealNVP)**: Maps the latent distribution of normal triplets to a standard Gaussian $\mathcal{N}(0, I)$. Anomaly detection is performed via negative log-likelihood:
$$a = -\log p(\mathbf{z}) = -\log p(\mathbf{u}) - \log\left|\det\frac{\partial f_{flow}}{\partial \mathbf{z}}\right|$$
Triplets that deviate from the normal distribution receive high anomaly scores.

### Loss & Training
- Autoencoder: $\mathcal{L}_{AE} = \frac{1}{|\mathcal{T}|}\sum\|\mathbf{t} - \hat{\mathbf{t}}\|^2$
- Normalizing flow: $\mathcal{L}_{flow} = -\frac{1}{2}\|\mathbf{u}\|_2^2 + \log|\det\frac{\partial f_{flow}}{\partial \mathbf{z}}|$ (maximizing the likelihood of normal data)

## Key Experimental Results

### Main Results: Comparison on the SARD Dataset

| Method | Office AUROC↑ | Restaurant AUROC↑ | Training Required | Speed |
|--------|--------------|-------------------|-------------------|-------|
| SARD-o (counting baseline) | ~75% | ~70% | None | Slower |
| SARD-c (corrected data) | ~77% | ~72% | None | Slower |
| **BUSSARD** | **~87%** | **~80%** | Learning-based | **5× faster** |

### Ablation Study: Robustness and Generalizability

| Test Condition | SARD Baseline Deviation | BUSSARD Deviation |
|----------------|------------------------|-------------------|
| Original vocabulary | Reference | Reference |
| Synonym substitution | 17.5% performance fluctuation | Stable (~0%) |

### Latent Dimension Ablation

| $d_z$ | Performance |
|-------|-------------|
| 256 | Suboptimal |
| **512** | **Best** |
| 768 | Slightly lower |

### Key Findings
- BUSSARD achieves approximately 10% higher AUROC than the baseline while being 5× faster at inference.
- Semantic embeddings confer high robustness to synonyms (baseline deviation: 17.5% vs. BUSSARD: ~0%).
- Autoencoder-based dimensionality reduction is critical for the training stability of the normalizing flow.

## Highlights & Insights
- BUSSARD is the first learning-based method for SARD, demonstrating a substantial advantage of learned approaches over counting-based methods in relational anomaly detection.
- The multimodal design integrates scene graphs (structured visual information) with language model embeddings (semantic knowledge), enabling complementary fusion of both modalities.
- Leveraging pretrained word embeddings naturally addresses long-tail and synonym challenges in a simple yet effective manner.

## Limitations & Future Work
- The SARD dataset is relatively small (~120 images); the method's performance on larger-scale data remains to be verified.
- The approach depends on the EGTR scene graph generator—the quality of the upstream SGG directly constrains downstream detection performance.
- Validation is limited to indoor scenes (office/restaurant); generalization to open-world scenarios is unknown.

## Related Work & Insights
- Compared to ComplexVAD: the latter uses scene graphs for video anomaly detection, whereas BUSSARD focuses on image-level relational anomalies.
- The combination of normalizing flows and autoencoders is common in industrial anomaly detection (e.g., FastFlow), but its application to scene graph triplets is novel.
- Insight: The paradigm of pretrained embeddings combined with normalizing flows is generalizable to anomaly detection in other forms of structured data.

## Rating
- Novelty: ⭐⭐⭐⭐ (First learning-based solution to SARD; novel application of normalizing flows to scene graphs)
- Experimental Thoroughness: ⭐⭐⭐ (Small dataset, only 2 scenes, but ablations are thorough)
- Writing Quality: ⭐⭐⭐⭐ (Method is described clearly with intuitive pipeline diagrams)
- Value: ⭐⭐⭐ (Narrow task domain, but the methodological framework has broader applicability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Scene-VLM: Multimodal Video Scene Segmentation via Vision-Language Models](scene-vlm_multimodal_video_scene_segmentation_via_vision-language_models.md)
- [\[CVPR 2026\] FlowComposer: Composable Flows for Compositional Zero-Shot Learning](flowcomposer_composable_flows_for_compositional_zeroshot_learning.md)
- [\[CVPR 2026\] EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval](eaglenet_energy-aware_fine-grained_relationship_learning_network_for_text-video_.md)
- [\[CVPR 2026\] CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods](crosshoi-bench_a_unified_benchmark_for_hoi_evaluation_across_vision-language_mod.md)
- [\[CVPR 2026\] Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training](towards_real-world_document_parsing_via_realistic_scene_synthesis_and_document-a.md)

</div>

<!-- RELATED:END -->
