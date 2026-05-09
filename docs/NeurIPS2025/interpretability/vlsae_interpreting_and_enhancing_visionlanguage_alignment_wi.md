---
title: >-
  [Paper Note] VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set
description: >-
  [NeurIPS 2025][Sparse Autoencoder] This paper proposes VL-SAE, a sparse autoencoder with a distance-based encoder and modality-specific decoders that maps the semantics of both visual and linguistic representations onto a unified concept set, thereby interpreting and enhancing the vision-language alignment mechanism of VLMs. The approach yields an average improvement of 0.6–0.9% on zero-shot classification and outperforms the dedicated method VCD on POPE hallucination mitigation.
tags:
  - NeurIPS 2025
  - Sparse Autoencoder
  - vision-language alignment
  - unified concept set
  - interpretability
  - hallucination mitigation
date: 2026-05-08
content_hash: 2247a39e529116db
---

# VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set

**Conference**: NeurIPS 2025
**arXiv**: [2510.21323](https://arxiv.org/abs/2510.21323)
**Code**: [https://github.com/ssfgunner/VL-SAE](https://github.com/ssfgunner/VL-SAE)
**Area**: Multimodal VLM / Interpretability / Representation Alignment
**Keywords**: Sparse Autoencoder, vision-language alignment, unified concept set, interpretability, hallucination mitigation

## TL;DR
This paper proposes VL-SAE, a sparse autoencoder with a distance-based encoder and modality-specific decoders that maps the semantics of both visual and linguistic representations onto a unified concept set, thereby interpreting and enhancing the vision-language alignment mechanism of VLMs. The approach yields an average improvement of 0.6–0.9% on zero-shot classification and outperforms the dedicated method VCD on POPE hallucination mitigation.

## Background & Motivation
The core capability of VLMs (e.g., CLIP, LLaVA) stems from the alignment between visual and linguistic representations, yet the interpretability of this alignment mechanism has received little attention. Existing representation interpretation methods either focus exclusively on the visual side (e.g., SpLiCE decomposes CLIP visual representations into text) or on the language side (e.g., Parekh et al. decompose token representations into concepts), and are thus unable to map the semantics of both modalities into a shared concept space for comparative analysis.

An intuitive solution is to train separate SAEs for each modality; however, since the concept set of an SAE is learned in a self-supervised and uncontrolled manner, neurons at the same position in two independently trained SAEs will associate with different concepts—the so-called *concept mismatch* problem. Even with a shared SAE, semantically similar visual and linguistic representations cannot be guaranteed to produce consistent activation patterns due to the distributional gap between modalities.

## Core Problem
**How can the semantics of both visual and linguistic representations be mapped onto a unified concept set?** Two key challenges arise: (1) different VLMs employ different alignment strategies (CLIP uses contrastive loss for explicit alignment; LLaVA uses QA tasks for implicit alignment), making it non-trivial to measure cross-modal semantic similarity in a unified way; (2) multimodal representations occupy different regions of the feature space, making it difficult to ensure that semantically similar cross-modal representations receive consistent sparse activations while maintaining reconstruction quality.

## Method

### Overall Architecture
Given an image-text pair, a VLM extracts visual representations $\mathbf{x}_v$ and linguistic representations $\mathbf{x}_l$. For LVLMs (e.g., LLaVA), an auxiliary autoencoder first converts implicitly aligned representations into explicitly aligned intermediate representations $\mathbf{x}_v^e, \mathbf{x}_l^e$ via InfoNCE alignment of cosine similarities. VL-SAE then employs a shared distance-based encoder $E_s$ to encode both modalities into sparse activations $\mathbf{h}_v, \mathbf{h}_l$, which are subsequently reconstructed by two modality-specific decoders $D_v^s$ and $D_l^s$.

### Key Designs

1. **Explicit Representation Alignment (Auxiliary Autoencoder)**: For CVLMs (e.g., CLIP), visual and linguistic representations are naturally aligned in terms of cosine similarity, so original representations are used directly. For LVLMs (e.g., LLaVA), alignment is achieved implicitly through QA tasks, and cosine similarity does not directly reflect semantic similarity. An auxiliary autoencoder is therefore trained with InfoNCE loss (temperature 0.07) to align intermediate representations into cosine-similarity form, while a reconstruction loss preserves information completeness.

2. **Distance-Based Encoder**: The key innovation. Conventional SAEs use a linear transformation followed by ReLU as the encoder ($\mathbf{h} = \sigma(W\mathbf{x} + b)$), whose activations depend on inner products and cannot guarantee that semantically similar cross-modal representations yield similar activations. This paper proposes an encoder based on normalized Euclidean distance:
    $$E_s(\mathbf{x}^e)[i] = 2 - \left\|\frac{\mathbf{x}^e}{\|\mathbf{x}^e\|_2} - \frac{\mathbf{w}_i}{\|\mathbf{w}_i\|_2}\right\|_2 = 2 - \sqrt{2 - 2\cos(\mathbf{x}^e, \mathbf{w}_i)}$$
   This distance satisfies the **triangle inequality**: $|g(\mathbf{x}_v^e, \mathbf{w}_i) - g(\mathbf{x}_l^e, \mathbf{w}_i)| \leq g(\mathbf{x}_v^e, \mathbf{x}_l^e)$, which theoretically guarantees that when visual and linguistic representations have high cosine similarity, the upper bound on their activation difference for any given neuron is small. Top-K sparsification retains the $k=256$ largest activations.

3. **Modality-Specific Decoders**: Using a single shared decoder would force the encoder to embed modality-discriminative information into the activations in order to simultaneously reconstruct representations from both modalities (which have different distributions), causing semantically similar cross-modal representations to exhibit inconsistent activations. Two independent linear decoders $D_v^s$ and $D_l^s$ are used instead, each storing modality-specific distributional information, so that encoder activations encode only semantic information.

### Loss & Training
- **Auxiliary autoencoder** (required for LVLMs only): $\mathcal{L} = \text{InfoNCE}(\mathbf{x}_v^e, \mathbf{x}_l^e, \mathbf{x}_v^{e-}, \mathbf{x}_l^{e-}) + \|\hat{\mathbf{x}}_v - \mathbf{x}_v\|_2^2 + \|\hat{\mathbf{x}}_l - \mathbf{x}_l\|_2^2$; trained for 50 epochs, batch size 2048.
- **VL-SAE**: $\mathcal{L} = \|\hat{\mathbf{x}}_v^e - \mathbf{x}_v^e\|_2^2 + \|\hat{\mathbf{x}}_l^e - \mathbf{x}_l^e\|_2^2$; trained for 10 epochs, batch size 512.
- Training data: CC3M (3 million image-text pairs), 4:1 train/test split.
- Hidden dimension is 8× the representation dimension (e.g., LLaVA's 4096-d → 32,768 hidden neurons).
- All models trained on a single RTX 4090; training cost is extremely low (VL-SAE for ViT-B/16 requires only 0.03G FLOPs and 132 seconds).

## Key Experimental Results

### Zero-Shot Image Classification (OpenCLIP + VL-SAE)

| Model | Avg. Acc. | Avg. Acc. + VL-SAE | Gain |
|---|---|---|---|
| ViT-B/32 | 68.7% | 69.5% | +0.8 |
| ViT-B/16 | 69.8% | 70.4% | +0.6 |
| ViT-L/14 | 72.2% | 72.9% | +0.7 |
| ViT-H/14 | 76.9% | 77.8% | +0.9 |

Consistent improvements are observed across 14 classification datasets. Predictions are made by combining original cosine similarity with concept-level activation cosine similarity: $y = \cos(\mathbf{x}_v, \mathbf{x}_l) + \alpha_c \cos(\mathbf{h}_v, \mathbf{h}_l)$.

### POPE Hallucination Mitigation (LLaVA 1.5)

| Setting | Method | F1 |
|---|---|---|
| Random | Regular | 80.87 |
| Random | VCD | 84.04 |
| Random | **VL-SAE** | **85.50** |
| Popular | Regular | 79.27 |
| Popular | VCD | 82.31 |
| Popular | **VL-SAE** | **84.37** |
| Adversarial | Regular | 77.16 |
| Adversarial | VCD | 80.13 |
| Adversarial | **VL-SAE** | **82.29** |

### CHAIR Hallucination Mitigation (Open-Ended Description Generation)

| Model | Method | CHAIR_S↓ | CHAIR_I↓ | Recall↑ |
|---|---|---|---|---|
| LLaVA1.5 | Regular | 53.4 | 17.6 | 72.3 |
| LLaVA1.5 | VCD | 55.0 | 16.3 | 76.0 |
| LLaVA1.5 | **VL-SAE** | **47.8** | **13.3** | **76.3** |
| Qwen-VL | Regular | 44.6 | 16.1 | 60.7 |
| Qwen-VL | **VL-SAE** | **39.6** | **10.7** | **63.3** |

### Ablation Study
- The **distance-based encoder** is the core contribution: replacing a standard SAE encoder with the distance-based encoder improves Intra-Sim from 0.1890 → 0.2016 (OpenCLIP) and 0.2086 → 0.2216 (LLaVA).
- **Modality-specific decoders** yield further gains: adding them brings Intra-Sim to 0.2134/0.2257 and reduces Inter-Sim to 0.1149/0.1828.
- The **auxiliary autoencoder** is indispensable for LVLMs: removing it (i.e., training VL-SAE directly on original representations) drops Intra-Sim from 0.2257 to 0.2084 and raises Inter-Sim from 0.1828 to 0.2034, indicating severe degradation of concept quality.
- Top-K sparsification outperforms L1 regularization: Intra-Sim 0.2442 vs. 0.2142; Inter-Sim 0.1373 vs. 0.1809.
- Concept quality improves monotonically with data volume: scaling from 20% to 100% of CC3M raises Intra-Sim from 0.2029 to 0.2299.
- VL-SAE learns more effective concepts: only 15 dead neurons, compared to 54 (SAE-D) and 46 (SAE-S).

## Highlights & Insights
- **The distance-based encoder is elegantly motivated**: by exploiting the relationship between normalized Euclidean distance and cosine similarity, it naturally satisfies the triangle inequality, providing theoretical guarantees that semantically similar cross-modal representations yield similar activations. The design is both concise and mathematically principled.
- **The insight behind modality-specific decoders is profound**: a shared decoder compels the encoder to embed modality-discriminative information into its activations, and this observation explains why a naïve shared SAE fails.
- **Training cost is negligible**: VL-SAE amounts to essentially two linear layers in terms of parameter count, trainable on a single RTX 4090 in minutes. As a plug-and-play inference module, it has virtually no impact on throughput (935 → 935 samples/s).
- **Interpretation and enhancement form a closed loop**: the work goes beyond explaining the model—it leverages the interpretation results (concept-level alignment) to improve model performance, with demonstrated gains in both classification and hallucination mitigation.
- **Human evaluation** overwhelmingly favors VL-SAE's concept quality (65.9% vs. SAE-S 33.5% vs. SAE-D 0.6%).

## Limitations & Future Work
- **Limitations of concept evaluation metrics**: numerical differences in CLIP similarity scores may not fully capture differences in concept quality, as the score gap between semantically similar and dissimilar image-text pairs is sometimes small. A more comprehensive evaluation framework is needed.
- **Dead neuron problem persists**: although VL-SAE exhibits fewer dead neurons than baselines, they still exist, and high-frequency neurons require reweighting.
- **No modeling of inter-concept relationships**: each neuron independently associates with a single concept, without capturing hierarchical or compositional relationships among concepts.
- **Enhancement strategies are relatively simple**: zero-shot classification merely linearly combines original and concept-level predictions; hallucination mitigation injects visual concept activations via contrastive decoding. More fine-grained concept-level intervention strategies remain to be explored.
- **The hyperparameter $\alpha_c$ requires per-task tuning** (though at low cost); task-agnostic settings still yield improvements but are suboptimal compared to task-specific tuning.

## Related Work & Insights
- **SpLiCE (NeurIPS 2024)**: decomposes CLIP visual representations into sparse combinations of text concepts, but operates only on the visual side and cannot perform cross-modal concept comparison. VL-SAE processes both modalities with a shared concept set.
- **SAE-V (2025)**: trains SAEs on multimodal token representations in LVLMs, but targets efficient data sampling rather than interpreting alignment mechanisms. VL-SAE focuses specifically on alignment interpretation and enhancement.
- **Parekh et al. (NeurIPS 2024)**: a concept interpretation framework that decomposes LVLM token representations into visual-linguistic concepts, but lacks a unified concept set, making it impossible to directly compare concept activations across modalities to understand alignment or misalignment.
- **VCD (CVPR 2024)**: a dedicated hallucination mitigation method based on contrastive decoding. VL-SAE, as a general interpretability tool, nevertheless surpasses VCD on hallucination mitigation, demonstrating the practical value of the *understand the model → improve the model* pathway.

The triangle inequality trick of the distance-based encoder is transferable to other scenarios requiring consistent cross-modal representations (e.g., audio-visual, tactile-visual). The idea of enhancing inference through concept-level alignment may also benefit multimodal RAG and few-shot capability improvement in VLMs.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The distance-based encoder and modality-specific decoder have clear mathematical motivation, and the unified concept set idea is novel, though the overall framework is not revolutionary.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 CVLMs and 2 LVLMs, 14 classification datasets, POPE and CHAIR benchmarks, detailed ablations, and human evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem motivation is clear, method derivation is rigorous, figures are intuitive, and the narrative is logically coherent.
- **Value**: ⭐⭐⭐⭐ Establishes a closed loop from alignment interpretation to alignment enhancement, with low training cost, plug-and-play deployment, and strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](probabilistic_token_alignment_for_large_language_model_fusion.md)
- [\[NeurIPS 2025\] Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions](explaining_similarity_in_vision-language_encoders_with_weighted_banzhaf_interact.md)
- [\[NeurIPS 2025\] Efficient Vision-Language Reasoning via Adaptive Token Pruning](efficient_vision-language_reasoning_via_adaptive_token_pruning.md)
- [\[NeurIPS 2025\] Out of Control -- Why Alignment Needs Formal Control Theory (and an Alignment Control Stack)](out_of_control_--_why_alignment_needs_formal_control_theory_and_an_alignment_con.md)
- [\[ICLR 2026\] Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language](../../ICLR2026/interpretability/semantic_regexes_auto-interpreting_llm_features_with_a_structured_language.md)

</div>

<!-- RELATED:END -->
