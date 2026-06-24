---
title: >-
  [Paper Note] Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models
description: >-
  [ICCV 2025][Multimodal VLM][Knowledge Editing] This paper proposes a vision-oriented fine-grained multimodal knowledge editing task and the FGVEdit benchmark, designing the MSCKE framework that integrates visual and textual information via a multimodal scope classifier. This achieves precise knowledge updates for multiple interacting entities within an image, significantly outperforming text-only editing methods.
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Knowledge Editing"
  - "MLLM"
  - "Fine-Grained Visual Editing"
  - "Multimodal Classifier"
  - "FGVEdit"
date: 2026-05-08
content_hash: bc4a2b58351ac8f6
---

# Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models

**Conference**: ICCV 2025  
**arXiv**: [2411.12790](https://arxiv.org/abs/2411.12790)  
**Code**: Not provided  
**Area**: Multimodal VLM  
**Keywords**: Knowledge Editing, MLLM, Fine-Grained Visual Editing, Multimodal Classifier, FGVEdit  

## TL;DR

This paper proposes a vision-oriented fine-grained multimodal knowledge editing task and the FGVEdit benchmark, designing the MSCKE framework that integrates visual and textual information via a multimodal scope classifier. This achieves precise knowledge updates for multiple interacting entities within an image, significantly outperforming text-only editing methods.

## Background & Motivation

Knowledge editing aims to correct erroneous knowledge in models efficiently and cost-effectively. With the rise of Multimodal Large Language Models (MLLMs), knowledge editing needs to be extended to multimodal scenarios. However, existing works suffer from several key limitations:

**Limitations of Coarse-Grained Editing**: Existing multimodal knowledge editing benchmarks (MMEdit, KEBench, MIKE) treat the entire image as a single entity. The editing operation is essentially a simple text replacement (e.g., replacing "bird" with "kite") without requiring access to the visual modules, which explains why LLM editing methods perform well under these setups.

**Neglecting Multi-Entity Interaction**: In real-world scenarios, an image typically contains multiple interacting entities. It is necessary to edit knowledge related to a specific entity without affecting other entities.

**Lack of Visual Information**: Relying solely on textual semantics cannot capture the subtle relationships among different entities in the same image.

As shown in Figure 1, changing "bird" to "kite" in coarse-grained editing only requires text manipulation. In contrast, fine-grained editing requires precisely locating the target entity (e.g., one of the kites) within the same image and updating its associated knowledge while keeping other entities unaffected.

## Method

### Problem Formulation

Given a pre-trained MLLM $f_\theta$, an edit sample is denoted as $(i_e, t_e, y_e)$, where $i_e$ is an image containing multiple entities, $t_e$ is the text prompt, and $y_e$ is the target output. The knowledge editing operation $\mathcal{E}$ updates parameters to $\theta_e = \mathcal{E}(\theta, i_e, t_e, y_e)$, such that $y_e = f_{\theta_e}(i_e, t_e)$.

The key challenge lies in defining the **fine-grained edit scope**: for an edit sample, there is an edit scope $S(i_e, t_e, y_e)$ jointly determined by visual and textual information:
- **In-scope inputs**: Must produce the corrected output.
- **Out-of-scope inputs** (same image but pointing to different entities): Must keep the original output unchanged.

### MSCKE Framework

Improved based on the SERAC method, MSCKE consists of four core components:

1. **Multimodal Edit Memory**: Stores edit samples without modifying the base model parameters.
2. **Multimodal Scope Classifier**: Evaluates the relevance of the input to the edit samples.
3. **Base Multimodal Model** $f_{\text{base}}$: Parameter-frozen, handles out-of-scope inputs.
4. **Counterfactual Multimodal Model** $f_{\text{cfr}}$: Trainable parameters, handles in-scope inputs.

Decision logic during inference:

$$y_{\text{test}} = \begin{cases} f_{\text{base}}(i_{\text{test}}, t_{\text{test}}), & \rho < 0.5 \\ f_{\text{cfr}}(t_e, y_e, i_{\text{test}}, t_{\text{test}}), & \rho \geq 0.5 \end{cases}$$

where $\rho$ is the similarity score computed by the multimodal scope classifier.

### Multimodal Scope Classifier

This is the core innovation of MSCKE. Unlike SERAC's text-only classifier, this classifier fuses both visual and textual modalities:

**Feature Extraction and Alignment**: A pre-trained CLIP model is utilized to extract image and text features separately and map them to a unified space:

$$h_e^i = A_i(E_i(i_e)), \quad h_e^t = A_t(E_t(t_e, y_e))$$

$$h_{\text{test}}^i = A_i(E_i(i_{\text{test}})), \quad h_{\text{test}}^t = A_t(E_t(t_{\text{test}}))$$

**Feature Fusion**: Dot-product attention is used to fuse visual and textual features:

$$z_e = \text{Fusion}(h_e^i, h_e^t), \quad z_{\text{test}} = \text{Fusion}(h_{\text{test}}^i, h_{\text{test}}^t)$$

**Similarity Calculation**:

$$\rho = \text{Sim}(z_e, z_{\text{test}})$$

### Loss & Training

The classifier is trained as a binary classifier using binary cross-entropy loss:

$$\mathcal{L}_{cls} = -\frac{1}{N} \sum_{k=1}^{N} [\log(\rho_{\text{in}}^k) + \log(1 - \rho_{\text{out}}^k)]$$

where $\rho_{\text{in}}^k$ and $\rho_{\text{out}}^k$ are the similarity scores of in-scope and out-of-scope samples, respectively.

### FGVEdit Benchmark Construction

Constructed based on VQAv2, containing 11,112 samples (Train:Test = 3:1):

- **Specificity Data**: Employs GPT-4o-mini for a two-stage classification—first determining logical entailment based on image+text, then filtering out "hard" samples based solely on text.
- **Locality Data**: Uses the NQ dataset (irrelevant Q&A pairs).
- **Generality Data**: GPT-4o-mini rewrites the questions.

### Evaluation Metrics

The newly proposed **Specificity** metric consists of two components:

$$M_{\text{specificity}} = \frac{1}{2}[M_{\text{in}}^v + M_{\text{out}}^v]$$

- $M_{\text{in}}^v$: Accuracy rate of in-scope visual questions (should reflect edited knowledge).
- $M_{\text{out}}^v$: Retention rate of original answers for out-of-scope visual questions (should be unaffected by editing).

## Experimental Results

### Main Results (BLIP-2 OPT / MiniGPT-4)

| Method | Reliability | Locality | Generality | Specificity |
|------|------|------|------|------|
| FT-LLM | 100.0/93.4 | 76.9/86.3 | 100.0/93.4 | 24.2/35.0 |
| IKE | 99.9/100.0 | 48.5/52.5 | 98.0/98.9 | 20.1/25.3 |
| SERAC | 93.1/99.5 | 99.9/100.0 | 96.8/92.9 | 31.9/37.9 |
| MEND | 97.0/94.9 | 98.6/98.6 | 96.4/94.8 | 65.9/67.4 |
| **MSCKE** | 99.1/99.5 | **100.0/100.0** | 98.6/93.0 | 61.6/57.2 |
| **MSCKE-MEND** | 97.4/97.1 | **100.0/100.0** | 96.5/96.7 | **68.4/72.0** |

**Key Findings**:
- On traditional metrics, differences among methods are minor; however, on the crucial Specificity metric, MSCKE (61.6) significantly outperforms SERAC (31.9), nearly doubling the performance.
- MSCKE-MEND further improves Specificity to 68.4/72.0.

### Ablation Study

| Component | CLIP-ViT-B/32 | CLIP-ViT-L/14 |
|------|------|------|
| Concatenation Fusion | 63.70 | 63.80 |
| Cross-Attention | 64.45 | 64.35 |
| **Dot-Product Attention** | **64.73** | **64.85** |

**Key Findings**:
- ViT-B/32 paired with ViT-L/14 achieves comparable performance, indicating that a lightweight backbone is sufficient.
- Dot-product attention fusion is optimal and yields the minimum computational overhead.

### Transferability Experiments

| Source $\rightarrow$ Target | Specificity (Transfer / Retrain) |
|------|------|
| BLIP-2 → BLIP-2 (MSCKE-MEND) | 68.35 / 68.38 |
| BLIP-2 → MiniGPT-4 (MSCKE) | 57.09 / 57.20 |
| BLIP-2 → MiniGPT-4 (MSCKE-MEND) | 72.16 / 71.98 |

The performance of the classifier remains almost lossless when transferred across different models, demonstrating excellent model-agnostic capabilities.

### Computational Cost

| Component | Inference Time | Model Size |
|------|------|------|
| Multimodal Classifier | 36ms | 0.56G |
| Base Model | 121ms | 9.10G |
| Counterfactual Model | 85ms | 4.22G |

The classifier introduces only minimal additional overhead (36ms / 0.56G).

## Highlights & Insights

1. **Contribution to Problem Definition**: This work is the first to explicitly propose the new task of "visual-oriented fine-grained knowledge editing", revealing the limitations of coarse-grained editing benchmarks.
2. **Multimodal Scope Discrimination**: Text-only classifiers fail in fine-grained scenarios (due to high textual similarity between in-scope and out-of-scope samples); the multimodal classifier correctly distinguishes them by introducing visual information.
3. **Advantages of Decoupled Design**: The classifier, base model, and counterfactual model are completely decoupled, supporting flexible combinations and transfer.
4. **Specificity Metric**: Fills the gap in existing evaluation frameworks regarding the assessment of multi-entity editing within the same image.

## Limitations & Future Work

1. The highest Specificity is only around 72%, leaving significant room for improvement.
2. It relies on the alignment capability of CLIP for cross-modal matching, which may fail for similar entities that CLIP struggles to distinguish.
3. The FGVEdit benchmark is constructed based on VQAv2, so the scene diversity might be limited by the original dataset.
4. The edit memory grows linearly with the number of edit samples, potentially impacting retrieval efficiency.

## Related Work & Insights

- **LLM Knowledge Editing**: Parameter-preserving methods (SERAC, IKE, MemPrompt) vs. parameter-modifying methods (ROME, MEMIT, MEND).
- **MLLM Knowledge Editing**: MMEdit as the first multimodal benchmark, KEBench introducing generalization metrics, and MIKE & MC-MKE exploring fine-grained but still text-centric editing.
- **Foundation Models**: BLIP-2 OPT and MiniGPT-4 serving as the edited MLLMs.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | 7.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)
- [\[ICCV 2025\] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](basic_boosting_visual_alignment_with_intrinsic_refined_embeddings_in_multimodal_.md)
- [\[ICLR 2026\] GranViT: A Fine-Grained Vision Model For Autoregressive Multimodal Large Language Models](../../ICLR2026/multimodal_vlm/granvit_a_fine-grained_vision_model_for_autoregressive_multimodal_large_language.md)
- [\[NeurIPS 2025\] VITRIX-CLIPIN: Enhancing Fine-Grained Visual Understanding in CLIP via Instruction Editing Data and Long Captions](../../NeurIPS2025/multimodal_vlm/vitrix-clipin_enhancing_fine-grained_visual_understanding_in_clip_via_instructio.md)
- [\[CVPR 2026\] OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/oddgridbench_exposing_the_lack_of_fine-grained_visual_discrepancy_sensitivity_in.md)

</div>

<!-- RELATED:END -->
