---
title: >-
  [Paper Note] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models
description: >-
  [ACL2025][Hallucination Detection][Object Hallucination] Proposes Retrieval Visual Contrastive Decoding (RVCD), which constructs positive/negative logit sets by retrieving AI-generated single-concept explicit images to mitigate object hallucinations in Large Vision-Language Models (LVLMs) during the decoding stage, achieving performance significantly superior to existing decoding methods without requiring extra training.
tags:
  - "ACL2025"
  - "Hallucination Detection"
  - "Object Hallucination"
  - "Contrastive Decoding"
  - "LVLM"
  - "Image Retrieval"
  - "Plug-and-Play"
date: 2026-05-08
content_hash: 39aa6a39fe513612
---

# Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models

**Conference**: ACL2025  
**arXiv**: [2505.20569](https://arxiv.org/abs/2505.20569)  
**Authors**: Jihoon Lee, Min Song (Yonsei University, Onoma AI)
**Code**: [GitHub](https://github.com/JiHoonLee9898/RVCD)  
**Area**: Hallucination Detection  
**Keywords**: Object Hallucination, Contrastive Decoding, LVLM, Image Retrieval, Plug-and-Play

## TL;DR

Proposes Retrieval Visual Contrastive Decoding (RVCD), which constructs positive/negative logit sets by retrieving AI-generated single-concept explicit images to mitigate object hallucinations in Large Vision-Language Models (LVLMs) during the decoding stage, achieving performance significantly superior to existing decoding methods without requiring extra training.

## Background & Motivation

**Object hallucination remains a severe challenge**: Large Vision-Language Models (LVLMs) frequently exhibit three types of object hallucinations when generating text descriptions—existential hallucination (generating non-existent objects), attribute hallucination (incorrectly describing object attributes), and relation hallucination (incorrectly describing relationships between objects)—which severely degrades model reliability.

**Limitations of existing contrastive decoding methods**: Prior methods like VCD and HALC only transform the original input image (e.g., distorting or cropping local views) to generate steering logits, failing to fully exploit the potential of visual contrastive decoding—the images used for steering do not necessarily have to be restricted to transformations of the original image.

**Inadequate self-detection capability of LVLMs for object hallucinations**: Experiments reveal that the accuracy of LVLMs in detecting hallucinated objects via Visual Question Answering (VQA) is much lower than that of traditional object detection (OD) models (such as YOLO). This suggests that the detection capabilities of OD models can be leveraged to assist LVLMs in mitigating hallucinations.

## Method

### Overall Workflow

RVCD adopts a two-stage decoding strategy:

1. **Draft Decoding + Object Detection**: The LVLM is initially used to perform greedy decoding on the input image to generate a draft caption, while YOLO (YOLOv8x) simultaneously performs object detection on the same image to obtain a list of detected objects.
2. **Contrastive Identification of Positive/Negative Objects**: Objects mentioned in the draft caption but not detected by YOLO are defined as negative objects $N$ (suspected hallucinations), while objects detected by both are defined as positive objects $P$ (truly existing).
3. **Retrieval of Explicit Images**: For each object in $N$ and $P$, their corresponding reference images are retrieved from a pre-constructed single-concept image database.
4. **RVCD Steering Decoding**: At each decoding step $t$, negative/positive images are used to generate corresponding logit sets $N_t$ and $P_t$, steering the original logits through a formulation to suppress hallucinated objects and preserve real ones.

### Single-Concept Image Database Construction

- FLUX.1-dev is used to generate single-concept images covering all 400+ words in the CHAIR dictionary (prompt format: "An/A {object}, white background").
- Validation of the generated images is performed via an LVLM (LLaVA-1.5): only when the LVLM description contains the target object is the image added to the database, ensuring semantic alignment between the image generation model and the LVLM.
- The final database maps each word to a high-quality single-concept reference image.

### Core Formulation

The adjusted logit is formulated as:

$$f_{adjusted_t} = f_\theta(\cdot|v,x,y_{<t}) \cdot (1+\alpha \cdot len(N) - \beta \cdot len(P)) - (\alpha \cdot sum(N_t) - \beta \cdot sum(P_t))$$

Where $\alpha$ controls the strength of negative logits (hallucination suppression) and $\beta$ controls the strength of positive logits (real object protection). The optimal setting is $\alpha=1, \beta=0.1$.

### Necessity of Parameter $\beta$ and Positive Logits

Due to co-occurrence bias in LVLMs (e.g., predicting fork, knife, spoon altogether upon seeing a fork image), simply subtracting negative logits could inadvertently damage the representation of real objects. Introducing $\beta$ and positive logits helps recover incorrectly suppressed real object information. Ablation studies confirm that setting $\beta=0.1$ yields improvements in both CHAIR and BLEU metrics.

## Key Experimental Results

### Table 1: CHAIR and BLEU Results (MSCOCO, 500 images x 5 sampling runs)

| Method | LLaVA-1.5 CHAIR_S↓ | CHAIR_I↓ | BLEU↑ | MiniGPT-4 CHAIR_S↓ | CHAIR_I↓ | BLEU↑ |
|------|---------------------|----------|-------|---------------------|----------|-------|
| Greedy | 22.08 | 7.08 | 16.06 | 20.32 | 7.03 | 16.17 |
| VCD | 23.24 | 7.73 | 14.97 | 21.72 | 8.08 | 15.92 |
| HALC | 18.60 | 6.03 | 16.32 | 15.36 | 5.55 | 17.83 |
| OPERA | 18.72 | 6.56 | 16.65 | 19.44 | 7.22 | 17.77 |
| **RVCD** | **11.32** | **3.87** | 15.48 | **9.00** | **3.61** | 15.98 |

RVCD significantly reduces CHAIR (CHAIR_S drop by around 40-50%) across all three backbones, while BLEU only experiences a minor decrease, indicating that text quality remains well-preserved.

### Table 2: POPE Evaluation Results

| Method | LLaVA-1.5 Acc↑ | Prec↑ | F1↑ | mPLUG-Owl2 Acc↑ | Prec↑ | F1↑ |
|------|----------------|-------|-----|-----------------|-------|-----|
| Greedy | 72.19 | 65.28 | 77.86 | 74.36 | 67.23 | 79.23 |
| Beam Search | 78.27 | 71.94 | 81.28 | 80.17 | 74.30 | 82.64 |
| HALC | 72.48 | 65.54 | 78.04 | 74.54 | 67.42 | 79.33 |
| **RVCD** | **88.54** | **89.92** | **88.43** | **87.45** | **87.91** | **87.41** |

RVCD substantially outperforms all baselines in POPE accuracy, precision, and F1, achieving gains of approximately 10-20 percentage points.

### Latency Analysis (Table 4)

| Method | Average Latency (s/token) | Relative Scale |
|------|-------------------|---------|
| Greedy | 0.034 | 1.0x |
| OPERA | 0.341 | 10.1x |
| HALC | 0.800 | 23.8x |
| RVCD (β=0) | 0.143 | 4.2x |
| RVCD (β≠0) | 0.204 | 6.1x |

The latency of RVCD is significantly lower than that of OPERA and HALC, demonstrating a clear efficiency advantage.

## Highlights

- **Novel Retrieval Paradigm**: Introduces external explicit image retrieval into visual contrastive decoding for the first time, breaking free from the limitation of only using original image transformations.
- **Training-free and Plug-and-play**: Can be directly applied to any open-source LVLM (e.g., MiniGPT-4, LLaVA-1.5, mPLUG-Owl2) without requiring any fine-tuning.
- **Bidirectional Steering of Positive/Negative Logits**: Cleverly designed to not only suppress hallucinated objects but also protect real objects from being suppressed using positive logits.
- **Comprehensive Ablation Analysis**: Systematically investigates the effects of detection accuracy, $\alpha/\beta$ parameters, and different OD models on performance, providing robust conclusions.

## Limitations & Future Work

1. **Dependency on the CHAIR Dictionary**: The single-concept image database is based on the limited dictionary of MSCOCO (around 400 words), making it challenging to generalize to open-vocabulary scenarios.
2. **Latency Positively Correlated with Draft Length**: When the draft caption mentions many different objects, extra logits must be generated for each object, increasing decoding latency.
3. **Dependency on OD Model Quality**: The performance of RVCD is positively correlated with the accuracy of the object detection model (validated in Table 3); any missed detections or false positives from the OD model directly impact the results.
4. **Only Focusing on Object-Level Hallucinations**: It does not address hallucinations at the attribute and relationship levels (although partially covered under the CHAIR framework, it lacks fine-grained details).

## Related Work

| Method | Strategy | Requires Training | Image Source |
|------|------|------------|---------|
| VCD | Distort original image to generate contrastive logits | No | Original image transformation |
| HALC | Crop local views containing key objects | No | Original image crop |
| OPERA | Penalty on over-trust in attention weights | No | No extra image |
| DoLA | Contrast logits from different layers | No | No extra image |
| **RVCD** | Retrieve external single-concept images for positive/negative contrast | No | External AI-generated image database |

The core distinction of RVCD lies in introducing external explicit images unrelated to the original image, making the steering signal for contrastive decoding more precise and controllable.

## Rating

- Novelty: ⭐⭐⭐⭐ (Retrieving external explicit images for contrastive decoding is a novel idea)
- Experimental Thoroughness: ⭐⭐⭐⭐ (CHAIR/POPE/MME/LLaVA-Bench + 3 backbones + detailed ablation)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation, intuitive flowcharts, comprehensive formula derivations)
- Value: ⭐⭐⭐⭐ (Plug-and-play method with significant results, though limited by the generalization of the CHAIR dictionary)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] First Logit Boosting: Visual Grounding Method to Mitigate Object Hallucination in Large Vision-Language Models](../../CVPR2026/hallucination/first_logit_boosting_visual_grounding_method_to_mitigate_object_hallucination_in.md)
- [\[ACL 2025\] Visual Evidence Prompting Mitigates Hallucinations in Large Vision-Language Models](visual_evidence_prompting.md)
- [\[CVPR 2026\] VES-RFT: Rewarding Visual Evidence Sensitivity to Mitigate Hallucinations in Large Vision-Language Models](../../CVPR2026/hallucination/ves-rft_rewarding_visual_evidence_sensitivity_to_mitigate_hallucinations_in_larg.md)
- [\[ACL 2025\] Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention](activation_steering_decoding_mitigating_hallucination_in_large_vision-language_m.md)
- [\[CVPR 2025\] Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding](../../CVPR2025/hallucination/octopus_alleviating_hallucination_via_dynamic_contrastive_decoding.md)

</div>

<!-- RELATED:END -->
