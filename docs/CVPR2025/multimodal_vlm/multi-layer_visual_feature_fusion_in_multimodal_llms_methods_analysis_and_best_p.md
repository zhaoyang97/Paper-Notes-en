---
title: >-
  [Paper Note] Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices
description: >-
  [Multimodal VLM] This paper systematically studies two core problems of multi-layer visual feature fusion in multimodal LLMs: **(1) how to select the most effective visual layers** and **(2) how to best fuse them into the language model**. The study reveals that selecting one layer from each representation similarity stage and applying external direct fusion is the best practice.
tags:
  - "Multimodal VLM"
date: 2026-05-08
content_hash: decbc9a3db341127
---

# Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices

| Information | Content |
|------|------|
| Conference | CVPR 2025 |
| arXiv | [2503.06063](https://arxiv.org/abs/2503.06063) |
| Code | [EIT-NLP/Layer_Select_Fuse_for_MLLM](https://github.com/EIT-NLP/Layer_Select_Fuse_for_MLLM) |
| Area | Multimodal Large Language Models / Visual Feature Fusion |
| Keywords | Multi-layer Visual Features, Fusion Strategy, Layer Selection, MLLM, LLaVA |

## TL;DR

This paper systematically studies two core problems of multi-layer visual feature fusion in multimodal LLMs: **(1) how to select the most effective visual layers** and **(2) how to best fuse them into the language model**. The study reveals that selecting one layer from each representation similarity stage and applying external direct fusion is the best practice.

---

## Background & Motivation

### Background

Multimodal Large Language Models (MLLMs) have achieved significant progress by combining pre-trained visual encoders with LLMs. However, most models (e.g., LLaVA, InternVL) only utilize the single-layer output of the visual encoder (usually the second-to-last layer), wasting information from other layers. Some works (Dense Connector, EVLM) attempt to use multi-layer features, but their selection methods and fusion strategies lack systematic investigations.

### Limitations of Prior Work

1. **Arbitrary layer selection**: Dense Connector selects layers proportionally, while EVLM directly uses features from the latter half of the layers, both lacking theoretical guidance.
2. **Chaotic fusion strategies**: Some fuse before the LLM input, while others fuse in the middle layers of the LLM; some use extra modules (cross-attention), while others directly concatenate, making the methods incomparable.
3. **Unfair experiments**: Many methods use larger datasets or more complex architectures, making it difficult to determine whether the improvements stem from the fusion strategy or the model capacity.

### Key Challenge

Though multi-layer visual features **are indeed effective**, there is a **lack of systematic study** to inform "which layers to select" and "how to fuse them".

### Key Insight

By controlling other variables (model size, training data) and only varying the layer selection and fusion strategies, this work conducts a comprehensive comparative experiment under a unified framework.

---

## Method

### Overall Architecture

Based on Mini-LLaVA (LLaVA-1.5 architecture + MobileLLaMA 1.4B instead of Vicuna 7B), this work systematically experiments with two dimensions: layer selection strategy $\times$ fusion strategy.

### Visual Layer Selection Strategy

- **Similarity-based**: Based on the cosine similarity of visual features from different layers, the 24-layer encoder is divided into three stages:
    - Beginning stage (low-level): Representative layer is Layer 3—capturing low-level detail features.
    - Middle stage (mid-level): Representative layer is Layer 18—encoding mid-level semantics.
    - Ending stage (high-level): Representative layer is Layer 23—containing high-level discriminative features.
    - Combinations: Single {18}, Double {3, 18}, Triple {3, 18, 23}
- **Proportion-based**: Divides the encoder into front and back halves based on depth:
    - Former {1-12}, Latter {13-24}, All {1-24}
- **Core Discovery**: Selecting **one representative layer from each of the different stages** yields the best performance; choosing **multiple layers from the same stage actually degrades performance**.

### Classification of Four Fusion Strategies

Classified by two dimensions:

| | Modular Fusion | Direct Fusion |
|---|---|---|
| **Internal Fusion** | Cross-attention modules inserted in LLM middle layers | Directly adding visual tokens to LLM middle layers |
| **External Fusion** | Input with text tokens after processing by an extra module | Directly input with text tokens after concatenation/addition |

- **Internal Modular Fusion**: Infuses visual features at corresponding LLM layers via pre-/post-/parallel cross-attention.
- **Internal Direct Fusion**: Directly adds visual tokens at corresponding LLM layers.
- **External Modular Fusion**: Processes multi-layer visual features with extra modules before inputting them into the LLM.
- **External Direct Fusion**: Directly concatenates or adds multi-layer features element-wise, then concatenates them with text tokens to input into the LLM.

### Mini-LLaVA Lightweight Experimental Platform

- **Function**: Reduces the computational cost of exploratory experiments, enabling a large number of ablation studies.
- **Configuration**: CLIP-ViT-L/14 (24 layers) + MobileLLaMA 1.4B (24 layers).
- **Training**: Pre-training phase with 558K image captions + Instruction tuning phase with 665K dialogues.
- **Design Motivation**: Both the visual encoder and the LLM have 24 layers, allowing one-to-one correspondence for internal fusion experiments.

---

## Key Experimental Results

### External Fusion vs. Internal Fusion (Triple {3, 18, 23})

| Fusion Strategy | GQA | MMB | TextVQA | POPE | Average |
|----------|-----|-----|---------|------|------|
| Mini-LLaVA (Baseline) | 56.95 | 46.91 | 35.47 | 85.83 | 48.51 |
| Internal Modular Fusion (Pre-Cross) | 57.56 | 49.66 | 34.06 | 84.69 | 46.91 |
| Internal Direct Fusion | 58.59 | 47.47 | 36.24 | 85.87 | 48.54 |
| **External Direct Fusion** | **59.12** | **51.20** | **36.87** | **86.10** | **49.85** |

### Layer Selection Comparison (External Direct Fusion)

| Layer Set | Average Performance |
|--------|----------|
| Single {18} | 49.20 |
| Double {3, 18} | 49.52 |
| **Triple {3, 18, 23}** | **49.85** |
| Former {1-12} | Decline |
| Latter {13-24} | Decline |
| All {1-24} | Unstable/Decline |

### Validation on LLaVA-1.5 (7B)

| Method | GQA | MMB | TextVQA | Average |
|------|-----|-----|---------|------|
| LLaVA-1.5 Baseline | 62.0 | 64.3 | 58.2 | — |
| + External Direct Fusion (Triple) | **63.1** | **65.7** | **59.4** | — |

Consistent gains validate the transferability of the conclusions.

### Summary of Core Experimental Findings

1. **Cross-stage layer selection outperforms multi-selection from the same stage**: Selecting one layer from each of the beginning/middle/ending stages (Triple {3, 18, 23}) is optimal.
2. **External direct fusion is consistently superior**: Simple, stable, and parameter-efficient.
3. **Internal modular fusion suffers from training difficulties**: The loss is hard to converge when the number of layers increases, and the "All" configuration even fails to complete training.
4. **Differences among the three cross-attention variants are minor**: Pre-cross, post-cross, and parallel perform comparably.
5. **Internal direct fusion shows potential**: It may catch up with external fusion when trained on larger datasets.

---

## Highlights & Insights

1. **Highly systematic**: 2 layer selection criteria $\times$ 4 fusion strategies = a comprehensive experimental matrix, representing the most systematic study in this field.
2. **Clear and actionable conclusions**: Selecting one layer from each of the different similarity stages + external direct fusion = a simple yet effective best practice.
3. **Counter-intuitive finding**: More layers are not necessarily better—redundant multi-layer features from the same stage can be harmful.
4. **Fair controlled variables**: All experiments use the same model and data, eliminating interference from capacity differences.

## Limitations & Future Work

1. The choice of representative layers (3/18/23) is based on empirical training filtering, lacking an automated layer selection method.
2. Whether the findings on Mini-LLaVA (1.4B) hold true for larger models (70B+) remains to be verified.
3. Experiments were only conducted on one visual encoder (CLIP-ViT-L/14); the inter-layer characteristics of other encoders (e.g., SigLIP, InternViT) may differ.
4. The "optimal fusion operation" (addition vs. concatenation) for external direct fusion has not been fully explored.

## Related Work & Insights

- **Comparison with Dense Connector**: Dense Connector selects layers proportionally and fuses them using an extra connector; this work proves that a simpler strategy is superior.
- **Comparison with EVLM**: EVLM uses features from the latter half of the layers; this work finds that layers should be selected from three distinct stages.
- **Comparison with DeepStack**: DeepStack performs internal fusion (injecting high-resolution sub-images into LLM middle layers); this work finds external fusion to be more stable.
- **Insight**: Features from different layers of the visual encoder exhibit "phased" characteristics—redundant within the same stage, but complementary across different stages. This finding can guide the design of all visual tasks requiring multi-layer features.

---

## Rating

⭐⭐⭐⭐ — Highly systematic, solid experiments, and practical conclusions that can directly guide engineering practice; however, the layer selection mechanism is still somewhat empirical, lacking an automated scheme.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](../../ICCV2025/multimodal_vlm/hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](../../ACL2025/multimodal_vlm/table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](../../ACL2025/multimodal_vlm/teaching_vlm_ask_ambiguity.md)
- [\[ICCV 2025\] MM-IFEngine: Towards Multimodal Instruction Following](../../ICCV2025/multimodal_vlm/mm-ifengine_towards_multimodal_instruction_following.md)

</div>

<!-- RELATED:END -->
