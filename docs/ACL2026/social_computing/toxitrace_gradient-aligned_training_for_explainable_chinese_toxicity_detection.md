---
title: >-
  [Paper Note] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection
description: >-
  [ACL 2026][Social Computing][Chinese Toxicity Detection] ToxiTrace proposes an explainable Chinese toxicity detection method for BERT-class encoders, combining CuSA (LLM-guided weak annotation), GCLoss (gradient-constrained loss), and ARCL (adversarial reasoning contrastive learning) to achieve both high sentence-level classification accuracy and contiguous toxic span extraction while maintaining efficient encoder inference.
tags:
  - ACL 2026
  - Social Computing
  - Chinese Toxicity Detection
  - Explainability
  - Gradient Constraint
  - Fine-grained Evidence Extraction
  - Contrastive Learning
date: 2025-05-08
content_hash: c68664a8f379679b
---

# ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection

**Conference**: ACL 2026  
**arXiv**: [2604.12321](https://arxiv.org/abs/2604.12321)  
**Code**: [https://huggingface.co/ArdLi/ToxiTrace](https://huggingface.co/ArdLi/ToxiTrace)  
**Area**: Social Computing  
**Keywords**: Chinese Toxicity Detection, Explainability, Gradient Constraint, Fine-grained Evidence Extraction, Contrastive Learning

## TL;DR

ToxiTrace proposes an explainable Chinese toxicity detection method for BERT-class encoders, combining CuSA (LLM-guided weak annotation), GCLoss (gradient-constrained loss), and ARCL (adversarial reasoning contrastive learning) to achieve both high sentence-level classification accuracy and contiguous toxic span extraction while maintaining efficient encoder inference.

## Background & Motivation

**Background**: Existing Chinese toxicity detection methods primarily target sentence-level classification and have achieved strong performance through pre-trained language models (e.g., RoBERTa, MacBERT) and large language models.

**Limitations of Prior Work**:
- Most methods only perform sentence-level classification and cannot pinpoint which specific spans within a sentence are toxic, lacking explainability
- Chinese uses character-level tokenization, causing gradient/attention attribution signals to fragment across individual characters and fail to form human-readable contiguous spans
- LLMs have strong explanation capabilities but underperform encoders in direct classification and incur high inference costs

**Key Challenge**: Encoder models are accurate classifiers but poor explainers (fragmented attributions), while LLMs excel at explanation but are weak classifiers with slow inference—the strengths of both cannot be combined.

**Goal**: Maintain efficient encoder inference while enabling the model to both classify accurately and extract contiguous, readable toxic spans as explanations.

**Key Insight**: By explicitly constraining gradient signals during training, the encoder's token-level attributions naturally focus on toxic evidence, enabling direct contiguous span extraction from saliency maps at inference time.

**Core Idea**: Elevate gradient attribution from "post-hoc explanation" to "training objective"—use LLM-generated weak annotations to guide gradients toward toxic tokens while sharpening the toxic/non-toxic semantic boundary through contrastive learning.

## Method

### Overall Architecture

ToxiTrace follows a four-stage pipeline: (1) warm-up training for baseline classification; (2) CuSA leverages encoder attribution cues + LLM refinement to generate weakly annotated toxic spans; (3) GCLoss explicitly boosts gradient responses on toxic tokens and suppresses non-toxic ones; (4) ARCL sharpens the semantic boundary between toxic and non-toxic content. At inference, the model first classifies, then applies the BiCSE algorithm to extract contiguous spans from the saliency map for toxic inputs.

### Key Designs

1. **CuSA (Clue-guided Span Annotation)**:

    - Function: Automatically generates weak supervision signals for toxic spans without fine-grained annotations
    - Mechanism: First warm-up trains the encoder for baseline classification, then computes token-level saliency scores. The BiCSE (Bidirectional Cliff Scan and Extraction) algorithm extracts initial high-saliency spans as clues, which are fed to an LLM (Gemini 2.5 Pro) to refine span boundaries
    - Design Motivation: Existing Chinese toxicity datasets have only coarse-grained labels, but direct LLM annotation lacks localization cues. Using the encoder's own attribution signals to provide candidate regions, then leveraging the LLM's comprehension ability to refine them, achieves complementary strengths

2. **GCLoss (Gradient-Constrained Loss)**:

    - Function: Explicitly shapes token-level gradient distributions so that toxic tokens have higher gradient responses than non-toxic ones
    - Mechanism: Comprises two components—PGR Loss enforces a margin between toxic/non-toxic token gradients; PPT Loss uses within-sample statistics (15th percentile + α·max) to cap the gradient upper bound for non-toxic tokens and set a gradient lower bound for toxic tokens
    - Design Motivation: Models trained solely with classification loss produce scattered, inaccurate token-level attributions. Directly constraining gradients concentrates attributions on toxic evidence, making span extraction at inference more reliable

3. **ARCL (Adversarial Reasoning Contrastive Learning)**:

    - Function: Sharpens the semantic boundary between toxic and non-toxic text
    - Mechanism: Uses an LLM (Gemini 2.5 Flash) to generate pro and con reasoning arguments ("Assume the text is toxic/non-toxic, generate supporting rationales"), which serve as positive and negative samples for adaptive InfoNCE contrastive learning
    - Design Motivation: GCLoss only constrains intra-sentence token-level gradient relationships and cannot capture inter-sentence semantic differences. Reasoning content generated via the LLM debate mechanism is more targeted and semantically deeper than simple data augmentation

### Loss & Training

Overall training objective: $\mathcal{L} = \mathcal{L}_{CE} + \lambda_{grad}(\mathcal{L}_{PGR} + \mathcal{L}_{PPT}) + \lambda_{sem}\mathcal{L}_{con}$

Training procedure: warm-up for 3 epochs (cross-entropy only) → introduce GCLoss + ARCL for joint training. Too many or too few warm-up steps both degrade final performance.

## Key Experimental Results

### Main Results (Classification)

| Dataset | Metric | Ours (RoBERTa+ToxiTrace) | Prev. SOTA (RoBERTa) | Gain |
|---------|--------|--------------------------|----------------------|------|
| COLD | Macro-F1 | **83.68%** | 82.56% | +1.12% |
| COLD | Acc | **83.84%** | 82.68% | +1.16% |
| ToxiCN | Macro-F1 | **83.83%** (MacBERT) | 82.81% | +1.02% |

### Span Extraction (CNTP)

| Model | Overlap F1 | Character F1 | IoU | Inference Time |
|-------|-----------|-------------|-----|----------------|
| RoBERTa+ToxiTrace* | **77.90%** | **77.63%** | **61.56%** | 1m 58s |
| Qwen3-8B | 77.87% | 74.74% | 59.67% | 14m 33s |
| Gemini 2.5 Pro | 80.39% | 79.67% | 66.22% | ~1.5h |

### Ablation Study

| Config | Classification Macro-F1 | Extraction F1 | Note |
|--------|------------------------|---------------|------|
| Full model | 83.68% | 77.90% | full model |
| w/o CuSA | 82.90% | 71.96% | Weak annotation degrades to raw BiCSE; extraction recall drops sharply |
| w/o ARCL | 83.12% | 75.16% | Missing semantic contrast; both classification and extraction decline |
| w/o GCLoss | 83.36% | 65.15% | **Largest extraction F1 drop (−12.75%)** |
| RoBERTa baseline | 82.76% | 65.08% | Baseline |

### Key Findings

- GCLoss contributes far more to span extraction than ARCL (−12.75% vs −2.74%), making it the core component
- Encoder+ToxiTrace achieves span extraction F1 comparable to the strongest LLM (Qwen3-8B) in ~1/7 of the inference time
- The BiCSE algorithm significantly outperforms traditional top-k selection for extraction (RoBERTa 52.34→65.08 F1)
- Masking toxic spans causes a sharp drop in model confidence, validating the causal faithfulness of gradient attributions

## Highlights & Insights

- The idea of elevating gradient attribution from a passive analysis tool to an active training objective is novel, closing the loop from "shaping gradients during training → extracting spans at inference"
- Clever dual use of LLMs: as a weak annotation refiner in CuSA and as an adversarial reasoning generator in ARCL—neither role involves direct classification, avoiding the LLM's classification weakness
- The BiCSE bidirectional cliff scan algorithm addresses the practical problem of attribution fragmentation under Chinese character-level tokenization
- Clear efficiency advantage: encoder inference ~2 min vs LLM ~15 min, with comparable span extraction quality

## Limitations & Future Work

- Does not handle homophone substitution, pinyin obfuscation, or other "implicit toxic expressions"
- Validated only on Chinese; applicability to other character-level languages (Japanese, Korean) requires further investigation
- LoRA-based transfer to decoder LLMs yields limited results; deeper parameter-efficient gradient shaping strategies may be needed
- CuSA depends on an external LLM for annotation refinement, introducing additional cost

## Related Work & Insights

- **vs Traditional Attribution Methods (LIME/IG/Attention)**: Traditional methods provide post-hoc explanations with scattered token selections; ToxiTrace shapes gradients during training to extract contiguous spans
- **vs Direct LLM Detection**: LLMs explain well but classify poorly and slowly; ToxiTrace gives encoders the best of both worlds
- **vs CRF Sequence Labeling**: CRF requires explicit annotation for training; ToxiTrace achieves comparable results via weak supervision + gradient constraints

## Rating

- Novelty: ⭐⭐⭐⭐ Gradient attribution as a training objective + LLM debate contrastive learning is a creative combination
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-model comparisons, ablations, and faithfulness verification are comprehensive
- Writing Quality: ⭐⭐⭐⭐ Clear framework with logically sound motivation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway](toxreason_a_benchmark_for_mechanistic_chemical_toxicity_reasoning_via_adverse_ou.md)
- [\[CVPR 2026\] Learning from Synthetic Data via Provenance-Based Input Gradient Guidance](../../CVPR2026/social_computing/learning_from_synthetic_data_via_provenance-based_input_gradient_guidance.md)
- [\[ICCV 2025\] Gradient Extrapolation for Debiased Representation Learning](../../ICCV2025/social_computing/gradient_extrapolation_for_debiased_representation_learning.md)
- [\[ACL 2026\] Is this chart lying to me? Automating the detection of misleading visualizations](is_this_chart_lying_to_me_automating_the_detection_of_misleading_visualizations.md)
- [\[AAAI 2026\] Argumentative Debates for Transparent Bias Detection](../../AAAI2026/social_computing/argumentative_debates_for_transparent_bias_detection_technic.md)

</div>

<!-- RELATED:END -->
