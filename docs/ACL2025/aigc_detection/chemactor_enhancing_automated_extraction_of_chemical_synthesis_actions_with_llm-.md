---
title: >-
  [Paper Note] ChemActor: Enhancing Automated Extraction of Chemical Synthesis Actions with LLM-Generated Data
description: >-
  [ACL 2025][AIGC Detection][Chemical synthesis action extraction] This paper proposes ChemActor, a fully fine-tuned LLM chemical executor, which addresses the data scarcity issue in chemical synthesis action extraction through a sequential LLM-generated data framework and a distribution divergence-based data selection module, outperforming baseline models by 10% on R2D and D2A tasks.
tags:
  - "ACL 2025"
  - "AIGC Detection"
  - "Chemical synthesis action extraction"
  - "LLM data generation"
  - "distribution divergence data filtering"
  - "circle review metric"
  - "reaction-to-description conversion"
date: 2026-05-08
content_hash: 16bb43a95d24d0d9
---

# ChemActor: Enhancing Automated Extraction of Chemical Synthesis Actions with LLM-Generated Data

**Conference**: ACL 2025  
**arXiv**: [2506.23520](https://arxiv.org/abs/2506.23520)  
**Code**: [https://github.com/Zhanghahah/ChemActor](https://github.com/Zhanghahah/ChemActor)  
**Area**: AIGC Detection  
**Keywords**: Chemical synthesis action extraction, LLM data generation, distribution divergence data filtering, circle review metric, reaction-to-description conversion

## TL;DR

This paper proposes ChemActor, a fully fine-tuned LLM chemical executor, which addresses the data scarcity issue in chemical synthesis action extraction through a sequential LLM-generated data framework and a distribution divergence-based data selection module, outperforming baseline models by 10% on R2D and D2A tasks.

## Background & Motivation

**Background**: With the rise of robotic synthesis in the field of organic chemistry, automatically extracting chemical experimental steps from literature and converting them into machine-executable action sequences has become increasingly important. This process involves conversion in two directions: Reaction-to-Description (R2D)—generating experimental descriptions from structured reaction information, and Description-to-Action (D2A)—extracting structured action sequences from unstructured experimental descriptions.

**Limitations of Prior Work**: Chemical language itself is highly ambiguous—the same operation can have multiple natural language expressions, and the same expression can refer to different operations under different chemical contexts. Furthermore, high-quality human-annotated data is extremely scarce and costly to annotate, as annotators must possess both chemistry expertise and natural language understanding capabilities. Existing annotated datasets are small in scale and vary in quality, severely limiting the performance of extraction models.

**Key Challenge**: There is a sharp contradiction between the demand for high-quality annotated data and the high cost of human annotation. Simply fine-tuning LLMs with a small amount of human-annotated data yields limited effectiveness, while directly generating large amounts of data using LLMs faces quality assurance issues.

**Goal**: To design a systematic LLM data generation framework that can highly efficiently generate a large volume of high-quality chemical synthesis action training data starting from a small amount of seed data, and use this data to train a specialized chemical action extraction model.

**Key Insight**: The authors observe that although general LLMs (such as GPT-4) possess some chemical knowledge, their accuracy is insufficient when directly applied to professional chemical action extraction. However, if the generation capabilities of general LLMs can be utilized to augment training data, and his data is used to fine-tune a specialized model, the coverage of general LLMs can be combined with the precision of specialized models.

**Core Idea**: Utilizing general LLMs to generate chemical experimental action data from single molecule inputs, filtering high-quality data through a distribution divergence selection mechanism, and fine-tuning a specialized LLM with this data to complete bidirectionally converted chemical synthesis actions.

## Method

### Overall Architecture

The ChemActor framework consists of three main stages: (1) Seed Data Preparation—extracting high-quality samples from existing small-scale annotated datasets to serve as seeds; (2) LLM Data Generation—using general LLMs to generate new synthesis action sequences starting from molecular inputs, and filtering out high-quality data consistent with the target distribution through a distribution divergence selection module; (3) Model Fine-Tuning—jointly fine-tuning an LLM using the filtered LLM-generated data and the original seed data to make it a specialized chemical action executor, ChemActor.

### Key Designs

1. **Sequential LLM-Generated Data Framework**:

    - **Function**: Systematically utilizing general LLMs to generate chemical synthesis action annotated data.
    - **Mechanism**: Given the SMILES representation of a target molecule as input, general LLMs (such as GPT-4) are used through meticulously designed prompt templates to generate the complete synthetic experimental description of the molecule and the corresponding machine-executable action sequence. The generation process is divided into multiple rounds: first generating the experimental description (R2D direction), then generating the action sequence from the description (D2A direction), and finally conducting cross-validation to ensure consistency.
    - **Design Motivation**: Generating starting from molecules ensures that the data covers a broad chemical space, while sequential multi-step generation and cross-validation can enhance internal consistency.

2. **Distribution Divergence-based Data Selection**:

    - **Function**: Filtering high-quality samples from a large pool of candidate data generated by LLMs.
    - **Mechanism**: Computing the distribution divergence (such as KL divergence or JS divergence) between the LLM-generated data and the real human-annotated data, selecting generated samples that are closer to the real data in distribution characteristics. Specifically, the deviation of each generated sample from the real data distribution is measured across multiple dimensions, including lexical distribution, action type distribution, and sequence length distribution, prioritizing samples with low deviation.
    - **Design Motivation**: Data generated by LLMs inevitably contains noise and unreasonable samples; using all of it directly would introduce distribution shifts. Distribution divergence filtering achieves the optimal balance between data volume and quality.

3. **Multi-round LLMs Circle Review Metric**:

    - **Function**: Providing a new evaluation metric to measure the model's deep understanding of chemical experimental workflows.
    - **Mechanism**: Letting the model perform multiple rounds of R2D and D2A conversions ($Description \to Action \to Description \to Action \dots$) for the same synthesis task, and checking whether the information remains consistent after multiple rounds of conversions. If the model truly understands the chemical experimental workflow, multi-round conversions should maintain semantic stability; if it is just surface-level matching, information will gradually degrade across multiple rounds of conversions.
    - **Design Motivation**: Traditional single-pass evaluations (such as BLEU, ROUGE) fail to reflect whether the model truly "understands" the chemical process. The Circle Review metric measures deeper understanding capability by testing multi-round consistency.

### Loss & Training

ChemActor uses the standard language modeling cross-entropy loss for fine-tuning. The training data consists of a mixture of filtered LLM-generated data and original seed data in a certain ratio. Full parameter fine-tuning is adopted instead of parameter-efficient methods like LoRA to maximize the performance of the specialized model.

## Key Experimental Results

### Main Results

| Model/Method | D2A (Action F1) | R2D (BLEU-4) | R2D (ROUGE-L) | Circle Review |
|----------|-------------|-------------|---------------|---------------|
| GPT-4 (zero-shot) | 52.3 | 18.5 | 35.2 | 42.1|
| GPT-3.5 (zero-shot) | 45.6 | 15.2 | 30.8 | 35.4 |
| Baseline Fine-tuned Model | 65.8 | 28.3 | 48.5 | 55.2 |
| + Unfiltered LLM Data | 69.2 | 31.5 | 52.1 | 58.6 |
| **ChemActor** | **75.4** | **35.8** | **56.3** | **65.8** |
| ChemActor Gain | +10% | +7.5 | +7.8 | +10.6 |

### Ablation Study

| Configuration | D2A (F1) | R2D (BLEU) | Description |
|------|---------|-----------|------|
| Full ChemActor | 75.4 | 35.8 | Full model |
| w/o Distribution Divergence Selection | 69.2 | 31.5 | Using all LLM data directly without selection, drops 6.2% |
| w/o LLM-generated Data | 65.8 | 28.3 | Fine-tuning with only seed data, drops 9.6% |
| Random Selection (Same Data Volume) | 71.5 | 33.2 | Randomly selecting the same volume of data, drops 3.9% |
| Only LLM Data (No Seeds) | 67.8 | 30.1 | No mixture of seed data, drops 7.6% |

### Key Findings

- The addition of LLM-generated data significantly boosts model performance, but using it directly without screening yields limited effects; distribution divergence filtering is critical.
- ChemActor substantially outperforms baselines in both D2A and R2D directions, demonstrating the effectiveness of the framework.
- The Circle Review metric positively correlates with the model's performance in single-pass evaluations, but can reveal finer-grained differences in understanding capability.
- General LLMs (GPT-4) in zero-shot settings perform far worse than fine-tuned models, indicating that chemical action extraction indeed requires specialized training.
- The mixing ratio of seed data and LLM-generated data has a significant impact on final performance, with the optimal ratio being around 1:3.

## Highlights & Insights

- **Direct training data generation from molecules**: Through the $SMILES \to \text{experimental description} \to \text{action sequence}$ generation chain, the chemical knowledge of general LLMs is ingeniously leveraged to expand the specialized dataset. This strategy can be transferred to data scarcity problems in other scientific domains.
- **Distribution divergence filtering**: The idea of "pruning" LLM-generated data using distribution divergence is simple and effective. Compared to methods that grade each data point independently, filtering from the distribution level offers better statistical guarantees.
- **Circle Review metric**: Evaluating deep understanding by testing consistency through multi-round bidirectional conversions. This metric design idea can be generalized to any task requiring bidirectional conversion (such as translation, summarization, etc.).

## Limitations & Future Work

- The sequential generation framework highly depends on the quality of the general LLM's chemical knowledge; for rare reaction types unfamiliar to the LLM, it may generate poor-quality data.
- Distribution divergence filtering assumes that LLM-generated data should share a consistent distribution with human-annotated data, but real-world chemical reaction distributions might be much broader.
- Currently only evaluated in the field of organic chemical synthesis; whether it applies to other chemical sub-domains (such as inorganic chemistry, biochemistry) remains to be validated.
- In the future, combining Retrieval-Augmented Generation (RAG) can be explored to improve the accuracy of chemical data generated by LLMs.

## Related Work & Insights

- **vs Ord-RL (Prev. SOTA)**: Ord-RL uses reinforcement learning to optimize chemical action extraction, but is constrained by the scale of annotated data. ChemActor breaks through the data bottleneck via LLM data augmentation.
- **vs Direct GPT-4 Application**: Although GPT-4 has chemical knowledge, its zero-shot extraction performance is poor, demonstrating the necessity of specialized fine-tuned models. The innovation of ChemActor lies in "using general models to generate data to train specialized models".
- **vs Self-Instruct**: Self-Instruct also leverages LLMs to generate training data but lacks domain-specific quality filtering mechanisms. ChemActor's distribution divergence selection is the key differentiating factor.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of the LLM data generation framework and distribution divergence filtering is creative, and the Circle Review metric is a novel evaluation approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual-direction evaluations for both R2D and D2A are comprehensive, and ablation studies thoroughly validate the contributions of each component.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, the method description is systematic, and the experiments are organized logically.
- Value: ⭐⭐⭐ Provides significant promotion to the field of chemical information extraction, though the application scenario remains rather specialized and niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Data Provenance for Image Auto-Regressive Generation](../../ICLR2026/aigc_detection/data_provenance_for_image_auto-regressive_generation.md)
- [\[ACL 2025\] Learning to Rewrite: Generalized LLM-Generated Text Detection](learning_to_rewrite_generalized_llm-generated_text_detection.md)
- [\[ICML 2026\] Dissect and Prune: Enhancing Robustness in AI-Generated Image Detection](../../ICML2026/aigc_detection/dissect_and_prune_enhancing_robustness_in_ai-generated_image_detection.md)
- [\[ACL 2025\] Low-Perplexity LLM-Generated Sequences and Where To Find Them](low-perplexity_llm-generated_sequences_and_where_to_find_them.md)
- [\[ACL 2025\] Comparing LLM-generated and human-authored news text using formal syntactic theory](llm_vs_human_formal_syntax.md)

</div>

<!-- RELATED:END -->
