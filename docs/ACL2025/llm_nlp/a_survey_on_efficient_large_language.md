---
title: >-
  [Paper Note] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives
description: >-
  [ACL 2025][LLM (Other)][Data-efficient training] This paper presents the first systematic survey framework for "data-efficient LLM post-training", categorizing methods into five major areas: data selection, data quality enhancement, synthetic data generation, data distillation & compression, and self-evolving data ecosystems, thereby constructing a comprehensive "data value flywheel" system.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Data-efficient training"
  - "Post-training"
  - "Data selection"
  - "Synthetic data"
  - "Knowledge distillation"
  - "Self-evolving data"
date: 2026-05-08
content_hash: 76ad1baf340ef26a
---

# A Survey on Efficient Large Language Model Training: From Data-centric Perspectives

**Conference**: ACL 2025  
**arXiv**: [2510.25817](https://arxiv.org/abs/2510.25817)  
**Code**: [GitHub](https://github.com/luo-junyu/Awesome-Data-Efficient-LLM)  
**Area**: LLM Training / Data Efficiency  
**Keywords**: Data-efficient training, Post-training, Data selection, Synthetic data, Knowledge distillation, Self-evolving data  

## TL;DR

This paper presents the first systematic survey framework for "data-efficient LLM post-training", categorizing methods into five major areas: data selection, data quality enhancement, synthetic data generation, data distillation & compression, and self-evolving data ecosystems, thereby constructing a comprehensive "data value flywheel" system.

## Background & Motivation

- **Core Problem**: LLM post-training (SFT, RLHF, etc.) is a critical phase for unlocking model generalization capabilities, yet it faces severe data bottlenecks: high manual annotation costs, diminishing marginal returns on data scaling, and static data failing to adapt to knowledge evolution.
- **The Necessity of Data-Efficient Training**: The successful cases of models like DeepSeek-R1 achieving data-efficient post-training via reinforcement learning further demonstrate the importance of data efficiency. The core rationale is that breaking efficiency bottlenecks requires establishing value extraction mechanisms across the entire data lifecycle, rather than merely scaling up data volume.
- **Research Gap**: Although surveys on specific aspects like data selection or synthetic data exist, there is a lack of a systematic survey from the unified perspective of data efficiency.

## Method

### Overall Architecture: Data Value Flywheel

Five key components form a closed loop: **Data Selection** (filtering high-value subsets) $\rightarrow$ **Quality Enhancement** (improving utility of existing data) $\rightarrow$ **Synthetic Generation** (creating new training data) $\rightarrow$ **Distillation & Compression** (extracting core knowledge) $\rightarrow$ **Self-Evolving Ecosystem** (building self-evolutionary mechanisms). These five components complement each other: selection filters quality data, enhancement improves utility, generation expands coverage, distillation condenses knowledge, and self-evolution drives continuous improvement.

### Key Designs

#### 1. Data Selection

- **Static Filtering**: Alpagasus achieves comparable performance using only 17% of the data; offline selection based on quality/information-theoretic metrics.
- **Dynamic Selection**: Active Instruction Tuning prioritizes high-value samples based on uncertainty; LESS utilizes low-rank gradient features for optimizer-aware similarity search.
- **Agent Strategies**: CLUES multi-model voting mechanism; DATA ADVISOR red-team filtering.
- **Annotation Efficiency**: SELF-INSTRUCT autonomously generates instruction data; LLMaAA employs LLMs as annotators.

#### 2. Data Quality Enhancement

- **Semantic Rewriting**: CoachLM automatically modifies complex instructions to reduce ambiguity; LLM2LLM iteratively improves low-confidence samples.
- **Toxicity Control**: ToxiCraft generates adversarial datasets to stress-test model safety boundaries.
- **Distribution Stabilization**: Synthetic oversampling addresses class imbalance; RobustFT utilizes multi-expert collaborative noise detection and entropy-based data selection.

#### 3. Synthetic Data Generation

- **Instruction-Driven**: SynPO generates preference pairs for alignment (ROUGE-L +12%); Magpie achieves template-free instruction generation.
- **Knowledge-Guided**: Incorporating knowledge graphs/structured knowledge to ensure factual accuracy; hybrid generation reduces API costs by 70%.
- **Adversarial Generation**: Probing model vulnerabilities to enhance robustness.

#### 4. Distillation & Compression

- **Model Distillation**: Impossible Distillation creates high-quality student models from low-quality teachers; cross-tokenizer distillation.
- **Data Distillation**: LLMLingua-2 achieves prompt compression through token-level distillation.
- **Joint Compression**: LLaMA-7B is compressed to 2.8B parameters with minimal performance loss.

#### 5. Self-Evolving Data Ecosystem

- **Self-Iterative Optimization**: Self-Rewarding, Self-Refine — models autonomously improve using their own outputs.
- **Dynamic Evaluation Feedback**: Multi-agent real-time adjustment of evaluation and optimization.
- **LLM-as-a-Judge**: Self-evaluation paradigm replacing external evaluation.

### Method Comparison

| Category | Data Dependency | Computational Cost | Model Dependency | Data Value Mining |
|------|:---:|:---:|:---:|:---:|
| Data Selection | ++ | + | + | +++ |
| Quality Enhancement | ++ | ++ | ++ | ++ |
| Synthetic Generation | + | +++ | +++ | + |
| Distillation & Compression | + | + | +++ | +++ |
| Self-Evolution | + | +++ | +++ | +++ |

## Key Experimental Results

This is a survey paper with no new experiments. However, it systematically compiles the key experimental findings from various sub-domains:

### Summary of Representative Method Performance

| Method | Category | Key Performance |
|------|------|------|
| Alpagasus | Data Selection | 17% data achieves comparable performance |
| SynPO | Synthetic Generation | ROUGE-L +12% |
| Hybrid Generation (Chan et al.) | Synthetic Generation | API cost reduced by 70% |
| LLaMA-7B Compression | Joint Compression | 2.8B parameters, minimal performance loss |
| Magpie | Synthetic Generation | AlpacaEval 98% accuracy |

### Key Findings

- Merely scaling up data volume yields diminishing marginal returns, necessitating a transition toward data value mining.
- The five methodologies are complementary rather than mutually exclusive; a unified framework should be established.
- Self-evolution and LLM-as-a-Judge represent critical directions for reducing human intervention.
- Domain-specific data synthesis is more effective than generation using general-purpose models.

## Highlights & Insights

- The first work to systematically survey LLM post-training from the unified perspective of data efficiency.
- Formulates the concept of the "data value flywheel" to organize disparate research lines into an organic whole.
- Presents a clear taxonomy (5 major categories $\times$ multiple sub-categories) with comprehensive coverage.
- Provides an accompanying awesome list that is continuously updated.

## Limitations & Future Work

- The field is evolving rapidly, and some emerging technologies might not be fully covered.
- The synergistic effects and interaction mechanisms among the five methodologies have not been thoroughly explored.
- Discussion on reliability and scalability remains insufficient.
- Lacks unified experimental validation across different methodologies.

## Related Work & Insights

- **Data Selection Surveys**: Wang et al. (24b) — focuses solely on the dimension of data selection.
- **Synthetic Data Surveys**: Long et al. (2024); Tan et al. (2024) — only covers synthetic generation.
- **Model Self-Feedback**: Liang et al. (24a); Pan et al. (2023) — self-evolving direction.
- **Self-Evolution Surveys**: Tao et al. (2024) — model self-evolution.
- **Training Efficiency Surveys**: Wan et al. (2023) — focuses on temporal efficiency rather than data efficiency.

## Rating

| Dimension | Rating |
|------|:---:|
| Novelty | ★★★☆☆ |
| Utility | ★★★★★ |
| Experimental Thoroughness | ★★★☆☆ |
| Writing Quality | ★★★★☆ |
| Overall Rating | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions](language_model_fine-tuning_on_scaled_survey_data_for_predicting_distributions_of.md)
- [\[ACL 2025\] Understanding Silent Data Corruption in LLM Training](understanding_silent_data_corruption_in_llm_training.md)
- [\[ACL 2025\] Training Language Model to Critique for Better Refinement](training_language_model_to_critique_for_better_refinement.md)
- [\[ACL 2025\] Personalized Generation In Large Model Era: A Survey](personalized_generation_in_large_model_era_a_survey.md)
- [\[ACL 2025\] Large Language Models in Bioinformatics: A Survey](large_language_models_in_bioinformatics_a_survey.md)

</div>

<!-- RELATED:END -->
