---
title: >-
  [Paper Note] MIDB: Multilingual Instruction Data Booster for Enhancing Cultural Equality in Multilingual Instruction Synthesis
description: >-
  [AAAI 2026][Multilingual & Machine Translation][Multilingual instruction tuning] This paper proposes MIDB (Multilingual Instruction Data Booster), a unified model trained on 36.8k expert-annotated revision samples…
tags:
  - "AAAI 2026"
  - "Multilingual & Machine Translation"
  - "Multilingual instruction tuning"
  - "data quality enhancement"
  - "cultural fairness"
  - "machine translation correction"
  - "LLM multilingual capability"
date: 2026-05-08
content_hash: c78a6489958fa1c5
---

# MIDB: Multilingual Instruction Data Booster for Enhancing Cultural Equality in Multilingual Instruction Synthesis

**Conference**: AAAI 2026
**arXiv**: [2505.17671](https://arxiv.org/abs/2505.17671)  
**Code**: [github](https://github.com/zhaocorey/MIDB)  
**Area**: Multilingual Translation
**Keywords**: Multilingual instruction tuning, data quality enhancement, cultural fairness, machine translation correction, LLM multilingual capability

## TL;DR

This paper proposes MIDB (Multilingual Instruction Data Booster), a unified model trained on 36.8k expert-annotated revision samples, which automatically repairs content errors, machine translation defects, and localization deficiencies in multilingual synthetic instruction data, significantly improving instruction data quality across 16 languages and enhancing downstream LLM multilingual/cultural understanding capabilities.

## Background & Motivation

Current LLMs exhibit severely limited multilingual capabilities, fundamentally due to the English-centric nature of pretraining corpora (e.g., non-English data accounts for only ~2% in LLaMA-2). Multilingual instruction tuning (IT) is the prevailing approach to mitigate this issue, yet acquiring high-quality multilingual instruction data faces three core challenges:

**High cost of human annotation**: Annotation costs in multilingual settings are several times higher than monolingual settings, particularly for low-resource languages.

**Quality risks in synthetic data**: The mainstream practice of translating English synthetic data (e.g., Alpaca-52k) into target languages via machine translation introduces three layers of problems:
   - Content errors in the English source data caused by LLM hallucinations (factual inaccuracies, logical inconsistencies, etc.)
   - Translation defects introduced by machine translation (~30% error rate, Lai et al. 2024)
   - Severely insufficient cultural localization (e.g., Greek-translated content still reflects American cultural contexts)

**Cultural inequality**: English-centric data causes LLMs to exhibit systematic biases toward non-English cultures; a Stanford report identifies this as a form of "digital divide."

Existing English instruction data quality improvement methods (e.g., CoachLM) cannot be directly transferred to multilingual settings, as multilingual data presents additional challenges in translation quality and cultural adaptation.

## Method

### Overall Architecture

The core mechanism of MIDB is as follows: language experts first construct a multilingual revision dataset (MEB); this dataset is then used to train a unified booster model (MIDB); finally, MIDB is applied to automatically repair large-scale multilingual synthetic data. The pipeline consists of three stages: data construction, model training, and inference application.

### Key Designs

#### 1. **MEB Dataset Construction**: Human Expert-Driven Multilingual Revision

**Data sources and expert team**: Language experts with an average of 6.5+ years of experience are recruited from international enterprise language service centers, with expertise in translation, localization, and editing. 23 experts are responsible for MEB dataset construction, 20 for benchmark localization, and 7 for human evaluation, with strict task separation to avoid evaluation bias.

**Quality issue taxonomy and revision standards** (Table 1):

| Category | Proportion | Revision Standards |
|---|---|---|
| Content Enhancement | 22.9% | Contextual relevance, pertinence, feasibility, timeliness, humanization, comprehensiveness, richness, correctness, readability, safety |
| Translation Correction | 24.4% | Fluency, grammar, translation elegance, omission, spelling, mistranslation |
| Localization | 52.7% | Cultural localization, geocultural terminology correction, ideological localization, expression localization |

**Four innovative dimensions of localization standards**:
- **Cultural relevance**: Adapting instruction pairs to local culture (music, film, cuisine, etc.)
- **Geocultural terminology**: The same entity may have different names across regions (e.g., the Himalayas vs. Mount Everest)
- **Ideological localization**: Differences in religion, history, and media necessitate entirely different responses to the same question
- **Localized expressions**: Replacing literal translations with locally habitual expressions to preserve linguistic character

The dataset ultimately covers 16 languages (including 4 low-resource languages), requiring 485+ person-days of effort, yielding 36.8k revised samples (~2.3k pairs per language).

#### 2. **MIDB Training Design**: Joint Multilingual Optimization

**Training sample construction**: Each training sample consists of a Prompt (revision instruction), Input (concatenated text of the original low-quality instruction pair), and Output (the high-quality instruction pair revised by experts).

**Joint training objective**: A unified model covering all 16 languages is trained with the following optimization objective:

$$\theta_m = \arg\max_\theta \sum_{i \in [1,16]} \sum_{x_j \in C_i} \log P(y_j | x_j; \theta)$$

where $C_i$ is the training subset for language $i$. A key design is the introduction of a **quality control coefficient $\alpha$**: only the top-α% samples by edit distance are used (default 30%), as high-edit samples contain richer learning patterns.

**Model configuration**: LLaMA3.1-8B-Instruct is used as the backbone with LoRA (rank=64) for parameter-efficient fine-tuning, trained for 3 epochs with a learning rate of $4 \times 10^{-4}$ and a global batch size of 128.

#### 3. **Multilingual Evaluation Benchmark Construction**

Multilingual versions of existing English benchmarks produced via direct translation suffer from the same translation defects. The authors engage 20 professional translators over 175 person-days to professionally translate AlpacaEval and MT-Bench into 16 language versions (AlpacaEval-16L and MT-Bench-16L), and additionally adopt the BLEnD cultural understanding benchmark.

### Loss & Training

- Standard autoregressive language modeling loss based on cross-entropy
- Parameter-efficient fine-tuning via LoRA, updating only a small fraction of parameters
- Quality control filtering: samples ranked by edit distance, retaining the top-30% with highest revision magnitude
- Beam size set to 1 during inference

## Key Experimental Results

### Main Results

**Data quality evaluation (LLM-as-Judge)**: 520 samples are randomly drawn per language across all 16 languages; the win rate of MIDB-boosted data significantly exceeds its loss rate across all languages (e.g., Portuguese win rate 46% vs. loss rate 9%).

**Model performance evaluation**: Alpaca-MIDB vs. Alpaca-Original

| Metric | Benchmark | Result |
|---|---|---|
| AlpacaEval-16L | Average performance | Alpaca-MIDB maintains consistent advantage |
| MT-Bench-16L R1 | Low-resource languages (Thai, Greek) | Significantly higher scores |
| MT-Bench-16L R2 | Low-resource languages | Significantly higher scores |
| BLEnD cultural understanding | 5 non-English cultures | Accuracy improves by 19.5% |

**Human evaluation results (Winning Score, >1 = MIDB wins)**:

| Language | AlpacaEval | MT-Bench R1 | MT-Bench R2 |
|---|---|---|---|
| French | 1.68 | 1.56 | 1.52 |
| Greek | 1.56 | 1.46 | 1.28 |
| Japanese | 1.72 | 1.68 | 1.40 |
| Korean | 1.36 | 1.38 | 1.38 |
| Portuguese | 1.62 | 1.84 | 1.88 |
| Russian | 1.68 | 1.72 | 1.52 |

### Ablation Study

**Cultural understanding improvement (BLEnD benchmark)**:

| Language | Original | MIDB-Boosted | Gain |
|---|---|---|---|
| Arabic | 15.03 | 16.85 | +12.1% |
| Greek | 18.72 | 22.03 | +17.7% |
| Spanish | 25.00 | 28.49 | +14.0% |
| Indonesian | 20.62 | 25.30 | +22.7% |
| Korean | 18.50 | 24.19 | +30.8% |

**Sensitivity analysis**:
- **Backbone model**: LLaMA3.1-8B significantly outperforms Qwen2.5-7B and Qwen3-8B (stronger multilingual capability)
- **Quality control coefficient α**: α=30% yields the best performance; incorporating more low-edit samples degrades performance
- **OOD generalization**: Significant quality improvements are maintained on Dolly-15k (human-collected, out-of-distribution data)

### Key Findings

1. MIDB's improvement is more pronounced in human evaluation than in LLM-based evaluation, as humans can perceive subtle yet important improvements such as "humanized tone" and "culturally adapted expressions."
2. Localization revisions account for over 52% of all edits, indicating that cultural adaptation represents the largest quality gap in multilingual data.
3. Using only ~5% of the data volume as human revisions (relative to the full Alpaca dataset) is sufficient to achieve significant quality improvement across the entire dataset.
4. Vietnamese performance improves by 25.9% and Korean cultural understanding by 30.8%, demonstrating particular benefits for low-resource languages.

## Highlights & Insights

- **Precise problem formulation**: Multilingual instruction data quality is decomposed into three dimensions—content, translation, and localization—with clearly defined quality standards.
- **Human-machine collaboration paradigm**: A small volume of high-quality human revisions drives automated large-scale quality improvement, balancing cost and effectiveness.
- **Social impact orientation**: The paper explicitly articulates its societal significance in bridging the digital divide and mitigating cultural inequality.
- **Unified model design**: A single MIDB model handles 16 languages simultaneously, reducing deployment costs and facilitating cross-lingual knowledge transfer.

## Limitations & Future Work

1. Coverage is limited to 16 languages, whereas thousands of languages exist globally; expansion is constrained by expert resource availability.
2. The human-annotated training set construction approach is difficult to scale; crowdsourcing or iterative self-training may be considered.
3. Effectiveness on complex tasks such as advanced reasoning and mathematical computation has not been validated.
4. Data quality evaluation relies primarily on LLM-as-Judge, which is subject to known issues such as positional bias.
5. Differences in the backbone model's inherent multilingual capabilities affect MIDB's performance (e.g., larger variance for certain languages).

## Related Work & Insights

- **CoachLM** (Liu et al., 2024): An English instruction revision framework and the direct inspiration for MIDB; however, CoachLM addresses English only.
- **Self-Instruct / Alpaca**: Foundational work in instruction synthesis; MIDB directly addresses the data quality issues of these pipelines.
- **CulFiT** (Feng et al., 2025): A concurrent work that enhances cultural understanding by synthesizing culturally relevant data and translation; MIDB focuses more on fine-grained repair.
- Insight: Multilingual LLM evaluation benchmarks themselves require high-quality localization—an observation with broad implications for the field.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Data quality enhancement in the multilingual dimension addresses an important and underexplored problem; the proposed localization standards are valuable contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Coverage of 16 languages, automatic and human evaluation, OOD testing, and sensitivity analysis constitute a highly comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear and social impact is well articulated, though some details are somewhat redundant.
- **Value**: ⭐⭐⭐⭐⭐ — Directly applicable to improving training data quality for any multilingual LLM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](../../NeurIPS2025/multilingual_mt/xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)
- [\[ACL 2026\] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models](../../ACL2026/multilingual_mt/exploring_continual_fine-tuning_for_enhancing_language_ability_in_large_language.md)
- [\[NeurIPS 2025\] Enhancing Multilingual LLM Pretraining with Model-Based Data Selection](../../NeurIPS2025/multilingual_mt/enhancing_multilingual_llm_pretraining_with_model-based_data_selection.md)
- [\[ACL 2026\] DFKI-MLT at SemEval-2026 TASK 7: Steering Multilingual Models Towards Cultural Knowledge](../../ACL2026/multilingual_mt/dfki-mlt_at_semeval-2026_task_7_steering_multilingual_models_towards_cultural_kn.md)
- [\[AAAI 2026\] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)

</div>

<!-- RELATED:END -->
