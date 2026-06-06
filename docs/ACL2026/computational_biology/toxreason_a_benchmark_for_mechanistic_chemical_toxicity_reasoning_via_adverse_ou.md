---
title: >-
  [Paper Note] ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway
description: >-
  [ACL 2026][Computational Biology][Toxicity Reasoning] This paper proposes ToxReason, a mechanistic chemical toxicity reasoning benchmark based on the Adverse Outcome Pathway (AOP) framework. It integrates drug-target exp…
tags:
  - "ACL 2026"
  - "Computational Biology"
  - "Toxicity Reasoning"
  - "Adverse Outcome Pathway"
  - "Benchmark"
  - "Reinforcement Learning"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: b1a4cc13fc88e613
---

# ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway

**Conference**: ACL 2026  
**arXiv**: [2604.06264](https://arxiv.org/abs/2604.06264)  
**Code**: None  
**Area**: Computational Biology  
**Keywords**: Toxicity Reasoning, Adverse Outcome Pathway, Benchmark, Reinforcement Learning, LLM Evaluation

## TL;DR

This paper proposes ToxReason, a mechanistic chemical toxicity reasoning benchmark based on the Adverse Outcome Pathway (AOP) framework. It integrates drug-target experimental data with toxicity labels, requiring models to reason from molecular initiating events to organ-level adverse outcomes. A 4B model trained via GRPO reinforcement learning outperforms larger models such as GPT-5 in both toxicity prediction (F1 71.4%) and reasoning quality.

## Background & Motivation

**Background**: LLMs have been applied to molecular reasoning and toxicity prediction tasks. Existing benchmarks (e.g., Tox21, ClinTox) focus primarily on predicting structure-property relationships, treating toxicity as a simple classification task.

**Limitations of Prior Work**: Toxicity inherently stems from complex biological mechanisms (molecular targets → cellular events → organ responses), rather than being determined solely by chemical structure. LLMs can generate fluent but biologically unreliable explanations, meaning that high prediction accuracy does not equate to reliable reasoning. Reasoning in existing datasets (e.g., UniTox) is based on clinical observations rather than causal mechanistic pathways.

**Key Challenge**: There is a significant disconnect between prediction performance and reasoning quality—models may "guess the answer correctly" but provide incorrect mechanistic explanations, which is unacceptable in high-risk scenarios such as drug safety assessment.

**Goal**: Construct a benchmark to evaluate mechanistic toxicity reasoning, requiring models to perform step-by-step causal reasoning from Molecular Initiating Events (MIE) to Adverse Outcomes (AO), and explore training strategies to enhance reasoning capabilities.

**Key Insight**: The AOP framework in toxicology naturally describes the causal chain from MIE → Key Events (KE) → AO, which aligns closely with the multi-step reasoning paradigm in NLP.

**Core Idea**: Use AOP causal chains as the ground truth for toxicity reasoning to build an evaluation benchmark and simultaneously improve prediction and reasoning capabilities through reasoning-aware training.

## Method

### Overall Architecture

The construction process of ToxReason includes three steps: (1) Screening 23 AOPs and 25 MIE targets related to liver, heart, and kidney toxicity from AOP-Wiki; (2) Integrating CTD disease-chemical associations with ChEMBL experimental activity data, inferring MIEs through structural similarity; (3) Constructing training and test sets for model learning and rigorous evaluation, respectively. Evaluation is conducted regarding both toxicity prediction (F1) and reasoning quality (LLM-as-a-Judge four-dimensional scoring).

### Key Designs

1.  **AOP Selection and Chemical-AOP Association Derivation**:
    - **Function**: Construct reasoning annotation data based on biological causal mechanisms.
    - **Mechanism**: Filter AOPs related to liver/heart/kidney toxicity from AOP-Wiki and treat their AOs as disease concepts to retrieve associated chemicals in CTD. Extract EC50/IC50 activity data for MIE targets from ChEMBL (activity < 10,000nM is considered active), and infer the MIE direction (activation/inhibition) for candidate chemicals via structural similarity majority voting.
    - **Design Motivation**: Address the MIE inference problem when direct experimental data is unavailable by performing evidence aggregation through known activities of similar molecules.

2.  **Training and Test Set Construction**:
    - **Function**: Design separated data to support learning and evaluation.
    - **Mechanism**: The training set is divided into two complementary sets: MIE-matched (satisfying only MIE conditions with Dice similarity $\ge 0.5$) and MIE-AO-matched (satisfying both MIE and AO). The test set uses strictly curated associations and chemicals with identical structures to ensure no leakage.
    - **Design Motivation**: MIE-matched data expands coverage to help learn initial patterns, while MIE-AO-matched data encourages reasoning across molecular interactions and downstream toxicity outcomes.

3.  **GRPO Reinforcement Learning Training Framework**:
    - **Function**: Explicitly optimize the joint objective of toxicity prediction and mechanistic reasoning.
    - **Mechanism**: A two-stage training approach is adopted—first SFT to align task formats, followed by the GRPO framework to optimize causal consistency and biological faithfulness of AOP reasoning.
    - **Design Motivation**: SFT only learns output formats but does not improve reasoning quality; RL guides the model to generate reasoning chains aligned with AOP paths through explicit reward signals.

### Loss & Training

A GRPO (Group Relative Policy Optimization) framework is employed, where reward signals integrate toxicity prediction accuracy and the alignment between the reasoning chain and the AOP. Parameter-efficient fine-tuning is performed using LoRA.

## Key Experimental Results

### Main Results

| Model | Kidney Tox F1 | Heart Tox F1 | Liver Tox F1 | Average F1 | Reasoning Overall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-5 | 56.4 | 72.7 | 65.0 | 64.7 | 5.420 |
| GPT-5.1 | 50.3 | 71.2 | 58.9 | 60.1 | 5.523 |
| o3 | 60.0 | 72.5 | 58.8 | 63.8 | 5.326 |
| DeepSeek-R1-70B | 59.1 | 78.5 | 59.6 | 65.7 | 4.487 |
| Qwen3-4B (base) | 56.9 | 71.1 | 57.3 | 61.8 | 4.523 |
| ToxReason-4B-SFT | 57.9 | 74.3 | 57.4 | 63.2 | 4.554 |
| **ToxReason-4B-GRPO** | **73.4** | 72.7 | **68.2** | **71.4** | **5.642** |

### Ablation Study

| Configuration | Average F1 | Reasoning Overall | Description |
| :--- | :--- | :--- | :--- |
| Qwen3-4B base | 61.8 | 4.523 | Base Model |
| + ICL 1-shot | 68.8 | 5.373 | Best Few-shot |
| + ICL 2-shot | 59.1 | 4.373 | More examples introduce noise |
| + SFT | 63.2 | 4.554 | Limited improvement from fine-tuning |
| + GRPO | 71.4 | 5.642 | Significant improvement from RL |

### Key Findings

- A significant disconnect exists between prediction performance and reasoning quality: GPT-5.1 achieves the best reasoning but the poorest prediction (60.1%), while DeepSeek-R1 achieves the best prediction but poorer reasoning.
- SFT contributes minimally to reasoning quality, whereas GRPO significantly improves both prediction (+9.6%) and reasoning (+1.1 points).
- ICL is most effective at 1-shot; increasing the number of shots introduces noise that degrades performance.
- NW alignment scores are highly correlated with LLM-as-a-Judge ratings (Spearman $\rho=0.837$), validating the reliability of the evaluation method.

## Highlights & Insights

- **Discovery of Prediction-Reasoning Decoupling**: Reveals that LLMs can perform well in toxicity prediction while having entirely incorrect reasoning mechanisms, which serves as an important warning for safety-critical applications.
- **4B Model Outperforming GPT-5**: Through GRPO reasoning-aware training, a model with 4B parameters surpasses closed-source large models in both prediction and reasoning, proving the value of explicit reasoning optimization.
- **Mapping AOP Framework to NLP Multi-step Reasoning**: Transforms toxicological causal chains into evaluable NLP reasoning tasks; this approach can be generalized to the mechanistic reasoning evaluation of other scientific fields.

## Limitations & Future Work

- Coverage is limited to liver, heart, and kidney organ toxicities due to the scope of AOP-Wiki.
- MIE inference is based on structurally similar molecules rather than direct prediction from molecular structure, which limits applicability to entirely novel chemicals.
- LLM-as-a-Judge evaluation is inherently subjective; although validated with the NW algorithm, it should still be considered a relative measure.
- Future work could extend to more organ systems and more complex AOP networks.

## Related Work & Insights

- **vs CoTox**: CoTox improves prediction through CoT but does not evaluate whether reasoning aligns with causal paths, whereas ToxReason treats reasoning evaluation as a core objective.
- **vs Tox21/ClinTox**: Traditional toxicity benchmarks only perform outcome prediction, while ToxReason requires the model to explain "why it is toxic."
- **vs UniTox**: UniTox provides explanations based on clinical observations, whereas ToxReason requires step-by-step reasoning based on AOP causal mechanisms.

## Rating

- Novelty: ⭐⭐⭐⭐ First benchmark to systematically evaluate LLM toxicity mechanistic reasoning; the finding of prediction-reasoning decoupling is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various open/closed-source models and learning strategies, though limited to three organ toxicities.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed AOP background introduction.
- Value: ⭐⭐⭐⭐ Practical significance for drug safety and the field of trustworthy AI reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](../../NeurIPS2025/computational_biology/fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)
- [\[ACL 2026\] AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling](aroma_augmented_reasoning_over_a_multimodal_architecture_for_virtual_cell_geneti.md)
- [\[ICML 2026\] TadA-Bench: A Million-Variant Benchmark for Future-Round Discovery Toward Agentic Protein Engineering](../../ICML2026/computational_biology/tada-bench_a_million-variant_benchmark_for_future-round_discovery_toward_agentic.md)
- [\[ICLR 2026\] mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules](../../ICLR2026/computational_biology/mclm_a_modular_chemical_language_model_that_generates_functional_and_makeable_mo.md)
- [\[ICLR 2026\] HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction](../../ICLR2026/computational_biology/histoprism_unlocking_functional_pathway_analysis_from_pan-cancer_histology_via_g.md)

</div>

<!-- RELATED:END -->
