---
title: >-
  [Paper Note] PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis
description: >-
  [AAAI 2026][Medical NLP][Medical multimodal model] This paper proposes PulseMind, a multimodal medical diagnostic model comprising three core contributions: MediScope, a large-scale multi-turn diagnostic dialogue dataset…
tags:
  - "AAAI 2026"
  - "Medical NLP"
  - "Medical multimodal model"
  - "multi-turn diagnostic dialogue"
  - "reinforcement learning"
  - "comparative reward"
  - "clinical evaluation benchmark"
date: 2026-05-08
content_hash: e03c9fb4d9d59ebc
---

# PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis

**Conference**: AAAI 2026
**arXiv**: [2601.07344](https://arxiv.org/abs/2601.07344)  
**Code**: [GitHub](https://github.com/AQ-MedAI/PulseMind)  
**Area**: Medical Imaging
**Keywords**: Medical multimodal model, multi-turn diagnostic dialogue, reinforcement learning, comparative reward, clinical evaluation benchmark

## TL;DR

This paper proposes PulseMind, a multimodal medical diagnostic model comprising three core contributions: MediScope, a large-scale multi-turn diagnostic dialogue dataset; PulseMind Benchmark, a multi-dimensional clinical dialogue evaluation benchmark; and CRPO, a comparison-based reinforcement policy optimization method. The system achieves superior performance in real-world clinical diagnostic dialogue scenarios.

## Background & Motivation

Recent advances in vision-language models (VLMs) have driven substantial progress in multimodal understanding, spurring extensive research on medical multimodal models (e.g., LLaVA-Med, Med-GEMMA). However, existing medical VLMs primarily focus on **specialized image analysis** (e.g., dermatology, pathology slides, radiology), leaving a fundamental gap with **real-world clinical diagnosis** scenarios:

**Limitations of training data**: Most medical datasets are limited to either single-turn VQA or single imaging modalities, lacking the heterogeneous multi-source inputs and multi-turn physician–patient dialogues characteristic of real clinical practice.

**Insufficient evaluation benchmarks**: Existing medical multimodal benchmarks fail to reflect the complexity of real clinical settings—in practice, physicians must proactively elicit missing information, integrate diverse data sources (lab reports, imaging, medical records), and maintain contextual coherence across multiple interaction turns.

**Limitations of optimization methods**: Common reinforcement learning approaches (e.g., GRPO) use absolute scores as reward signals, which presents two problems in clinical dialogue settings: (a) model-based scoring is unstable and highly subjective; and (b) absolute scores often fail to discriminate subtle differences among top-performing models.

The core motivation of this paper is to **bridge the triple gap between medical VLMs and real-world clinical diagnosis**—spanning data, evaluation, and optimization methodology.

## Method

### Overall Architecture

PulseMind comprises three core components (see Figure 2):

1. **MediScope Dataset**: Large-scale multimodal diagnostic dialogue data
2. **PulseMind Benchmark**: Multi-dimensional clinical dialogue evaluation benchmark
3. **CRPO Training Framework**: Comparison-based reinforcement policy optimization

The backbone model is Qwen2.5-VL (in 72B and 32B variants), fine-tuned via LoRA (rank-64) for parameter-efficient adaptation, trained on 128 A100 GPUs.

### Key Designs

1. **MediScope Dataset Construction**

   The dataset is constructed following a rigorous four-stage pipeline:
   - **Collection**: De-identified data collected from real clinical scenarios, encompassing examination reports and multi-turn physician–patient dialogues.
   - **Anonymization**: A secondary anonymization pass using OCR and NER techniques is applied to both text and images to ensure complete removal of personally identifiable information.
   - **Expansion**: LLMs such as GPT-4o and Gemini are used to refine and augment physician responses—filtering meaningless filler utterances and supplementing clinically relevant content.
   - **Proofreading**: Comprehensive review by medical experts and licensed physicians to ensure clinical validity and ethical compliance.

   The resulting dataset contains **98,000 real multi-turn consultation dialogues** and **601,500 medical images**, spanning **10+ major clinical departments** and **200+ subspecialties**. Data types include laboratory results, examination reports, prescriptions, medical images, and surgical records. 40.9% of dialogues contain 6–10 turns, and 6% exceed 20 turns.

2. **PulseMind Benchmark**

   The benchmark consists of two subsets totaling 1,200+ samples:
   - **MedDiagnose** (237 samples): An in-house multimodal consultation set containing imaging data and expert-validated dialogues.
   - **CMtMedQA-test** (1,000 samples): An extended text-only multi-turn reasoning consultation set.

   A four-dimensional evaluation protocol is employed:
   - **Proactiveness**: Whether the model proactively elicits missing but clinically critical information.
   - **Accuracy**: Whether diagnostic suggestions are medically sound and free of factual errors.
   - **Usefulness**: The practical value of responses, including clarity and actionability.
   - **Language Quality**: Fluency, professionalism, and communicative effectiveness.

   GPT-4 serves as the automatic evaluator, with pairwise comparisons used to compute win rate as the primary metric.

3. **CRPO (Comparison-based Reinforcement Policy Optimization)**

   CRPO is the methodological centerpiece of this work. Its design motivation stems from a key observation: **humans are better at judging which of two responses is superior than at assigning absolute scores to individual responses**.

   The procedure is as follows: given a query $q$, the policy model generates $G$ candidate responses $\{o_1, \dots, o_G\}$. Each candidate $o_g$ is then compared against responses from 5 counterpart models $\{CP_1, \dots, CP_5\}$ across 4 evaluation dimensions:

   $r_{g,c,d} = \begin{cases} 1, & \text{if } o_g \succ CP_c \text{ on dimension } d \\ 0, & \text{otherwise} \end{cases}$

   The reward for a candidate response is the average over all counterpart models and dimensions:

   $R_g = \frac{1}{C \times D} \sum_{c=1}^{C} \sum_{d=1}^{D} r_{g,c,d}$

   where $C=5$ (number of counterpart models) and $D=4$ (number of evaluation dimensions). Subsequent advantage computation and the loss function follow the same formulation as GRPO.

### Loss & Training

Training proceeds in two stages:

1. **Supervised Fine-Tuning (SFT)**: Domain knowledge is first injected via training on Huatuo26M, followed by fine-tuning on MediScope combined with public datasets to unlock multimodal and multi-turn dialogue capabilities.
2. **Reinforcement Learning (CRPO)**: Comparative reward signals are used to further optimize diagnostic response quality.

The technical stack comprises HuggingFace Transformers + PEFT + DeepSpeed ZeRO-3, BF16 mixed precision, AdamW with cosine annealing, and dropout of 0.1.

## Key Experimental Results

### Main Results

| Dataset | Metric | PulseMind-72B | InternVL3-78B | Qwen2.5VL-72B | GPT-4o | o1 |
|---------|--------|--------------|---------------|---------------|--------|-----|
| VQA-RAD | Acc | **87.1** | 73.6 | 80.3 | 71.2 | 63.0 |
| PMC-VQA | Acc | **70.3** | 56.6 | 59.3 | 55.2 | 54.5 |
| SLAKE | Acc | **85.6** | 77.4 | 78.3 | 67.4 | 69.9 |
| PathVQA | Acc | **64.9** | 51.0 | 42.3 | 55.5 | 57.3 |
| MedQA | Acc | **94.8** | 93.3 | 91.3 | 55.7 | 86.6 |
| MMMU | Acc | **69.4** | 69.1 | 66.4 | 57.3 | 57.8 |

On the PulseMind Benchmark, PulseMind achieves win rates of 94% (MedDiagnose) and 73% (CMtMedQA) against GPT-4o, 89% and 83% against o1, and 54% and 72% against Gemini 2.5 Pro.

### Ablation Study

| Configuration | PulseMind-B Win Rate | MMMU | VQA-RAD | SLAKE |
|---------------|---------------------|------|---------|-------|
| Public data only | 26.4% | 67.3 | 86.6 | 84.7 |
| + MediScope | 65.2% | 68.1 | 86.9 | 85.3 |
| + RL (CRPO) | **76.0%** | **69.4** | **87.1** | **85.6** |
| GRPO instead of CRPO | 54.7% | 66.7 | 86.9 | 85.2 |

### Key Findings

1. **MediScope is critical**: Its inclusion raises the PulseMind Benchmark win rate from 26.4% to 65.2%.
2. **CRPO outperforms GRPO**: Comparative reward signals substantially outperform absolute score rewards in diagnostic dialogue (76.0% vs. 54.7%).
3. **Reliability of relative vs. absolute evaluation**: Compared against the judgments of 50 medical experts, the relative evaluation strategy achieves 86.1% agreement, versus only 51.5% for absolute evaluation.
4. Under absolute scoring, all models cluster within a narrow range of 4.01–4.35 (on a 5-point scale), rendering discrimination nearly impossible.

## Highlights & Insights

1. **End-to-end, systematic contribution**: The simultaneous release of a dataset, evaluation benchmark, and training methodology constitutes a complete ecosystem for clinical diagnostic dialogue—offering greater practical value than model-only improvements.

2. **Deep design insight behind CRPO**: The reward function is grounded in the cognitive principle that humans are better at making comparisons than assigning absolute scores. Experiments strongly validate this assumption—relative evaluation achieves nearly twice the agreement with human experts compared to absolute evaluation.

3. **Rigorous data quality control**: The four-stage pipeline (Collection → Anonymization → Expansion → Proofreading) ensures both scale and clinical validity through expert review—a level of methodological rigor that is rare in medical AI research.

## Limitations & Future Work

1. **No support for 3D medical imaging**: High-dimensional modalities such as volumetric CT reconstructions and 3D MRI are not yet covered.
2. **Extremely high computational requirements**: Training costs of 128 A100 GPUs limit applicability in resource-constrained settings.
3. **MediScope dataset is not publicly released**: Its origin in real clinical data raises privacy concerns that may hinder reproducibility.
4. **Evaluation relies primarily on GPT-4**: Potential biases in the automatic evaluator may affect conclusions, despite the relatively high agreement with human experts.
5. **Counterpart model selection strategy in CRPO is underspecified**: The choice of the 5 counterpart models may influence training outcomes, yet this aspect receives limited discussion.

## Related Work & Insights

- **LLaVA-Med / HuatuoGPT-Vision**: Early medical multimodal models—MediScope's multi-turn dialogue characteristics represent an important complement to these prior works.
- **GRPO (Shao et al. 2024)**: The foundational framework for CRPO—CRPO replaces absolute scoring rewards with pairwise comparative judgments.
- **Lingshu (Xu et al. 2025)**: A general-purpose medical VLM—PulseMind significantly outperforms Lingshu in diagnostic dialogue scenarios.
- The "comparison over scoring" principle underlying CRPO aligns with preference-based reward modeling in RLHF and can potentially be extended to other domains requiring fine-grained evaluation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Significant contributions in the dataset and evaluation benchmark; CRPO offers a novel and practical approach.
- **Technical Depth**: ⭐⭐⭐ — Each component (data, evaluation, training) is soundly designed, though individual modules offer limited technical innovation in isolation.
- **Practicality**: ⭐⭐⭐⭐⭐ — Directly targeting real-world clinical scenarios, the systematic solution demonstrates high deployment value.
- **Clarity**: ⭐⭐⭐⭐ — System architecture is clearly presented; experimental comparisons are comprehensive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CGBench: Benchmarking Language Model Scientific Reasoning for Clinical Genetics Research](../../NeurIPS2025/medical_nlp/cgbench_benchmarking_language_model_scientific_reasoning_for_clinical_genetics_r.md)
- [\[ACL 2026\] Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents](../../ACL2026/medical_nlp/beyond_the_individual_virtualizing_multi-disciplinary_reasoning_for_clinical_int.md)
- [\[ICLR 2026\] ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](../../ICLR2026/medical_nlp/atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue.md)
- [\[ACL 2026\] IndicMedDialog: A Parallel Multi-Turn Medical Dialogue Dataset for Accessible Healthcare in Indic Languages](../../ACL2026/medical_nlp/indicmeddialog_a_parallel_multi-turn_medical_dialogue_dataset_for_accessible_hea.md)
- [\[ACL 2026\] SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning](../../ACL2026/medical_nlp/sema-rag_a_self-evolving_multi-agent_retrieval-augmented_generation_framework_fo.md)

</div>

<!-- RELATED:END -->
