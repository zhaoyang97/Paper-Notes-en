---
title: >-
  [Paper Note] ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection
description: >-
  [ACL 2026][Medical NLP][GRPO] ProMedical utilizes hierarchical fine-grained clinical rubrics, constructed with physician participation, across preference data, reward models, and benchmarks. Through explicit criteria injection to train multi-dimensional reward models, it achieves a improvement of 22.3% in overall accuracy and 21.7% in safety compli
tags:
  - ACL 2026
  - Medical NLP
  - GRPO
date: 2026-05-08
content_hash: 61961650439a119c
---
# ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection

**Conference**: ACL 2026  
**arXiv**: [2604.08326](https://arxiv.org/abs/2604.08326)  
**Code**: The paper states that public data, reward models, and benchmarks are released; specific URLs were not provided in the cache.  
**Area**: Medical NLP  
**Keywords**: Medical LLM alignment, fine-grained rubric, safety veto, reward model, GRPO

## TL;DR
ProMedical utilizes hierarchical fine-grained clinical rubrics, constructed with physician participation, across preference data, reward models, and benchmarks. Through explicit criteria injection to train multi-dimensional reward models, it achieves a improvement of 22.3% in overall accuracy and 21.7% in safety compliance for Qwen3-8B during medical alignment.

## Background & Motivation

**Background**: Medical LLMs are already capable of answering questions regarding symptoms, diagnosis, treatment, and health management. Closed-source models have approached the level of clinical experts on several medical benchmarks. However, evaluation standards in medical scenarios are becoming increasingly granular: they must not only provide correct facts but also avoid hallucinations, identify risks, follow clinical boundaries, and demonstrate empathy and clear reasoning.

**Limitations of Prior Work**: Mainstream alignment data still primarily consists of coarse-grained preference pairs or overall scores. Models only know which response is better without understanding whether it is due to safety, facts, completeness, tone, or clinical workflow. For high-risk medical errors, such binary signals allow models to easily mistake "fluent and helpful" for "safe and professional."

**Key Challenge**: The evaluation side requires fine-grained clinical criteria, whereas the training side provides coarse-grained preference signals. Inconsistency between training objectives and real-world clinical evaluation makes it difficult for models to internalize complex medical protocols.

**Goal**: To build a unified framework where instruction-specific clinical rubrics are not just post-hoc evaluation tools but are integrated into preference construction, reward modeling, and the RL alignment process.

**Key Insight**: The authors categorize medical response quality into three orthogonal dimensions: Proficiency, Excellence, and Safety, designing Safety as a strict veto constraint to prevent models from offsetting safety violations with high-utility responses.

**Core Idea**: Explicitly inject fine-grained criteria for each medical instruction into the reward model, enabling the reward model to judge preferences under "specific rubric conditions" rather than outputting a black-box scalar that mixes all factors.

## Method

### Overall Architecture
ProMedical consists of three layers: the first is ProMedical-Rubrics, which maps each medical instruction to clinical criteria; the second is ProMedical-Preference-50k and ProMedical-Bench, used for training and evaluation respectively; the third is Explicit Criteria Injection, which trains a Rubric-Aware Reward Model to guide Qwen3-8B for GRPO alignment. Its core is not proposing a new medical QA model, but reshaping the supervisory signals for medical alignment.

```remarkable
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Medical Instruction"] --> B["Three-Component Clinical Rubric and Safety Veto<br/>Proficiency / Excellence / Safety, Lexicographical Comparison"]
    B --> C
    subgraph C["Human-in-the-Loop Rubric Data Construction"]
        direction TB
        C1["Filtering & Deduplication + Strong Model Candidate Generation"] --> C2["Gemini Generates Rubric, Doctors Review 500 Refined Examples per Round"]
        C2 --> C3["Output ProMedical-Preference-50k + ProMedical-Bench"]
    end
    C --> D["Explicit Criteria Injection<br/>Rewrite Preference as P(yw≻yl | x, c), Expand by Criterion"]
    D --> E["Train Rubric-Aware Reward Model (ProMedical-RM)"]
    E --> F["GRPO Alignment for Qwen3-8B<br/>Safety Veto → Main Proficiency → Excellence Hierarchical Aggregation"]
```

### Key Designs

**1. Three-component clinical rubric and safety veto: Decomposing "which response is better" into three interpretable and constrained dimensions**

The fundamental problem with coarse-grained preferences is that models only know a response is overall superior but do not know if it is due to safety, facts, completeness, or tone. Consequently, they often mistake "fluent and helpful" for "safe and professional." ProMedical decomposes medical response quality into three orthogonal components: Proficiency $S_1$ measures basic clinical accuracy and completeness; Excellence $S_2$ rewards attributes beyond the passing line, such as empathy and logical clarity; Safety $S_3$ detects severe hallucinations, harmful suggestions, or boundary violations. Crucially, the final preference is not a simple sum of the three but a lexicographical comparison—first checking safety violations, then proficiency, and finally excellence. Thus, a "very helpful but severely unsafe" response can never win against a safer response, establishing safety as a hard constraint rather than a soft item offset by other dimensions.

**2. Human-in-the-Loop rubric data construction: Balancing scalable generation with physician expertise**

Purely manual rubric writing is too costly, while purely automated generation is prone to medical hallucinations. ProMedical-Preference-50k undergoes source filtering, semantic deduplication, difficulty screening, and expert-guided classification, followed by candidate generation from multiple strong models. The rubrics themselves are generated by Gemini-3-Pro-thinking combined with static expert system instructions and dynamic few-shot examples, while physicians review 500 entries per round, re-injecting corrected gold standards into the example pool. This iterative HITL loop converges generation quality—the example pool aligns closer to clinical consensus, and newly generated rubrics become more reliable, with a reported strict expert evaluation pass rate of 96.40%.

**3. Reward model with explicit Criteria Injection: Enabling reward models to compare responses "under a specific criterion"**

A drawback of scalar rewards is that they lump safety, professionalism, and expression quality into a single number, making the supervisory signal a mixed black box. ProMedical rewrites the traditional reward model learning objective $P(y_w \succ y_l \mid x)$ into a criterion-conditioned form $P(y_w \succ y_l \mid x, c)$, where $c$ is a specific rubric criterion. A response pair is expanded into multiple criterion-conditioned training instances, each labeled independently for "who is better under this dimension." This explicitly deconstructs the supervisory signal—safety, capability, and excellence are separated—and they are later aggregated hierarchically (Safety Veto → Main Proficiency → Excellence), preserving fine-grained judgment while enforcing lexicographical constraints during training.

### Loss & Training
The reward model utilizes a Bradley-Terry style pairwise loss, with inputs including the instruction, candidate responses, and specific criteria, optimizing the criterion-conditioned reward margin. During the policy alignment phase, ProMedical-RM serves as a proxy oracle to calculate hierarchical rewards for sampled outputs from Qwen3-8B's GRPO. The penalty coefficient for safety violations is set high enough to outweigh any positive utility, ensuring that safety issues cannot be offset by high scores in other dimensions.

## Key Experimental Results

### Main Results
ProMedical-Bench includes 795 held-out samples expanded into 5,505 criterion-level pairs: 3,625 for Proficiency, 1,650 for Excellence, and 230 for Safety. Double-blind physician adjudication yielded a weighted Cohen's Kappa of 0.88.

| Model | Pointwise Proficiency | Pointwise Safety | Pairwise Safety | Overall Accuracy |
|------|-----------------------|------------------|-----------------|------------------|
| GPT-5 | 91.50 | 76.45 | 77.39 | 76.42 |
| Gemini-3-Pro | 89.80 | 64.10 | 65.65 | 64.80 |
| DeepSeek-R1 | 89.50 | 78.80 | 80.00 | 78.55 |
| Qwen3-8B | 50.15 | 62.79 | 65.64 | 64.30 |
| PairRM-LLaMA3-8B | 76.50 | 58.80 | 60.43 | 58.95 |
| medical_o1_verifier_3B | 75.20 | 51.90 | 53.04 | 51.10 |
| ProMedical-RM-8B (Llama) | 90.15 | 87.20 | 86.10 | 85.40 |
| ProMedical-RM-8B (Qwen3) | 90.85 | 88.50 | 87.39 | 86.55 |

### Ablation Study

| Model | Safety Precision | Safety Recall | Safety F1 | Description |
|------|------------------|---------------|-----------|------|
| GPT-5 | 79.24 | 73.85 | 76.45 | Strong closed-source models still miss some safety vetoes |
| DeepSeek-R1 | 81.50 | 76.28 | 78.80 | Strong open-source reasoning model, but lower than ProMedical-RM |
| PairRM-LLaMA3-8B | 62.45 | 59.80 | 61.10 | Prone to confusing safety with text fluency |
| medical_o1_verifier_3B | 55.30 | 50.80 | 52.95 | Significant lack of recall |
| ProMedical-RM (Llama) | 89.40 | 85.10 | 87.20 | Fine-grained supervision brings stable improvement |
| ProMedical-RM (Qwen3) | 91.50 | 86.80 | 89.09 | Best Safety Veto detection |

### External Generalization and Policy Alignment

| Method | Q | Q+Criteria | Q+Sub | Conclusion |
|------|---|------------|-------|------|
| Ultra-Medical | 80.53 | - | - | Standard preference optimization baseline |
| RaR | 79.03 | 80.10 | 81.32 | Rubric-related baseline |
| InfiMed-ORBIT | 80.85 | 81.07 | 81.63 | Fine-grained preference baseline |
| Ours | 81.94 | 82.32 | 83.60 | Higher across all three granularities |
| Ours + RAG | 81.60 | 83.20 | 84.28 | Q+Sub is optimal after external knowledge enhancement |

### Key Findings
- ProMedical-RM-8B (Qwen3) achieved an Overall Accuracy of 86.55%, surpassing GPT-5's 76.42 and DeepSeek-R1's 78.55, indicating that specialized rubric-aware reward models can outperform general strong models on fine-grained clinical standards.
- The Llama backbone version also reached 85.40%, only 1.2 points lower than the Qwen3 version, proving that gains primarily stem from explicit criteria injection rather than the specific backbone's capability.
- Meditron-70B's Overall Accuracy was only 53.40%, showing that parameter scale and medical pre-training do not automatically lead to safety constraint compliance.
- Safety Veto F1 improved from 76.45 for GPT-5 to 89.09 for ProMedical-RM (Qwen3), with gains concentrated in high-risk medical boundary identification.

## Highlights & Insights
- The most critical contribution of the paper is moving clinical rubrics from the evaluation stage to the training stage. Medical alignment is not just about "producing more preference data" but about ensuring preference labels have clear clinical justifications.
- Treating Safety as a veto rather than a soft penalty is vital. While many general alignment methods allow dimensions to offset each other, a single severe hallucination in a medical context is enough to invalidate the entire response.
- The double-blind physician adjudication and 0.88 Kappa of ProMedical-Bench enhance the benchmark's credibility and make the reward model's gains more convincing.
- The concept of criteria-conditioned reward models can be transferred to other high-risk fields like law, finance, and education: first decompose standards into explicit criteria, then train the model to evaluate based on those standards.

## Limitations & Future Work
- The framework depends on expert consensus; in medical issues with controversy, inconsistent guidelines, or significant regional differences, rubrics may be difficult to define.
- Currently, only text modality is handled, failing to cover medical imaging, lab results, vital signs, and structured medical records common in clinical workflows.
- The HITL pipeline remains costly; while more scalable than purely manual efforts, new specialties or regional standards may requires re-calibration.
- Although the reward model guides the generative model, final responses may still produce medical hallucinations; real-world deployment requires human physician supervision.
- Benchmark and data construction rely on strong models for candidate generation and initial rubric drafts, necessitating continuous monitoring of generative model bias on data distribution.

## Related Work & Insights
- **vs UltraMedical**: UltraMedical provides large-scale medical preference data; ProMedical further injects fine-grained rubrics for each instruction and distinguishes between safety, capability, and excellence.
- **vs HealthBench**: HealthBench emphasizes physician-written evaluation rubrics; this paper applies similar ideas to training reward models and GRPO alignment.
- **vs General Reward Models**: Models like PairRM can learn general preferences but cannot reliably handle medical safety vetoes; ProMedical-RM's advantage stems from criterion-conditioned supervision.

## Rating
- Novelty: ⭐⭐⭐⭐ Explicitly injecting instruction-specific rubrics into the reward model is a robust alignment design for high-risk scenarios.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage includes datasets, benchmarks, reward models, safety metrics, and external generalization.
- Writing Quality: ⭐⭐⭐⭐ Methodological lines are clear, and tables are information-dense; some formula formatting is slightly complex.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for medical LLM alignment and interpretable reward modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework](region-grounded_report_generation_for_3d_medical_imaging_a_fine-grained_dataset_.md)
- [\[ACL 2026\] CT-FineBench: A Diagnostic Fidelity Benchmark for Fine-Grained Evaluation of CT Report Generation](ct-finebench_a_diagnostic_fidelity_benchmark_for_fine-grained_evaluation_of_ct_r.md)
- [\[ACL 2026\] PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment](principlismqa_a_philosophy-grounded_approach_to_assessing_llm-human_clinical_med.md)
- [\[ACL 2026\] Beyond Prompt: Fine-grained Simulation of Cognitively Impaired Standardized Patients via Stochastic Steering](beyond_prompt_fine-grained_simulation_of_cognitively_impaired_standardized_patie.md)
- [\[AAAI 2026\] GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](../../AAAI2026/medical_nlp/gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)

</div>

<!-- RELATED:END -->
