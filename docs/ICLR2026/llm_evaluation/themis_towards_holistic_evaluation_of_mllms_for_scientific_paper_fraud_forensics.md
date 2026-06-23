---
title: >-
  [Paper Note] THEMIS: Towards Holistic Evaluation of MLLMs for Scientific Paper Fraud Forensics
description: >-
  [ICLR 2026][LLM Evaluation][Paper Note] THEMIS constructs a multi-task benchmark for "scientific paper image fraud forensics" (4,054 questions, 5 fraud types, 16 fine-grained manipulations, 7 real academic scenarios). It maps fraud types to 5 expert-level visual reasoning abilities and evaluates 16 mainstream MLLMs. The study reveals systematic shortcomings
tags:
  - ICLR 2026
  - LLM Evaluation
date: 2026-05-08
content_hash: 3b827c04536793da
---
# THEMIS: Towards Holistic Evaluation of MLLMs for Scientific Paper Fraud Forensics

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=y3UkklvoW9](https://openreview.net/forum?id=y3UkklvoW9)  
**Code**: https://github.com/BUPT-Reasoning-Lab/THEMIS (Including project page and HuggingFace dataset)  
**Area**: Multimodal VLM Evaluation / Academic Fraud Forensics / Visual Forgery Benchmarking  
**Keywords**: MLLM Evaluation, Image Forgery Detection, Academic Fraud, Visual Reasoning, Multi-task Benchmark

## TL;DR
THEMIS constructs a multi-task benchmark for "scientific paper image fraud forensics" (4,054 questions, 5 fraud types, 16 fine-grained manipulations, 7 real academic scenarios). It maps fraud types to 5 expert-level visual reasoning abilities and evaluates 16 mainstream MLLMs. The study reveals systematic shortcomings in "forensics" capabilities, with even the strongest GPT-5 achieving an overall score of only 56.15% in complex real-world scenarios.

## Background & Motivation
**Background**: Multi-modal Large Language Models (MLLMs) have advanced from basic perception (recognition, localization, description) to complex reasoning. Existing evaluations typically focus on "knowledge-intensive" tasks like mathematics or chart analysis, where scores continue to rise.

**Limitations of Prior Work**: The authors argue these high scores are "inflated"—many multimodal problems can be **translated into text without loss of information**, allowing the underlying LLM's strong textual reasoning to solve them without truly testing vision. Consequently, whether models possess expert-level visual understanding remains obscured. Additionally, existing forgery benchmarks have significant flaws: FakeBench / DiffuSyn / SHIELD cover narrow fraud types; MFC-Bench / MMFakeBench focus only on coarse-grained binary classification; Forensics-Bench relies almost entirely on synthetic data and lacks real-world cases.

**Key Challenge**: To expose the "intrinsic visual capabilities" of MLLMs, it is necessary to identify boundary tasks where image evidence **cannot be losslessly replaced by text**. Scientific paper fraud forensics satisfies this: it requires both pixel-level anomaly perception (splicing seams, copy-move traces, generative textures) and an understanding of scientific context and logical consistency, which text descriptions cannot fully capture.

**Goal**: To create a multi-task benchmark mirroring the complexity of real-world academic fraud that can (1) cover complex scenarios from actual retracted cases, (2) distinguish various fraud techniques at a fine-grained level, and (3) decompose fraud types into diagnosable multi-dimensional abilities.

**Key Insight**: Using "paper image fraud" as a stress-test boundary scenario, combining real fraud cases from retraction databases with high-fidelity synthetic data, and designing a mapping from "fraud types to core reasoning abilities" to provide diagnostic insights rather than just a total score.

**Core Idea**: Anchor difficulty with real retracted cases, stack high-complexity synthetic data using 16 controllable manipulations, and map fraud techniques to 5 expert-level visual reasoning abilities to create a forensic benchmark that is both realistic and fine-grained.

## Method

### Overall Architecture
THEMIS is not a new model but a **comprehensive benchmark + data construction pipeline + capability diagnostic system**. It is organized around three pillars: (1) Real scenarios and complexity—4,054 questions covering 7 academic scenarios (micrographs, medical imaging, stained sections, charts, physical objects, etc.), with 60.47% involving complex textures where manipulation is harder to detect; (2) Diversity and granularity of fraud types—systematic coverage of 5 major fraud techniques (Splicing, Copy-Move, AI-Generated, Duplication, Text–Image Inconsistency) under 16 fine-grained operations, with each sample averaging 2.08 overlapping operations; (3) Multi-dimensional capability evaluation—mapping fraud techniques to 5 core visual forensic abilities.

The data workflow follows "collection and parsing → fraud generation and quality control (QC) → task design → model evaluation": first, real retracted cases are collected from Retraction Watch / PubPeer and high-quality papers from PubMed Central are used for synthetic materials, with YOLOv7 used to segment images into "panels" (sub-graph units with independent semantics); then, generation pipelines for each of the 5 fraud types create manipulated samples, which are QCed by 16 experts and GPT-4o mini; finally, samples are organized into 3 core tasks to evaluate 16 MLLMs and perform capability attribution.

### Key Designs

**1. Dual-source Data (Real Retractions + High-Fidelity Synthesis): Aligning difficulty with real-world fraud while ensuring scalability.**

Purely real cases are scarce and hard to annotate, while purely synthetic data is often too "fake," failing to challenge humans or models. THEMIS adopts a dual-source approach: On the real side, experts extracted 194 real "fraud panels" from 41 high-quality manuscripts out of 1,432 retracted papers to form a gold standard of 152 real forensic cases; on the synthetic side, 41,422 high-resolution panels were cut from 879 high-quality papers, with 150 `(image, caption, relevant sentence)` triplets selected for cross-modal tasks. Crucially, experiments verified that the "deceptiveness" of the synthetic data is comparable to real data—synthetic data was even harder than real data in composite manipulation recognition tasks, except for Splicing where real data was harder (due to more subtle splicing).

**2. 5 Fraud Types × 16 Fine-grained Manipulations: Dialing "difficulty" to expert levels via controllable stacking.**

THEMIS goes beyond binary classification. For the 5 techniques, it designed generation pipelines with fine-grained operations: Splicing uses foreground/background recombination + bidirectional splicing + boundary blending for visual coherence; Copy-Move uses SAM for automatic matting + adaptive grid positioning to transfer objects within the same panel; AI-Generated includes Image Inference Forgery (full image generation via Flux) and Targeted Region Editing (localized editing via Stable Diffusion / GPT-Image-1); Duplication covers global duplication (direct reuse or geometric/parametric transformation) and local duplication (crop-transform-recompose); Text–Image Inconsistency modifies captions or sentences to create numerical or trend inconsistencies. An average of 2.08 operations per sample pushes the task to expert-level—GPT-5's F1 score on Duplication identification plummeted from 38% to 17% as operations increased from 1 to 3.

**3. Mapping Fraud Types to 5 Core Reasoning Abilities: Decomposing total scores into diagnostic profiles.**

Instead of providing only accuracy, THEMIS maps the 5 techniques to 5 expert-level visual forensic abilities: **Expert Knowledge Utilization** (qualifying manipulations using domain priors), **Visual Recognition** (perceiving anomalies in complex textures), **Spatial Reasoning** (understanding positional/structural relationships between components), **Region Localization** (pointing out manipulated areas at the sub-image level), and **Comparative Reasoning** (cross-image or image-text evidence comparison). Each question has weighted abilities (e.g., Figure 4 in the paper). This allows for "capability profiles"—for example, GPT-5 scores 53.50% in visual recognition but only 24.14% in region localization, highlighting a significant imbalance.

**4. Three Tasks + Metric Suites + Balanced Robustness Index (BRI): Measuring performance by task nature and penalizing "imbalance."**

THEMIS designs 3 core tasks: **Task 1: Single-Mode Forgery Identification and Localization (SMF-IL)** for Splicing / Copy-Move / AI-Generated, requiring models to distinguish between no fraud / uncertain / three types of forgery and perform block-level localization (metrics: ACC and IoU); **Task 2: Composite Manipulation Operation Identification (CMO-I)** for Duplication, selecting from 7 categories including scaling/rotation (metrics: Set-based F1); **Task 3: Cross-Modal Inconsistency Identification and Localization (CMI-IL)** for text-image inconsistency, identifying numerical/trend errors and locating/correcting the minimal sentence unit (metrics: ACC and text-level F1). Finally, the **Balanced Robustness Index (BRI)** aggregates performance while penalizing high variance across tasks to distinguish consistently performing models from those that excel in only one area.

## Key Experimental Results

### Main Results
Testing 16 MLLMs (9 closed-source + 7 open-source) on synthetic data, reporting identification (Id) and localization (Loc) scores, aggregated into BRI. Even the strongest model, GPT-5, achieved a BRI of only 56.15%.

| Model | SMF Id Avg | SMF Loc Avg | DUP Id | TII Id | TII Loc | BRI |
|------|------|------|------|------|------|------|
| GPT-5 | 53.50 | 24.14 | 33.32 | 60.67 | 27.44 | **56.15** |
| OpenAI o4-mini-high | 51.68 | 20.00 | 30.34 | 66.33 | 32.22 | 52.34 |
| Qwen-VL-Max | 51.07 | 41.42 | 23.33 | 56.00 | 15.36 | 49.83 |
| Gemini 2.5 Flash | 55.47 | 47.52 | 24.96 | 36.33 | 28.24 | 44.70 |
| Qwen2.5-VL-72B (Best Open) | 54.91 | 47.46 | 16.75 | 61.33 | 12.32 | 47.16 |

Core findings: (1) **Limited overall capacity**—SOTA peaks at 56.15%. Open-source Qwen2.5-VL-72B reached 47.16%, competing with closed-source models; (2) **Universal imbalance**—most models are strong in only one or two sub-tasks; (3) **Localization is much harder than identification**—all models dropped significantly when moving from identification to localization (GPT-5 dropped 55%, o4-mini-high dropped 61%, while Gemini 2.5 Flash dropped only 14%, showing better spatial perception); (4) **Weak fine-grained cross-modal alignment**—Models are accurate at identifying text-image inconsistencies but struggle to locate the specific problematic text segments.

### Ablation Study
Evaluation of 152 real retracted cases (Table 3) showed Qwen2.5-VL-72B leading in SMF-IL identification (50.00) and Gemini 2.5 Flash leading in CMO-I (49.58). Combined analyses (Figure 5) revealed:

| Analysis Dimension | Key Observations |
|------|------|
| Composite Robustness | GPT-5's F1 on Duplication drops from 38% to 17% as operations increase 1→3. |
| Input Perturbations | Gaussian blur caused the largest performance drop, followed by JPEG compression and scaling. |
| Edge Perception | Copy-Move detection is generally better than Splicing (the former uses both edge and similarity cues). |
| Transformation Sensitivity | Models perform okay on direct reuse but degrade significantly under geometric/appearance changes. |
| CoT Prompting | Consistent performance gains, more pronounced in weaker models (e.g., Llama 4 Maverick). |
| Few-shot Prompting | Unstable gains; some sub-tasks dropped, likely due to models overfitting to local patterns in examples. |

### Key Findings
- **Localization is a universal bottleneck**: The drop from identification to localization suggests MLLMs "know it's fake but don't know where," with region-level forensics far from professional standards.
- **Composite operations act as amplifiers**: Overlapping manipulations halved the F1 score of top models, exposing a lack of robust reasoning for "combined transformations."
- **Error attribution to capability dimensions**: Analysis of 100 GPT-5 failures identified the main bottlenecks as expert knowledge utilization (43/100) and visual recognition (37/100).

## Highlights & Insights
- **Optimal Boundary Task Selection**: Using scientific paper fraud forensics bypasses the common flaw in multimodal benchmarks where images can be translated into text, forcing the extraction of true "intrinsic visual capabilities."
- **Dual-source Strategy**: Real retracted cases anchor the difficulty, while controllable synthesis allows for systematic stacking of operations. The verification that synthetic "deceptiveness" matches real-world samples is a crucial validation.
- **Capability Mapping + BRI**: Upgrading from a single accuracy score to diagnostic capability profiles and penalizing "imbalance" provides a roadmap for multi-dimensional evaluation applicable to other fields.
- **High Reproducibility**: Code, data, and scripts are fully open-sourced. 16 experts spent 200 hours on QC, removing ~20% of low-quality samples.

## Limitations & Future Work
- **Evaluation only**: THEMIS identifies weaknesses but does not provide architectural solutions to improve localization or composite robustness.
- **Dependency on specific generators**: The AI-Generated subset relies on current models like Flux/Stable Diffusion; as generators evolve, the benchmark will require continuous updates.
- **Real case scale**: 152 real cases is relatively small compared to 4,054 questions, limited by the availability of high-quality retraction data.
- **Metric Details**: The BRI index is variance-dependent; users must refer to Appendix B.3 for the precise formula to avoid misuse.
- **Future directions**: Incorporating forensic-specific training data or tool-augmentation (e.g., domain prior retrieval, region proposal modules) could address current shortcomings.

## Related Work & Insights
- **vs Forensics-Bench**: While both cover 5 fraud types, Forensics-Bench is mostly synthetic and disconnected from real-world difficulty; THEMIS anchors difficulty with 152 real cases.
- **vs MFC-Bench / MMFakeBench**: These focus on coarse-grained binary classification; THEMIS provides multi-dimensional diagnostic profiles.
- **vs FakeBench / DiffuSyn / SHIELD**: These cover narrower fraud types; THEMIS provides systematic coverage of 5 themes and 16 operations.
- **vs Traditional Image Forensics (CIMD / GIM)**: Traditional benchmarks evaluate specific detectors; THEMIS is designed for the holistic "identification + localization + cross-modal" reasoning of MLLMs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

---

## Related Papers

- [\[ICLR 2026\] Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation](holistic_agent_leaderboard_the_missing_infrastructure_for_ai_agent_evaluation.md)
- [\[ACL 2026\] Reward Modeling for Scientific Writing Evaluation](../../ACL2026/llm_evaluation/reward_modeling_for_scientific_writing_evaluation.md)
- [\[ICLR 2026\] NAIPv2: Debiased Pairwise Learning for Efficient Paper Quality Estimation](naipv2_debiased_pairwise_learning_for_efficient_paper_quality_estimation.md)
- [\[ACL 2026\] HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing](../../ACL2026/llm_evaluation/howtobench_holistic_evaluation_for_llms_capability_in_human-level_writing_using_.md)
- [\[ACL 2026\] SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models](../../ACL2026/llm_evaluation/scicustom_a_framework_for_custom_evaluation_of_scientific_capabilities_in_large_.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DISCO: Diversifying Sample Condensation for Efficient Model Evaluation](disco_diversifying_sample_condensation_for_efficient_model_evaluation.md)
- [\[ICLR 2026\] Teach2Eval: An Interaction-Driven LLMs Evaluation Method via Teaching Effectiveness](teach2eval_an_interaction-driven_llms_evaluation_method_via_teaching_effectivene.md)
- [\[ICLR 2026\] SparseEval: Efficient Evaluation of Large Language Models by Sparse Optimization](sparseeval_efficient_evaluation_of_large_language_models_by_sparse_optimization.md)
- [\[ICLR 2026\] PRISM-Physics: Causal DAG-Based Process Evaluation for Physics Reasoning](prism-physics_causal_dag-based_process_evaluation_for_physics_reasoning.md)
- [\[ICLR 2026\] Rethinking LLM Evaluation: Can We Evaluate LLMs with 200× Less Data?](rethinking_llm_evaluation_can_we_evaluate_llms_with_200_less_data.md)

</div>

<!-- RELATED:END -->
