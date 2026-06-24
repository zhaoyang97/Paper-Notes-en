---
title: >-
  [Paper Note] PCB-Bench: Benchmarking LLMs for Printed Circuit Board Placement and Routing
description: >-
  [ICLR 2026][LLM Evaluation][PCB design] PCB-Bench is the first comprehensive benchmark to systematically evaluate the capabilities of (multimodal) large language models in printed circuit board (PCB) placement and routing tasks. By utilizing three types of tasks—"pure text QA/CQ + image-text multimodal + real-world design understanding"—it covers approximately 3,700 text-based questions, 500 image-text questions, and 174 real-world engineering projects…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "PCB design"
  - "EDA"
  - "multimodal reasoning"
  - "benchmark dataset"
date: 2026-05-08
content_hash: ecf8dfaf7715c2bf
---

# PCB-Bench: Benchmarking LLMs for Printed Circuit Board Placement and Routing

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Q5QLu7XTWx](https://openreview.net/forum?id=Q5QLu7XTWx)  
**Code**: https://github.com/digailab/PCB-Bench  
**Area**: LLM Evaluation / Engineering AI / Multimodal Benchmark  
**Keywords**: PCB design, EDA, LLM evaluation, multimodal reasoning, benchmark dataset

## TL;DR
PCB-Bench is the first comprehensive benchmark to systematically evaluate the capabilities of (multimodal) large language models in printed circuit board (PCB) placement and routing tasks. By utilizing three types of tasks—"pure text QA/CQ + image-text multimodal + real-world design understanding"—it covers approximately 3,700 text-based questions, 500 image-text questions, and 174 real-world engineering projects, revealing that current state-of-the-art models still have significant shortcomings in spatial layout reasoning, rule constraint following, and engineering drawing interpretation.

## Background & Motivation
**Background**: PCB placement and routing are the core and most difficult stages in the electronic design automation (EDA) workflow. Engineers must determine the precise position of each component and the interconnection routing under strict physical, electrical, and manufacturing constraints. Traditional methods rely on classical EDA algorithms (analytical global placement, congestion-driven routing). In recent years, reinforcement learning (RL) has shown competitiveness in IC macro cell placement and joint placement/routing, and the first RL systems for PCB have begun to appear. Simultaneously, large language models (LLMs) such as GPT-4o, GPT-5, Claude, Gemini, DeepSeek, and Qwen have demonstrated amazing capabilities in open reasoning, code generation, and multimodal grounding. This naturally leads to the question: Can these general-purpose foundation models understand and operate expert-level engineering tasks like PCB design?

**Limitations of Prior Work**: This question has remained largely unanswered for two reasons. First, there is no standardized benchmark—existing PCB datasets (PCB-DSLR, DeepPCB, FICS-PCB, FPIC, PCB-Vision, etc.) are almost entirely RGB image sets for defect detection and component recognition, lacking a cross-modal, rule-driven placement and routing evaluation covering "text $\leftrightarrow$ image $\leftrightarrow$ design file." Second, PCB data is scarce and expensive—design costs are as high as $0.2 \sim 0.5$ USD per pin, requiring hardware tape-out verification. Coupled with intellectual property restrictions, very little is open-sourced, and even top academic research often experiments on fewer than 20 boards, severely limiting reproducibility.

**Key Challenge**: Existing LLM-for-EDA work (text-to-HDL assistants, Verilog code generation evaluations) only examines textual capabilities, while RL placement methods only optimize geometric objectives under fixed rules. Neither touches the core capability of "performing cross-modal alignment and rule reasoning under real-world PCB engineering constraints," leaving a clear evaluation gap.

**Goal**: Construct a unified benchmark that can simultaneously examine textual reasoning, image-text alignment, and real-world design drawing understanding, incorporating abstract semantics (design principles) and physical constraints (signal integrity, routing rules) into quantifiable evaluation tasks.

**Key Insight**: The authors observe that real-world PCB design is naturally multimodal—text evaluates reasoning, images evaluate spatial grounding, and structured design files provide domain knowledge. Thus, the benchmark is designed with three complementary tasks, aligning each category with a specific modal combination and reasoning goal.

**Core Idea**: Use expert-annotated, dual-format (open-ended QA + multiple-choice CQ), cross-modal question banks to turn the question of "whether LLMs understand PCB placement and routing" into a systematically comparable evaluation rather than scattered qualitative observations.

## Method

### Overall Architecture
PCB-Bench is not a single model but a comprehensive benchmark consisting of "three task categories + a construction pipeline + a set of evaluation protocols." The three task categories correspond to three modal combinations: **Task 1 Pure Text QA/CQ** (~1,800 open-ended questions, each with a single-choice version, totaling ~3,700 questions, testing PCB knowledge reasoning at the text level); **Task 2 Image-Text Multimodal QA/CQ** (~500 questions, pairing PCB layouts with text prompts to test vision-semantic alignment); **Task 3 Real-world Design Understanding** (174 engineering projects, providing only EDA screenshots for models to generate functional/structural descriptions, testing pure visual engineering drawing interpretation).

The question bank itself is generated via an expert-driven construction pipeline: knowledge points are first collected from multiple sources such as textbooks, websites, commercial MLLM outputs, domain experts, and PCB education curricula. Human experts design a structured syllabus and define the knowledge scope. This is followed by multi-stage question generation (dual-form QA/CQ $\rightarrow$ labeling schema and format standardization $\rightarrow$ quality assurance to remove ambiguity/redundancy). Finally, all questions undergo iterative expert review to ensure technical correctness, clarity, and engineering relevance. The evaluation employs a unified zero-shot protocol across models, applying different metrics based on task type. These three tasks, the pipeline, and the protocols form a complete closed loop of "generation—review—evaluation."

### Key Designs

**1. Tri-modal Task System: Mapping the multimodal nature of real-world PCB design into three complementary evaluations**

The authors address the pain point that "existing benchmarks only test single modalities and fail to cover the multimodal needs of real PCB design." The solution is to split the benchmark into three tasks aligned with different modal combinations: Task 1 (pure text) covers two domains (placement and routing), two difficulty levels (Easy/Hard), and two scales (Macro such as module/power planning and Micro such as signal integrity/high-speed routing), spanning over 25 sub-topics. Task 2 (image-text) pairs real/simulated PCB layouts with natural language prompts, covering sub-tasks like component recognition, functional block identification, routing type detection, via inspection, and differential pair continuity analysis. Task 3 (screenshots only) requires models to provide open-ended descriptions of real, noisy, and heterogeneous PCB designs, simulating scenarios where engineers perform initial visual inspections. These tasks progress from "textual knowledge $\rightarrow$ image-text alignment $\rightarrow$ pure visual interpretation," ensuring a full spectrum of coverage from abstract semantics to physical space.

**2. Dual-format QA+CQ Design: Supporting both generative and discriminative evaluation for the same knowledge point**

The limitation is that multiple-choice questions (CQ), while possessing objective scoring standards, allow models to guess or use elimination, potentially overestimating capability. Conversely, open-ended questions (QA) reflect real-world usage but are difficult to judge via exact string matching. The authors' approach is to generate an open-ended QA and a corresponding single-choice CQ for every knowledge point. Thus, CQ uses top-1 accuracy to test factual discrimination, while QA uses semantic metrics to test generative reasoning fidelity. Experiments validate this design—CQ is generally easier than QA, and CQ accuracy does not always correlate with QA semantic alignment (e.g., InternVL3-78B is strong in CQ but lower in QA semantic alignment), indicating that discriminative and generative abilities are distinct dimensions.

**3. Expert-driven Multi-source Construction and Iterative Review Pipeline: Ensuring engineering authenticity rather than LLM self-generation**

PCB questions require high professional accuracy; direct LLM generation often results in plausible but incorrect answers. The authors utilize a human-expert-centric pipeline: knowledge sources are multi-source and complementary. Experts first design a structured syllabus; each knowledge point then undergoes "dual-format generation $\rightarrow$ labeling schema standardization $\rightarrow$ quality assurance" to produce candidate questions. Finally, all questions are iteratively reviewed by experts to verify technical correctness and refine phrasing for engineering reasoning. The 174 real designs strictly follow open-source licenses, retaining URLs from OSHWHub/JLCPCB, and include feedback from engineers at a PCB manufacturing firm to verify industrial relevance.

**4. Structured Evaluation Protocols for Generation and Discrimination: Matching metrics based on task types**

Different tasks require different definitions of "correctness." The authors assign metrics accordingly: CQ uses top-1 accuracy; open-ended QA uses BERTScore and Sentence-BERT (SBERT) similarity to measure consistency with reference answers—as PCB answers involve technical terms and diverse expressions where exact matching is inappropriate; Task 3 further reports precision/recall/F1 to capture complementary aspects of prediction quality. All models are evaluated under a unified zero-shot setting to simulate real deployment without fine-tuning. This protocol allows for quantitative comparison and reveals that existing semantic metrics may sometimes miss PCB-specific technical correctness.

## Key Experimental Results

### Main Results

Evaluations were conducted under a unified zero-shot setting, covering state-of-the-art closed-source models (GPT-4o/5, Claude-Opus-4.1, Gemini-2.5-Pro, DeepSeek-V3.1), open-source large models (LLaMA-4-Maverick-400B, InternVL3-78B, Qwen series), and two domain-specific variants (QLoRA fine-tuned and RAG versions of Qwen2.5-7B).

Selected results for Task 1 (Text QA/CQ) (CQ accuracy, %):

| Model | Placement-Macro CQ | Placement-Micro CQ | Routing-Macro CQ | Routing-Micro CQ |
|------|------|------|------|------|
| Claude-Opus-4.1 | 93.30 | 94.35 | 99.16 | 92.32 |
| GPT-4o | 92.74 | 93.82 | 98.32 | 91.13 |
| GPT-5 | 88.27 | 91.79 | 99.16 | 90.17 |
| DeepSeek-Chat-V3.1-671B | 92.74 | 93.64 | 97.48 | 88.49 |
| InternVL3-78B | 90.50 | 93.91 | 97.48 | 91.37 |
| Ministral-3B | 59.21 | 42.71 | 74.79 | 54.68 |

GPT-4o, Claude-Opus-4.1, and DeepSeek-V3 form the top tier. Claude leads in CQ accuracy, while GPT-4o and DeepSeek are stronger in QA semantic alignment. Smaller models (Ministral-3B, MythoMax-L2-13B) show a significant performance drop on Hard/Micro questions.

Selected results for Task 2 (Image-Text Multimodal):

| Model | CBC Acc(%) | CBFB Acc(%) | BWEI Acc(%) | QAR-BERT |
|------|------|------|------|------|
| GPT-5 | 83.26 | 75.67 | 90.90 | 0.8561 |
| Gemini-2.5-Pro | 81.08 | 84.00 | 100.00 | 0.8362 |
| LLaMA-4-Maverick | 77.60 | 70.66 | 54.54 | 0.8226 |
| InternVL-3-78B | 76.83 | 54.66 | 45.45 | 0.8357 |
| Qwen3-VL-8B-Instruct | 75.28 | 58.66 | 45.45 | 0.8161 |

(CBC=Component Recognition CQ, CBFB=Component Recognition Fill-in-Blank, BWEI=Basic Wiring Error Identification, QAR=Routing QA). GPT-5 shows balanced performance, while Gemini-2.5-Pro excels in CBFB and BWEI (100%). Open-source models drop to 45%~55% on BWEI, which requires fine-grained visual judgment.

### Ablation Study

While the paper is a benchmark rather than a single model, it utilizes domain-specific variants for comparative analysis:

| Configuration | Key Observation | Description |
|------|---------|------|
| General LLM vs. SLM | Strength correlates with scale | Gemini-2.5, GPT-4o/5, Claude, and Qwen-VL-Max lead; 3B/12B models fall behind on Hard questions. |
| Qwen2.5-7B + QLoRA(D.S.) | Gain in SBERT alignment | Domain fine-tuning improved SBERT from ~0.22 to ~0.48~0.60, but CQ accuracy did not surpass SOTA models. |
| Qwen2.5-7B + RAG(D.S.) | Gain in semantic similarity | Retrieval augmentation improved semantic alignment but remained inferior to closed-source SOTA. |
| CQ vs. QA | CQ is consistently easier | Multiple-choice allows for guessing; QA semantic alignment is generally lower. |
| BERTScore vs. SBERT | Lack of consistency | Existing semantic metrics measure different aspects and may miss PCB-specific correctness. |

### Key Findings
- **Discrimination $\neq$ Generation**: CQ accuracy often diverges from QA semantic alignment. Models may perform well on selection while failing to provide correct open-ended explanations—a phenomenon captured by the dual-format design.
- **Multimodality is the greatest weakness**: On Task 2's Wiring Error Identification (BWEI), open-source models drop significantly, indicating that visual grounding and textual reasoning alignment are still immature.
- **Domain specialization is effective but insufficient**: QLoRA and RAG improve semantic similarity but fail to surpass general SOTA closed-source models, suggesting that lightweight adaptation of general bases is insufficient for expert-level PCB tasks.
- **Task 3 has low discriminative power**: Models show very similar metrics (BERTScore ~0.82, F1 ~0.85) on real design understanding, suggesting that while high-level functional descriptions are achievable, deep structural understanding remains difficult.

## Highlights & Insights
- **Dual-format QA+CQ is the most clever design**: By transforming the same knowledge point into both generative and discriminative tasks, the benchmark preserves the realism of open-ended answers while using CQs as objective anchors. This reveals performance dimensions often hidden by single metrics.
- **Expert Review + Traceability**: In an era where LLMs often generate erroneous training data, the insistence on human expert review and traceability to source URLs ensures benchmark credibility and provides high-quality EDA corpora.
- **Task Difficulty Gradient**: Moving from pure text to image-text alignment and then to pure visual interpretation effectively mirrors the professional workflow of an engineer.

## Limitations & Future Work
- **Ours vs. Human Replacement**: The authors admit current LLMs only possess foundational PCB knowledge and are far from replacing humans in open-ended scenarios.
- **Understanding vs. Generating Layouts**: The benchmark tests whether models can "understand" or "answer" PCB questions, not whether they can produce manufacturable layouts—a significant gap remains toward end-to-end "LLM-driven PCB design."
- **Task 3 Metric Limitations**: Similar F1 scores and inconsistent BERTScore/SBERT results suggest that general semantic metrics may fail to detect true structural understanding differences in engineering drawings.
- **Zero-shot Constraint**: The lack of prompt engineering or few-shot examples may underestimate the upper bounds of model capability.

## Related Work & Insights
- **vs. Existing PCB Datasets**: Prior sets like PCB-DSLR or DeepPCB focus only on visual defect detection. PCB-Bench is the first to cover "Text + Schematic + Source + Manufacturing" to evaluate LLM design understanding.
- **vs. LLM-for-EDA Benchmarks**: Existing work focuses on HDL/Verilog code generation (textual/logic); PCB-Bench adds the missing components of visual and physical design files.
- **vs. RL Placement**: RL methods optimize geometric objectives under fixed rules without multimodal semantic reasoning. This work provides a complementary perspective on "engineering semantics."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First LLM benchmark for PCB placement/routing spanning three modalities; fills a clear gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 13+ models across three tasks with multiple metrics, though missing generative layout evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear task definitions and standardized tables; some metric consistency issues noted.
- Value: ⭐⭐⭐⭐⭐ Provides a standardized platform for "Engineering LLMs," with lasting reference value for the EDA+AI field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LMGame-Bench: How Good are LLMs at Playing Games?](lmgame-bench_how_good_are_llms_at_playing_games.md)
- [\[ICLR 2026\] DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)
- [\[ICLR 2026\] Benchmarking Overton Pluralism in LLMs](benchmarking_overton_pluralism_in_llms.md)
- [\[ICLR 2026\] LogiConBench: Benchmarking Logical Consistencies of LLMs](logiconbench_benchmarking_logical_consistencies_of_llms.md)
- [\[ACL 2026\] AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation](../../ACL2026/llm_evaluation/aj-bench_benchmarking_agent-as-a-judge_for_environment-aware_evaluation.md)

</div>

<!-- RELATED:END -->
