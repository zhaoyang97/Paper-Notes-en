---
title: >-
  [Paper Note] RedSage: A Cybersecurity Generalist LLM
description: >-
  [ICLR 2026][LLM Safety][Cybersecurity LLM] This paper introduces RedSage—the first fully open-source cybersecurity generalist LLM—built upon large-scale domain continual pre-training on 11.7B tokens…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "Cybersecurity LLM"
  - "Continual Pre-training"
  - "Agentic Data Augmentation"
  - "Security Evaluation Benchmark"
date: 2026-05-08
content_hash: b968067b18367818
---

# RedSage: A Cybersecurity Generalist LLM

**Conference**: ICLR 2026
**arXiv**: [2601.22159](https://arxiv.org/abs/2601.22159)
**Code**: [GitHub](https://github.com/AraLabs-AI/RedSage) (open-source data + model + code)
**Area**: AI Security / Cybersecurity
**Keywords**: Cybersecurity LLM, Continual Pre-training, Agentic Data Augmentation, Security Evaluation Benchmark

## TL;DR

This paper introduces RedSage—the first fully open-source cybersecurity generalist LLM—built upon large-scale domain continual pre-training on 11.7B tokens, agentic-augmentation SFT with 266K samples, and RedSage-Bench, the first comprehensive evaluation benchmark covering knowledge, skills, and tools. The resulting 8B-parameter model surpasses same-scale SOTA on cybersecurity benchmarks by +5.4 pp and approaches Qwen3-32B, while simultaneously improving general-purpose performance (+8.4 pp vs. Qwen3-8B).

## Background & Motivation

**Background**: Cybersecurity threats are growing in complexity, and tasks such as APT attack analysis, vulnerability management, and incident response demand deep domain expertise and tool-operating capabilities. The global cybersecurity talent gap has reached millions (ISC² report), driving demand for LLM-assisted security analyst tools. Several cybersecurity LLMs have emerged in recent years (Foundation-Sec, PRIMUS, DeepHat, etc.), yet each suffers from notable shortcomings.

**Limitations of Prior Work**: Existing cybersecurity LLMs exhibit deficiencies along three dimensions. (1) Incomplete training pipelines: PRIMUS (Trend Micro) performs continual pre-training on 2.57B tokens but SFT on only 835 samples; Foundation-Sec-8B (Cisco) includes pre-training but keeps its data closed-source; DeepHat performs only SFT without pre-training. (2) Limited SFT data quality: most approaches rely on static Q&A pairs or small-scale human annotations, failing to simulate the multi-turn interaction patterns of real-world security workflows. (3) Insufficient benchmark coverage: SecEval/CyberMetric evaluate only knowledge-level MCQs, CyberSecEval evaluates only skills, and no benchmark jointly covers tool-use evaluation and open-ended answer quality assessment.

**Key Challenge**: Building a practical cybersecurity LLM requires simultaneously addressing data scale, training pipeline completeness, and evaluation comprehensiveness—yet existing work covers at most one or two of these. More critically, most works do not release data and code (Foundation-Sec withholds data; SecGemini withholds its model), severely limiting reproducibility and community development.

**Goal**: To construct a fully open-source cybersecurity LLM system covering the complete pipeline from data filtering, continual pre-training, agentic-augmented SFT, and preference alignment to comprehensive evaluation—with all components publicly released.

**Key Insight**: A data-centric philosophy is applied throughout the entire pipeline: a classifier filters domain corpora from FineWeb for large-scale pre-training; high-quality seed data is curated across three dimensions (knowledge, skills, tools); an agentic pipeline automatically converts static documents into multi-turn dialogues; and a hierarchically validated evaluation benchmark is constructed.

**Core Idea**: Large-scale domain pre-training, agentic-augmented SFT, and a three-dimensional evaluation benchmark are combined to build the first fully open-source cybersecurity generalist LLM.

## Method

### Overall Architecture

RedSage is built on Qwen3-8B-Base through a three-stage training process. **Stage 1 (Continual Pre-training, CPT)**: CyberFineWeb (11.7B tokens of cybersecurity-filtered web text with 30% general replay) is used for continual pre-training to obtain RedSage-CFW; subsequently, high-quality curated data RedSage-Seed (28,637 samples, 150M tokens) and non-classifier dumps (459K documents, 700M tokens) are used for further training to obtain RedSage-Base. **Stage 2 (Supervised Fine-tuning, SFT)**: 266K multi-turn dialogues generated via agentic augmentation from seed data (RedSage-Conv, 353M tokens), combined with general instruction data from SmolLM3, are used for SFT to obtain RedSage-Ins. **Stage 3 (Preference Alignment, DPO)**: Tulu 3 8B open-source preference data is used for DPO alignment to produce the final RedSage-DPO. Concurrently, the RedSage-Bench evaluation benchmark is constructed (30K MCQs + 240 open-ended questions) to assess model capabilities across knowledge, skills, and tools.

### Key Designs

1. **CyberFineWeb Domain Corpus Construction and Catastrophic Forgetting Mitigation**
    - **Function**: Efficiently filters cybersecurity text from large-scale web corpora while preserving general capabilities through a replay mechanism.
    - **Mechanism**: A ModernBERT-base binary classifier is fine-tuned to filter FineWeb (Common Crawl 2013–2024, ~15T tokens), yielding a candidate pool of ~125M documents (89.8B tokens). The key design is mixing 30% FineWeb-Edu general educational text as replay to prevent catastrophic forgetting. After applying MinHash-LSH global near-deduplication, ~52M documents (46.8B tokens) remain. The corpus is divided into 20 chronological chunks for sequential training, with early stopping after the 5th chunk to balance cost, resulting in 13M documents (11.7B tokens).
    - **Design Motivation**: Training on the full 89.8B tokens is prohibitively expensive; chronological chunking with early stopping captures the most valuable data within a limited compute budget. Experiments confirm the effectiveness of the 30% replay ratio—RedSage-DPO achieves a mean of 74.33% on the Open LLM Leaderboard, surpassing Qwen3-32B (73.17%), with general capability improving rather than degrading.

2. **Agentic Data Augmentation Pipeline**
    - **Function**: Automatically converts curated static cybersecurity resources into high-quality multi-turn dialogues for SFT training.
    - **Mechanism**: A two-stage agentic framework is employed. The **Planner Agent** analyzes each seed data chunk and dynamically derives candidate skill sets (e.g., vulnerability analysis, tool command generation, penetration testing workflows) and augmentation strategies (how to convert content into dialogue and enrich explanations), without using fixed templates. The **Augmenter Agent** instantiates each plan into role-based multi-turn dialogues in an expert–assistant format, simulating real-world cybersecurity workflows. Outputs undergo three-level filtering for format validity, consistency, and topical relevance. Seed data is curated across three categories—Knowledge (MITRE ATT&CK/CWE/OWASP frameworks, 6,924 + 3,715 samples), Skills (HackTricks/penetration testing writeups, 4,032 samples), and Tools (CLI cheatsheets/Kali documentation, 12,943 + 1,023 samples). The pipeline expands 28,637 seed samples into 266K dialogues (9.2× sample count, 2.3× token count), spanning Knowledge (67K), Skills (39K), and Tools (120K).
    - **Design Motivation**: Manually constructing cybersecurity SFT data is extremely costly and difficult to cover all dimensions. Unlike AgentInstruct's fixed skill templates, the Planner dynamically generates strategies based on content to ensure diversity. The dialogue format better approximates real-world usage than static documents, as security analysts typically complete tasks through multi-turn interactions.

3. **RedSage-Bench Three-Dimensional Evaluation Benchmark**
    - **Function**: The first cybersecurity LLM evaluation benchmark to jointly cover knowledge, skills, and tool use, supporting both MCQ and open-ended question assessment.
    - **Mechanism**: **MCQ Generation**—A 70B instruction model (Llama-3.3-70B/Qwen2.5-72B) generates four-choice questions from seed data, validated through two stages: Stage 1 checks structural validity (format, correctness, distractor quality; pass/fail), Stage 2 scores quality (retained only if >8/10), with quota sampling to ensure categorical balance, yielding 30K MCQs. **Open-Ended QA**—Generated through a two-stage Evaluation-Planner and Q&A Generator, evaluated by LLM-as-Judge on factual correctness (T/F) and answer quality (0–10, covering helpfulness, relevance, and depth), with human verification retaining 240 questions. **Decontamination**—Training samples with semantic similarity >0.9 to benchmark items are removed (2.96%), preventing training leakage.
    - **Design Motivation**: Existing benchmarks evaluate either knowledge only (SecEval, etc.) or skills only (CyberSecEval), with none assessing tool use. MCQs can only evaluate correctness; open-ended QA with quality scoring is necessary to assess response helpfulness and depth.

### Loss & Training

Continual pre-training is conducted on Qwen3-8B-Base using 32× A100-64GB GPUs, DeepSpeed ZeRO Stage 3 distributed training, AdamW optimizer, a fixed learning rate of $2.5\times10^{-6}$ with linear warmup, and single-epoch training (global batch size 1024). SFT uses 2 epochs with cosine learning rate scheduling. DPO uses the Tulu 3 8B Preference Mixture dataset with its original hyperparameters. The entire pipeline is implemented in the Axolotl framework and is fully reproducible via configuration files.

## Key Experimental Results

### Main Results

RedSage-Bench MCQ evaluation (0-shot, accuracy %):

| Model | Macro Avg. | General Knowledge | Frameworks | Offensive/Defensive Skills | CLI Tools | Kali Tools |
|---|---|---|---|---|---|---|
| Lily-Cybersecurity-7B | 71.19 | 68.78 | 67.44 | 76.61 | 71.44 | 66.26 |
| Foundation-Sec-8B-Ins | 76.12 | 74.50 | 77.10 | 80.91 | 74.98 | 68.30 |
| DeepHat-V1-7B | 80.18 | 77.26 | 76.90 | 85.07 | 81.94 | 74.82 |
| Qwen3-8B | 81.85 | 80.46 | 78.82 | 86.16 | 83.92 | 75.56 |
| **RedSage-8B-Ins** | **85.73** | **84.20** | **84.98** | **89.06** | **86.80** | **80.30** |
| RedSage-8B-DPO | 84.83 | 82.48 | 83.80 | 88.54 | 86.30 | 79.30 |
| Qwen3-32B | 85.40 | 84.08 | 82.32 | 89.00 | 87.60 | 80.40 |

External cybersecurity benchmark evaluation (accuracy %):

| Model | Mean | CTI-MCQ | CTI-RCM | CyMtc-500 | MMLU-CSec | SecBench-En |
|---|---|---|---|---|---|---|
| Qwen3-8B-Base | 80.81 | 68.80 | 63.50 | 92.00 | 83.00 | 82.84 |
| Foundation-Sec-8B | 76.90 | 62.40 | 75.40 | 86.60 | 80.00 | 69.86 |
| **RedSage-8B-Base** | **84.56** | **71.04** | **78.40** | **92.60** | **87.00** | **81.76** |
| Qwen3-8B (instruct) | 75.71 | 62.76 | 54.00 | 88.60 | 76.00 | 73.26 |
| **RedSage-8B-DPO** | **81.10** | **70.84** | **70.60** | **90.00** | **79.00** | **80.06** |

### Ablation Study

Contribution of each training stage (base models, RedSage-Bench macro-average accuracy %):

| Training Configuration | Bench Macro Avg. | External Benchmark Mean | Key Changes |
|---|---|---|---|
| Qwen3-8B-Base (baseline) | 84.24 | 80.81 | — |
| + CyberFineWeb (CFW) | 84.86 (+0.62) | 82.66 (+1.85) | Frameworks +3.00, SecBench +0.78 |
| + Seed only | 85.21 (+0.97) | 84.45 (+3.64) | CTI-RCM +15.1, Kali +1.04 |
| + CFW + Seed (Base) | 85.05 (+0.81) | 84.56 (+3.75) | Best overall |
| + SFT (Ins) | 85.73 (+1.49) | 81.30 | Best instruct model |
| + DPO | 84.83 (+0.59) | 81.10 | Best open-ended QA quality |

General capability retention (Open LLM Leaderboard instruct model mean %):

| Model | Mean | MMLU | ARC-C | GSM8K | IFEval |
|---|---|---|---|---|---|
| Qwen3-8B | 65.92 | 73.59 | 62.54 | 75.66 | 85.21 |
| Foundation-Sec-8B-Ins | 69.28 | 64.11 | 63.91 | 77.79 | 76.17 |
| **RedSage-8B-DPO** | **74.33** | **77.07** | **71.76** | **82.71** | **83.44** |
| Qwen3-32B | 73.17 | 82.11 | 69.28 | 87.49 | 88.26 |

### Key Findings

- RedSage-8B-Ins (85.73) surpasses Qwen3-32B (85.40)—a model with 4× the parameters—on the in-house benchmark, demonstrating that domain-targeted training can compensate for parameter scale.
- In open-ended QA, RedSage-DPO outperforms the second-best Qwen3-8B by +7% absolute correctness rate and +0.07 quality score, confirming DPO's significant contribution to response quality.
- CyberFineWeb and Seed provide complementary gains: CFW yields the largest improvements on SecBench/CyMtc, while Seed yields the largest gains on knowledge-intensive CTI-RCM (+15.1 pp).
- General capability improves rather than degrades: RedSage-DPO (74.33%) surpasses Qwen3-32B (73.17%) on the Open LLM Leaderboard, confirming the effectiveness of 30% replay in preventing forgetting.
- Tool use is the weakest dimension for current LLMs: tool-related open-ended questions exhibit the lowest median scores and the longest-tailed distribution.

## Highlights & Insights

- Full-stack openness is the core differentiator: data (11.7B pre-training + 266K SFT), models, code, and the evaluation benchmark are all publicly released, distinguishing RedSage from Foundation-Sec (closed-source data) and SecGemini (closed-source model), and providing a substantial contribution to the community.
- The Planner→Augmenter two-stage framework of the agentic augmentation pipeline offers methodological generality and is transferable to specialized LLM construction in domains such as medicine and law.
- RedSage-Bench's MCQ + open-ended QA + LLM-judge quality scoring design achieves, for the first time, three-dimensional evaluation of knowledge, skills, and tools in the cybersecurity domain.
- Fine-tuning Qwen3-32B on a subset of data via QLoRA also yields improvements, demonstrating that the data pipeline is equally effective for larger models.

## Limitations & Future Work

- The 8B parameter scale limits complex reasoning, with a remaining gap of ~5 pp compared to GPT-5 (86.29 vs. 81.10 mean).
- Tool evaluation is limited to CLI commands and document comprehension, without covering scenarios requiring environment interaction such as CTF challenges.
- LLM-generated training data may propagate biases or inaccuracies despite filtering and validation.
- Cybersecurity knowledge evolves rapidly, making temporal maintenance of the model a persistent challenge.
- Open-sourcing offensive and defensive knowledge carries dual-use risks and requires responsible deployment.

## Related Work & Insights

- Foundation-Sec-8B (Cisco, 5.1B token pre-training + 28K SFT) vs. PRIMUS (Trend Micro, 2.57B pre-training + 835 SFT): RedSage comprehensively leads in data scale (11.7B + 266K), methodology (agentic augmentation), and openness.
- The agentic augmentation approach inherits the spirit of AgentInstruct but innovates by having the Planner dynamically generate skill sets rather than relying on fixed templates.
- The 30% general replay is a classic continual learning strategy; RedSage's contribution lies in embedding it directly into static corpus construction rather than dynamically adjusting the ratio.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Systematic engineering contribution outweighs single-point algorithmic innovation; the agentic augmentation and three-dimensional evaluation design are genuinely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three categories of benchmarks (in-house + external cybersecurity + general), multi-stage ablations, open-ended QA quality evaluation, and large-model scaling validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with rich tables and figures; pipeline description is complete.
- **Value**: ⭐⭐⭐⭐⭐ Full-stack openness greatly advances the cybersecurity AI community; the data pipeline methodology is transferable to other specialized domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates](inference-time_backdoors_via_hidden_instructions_in_llm_chat_templates.md)
- [\[ICLR 2026\] LH-Deception: Simulating and Understanding LLM Deceptive Behaviors in Long-Horizon Interactions](lh-deception_simulating_and_understanding_llm_deceptive_behaviors_in_long-horizo.md)
- [\[ICLR 2026\] From Static Benchmarks to Dynamic Protocol: Agent-Centric Text Anomaly Detection for Evaluating LLM Reasoning](from_static_benchmarks_to_dynamic_protocol_agent-centric_text_anomaly_detection_.md)
- [\[AAAI 2026\] LLM Targeted Underperformance Disproportionately Impacts Vulnerable Users](../../AAAI2026/llm_safety/llm_targeted_underperformance_disproportionately_impacts_vulnerable_users.md)
- [\[ACL 2026\] Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations](../../ACL2026/llm_safety/two_pathways_to_truthfulness_on_the_intrinsic_encoding_of_llm_hallucinations.md)

</div>

<!-- RELATED:END -->
