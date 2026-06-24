---
title: >-
  [Paper Note] MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark
description: >-
  [ACL 2025][Multimodal VLM][multimodal benchmark] A more robust MMMU-Pro benchmark is constructed based on MMMU through a three-step hardening process (filtering text-only solvable questions, expanding options to 10, and introducing vision-only input). Performance across all models drops by $16.8\%$ to $26.9\%$, revealing that current multimodal models are far from achieving true cross-modal understanding.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "multimodal benchmark"
  - "MMMU"
  - "vision-only evaluation"
  - "shortcut exploitation"
  - "robust evaluation"
date: 2026-05-08
content_hash: 4a153ce69e2f3b19
---

# MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark

**Conference**: ACL 2025  
**arXiv**: [2409.02813](https://arxiv.org/abs/2409.02813)  
**Code**: [https://mmmu-benchmark.github.io/#leaderboard](https://mmmu-benchmark.github.io/#leaderboard)  
**Area**: Multimodal VLM / Benchmark  
**Keywords**: multimodal benchmark, MMMU, vision-only evaluation, shortcut exploitation, robust evaluation

## TL;DR
A more robust MMMU-Pro benchmark is constructed based on MMMU through a three-step hardening process (filtering text-only solvable questions, expanding options to 10, and introducing vision-only input). Performance across all models drops by $16.8\%$ to $26.9\%$, revealing that current multimodal models are far from achieving true cross-modal understanding.

## Background & Motivation

**Background**: MMMU is the gold standard benchmark for evaluating the multi-discipline understanding capabilities of large multimodal models, containing 11.5K college-level multi-discipline questions. GPT-4o has achieved $69.1\%$ accuracy on MMMU, seemingly close to human expert levels.

**Limitations of Prior Work**: In-depth analysis reveals that many MMMU questions suffer from severe "shortcut" issues—text-only LLMs (without visual input) can answer a portion of the questions correctly. There are two reasons: (1) some questions do not actually depend on images to be solved; (2) models exploit statistical patterns in the options and pre-training knowledge to guess the answers without truly understanding the images. For example, Llama-3-70B can find shortcuts among options on some questions relying solely on text.

**Key Challenge**: The 4-option multiple-choice format gives models a $25\%$ random guessing probability. Coupled with semantic cues within the options, the actual guessing probability is even higher. This implies that the high scores on MMMU might be significantly inflated—the models' true multimodal understanding capabilities are being overestimated.

**Goal**: (1) How to filter out "pseudo-multimodal" questions that are solvable with text only? (2) How to reduce the success rate of option guessing? (3) How to test the models' integrated "see + read" capability when text is embedded in images?

**Key Insight**: Starting from human cognitive abilities—humans naturally process text embedded in visual scenes simultaneously in daily life (e.g., reading screenshots, looking at posters). This seamless visual-textual integration is a core cognitive ability. The Vision-only setting directly tests whether models possess this capability.

**Core Idea**: Upgrade MMMU to MMMU-Pro, which better reflects real multimodal understanding capabilities, through a three-step hardening process of filtering, option expansion, and Vision-only input.

## Method

### Overall Architecture
The three-step construction pipeline: starting from MMMU, filtering, augmentation, and visualization are executed sequentially, resulting in 3,460 questions (1,730 standard + 1,730 screenshots/photos). The total score of MMMU-Pro is the average of the 10-option standard performance and the Vision-only performance.

### Key Designs

1. **Text-only Filtering**:

    - **Function**: Remove "pseudo-multimodal" questions that do not require images to be solved.
    - **Mechanism**: Four strong open-source LLMs (Llama3-70B, Qwen2-72B, Yi-1.5-34B, Mixtral-8×22B) are selected, and each question is answered 10 times under the text-only condition. If a model answers correctly more than 5 times, the question is marked as "solvable". Questions marked as solvable by $\ge 3$ models are excluded. From the remaining questions, 1,800 questions are evenly sampled (60 questions for each of the 30 disciplines).
    - **Design Motivation**: A majority voting mechanism of $4$ models $\times 10$ repeated trials ensures the robustness of filtering, preventing false exclusions caused by accidental guesses from a single model.

2. **Option Augmentation**:

    - **Function**: Reduce the random guessing probability of multiple-choice questions (from $25\%$ to $10\%$).
    - **Mechanism**: Human experts generate additional options with the help of GPT-4o, Claude 3.5 filters out unreasonable options, and then two rounds of manual review are conducted for validation. Simultaneously, the correlation between the original questions and images is reviewed, and incoherent questions are removed (filtering out 70 questions, keeping 1,730 final questions).
    - **Design Motivation**: Experiments confirm that simply increasing the number of options can lead to a significant drop in the accuracy of text-only LLMs (Figure 3), effectively suppressing option-based guessing strategies.

3. **Vision-only Input Setting**:

    - **Function**: Test the model's true "see + read" capability when text is embedded in images.
    - **Mechanism**: Human annotators embed the question text and options into screenshots/photos, changing backgrounds, font styles, and font sizes to simulate real-world scene diversity. Models receive only image input, without any explicit text.
    - **Design Motivation**: Mimic actual user habits (sharing screenshots instead of manually typing text) to test whether models possess the core human cognitive capability of "seamlessly integrating visual and textual information".

### Human Expert Performance Estimation
Estimated based on the original MMMU manual evaluation data—the core question content remains unchanged, and human experts are required to write down the full problem-solving process (reducing guessing), while visual-textual integration is a natural human capability. Three tiers of Human Expert: $75.4\% / 82.1\% / 88.6\%$ (Low/Medium/High), far exceeding all models.

## Key Experimental Results

### Main Results

| Model | Standard (4 options) | Standard (10 options) | Vision-only | MMMU (Val) | $\Delta$ (10 options - MMMU) |
|------|----------------|-----------------|-------------|-----------|----------------|
| GPT-4o | $64.7\%$ | $54.0\%$ | $49.7\%$ | $69.1\%$ | $-15.1\%$ |
| Claude 3.5 Sonnet | $63.7\%$ | $55.0\%$ | $48.0\%$ | $68.3\%$ | $-13.3\%$ |
| Gemini 1.5 Pro | $60.6\%$ | $49.4\%$ | $44.4\%$ | $65.8\%$ | $-16.4\%$ |
| InternVL2-76B | $55.0\%$ | $41.9\%$ | $38.0\%$ | $58.3\%$ | $-16.4\%$ |
| LLaVA-OneVision-72B | $52.3\%$ | $38.0\%$ | $24.0\%$ | $56.8\%$ | $-18.8\%$ |
| VILA-1.5-40B | $46.8\%$ | $35.9\%$ | $14.1\%$ | $51.9\%$ | $-16.0\%$ |
| Human Expert (High) | $88.6\%$ | $85.4\%$ | $85.4\%$ | $88.6\%$ | $-3.2\%$ |

### Ablation Study: Impact of CoT and OCR

| Model | Standard w/o CoT | Standard w/ CoT | OCR Acc | Vision w/ OCR | Vision w/o OCR |
|------|-----------------|-----------------|---------|---------------|----------------|
| Claude 3.5 Sonnet | $42.7\%$ | $55.0\%$ | - | - | - |
| GPT-4o | - | - | $92.3\%$ | $49.7\%$ | $49.4\%$ |
| InternVL2-40B | - | - | $85.5\%$ | $32.1\%$ | $28.9\%$ |
| MiniCPM-V2.6 | - | - | $67.0\%$ | $24.2\%$ | $21.1\%$ |

### Key Findings
- Expanding options from 4 to 10 causes GPT-4o to drop by $10.7\%$ ($64.7\% \rightarrow 54.0\%$), and the Vision-only setting drops it further by $4.3\%$ ($54.0\% \rightarrow 49.7\%$), introducing a total drop of $19.4\%$.
- LLaVA-OneVision-72B plummets by $14.0\%$ ($38.0\% \rightarrow 24.0\%$) in the Vision-only setting, exposing its severe deficiency in understanding text-embedded images.
- OCR accuracy is generally high (GPT-4o at $92.3\%$), but explicit OCR prompts have almost no impact on accuracy ($49.7\%$ vs. $49.4\%$), indicating that the bottleneck is not character recognition but deep understanding.
- CoT brings significant improvements in reasoning-intensive subjects like engineering/science (GPT-4o $+14.5\%$), while having limited or even negative effects on subjective disciplines like art and design.
- Human experts experience only a $\sim 3\%$ drop across all hardening steps, while models drop by $15\% - 27\%$, indicating a huge gap.

## Highlights & Insights
- The Vision-only setting is an effective and simple enhancement method—it is low-cost (only requiring manual screenshots) but directly hits the model's Achilles' heel: when text is no longer presented as explicit input, "reading characters in images" becomes an unprecedented challenge.
- The simple operation of "increasing the number of options" can make a benchmark more robust, which also provides inspiration for other multiple-choice benchmarks (such as ARC, ScienceQA).
- The decoupling of OCR capability and visual understanding capability is an important finding—models can accurately extract text from images, but fail to correctly understand the relationships and context between text and visual elements.

## Limitations & Future Work
- Human expert performance is an approximation rather than a re-evaluation, which may overestimate human performance in the Vision-only setting.
- Vision-only photos/screenshots are manually captured by annotators, limiting scale and diversity.
- The discipline coverage still follows MMMU's 30 disciplines, without adding practical fields such as programming or law.
- Updated models released after GPT-4o (e.g., o1, o3) are not tested, and the conclusions might be partially outdated due to model iterations.

## Related Work & Insights
- **vs MMMU**: MMMU-Pro is a strictly hardened version of MMMU. It inherits the question content but eliminates shortcuts through a three-step construction, positioning itself as a "same benchmark, harder version".
- **vs MathVista / ScienceQA**: These are also multi-discipline visual reasoning benchmarks, but the Vision-only setting and 10-option design of MMMU-Pro are significantly more robust.
- **vs MMBench**: MMBench focuses on perception capabilities, while MMMU-Pro focuses on academic knowledge reasoning, making them complementary to each other.
- **Inspiration**: The "three-step hardening" methodology for benchmark design (filtering pseudo-questions $\rightarrow$ increasing difficulty $\rightarrow$ changing input modalities) can serve as a universal paradigm for robustifying benchmarks.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of the Vision-only setting and option expansion brings a truly discriminative benchmark.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with 20+ models, 3 settings, CoT/OCR ablation, and disciplinary dimension analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, transparent construction process, and intuitive presentation of results.
- Value: ⭐⭐⭐⭐⭐ MMMU-Pro has become one of the standard evaluation benchmarks when new multimodal models are released.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AGRI-CM3: A Chinese Massive Multi-Modal Multi-Level Benchmark for Agricultural Understanding](agri-cm3_a_chinese_massive_multi-modal_multi-level_benchmark_for_agricultural_un.md)
- [\[NeurIPS 2025\] DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding](../../NeurIPS2025/multimodal_vlm/danmakutppbench_a_multimodal_benchmark_for_temporal_point_pr.md)
- [\[ACL 2025\] REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)
- [\[ACL 2025\] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)
- [\[NeurIPS 2025\] Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants](../../NeurIPS2025/multimodal_vlm/face-human-bench_a_comprehensive_benchmark_of_face_and_human_understanding_for_m.md)

</div>

<!-- RELATED:END -->
