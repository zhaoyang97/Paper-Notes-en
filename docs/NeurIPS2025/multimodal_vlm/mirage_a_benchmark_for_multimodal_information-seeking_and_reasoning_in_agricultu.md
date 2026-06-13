---
title: >-
  [Paper Note] MIRAGE: A Benchmark for Multimodal Information-Seeking and Reasoning in Agriculture
description: >-
  [NeurIPS 2025][Multimodal VLM][benchmark] MIRAGE is the first multimodal benchmark constructed from real agricultural expert consultation dialogues (35,000+)…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "benchmark"
  - "multimodal"
  - "VLM"
  - "agriculture"
  - "visual grounding"
  - "multi-turn dialogue"
date: 2026-05-08
content_hash: 5051449ed705904e
---

# MIRAGE: A Benchmark for Multimodal Information-Seeking and Reasoning in Agriculture

**Conference**: NeurIPS 2025
**arXiv**: [2506.20100](https://arxiv.org/abs/2506.20100)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: benchmark, multimodal, VLM, agriculture, visual grounding, multi-turn dialogue

## TL;DR
MIRAGE is the first multimodal benchmark constructed from real agricultural expert consultation dialogues (35,000+), evaluating vision-language models on domain-level entity identification, causal reasoning, and clarify-or-respond decision-making. It reveals a severe challenge in which even GPT-4.1 achieves only 43.9% identification accuracy.

## Background & Motivation

1. **Limitations of existing benchmarks**: Mainstream VLM benchmarks (MMMU, VQA) focus on short-text QA or constrained multiple-choice formats, failing to reflect the interactivity and decision demands of real expert consultation scenarios.
2. **Domain knowledge gap**: In knowledge-intensive fields such as agriculture, user queries frequently exhibit ambiguity, incomplete context, and open-world entities that existing datasets do not adequately cover.
3. **Insufficient multimodal integration**: In practice, farmers send crop photos, describe symptoms, and provide geographic and temporal metadata, yet existing benchmarks rarely integrate text, images, and metadata simultaneously.
4. **Unevaluated interactive decision-making**: Experts must determine whether to answer directly or first ask clarifying questions—a clarify-or-respond decision capability that has not been systematically evaluated.
5. **Open-world generalization challenge**: Existing classification datasets adopt closed-category taxonomies, whereas real-world scenarios involve 7,000+ biological entities (including many rare species), leaving model generalization ability unknown.
6. **High risk of erroneous outputs**: Inaccurate recommendations from agricultural AI can have serious consequences (e.g., misdiagnosis of disease, incorrect pesticide application), underscoring the necessity of rigorous evaluation frameworks.

## Method

### Overall Architecture

- **Function**: Construct the MIRAGE benchmark, comprising two evaluation tasks—MMST (multimodal single-turn) and MMMT (multimodal multi-turn)—covering entity identification, management recommendation generation, and clarification decision-making.
- **Motivation**: Traditional benchmarks lack authentic interactive scenarios, domain-expert annotations, and open-world settings, preventing comprehensive evaluation of VLMs in knowledge-intensive domains.
- **Approach**: Over 218,000 interaction records are collected from the AskExtension platform and processed through a four-stage pipeline (cleaning → categorization → formatting → splitting) to produce a standard subset (8,184 instances), a contextual subset (3,934 instances), and a multi-turn dialogue set (861 conversations).

### Key Designs

#### MMST: Single-Turn Reasoning Task
- **Function**: Given a user query, image(s), and metadata, the model must generate a structured response covering entity identification (ID) and management guidance (MG).
- **Motivation**: To assess models' ability to perform causal reasoning from visual symptoms and generate actionable recommendations, simulating real-world long-form visual question answering.
- **Approach**: Data are split into a standard subset (self-contained queries) and a contextual subset (requiring inference of implicit temporal, spatial, or agricultural background information). The ID task is evaluated with binary matching accuracy; the MG task is scored along four dimensions (Accuracy / Relevance / Completeness / Parsimony).

#### MMMT: Multi-Turn Decision Task
- **Function**: Across multi-turn dialogue, the model must determine whether to issue a clarification (Clarify) or a direct response (Respond) given the current context, and generate the corresponding utterance.
- **Motivation**: In real expert consultations, user information is often incomplete; experts must proactively identify knowledge gaps and guide the conversation—a core capability for AI assistants.
- **Approach**: Multi-turn dialogues are truncated at a given user utterance, with subsequent user replies serving as "revealed facts" to reconstruct decision points. GPT-4o is used to produce structured annotations. Evaluation covers decision accuracy and the goal-relevance of generated content.

#### Evaluation Framework
- **Function**: Employs a multi-reasoning LLM ensemble judging scheme in place of single-model, single-pass evaluation.
- **Motivation**: A single judge exhibits systematic bias; ensemble judging improves reliability and reproducibility.
- **Approach**: Three reasoning LLMs—DeepSeek-R1-Distilled, Qwen3-32B, and Phi-4-Reasoning—serve as judges. Each sample undergoes 3 generations × 3 judges = 9 evaluations; inter-rater agreement is validated with Fleiss' $\kappa$ (0.75–0.88).

## Key Experimental Results

### Main Results: MMST Single-Turn Identification and Management

| Model | ID Acc (%) | Reasoning | MG-Acc | MG-Rel | MG-Comp | MG-Pars | W-Sum |
|---|---|---|---|---|---|---|---|
| GPT-4.1 | **43.9** | **3.01** | **3.24** | **3.60** | **3.22** | **3.01** | **0.82** |
| Claude-3.7-Sonnet | 33.9 | 2.64 | 2.82 | 3.23 | 2.69 | 2.88 | 0.72 |
| Qwen2.5-VL-72B | 29.8 | 2.47 | 2.72 | 3.09 | 2.56 | 2.61 | 0.69 |
| Qwen2.5-VL-32B | 25.1 | 2.43 | 2.87 | 3.19 | 2.88 | 2.43 | 0.71 |
| InternVL3-78B | 22.4 | 2.24 | 2.60 | 2.98 | 2.31 | 2.87 | 0.67 |
| LLaVA-v1.6-7B | 7.1 | 1.34 | 2.20 | 2.50 | 1.86 | 2.20 | 0.55 |

**Key Findings**: Even the strongest model, GPT-4.1, achieves only 43.9% identification accuracy, while the best open-source model falls below 30%, demonstrating the extreme difficulty of open-world agricultural entity identification. The gap between closed-source and open-source models is approximately 14 percentage points.

### Main Results: MMMT Multi-Turn Decision

| Model | Zero-Shot Acc% | CoT Acc% | Clarify Rel | Respond Rel |
|---|---|---|---|---|
| GPT-4o | 62.98 | **65.50** | 72.80 | 78.50 |
| Claude-3.7-Sonnet | 57.80 | 62.40 | 34.90 | 28.70 |
| LLaMA-4-Maverick | 53.75 | 59.80 | 69.10 | 74.20 |
| Qwen-72B | 31.33 | 37.40 | 63.90 | 76.50 |

**Key Findings**: Under partially observable conditions, even GPT-4o achieves only approximately 63–65% decision accuracy, indicating that inferring users' implicit goals and knowledge gaps remains a core challenge. Chain-of-thought prompting consistently improves performance across all models (+2.5%–+6%).

### Fine-Tuning Experiments

After LoRA fine-tuning of Qwen2.5-VL-3B, accuracy on seen entities improves from 22.3% to 28.4%, yet accuracy on unseen entities reaches only 14.6%, yielding a generalization gap of approximately 14 percentage points. The 32B model peaks at 37.6%.

## Highlights & Insights

- The first agricultural multimodal benchmark grounded in 35,000+ real expert dialogues covering 7,000+ biological entities, offering exceptionally high ecological validity.
- Introduces a clarify-or-respond decision evaluation dimension that transcends the conventional QA paradigm.
- The three-judge ensemble framework achieves high inter-rater agreement ($\kappa$ = 0.75–0.88) and yields a reusable evaluation scheme.
- The three-subset design (standard / contextual / multi-turn) progressively reveals model weaknesses across distinct capability dimensions.

## Limitations & Future Work

- Data are biased toward small-scale farming and home gardening in the United States, without coverage of large-scale industrial agriculture.
- The MMMT task does not simulate genuinely dynamic interaction (e.g., user simulator dialogue), evaluating only single-step decision-making.
- Metadata (temporal and geographic) contributes limited performance gains (only +1.6% for GPT-4.1), suggesting that current models fail to effectively leverage such information.
- Visual follow-ups in multi-turn dialogue are not modeled; all turns after the first are assumed to be text-only.

## Related Work & Insights

| Dimension | MIRAGE | AgMMU |
|---|---|---|
| Data Source | Real expert–user dialogues | Synthetic short-text multiple choice |
| Task Type | Open-ended long-form QA + multi-turn decision | Closed-form multiple choice |
| Multimodality | Image + text + metadata | Image + text |
| Biological Entity Coverage | 7,000+ species | Limited categories |
| Evaluation Dimensions | Identification + reasoning + decision + generation quality | Accuracy |

| Dimension | MIRAGE | CROP |
|---|---|---|
| Data Source | AskExtension real dialogues | Two-crop dataset |
| Multimodal Support | ✓ (images) | ✗ (text only) |
| Crop Coverage | Thousands of crops/pests/diseases | Only 2 crops |
| Multi-Turn Dialogue | ✓ (861 conversations with decision annotations) | ✓ (but no decision dimension) |

## Rating

- ⭐⭐⭐⭐ **Novelty**: First benchmark to incorporate clarification decision-making into multimodal evaluation, filling an important gap.
- ⭐⭐⭐⭐ **Technical Quality**: Rigorous data filtering pipeline, well-designed three-judge ensemble scheme, systematic evaluation of 22 models.
- ⭐⭐⭐⭐ **Value**: Provides direct guidance for agricultural AI system development; benchmark and code are publicly available.
- ⭐⭐⭐ **Writing Quality**: Paper structure is clear, but the volume of content is large; some experimental details require reference to the appendix.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness](mmperspective_do_mllms_understand_perspective_a_comprehensive_benchmark_for_pers.md)
- [\[ICLR 2026\] Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading](../../ICLR2026/multimodal_vlm/decoding_open-ended_information_seeking_goals_from_eye_movements_in_reading.md)
- [\[ICLR 2026\] LiveWeb-IE: A Benchmark For Online Web Information Extraction](../../ICLR2026/multimodal_vlm/liveweb-ie_a_benchmark_for_online_web_information_extraction.md)
- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [\[NeurIPS 2025\] FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models](flowcut_rethinking_redundancy_via_information_flow_for_effic.md)

</div>

<!-- RELATED:END -->
