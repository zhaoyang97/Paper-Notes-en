---
title: >-
  [Paper Note] CulturalBench: A Robust, Diverse, and Challenging Cultural Benchmark by Human-AI CulturalTeaming
description: >-
  [ACL 2025][LLM Evaluation][Cultural Knowledge] CulturalBench is constructed through a Human-AI CulturalTeaming pipeline, comprising 1,696 human-written and five-way independently verified cultural knowledge questions across 45 global regions and 17 themes. CulturalBench-Hard (True/False format) yields only 61.5% accuracy even for the strongest model (OpenAI o1), far below the human performance of 92.4%, revealing models' mode-seeking tendencies in multi-answer questions and i…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Cultural Knowledge"
  - "Human-AI Co-red Teaming"
  - "Multi-region Coverage"
  - "Mode-seeking Bias"
  - "True/False Evaluation"
date: 2026-05-08
content_hash: d508753bdcda25c8
---

# CulturalBench: A Robust, Diverse, and Challenging Cultural Benchmark by Human-AI CulturalTeaming

**Conference**: ACL 2025  
**arXiv**: [2410.02677](https://arxiv.org/abs/2410.02677)  
**Code**: [HuggingFace](https://hf.co/spaces/kellycyy/CulturalBench)  
**Area**: Cultural Knowledge Evaluation / LLM Benchmarks  
**Keywords**: Cultural Knowledge, Human-AI Co-red Teaming, Multi-region Coverage, Mode-seeking Bias, True/False Evaluation

## TL;DR

CulturalBench is constructed through a Human-AI CulturalTeaming pipeline, comprising 1,696 human-written and five-way independently verified cultural knowledge questions across 45 global regions and 17 themes. CulturalBench-Hard (True/False format) yields only 61.5% accuracy even for the strongest model (OpenAI o1), far below the human performance of 92.4%, revealing models' mode-seeking tendencies in multi-answer questions and imbalanced performance in cross-regional cultural knowledge.

## Background & Motivation

The uneven cultural representation of LLMs is a long-standing issue, but constructing high-quality cultural knowledge benchmarks faces multiple challenges:

**Insufficient Robustness of Existing Benchmarks**:
   - Inadequate quality verification: Most benchmarks only perform quality checks during intermediate data collection steps rather than verifying the entire final dataset.
   - Over-reliance on web data sources: Sources like Wikipedia might have already been seen by models during pre-training.
   - Risk of bias propagation in LLM-generated benchmarks.

**Narrow Topic Coverage**:
   - Most benchmarks rely on pre-defined topics (e.g., food, dating), which fail to capture cultural elements unique to different regions.
   - Covering only 1-12 topics, lacking diversity.

**Limitations of Evaluation Formats**:
   - Multiple-choice formats allow models to achieve accuracy far exceeding random guessing (40.4% vs. 25% random) using heuristic methods (e.g., embedding similarity between options and country names) without actually understanding the question content.
   - Models might be guessing rather than demonstrating true cultural understanding.

CulturalBench aims to address these issues by constructing a robust, diverse, and challenging benchmark.

## Method

### Overall Architecture

The CulturalTeaming data collection pipeline consists of three steps:
1. Red-teaming data collection (human-AI collaboration)
2. Human quality verification (five-person independent verification)
3. Majority vote filtering

### Key Designs

#### 1. Human-AI Red-Teaming Data Collection

- **Function**: Guides human annotators to iteratively propose cultural questions that challenge models.
- **Mechanism**:
    - **Question Construction**: Annotators brainstorm culture-related scenarios based on their own cultural experiences (e.g., "Singaporeans using tissues to reserve seats"), and an AI assistant converts these scenarios into structured multiple-choice questions with four options.
    - **Question Verification & Refinement**: Annotators challenge an AI validator on an interactive platform using the constructed questions. The platform provides refinement strategies and examples (e.g., "question reversal") to make the questions more challenging.
    - **Internal Filtering**: Researchers filter out questions unrelated to specific regions from over 3,600 questions, retaining more than 3,000.
- **Design Motivation**: Adopts the concept of AI safety red-teaming to collect challenging data through human-AI competition.
- **Discovery-based Topic Approach**: Does not pre-define topic sets, encouraging annotators to freely explore based on their personal experiences.

#### 2. Five-Person Independent Human Quality Verification

- **Function**: Every question is verified by 5 independent annotators.
- **Mechanism**:
    - Recruit via the Prolific platform, requiring annotators' nationality and primary residence before age 18 to match the region associated with the question.
    - Adopt a multi-label selection setting: Annotators can select multiple correct answers.
    - Provide additional "no correct option" and "no relevant knowledge" options to prevent guessing.
- **Design Motivation**: The correctness of cultural knowledge is difficult to verify, necessitating expert-level human verification for the entire final dataset.
- Majority vote threshold: $\ge 4/5$ annotator agreement.

#### 3. Dual-Format Benchmark Construction

**CulturalBench-Easy (Multiple-Choice Questions)**:
- 1,696 four-option multiple-choice questions.
- Single-mode questions (one correct answer): Used directly.
- Multi-mode questions (multiple correct answers): Restructured into compound options (e.g., "A. (i) and (iv)") with instructions to "select all that apply".

**CulturalBench-Hard (True/False)**:
- $1,696 \times 4 = 6,784$ binary classification questions.
- Each of the four options from the original question becomes a True/False question.
- A question is considered correctly answered only if all four decisions are correct.
- Random baseline: $0.5^4 = 6.25\%$

### Topic Discovery

Through GPT-4o classification, 17 topics are identified, falling into three major categories:
- **Daily Life**: Food, workplace, etc.
- **Social Etiquette**: Greetings, social norms, etc.
- **Broader Society**: Celebrations, religion, etc.

Annotators from different regions focus on different topics: Italians lean toward food (38.9%), while Israelis focus on religion (23.8%).

## Key Experimental Results

### Main Results: Performance of 29 LLMs on CulturalBench-Hard

| Model | CulturalBench-Easy | CulturalBench-Hard |
|------|-------------------|-------------------|
| **Human** | **92.4%** | **92.4%** |
| Random | 25.0% | 6.25% |
| OpenAI o1 | 89.6% | 61.5% |
| GPT-4o | - | 60.4% |
| Claude 3.5 Sonnet | - | ~56% |
| Llama-3.1-70B | - | 54.6% |
| Llama-3.1-8B | - | 36.0% |
| GPT-3.5 Turbo | - | 34.5% |
| Cohere Aya-8b | - | 28.7% |

The gap between the best model and humans on the Hard version is 30.9 percentage points.

### Ablation Study: Question Type Analysis

| Question Type | Model Average Accuracy | Best Model (o1) | Human |
|---------|-------------|------------|------|
| Single-mode (1 correct answer, $N=1554$) | 49.6% | ~65% | ~95% |
| Multi-mode (multiple correct answers, $N=142$) | 20.9% | ~20% | ~89% |
| **Gap** | **28.7%** | **45.5%** | **6.1%** |

Models' performance drops precipitously on multi-answer questions, whereas human performance decreases only slightly.

### Regional Performance Differences

| Region | Model Average Accuracy |
|------|-------------|
| North America | 57.9% |
| Northern Europe | 51.8% |
| South Asia | 51.5% |
| South America | 41.5% |
| Eastern Europe | 41.5% |
| Middle East / Western Asia | 37.8% |

### Heuristic Baseline Analysis

| Method | CulturalBench-Easy Accuracy |
|------|-------------------------|
| Random Guessing | 25.0% |
| Option vs. Country Name Embedding Similarity | 40.4% |
| Best Model | 89.6% |

Even without the question, an accuracy of 40.4% can be reached solely based on the similarity between options and country names, indicating that the multiple-choice format in the Easy version contains shortcuts.

### Key Findings

1. **CulturalBench-Hard is highly challenging**: The best model achieves only 61.5%, far below the human performance of 92.4%.
2. **Multiple-choice format contains shortcuts**: The embedding similarity heuristic achieves 40.4%, indicating that the Easy version may overestimate LLMs' cultural knowledge.
3. **Models' mode-seeking tendencies**: Models perform extremely poorly on multi-answer questions ($-28.7\%$), tending to overfit to a single most likely answer.
4. **Positive correlation with model size**: Within the same family, larger models perform better.
5. **Imbalanced regional performance**: North America, Northern Europe, and South Asia show better performance compared to South America, Eastern Europe, and the Middle East.
6. **Lack of cultural advantage for local providers**: Qwen/DeepSeek in East Asia and Mistral in Western Europe do not outperform GPT-4o.
7. **Performance ceiling**: Improvements across versions within the same model family are becoming increasingly smaller, potentially approaching a performance bottleneck.

## Highlights & Insights

- **Human-AI CulturalTeaming Pipeline**: Creatively applies the concept of AI safety red-teaming to cultural knowledge benchmark construction.
- **Five-way Comprehensive Verification**: 100% of final questions are verified by five independent annotators, providing quality assurance that far exceeds similar work.
- **Discovery-based Topic Approach**: Does not pre-set topics, allowing annotators to freely explore, thereby capturing 17 diverse themes.
- **Exquisite Hard Version Design**: The True/False format effectively eliminates heuristic shortcuts inherent in multiple-choice questions.
- **Multi-answer Questions Reveal Mode-Seeking Bias**: Exposes the fundamental weakness of LLMs in handling cultural diversity.

## Limitations & Future Work

1. **English Only**: The performance of models on cultural knowledge in local languages is not evaluated, potentially omitting scenarios of "understanding the language but not the culture".
2. **Small Verifier Sample Size**: In some underrepresented regions (e.g., Bangladesh), active annotators on Prolific number fewer than 30, limiting recruitment to just 5 people.
3. **Coarse Country/Region Granularity**: Cultural diversity within the same country (e.g., Wales vs. England in the UK) is not fully captured.
4. **Annotator Representativeness Issues**: Due to limitations of the Prolific platform, certain cultural perspectives might be over- or under-represented.
5. **No Multimodal Testing**: Limited to text-only formats, omitting visual cultural knowledge.

## Related Work & Insights

- Systematically compares with cultural benchmarks such as FORK, BERTAQA, CVQA, NormAd, and Blend.
- CulturalBench leads comprehensively across three dimensions: verification coverage (100%), theme diversity (17 themes), and challenging nature (best model at 61.5%).
- The human-AI collaborative red-teaming paradigm can be generalized to the construction of other highly subjective evaluation benchmarks.
- The True/False evaluation format also serves as a reference for assessing other multiple-choice benchmarks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The CulturalTeaming pipeline is novel, and the Hard version design is clever)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (29 models, regional analysis, question type analysis, heuristic baseline analysis, temporal version analysis)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure, comprehensive analysis, and sufficient comparison with related work)
- **Value**: ⭐⭐⭐⭐⭐ (High-quality open-source benchmark, reveals systematic weaknesses in LLM cultural knowledge, and the methodology is reusable)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ChatBench: From Static Benchmarks to Human-AI Evaluation](chatbench_from_static_benchmarks_to_human-ai_evaluation.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)
- [\[ICML 2026\] From Human-Level AI Tales to AI Leveling Human Scales](../../ICML2026/llm_evaluation/from_human-level_ai_tales_to_ai_leveling_human_scales.md)
- [\[ACL 2025\] WiCkeD: A Simple Method to Make Multiple Choice Benchmarks More Challenging](wicked_a_simple_method_to_make_multiple_choice_benchmarks_more_challenging.md)

</div>

<!-- RELATED:END -->
