---
title: >-
  [Paper Note] Evaluating Language Models as Synthetic Data Generators
description: >-
  [ACL 2025][LLM (Other)][Synthetic data] This work proposes AgoraBench, a benchmark to systematically evaluate the data generation capabilities of 6 LLMs across 3 domains $\times$ 3 data generation methods. By training 99 student models, the study reveals that the data generation capabilities of LLMs are not directly correlated with their problem-solving abilities; GPT-4o performs best in instance generation, while Claude-3.5-Sonnet excels in quality enhancement.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Synthetic data"
  - "Data generation"
  - "Benchmark"
  - "LLM evaluation"
  - "Training data quality"
date: 2026-05-08
content_hash: ed9b8ded73db1964
---

# Evaluating Language Models as Synthetic Data Generators

**Conference**: ACL 2025  
**arXiv**: [2412.03679](https://arxiv.org/abs/2412.03679)  
**Code**: [https://github.com/neulab/data-agora](https://github.com/neulab/data-agora)  
**Area**: LLM/NLP  
**Keywords**: Synthetic data, Data generation, Benchmark, LLM evaluation, Training data quality

## TL;DR
This work proposes AgoraBench, a benchmark to systematically evaluate the data generation capabilities of 6 LLMs across 3 domains $\times$ 3 data generation methods. By training 99 student models, the study reveals that the data generation capabilities of LLMs are not directly correlated with their problem-solving abilities; GPT-4o performs best in instance generation, while Claude-3.5-Sonnet excels in quality enhancement.

## Background & Motivation
**Background**: Synthetic data has been widely used for LLM post-training. Methods such as Self-Instruct, Alpaca, and WizardLM have demonstrated the effectiveness of different data generation strategies, but experimental setups lack uniformity—employing different generator models, data volumes, base models, and evaluation benchmarks.

**Limitations of Prior Work**: There is a lack of controlled variable comparative studies, making it impossible to determine whether the improvements in student models stem from the data generation method itself or the generator model. Various API providers claim their models are suitable for generating training data, yet systematic verification is lacking.

**Key Challenge**: Does the intuitive assumption that "a good problem solver is definitely a good data generator" hold true? If not, what factors determine the quality of data generation?

**Goal**: To systematically compare the data generation capabilities of different LLMs under a unified setup, revealing key factors that influence data quality.

**Key Insight**: By fixing all variables (seed data, prompt templates, data volume, student models, evaluation benchmarks) and only varying the data generator, the training effect of the generated data is measured using the standardized metric PGR.

**Core Idea**: The first standardized benchmark of LLM data generation capabilities, revealing that data generation capability is not equivalent to problem-solving capability.

## Method

### Overall Architecture
AgoraBench covers 3 domains (math / code / instruction following) $\times$ 3 data generation methods (instance generation / response generation / quality enhancement) = 9 settings. In each setting, 6 LLMs each generate 10K training samples, which are then used to train Llama-3.1-8B as the student model and evaluated on fixed benchmarks.

### Key Designs

1. **Performance Gap Recovered (PGR) Metric**:

    - **Function**: Measures the training effect of the student model relative to the reference model
    - **Mechanism**: $\text{PGR}(G,B) = \frac{\text{score}_B(S_{D_G}) - \text{score}_B(S_\emptyset)}{\text{score}_B(S_{ref}) - \text{score}_B(S_\emptyset)} \times 100$, where $S_\emptyset$ is the base model (Llama-3.1-8B) and $S_{ref}$ is the reference model (Llama-3.1-8B-Instruct)
    - **Design Motivation**: A PGR of 50% means that using only 10K synthetic data recovers half the effectiveness of Meta's training with 10M+ human data. PGR reflects the actual training value of the data more directly than intrinsic metrics (such as response quality scores)

2. **Three Data Generation Methods**:

    - **Instance Generation**: Expanding to generate new instruction-response pairs from a small amount of seed data—the Self-Instruct approach
    - **Response Generation**: Generating corresponding responses given a set of instructions—the most common distillation paradigm
    - **Quality Enhancement**: Improving instructions and/or responses given existing low-quality instances—the WizardLM approach
    - **Design Motivation**: Covers the three most commonly used synthetic data strategies in practical scenarios

3. **Analysis of Intrinsic Quality Features**:

    - **Function**: Studies which intrinsic data quality features can predict PGR
    - **Mechanism**: Measures multiple intrinsic metrics (response quality, instruction difficulty, perplexity, diversity, etc.) and performs PCA to analyze their relationship with PGR
    - **Key Findings**: The top-5 principal components account for 93.4% of the variance in PGR—indicating that data quality is multidimensional and a single metric is insufficient

### Loss & Training
- Student models are trained using standard SFT, calculating loss only on response tokens
- No data filtering or augmentation is applied, directly using raw generated data to evaluate "naked" data generation capability
- In total, 1.26 million training samples were generated, training 99 student models

## Key Experimental Results

### Main Results (Average PGR)

| Data Generator | API Cost (In/Out) | Problem Solving | Data Generation PGR |
|-----------|-------------|---------|-------------|
| **GPT-4o** | $2.5/$10 | 80.9 | **29.5%** |
| Claude-3.5-Sonnet | $3/$15 | 80.5 | 23.6% |
| GPT-4o-mini | $0.15/$0.6 | 75.4 | 19.2% |
| Llama-3.1-8B | $0.055 | 50.2 | 15.9% |
| Llama-3.1-70B | $0.35/$0.4 | 69.6 | 14.1% |
| Llama-3.1-405B | $1.79 | 75.0 | 11.3% |

### Performance Variance by Method

| Method | Best Generator | PGR | Runner-Up | PGR |
|------|----------|-----|-------|-----|
| Instance Generation | GPT-4o | 46.8% | Claude | 24.1% |
| Response Generation | GPT-4o | 35.2% | Claude | 33.0% |
| Quality Enhancement | Claude-3.5-Sonnet | 17.9% | GPT-4o | 6.7% |

### Key Findings
- **Data generation capability $\neq$ problem-solving capability**: Llama-3.1-8B (the weakest in problem solving) achieves a PGR of 55.7% in code instance generation, outperforming Claude-3.5-Sonnet's 23.4%
- GPT-4o leads by a large margin in instance generation (+46.8%) but performs mediocrely in quality enhancement (+6.7%)
- Claude-3.5-Sonnet is the best choice for quality enhancement, but underperforms compared to GPT-4o in instance generation
- The top-5 principal components of intrinsic quality metrics explain 93.4% of the variance in PGR—indicating the necessity of evaluating data quality multidimensionally
- Output format (Markdown vs. plain text vs. JSON) has a significant impact on PGR—different tasks prefer different formats
- **Value**: Llama-3.1-8B's PGR (15.9%) is close to Llama-405B (11.3%) at only 1/50 of GPT-4o's cost

## Highlights & Insights
- **"A good problem solver is not necessarily a good problem designer"** is the core insight—this challenges the common assumption and offers direct guidance for practitioners in selecting data generation models.
- **Elegant design of the PGR metric**—using the full post-training of Llama-3.1-8B-Instruct as a 100% reference, it clearly and intuitively represents what proportion of the performance can be recovered with only 10K data.
- The experimental scale of **1.26 million samples + 99 student models** ensures the reliability of the findings.
- Intrinsic quality analysis reveals that data quality is a multidimensional concept—one cannot simply evaluate "whether the response is correct," but must also consider difficulty, diversity, perplexity, etc.

## Limitations & Future Work
- Only Llama-3.1-8B is used as the student model; student models of different scales/architectures may exhibit different preferences.
- SFT is the sole training method; conclusions may vary under other training paradigms such as DPO or RLHF.
- The data volume is fixed at 10K, leaving scaling effects unexplored.
- Scenarios involving mixed training from multiple data sources are not considered.
- PGR is highly dependent on the choice of the reference model.

## Related Work & Insights
- **vs. Self-Instruct/Alpaca**: They validate the effectiveness of specific data generation methods, but ours is the first to compare different generators under a unified setup.
- **vs. Xu et al. (2024c)**: Concurrent work only investigates response generation, whereas this study covers three data generation paradigms and analyzes intrinsic quality.
- **vs. DataComp/HELM**: Shares a similar standardized evaluation methodology but applies it to data generation instead of model capabilities.
- Serves as an important guide for practical data production—different scenarios warrant different generator choices.

## Rating
- Novelty: ⭐⭐⭐⭐ The first systematic benchmark of LLM data generation capabilities, featuring an elegantly designed PGR metric.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 1.26 million samples + 99 models + multidimensional analysis, highly comprehensive scale.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear diagrams, well-organized findings, and in-depth analysis.
- Value: ⭐⭐⭐⭐⭐ Direct practical guidance value for synthetic data pipelines, with findings being counterintuitive yet verifiable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](difflm_controllable_synthetic_data_generation_via_diffusion_language_models.md)
- [\[ACL 2025\] Theorem Prover as a Judge for Synthetic Data Generation](theorem_prover_as_a_judge_for_synthetic_data_generation.md)
- [\[ACL 2025\] From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts](from_data_to_knowledge_evaluating_how_efficiently_language_models_learn_facts.md)
- [\[NeurIPS 2025\] Valid Inference with Imperfect Synthetic Data](../../NeurIPS2025/llm_nlp/valid_inference_with_imperfect_synthetic_data.md)
- [\[ACL 2025\] Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models](genetic_instruct_scaling_up_synthetic_generation_of_coding_instructions_for_larg.md)

</div>

<!-- RELATED:END -->
