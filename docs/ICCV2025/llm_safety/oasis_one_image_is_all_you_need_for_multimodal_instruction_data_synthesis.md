---
title: >-
  [Paper Note] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis
description: >-
  [ICCV 2025][LLM Safety][multimodal data synthesis] This paper proposes Oasis, a method that induces MLLMs to autoregressively generate high-quality multimodal instruction-following data using only an input image (without…
tags:
  - "ICCV 2025"
  - "LLM Safety"
  - "multimodal data synthesis"
  - "instruction-following data"
  - "quality control"
  - "LLaVA"
  - "MLLM"
date: 2026-05-08
content_hash: 02d9c58b3dfd43b1
---

# Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis

**Conference**: ICCV 2025
**arXiv**: [2503.08741](https://arxiv.org/abs/2503.08741)
**Code**: [https://github.com/Letian2003/MM_INF](https://github.com/Letian2003/MM_INF)
**Area**: AI Safety / Multimodal Data Synthesis
**Keywords**: multimodal data synthesis, instruction-following data, quality control, LLaVA, MLLM

## TL;DR

This paper proposes Oasis, a method that induces MLLMs to autoregressively generate high-quality multimodal instruction-following data using only an input image (without any text prompt). Combined with a fine-grained instruction quality control mechanism, synthesizing 500K samples yields an average 3.1% overall performance gain for LLaVA-NeXT, surpassing other data synthesis methods.

## Background & Motivation

The success of multimodal large language models (MLLMs) relies heavily on large-scale training data, yet three major bottlenecks exist:

**Data unavailability**: Training data of top-tier MLLMs is not publicly released due to privacy concerns.

**High collection cost**: Multimodal data annotation is expensive and labor-intensive.

**Limitations of existing synthesis methods**:
- Fixed pipelines and invariant prompts constrain data diversity.
- Insufficient quality control makes it difficult to generate data that genuinely improves model representation capacity.
- Complex frameworks require extensive manual design of data patterns and prompts.

Core insight: Inspired by Magpie (prompt-free synthesis for text), given that the autoregressive nature of MLLMs can produce diverse outputs, can an MLLM automatically generate instruction data when provided with only an image?

## Method

### Overall Architecture

The Oasis pipeline consists of three steps:
1. **Data synthesis**: A "hooking prompt" is used to induce the MLLM to generate instructions.
2. **Data categorization**: An LLM filters instruction-following data and removes purely descriptive data.
3. **Instruction quality control**: Multi-dimensional scoring filters out low-quality instructions.

### Key Designs

1. **Hooking Prompt Data Synthesis**

   A conventional MLLM input consists of four components: pre-query template + visual content + instruction + post-query template. The core operation in Oasis is to **remove the instruction and post-query template**, retaining only the pre-query template and the image:

   $\text{Inst} = \Theta(\text{vision})$

   rather than the conventional $\text{Resp} = \Theta(\text{vision}, \text{instruction})$.

   Since only an image is provided, the MLLM autoregressively generates diverse instructions based on its own knowledge. The absence of manually crafted text prompts means:
   - Generated instructions are not constrained by fixed-prompt biases.
   - Coverage naturally spans 46 languages (LLaVA-NeXT covers only English).
   - Root verb and noun-object distributions are more naturally diverse.

2. **Data Categorization**

   Among generated data, approximately 49.9% are descriptive (caption) and 50.1% are instruction-following. An LLM is used as a few-shot classifier to distinguish the two categories:
   - Instruction-following data is retained and instructions are extracted.
   - Descriptive data is first filtered out, then partially recovered via rule-based selection and LLM cleaning to yield 250K high-quality captions.

3. **Instruction Quality Control**

   Four characteristic dimensions of high-quality instructions are identified, each scored on a 1–5 scale:
   - **Solvability**: Whether the image provides sufficient information to answer the question.
   - **Clarity**: Whether the question precisely conveys its intent.
   - **Hallucination**: The alignment between question content and the actual image content.
   - **Nonsense**: Grammatical correctness and semantic coherence.

   The first three dimensions are evaluated by an MLLM (requiring visual information); the fourth is evaluated by an LLM (more sensitive to linguistic quality). The pass rate for high-quality instructions is approximately 50.9%.

### Loss & Training

Standard two-stage LLaVA-NeXT training is adopted:
- **Pre-training**: LLaVA-Pretrain-558K, training only the randomly initialized projector.
- **Fine-tuning**: LLaVA-NeXT official SFT data + Oasis synthesized data, with all parameters trainable.

Training details: AdamW optimizer, cosine learning rate schedule, warmup ratio 0.03, batch size 128. Pre-training LR = 1e-3, fine-tuning LR = 1e-5.

Synthesis tools: Qwen2.5-VL-72B-Instruct (MLLM) + Qwen2.5-72B-Instruct (LLM); images sourced from Cambrian-10M.

## Key Experimental Results

### Main Results

Performance comparison of LLaVA-NeXT on 14 benchmarks (Vicuna-7B-v1.5 backbone):

| Method | MMBench | MME | MMStar | MMVet | DocVQA | TextVQA | OCRBench | Avg. |
|--------|---------|-----|--------|-------|--------|---------|----------|------|
| Baseline | 64.2/54.4 | 1482/291 | 37.1 | 28.0 | 71.7 | 63.4 | 52.9 | 53.0 |
| +LLaVA (upsampled) | 64.8/54.9 | 1461/353 | 37.6 | 34.3 | 67.8 | 64.0 | 52.6 | 53.7 |
| +DenseFusion | 67.4/56.2 | 1523/333 | 37.8 | 30.2 | 69.2 | 65.4 | 55.4 | 54.3 |
| +Cambrian | 66.8/56.6 | 1504/329 | 37.8 | 32.4 | 73.8 | 63.7 | 52.3 | 54.9 |
| +MMEvol | 63.6/53.8 | 1503/316 | 32.3 | 34.9 | 64.7 | 62.8 | 51.7 | 51.6 |
| **+Oasis** | **65.6/56.7** | **1532/357** | **38.0** | **37.2** | **76.0** | **66.1** | **55.0** | **56.1** |

Oasis yields significant improvements across three backbones: Vicuna +3.1%, Qwen2.5 +1.8%, Llama3 +3.2%.

### Ablation Study

**Effect of instruction quality control** (200K data comparison):

| Configuration | DocVQA | InfoVQA | TextVQA | Avg. |
|---------------|--------|---------|---------|------|
| With quality control | **74.8** | **38.7** | **64.5** | **54.9** |
| Without quality control | 67.7 | 31.4 | 63.6 | 53.9 |

Quality control yields an overall 1% improvement, with DocVQA and InfoVQA each gaining over 7%.

**Response quality control is ineffective**: Both NLL sampling and MLLM scoring lead to performance degradation (−0.7% and −1.6%), indicating that high-quality instructions alone elicit strong responses from a state-of-the-art MLLM.

**Data scaling** (added on top of 100K LLaVA data):

| Oasis Data Size | Avg. Score | Gain |
|-----------------|------------|------|
| 0 | 46.5 | — |
| 150K | 46.6 | +0.1 |
| 300K | 47.7 | +1.2 |
| 500K | 51.7 | **+5.2** |

### Key Findings

- **Oasis data is longer and more diverse**: Average instruction length 76.8 vs. 45.2 for LLaVA-NeXT; response length 71.2 vs. 34.2.
- **Multilingual coverage**: Automatically covers 46 languages, making it the first synthetic multimodal dataset with such linguistic diversity.
- **Unbiased root verb distribution**: "Answer question" dominates LLaVA-NeXT data, while Oasis exhibits a more balanced and natural distribution.
- **Domain-specific capability**: In OCR-domain experiments, adding 70K OCR-domain Oasis data yields +3.3% on DocVQA and +3.5% on ChartQA.
- **Caption recovery is valuable**: The 250K cleaned captions recovered from filtered descriptive data outperform the baseline on 12/16 metrics.
- **Instruction quality control is critical**: Response quality control, however, is harmful—high-quality instructions naturally induce high-quality responses.

## Highlights & Insights

1. **Extremely simple method design**: Only one image is needed with no text prompt engineering; the MLLM's own knowledge drives diverse instruction generation.
2. **Insightful quality control**: Instruction quality is identified as the core factor (controlling instructions controls data quality), while response quality control introduces unwanted bias.
3. **Domain controllability**: The source of images directly determines the domain of generated data, enabling domain-specific data production without modifying the pipeline.
4. **Strong scalability**: Data scale grows linearly with image count; a significant ~4% gain is still observed when scaling from 300K to 500K.

## Limitations & Future Work

- Relies on a powerful MLLM such as Qwen2.5-VL-72B as the data generator; data quality from smaller models remains unvalidated.
- Scoring criteria in quality control require manual design and may need adjustment for different domains.
- The scaling law for larger-scale (e.g., million-level) synthesized data remains unexplored.
- Oasis data is primarily validated on the LLaVA-NeXT architecture; generalization to other architectures requires further confirmation.

## Related Work & Insights

- Shares the same prompt-free synthesis philosophy as Magpie for text, representing its natural extension to the multimodal domain.
- The finding that "instruction quality > response quality" provides an important reference for the data synthesis community.
- The image-as-domain property makes constructing targeted datasets straightforward.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of "removing text prompts" is remarkably concise yet effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 14 benchmarks, 3 backbones, 5 synthesis method comparisons, and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, thorough data property analysis, and rich visualizations.
- **Value**: ⭐⭐⭐⭐⭐ Simple, reproducible, open-source data and code; extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Attention Smoothing Is All You Need For Unlearning](../../ICLR2026/llm_safety/attention_smoothing_is_all_you_need_for_unlearning.md)
- [\[ICCV 2025\] Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation](temporal_unlearnable_examples_preventing_personal_video_data_from_unauthorized_e.md)
- [\[ICLR 2026\] Train Once, Answer All: Many Pretraining Experiments for the Cost of One](../../ICLR2026/llm_safety/train_once_answer_all_many_pretraining_experiments_for_the_cost_of_one.md)
- [\[NeurIPS 2025\] PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning](../../NeurIPS2025/llm_safety/pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)
- [\[ICCV 2025\] ChartCap: Mitigating Hallucination of Dense Chart Captioning](chartcap_mitigating_hallucination_of_dense_chart_captioning.md)

</div>

<!-- RELATED:END -->
