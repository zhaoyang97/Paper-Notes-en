---
title: >-
  [Paper Note] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models
description: >-
  [ACL 2026][LLM Evaluation][diffusion LLM] DIA is a training-free format-constrained generation method for diffusion large language models. By predicting the end anchor position before iteratively filling between anchors…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "diffusion LLM"
  - "format constraints"
  - "dynamic anchors"
  - "structured generation"
  - "JSON generation"
date: 2026-05-08
content_hash: 3968ecbda27568ba
---

# Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2606.04535](https://arxiv.org/abs/2606.04535)  
**Code**: https://github.com/Westlake-AGI-Lab/DIA  
**Area**: Diffusion Large Language Models / Format-Constrained Generation  
**Keywords**: diffusion LLM, format constraints, dynamic anchors, structured generation, JSON generation

## TL;DR
DIA is a training-free format-constrained generation method for diffusion large language models. By predicting the end anchor position before iteratively filling between anchors, it significantly improves format accuracy for reasoning templates and JSON outputs while alleviating truncation or redundancy caused by fixed anchors.

## Background & Motivation
**Background**: Unlike autoregressive LLMs, diffusion large language models (dLLMs) use bidirectional attention and parallel denoising for generation. They are naturally capable of pre-filling certain fixed tokens within an initial fully masked sequence. Consequently, they appear well-suited for structured outputs, such as `<think>...</think><answer>...</answer>` or parseable JSON.

**Limitations of Prior Work**: Placing begin/end anchors at fixed positions constrains the format but partitions the generation space into fixed lengths. If the reasoning span is too short, the model truncates early; if the span is too long, the model generates repetitive or redundant content. Prompt constraints, post-processing, and constrained decoding also have drawbacks: prompts are unstable, post-processing may destroy semantics, and strict decoding affects efficiency and flexibility.

**Key Challenge**: Format constraints require stable structural boundaries, but high-quality generation requires variable lengths. Fixed templates bind these two aspects together, making it difficult to achieve both structural correctness and semantic quality.

**Goal**: The authors aim to leverage the dLLM's perception of global mask sequences and end positions to dynamically estimate anchor positions without fine-tuning the model, allowing the model to plan the required length of each structural segment before generating its content.

**Key Insight**: The paper observes that dLLMs can estimate the `eos` or end positions through one or two prediction steps. Therefore, end anchors do not need to be hardcoded in advance and can be found dynamically through single-step prediction and confidence thresholds.

**Core Idea**: The task of format-constrained generation is decomposed into two stages: "length adjustment" and "in-anchor infilling." The method iteratively extends mask blocks until the model predicts an end anchor with high confidence, then fixes the anchors and completes diffusion-based content generation within the boundaries.

## Method
DIA is designed for pre-trained diffusion LLMs and introduces no new parameters. It divides the output sequence into several blocks, each corresponding to a structural segment (e.g., reasoning segment and answer segment). Each segment starts with a begin anchor, followed by a prediction to determine where the end anchor might appear. If the current space is insufficient, the block is extended until a high-confidence end anchor is found or the maximum length is reached.

### Overall Architecture
Given a user query $Q$ and a fully masked target sequence $X_L$, DIA first partitions $X_L$ into blocks $\mathcal{C}=\{C_1,\dots,C_{|\mathcal{B}|}\}$ and places a begin anchor at the start of each block. Stage 1 performs length adjustment for each block: calling the dLLM for a single-step prediction to search for an end anchor or partial anchor. If the confidence exceeds a threshold $c$, the block is truncated and the end anchor is completed; otherwise, it is extended by a step size $\Delta$. Stage 2 performs iterative denoising with infilling after boundaries are fixed, generating specific reasoning or answer content between the fixed anchors.

### Key Designs
1. **Dynamic End Anchor Prediction**:
    - **Function**: Automatically estimates the appropriate length for each structural segment instead of manually fixing the span.
    - **Mechanism**: Performs single-step prediction after placing a begin anchor before the block. If the model predicts an end anchor or a partial end anchor within the block with confidence exceeding a threshold, the current length is considered sufficient; otherwise, $\Delta$ mask tokens are added.
    - **Design Motivation**: dLLMs learn priors about termination positions during pre-training. Leveraging this prior allows planning boundaries before actual content generation, avoiding fixed anchors that are too early or too late.

2. **Left-Priority and Maximum Length Constraint**:
    - **Function**: Avoids repetitive end anchors and infinite expansion.
    - **Mechanism**: When multiple positions meet the confidence threshold, DIA selects the leftmost position as the end anchor and truncates redundant masks following it. If no valid anchor is found, expansion stops at the maximum block length $M$.
    - **Design Motivation**: Structured generation is most compromised when models repeatedly open/close tags in the same segment. Left-priority provides a conservative boundary, while maximum length prevents uncontrollable latency for complex samples.

3. **Sequential In-Anchor Iterative Infilling**:
    - **Function**: Generates semantically coherent content within determined structural boundaries.
    - **Mechanism**: For think-answer tasks, DIA first determines and generates the thinking block, then utilizes the generated reasoning content to decide the length and content of the answering block. Anchors remain fixed during generation, and only the intermediate mask tokens are iteratively updated.
    - **Design Motivation**: The length and content of the answer depend on the reasoning. If two blocks are planned independently, the final answer may deviate from the reasoning; sequential processing passes information from the previous segment to the next.

### Loss & Training
DIA is a training-free method with no additional training loss. Experiments use `Dream-7B-Base-v0` and `Dream-7B-Instruct-v0`, based on modifications to the official code. GSM8K uses 1,319 samples and MATH uses 5,000 samples. Max new tokens are 256 for GSM8K and 512 for MATH. Confidence thresholds are 0.065 and 0.05, respectively. Expansion size $\Delta=4$, max block length $M=512$, diffusion steps are 512, and batch sizes are 1 and 3. The experimental environment consists of PyTorch 2.5.1, Python 3.10, and NVIDIA vGPU 32G/48G.

## Key Experimental Results

### Main Results
| Dataset | Metric | Ours (DIA) | Prev. Method | Gain / Note |
|--------|------|------|----------|------|
| GSM8K 0-shot | Format Score | 72.63 | Infilling: 58.83, Base/Instruct: 0.00 | Significant improvement in format accuracy |
| GSM8K 0-shot | Accuracy | 46.78 | Infilling: 14.86, Instruct: 15.01, Base: 68.99 | Significant improvement over fixed infilling, but lower than unformatted Base |
| MATH-500 0-shot | Format Score | 76.82 | Infilling: 29.10, Base/Instruct: 0.00 | Maximum format benefit in complex math scenarios |
| MATH-500 0-shot | Accuracy | 20.08 | Infilling: 21.52, Base: 25.14, Instruct: 25.28 | Maintains comparable but not highest accuracy |
| WikiBio JSON | Valid JSON / Hallucination | 79.84 / 0.15 | Instruct raw: 52.80 / 4.81, Infilling: 0.01 / 0.00 | Consistent results under raw matching and regex extraction |

### Ablation Study
| Configuration | Key Metric | Note |
|------|---------|------|
| DIA w/o Stage 1, GSM8K | Acc. 10.31, Format 0.00, Latency 14.99 | Format nearly collapses without confidence prediction |
| DIA Full, GSM8K | Acc. 47.54, Format 59.67, Latency 25.86 | Complete method is significantly better under appendix hyperparameter settings |
| DIA w/o Stage 1, MATH | Acc. 6.73, Format 0.84, Latency 15.33 | Complex tasks rely more on length planning |
| DIA Full, MATH | Acc. 20.20, Format 75.62, Latency 29.37 | Complete two-stage method preserves structural boundaries |
| GSM8K latency | DIA 26.52 vs Base 10.72 | Dynamic planning introduces additional latency |
| MATH latency | DIA 30.62 vs Base 31.71 | Proactive length planning reduces redundant computation on complex tasks |

### Key Findings
- Fixed infilling can partially preserve anchors but cannot guarantee overall format correctness and suppresses answer quality. On GSM8K, fixed infilling accuracy is only 14.86, while DIA reaches 46.78.
- DIA's performance in JSON generation is most stable. In WikiBio, both raw matching and regular expressions achieve 79.84% valid JSON, with a hallucination score of only 0.15%.
- Anchor retention analysis shows that DIA stably preserves four types of anchors (`<think>`, `</think>`, `<answer>`, `</answer>`) on GSM8K and MATH; anchor retention for Base and Instruct models drops significantly.
- Hyperparameter analysis indicates that $\Delta$ acts as a knob between format strictness and reasoning depth. Smaller $\Delta$ may truncate too early, leading to high format rates but low accuracy; larger $\Delta$ provides more space for reasoning but may decrease the format score or increase latency.

## Highlights & Insights
- DIA moves format control from "post-generation JSON/tag repair" forward to "pre-generation boundary planning," which aligns well with the bidirectional and parallel generation characteristics of diffusion LLMs.
- Being training-free, the method is suitable for rapid deployment on existing dLLMs; it is more lightweight than fine-tuning models for each output schema.
- The paper clearly demonstrates the intrinsic problem of fixed anchors: it is not the absence of structural tokens, but the rigidity of token positions that leads to a mismatch in content space.
- An insight for structured reasoning is that models need to know not only the output format but also the generation budget for each format segment.

## Limitations & Future Work
- DIA still relies on manually specified anchors and their semantic roles. For open-domain dialogues, multi-turn tool calls, or creative writing where structural boundaries themselves change dynamically, manual anchors may lack flexibility.
- Length adjustment introduces additional inference overhead. DIA's latency on GSM8K is 26.52, significantly higher than the Base 10.72, making it unsuitable for all real-time scenarios.
- There is a trade-off between accuracy and format. DIA significantly improves format accuracy, but its accuracy on MATH is lower than unformatted Base/Instruct and fixed infilling.
- Current evaluations focus on reasoning templates and WikiBio JSON. Future work needs to validate more complex schemas, code, proofs, multimodal structured outputs, and nested tool calls.

## Related Work & Insights
- **vs prompt-based constraints**: Relying solely on prompts to output JSON or tags often results in lost boundaries during long reasoning; DIA controls anchors directly in the initial mask sequence, making it closer to the decoding process.
- **vs post-processing / repair**: Post-processing can fix formatting but may alter semantics or lose reasoning. DIA plans the structure before generation, reducing dependence on post-processing.
- **vs constrained decoding**: Constraints from grammar or FSM decoding are strong but inflexible; DIA maintains generation freedom through dynamic length allocation.
- **Insights for Diffusion LLMs**: The advantage of dLLMs is not just parallel acceleration, but also the ability to plan global structures before filling content, which could be an important application direction distinct from AR LLMs.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Dynamic anchor position prediction fits the dLLM mechanism well, and the training-free design is practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers GSM8K, MATH, WikiBio, stage ablation, hyperparameters, and latency, though task types remain somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear motivation and intuitive algorithm; however, numerical settings differ between some main and appendix tables, requiring readers to distinguish experimental conditions.
- **Value**: ⭐⭐⭐⭐☆ Highly insightful for structured output, format-constrained reasoning, and dLLM decoding design, especially for applications requiring parseable output.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)
- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)

</div>

<!-- RELATED:END -->
