---
title: >-
  [Paper Note] Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation
description: >-
  [CVPR 2025][Reasoning][Agent Distillation] AoTD uses an LLM agent to decompose complex video questions into subtasks, invokes expert vision models to execute them, and collects intermediate results as a Chain-of-Thought (CoT). After quality filtering using an LLM, the CoT is distilled into a Video-LLM, enabling the end-to-end model to achieve both accurate answers and interpretable multi-step reasoning capabilities.
tags:
  - "CVPR 2025"
  - "Reasoning"
  - "Agent Distillation"
  - "Video Question Answering"
  - "Chain-of-Thought (CoT)"
  - "Spatiotemporal Grounding"
  - "Multi-step Reasoning"
date: 2026-05-08
content_hash: cedec9f4d04de77c
---

# Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation

**Conference**: CVPR 2025  
**arXiv**: [2412.01694](https://arxiv.org/abs/2412.01694)  
**Code**: [https://zhengrongz.github.io/AoTD/](https://zhengrongz.github.io/AoTD/)  
**Area**: Video Understanding / Agent  
**Keywords**: Agent Distillation, Video Question Answering, Chain-of-Thought (CoT), Spatiotemporal Grounding, Multi-step Reasoning

## TL;DR
AoTD uses an LLM agent to decompose complex video questions into subtasks, invokes expert vision models to execute them, and collects intermediate results as a Chain-of-Thought (CoT). After quality filtering using an LLM, the CoT is distilled into a Video-LLM, enabling the end-to-end model to achieve both accurate answers and interpretable multi-step reasoning capabilities.

## Background & Motivation

**Background**: The field of Video Question Answering (VideoQA) has two main paradigms: (1) end-to-end Video-LLMs (e.g., VideoLLaMA2, LLaVA-NeXT-Video), which are trained directly on QA pairs and exhibit strong performance but lack interpretability and spatiotemporal grounding capabilities; (2) agent-based systems, which utilize LLMs to decompose problems and invoke expert models, offering good interpretability but suffering from slow inference speeds (47s vs 10s), high memory footprint (65GB vs 18GB), and dependency on the capabilities of tool models.

**Limitations of Prior Work**: End-to-end models only learn the "question $\to$ answer" mapping without acquiring intermediate reasoning processes, rendering them prone to errors and incapable of explaining reasoning paths when facing complex compositional, temporal, or causal questions. Conversely, while agent systems possess reasoning chains, they are impractical for real-world deployment due to latency and resource constraints.

**Key Challenge**: The reasoning process of agent systems is highly valuable (interpretable and physically grounded in space-time), but its format (sequential invocation of multiple models) is unsuitable for practical deployment. Can the reasoning capabilities of agents be "taught" to end-to-end models?

**Goal**: How to automatically generate high-quality multi-step Chains-of-Thought (CoT) for any VideoQA dataset and distill them into a Video-LLM to enhance its reasoning capabilities?

**Key Insight**: Unlike directly generating CoT using MLLMs (which are prone to hallucination), the authors utilize reliable expert vision models as "proxies for thinking"—where the output of each subtask is an actual visual analysis result (e.g., bounding boxes, temporal windows), which is far more reliable than pure text reasoning.

**Core Idea**: Construct video reasoning chains from the execution trajectories of an agent system (rather than MLLM hallucination) and distill them into an end-to-end Video-LLM after validation by an LLM.

## Method

### Overall Architecture
AoTD consists of four steps: (1) evaluate and select the best vision models for each subtask; (2) use an LLM to decompose the video question into a Python program, sequentially execute the expert models, and record intermediate results; (3) employ an LLM to convert the execution trajectories into natural language CoT and perform a two-step quality verification; (4) distill the verified CoT alongside the QA pairs into the Video-LLM. The final model can switch between outputting a concise answer or a detailed reasoning chain based on the prompt.

### Key Designs

1. **Subtask Expert Model Selection and Agent Execution**:

    - **Function**: Automatically decompose complex video questions into chains of executable subtasks and solve them sequentially using optimal models.
    - **Mechanism**: Define 5 types of subtasks—problem decomposition (DeepSeek-Coder 85.7% Acc), object detection (OWL-ViT v2 63.0% IoU), temporal localization (UniVTG 24.7% IoU), action recognition (LLaVA-NeXT-Video-DPO 18.2% Top1), and question answering (LLaVA-NeXT-Video-DPO 53.4% Acc). Evaluate and select the best model for each subtask using annotated programs from the STAR dataset. During execution, the LLM reads model documentation and decomposes the question into Python code to invoke the respective models.
    - **Design Motivation**: Rather than relying on a single model's "imagination" to construct a CoT, utilize the actual outputs of expert models (e.g., bounding boxes, time intervals) as the basis for reasoning, which is more reliable. Independent evaluation of each subtask also exposes the performance bottlenecks of current vision models (e.g., temporal localization achieving only 24.7% IoU).

2. **Two-Step CoT Quality Verification**:

    - **Function**: Filter out incorrect or low-quality reasoning chains to ensure the reliability of the distilled data.
    - **Mechanism**: First step—Execution outcome filtering: for multiple-choice questions, the agent's output is required to perfectly match the ground truth; for open-ended questions, an LLM is used to verify consistency. Second step—Logical quality filtering: use an LLM to evaluate whether the CoT follows a clear step-by-step reasoning process and contains key information required for the answer (binary Yes/No classification). Out of 158.6K QA pairs, 32.3K high-quality CoTs are ultimately retained (approximately 20% pass rate).
    - **Design Motivation**: Directly distilling without filtering leads to performance degradation (as shown in ablation studies: 55.6% on MVBench with filtering vs. 53.7% without), indicating that low-quality CoTs can mislead the model's learning process.

3. **Dual-Mode Distillation Training**:

    - **Function**: Enable the model to support both direct answering and reasoning chain generation output modes.
    - **Mechanism**: During training, samples with CoTs are appended with the suffix prompt "Explain the rationale", while samples without CoTs use standard QA prompts. The loss function is $\mathcal{L} = \mathcal{L}_{label} + \lambda \mathcal{L}_{rationale}$ where $\lambda=1$. During inference, the output mode is selected based on the prompt—directly outputting the answer for fast responses, or generating the full reasoning chain when an explanation is needed.
    - **Design Motivation**: Training with reasoning chains not only enhances interpretability but also boosts direct-answer accuracy (as the model internalizes the reasoning process) while maintaining deployment flexibility.

### Loss & Training
Standard cross-entropy loss is used, with answer loss and rationale loss equally weighted ($\lambda=1$). For samples without CoTs, the rationale loss is set to 0. Instruction tuning is performed based on LLaVA-NeXT-Video-7B. Training data includes VideoQA datasets such as STAR, NExT-QA, AGQA, ANetQA, and CLEVRER.

## Key Experimental Results

### Main Results

| Benchmark | Metric | LNV-AoTD | LNV-Instruct | Gain |
|--------|------|------|----------|------|
| STAR (Compositional) | Acc | **74.3%** | 72.2% | +2.1% |
| NExT-QA (Causal) | Acc | **81.2%** | 79.7% | +1.5% |
| Perception-Test | Acc | **58.8%** | 57.1% | +1.7% |
| MVBench | Acc | **55.6%** | 53.1% | +2.5% |
| AGQA (Open) | Acc/Score | 60.9/3.6 | 59.3/3.4 | +1.6/+0.2 |
| ActivityNet-QA | Score | **3.55** | 3.52 | +0.03 |

### Ablation Study

| Configuration | MVBench | STAR | AGQA |
|------|---------|------|------|
| LNV-AoTD (w/ filtering) | **55.6** | **74.3** | 60.9/3.6 |
| LNV-AoTD (w/o filtering) | 53.7 | 73.3 | 59.5/3.5 |
| LLaVA-OneVision + AoTD | **60.5** | **76.6** | 65.7/3.7 |
| LLaVA-OneVision Instruct | 59.2 | 75.8 | 65.6/3.7 |
| Qwen2-VL + AoTD | **66.5** | **73.1** | 61.2/3.7 |
| Qwen2-VL Instruct | 65.6 | 71.4 | 59.8/3.6 |

### Key Findings
- **CoT filtering is critical**: Distilling unfiltered CoT can introduce noise instead (dropping by 1.9% on MVBench), indicating that about 80% of the reasoning chains produced by the agent system do not meet quality standards.
- **The method is generalizable to different Video-LLMs**: Consistent improvements are observed on LLaVA-OneVision, VideoLLaMA2, and Qwen2-VL, validating the generalizability of AoTD.
- **The distilled model genuinely learns spatiotemporal reasoning**: Evaluating the temporal localization (IoU 21.7% vs. UniVTG 22.8%) and spatial localization (IoU 45.2% vs. OWL-ViT 64.7%) in the rationales on STAR shows that the end-to-end model's grounding capabilities are close to those of the expert models.
- **Significant efficiency gains**: The agent system (47.9s / 65GB) $\to$ distilled model (10.6s / 18GB), representing a 4.5 $\times$ reduction in inference latency and a 3.6 $\times$ reduction in GPU memory footprint.

## Highlights & Insights
- **"Using agent execution trajectories as CoTs" is more reliable than "letting MLLMs hallucinate CoTs"**: Since intermediate results are the actual outputs of vision models rather than being fabricated out of thin air. This paradigm of constructing CoTs based on tool execution results can be generalized to other multi-step reasoning tasks.
- **The elegance of the distillation approach**: It requires no architectural changes to the Video-LLMs and merely incorporates reasoning chains at the training data level, achieving simultaneous improvements in accuracy and interpretability with minimal cost and clear benefits.
- **Subtask evaluation exposes the limitations of current vision models**: Temporal localization (24.7% IoU) and action recognition (18.2% Top1-Acc) remain weak, serving as bottlenecks for the agent system. As these foundation models improve, there is significant potential for further performance gains in AoTD.

## Limitations & Future Work
- Temporal localization and action recognition models are still weak (coupling only 24.7% IoU), which means the temporal information in CoTs may not be precise enough, restricting the performance ceiling of the distilled model.
- The CoT pass rate is only about 20% (158.6K $\to$ 32.3K), leaving a large number of QA pairs without CoT assistance during training.
- The spatial localization ability of the distilled model (IoU 45.2%) is significantly weaker than that of the expert model (64.7%), indicating that there is still information loss during end-to-end distillation.
- Open-ended VQA evaluation (GPT-based scoring) suffers from bias, making it difficult to accurately reflect the true capabilities of the model.
- Iterative distillation was not explored—using the distilled model to generate better CoTs for subsequent training rounds.

## Related Work & Insights
- **vs VPD (Visual Program Distillation)**: VPD performs a similar task on images; AoTD is the first to extend agent distillation to the video domain, which requires handling additional challenges like temporal localization and action recognition.
- **vs Video-STaR**: Video-STaR constructs CoTs using videos and existing labels without requiring an agent system; the advantage of AoTD is that its CoTs are grounded in the actual observations of vision models, making them more well-founded.
- **vs MoReVQA/VURF**: These are agent systems themselves, which are heavy and slow during inference; AoTD distills the capabilities of the agent into a lightweight model, making physical deployment much more feasible.

## Rating
- Novelty: ⭐⭐⭐ Siginificant innovation. Distilling agent execution trajectories into CoT is a natural yet effective idea, pioneered in the video domain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple VideoQA benchmarks, multi-model transfer tests, CoT quality assessments, and efficiency comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, with detailed descriptions of subtask evaluation and filtering mechanisms.
- Value: ⭐⭐⭐⭐ The method is generalizable, low-cost, and yields consistent performance improvements, offering valuable guidance to the Video-LLM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Video-T1: Test-Time Scaling for Video Generation](../../ICCV2025/llm_reasoning/video-t1_test-time_scaling_for_video_generation.md)
- [\[ACL 2025\] Unifying Language Agent Algorithms with Graph-based Orchestration Engine for Reproducible Agent Research](../../ACL2025/llm_reasoning/unifying_language_agent_algorithms_with_graph-based_orchestration_engine_for_rep.md)
- [\[NeurIPS 2025\] Atom of Thoughts for Markov LLM Test-Time Scaling](../../NeurIPS2025/llm_reasoning/atom_of_thoughts_for_markov_llm_testtime_scaling.md)
- [\[CVPR 2025\] VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection](videoespresso_a_large-scale_chain-of-thought_dataset_for_fine-grained_video_reas.md)
- [\[ICML 2025\] ProofCompass: Enhancing Specialized Provers with LLM Guidance](../../ICML2025/llm_reasoning/proofcompass_enhancing_specialized_provers_with_llm_guidance.md)

</div>

<!-- RELATED:END -->
