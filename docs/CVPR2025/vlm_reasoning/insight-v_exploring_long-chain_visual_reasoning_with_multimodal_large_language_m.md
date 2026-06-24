---
title: >-
  [Paper Note] Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models
description: >-
  [CVPR 2025][VLM Reasoning][Long-Chain Reasoning] Insight-V proposes a visual reasoning enhancement scheme consisting of a data generation pipeline and a multi-agent reasoning system: it constructs high-quality long-chain reasoning data through progressive generation and multi-granular evaluation, designs a Reasoning Agent and a Summary Agent to collaboratively solve problems, and incorporates iterative DPO to further improve reasoning quality…
tags:
  - "CVPR 2025"
  - "VLM Reasoning"
  - "Long-Chain Reasoning"
  - "Multi-Agent System"
  - "Visual Reasoning"
  - "Preference Optimization"
  - "Data Generation"
date: 2026-05-08
content_hash: fddf2fda2e4ac5ca
---

# Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models

**Conference**: CVPR 2025  
**arXiv**: [2411.14432](https://arxiv.org/abs/2411.14432)  
**Code**: [https://github.com/dongyh20/Insight-V](https://github.com/dongyh20/Insight-V)  
**Area**: Multimodal VLM  
**Keywords**: Long-Chain Reasoning, Multi-Agent System, Visual Reasoning, Preference Optimization, Data Generation

## TL;DR
Insight-V proposes a visual reasoning enhancement scheme consisting of a data generation pipeline and a multi-agent reasoning system: it constructs high-quality long-chain reasoning data through progressive generation and multi-granular evaluation, designs a Reasoning Agent and a Summary Agent to collaboratively solve problems, and incorporates iterative DPO to further improve reasoning quality, achieving an average improvement of 7% across seven visual reasoning benchmarks.

## Background & Motivation
- **Background**: LLMs have significantly improved reasoning capabilities through long-chain reasoning (such as CoT, o1), but long-chain visual reasoning in the multimodal field is still in its early stages.
- **Limitations of Prior Work**: (1) Lack of large-scale, high-quality long-chain visual reasoning data, and manual annotation is cost-prohibitive; (2) directly training MLLMs using CoT data yields limited effectiveness, as a single model struggles to handle both reasoning and final answering simultaneously.
- **Key Challenge**: Errors are easily introduced during the long-chain reasoning process, and a single model's judgment declines as the reasoning chain lengthens, where verbose reasoning can conversely lead to incorrect answers.
- **Goal**: To provide a scalable long-chain reasoning data generation scheme and an efficient training pipeline for reasoning enhancement.
- **Key Insight**: Decompose the reasoning process into two independent tasks: "reasoning" and "summarization", which are handled by specialized agents respectively.
- **Core Idea**: Decomposition of reasoning and summarization + robust summarization against reasoning errors = better visual reasoning.

## Method

### Overall Architecture
Insight-V consists of three components: (1) a long-chain reasoning data generation pipeline (progressive generation + multi-granular evaluation); (2) a multi-agent reasoning system (Reasoning Agent + Summary Agent); and (3) a two-stage training workflow (Supervised Fine-Tuning + Iterative DPO). The Reasoning Agent generates detailed step-by-step reasoning processes, while the Summary Agent evaluates the reasoning quality and selectively utilizes the reasoning results to provide the final answer.

### Key Designs
1. **Progressive Long-Chain Reasoning Data Generation**:
    - **Function**: Automatically generate structured long-chain reasoning data without human annotation.
    - **Mechanism**: Utilize a reasoning generator to progressively generate the reasoning process in JSON format. Each step contains a summary of the current step, detailed reasoning, and the next action ($continue$ or $summary$). For each question, iteratively sample $N$ times to obtain diverse reasoning paths. This is formalized as $R_t = M(I, Q, [R_1 \cdots R_{t-1}], A)$.
    - **Design Motivation**: Traditional CoT data lacks structure and sufficient reasoning depth. The progressive strategy allows the model to adaptively determine the length of the reasoning chain, and multiple samplings ensure the diversity of reasoning paths.

2. **Multi-Granular Evaluation System**:
    - **Function**: Performs quality filtering and ranking of the generated reasoning paths.
    - **Mechanism**: Two-level evaluation: (1) using an LLM (e.g., Qwen2) for correctness filtering of the final answer; (2) using a multimodal model (e.g., Qwen2-VL) to perform step-by-step scoring (1-100 scale) of the reasoning path, evaluating both step-by-step accuracy and the level of reasoning detail. All answers to the same question are scored in a single pass to ensure scoring consistency.
    - **Design Motivation**: Solely relying on answer correctness is insufficient to guarantee the quality of the reasoning process. Multi-granular evaluation is required to filter out the optimal reasoning chains.

3. **Multi-Agent Reasoning System**:
    - **Function**: Decomposes the problem-solving process into reasoning and summarization stages.
    - **Mechanism**: The Reasoning Agent focuses on generating detailed reasoning processes (trained on the highest-scoring reasoning paths); the Summary Agent evaluates the reasoning quality and selectively adopts the reasoning conclusions. Key design: the training data for the Summary Agent contains a mixture of optimal reasoning and flawed reasoning samples, avoiding simple copying of reasoning results and cultivating critical evaluation capability. Flawed samples are sampled across score ranges to cover different error levels.
    - **Design Motivation**: A single model's judgment degrades when the reasoning chain becomes long. Separating reasoning and summarization allows each agent to focus on its own task. The Summary Agent's robustness to reasoning errors is the key to the system's success.

### Loss & Training
- **Two-stage training**: Stage 1 performs supervised fine-tuning to obtain the two agents; Stage 2 performs iterative DPO on the Reasoning Agent.
- **Iterative DPO**: Solves the data distribution shift issue inherent in traditional offline DPO. Train a sequence of models $M_1, \ldots, M_T$, where each $M_{t+1}$ is trained using the preference data generated by $M_t$. A total of 3 iterations are performed.
- **DPO Loss**: Based on the Bradley-Terry model, $p^*(y_1 \succ y_2 | x) = \sigma(r^*(x,y_1) - r^*(x,y_2))$.
- Reasoning Agent training data: 200K images, 2 epochs, lr=5e-6.
- Summary Agent training data: 1.2M images (including 1M general image-text pairs to preserve base capability), 1 epoch, lr=1e-5.
- DPO training: 15K preference data, 1 epoch per round, lr=5e-7.

## Key Experimental Results

### Main Results

| Model | MMMU | MMMU-Pro | MMBench | ChartQA | MathVista | MMStar | Average |
|------|------|----------|---------|---------|-----------|--------|------|
| LLaVA-NeXT-LLaMA3 (8B) | 36.9 | 13.2 | 72.3 | 69.4 | 45.9 | 43.1 | 40.2 |
| + Multi-Agent | 40.8 | 17.8 | 77.6 | 74.6 | 47.4 | 52.6 | 44.5 |
| + Iterative DPO (Insight-V-LLaVA) | 42.0 | 21.0 | 81.7 | 77.4 | 49.8 | 57.4 | **47.2 (+7.0)** |
| Base Model (7B) | 47.1 | 22.6 | 81.3 | 75.7 | 56.9 | 57.0 | 48.7 |
| + Iterative DPO (Insight-V) | 50.2 | 24.9 | 82.3 | 81.5 | 59.9 | 61.5 | **51.6 (+2.9)** |

### Ablation Study

| Configuration | MMMU | ChartQA | MathVista | MMStar | Average |
|------|------|---------|-----------|--------|------|
| Baseline | 47.1 | 75.7 | 56.9 | 57.0 | 59.2 |
| Vanilla Direct SFT (Single-Model CoT) | 47.0 | 79.2 | 57.6 | 58.4 | 60.6 |
| Multi-Turn Supervised | 48.1 | 79.6 | 57.9 | 58.2 | 61.0 |
| Summary Agent Only | 47.5 | 76.3 | 57.3 | 57.9 | 59.8 |
| **Multi-Agent** | **49.7** | **81.2** | **58.7** | **58.6** | **62.1** |

### Key Findings
- The Multi-Agent system outperforms all single-model variants (Direct SFT, Multi-Turn), proving that decomposing reasoning and summarization is the core design.
- Using only the Summary Agent (without the reasoning process) yielding extremely limited improvement demonstrates that the detailed reasoning provided by the Reasoning Agent is indispensable.
- Increasing the training data of the Reasoning Agent from 50K to 200K continuously yields improvements, demonstrating clear data-scaling behavior.
- Iterative DPO (3 rounds) brings an additional 0.6% improvement compared to single-round DPO and outperforms external RLAIF-V datasets (which only yield a 0.2% improvement), indicating that preference pairs constructed from self-generated reasoning data are more effective.
- Insight-V does not degrade but rather improves on perception benchmarks (TextVQA/DocVQA/OCRBench), proving that the multi-agent system does not sacrifice base visual capability.

## Highlights & Insights
- The idea of separating reasoning and summarization is simple yet effective, and the robustness design of the Summary Agent against reasoning errors (trained with mixed flawed reasoning) is a key innovation.
- The progressive data generation pipeline enables zero-human-intervention-scale production of reasoning data, which is transferable to other tasks.
- The +7% advancement on LLaVA-NeXT proves that the method is more effective on weaker models, reducing the dependency on strong backbones.
- Iterative DPO addresses the distribution shift issue of offline DPO, offering a robust mechanism for continuous improvement in reasoning quality.

## Limitations & Future Work
- The reasoning data for the Reasoning Agent is primarily generated by a strong model, meaning the reasoning style might be limited by the generator.
- The multi-agent system requires two forward passes during inference, which doubles the computation overhead.
- It is currently validated only in single-image scenarios; long-chain reasoning on multi-image and video domains remains to be explored.
- The returns from 3-round iterative DPO diminish significantly, and whether more iterations are meaningful requires further validation.

## Related Work & Insights
- **vs Chain-of-Thought**: Direct CoT has limited effectiveness on MLLMs. Insight-V resolves the error accumulation caused by overly long CoT chains by separating reasoning and judgment.
- **vs OpenAI o1**: o1 focuses on pure text reasoning. Insight-V introduces a similar spirit into the multimodal domain, but adopts a multi-agent system rather than single-model long-chain reasoning.
- **vs POINTS/IXC-2.5**: These methods enhance single-model capabilities through better data or architecture, whereas Insight-V achieves larger gains through a system-level design (multi-agent + DPO).
- **vs Cambrian-1**: Cambrian-1 emphasizes visual backbone design, while Insight-V focuses on the reasoning process workflow design, offering a different entry point.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-agent reasoning system concept is novel and the data generation pipeline is practical, though core components (DPO, agent separation) are not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 reasoning benchmarks + 4 perception benchmarks, with comprehensive ablation and scaling analyses.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear and the methodology is described in detail, though abundant mathematical notation slightly elongates the text.
- Value: ⭐⭐⭐⭐ Provides an effective and reproducible scheme for enhancing MLLM visual reasoning.

---

> This note is generated based on a full reading of the paper, covering all contents of Methodology, Experiments, and Ablation Studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Grounded Chain-of-Thought for Multimodal Large Language Models](../../CVPR2026/vlm_reasoning/grounded_chain-of-thought_for_multimodal_large_language_models.md)
- [\[CVPR 2025\] Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces](thinking_in_space_how_multimodal_large_language_models_see_remember_and_recall_s.md)
- [\[CVPR 2025\] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)
- [\[CVPR 2025\] SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model](seqafford_sequential_3d_affordance_reasoning_via_multimodal_large_language_model.md)
- [\[ACL 2025\] Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning](../../ACL2025/vlm_reasoning/benchmarking_and_improving_large_vision-language_models_for_fundamental_visual_g.md)

</div>

<!-- RELATED:END -->
