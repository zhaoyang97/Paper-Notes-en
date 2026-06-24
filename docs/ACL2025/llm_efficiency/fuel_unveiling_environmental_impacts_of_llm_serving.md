---
title: >-
  [Paper Note] FUEL: Unveiling Environmental Impacts of Large Language Model Serving: A Functional Unit View
description: >-
  [ACL 2025][LLM Efficiency][Carbon Emissions] Proposes the FUEL framework, which for the first time introduces the concept of "Functional Unit" from life cycle assessment (LCA) as a standardized baseline for comparison. It evaluates carbon emissions of different LLM serving configurations under unified quality, performance, and workload constraints, revealing several counter-intuitive green AI insights through three case studies: model size, quantization strategy…
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Carbon Emissions"
  - "LLM Serving"
  - "Functional Unit"
  - "Quantification"
  - "Hardware Selection"
date: 2026-05-08
content_hash: adcc6c39a40662e3
---

# FUEL: Unveiling Environmental Impacts of Large Language Model Serving: A Functional Unit View

**Conference**: ACL 2025  
**arXiv**: [2502.11256](https://arxiv.org/abs/2502.11256)  
**Code**: [https://github.com/jojacola/FUEL](https://github.com/jojacola/FUEL)  
**Area**: LLM Efficiency / Green AI  
**Keywords**: Carbon Emissions, LLM Serving, Functional Unit, Quantification, Hardware Selection

## TL;DR

Proposes the FUEL framework, which for the first time introduces the concept of "Functional Unit" from life cycle assessment (LCA) as a standardized baseline for comparison. It evaluates carbon emissions of different LLM serving configurations under unified quality, performance, and workload constraints, revealing several counter-intuitive green AI insights through three case studies: model size, quantization strategy, and hardware selection.

## Background & Motivation

**Background**: LLM serving introduces significant environmental impacts—processing a single prompt in ChatGPT generates over 4 grams of $CO_2$eq, which is more than 20 times that of a standard search query. Existing studies (e.g., LLMCarbon, LLMCO2, GreenLLM) analyze the carbon emissions of LLMs through modeling and profiling.

**Limitations of Prior Work**: Existing research suffers from two key limitations: (1) focusing on individual LLMs rather than cross-model comparisons; (2) lacking a standardized carbon emission baseline—carbon emissions of different models under varying quality, latency, and throughput conditions cannot be fairly compared. Merely looking at emissions per token ignores differences in output quality.

**Key Challenge**: A small model might have lower emissions per token, but if it requires more tokens to achieve the same quality, the total carbon emissions can be higher. The lack of a unified comparison baseline makes the question of "which configuration is greener" impossible to answer rigorously.

**Goal**: Establish a standardized evaluation framework considering quality and performance constraints to compare carbon emissions of different LLM serving configurations under fair conditions.

**Key Insight**: Drawing inspiration from the Life Cycle Assessment (LCA) methodology in environmental sustainability, this work introduces the concept of "Functional Unit"—instead of comparing "emissions per token", it compares "emissions per token meeting the constraint conditions".

**Core Idea**: Define the functional unit in LLM serving as the tokens that satisfy specific workload intensity, performance constraints (TTFT $\leq$ 1s, TPOT $\leq$ 200ms), and quality constraints (Qscore meets the threshold), based on which the carbon functional unit (CFU) intensity is calculated.

## Method

### Overall Architecture

The FUEL framework consists of four steps: (1) Input definition—models, comparison configurations (size/quantization/hardware), and serving constraints; (2) Functional unit definition—based on QPS, TTFT/TPOT performance constraints, and Qscore quality constraints; (3) Profiling—running LLMs to collect performance and energy consumption data; (4) Carbon modeling—calculating carbon emissions per functional unit (CFU), which includes operational and embodied carbon.

### Key Designs

1. **Functional Unit Definition**:

    - **Function**: Establish a standardized baseline for cross-model carbon comparison.
    - **Mechanism**: A functional unit represents a token that satisfies serving constraints during generation. $N_f = \sum_{i=1}^N \mathbb{I}(Q_i \geq \alpha) \cdot \mathbb{I}(TTFT_i \leq \beta) \cdot \mathbb{I}(TPOT_i \leq \gamma)$, and CFU = Total Carbon / $N_f$. Quality is evaluated using the Qscore from the Skywork reward model, and performance constraints are set to TTFT $\le$ 1s and TPOT $\le$ 200ms (aligned with human reading speed).
    - **Design Motivation**: Directly comparing carbon emissions per token is misleading—small models have cheaper single tokens but worse quality, leading to more wasted tokens. The functional unit incorporates quality and performance to make comparisons fairer.

2. **Carbon Emission Modeling (Operational + Embodied)**:

    - **Function**: Comprehensively calculate the carbon footprint of LLM serving.
    - **Mechanism**: Total carbon = Operational carbon + Embodied carbon. Operational carbon is $C_{op} = E_{op} \times CI$ (energy consumption $\times$ carbon intensity), sampling power every 200ms using pynvml/psutil. Embodied carbon is $C_{em} = (t / LT) \times C_{em,total}$ (ratio of running time to hardware lifetime $\times$ hardware lifecycle carbon emissions), calculated using the ACT tool.
    - **Design Motivation**: Only looking at operational carbon ignores the massive carbon footprint of manufacturing new hardware—the embodied carbon of an H100 is about 29.92 kg$CO_2$eq, while that of an L40 is 26.6 kg$CO_2$eq.

3. **Quality Evaluation Method (Reward Model as Quality Evaluator)**:

    - **Function**: Quantitatively evaluate LLM output quality as the quality constraint of the functional unit.
    - **Mechanism**: Employ the open-source Skywork reward model (which ranks high on RewardBench) to calculate the Qscore of each response as a quality metric. A higher Qscore indicates better output quality and stronger alignment with human preferences.
    - **Design Motivation**: Traditional quality evaluations rely on specific datasets or reference answers, while a reward model can consistently evaluate quality across tasks and aligns well with human preferences.

### Loss & Training

Non-training work. The experiments use the vLLM serving platform, with carbon intensity set to 518 g$CO_2$eq/kWh (the 12-month average of the region where the server is located), and the temperature set to 0.

## Key Experimental Results

### Main Results: Model Size Case Study

| Configuration | Low Quality (-5) QPS=1 | High Quality (15) QPS=1 |
|------|-------------------|-------------------|
| Qwen 7B | Lowest Carbon | Highest Carbon (1.8× 32B) |
| Qwen 14B | Medium | Medium |
| Qwen 32B | Highest Carbon | **Lowest Carbon (saved 40%+)** |

### Quantization Case Study

| Method | Description | Is It Always Greener? |
|------|------|----------------|
| AWQ (Weight Quantization) | TPOT accelerates at low QPS, TTFT is always slower | **No**—increases carbon emissions under large models and high QPS |
| W8A8 (Activation Quantization) | Both TPOT and TTFT consistently accelerate | **Yes**—consistently reduces carbon emissions in all scenarios |

### Hardware Case Study

| Hardware | L40 (2022) | H100 (2023) |
|------|------------|-------------|
| GPU TDP | 300W | 700W |
| Embodied Carbon (GPU) | 26.6 kg$CO_2$eq | 29.92 kg$CO_2$eq |
| Low QPS Carbon Efficiency | **Better** | Worse |
| High QPS Carbon Efficiency | Worse | **Better** |

### Key Findings

- **Small models are not always greener**: When output quality requirements are high, larger models are actually more carbon-efficient because smaller models have a lower proportion of tokens meeting quality constraints.
- **The carbon-saving effect of quantization depends on the type**: Weight quantization (AWQ) increases carbon emissions in certain scenarios (because it requires dequantization back to 16-bit during inference), whereas activation quantization (W8A8) is consistently effective.
- **New hardware is not always greener**: Although H100 has powerful performance, its high power consumption and large embodied carbon make older hardware like L40 more carbon-efficient at low QPS.
- **There is no "one-size-fits-all green" configuration**: The optimal choice depends on the combination of QPS, quality requirements, and model size.

## Highlights & Insights

- **The introduction of the functional unit is the core innovation**: Transferring the LCA methodology from environmental science to the AI domain solves the fundamental problem of "how to fairly compare carbon emissions across different configurations." This framework can be extended to the environmental impact assessment of all AI services.
- **Counter-intuitive findings possess high practical value**: Intuitions like "small models are greener," "new hardware is greener," and "quantization is greener" are proven to be conditional, providing quantitative bases for practical deployment decisions.
- **The importance of embodied carbon is revealed**: The embodied carbon of an Intel Xeon 8480+ (42.81 kg$CO_2$eq) is more than 4 times that of an AMD EPYC 7443 (9.98 kg$CO_2$eq), a difference that is completely ignored when only looking at operational carbon.

## Limitations & Future Work

- Only two model families, Qwen2.5 and Llama2, were tested, which do not cover multimodal or code models.
- Experiments were limited to a single GPU, without exploring communication overheads and carbon emissions in multi-GPU distributed environments.
- Quality evaluation relies on a reward model; more advanced quality assessment methods might alter the conclusions.
- Future work can explore carbon-aware adaptive serving strategies—dynamically choosing model configurations based on real-time carbon intensity and load.

## Related Work & Insights

- **vs LLMCarbon/LLMCO2**: These frameworks provide end-to-end carbon modeling but lack standardized comparison baselines. FUEL resolves the cross-model comparability issue through functional units.
- **vs GreenLLM/Sprout**: These works optimize carbon emissions based on profiling but do not consider quality constraints, which can lead to suboptimal decisions that "reduce carbon but degrade quality."

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The introduction of the functional unit concept is a paradigm shift in this field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ The three case studies are comprehensive, though the variety of models and hardware is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Highly structured, motivated by three questions for each case study, demonstrating strong logic.
- **Value**: ⭐⭐⭐⭐⭐ Holds direct practical guidance value for green AI deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reasoning Language Model Inference Serving Unveiled: An Empirical Study](../../ICLR2026/llm_efficiency/reasoning_language_model_inference_serving_unveiled_an_empirical_study.md)
- [\[ACL 2025\] LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)
- [\[ICLR 2026\] Cascadia: An Efficient Cascade Serving System for Large Language Models](../../ICLR2026/llm_efficiency/cascadia_an_efficient_cascade_serving_system_for_large_language_models.md)
- [\[ACL 2025\] LongSafety: Evaluating Long-Context Safety of Large Language Models](longsafety_evaluating_long-context_safety_of_large_language_models.md)
- [\[ACL 2025\] A Drop-In Solution for On-the-Fly Adaptation of Speculative Decoding in Large Language Models](a_drop-in_solution_for_on-the-fly_adaptation_of_speculative_decoding_in_large_la.md)

</div>

<!-- RELATED:END -->
