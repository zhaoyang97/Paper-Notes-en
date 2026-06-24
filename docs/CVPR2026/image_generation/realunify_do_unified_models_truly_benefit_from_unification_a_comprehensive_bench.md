---
title: >-
  [Paper Note] RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark
description: >-
  [CVPR 2026][Image Generation][Unified models] This paper proposes RealUnify, the first benchmark specifically designed to evaluate the bidirectional synergy between understanding and generation capabilities in unified models. Through 1000 human-annotated instances and a dual evaluation protocol (direct and stepwise), it reveals that while current unified models possess both understanding and generation capabilities, they fail to achieve true capability synergy in end-to-end s…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Unified models"
  - "capability synergy"
  - "understanding and generation"
  - "benchmark"
  - "bidirectional evaluation"
date: 2026-05-08
content_hash: e3c0590f6f9c923b
---

# RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark

**Conference**: CVPR 2026  
**arXiv**: [2509.24897](https://arxiv.org/abs/2509.24897)  
**Code**: [https://github.com/FrankYang-17/RealUnify](https://github.com/FrankYang-17/RealUnify)  
**Area**: Image Generation  
**Keywords**: Unified models, capability synergy, understanding and generation, benchmark, bidirectional evaluation

## TL;DR
This paper proposes RealUnify, the first benchmark specifically designed to evaluate the bidirectional synergy between understanding and generation capabilities in unified models. Through 1000 human-annotated instances and a dual evaluation protocol (direct and stepwise), it reveals that while current unified models possess both understanding and generation capabilities, they fail to achieve true capability synergy in end-to-end scenarios.

## Background & Motivation
1. **Background**: Multimodal unified models (e.g., BAGEL, Janus-Pro), which integrate visual understanding (VQA) and visual generation (T2I) into a single architecture, have become a significant direction toward General AI.
2. **Limitations of Prior Work**: Existing evaluation frameworks (e.g., MME-Unify, UniEval) primarily evaluate understanding and generation separately or simply combine the two types of tasks, failing to determine whether unified models truly achieve a "1+1>2" synergistic effect.
3. **Key Challenge**: The primary value of unified models lies in bidirectional gains—using understanding to guide generation and using generation to assist understanding. However, there is currently a lack of rigorous benchmarks to verify whether this bidirectional synergy actually exists.
4. **Goal**: To design a benchmark that can precisely measure the degree of capability synergy in unified models, answering the question: "Does unification truly bring performance superior to separate capabilities?"
5. **Key Insight**: Designing tasks that inherently rely on understanding-generation synergy and diagnosing the source of bottlenecks through a stepwise evaluation protocol.
6. **Core Idea**: For the first time, systematically examine whether unified models achieve true synergy between understanding and generation through meticulously designed bidirectional synergistic tasks and a dual direct/stepwise evaluation protocol.

## Method

### Overall Architecture
RealUnify comprises 1000 human-annotated instances covering 10 categories and 32 subtasks. The core design revolves around two axes: **Understanding-Enhanced Generation (UEG)**, which requires reasoning (common sense, logic, etc.) to guide image generation; and **Generation-Enhanced Understanding (GEU)**, which requires mental simulation or reconstruction to solve reasoning tasks. Evaluation utilizes two protocols: direct evaluation and stepwise evaluation.

### Key Designs

**1. Understanding-Enhanced Generation (UEG): Forcing the model to "think clearly" before drawing**

Ordinary T2I benchmarks (e.g., aesthetic scores, text alignment) only observe whether the image looks good or matches the prompt, but fail to test if the model applies knowledge and reasoning to generation. The UEG axis specifically designs tasks requiring "understanding before generation," divided into 6 categories: world knowledge (generating images requiring objective facts), common sense reasoning (generating images following everyday phenomena), mathematical reasoning (calculating correctly before generating the result), logical reasoning (generation satisfying logical constraints), scientific reasoning (applying physics/chemistry/biology principles), and code-to-image (parsing code logic to generate the corresponding scene). The commonality is that basic drawing capability is insufficient; the model must first invoke its understanding to "think" of the correct answer. A wrong image indicates the synergy broke at the "understanding → generation" step.

**2. Generation-Enhanced Understanding (GEU): Forcing the model to "think with images" to answer questions**

Conversely, this axis tests the other direction: whether the model can leverage its generation capability to assist its understanding. It contains 4 task types: mental reconstruction (reassembling shuffled image patches mentally before answering), mental tracking (tracking the final state of colored segments after multi-step transformations), attention focusing (using generation to highlight key areas to assist recognition), and cognitive navigation (maze/map pathfinding requiring intermediate visualization for reasoning). These problems are often difficult if relying solely on linguistic chain-of-thought reasoning; an ideal unified model should explicitly generate intermediate states in a "generate while looking" manner. GEU examines whether the model is truly using generation to assist understanding or merely bypassing generation to guess answers via linguistic reasoning.

**3. Dual Evaluation Protocol (Direct + Stepwise): Separating "lack of capability" from "failure to integrate"**

Looking only at end-to-end results is ambiguous: if a model fails, is it because the base capabilities (understanding or generation) are weak, or are both present but unable to spontaneously combine? To address this, every item is tested under two protocols. Direct Evaluation requires end-to-end completion in one go. Stepwise Evaluation decomposes the task into independent understanding and generation segments—for UEG, it understands first then generates; for GEU, it generates first then understands. Comparing the two identifies the bottleneck: if Stepwise is significantly higher than Direct, the capabilities exist but the issue lies in integration; if Stepwise is also low, the problem is with the base capabilities. This design is the core distinction of RealUnify—it provides a diagnosis rather than just a score.

**4. Polling Evaluation: Validating via QA instead of aesthetic scoring**

The correctness of images generated in UEG cannot be judged by aesthetic or similarity metrics like FID/CLIP—whether the "calculation result in the image is correct" or the "physical phenomenon is drawn right" is a matter of content. RealUnify pre-defines a set of verification questions for each generation target and uses Gemini 2.5 Pro as a judging model to vote on the generated images. Correctness is judged by whether the content hits the target. Compared to traditional metrics, this QA-based automated verification is better suited for the evaluation of knowledge-intensive generation.

### Loss & Training
This work is a benchmark study and does not involve model training. Regarding data construction: UEG tasks were manually designed by 10 human experts and cross-verified by 3 reviewers; GEU tasks were partially automatically generated and then annotated by experts. The reliability of Gemini 2.5 Pro as a judging model was verified through consistency with human expert scores.

## Key Experimental Results

### Main Results

| Model | UEG Direct | UEG Step | GEU Direct | GEU Step | Total |
|------|-----------|----------|-----------|----------|------|
| Nano Banana (Closed) | 63.0 | - | 31.8 | - | 50.5 |
| BAGEL (Best Open) | 32.7 | 47.7 | 39.3 | 35.8 | 35.3/42.9 |
| UniPic2 | 37.5 | 40.5 | 24.0 | 23.8 | 32.1/33.8 |
| OneCAT | 37.5 | 39.0 | 31.3 | 29.3 | 35.0/35.1 |
| Oracle (Gemini+GPT-Image) | - | 72.7 | - | 31.8 | - |

### Ablation Study

| Eval Method | BAGEL UEG | BAGEL GEU | Description |
|---------|----------|----------|------|
| Direct | 32.7 | 39.3 | End-to-end, fails to integrate spontaneously |
| Stepwise | 47.7 | 35.8 | UEG improves significantly, GEU decreases |
| Oracle (GT Intermediates) | Higher | Higher | Indicates base capabilities exist but integration is lacking |

### Key Findings
- **Significant improvement in UEG Stepwise**: BAGEL improved from 32.7% to 47.7%, indicating the model possess internal knowledge but cannot spontaneously integrate it into generation.
- **GEU Stepwise unexpectedly decreases**: Performance dropped after decomposition, suggesting that in direct evaluation, the model relies on "understanding shortcuts" rather than truly utilizing its generation capability.
- **Huge gap between open and closed source**: On UEG, the best open-source model reached 37.5% vs. 63.0% for closed-source; however, on GEU, open-source models (BAGEL 39.3%) actually outperformed closed-source models (31.8%).
- **Oracle upper bound far from reached**: Combined expert models reached 72.7% on UEG, while the current best unified model reached only 47.7% (stepwise), a massive gap.

## Highlights & Insights
- **Stepwise evaluation reveals the "possess but cannot use" phenomenon**: This is the most core finding—models have the understanding and generation capabilities but cannot spontaneously integrate them in end-to-end scenarios. This diagnostic evaluation design can be transferred to the evaluation of other AI systems requiring multi-capability synergy.
- **Discovery of "understanding shortcuts" in GEU tasks**: On tasks requiring "generation then understanding," models actually bypassed generation to answer directly via understanding. Forced generation in stepwise evaluation resulted in worse performance, revealing that current models severely underutilize their generation capabilities.
- **Polling Evaluation Mechanism**: Using question lists + LLM judgment to verify the correctness of generated images is more suitable for knowledge-intensive generation evaluation than traditional FID/CLIP metrics.

## Limitations & Future Work
- Evaluation depends on Gemini 2.5 Pro as a judge, posing a risk of evaluation bias (though consistency with human scores was verified).
- The benchmark only contains 1000 instances; some subtask sample sizes might be insufficient to support statistical significance.
- Lack of exploration into training methods to improve synergy—the study diagnoses the problem but does not propose a solution.
- Future work could explore specific training strategies (e.g., interleaved training, synergistic rewards) to promote true capability fusion.

## Related Work & Insights
- **vs MME-Unify**: The latter evaluates understanding and generation simultaneously but does not test their synergy; RealUnify specifically designs tasks that can only be completed through synergy.
- **vs T2I-CoReBench/WISE**: These benchmarks initially explored how understanding helps generation, but they are not systematic, not bidirectional, and lack stepwise diagnosis.
- **vs Expert Model Combinations**: Oracle experiments show that a simple combination of best-in-class expert models (Gemini + GPT-Image) can reach 72.7%, far exceeding any unified model, suggesting that the unified architecture itself is not the core—training strategies and inductive biases are key.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic benchmark for unified model capability synergy, with a sophisticated stepwise evaluation protocol.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 unified models + 6 expert baselines, dual evaluation protocol, and verification of judgment reliability.
- Writing Quality: ⭐⭐⭐⭐ Logical clarity, rich charts, and persuasive conclusions.
- Value: ⭐⭐⭐⭐ Clearly highlights the direction of "what truly needs optimization" for unified model research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] I2I-Bench: A Comprehensive Benchmark Suite for Image-to-Image Editing Models](i2i-bench_a_comprehensive_benchmark_suite_for_image-to-image_editing_models.md)
- [\[CVPR 2026\] EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories](emma_concept_erasure_benchmark_with_comprehensive_semantic_metrics_and_diverse_c.md)
- [\[CVPR 2026\] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization](vistorybench_comprehensive_benchmark_suite_for_story_visualization.md)
- [\[CVPR 2026\] Do Less, Achieve More: Do We Need Every-Step Optimization for RL Fine-tuning of Diffusion Models?](do_less_achieve_more_do_we_need_every-step_optimization_for_rl_fine-tuning_of_di.md)
- [\[CVPR 2026\] UniGen-1.5: Enhancing Image Generation and Editing through Reward Unification in RL](unigen-15_enhancing_image_generation_and_editing_through_reward_unification_in_r.md)

</div>

<!-- RELATED:END -->
