---
title: >-
  [Paper Note] We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?
description: >-
  [ACL 2025][VLM Reasoning][Visual Mathematical Reasoning] This paper proposes the We-Math benchmark, containing 6.5K visual mathematical problems and 67 hierarchical knowledge concepts. By decomposing composite problems into sub-problems, it introduces a four-dimensional evaluation metric (Insufficient Knowledge (IK), Insufficient Generalization (IG), Complete Mastery (CM), and Rote Memorization (RM)), systematically evaluating the mathematical reasoning process of LMMs from t…
tags:
  - "ACL 2025"
  - "VLM Reasoning"
  - "Visual Mathematical Reasoning"
  - "Knowledge Concepts"
  - "Reasoning Evaluation"
  - "Multimodal"
  - "Benchmark"
date: 2026-05-08
content_hash: 239fa0979cc10752
---

# We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?

**Conference**: ACL 2025  
**arXiv**: [2407.01284](https://arxiv.org/abs/2407.01284)  
**Code**: [https://github.com/We-Math/We-Math](https://github.com/We-Math/We-Math)  
**Area**: Multimodal VLM  
**Keywords**: Visual Mathematical Reasoning, Knowledge Concepts, Reasoning Evaluation, Multimodal, Benchmark

## TL;DR
This paper proposes the We-Math benchmark, containing 6.5K visual mathematical problems and 67 hierarchical knowledge concepts. By decomposing composite problems into sub-problems, it introduces a four-dimensional evaluation metric (Insufficient Knowledge (IK), Insufficient Generalization (IG), Complete Mastery (CM), and Rote Memorization (RM)), systematically evaluating the mathematical reasoning process of LMMs from the perspective of knowledge mastery for the first time, rather than focusing solely on the final results.

## Background & Motivation
1. **Background**: LMMs have achieved progress in visual mathematical reasoning, but existing benchmarks (e.g., MathVista, MathVerse) only focus on the correctness of final answers, ignoring the knowledge mastery during the reasoning process.
2. **Limitations of Prior Work**: (1) Focusing only on results leads to counter-intuitive conclusions (e.g., LMMs performing better on university-level problems than primary school-level ones); (2) Correct answers do not necessarily reflect genuine reasoning ability (they may stem from rote memorization); (3) Incorrect answers do not necessarily imply a complete lack of foundational knowledge.
3. **Key Challenge**: Humans solve mathematical problems by step-by-step mastering and generalizing knowledge concepts, but existing evaluations cannot distinguish whether a model "truly understands" or is just "making a wild guess."
4. **Goal**: Design an evaluation benchmark based on knowledge concepts that can reveal the inherent issues in the mathematical reasoning of LMMs.
5. **Key Insight**: Decompose composite problems into sub-problems based on a single knowledge concept, and judge the true reasoning ability of models by comparing the correctness of sub-problems and original problems.
6. **Core Idea**: Hierarchical knowledge concepts + problem decomposition + four-dimensional evaluation metrics.

## Method

### Overall Architecture
Mathematical textbook knowledge system $\rightarrow$ 5 major categories, 12 typical problems, 67 knowledge concepts $\rightarrow$ collection of 6.5K problems $\rightarrow$ annotation of knowledge concepts and reasoning steps $\rightarrow$ manual decomposition of 1.5K multi-step problems into sub-problems $\rightarrow$ LMMs answering both sub-problems and original problems simultaneously $\rightarrow$ four-dimensional metric evaluation $\rightarrow$ Knowledge Concept Augmentation (KCA) strategy.

The 5 major categories are: Planar Figures (PF), Solid Figures (SF), Transformation and Motion (TMF), Position and Direction (PD), Measurement (Mem). Each terminal knowledge concept contains 10-40 samples to ensure balance, such as "Angle and Length" (AL), "Unit Comprehension and Conversion" (UCU), "Coordinate and Position Correspondence" (CCP), etc.

### Key Designs

1. **Hierarchical Knowledge Structure**:

    - Function: Ensure that the evaluation covers all foundational aspects of mathematical reasoning.
    - Mechanism: Organize mathematical problems into 5 layers according to the textbook knowledge system: Planar Figures, Solid Figures, Transformation and Motion, Position and Direction, and Measurement. Each layer is decomposed into 12 typical problems and 67 terminal knowledge concepts, with each concept containing 10-40 samples to ensure balance.
    - Design Motivation: Classifications in existing benchmarks are not systematic enough, leading to incomplete evaluations.

2. **Knowledge-Driven Problem Decomposition and Four-Dimensional Metrics**:

    - Function: Evaluate the mathematical capability of LMMs from the reasoning process rather than just the final results.
    - Mechanism: For a composite problem containing $k$ knowledge concepts, decompose it into $k$ single-concept sub-problems. Allow LMMs to answer all sub-problems and the original problem simultaneously, and then classify them into four categories: IK (incorrect sub-problems + incorrect original problem = Insufficient Knowledge), IG (correct sub-problems + incorrect original problem = Insufficient Generalization), CM (correct sub-problems + correct original problem = Complete Mastery), and RM (incorrect sub-problems + correct original problem = Rote Memorization). The hierarchy of capability is: IK < IG < CM.
    - Design Motivation: Just looking at correctness cannot distinguish between "insufficient knowledge" and "insufficient generalization," but the two require different improvement strategies.

3. **Knowledge Concept Augmentation Strategy (KCA)**:

    - Function: Alleviate the insufficient knowledge problem of LMMs by supplementing descriptions of knowledge concepts.
    - Mechanism: Construct descriptions for 67 knowledge concepts from Wikipedia and textbooks, and serve them as additional knowledge inputs to LMMs during reasoning.
    - Design Motivation: If IK is the primary issue, directly supplementing knowledge should bring improvement.

### Loss & Training
We-Math is an evaluation benchmark and does not involve training. Evaluated 17 LMMs (4 closed-source + 13 open-source), including GPT-4o, GPT-4V, Gemini 1.5 Pro, Qwen-VL-Max, LLaVA-NeXT-110B/70B, DeepSeek-VL, etc. The `testmini` subset (1,740 samples: 1,215 one-step problems, 360 two-step problems, 165 three-step problems) was used to accelerate evaluation. All problems were standardized into multiple-choice formats with an additional "Uncertain" option to prevent LMMs from deriving answers from the choices. Problem decomposition was completed by expert annotators, with cross-validation ensuring quality.

## Key Experimental Results

### Main Results

| Model | S1 Accuracy | S2 Accuracy | S3 Accuracy | Weakest Domain |
|------|---------|---------|---------|----------|
| GPT-4o | 72.84% | 58.06% | 43.64% | Angle & Length (39.12%) |
| GPT-4V | 65.51% | 49.17% | 38.18% | Angle & Length (38.42%) |
| Gemini 1.5 Pro | 56.13% | 51.39% | 33.94% | Angle & Length (31.23%) |
| LLaVA-NeXT-110B | High | Medium | Medium | Fine-grained Measurement |
| DeepSeek-VL-1.3B | Low | Very Low | Very Low | Most Domains |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Without KCA | Baseline | Standard reasoning |
| +KCA | Significant reduction in IK | Knowledge augmentation is effective |
| 1-step problems only | Highest accuracy | The more steps, the more difficult |
| 3-step problems | Lowest accuracy | Number of knowledge concepts is positively correlated with difficulty |

### Key Findings
- The number of solving steps is negatively correlated with performance—multi-step problems are significantly more difficult, indicating that the combination of knowledge concepts is a core challenge.
- GPT-4o transitioned from the IK stage to the IG stage for the first time, becoming the first LMM to move towards knowledge generalization.
- Most LMMs suffer from serious RM problems—they can answer composite problems correctly but fail on sub-problems, suggesting that they might not perform genuine reasoning.
- The KCA strategy effectively reduces IK problems but has limited impact on IG and RM.
- LMMs perform worst in fine-grained visual measurements such as "Angle and Length."

## Highlights & Insights
- **Depth of the Four-Dimensional Evaluation**: The discovery of RM (rote memorization) is alarming—models may get correct answers by exploiting pattern matching rather than genuine reasoning.
- **Knowledge-Concept-Driven Evaluation Paradigm**: Evaluating knowledge mastery rather than just the final answer is closer to human educational assessment systems.
- **Transferability to Other Reasoning Evaluations**: The four-dimensional metric framework can be generalized to any reasoning tasks that can be decomposed into sub-knowledge, such as code reasoning and scientific reasoning.
- **Significance of the Transition from the IK to IG Stage**: GPT-4o entered the insufficient generalization stage from the insufficient knowledge stage for the first time, showing that large-scale pre-training can indeed resolve the lack of foundational knowledge, but knowledge combination and generalization remain the core bottlenecks.
- **Systematic Explanation of Counter-Intuitive Phenomena**: The counter-intuitive conclusion in existing benchmarks that "university problems are simpler than primary school problems" is reasonably explained under the We-Math framework—university problems involve fewer combinations of knowledge concepts, thus resulting in lighter IG issues.
- **Additional Findings in Evaluation**: GPT-4o achieved up to 86.61% accuracy on "Unit Comprehension and Conversion" (UCU) but only 39.12% on "Angle and Length" (AL), reflecting that fine-grained visual measurement is a common shortcoming of all LMMs. Closed-source models overall significantly outperform open-source models, but large-scale open-source models (such as LLaVA-NeXT-110B) are already close to the level of closed-source GPT-4V.

## Limitations & Future Work
- Problem decomposition relies on human experts, which is costly and potentially subjective.
- Only covers basic visual mathematics (5 major categories, 12 typical problems) and does not involve advanced mathematics (calculus, linear algebra, etc.).
- The cause of RM has not been deeply analyzed (data leakage or shortcut learning?).
- The Knowledge Concept Augmentation (KCA) strategy has limited effectiveness—it is effective for IK but basically ineffective for IG and RM, indicating that simply adding knowledge descriptions cannot replace genuine reasoning ability training.
- Dependencies between the 67 knowledge concepts are not modeled; future work could explore knowledge graph-driven evaluation.
- Evaluation is limited to multiple-choice and fill-in-the-blank questions, and does not involve open-ended mathematical proofs.
- Imbalance in sample sizes for some areas: The high accuracy of 86.61% in UCU might be related to simpler problem design, whereas the 39.12% in AL reflects the inherent difficulty of visual measurement.

## Related Work & Insights
- **vs MathVista**: MathVista only evaluates the final result, whereas We-Math evaluates the reasoning process and knowledge mastery. The counter-intuitive conclusion in MathVista that "university problems are simpler than primary school problems" is explained in We-Math.
- **vs MathVerse**: MathVerse attempts to evaluate reasoning paths but is based on reference answers, while We-Math is more systematic based on knowledge concept decomposition.
- **vs G-LLaVA**: G-LLaVA-13B performs well in some specialized areas but has an RM ratio as high as ~36%, showing that it might rely on patterns in training data rather than true reasoning.
- **Connection to Educational Evaluation**: The four-dimensional metrics of IK/IG/CM/RM can be analogized to knowledge diagnostic frameworks in educational psychology, providing an educational theoretical foundation for AI assessment.

## Rating

### Implementation Details
Using `bert-base-uncased` for embedding, all experiments were conducted on NVIDIA A100. Evaluation uses regular expression matching for LMM predictions and calculates accuracy.
- Novelty: ⭐⭐⭐⭐⭐ The four-dimensional evaluation metrics and knowledge-driven decomposition are both highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 17 LMMs with insightful findings.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation and systematic framework.
- Value: ⭐⭐⭐⭐⭐ Changes the paradigm of mathematical reasoning evaluation; the `testmini` subset facilitates rapid evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ACT as Human: Multimodal Large Language Model Data Annotation with Critical Thinking](../../NeurIPS2025/vlm_reasoning/act_as_human_multimodal_large_language_model_data_annotation.md)
- [\[ICLR 2026\] We-Math 2.0: A Versatile MathBook System for Incentivizing Visual Mathematical Reasoning](../../ICLR2026/vlm_reasoning/we-math_20_a_versatile_mathbook_system_for_incentivizing_visual_mathematical_rea.md)
- [\[ACL 2025\] The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights](the_role_of_visual_modality_in_multimodal_mathematical_reasoning_challenges_and_.md)
- [\[ACL 2025\] MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning](mathcoder-vl_bridging_vision_and_code_for_enhanced_multimodal_mathematical_reaso.md)
- [\[CVPR 2025\] MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts](../../CVPR2025/vlm_reasoning/mv-math_evaluating_multimodal_math_reasoning_in_multi-visual_contexts.md)

</div>

<!-- RELATED:END -->
