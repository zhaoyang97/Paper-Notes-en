---
title: >-
  [Paper Note] Automated Knowledge Component Generation and Interpretable Knowledge Tracing in Coding Problems
description: >-
  [ACL2026][Video Understanding][Knowledge Component] This paper utilizes LLMs to automatically generate and cluster Knowledge Components (KCs) for open-ended programming problems. It proposes KCGen-KT…
tags:
  - "ACL2026"
  - "Video Understanding"
  - "Knowledge Component"
  - "Knowledge Tracing"
  - "Programming Education"
  - "LLM"
  - "Interpretable Student Modeling"
date: 2026-05-08
content_hash: 8d05c5022916ffe4
---

# Automated Knowledge Component Generation and Interpretable Knowledge Tracing in Coding Problems

**Conference**: ACL2026  
**arXiv**: [2502.18632](https://arxiv.org/abs/2502.18632)  
**Code**: https://github.com/umass-ml4ed/kcgen-kt  
**Area**: AI in Education / Knowledge Tracing  
**Keywords**: Knowledge Component, Knowledge Tracing, Programming Education, LLM, Interpretable Student Modeling  

## TL;DR
This paper utilizes LLMs to automatically generate and cluster Knowledge Components (KCs) for open-ended programming problems. It proposes KCGen-KT, which converts student mastery levels for each KC into soft tokens as input for Llama 3, simultaneously improving correctness prediction and student code generation on CodeWorkout and FalconCode datasets.

## Background & Motivation
**Background**: Knowledge Tracing (KT) requires estimating a student's mastery of fine-grained knowledge points, often relying on Knowledge Components (KCs) as skill labels. Traditional KCs are typically authored by teachers or domain experts and manually mapped to problems.

**Limitations of Prior Work**: Manual KC design is costly and prone to bias. Furthermore, open-ended programming problems are significantly harder to label than multiple-choice questions. A single programming problem may have multiple correct solutions involving different skills, and student errors are diverse, unlike fixed options in multiple-choice questions.

**Key Challenge**: Knowledge tracing models require fine-grained, interpretable, and transferable KCs; however, finer-grained KCs increase the cost of manual design and annotation. Existing automated KC generation mostly focuses on multiple-choice questions and lacks support for open-ended student code submissions.

**Goal**: The authors aim to use LLMs to automatically generate the KCs required for programming problems and allow these natural language KC descriptions to directly assist KT models in predicting future student performance and code submissions while maintaining interpretability regarding "which KCs a student is weak at."

**Key Insight**: The paper connects KC generation and KT modeling into a closed loop: it first uses actual correct student code to help the LLM identify problem skills, then uses skill descriptions and student mastery as part of the LLM input for predicting the next response.

**Core Idea**: Use LLMs to generate readable KCs, employ clustering to control the level of abstraction, and project each student's mastery of KCs into differentiable soft text tokens. This allows LLM-based KT to gain both semantic knowledge and an interpretable student state.

## Method

### Overall Architecture
The approach consists of two parts. The first part is an automated KC generation and annotation pipeline: for each programming problem, diverse correct student submissions are sampled, and GPT-4o is prompted to generate necessary KCs. Similar KCs are then merged using Sentence-BERT embeddings and Hierarchical Agglomerative Clustering (HAC). Finally, GPT-4o generates the final names for each cluster and maps problems to these KC clusters, forming a Q-matrix.

The second part is the KCGen-KT model: it maintains a mastery vector for each student across all KCs. These mastery values are converted into soft tokens and fed into Llama 3 along with the next problem text and KC descriptions to predict both the correctness of the next response and the likely code submission.

### Key Designs
1. **LLM-based KC Generation and Clustering**:

	- **Function**: Automatically generate fine-grained and readable KCs for open-ended programming problems while controlling abstraction levels.
	- **Mechanism**: First, correct student code is clustered using CodeBERT embeddings. Representative solutions are sampled from different clusters, and GPT-4o is tasked with generating KCs based on the problem and diverse solutions. Subsequently, KCs are vectorized using Sentence-BERT, merged through HAC, and finally named by GPT-4o.
	- **Design Motivation**: Programming problems can have multiple correct logical paths; looking only at the problem statement or a single solution may miss necessary skills. Clustering prevents the LLM from generating redundant, overly specific, or non-generalizable KCs.

2. **KC Mastery Soft Token Conversion**:

	- **Function**: Integrate student mastery of each KC into the textual input space of the LLM.
	- **Mechanism**: The model first updates a 512-dimensional student knowledge state $h_t$ via an LSTM, followed by a linear layer and sigmoid to obtain a $k$-dimensional mastery vector $m_t\in[0,1]^k$. For the $j$-th KC, a soft token is generated as $s_t^j=m_t^j\cdot emb^{true}+(1-m_t^j)\cdot emb^{false}$, representing the student's mastery level for that KC.
	- **Design Motivation**: While LLMs excel at processing text descriptions, traditional continuous mastery values cannot be directly used as standard text tokens. Soft tokens make mastery information differentiable while allowing it to blend into the LLM representation space.

3. **Multi-task KT Objectives and Interpretable Regularization**:

	- **Function**: Simultaneously predict correctness, generate student code, and ensure monotonic interpretability between KC mastery and performance.
	- **Mechanism**: Correctness prediction uses the hidden states of Llama 3 for sigmoid classification; code prediction is performed via token-by-token generation. KC regularization uses the average mastery of relevant KCs to predict correctness. The final loss is $\mathcal{L}_{KCGen-KT}=\lambda(\mathcal{L}_{CodeGen}+\mathcal{L}_{CorrPred})+(1-\lambda)\mathcal{L}_{KC}$.
	- **Design Motivation**: Optimizing for prediction accuracy alone may result in uninterpretable hidden states. The KC loss forces "high KC mastery to correspond to high correctness probability," ensuring student profiles align with educational intuition.

### Loss & Training
The training objective includes three parts: BCE loss for correctness prediction, token-level negative log-likelihood for code generation, and BCE regularization for KC mastery against correctness. The model utilizes instruction-tuned Llama 3 8B with LoRA fine-tuning and 8-bit quantization. In KCGen-KT, the learning rates are 1e-5 for Llama 3, 5e-4 for LSTM, and 1e-4 for the mastery linear layer. Experiments are repeated across 5 random train-validation-test splits.

## Key Experimental Results

### Main Results
Two datasets consisting of real open-ended programming submissions were used: CodeWorkout (246 students, 50 Java problems, 10,834 first submissions) and FalconCode (3,267 students, 157 Python problems, 28,617 first submissions).

| Dataset | Method | AUC | F1 | Accuracy | CodeBLEU |
|--------|------|-----|----|----------|----------|
| CodeWorkout | Code-DKT | 0.766 | 0.672 | 0.724 | - |
| CodeWorkout | TIKTOC* | 0.788 | 0.666 | 0.726 | 0.507 |
| CodeWorkout | KCGen-KT (Human KCs) | 0.797 | 0.706 | 0.727 | 0.557 |
| CodeWorkout | KCGen-KT (Generated KCs) | 0.816 | 0.727 | 0.746 | 0.580 |
| FalconCode | Code-DKT | 0.709 | 0.552 | 0.617 | - |
| FalconCode | TIKTOC* | 0.728 | 0.585 | 0.633 | 0.427 |
| FalconCode | KCGen-KT (Human KCs) | 0.752 | 0.599 | 0.700 | 0.473 |
| FalconCode | KCGen-KT (Generated KCs) | 0.771 | 0.645 | 0.712 | 0.498 |

LLM-generated KCs outperformed human KCs in both datasets for correctness prediction and code generation tasks. The paper reports that improvements relative to baselines are statistically significant (p < 0.05).

### Ablation Study
| Configuration | AUC | F1 | Accuracy | CodeBLEU | Description |
|------|-----|----|----------|----------|------|
| KCGen-KT | 0.812 | 0.723 | 0.724 | 0.569 | CodeWorkout ablation baseline |
| w/o Correct Sol. | 0.789 | 0.674 | 0.704 | 0.529 | Missing correct student solutions leads to incomplete KCs |
| w/ Incorrect Sol. | 0.773 | 0.651 | 0.700 | 0.516 | Adding incorrect solutions introduces noise |
| w/o KC Loss | 0.791 | 0.680 | 0.709 | 0.540 | Interpretable mastery regularization contributes to performance |
| w/o ICL Ex. | 0.782 | 0.677 | 0.705 | 0.539 | KCs become more abstract and less effective without examples |
| Code → AST | 0.784 | 0.691 | 0.715 | 0.546 | AST representation is secondary to raw code text |
| Generated Code | 0.807 | 0.706 | 0.721 | 0.557 | LLM-generated solutions are less diverse than real student code |

### Key Findings
- KC abstraction level is critical. In CodeWorkout, 50 medium-grained clusters achieved the best results (0.816 AUC). Reducing this to 10 high-level KCs dropped performance to 0.794 AUC.
- Actual correct student submissions are vital. Removing correct submissions or using only LLM-generated code degrades KC coverage, as real student solutions contain richer strategic diversity.
- Human evaluation confirms the quality of generated KCs: the interpretable proportion of LLM-generated KCs reached 98.6% (vs. 94.6% for baseline). KC mapping precision was 93.2% (vs. 92.5%). In recall comparisons, generated KCs were judged to have equal or better coverage in 96% of cases.
- Learning curve analysis shows LLM-generated KCs align better with the power law of practice: weighted $R^2$ was 0.21, higher than the 0.18 recorded for human-written KCs.

## Highlights & Insights
- The key contribution is not merely "using LLM for labeling," but rather transforming labels into semantic inputs that downstream KT models can utilize, maintaining end-to-end differentiability via soft tokens.
- Sampling diverse correct code is a clever design choice. The skill set of an open-ended programming problem is defined not just by the problem statement, but by the space of feasible solutions. Real student code reflects this space better than a single "gold" answer.
- The fact that generated KCs outperform human KCs is illustrative: manual labels can be too coarse or outdated, whereas LLMs can generate natural, function-level skill descriptions better suited for LLM-based KT.

## Limitations & Future Work
- KC generation relies heavily on in-context examples; without human-curated examples, zero-shot generation tends to be abstract, failing to capture low-level skills.
- Multi-KC problems lack reliable ground-truth annotations; current KC mapping correctness labeling requires further validation.
- The experiment is limited to Computer Science education (Java/Python). Generalization to Math, Science, or conversational tutoring remains to be verified.
- While human evaluation proves KC interpretability and mapping quality, the ultimate value depends on improving actual student learning outcomes, which requires classroom deployment or A/B testing.

## Related Work & Insights
- **vs Code-DKT**: Code-DKT uses student code content for KT but lacks natural language KC semantics and explicit mastery interpretation; KCGen-KT uses code, problem text, and KC descriptions simultaneously.
- **vs TIKTOC**: TIKTOC uses Llama 3 for generative KT but does not explicitly model KC mastery; KCGen-KT derives its advantage from KC semantics and the KC loss.
- **vs Manual KC Labeling**: Manual labels are expensive and their granularity may not match model needs. LLM-generated KCs showed superior predictive performance and interpretable evaluation results in this study.
- **Insight**: LLMs in educational settings should serve not just as answer generators, but as "knowledge structure generators" that help build interpretable student models.

## Rating
- Novelty: ⭐⭐⭐⭐ While automated KC generation exists, its integration with soft-token LLM-based KT is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Solid evaluation across two real-world datasets, multiple strong baselines, abstraction analysis, ablation studies, human evaluation, and learning curves.
- Writing Quality: ⭐⭐⭐⭐ The methodological chain is clear, with educational motivations and model details well-explained.
- Value: ⭐⭐⭐⭐⭐ Strong practical value for programming education, intelligent tutoring systems, and interpretable knowledge tracing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition](../../AAAI2026/video_understanding/sugar_learning_skeleton_representation_with_visual-motion_knowledge_for_action_r.md)
- [\[ICLR 2026\] AnveshanaAI: A Multimodal Platform for Adaptive AI/ML Education through Automated Question Generation and Interactive Assessment](../../ICLR2026/video_understanding/anveshanaai_a_multimodal_platform_for_adaptive_aiml_education_through_automated_.md)
- [\[ACL 2026\] TRACE: Evidence Localization-based Multi-Video Event Understanding and Statement Generation](trace_evidence_grounding-guided_multi-video_event_understanding_and_claim_genera.md)
- [\[ICLR 2026\] AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference](../../ICLR2026/video_understanding/adaem_an_adaptively_and_automated_extensible_measurement_of_llms_value_differenc.md)
- [\[CVPR 2026\] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](../../CVPR2026/video_understanding/towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)

</div>

<!-- RELATED:END -->
