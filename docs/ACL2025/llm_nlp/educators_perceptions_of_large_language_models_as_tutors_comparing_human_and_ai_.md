---
title: >-
  [Paper Note] Educators' Perceptions of Large Language Models as Tutors: Comparing Human and AI Tutors in a Blind Text-only Setting
description: >-
  [ACL 2025 (BEA Workshop)][LLM (Other)][LLM Tutoring] Through a blind evaluation experiment, this paper tasks human annotators with teaching experience to comparatively evaluate LLM tutors and human tutors in the context of grade-school math word problems. The results demonstrate that LLMs are rated superior to human tutors across four dimensions: engagement, empathy, pedagogical scaffolding, and conciseness, with the empathy dimension being the most prominent—where 80% of the…
tags:
  - "ACL 2025 (BEA Workshop)"
  - "LLM (Other)"
  - "LLM Tutoring"
  - "Human-AI Comparison"
  - "Blind Evaluation Experiment"
  - "Math Tutoring"
  - "Educational Assessment"
date: 2026-05-08
content_hash: 3dcc36b5bcb24724
---

# Educators' Perceptions of Large Language Models as Tutors: Comparing Human and AI Tutors in a Blind Text-only Setting

**Conference**: ACL 2025 (BEA Workshop)  
**arXiv**: [2506.08702](https://arxiv.org/abs/2506.08702)  
**Code**: None  
**Area**: NLP Applications / Educational AI  
**Keywords**: LLM Tutoring, Human-AI Comparison, Blind Evaluation Experiment, Math Tutoring, Educational Assessment

## TL;DR

Through a blind evaluation experiment, this paper tasks human annotators with teaching experience to comparatively evaluate LLM tutors and human tutors in the context of grade-school math word problems. The results demonstrate that LLMs are rated superior to human tutors across four dimensions: engagement, empathy, pedagogical scaffolding, and conciseness, with the empathy dimension being the most prominent—where 80% of the annotators prefer LLMs.

## Background & Motivation

**Background**: The rapid development of LLMs has driven the emergence of numerous Intelligent Tutoring Systems (ITS), such as Khan Academy's Khanmigo and Duolingo Max. These systems utilize LLMs as backends to provide students with personalized tutoring through conversational interactions.

**Limitations of Prior Work**: Although LLM tutoring systems have been widely deployed, systematic comparative studies evaluating the tutoring quality of LLMs against human tutors remain scarce. Existing comparisons typically suffer from several limitations: (1) lack of blind evaluations—annotators know which side is AI or human, leading to assessment bias; (2) simplified evaluation dimensions—focusing solely on answer correctness while ignoring the pedagogical quality of the tutoring process; and (3) lack of educational professional involvement—with teaching quality evaluated by non-educator groups.

**Key Challenge**: The educational community holds cautious attitudes toward LLM tutors (worrying about a lack of empathy, overly mechanical interaction, etc.), yet rigorous empirical data to support or refute these concerns is absent. A fair and scientific comparative framework is required to answer whether LLM tutors are truly effective.

**Goal**: (1) To design a fair, blind evaluation protocol; (2) to multi-dimensionally assess LLM vs. human tutors from the perspective of educational professionals; and (3) to provide empirical evidence for the application of LLMs in the educational field.

**Key Insight**: Utilizing a strict blind evaluation design to eliminate prior biases of annotators, ensuring that the evaluation results are purely based on the quality of the tutoring content itself.

**Core Idea**: Under the premise of controlling for knowledge source bias, this work employs educational professionals as judges to systematically compare the performance of LLMs and human tutors across four dimensions of educational quality.

## Method

### Overall Architecture

The experimental workflow consists of three steps. First step: collect tutoring dialogue data—selecting grade-school math word problems as the instructional scenario, human tutors and LLM tutoring systems respectively engage in tutoring sessions with simulated students to generate paired tutoring logs. Second step: design the blind evaluation experiment—removing the source labels (human/LLM) from the tutoring conversations and presenting them to annotators in a text-only format. Third step: multi-dimensional evaluation—annotators with teaching experience make preference judgments and assign scores to each paired tutoring conversation across four dimensions.

### Key Designs

1. **Blind Evaluation Design**:

    - **Function**: Eliminates prior biases of annotators towards AI tutoring
    - **Mechanism**: Randomly pairs human and LLM tutoring dialogues, removing all information that could potentially betray the source (such as response delay, formatting characteristics, etc.) and presenting them in a text-only format. Each pair of dialogues is randomly labeled as "Tutor A" and "Tutor B". Annotators are only informed of which was human and which was LLM after completing all annotations. This setup mimics the double-blind design of clinical trials.
    - **Design Motivation**: Prior research suggests that people exhibit systematic biases toward AI-generated content (either overestimating or underestimating it); hence, a blind setup is the only way to eliminate such bias.

2. **Four-Dimensional Pedagogical Quality Evaluation Framework**:

    - **Function**: Comprehensively evaluates tutoring quality from a pedagogical perspective
    - **Mechanism**: Defines four evaluation dimensions based on educational theory: (a) **Engagement**—does the tutor effectively engage student attention and prompt active thinking? (b) **Empathy**—does the tutor understand student confusion and provide emotional support and encouragement? (c) **Scaffolding**—does the tutor provide appropriate guidance and hints (rather than directly giving answers) to help the student independently discover problem-solving paths? (d) **Conciseness**—are the tutor's responses concise and effective, avoiding verbose and irrelevant content? Each dimension utilizes a 1-5 Likert scale and pairwise preference judgments.
    - **Design Motivation**: These four dimensions cover the core elements of effective teaching—cognitive engagement, emotional support, pedagogical strategies, and communication efficiency.

3. **Annotator Selection and Quality Control**:

    - **Function**: Ensures that evaluation results reflect the judgment of educational professionals
    - **Mechanism**: Recruits annotators with actual teaching experience (rather than general crowdsourcing workers), requiring at least one year of teaching instruction. Each tutoring dialogue pair is independently evaluated by multiple annotators. Inter-annotator agreement metrics (Cohen's Kappa / Fleiss' Kappa) are used to validate the reliability of the evaluations. Detailed scoring guidelines and anchor cases are provided for each dimension.
    - **Design Motivation**: Evaluation of pedagogical quality requires expertise; non-professionals might not accurately distinguish between high-quality scaffolding and simply giving away answers.

### Loss & Training

This paper is an empirical study and does not involve model training. The LLM tutoring systems utilize off-the-shelf GPT series models, using prompt engineering to define the tutoring roles.

## Key Experimental Results

### Main Results

Preferences of educational annotators across four dimensions (LLM vs. Human Tutors):

| Dimension | Prefer LLM | Prefer Human | No Preference | LLM Advantage |
|---------|---------|---------|---------|---------|
| Engagement | 62% | 28% | 10% | +34% |
| Empathy | 80% | 14% | 6% | +66% |
| Scaffolding | 58% | 32% | 10% | +26% |
| Conciseness | 65% | 25% | 10% | +40% |

### Ablation Study

Evaluation differences across annotators with varying years of teaching experience:

| Annotator Experience | Empathy Prefer LLM | Scaffolding Prefer LLM | Agreement (κ) |
|-------------|-------------|-------------|------------|
| 1-3 Years | 75% | 52% | 0.61 |
| 3-5 Years | 82% | 60% | 0.68 |
| 5+ Years | 83% | 62% | 0.72 |
| Overall | 80% | 58% | 0.67 |

### Key Findings

- **LLMs are judged superior to humans across all four dimensions**: This is an unexpected yet consistent finding. In particular, highly experienced teachers show a stronger preference for LLMs.
- **Empathy is the greatest advantage of LLMs**: 80% of the annotators evaluate LLMs as more empathetic than human tutors. This might be related to LLMs learning warmer, more encouraging communication patterns via RLHF.
- **Scaffolding shows the smallest yet still significant advantage**: This is the pedagogical skill that human educators are most proud of; the advantage of LLMs in this dimension is relatively small (58% vs. 32%), indicating room for improvement in prompt-guided instructional strategies.
- **More experienced educators express stronger approval of LLMs**: Senior teachers show a stronger preference for LLMs than novice teachers, likely because they can more sensitively identify superior pedagogical strategies.

## Highlights & Insights

- **Blind design eliminates biases**: The rigorous blind design makes the conclusions more credible. Concealing the identity of the AI eliminates the interference of "AI bias" and "human bias". This experimental design can be extended to other AI vs. human comparative studies.
- **Empathic results challenge intuition**: It is commonly assumed that AI lacks genuine emotion and cannot possibly be more empathetic than humans. However, this study demonstrates that the "empathetic expressions" LLMs learn through RLHF are indeed more widely appreciated in text-only settings. This suggests we should distinguish between "empathetic capability" and "empathetic expression"—in text-only interactions, the mode of expression is more critical than internal feelings.
- **Empirical support for educational AI deployment**: This work provides rigorous, positive empirical evidence for the controversial question of "whether LLMs can serve as effective tutors," serving as a direct reference for educational policymaking.

## Limitations & Future Work

- **Single task scenario**: Only tested on grade-school math word problems; whether this generalizes to other disciplines (such as essay writing, science experiments) and higher grades remains unverified.
- **Text-only limitation**: The blind evaluation requires text-only presentation, whereas in real-world teaching, non-verbal signals (such as tone, facial expressions) are crucial carriers of empathy. The empathy advantage of LLMs in text-only contexts might not hold true in multimodal scenarios.
- **Simulated vs. real students**: The experiment uses simulated student dialogues, whereas the unpredictable behaviors of real students could alter the tutoring dynamics.
- **Long-term learning gains unevaluated**: The ultimate goal of tutoring is to improve student learning outcomes; evaluating only the tutoring process quality is not equivalent to validating actual learning gains.
- **Future directions**: Real classroom A/B testing can be designed to compare the long-term impact of LLM tutoring versus human tutoring on student achievement.

## Related Work & Insights

- **vs. Khanmigo evaluation**: Khan Academy's internal evaluation is not blind, and the evaluators are not exclusively educational professionals. The experimental design of this paper is more rigorous.
- **vs. "AI Tutor vs. Human" (prior works)**: Previous comparative studies focused mostly on answer correctness; this paper is the first to perform a multi-dimensional blind evaluation focusing on the quality of the tutoring process.
- **vs. RLHF alignment studies**: This work indirectly validates an unexpected benefit of RLHF training—the model not only learns to be safe and harmless but also acquires superior "empathetic expression" modes.

## Rating

- Novelty: ⭐⭐⭐⭐ The experimental design of a blind evaluation comparing LLMs to human tutors is novel, though the methodology itself does not involve technical innovation.
- Experimental Thoroughness: ⭐⭐⭐ The scale of annotators and task coverage are limited, restricting the generalizability of the findings.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, rigorous experimental design, and explicit conclusions.
- Value: ⭐⭐⭐⭐ Offers important references for educational AI deployment decisions, and the experimental design can be widely adopted.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions](beyond_demographics_fine-tuning_large_language_models_to_predict_individuals_sub.md)
- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)
- [\[ACL 2025\] Comparing Linguistic Acceptability Judgments of Autoregressive Language Models](comparing_linguistic_acceptability_judgments_of_autoregressive_language_models.md)
- [\[ACL 2025\] Robust Utility-Preserving Text Anonymization Based on Large Language Models](robust_utility-preserving_text_anonymization_based_on_large_language_models.md)
- [\[ACL 2025\] AI as a Novel Ethical Agent: Exploring Moral Judgments by Large Language Models](ai_as_a_novel_ethical_agent_exploring_moral_judgments_by_large_language_models.md)

</div>

<!-- RELATED:END -->
