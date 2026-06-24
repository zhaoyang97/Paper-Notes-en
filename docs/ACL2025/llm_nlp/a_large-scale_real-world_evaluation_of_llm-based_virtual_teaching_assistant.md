---
title: >-
  [Paper Note] A Large-Scale Real-World Evaluation of an LLM-Based Virtual Teaching Assistant
description: >-
  [ACL 2025][LLM (Other)][Virtual Teaching Assistant] A RAG-based LLM Virtual Teaching Assistant (VTA) was deployed in a graduate-level AI programming course with 477 students at KAIST. Through longitudinal analysis of three rounds of surveys (472 respondents) and 3869 interaction logs, the study revealed that the VTA significantly reduced students' psychological barriers to asking questions. While satisfaction among high-frequency users continuously improved over time…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Virtual Teaching Assistant"
  - "RAG"
  - "Educational Technology"
  - "User Study"
  - "LLM Deployment"
date: 2026-05-08
content_hash: dd9e555b90ea15c9
---

# A Large-Scale Real-World Evaluation of an LLM-Based Virtual Teaching Assistant

**Conference**: ACL 2025  
**arXiv**: [2506.17363](https://arxiv.org/abs/2506.17363)  
**Code**: [GitHub](https://github.com/sean0042/VTA)  
**Area**: LLM Applications / AI in Education  
**Keywords**: Virtual Teaching Assistant, RAG, Educational Technology, User Study, LLM Deployment

## TL;DR

A RAG-based LLM Virtual Teaching Assistant (VTA) was deployed in a graduate-level AI programming course with 477 students at KAIST. Through longitudinal analysis of three rounds of surveys (472 respondents) and 3869 interaction logs, the study revealed that the VTA significantly reduced students' psychological barriers to asking questions. While satisfaction among high-frequency users continuously improved over time, trust in the VTA remained lower than in human TAs.

## Background & Motivation

**Background**: LLM-driven Virtual Teaching Assistants (VTAs) have been pilot-deployed in several universities (e.g., JeepyTA at UPenn, Jill Watson at Georgia Tech), demonstrating potential in automating responses to student inquiries.

**Limitations of Prior Work**: (a) Most existing VTA studies rely on small-scale user surveys or automatic LLM evaluation, lacking empirical validation in large-scale real-world classrooms; (b) Deep analysis of instructor-student interaction logs is missing, making it difficult to understand the actual role of VTAs in the learning process; (c) Most VTA systems are closed-source, restricting reproduction and practical adoption.

**Key Challenge**: The contradiction between the need for personalized feedback in large-scale courses and limited teaching resources, combined with students' psychological barriers to asking human instructors due to fear of judgment.

**Goal**: To systematically answer the effectiveness of VTAs in real classrooms, the evolution of student acceptance, and their complementary relationship with human TAs through large-scale deployment and longitudinal evaluation.

**Key Insight**: Evaluating the VTA multi-dimensionally by combining three rounds of surveys (pre-, mid-, and post-deployment) with analysis of interaction logs.

## Method

### Overall Architecture

The VTA system is constructed using LangChain, Streamlit, and LangSmith: (1) Course materials (PDFs, Jupyter Notebooks, and transcriptions of lecture recordings) are chunked into 2048-token segments and stored in a FAISS vector database; (2) Upon receiving user questions, a context-aware search query is generated first to retrieve the top-5 documents; (3) GPT-4o-mini generates responses by combining the retrieved documents, conversation history, and system prompts.

### Key Designs

1. **Context-Aware Query Generation**:
    - **Function**: Synthesizes the conversation history and the latest question into a comprehensive search query using GPT-4o-mini within multi-turn dialogues.
    - **Mechanism**: Simply embedding the latest question can lose context (e.g., ambiguity in "what is that task?"); thus, integrating conversational context to generate complete queries is required.
    - **Design Motivation**: To ensure retrieval accuracy in multi-turn conversation scenarios.

2. **Course Materials Vector Database**:
    - **Function**: Processes 59 course documents (PDFs, notebooks, lecture audio) into 1,502 chunks.
    - **Mechanism**: Audio is transcribed using Whisper-1. Each chunk is prefixed with metadata (date and title) to provide context. FAISS is used for similarity search with `text-embedding-3-large` embeddings.
    - **Design Motivation**: To guarantee that VTA responses are grounded in course content and prevent irrelevant generic answers.

3. **Three-Round Longitudinal Survey Design**:
    - **Function**: Conducts mandatory surveys for all 472 students across three phases: pre-, mid-, and post-deployment.
    - **Mechanism**: Evaluates the VTA across four dimensions: Helpfulness, Trustworthiness, Appropriateness, and Comfortableness (in comparison to human TAs).
    - **Design Motivation**: To track the evolution of student perception over time instead of relying on a single snapshot evaluation.

### Loss & Training

No training required. The system operated for 14 weeks with an API cost of approximately $180. Student IDs were used for authentication, and LangSmith logged all conversation histories.

## Key Experimental Results

### Main Results

Approximately 50% of the 472 students utilized the VTA, generating 916 conversations and 3,869 QA interactions.

| Evaluation Dimension | Pre-Deployment | Post-Deployment | Human TA |
|---|---|---|---|
| Helpfulness | 3.64 | 3.54 | 3.86 |
| Trustworthiness | 2.97 | 3.21 | 3.71 |
| Appropriateness | 3.59 | 3.69 | 3.78 |
| Comfortableness (vs. Human) | +0.58 | +0.62 | - |

Question type distribution comparison (VTA vs. Human TA):

| Question Type | VTA Proportion | Human TA Proportion |
|---|---|---|
| Project-related | 49.1% | 52.1% |
| Theoretical | 26.2% | 9.7% |
| Programming | 14.1% | 18.8% |
| Administrative | 10.6% | 19.4% |

### Ablation Study

Changes in satisfaction grouped by usage frequency (High-frequency users: $\ge 18$ interactions):

| User Group | Change in Helpfulness | Change in Trustworthiness | Change in Comfortableness |
|---|---|---|---|
| High-frequency users (Groups A, B, C) | Significant increase $\uparrow$ ($p=0.043$) | Increase $\uparrow$ | Significant increase $\uparrow$ ($p < 0.001$) |
| Low-frequency users (Group D) | Decrease $\downarrow$ ($3.72 \rightarrow 3.26$) | Increase $\uparrow$ | Increase $\uparrow$ |
| Students hesitant to ask humans | Usage: 13.2 vs. 7.8 times | - | Higher comfortableness (0.76 vs. 0.47) |

### Key Findings

1. The volume of students' queries to the VTA was more than 25 times greater than that directed to human TAs, with a significantly larger proportion of theoretical questions, demonstrating that the VTA reduced psychological barriers to deep learning exploration.
2. Students from non-CS backgrounds used the VTA more frequently (80% of high-frequency users originated from this demographic), and students with no programming experience averaged 62.2 interactions.
3. 13% of conversations contained social interaction elements (greetings, appreciation, humor); these students showed an average usage frequency (27.8 times) that was 2.4 times higher than that of purely informational-driven users (11.4 times).
4. 58% of students admitted to avoiding asking questions to human TAs due to discomfort, and this subgroup consistently rated their comfort with the VTA highest.

## Highlights & Insights

- **Large-Scale Empirical Value**: A large-scale real deployment study involving 477 students, 14 weeks, and 3,869 interactions fills the gap in empirical research on VTAs.
- **Reducing Psychological Barrier is the VTA's Greatest Value**: The major benefit lies not in replacing humans with respect to answer quality, but rather in psychologically encouraging more students to ask questions.
- **More Usage Correlates with Higher Satisfaction**: High-frequency users exhibited significant improvements in helpfulness, trustworthiness, and comfortableness ratings. In contrast, ratings from low-frequency users decreased, possibly due to a mismatch between initial expectations and actual experience.
- **Unexpected Finding in Social Interaction**: Students' proactive efforts to establish human-like relationships with virtual agents correlated with higher usage frequencies.

## Limitations & Future Work

- Validated only within a programming course; the effectiveness in humanities and social sciences remains unexplored.
- Streaming outputs were not implemented, causing some students to perceive response delivery as slow.
- Vector retrieval coverage for colloquial classroom discussion content was insufficient; hybrid retrieval methods (dense + BM25) could be considered.
- A quantitative analysis of the VTA's direct impact on academic performance is missing.
- System prompts can be adjusted to encourage more comprehensive explanations that extend beyond only course materials.

## Related Work & Insights

- **Jill Watson (Georgia Tech)**: A pioneer in the VTA landscape, but reliant on IBM Watson classifier systems and unable to generate context-adaptive responses.
- **JeepyTA (UPenn)**: A similar system but lacking large-scale user studies.
- **Insights**: The VTA acts as a complement, rather than a replacement, to human TAs. Efforts should focus on lowering the barrier to asking questions and increasing accessibility, rather than blindly pursuing superior response quality to humans.

## Rating

- Novelty: ⭐⭐⭐ System-level contribution; the VTA architecture itself is standard, with the primary innovation lying in the large-scale evaluation design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Rigorous statistical methods involving a longitudinal survey of 472 participants paired with the analysis of 3,869 interaction logs.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with thorough data presentation and detailed appendices.
- Value: ⭐⭐⭐⭐ High practical reference value for LLM deployments in education; the open-sourced system lowers replication barriers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks](llm_vs_human_judges_study.md)
- [\[ICML 2025\] Expert Evaluation of LLM World Models: A High-Tc Superconductivity Case Study](../../ICML2025/llm_nlp/expert_evaluation_of_llm_world_models_a_high-t_c_superconductivity_case_study.md)
- [\[ACL 2025\] ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow Development](comfyui-copilot_an_intelligent_assistant_for_automated_workflow_development.md)
- [\[ACL 2025\] Conversational Quality Assessment: A Large-Scale Corpus and Comprehensive Study](conversational_quality_assessment_a_large-scale_corpus_and_comprehensive_study.md)
- [\[ACL 2025\] ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities](consistencychecker_tree_evaluation.md)

</div>

<!-- RELATED:END -->
