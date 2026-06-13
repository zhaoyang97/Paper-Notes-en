---
title: >-
  [Paper Note] Retrieval-Augmented Tutoring for Algorithm Tracing and Problem-Solving in AI Education
description: >-
  [ACL2026][Information Retrieval & RAG][RAG tutoring] Ours proposes KITE, a RAG tutoring system for course materials in algorithm tracing and problem-solving. Through intent-aware Socratic feedback and multi-stage retriev…
tags:
  - "ACL2026"
  - "Information Retrieval & RAG"
  - "RAG tutoring"
  - "Algorithm learning"
  - "Socratic scaffolding"
  - "Intent classification"
  - "Simulated student evaluation"
date: 2026-05-08
content_hash: acc0f391d46a3c4b
---

# Retrieval-Augmented Tutoring for Algorithm Tracing and Problem-Solving in AI Education

**Conference**: ACL2026  
**arXiv**: [2605.12988](https://arxiv.org/abs/2605.12988)  
**Code**: No public code  
**Area**: Information Retrieval / AI Education / Intelligent Tutoring Systems  
**Keywords**: RAG tutoring, Algorithm learning, Socratic scaffolding, Intent classification, Simulated student evaluation

## TL;DR
Ours proposes KITE, a RAG tutoring system for course materials in algorithm tracing and problem-solving. Through intent-aware Socratic feedback and multi-stage retrieval, it demonstrates superior grounding and pedagogical scaffolding across automated metrics, simulated students, and expert reviews.

## Background & Motivation
**Background**: Students widely use LLMs like ChatGPT for explanations, feedback, and problem-solving assistance. RAG provides a natural solution for educational scenarios by grounding responses in course slides, textbooks, and historical materials, reducing erroneous explanations that deviate from the course context.

**Limitations of Prior Work**: Course grounding does not equate to pedagogical effectiveness. A RAG system might provide a complete answer directly even if it retrieves relevant content, allowing students to bypass the reasoning process they ought to practice; it may also fail to distinguish between different help needs such as "asking about concepts," "debugging errors," or "validating algorithm traces."

**Key Challenge**: Educational assistants must accurately cite course materials while supporting learning in an appropriate manner. Algorithm courses, in particular, require students to perform tracing, debugging, and procedure application independently. Therefore, the system should not merely provide FAQ-style answers but offer layered hints, guiding questions, and error localization.

**Goal**: The authors construct KITE, combining RAG retrieval with pedagogical intent-aware response generation. It also proposes an evaluation framework that measures both the groundedness of non-procedural answers and the ability of the feedback to help correct reasoning using simulated students and expert reviews.

**Key Insight**: The paper transforms educational RAG from an "answer-giving QA tool" into a "tutor that modulates support based on student intent." It prioritizes response strategies over simple retrieval hit rates.

**Core Idea**: Use multi-stage course material retrieval to ensure content reliability, then use intent classification to decide whether the response should be a direct explanation, conceptual inquiry, validation, debugging, or algorithm tracing, thereby merging RAG grounding with Socratic scaffolding.

## Method

### Overall Architecture
KITE consists of five stages: document ingestion and cleaning, embedding generation, multi-stage retrieval, intent-aware generation, and session management. The system extracts course PDFs page-by-page, removes noise like headers and footers, and utilizes section-aware chunking to create chunks of approximately 500 characters with a 100-character overlap. Each chunk is encoded into a 3072-dimensional vector using text-embedding-3-large and stored in FAISS. Upon a student query, the system performs dense retrieval for the top-50, combines it with BM25, MMR, cross-encoder reranking, and course-source boosting to inject the top-8 chunks into the GPT-5 prompt. On the generation side, student intent is identified first, followed by the selection of a corresponding tutoring strategy.

### Key Designs
1. **Multi-stage Course Material Retrieval Pipeline**:

	- **Function**: Find evidence in course materials that is semantically relevant, terminologically precise, and low in redundancy.
	- **Mechanism**: Step 1: Dense bi-encoder retrieval for top-50; Step 2: Hybrid retrieval setting dense similarity to 70% and BM25 to 30%, balancing semantics with algorithmic terminology; Step 3: Deduplication using MMR with $\lambda=0.7$; Step 4: Reranking with cross-encoder/ms-marco-MiniLM-L-6-v2; Finally, source-based boosting for official course materials, adding 0.3 points to chunks with reranking scores above 0.6.
	- **Design Motivation**: Algorithm learning queries often contain specific terms, variable names, pseudo-code, and step names. Pure dense retrieval might miss literal matches, while pure BM25 might miss semantically equivalent expressions. The multi-stage pipeline improves groundedness and reduces irrelevant context pollution.

2. **Intent Classification and Pedagogical Strategy Routing**:

	- **Function**: Enable the system to choose different response styles based on student help needs.
	- **Mechanism**: KITE uses keywords and pattern-matching to classify queries into five categories: Direct Question, Conceptual Question, Algorithm Validation, Debugging, and Algorithm Tracing, with an additional "answer evaluation mode." Direct and conceptual questions are explained based on course materials; conceptual questions include reflective prompts; algorithm validation provides short evaluations, confirmation of correct parts, and leading questions; debugging uses step-by-step hints for self-checking; tracing maintains states like OPEN/CLOSED, current node, path, and cost according to course rules.
	- **Design Motivation**: The risk in educational scenarios is not just answering incorrectly, but "answering too directly." Intent-aware routing allows the system to control the level of support across different tasks, avoiding the leakage of full solutions where practice is required.

3. **Feedback Evaluation via Simulated Students + Expert Review**:

	- **Function**: Evaluate whether KITE's feedback improves student answers rather than just measuring single-turn answer similarity.
	- **Mechanism**: For procedural and tracing problems, the authors use Meta-Llama-3.1-70B-Instruct as a proxy student: in the first round, the student model answers independently; KITE provides feedback; in the second round, the student model revises the answer based on feedback. Experts review the Round 1, KITE feedback, and Round 2 to determine if the revision improved and score based on mistake remediation, scaffolding, guidance, coherence, and tone.
	- **Design Motivation**: A good tutor for procedural tasks does not necessarily output text most similar to a reference answer but instead promotes corrected reasoning in the learner. The simulated student pipeline is a low-cost evaluation method prior to deployment.

### Loss & Training
KITE is a system integration rather than a newly trained model. The retrieval side uses text-embedding-3-large, FAISS, BM25, MMR, and MiniLM cross-encoder. The generation side uses GPT-5 with the top-8 retrieved chunks injected into the prompt. Automated evaluation uses gpt-4o-mini as a RAGAs judge, and embedding similarity uses text-embedding-3-small.

## Key Experimental Results

### Main Results
Evaluation data comes from a university Introduction to AI course, totaling 109 questions with instructor-verified reference answers: 42 algorithmic questions, 51 procedural questions, and 16 direct-retrieval questions. RAGAs is used for the 58 non-procedural questions.

| RAGAs Metric | Mean | Std. Dev. | Explanation |
|------------|------|-----------|------|
| Faithfulness | 0.8486 | 0.2103 | Most answers are supported by the retrieved context |
| Answer Relevance | 0.7558 | 0.2032 | Responses are well-aligned with the original questions |
| Context Relevance | 0.9352 | 0.1905 | Retrieved contexts are highly relevant |
| Answer Similarity | 0.7586 | 0.0923 | Semantically close and stable compared to instructor answers |
| Factual Correctness | 0.4483 | 0.2477 | Claim-level metrics are relatively low, affected by reference phrasing |
| Answer Correctness | 0.6363 | 0.1810 | Synthesis of factual correctness and similarity |

Expert reviews covered 44 simulated student interaction triples with high annotation consistency (Cohen’s $\kappa=0.88$, raw agreement 98.15%).

| Expert Evaluation Dimension | Yes | No | N/A | Conclusion |
|--------------|-----|----|-----|------|
| Mistake Remediation: Identifying | 63.63% | 6.82% | 29.55% | N/A usually means student was correct in Round 1 |
| Mistake Remediation: Acknowledging | 63.63% | 6.82% | 29.55% | Recognizes and acknowledges errors when applicable |
| Scaffolding | 93.18% | 6.82% | 0% | Strong scaffolded support |
| Guidance | 93.18% | 6.82% | 0% | Clear instructions for the next step |
| Coherence: Naturalness | 93.18% | 6.82% | 0% | Natural dialogue |
| Tone: Encouraging | 93.18% | 6.82% | 0% | Highly supportive tone |

### Ablation Study
The paper does not perform traditional module ablation, but the Round 1 → Round 2 transition of the simulated student serves as a behavioral analysis of KITE's feedback effectiveness. Out of 27 interactions that were incorrect in the first round, 24 improved after KITE's feedback, an improvement rate of 88.89%.

| Answer State Transition | Count | Percentage | Interpretation |
|--------------|-------|------|------|
| Incorrect → Correct | 1 | 2.27% | Small number of complete corrections |
| Incorrect → Partially Correct | 3 | 6.82% | Progressed from incorrect to partially correct |
| Already Correct | 17 | 38.64% | Correct in round one; no correction needed |
| Partially Correct → Correct | 14 | 31.82% | Most common type of improvement |
| Partially Correct → Partially Correct with Improvement | 6 | 13.63% | Quality improved but not yet fully correct |
| N/A | 3 | 6.82% | Unsuitable for transition judgment |

### Key Findings
- KITE exhibits strong retrieval grounding: Context Relevance 0.9352 and Faithfulness 0.8486 indicate that the multi-stage retrieval effectively provides course-relevant evidence for responses.
- Factual Correctness is only 0.4483, but Answer Similarity reaches 0.7586; the authors suggest that RAGAs claim-level overlap is not friendly towards pedagogical paraphrasing and supplementary explanations.
- Experts found that 93.18% of the feedback possessed good scaffolding, guidance, coherence, and tone, showing KITE can express information in a pedagogically appropriate manner beyond just retrieval.
- Simulated students improved in 88.89% of incorrect interactions, particularly in the "Partially Correct → Correct" transition (31.82%), demonstrating that feedback helps bridge reasoning gaps.

## Highlights & Insights
- The highlight of the paper is extending the evaluation of educational RAG from "is the answer right" to "does the feedback help the student improve." This aligns better with the goals of tutoring systems.
- KITE's intent taxonomy is simple but practical. Direct, Conceptual, Validation, Debugging, and Tracing cover common help requests in algorithm courses.
- The parameters of the multi-stage retrieval pipeline are highly engineered: 70% dense, 30% BM25, MMR $\lambda=0.7$, and a 0.3 boost for official materials with scores > 0.6. These details are valuable for reproducing educational RAG.
- Simulated student evaluation is not final proof but serves as an effective safety screen before classroom deployment, identifying if feedback over-reveals answers or fails to drive revisions.

## Limitations & Future Work
- RAGAs Factual Correctness relies on claim decomposition and a single reference answer, potentially underestimating pedagogically different but semantically correct responses; multiple instructor references and manual factuality labels should be added.
- Simulated students only utilize Meta-Llama-3.1-70B-Instruct, which may not represent real students' misconception patterns, motivation, or learning transfer.
- The expert review sample size is small (44 interactions). Although consistency is high, it is insufficient to prove real-world classroom learning gains.
- KITE is currently a system for intra-course algorithm learning; intent classification and feedback strategies may need reconfiguration for tasks like mathematical proofs, programming assignments, or medical education.
- The paper lacks pre- and post-tests with real students or longitudinal analysis; thus, "learning outcomes" can only be cautiously interpreted as improvements in simulated revision quality.

## Related Work & Insights
- **vs MoodleBot / Edison**: These systems emphasize course-grounded Q&A and TA-in-the-loop evaluation; KITE goes further by distinguishing student intent and providing scaffolded feedback for procedural/tracing tasks.
- **vs AutoTutor**: AutoTutor emphasizes conversational hints and collaborative revision but does not focus on course material RAG grounding; KITE combines Socratic tutoring with course-specific retrieval.
- **vs KG-RAG / LPITutor**: These systems use knowledge graphs or prompt strategies to improve educational Q&A; KITE focuses more on tracing, debugging, and validation within algorithmic reasoning tasks.
- **Insight**: Educational RAG benchmarks should include "promotion of second-attempt improvement" as a core metric rather than just answer correctness.

## Rating
- Novelty: ⭐⭐⭐⭐☆ System components are mostly existing technologies, but the combination of intent-aware tutoring and simulated student evaluation is well-executed.
- Experimental Thoroughness: ⭐⭐⭐☆☆ Parallel tracks of automated metrics, expert review, and simulated students are good, but real student experiments and module ablations are missing.
- Writing Quality: ⭐⭐⭐⭐☆ System design and evaluation workflows are clear, and limitations are discussed honestly.
- Value: ⭐⭐⭐⭐☆ High practical reference value for educational RAG, course TA systems, and algorithm learning feedback design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs](../../NeurIPS2025/information_retrieval/is_prm_necessary_problem-solving_rl_implicitly_induces_prm_capability_in_llms.md)
- [\[ICML 2026\] Position: Reliable AI Needs to Externalize Implicit Knowledge: A Human-AI Collaboration Perspective](../../ICML2026/information_retrieval/reliable_ai_needs_to_externalize_implicit_knowledge_a_human-ai_collaboration_per.md)
- [\[ACL 2026\] Utility-Oriented Visual Evidence Selection for Multimodal Retrieval-Augmented Generation](utility-oriented_visual_evidence_selection_for_multimodal_retrieval-augmented_ge.md)
- [\[ACL 2026\] S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA](s2g-rag_structured_sufficiency_and_gap_judging_for_iterative_retrieval-augmented.md)
- [\[ACL 2026\] MM-BizRAG: Rethinking Multimodal Retrieval-Augmented Generation for General Purpose Enterprise Q&A](mm-bizrag_rethinking_multimodal_retrieval-augmented_generation_for_general_purpo.md)

</div>

<!-- RELATED:END -->
