---
title: >-
  [Paper Note] Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future
description: >-
  [ACL 2026][LLM (Other)][peer review] The authors provide a systematic survey of AI-assisted peer review methods in the LLM era. They categorize "review generation" into four paradigms: fine-tuning / agent / RL / generation enhancement, classify "after-review" into rebuttal / meta-review / paper revision, and present a four-quadrant evaluation taxonomy (hu
tags:
  - ACL 2026
  - LLM (Other)
  - peer review
  - LLM agent
  - RL
  - AI4Research
date: 2026-05-08
content_hash: c8d4765db2858b90
---
# Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future

**Conference**: ACL 2026  
**arXiv**: [2604.27924](https://arxiv.org/abs/2604.27924)  
**Code**: https://github.com/formula12/Awesome-Peer-Review (Available, Paper List)  
**Area**: LLM NLP / Academic Writing / Survey  
**Keywords**: peer review, LLM agent, RL, evaluation, AI4Research

## TL;DR
The authors provide a systematic survey of AI-assisted peer review methods in the LLM era. They categorize "review generation" into four paradigms: fine-tuning / agent / RL / generation enhancement, classify "after-review" into rebuttal / meta-review / paper revision, and present a four-quadrant evaluation taxonomy (human / reference-based / LLM-based / aspect-oriented). Finally, they discuss the future across six directions: novelty, automatic evaluation, cross-domain, multimodality, and ethics.

## Background & Motivation

**Background**: Peer review has become one of the most active sub-fields in AI4Research. The field has seen an explosion of methods over the years, starting from the 2018 PeerRead dataset, progressing to GPT-4 generating end-to-end reviews in 2023, and moving toward multi-agent panels (MARG, AgentReview, DeepReview) and RL-aligned methods (Remor, CycleResearcher, ReviewRL) from 2024 to 2026.

**Limitations of Prior Work**: (1) Existing surveys either treat peer review as a minor chapter of AI4Research with shallow coverage (e.g., Chen et al. 2025) or fail to keep up with the latest agent/RL trends (Zhuang et al. 2025); (2) Evaluation is highly fragmented across human, ROUGE, LLM-judge, and aspect-level systems without unified comparison; (3) "after-review" tasks (rebuttal, meta-review, revision) receive less focus than review generation and lack systematic organization; (4) Discussions on academic ethical risks (affiliation bias, AI-modified content estimated at $6.5-16.9\%$) are scattered.

**Key Challenge**: Peer review is a multi-stage, subjective, low-data task. The challenge lies in utilizing LLMs for speed while preventing them from introducing biases, hallucinations, or shallow critiques. The community lacks a roadmap connecting "Method + Evaluation + Ethics."

**Goal**: (1) Provide an AI agent-centric taxonomy for the entire peer review process; (2) Systematize four types of evaluation methods with their pros and cons; (3) Summarize differences in datasets across eras (pre-2023 vs. post-2023); (4) Outline six clear future directions, specifically focusing on going "beyond review generation."

**Key Insight**: Peer review is mapped along two axes: the vertical axis represents process stages (review / rebuttal / meta-review / revision), and the horizontal axis represents methodological paradigms (foundation → fine-tuning → agent → RL → enhancement). Evaluation is categorized into four quadrants based on "use of reference" and "use of LLM judge."

**Core Idea**: A unified taxonomy (Figure 1) is used to organize over 30 systems and 20 datasets into a searchable network, enabling researchers to quickly identify categories and baselines.

## Method

### Overall Architecture
This survey organizes the scattered literature on AI-assisted peer review using two dimensions: "process vertical axis × method horizontal axis" (Figure 1). Vertically, it covers task stages from review generation to after-review (rebuttal / meta-review / paper revision). Horizontally, it tracks the evolution of paradigms: foundation → fine-tuning → agent → RL → enhancement. Additionally, an evaluation axis divides the evaluation methods of 30+ systems and 20+ datasets into four quadrants based on the presence of references and LLM judges, concluding with six future directions.

### Key Designs

**1. Five Paradigms of Review Generation: Converging "direct generation" into an evolutionary lineage**

Addressing the prior confusion between "agent" and "RL" approaches, the authors explicitly define five paradigms where each step resolves limitations of its predecessor. Foundation approaches (pre-2023, PeerRead/NLPeer/MOPRD) focus on sub-tasks like acceptance prediction, score regression, and multi-document summarization. Fine-tuning (OpenReviewer-Llama8B, two-stage REVIEWER2, LimGen) addresses "overly positive zero-shot" results and formatting issues. Agent-based methods branch into two directions: task decomposition (MARG leader-worker, SWIF²T four agents, DeepReview three stages, MAMORX multimodal, DIAGPaper weakness diagnosis) for creating better reviews as tools, and process simulation (AgentReview / ReviewMT treating review as multi-turn dialogue) for studying panel dynamics. RL paradigms are categorized by reward design (Remor using GRPO + Human-aligned Peer Review Reward for shallow critiques, CycleResearcher using review-as-reward for a research-review-refine loop, ReviewRL using composite rewards for quality and consistency, REM-CTX using auxiliary context for correspondence-aware rewards). Enhancement includes RAG (ReviewRobot three KGs, novelty retrieval + reranking), iterative refinement (ReviewEval cycles, RbtAct/GoodPoint/ActReview using rebuttal as supervision), and structure control (TreeReview dynamic trees, AutoRev graphs, RevGAN style control).

**2. Four Evaluation Categories + Pros/Cons Matrix: Resolving long-term confusion in evaluation metrics**

The authors provide a $4\times2$ pros/cons matrix (Table 2) to justify evaluation choices. Human-centric (Robertson 2023 GPT-4 pilot, Liang 2023 $N=308$ user study, Reviewer Arena) is direct but expensive and subjective; Reference-based (ROUGE / BERTScore / hit rate / MCQ accuracy) is scalable but shallow, penalizing valid feedback that uses different wording; LLM-based (multi-judge ensemble, unsupervised judge) is scalable and flexible but suffers from position/verbosity bias and prompt dependency; Aspect-oriented (ReviewCritique 23 fine-grained error types, STRICTA step graphs, focus-level assessment, adversarial injection) is detailed for diagnostics but has high annotation costs.

**3. Cross-era Dataset Comparison + Three "After-review" Mechanisms**

The authors highlight that after-review tasks are more challenging due to the dynamic game between arguments and responses. Dataset navigation is provided for these sub-tasks. Rebuttal generation evolved from single-turn (Cheng 2020 APE argument pairs, Purkayastha JITSUPEER) to multi-turn (ReviewMT, Re2) and "verify-then-write" evidence frameworks (DRPG, Paper2Rebuttal), recently incorporating author-in-the-loop signals (DEFEND). Meta-review evolved from MetaGen's extract-then-write to sentence labeling in MReD, RAMMER's hierarchy in PeerSum, multi-stage introspection in ORSUM, and document-grounded dialogue in Purkayastha 2026. Paper revision from reviews, founded by ARIES, CASIMIR, and arXivEdits, remains the most underexplored sub-direction.

### Loss & Training
While the survey does not perform training, it organizes representative training paradigms for reference. The SFT route is exemplified by OpenReviewer (fine-tuning Llama-8B on 79k expert reviews) and REVIEWER2 (two-stage: aspect prompt → review text). The RL route includes Remor (GRPO + multi-objective rewards: criticism / relevance / actionable suggestion), CycleResearcher (SimPO + dual-agent), ReviewRL (rule-based composite reward + retrieval augmentation), and iterative refinement like RbtAct (rebuttal as implicit supervision) and ActReview (rubric-guided RL formalizing review-to-revision).

## Key Experimental Results

### Main Results
Method comparison table (selected systems, Table 1):

| Method | Paradigm | Dataset | Key Contribution |
|------|------|------|------|
| PeerRead (NAACL 2018) | Foundation | PeerRead | First large-scale paper+review dataset |
| OpenReviewer (NAACL 2025) | Fine-tuning | 79k expert reviews | Llama-8B SFT, captures critique patterns |
| REVIEWER2 (2024) | Fine-tuning | 27k papers / 99k reviews | Two-stage aspect prompt → review |
| MARG | Agent (Dec.) | – | Leader-worker, iterative refinement |
| DeepReview (ACL 2025) | Agent (Dec.) | DeepReview-13K | 3 stages: novelty / multi-dim / reliability |
| AgentReview (EMNLP 2024) | Agent (Sim.) | – | Simulates panels, studies authority bias |
| ReviewMT | Agent (Sim.) | ReviewMT | Rebuttal as dialogue, SFT > zero-shot |
| Remor | RL (GRPO) | PeerRT | Multi-objective HPRR reward |
| CycleResearcher (ICLR 2025) | RL (SimPO) | Review-5k + Research-14k | Research-review-refine closed loop |
| ReviewRL (EMNLP 2025) | RL (Rule) | ICLR 2025 papers | Composite-reward + grounded |
| ReviewRobot (INLG 2020) | RAG (KG) | – | Three KGs: paper / cited / background |
| TreeReview (EMNLP 2025) | Structure | Venue benchmark | Dynamic question tree |

### Key Findings
- Robertson 2023 pilot: GPT-4 reviews align with humans in helpfulness but have higher variance.
- Liang 2023 $N=308$: Users perceive LLM feedback as "valuable."
- Liang 2024 corpus-level: $6.5\%-16.9\%$ of reviews in top AI conferences may be significantly modified by LLMs, with surges reaching deadlines.
- von Wedel 2024: LLMs exhibit preference for authors from high-ranking institutions (affiliation bias) in single-blind settings.
- ReviewCritique: Identifies 23 fine-grained error types such as "Unstated Statement" and "Missing Reference."

### Key Findings
- **Clear Methodological Evolution**: Foundation (sub-task) → Fine-tuning (full review) → Agent (decomposed roles) → RL (alignment-aware) → Enhancement (RAG + iterative + structure); each step addresses limitations of the previous stage.
- **Evaluation Bottleneck**: LLM-as-judge bias (position/verbosity) makes single-judge systems unreliable; majority voting with at least two judges is required for stability.
- **NLP/ML Domain Dominance**: PeerRead, NLPeer, and ReviewMT are primarily derived from NLP/ML conferences; cross-domain generalization (e.g., MOPRD) remains under-verified.
- **After-review is significantly underexplored**: Rebuttal, meta-review, and revision tasks suffer from dataset scarcity despite their high actual utility for authors.
- **Ethical Risks are Quantified**: Evidence of bias (von Wedel 2024) and large-scale LLM inclusion (Liang 2024) forces the community to address "LLM-in-review" compliance.

## Highlights & Insights
- The explicit split of "agent-based" into task decomposition vs. process simulation is highly effective for distinguishing between tool-based review generation and research on panel dynamics.
- The "evaluation 4 quadrants" (Table 2) is a high-value output that provides structured guidance for future researchers to justify their metric choices.
- Categorizing RL paradigms by reward design (multi-objective HPRR / review-as-reward / composite rule-based / context-aware) helps developers quickly identify known pitfalls.
- Identifying "Beyond Review Generation" tasks as "underdeveloped" provides explicit prioritization for future research resource allocation.
- Separating "novelty assessment" (SchNovel, NovBench) as a distinct future direction highlights the gap where LLMs can write fluent reviews but struggle to judge true scientific novelty.

## Limitations & Future Work
- The survey reflects information up to early 2026; the rapid pace of publication necessitates continuous maintenance of the taxonomy.
- Coverage of reviews in biomedical and natural sciences is limited, as the majority of focus remains on NLP/ML conferences.
- Commercial review systems (e.g., Elsevier AI) are not included in the comparison, making their relative performance "dark matter" to the academic community.
- The evaluation pros/cons matrix is qualitative; there is a lack of unified quantitative comparisons across all systems using identical benchmarks.
- Future directions: (1) Establishing a unified leaderboard; (2) Evaluating review pipeline as an end-to-end process; (3) Systematizing the gap between commercial and open-source systems; (4) Cross-domain benchmarks.

## Related Work & Insights
- **vs. Chen et al. 2025 (AI4Research survey)**: While that paper covers the entire research lifecycle, this survey provides deeper specialization into agent/RL methodologies for reviewing.
- **vs. Zhuang et al. 2025**: This survey supplements previous ones by adding comprehensive RL sections and an evaluation matrix.
- **vs. Drozdz & Ladomery 2024 (BJBS)**: Complements medical-perspective surveys by providing an NLP-focused methodological and dataset-centric view.
- **Insights**: (1) The "Process × Paradigm" dual-axis taxonomy can be adapted for other AI4Science domains; (2) The "evaluation 4 quadrants" template is applicable to any long-form generation survey; (3) Explicitly labeling "underexplored" sub-tasks effectively guides community attention.

## Rating
- Novelty: ⭐⭐⭐ The taxonomy (5 paradigms + 4 quadrants + vertical stages) provides combinatorial novelty; the evaluation matrix is the most original contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 30+ systems and 20+ datasets over 8 years; appendices provide detailed backbone, tool, and timeline tables.
- Writing Quality: ⭐⭐⭐⭐ Clear organization; each paradigm includes system profiles and key contributions; intuitive taxonomy and flow diagrams.
- Value: ⭐⭐⭐⭐ Provides an excellent entry point for new researchers and a sustainable Awesome repository; future directions are well-motivated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Stop Automating Peer Review Without Rigorous Evaluation](../../ICML2026/llm_nlp/stop_automating_peer_review_without_rigorous_evaluation.md)
- [\[AAAI 2026\] Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback](../../AAAI2026/llm_nlp/position_on_llm-assisted_peer_review_addressing_reviewer_gap_through_mentoring_a.md)
- [\[ICML 2026\] Position: The ML Community Must Build an AI-Augmented Peer-Review Ecosystem](../../ICML2026/llm_nlp/position_the_ml_community_must_build_an_ai-augmented_peer-review_ecosystem.md)
- [\[ACL 2026\] Big AI is Accelerating the Metacrisis: What Can We Do?](big_ai_is_accelerating_the_metacrisis_what_can_we_do.md)
- [\[ACL 2026\] From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?](from_fallback_to_frontline_when_can_llms_be_superior_annotators_of_human_perspec.md)

</div>

<!-- RELATED:END -->
