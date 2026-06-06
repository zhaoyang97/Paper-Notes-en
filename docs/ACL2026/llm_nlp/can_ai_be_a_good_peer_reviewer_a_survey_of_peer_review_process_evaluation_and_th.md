---
title: >-
  [Paper Note] Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future
description: >-
  [ACL 2026][LLM/NLP][peer review] The authors systematically review methods for AI-assisted peer review throughout the entire process in the LLM era. They categorize "review generation" into four paradigms (fine-tuning…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "peer review"
  - "LLM agent"
  - "RL"
  - "evaluation"
  - "AI4Research"
date: 2026-05-08
content_hash: 5d86407593e85d61
---

# Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future

**Conference**: ACL 2026  
**arXiv**: [2604.27924](https://arxiv.org/abs/2604.27924)  
**Code**: https://github.com/formula12/Awesome-Peer-Review (Yes, paper list)  
**Area**: LLM NLP / Academic Writing / Survey  
**Keywords**: peer review, LLM agent, RL, evaluation, AI4Research

## TL;DR
The authors systematically review methods for AI-assisted peer review throughout the entire process in the LLM era. They categorize "review generation" into four paradigms (fine-tuning, agent, RL, and generation enhancement), classify "after-review" tasks into three types (rebuttal, meta-review, and paper revision), and provide a four-quadrant evaluation taxonomy (human, reference-based, LLM-based, and aspect-oriented). Finally, they discuss the future across six directions: novelty, automatic evaluation, cross-domain applications, multimodality, and ethics.

## Background & Motivation

**Background**: Peer review has become one of the most active sub-fields in AI4Research. Starting from the 2018 PeerRead dataset, continuing to end-to-end full review generation by GPT-4 in 2023, and reaching the multi-agent panels (MARG, AgentReview, DeepReview) and RL-aligned methods (Remor, CycleResearcher, ReviewRL) of 2024–2026, the methodological spectrum has exploded within a few years.

**Limitations of Prior Work**: (1) Existing surveys either treat peer review as a minor section of AI4Research (covering it too shallowly, e.g., Chen et al. 2025) or fail to keep up with the latest wave of agent/RL trends (Zhuang et al. 2025); (2) Evaluation methodologies are extremely fragmented—the four systems of human / ROUGE / LLM-judge / aspect-level are disconnected and lack unified comparison; (3) After-review tasks (rebuttal, meta-review, revision) receive significantly less attention than review generation and lack systematic organization; (4) Discussions on academic ethical risks (affiliation bias, AI-modified content estimated at 6.5–16.9%) are scattered.

**Key Challenge**: Peer review is a multi-stage, subjective, low-data task. It requires LLMs to accelerate the process while preventing them from introducing bias, hallucinations (pseudo-insights), or shallow critiques. The community lacks a roadmap that integrates methods, evaluation, and ethics.

**Goal**: (1) Provide an AI agent-oriented classification of the full-process peer review methodology; (2) Systematize 4 types of evaluation methods with their pros/cons; (3) Summarize dataset differences across different eras (pre-2023 vs. post-2023); (4) Outline 6 clear future directions, particularly focusing on "beyond review generation itself."

**Key Insight**: The authors divide peer review along two axes—the vertical axis represents process stages (review / rebuttal / meta-review / revision), and the horizontal axis represents methodological paradigms (foundation → fine-tuning → agent → RL → enhancement). Evaluation is categorized into four quadrants based on "whether references are used" and "whether an LLM judge is called."

**Core Idea**: Use a unified taxonomy (Figure 1) to weave over 30 systems and 20+ datasets into a searchable network, enabling researchers to quickly identify which category their work belongs to and whom to compare against.

## Method

### Overall Architecture
The survey is organized into 4 main chapters:
1. **Peer Review Generation** (§2): Five paradigms — foundation approaches → fine-tuning → agent-based (task decomposition + process simulation) → RL → review generation enhancement (RAG / iterative refinement / structure & style control).
2. **After Peer Review** (§3): Rebuttal generation / meta-review / paper revision from reviews.
3. **Benchmark Perspective** (§4): Four types of evaluation methods + chronological comparison of datasets.
4. **Discussion & Future Directions** (§5): Six open problems.

Each chapter is accompanied by a comprehensive table (Table 1 for method comparison, Table 2 for evaluation pros/cons, Table 7 for dataset eras).

### Key Designs

1. **Classification of the 5 Paradigms for Review Generation**:
    - **Function**: Converges all "direct model-generated review" methods into an evolutionary lineage.
    - **Mechanism**: (a) Foundation approaches (pre-2023) focused on sub-tasks (acceptance prediction, score regression, multi-doc summary), represented by datasets like PeerRead/NLPeer/MOPRD; (b) Fine-tuning (OpenReviewer-Llama8B, two-stage REVIEWER2, LimGen) addresses "overly positive zero-shot" and formatting issues; (c) Agent-based splits into two branches: task decomposition (MARG leader-worker / SWIF²T four-agent / DeepReview three-stage / MAMORX multimodal / DIAGPaper weakness diagnosis) and process simulation (AgentReview / ReviewMT treating review as multi-turn dialogue); (d) RL (Remor using GRPO + Human-aligned Peer Review Reward to solve "shallow critique," CycleResearcher making review-as-reward close the research-review-refine loop, ReviewRL using composite reward for quality and consistency, REM-CTX using auxiliary context for correspondence-aware reward); (e) Enhancement includes RAG (ReviewRobot with three KGs, Novelty retrieval + reranking), iterative refinement (ReviewEval internal/external loops, RbtAct/GoodPoint/ActReview using rebuttal as supervision), and structure control (TreeReview dynamic question trees, AutoRev document graphs, RevGAN style control).
    - **Design Motivation**: Previous literature often confused "agent" and "RL" paradigms. By explicitly separating them, researchers can choose between "task decomposition/simulation" for agents or "reward types" for RL.

2. **4 Evaluation Categories + Pros/Cons Matrix**:
    - **Function**: Resolves the long-standing confusion over which metrics to use for review systems.
    - **Mechanism**: (a) Human-centric (Robertson 2023 GPT-4 pilot, Liang 2023 $N=308$ user study, Reviewer Arena pairwise)—most direct but expensive and subjective; (b) Reference-based (ROUGE / BERTScore / hit rate / MCQ accuracy)—scalable but shallow, penalizing valid feedback that differs in phrasing; (c) LLM-based (multi-judge ensemble, unsupervised judge)—scalable and flexible but susceptible to position/verbosity bias and prompt-dependent; (d) Aspect-oriented (ReviewCritique 23 types of fine-grained error labels, STRICTA splitting reviews into reasoning step graphs, focus-level evaluation, adversarial review injection)—fine-grained and diagnostic but with high annotation costs. Table 2 provides a $4 \times 2$ pros/cons matrix for selection.
    - **Design Motivation**: Earlier works often used only one evaluation type. This matrix allows researchers to justify their choice (e.g., aspect-oriented for failure mode discovery vs. reference-based for fast ablation).

3. **Comparison of Datasets Across Eras + Three Enhancement Mechanisms (After-Review)**:
    - **Function**: Maps "beyond-review" tasks and data to specific sub-tasks.
    - **Mechanism**: (a) Rebuttal generation evolved from single-turn (Cheng 2020 APE argument pair extraction) to multi-turn (ReviewMT, Re2) to "verify-then-write" evidence frameworks (DRPG, Paper2Rebuttal), recently including author-in-the-loop signals (DEFEND). (b) Meta-review progressed from MetaGen extract-then-write to MReD sentence functional labels, PeerSum hierarchical handling, and ORSUM multi-stage venue introspection. (c) Paper revision from reviews, established by ARIES, CASIMIR, and arXivEdits, remains the most underexplored sub-area.
    - **Design Motivation**: The authors highlight that "after-review" tasks are more challenging due to the dynamic nature of arguments/responses, providing a navigation guide for future work.

### Loss & Training
(The survey summarizes training paradigms mentioned in the literature)
- **SFT**: OpenReviewer utilized 79k expert reviews on Llama-8B; REVIEWER2 employed a two-stage process (aspect prompt → review text).
- **RL**: Remor used GRPO + multi-objective rewards (criticism + relevance + actionable suggestion); CycleResearcher used SimPO + dual-agent; ReviewRL used rule-based composite rewards + retrieval augmentation.
- **Iterative Refinement**: RbtAct used rebuttal as implicit supervision; ActReview used rubric-guided RL to formulate "review-to-revision."

## Key Experimental Results

### Main Results
The authors summarized a comparison table of methods (key systems only, Table 1):

| Method | Paradigm | Dataset | Key Contribution |
| :--- | :--- | :--- | :--- |
| PeerRead (NAACL 2018) | Foundation | PeerRead | First large-scale paper+review dataset |
| OpenReviewer (NAACL 2025) | Fine-tuning | 79k expert reviews | Llama-8B fine-tuning, captures critique patterns |
| REVIEWER2 (2024) | Fine-tuning | 27k papers / 99k reviews | Two-stage aspect prompt → review |
| MARG | Agent (Dec.) | – | Leader-worker, iterative refinement |
| DeepReview (ACL 2025) | Agent (Dec.) | DeepReview-13K | 3 stages: novelty / multi-dim / reliability |
| AgentReview (EMNLP 2024) | Agent (Sim.) | – | Simulating panels, studying authority bias |
| ReviewMT | Agent (Sim.) | ReviewMT | Rebuttal as dialogue, SFT > zero-shot |
| Remor | RL (GRPO) | PeerRT | Multi-objective HPRR reward |
| CycleResearcher (ICLR 2025) | RL (SimPO) | Review-5k + Research-14k | Research-review-refine closed loop |
| ReviewRL (EMNLP 2025) | RL (rule) | ICLR 2025 papers | Composite-reward + grounded |
| ReviewRobot (INLG 2020) | RAG (KG) | – | Three KGs: paper / cited / background |
| TreeReview (EMNLP 2025) | Structure | Venue benchmark | Dynamic question tree |

### Key Findings
- **Clear Evolutionary Trajectory**: Foundation (sub-task) → fine-tuning (full review) → agent (decomposed roles) → RL (alignment-aware) → enhancement (RAG + iterative + structure). Each step addresses the previous step's limitations (e.g., zero-shot being too positive → fine-tune).
- **Evaluation Bottleneck in LLM-as-judge**: Position/verbosity bias makes single-judge unreliable; at least two-judge majority voting is required for stability.
- **NLP/ML Dominance in Datasets**: PeerRead, NLPeer, and ReviewMT are all from NLP/ML conferences. MOPRD is a rare multidisciplinary attempt, but cross-domain generalization is not yet fully verified.
- **After-review is Significantly Underexplored**: Rebuttal/meta-review/revision tasks have few datasets despite having higher actual utility for authors.
- **Ethical Risks are Empirically Present**: Quantitative estimates by Liang 2024 and evidence of bias by von Wedel 2024 show that the community can no longer avoid the issue of LLM compliance in reviewing.

## Highlights & Insights
- Explicitly splitting "agent-based" into task decomposition vs. process simulation is highly effective—the former is a tool (generating better reviews) while the latter is research-oriented (understanding panel dynamics).
- The "evaluation 4 quadrants" in Table 2 is the most reusable output of this survey, providing better long-term guidance than just listing methods.
- Categorizing RL paradigms by reward design (multi-objective HPRR / review-as-reward / composite rule-based) allows practitioners to quickly identify known pitfalls.
- Identifying "Beyond Review Generation" (rebuttal/meta-review/revision) as "underdeveloped" is a rare and helpful prioritization in a survey.
- Explicitly listing work on "novelty assessment" (SchNovel, NovBench, etc.) as a future direction highlights the gap where LLMs can write fluent reviews but struggle to judge true novelty.

## Limitations & Future Work
- The survey was written in early 2026, but the rapid publication pace of peer review research means the taxonomy requires continuous maintenance.
- Discussion focus is heavily centered on NLP/ML conference data, with thin coverage of differences in biomedical or natural science reviews.
- Commercial review systems (Elsevier AI, Editor Assistant, etc.) were not included in the comparison; the capabilities of proprietary systems remain "dark matter" to open-source research.
- The pros/cons matrix for evaluation (Table 2) is qualitative and lacks a unified quantitative comparison across systems.
- Improvement ideas: (1) Build a unified leaderboard running all 4 types of evaluation; (2) Treat review, rebuttal, meta-review, and revision as an end-to-end pipeline rather than isolated tasks; (3) Systematically test the gap between commercial and open-source systems; (4) Develop cross-domain (medicine, chemistry) peer review benchmarks.

## Related Work & Insights
- **vs. Chen et al. 2025 (AI4Research survey)**: That paper has broader coverage (research idea → writing → review → analysis); this paper is specialized and deep-dives into recent agent/RL work, making it a better entry point for the reviewing sub-field.
- **vs. Zhuang et al. 2025**: Also a survey of LLM-for-peer-review but lacks systematic RL and evaluation frameworks; this paper fills those gaps.
- **Inspiration**: (1) This "process × paradigm" dual-axis classification can be migrated to other AI4Science sub-fields; (2) The "evaluation 4 quadrants" template is applicable to any survey of long-form generation; (3) Explicitly labeling "underexplored" sub-tasks is a useful strategy for guiding community attention.

## Rating
- Novelty: ⭐⭐⭐ The survey achieves combinatorial novelty through its taxonomy (5 paradigms + 4 evaluations + process axis).
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 30+ systems and 20+ datasets over eight years (2018-2026), including backbone, tool, and timeline tables in the appendix.
- Writing Quality: ⭐⭐⭐⭐ Extremely clear organization; uses "Dec./Sim." markers for agent types to save reader time.
- Value: ⭐⭐⭐⭐ A practical entry point for newcomers and provides motivation for future work through 6 explicit directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback](../../AAAI2026/llm_nlp/position_on_llm-assisted_peer_review_addressing_reviewer_gap_through_mentoring_a.md)
- [\[ICML 2026\] Stop Automating Peer Review Without Rigorous Evaluation](../../ICML2026/llm_nlp/stop_automating_peer_review_without_rigorous_evaluation.md)
- [\[ACL 2026\] Big AI is Accelerating the Metacrisis: What Can We Do?](big_ai_is_accelerating_the_metacrisis_what_can_we_do.md)
- [\[ACL 2026\] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)
- [\[ACL 2026\] From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?](from_fallback_to_frontline_when_can_llms_be_superior_annotators_of_human_perspec.md)

</div>

<!-- RELATED:END -->
