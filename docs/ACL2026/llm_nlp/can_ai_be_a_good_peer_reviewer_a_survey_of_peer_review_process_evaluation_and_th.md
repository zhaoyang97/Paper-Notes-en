---
title: >-
  [Paper Note] Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future
description: >-
  [ACL 2026][LLM (Other)][peer review] The authors systematically review methods for AI-assisted peer review throughout the entire pipeline in the LLM era: "review generation" is categorized into four paradigms (fine-tuning / agent / RL / enhancement), "after-review" is divided into three types (rebuttal / meta-review / paper revision), and a four-quadrant
tags:
  - ACL 2026
  - LLM (Other)
  - peer review
  - LLM agent
  - RL
  - AI4Research
date: 2026-05-08
content_hash: 6060695f8d8de2bf
---
# Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future

**Conference**: ACL 2026  
**arXiv**: [2604.27924](https://arxiv.org/abs/2604.27924)  
**Code**: https://github.com/formula12/Awesome-Peer-Review (Yes, paper list)  
**Area**: LLM NLP / Academic Writing / Survey  
**Keywords**: peer review, LLM agent, RL, evaluation, AI4Research

## TL;DR
The authors systematically review methods for AI-assisted peer review throughout the entire pipeline in the LLM era: "review generation" is categorized into four paradigms (fine-tuning / agent / RL / enhancement), "after-review" is divided into three types (rebuttal / meta-review / paper revision), and a four-quadrant evaluation taxonomy (human / reference-based / LLM-based / aspect-oriented) is provided. Finally, the future is discussed across six directions: novelty, automatic evaluation, cross-domain, multi-modality, and ethics.

## Background & Motivation

**Background**: Peer review has become one of the most active sub-fields in AI4Research—from the PeerRead dataset in 2018, to GPT-4 end-to-end review writing in 2023, and to the multi-agent panels (MARG, AgentReview, DeepReview) and RL-aligned methods (Remor, CycleResearcher, ReviewRL) from 2024 to 2026, the method spectrum has exploded within a few years.

**Limitations of Prior Work**: (1) Existing surveys either treat peer review as a small chapter of AI4Research with shallow coverage (e.g., Chen et al. 2025) or fail to keep up with the latest agent/RL wave (Zhuang et al. 2025); (2) The evaluation side is extremely fragmented—four systems (human / ROUGE / LLM-judge / aspect-level) do not interface with each other, lacking unified comparison; (3) "after-review" tasks (rebuttal, meta-review, revision) receive less attention compared to review generation and lack systematic organization; (4) Discussions on academic ethical risks (affiliation bias, AI-modified content estimated at $6.5-16.9\%$) are scattered.

**Key Challenge**: Peer review is a multi-stage, subjective, low-data task—it requires LLM acceleration while preventing the introduction of bias, pseudo-insights, or shallow critiques. The community lacks a roadmap that integrates the three axes of "method + evaluation + ethics."

**Goal**: (1) Provide an AI agent-oriented method classification for the full peer review process; (2) Systematize four types of evaluation methods and their pros/cons; (3) Summarize cross-era (pre-2023 vs post-2023) dataset differences; (4) List six clear future directions, particularly moving "beyond review generation itself."

**Key Insight**: Peer review is sliced along two axes—the vertical axis represents the process stages (review / rebuttal / meta-review / revision), and the horizontal axis represents method paradigms (foundation → fine-tuning → agent → RL → enhancement); evaluation is divided into four quadrants based on "whether to use a reference / whether to call an LLM judge."

**Core Idea**: A unified taxonomy (Figure 1) is used to weave 30+ systems and 20+ datasets into a searchable network, allowing researchers to quickly locate "what category I am working on and who to compare with."

## Method

### Overall Architecture
This survey weaves scattered literature on AI-assisted peer review into a searchable map using two clues: "Process Vertical Axis × Method Horizontal Axis" (Figure 1). The vertical axis represents task stages—from review generation to after-review (rebuttal / meta-review / paper revision); the horizontal axis represents the evolutionary lineage of method paradigms—foundation → fine-tuning → agent → RL → enhancement. On top of this, an evaluation axis is overlaid, dividing evaluation methods for 30+ systems and 20+ datasets into four quadrants based on "whether to use a reference / whether to call an LLM judge," eventually leading to six clear future directions. This allow readers to quickly determine "what category I am doing, who to compare with, and what metrics to use."

### Key Designs

**1. Five-Paradigm Classification of Review Generation: Converging "direct model-written reviews" into an evolutionary lineage**

Previous literature had messy cross-references, often confusing "agent" and "RL." The authors explicitly extract five paradigms where each step addresses the pain points of the previous one. Foundation approaches (pre-2023, PeerRead/NLPeer/MOPRD) perform sub-tasks like accept prediction, score regression, and multi-doc summary; Fine-tuning (OpenReviewer-Llama8B, REVIEWER2 two-stage, LimGen) targets "too positive zero-shot" and formatting issues. Agent-based approaches branch into two—task decomposition (MARG leader-worker / SWIF²T four agents / DeepReview three stages / MAMORX multi-modal / DIAGPaper weakness diagnosis) focuses on tool-based generation of better reviews, while process simulation (AgentReview / ReviewMT treating reviews as multi-turn dialogues) studies panel dynamics. The RL paradigm is subdivided by reward design (Remor uses GRPO + Human-aligned Peer Review Reward to solve shallow critiques; CycleResearcher uses review-as-reward for a research-review-refine loop; ReviewRL uses composite reward to optimize quality and consistency; REM-CTX uses auxiliary context for correspondence-aware reward). Enhancement is further split into RAG (ReviewRobot three KGs, novelty retrieval + re-ranking), iterative refinement (ReviewEval internal/external loops, RbtAct/GoodPoint/ActReview using rebuttal as supervision), and structure control (TreeReview dynamic question tree, AutoRev document graph, RevGAN style control). Once decomposed, it becomes clear which branch to choose for agents or which reward type for RL.

**2. Four Evaluation Method Categories + Pros/Cons Matrix: Resolving long-term confusion in evaluation metrics**

Past works either ran only one type of evaluation (rejected by another method category) or ran a full set without justifying the reasons. The authors provide a $4 \times 2$ pros/cons matrix (Table 2) for evidence-based selection. Human-centric (Robertson 2023 GPT-4 pilot, Liang 2023 $N=308$ user study, Reviewer Arena pairwise) is direct but expensive and subjective; Reference-based (ROUGE / BERTScore / hit rate / MCQ accuracy) is scalable but shallow, penalizing effective feedback with different wording; LLM-based (multi-judge ensemble, unsupervised judge) is scalable and flexible but carries position/verbosity bias and depends on the prompt; Aspect-oriented (ReviewCritique 23 types of fine-grained error annotation, STRICTA decomposing reviews into reasoning step graphs, focus-level evaluation, adversarial review injection) is fine-grained and diagnostic but has high annotation costs and multidimensional profiles that are difficult to compare. The value of the matrix lies in making decisions explicit, such as "use aspect-oriented to discover specific failure modes, use reference-based for fast ablation runs."

**3. Cross-era Dataset Comparison + Three After-Review Enhancement Mechanisms: Mapping "beyond review" tasks and data to specific sub-tasks**

The authors clearly state that after-review is more challenging than review generation (requiring handling of the dynamic game between argument and response), but datasets and benchmarks are far from sufficient. This section serves as a navigation for "which sub-task to perform." Rebuttal generation evolved from single-turn (Cheng 2020 APE argument pair extraction, Purkayastha JITSUPEER attitude-root) to multi-turn (ReviewMT, Re2 treating rebuttal as dialogue), to "verify-then-write" evidence frameworks (DRPG, Paper2Rebuttal), and recently adding author-in-the-loop signals (DEFEND). Meta-review evolved from MetaGen's extract-then-write to MReD adding sentence functional labels, to PeerSum introducing RAMMER for review-rebuttal hierarchy, to ORSUM cross-venue multi-stage introspection, and recently Purkayastha 2026 treating it as document-grounded dialogue. Paper revision from reviews, founded by ARIES + CASIMIR + arXivEdits, remains the most underexplored sub-direction. In conjunction with cross-era dataset comparisons (pre-2023 vs post-2023, Table 7), this section clearly marks the gaps in "task-data."

### Loss & Training
The survey itself does not conduct training, but representative training paradigms are organized for horizontal reference. The SFT route is represented by OpenReviewer (fine-tuning Llama-8B with 79k expert reviews) and REVIEWER2 (two stages: aspect prompt → review text); the RL route includes Remor (GRPO + multi-objective rewards: criticism / relevance / actionable suggestion), CycleResearcher (SimPO + dual-agent), ReviewRL (rule-based composite reward + retrieval augmentation); iterative refinement includes RbtAct (using rebuttal as implicit supervision) and ActReview (rubric-guided RL formalizing "review-to-revision").

## Key Experimental Results

### Main Results
Method comparison table summarized by the authors (Selected key systems, Table 1):

| Method | Paradigm | Dataset | Key Contribution |
| :--- | :--- | :--- | :--- |
| PeerRead (NAACL 2018) | Foundation | PeerRead | First large-scale paper+review dataset |
| OpenReviewer (NAACL 2025) | Fine-tuning | 79k expert reviews | Fine-tuned Llama-8B, captures various critique patterns |
| REVIEWER2 (2024) | Fine-tuning | 27k papers / 99k reviews | Two-stage aspect prompt → review |
| MARG | Agent (Dec.) | – | leader-worker, iterative refinement |
| DeepReview (ACL 2025) | Agent (Dec.) | DeepReview-13K | 3 stages: novelty / multi-dim / reliability |
| AgentReview (EMNLP 2024) | Agent (Sim.) | – | Panel simulation, studies authority bias |
| ReviewMT | Agent (Sim.) | ReviewMT | Rebuttal as dialogue, SFT > zero-shot |
| Remor | RL (GRPO) | PeerRT | Multi-objective HPRR reward |
| CycleResearcher (ICLR 2025)| RL (SimPO) | Review-5k + Research-14k | research-review-refine loop |
| ReviewRL (EMNLP 2025) | RL (rule) | ICLR 2025 papers | composite-reward + grounded |
| ReviewRobot (INLG 2020) | RAG (KG) | – | Three KGs: paper / cited / background |
| TreeReview (EMNLP 2025) | Structure | venue benchmark | Dynamic question tree |

### Key Findings
- **Clear trajectory of method evolution**: foundation (sub-task) → fine-tuning (full review) → agent (decomposed roles) → RL (alignment-aware) → enhancement (RAG + iterative + structure); each step addresses limitations of the previous (e.g., zero-shot too positive → fine-tune; fine-tune too linear → agent; agent lacks alignment → RL).
- **Evaluation bottleneck in LLM-as-judge bias**: position/verbosity bias makes a single-judge unreliable; at least a 2-judge majority vote is stable.
- **NLP/ML absolute dominance in datasets**: PeerRead, NLPeer, and ReviewMT all originate from NLP/ML conferences; MOPRD is a rare multi-disciplinary attempt, and cross-domain generalization remains insufficiently verified.
- **After-review is the most underexplored area**: datasets for rebuttal/meta-review/revision tasks are scarce, but their actual utility is higher (rebuttals help authors directly, revisions directly improve papers).
- **Ethical risks empirically exist**: Liang 2024 quantitative estimates + von Wedel 2024 evidence of bias indicate that the community can no longer avoid the question of "whether LLM-in-review is compliant."

## Highlights & Insights
- Explicitly splitting "agent-based" into task decomposition vs. process simulation is very clear—the former is instrumental (generating better reviews), the latter is research-oriented (understanding panel dynamics); previous literature often mixed these discussions.
- The "evaluation 4 quadrants" in Table 2 is the most reusable output of this survey—it provides more sustainable guidance than just listing methods; it is suggested that future papers cite this table in their evaluation sections to justify choices.
- Categorizing RL paradigms by reward design (multi-objective HPRR / review-as-reward / composite rule-based / context-aware) is also practical—researchers working on RL-based reviews can quickly identify their reward category and known pitfalls.
- The "Beyond Review Generation" (rebuttal/meta-review/revision) directions in §5 are explicitly labeled as "underdeveloped"—this explicit prioritization is rare in surveys and aids resource allocation.
- Separately listing works on "novelty assessment" (SchNovel / NovBench / OpenReviewer / Reviewer2) as future direction 1 highlights a continuous gap: "LLMs can write fluent reviews but cannot accurately judge true novelty."

## Limitations & Future Work
- The survey was written in early 2026, but the pace of peer review publication is extremely fast (new papers monthly); the taxonomy requires continuous maintenance (as admitted in the limitations).
- Most discussions focus on NLP/ML conference data, with thin coverage of differences in biomedical/natural science reviews.
- Commercial review systems (Elsevier AI, Editor Assistant, etc.) are not included in the comparison—the capabilities of proprietary systems are "dark matter" compared to open-source academic work.
- The pros/cons matrix (Table 2) is qualitative, lacking a unified quantitative comparison table across systems (e.g., reporting ROUGE/judge/aspect scores for the same set of papers); this is both a hint and a drawback for future benchmark construction.
- Improvement ideas: (1) Build a unified leaderboard across four evaluation categories; (2) Evaluate review, rebuttal, meta-review, and revision as an end-to-end pipeline rather than isolated sub-tasks; (3) Systematically test the gap between commercial and open-source systems; (4) Cross-domain (medicine, chemistry) peer review benchmarks.

## Related Work & Insights
- **vs Chen et al. 2025 (AI4Research survey)**: That work has broader coverage (research idea → writing → review → analysis), with peer review as one chapter; ours is specialized and dives into recent agent/RL work, making it a better entry point for the reviewing sub-field.
- **vs Zhuang et al. 2025**: Also reviews LLM-for-peer-review but lacks systematic RL and evaluation integration; ours fills these gaps on both axes.
- **vs Drozdz & Ladomery 2024 (BJBS)**: That work takes a medical perspective, focusing on policy and ethics; ours takes an NLP perspective, focusing on methods and datasets, making them complementary.
- **Insights**: (1) This "Process × Method Paradigm" dual-axis taxonomy can be migrated to other AI4Science sub-fields (e.g., AI-for-Literature-Review); (2) The "evaluation 4 quadrants" template can be generalized to evaluation surveys for any long-form generation; (3) Explicitly labeling "underexplored" sub-tasks is a useful strategy for survey authors—it guides community attention better than plain narration.

## Rating
- **Novelty**: ⭐⭐⭐ The survey itself does not pursue method novelty; the taxonomy (5 paradigms + 4 evaluations + process axis) is combinatorically novel; the evaluation 4-quadrant table is the most original output.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 30+ systems + 20+ datasets over eight years (2018-2026); the appendix provides backbone tables, parsing tool tables, and dataset timeline tables, which are highly helpful for comparison.
- **Writing Quality**: ⭐⭐⭐⭐ Chapters are clearly organized, with each paradigm featuring brief system introductions and key contributions; Figure 1 taxonomy tree and Figures 2/3 flowcharts are intuitive; marking agent types as "Dec./Sim." saves reading time.
- **Value**: ⭐⭐⭐⭐ For researchers just entering AI-for-peer-review, it is the most practical entry point; the publicly available Awesome repository allows for sustainable maintenance; the six future directions can directly be converted into motivations for future papers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Stop Automating Peer Review Without Rigorous Evaluation](../../ICML2026/llm_nlp/stop_automating_peer_review_without_rigorous_evaluation.md)
- [\[AAAI 2026\] Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback](../../AAAI2026/llm_nlp/position_on_llm-assisted_peer_review_addressing_reviewer_gap_through_mentoring_a.md)
- [\[ACL 2026\] Big AI is Accelerating the Metacrisis: What Can We Do?](big_ai_is_accelerating_the_metacrisis_what_can_we_do.md)
- [\[ACL 2026\] From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?](from_fallback_to_frontline_when_can_llms_be_superior_annotators_of_human_perspec.md)
- [\[ACL 2026\] SteerEval: How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities](how_controllable_are_large_language_models_a_unified_evaluation_across_behaviora.md)

</div>

<!-- RELATED:END -->
