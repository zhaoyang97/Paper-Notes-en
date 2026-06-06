---
title: >-
  [Paper Note] Evaluating Legal Reasoning Traces with Legal Issue Tree Rubrics
description: >-
  [ACL 2026][LLM Evaluation][legal judgment prediction] LEGIT automatically extracts "hierarchical legal issue trees" from Korean civil and administrative judgments as rubrics. This enables the LLM-as-judge to evaluate bot…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "legal judgment prediction"
  - "issue tree"
  - "rubric-based judge"
  - "RAG"
  - "GRPO"
date: 2026-05-08
content_hash: f43b48c716dacb35
---

# Evaluating Legal Reasoning Traces with Legal Issue Tree Rubrics

**Conference**: ACL 2026  
**arXiv**: [2512.01020](https://arxiv.org/abs/2512.01020)  
**Code**: Proclaimed open source (GitHub repo)  
**Area**: LLM Evaluation / Law / Reasoning Chain Evaluation  
**Keywords**: legal judgment prediction, issue tree, rubric-based judge, RAG, GRPO

## TL;DR
LEGIT automatically extracts "hierarchical legal issue trees" from Korean civil and administrative judgments as rubrics. This enables the LLM-as-judge to evaluate both "issue coverage" and "issue correctness," revealing that RAG and RL have complementary effects: RAG provides global improvements, while RL sacrifices coverage for increased correctness.

## Background & Motivation

**Background**: LLM-as-a-judge has become the mainstream for evaluating chain-of-thought reasoning, but most benchmarks focus solely on final answer accuracy (math, code) or use coarse-grained Likert scales. In expert tasks like Legal Judgment Prediction (LJP), relying on final verdict accuracy obscures two dimensions critical to legal practice: whether the reasoning chain sufficiently covers key legal issues and whether correct conclusions are reached for each issue.

**Limitations of Prior Work**: (1) Manual rubric approaches like BigGen-Bench cannot scale due to the cost of expert labor; (2) Likert scale consistency across evaluators is poor and unstable; (3) Existing LJP datasets (CMDL, ECHR, Hwang2022, etc.) primarily cover criminal or binary judgments, leaving a void in civil and administrative cases which constitute 70-84% of court cases; (4) Even with correct final orders, instances of missing key issues or incorrect sub-conclusions remain hidden.

**Key Challenge**: Legal reasoning is naturally a **tree**—parent issues are implied by sub-issues, which are derived from facts and common sense. Compressing this into "single-label classification + scalar Likert scores" destroys this structure, preventing evaluators from distinguishing between "decomposition errors" and "deduction errors" and precluding the use of dense signals for RL rewards.

**Goal**: (1) Automatically extract large-scale, expert-level legal issue tree rubrics from judgments; (2) Verify the alignment between rubric-based LLM judges and licensed attorneys; (3) Use the LEGIT system to characterize failure modes (decomposition vs. deduction) of SOTA LLMs in complex civil/commercial cases; (4) Employ rubrics directly as RL rewards to observe how RAG and RL impact different parts of the issue tree.

**Key Insight**: Legal reasoning can be formalized as top-down "back-chaining" along an issue tree. Extracting every node ("parties' claims + court conclusion") from a judgment creates a natural rubric. Since judgments are abundant natural annotations, 24K instances are generated via LLM extraction followed by two-round refinement and difficulty categorization.

**Core Idea**: Utilize legal issue trees as "naturally scalable, expert-aligned" rubrics. Single-score evaluation is decomposed into {final order, issue coverage, issue correctness} components (weighted 5/2/3), providing both robust scoring and direct RL rewards.

## Method

LEGIT task input: Facts of the case + purpose of claim; Output: Free-form reasoning chain + final judgment. The evaluator scores the reasoning chain node-by-node against the issue tree, summing to a LEGIT score of 0–10.

### Overall Architecture

Data construction follows 4 steps: (1) **Judgment Filtering**: Crawling 24,406 Korean District Court civil/administrative judgments from LBOX, excluding non-deterministic cases (e.g., damages/negligence ratios) using keywords; (2) **Fact Extraction**: Gemini-2.0-Flash extracts atomic facts and synthesizes coherent case descriptions (1-shot prompt); (3) **Issue Tree Extraction**: The same model extracts hierarchical issue trees (3-shot) followed by a second round of refinement; (4) **Issue-to-Rubric Conversion**: Each issue is assigned binary labels for coverage and correctness by a judge model, which are then aggregated. Difficulty is split into easy/medium/hard based on issue count, with a test set of 300 cases manually corrected by authors.

Evaluation pipeline: 12 generation models produce reasoning chains → 10 judge models score them against the issue trees → Krippendorff’s $\alpha$ is calculated against licensed attorney annotations.

### Key Designs

1. **Hierarchical Legal Issue Tree as Rubric**:
    - **Function**: Automatically transforms judgments into a tree where nodes represent "(parties' claims, court conclusion)," allowing independent judgment at atomic nodes.
    - **Mechanism**: The root is fixed to the "purpose of claim" and final verdict. Branches drill down through legal arguments—e.g., "Insurer should pay" ← "Event covered by contract" ← "Death was sudden, fortuitous, external" ← "Cause not a pre-existing condition." LJP is equivalent to top-down back-chaining, where each step involves two actions: identifying sub-issues (decomposition) and making judgments based on facts and sub-conclusions (deduction).
    - **Design Motivation**: The tree structure captures both "what to consider" and "the conclusion for each point," breaking scalar evaluation into binary judgments per node so the evaluator does not need to process the entire reasoning chain at once.

2. **Additive LEGIT Score (5+2+3=10)**:
    - **Function**: Compresses multi-objective evaluation into a single optimizable scalar while retaining component interpretability.
    - **Mechanism**: ① **Final order accuracy** (5 points, binary)—5 points if the final verdict matches; ② **Issue coverage** (≤ 2 points)—if there are $N \geq 1$ non-root nodes, $2/N$ points per hit; ③ **Issue correctness** (≤ 3 points)—$3/N$ points if the node is hit and the conclusion is correct. Higher weight for the final order aligns with LJP goals; correctness is weighted more than coverage because reaching the correct conclusion is more informative than merely mentioning the issue.
    - **Design Motivation**: Compared to one-shot Likert scales, the additive score identifies exactly where failures occur; it also serves as a dense, decomposable reward signal for GRPO.

3. **Automated Rubric Extraction + LLM-as-judge Reliability Loop**:
    - **Function**: Generates 24K rubrics scalably and validates alignment with licensed attorneys.
    - **Mechanism**: Two-round refinement with Gemini-2.0-Flash to extract issue trees. Seven licensed Korean attorneys annotated 44 problems (300 issues) to calculate Krippendorff’s $\alpha$ and inter-annotator agreement. 10 judge LLMs were evaluated by comparing their $\alpha$ with humans and pairwise consistency against Likert evaluations.
    - **Design Motivation**: Validation showed inter-attorney $\alpha=0.87$, and strong LLM judges reached $\alpha=0.62-0.74$ with attorneys. LEGIT showed significantly higher LLM-LLM consistency than Likert scales, proving structured rubrics reduce evaluator subjectivity and are reliable for RL.

### Loss & Training

- **Data Construction**: All prompt-based (1/3-shot), no additional training.
- **RL with Rubric**: Gemma-3-4B underwent RL using GRPO on LEGIT scores. The training evaluator was Gemma-3-27B (decoupled from the test-time Gemini-2.0-Flash to prevent overfitting). Hyperparameters: KL coef 1e-3, lr 1e-6, bs 32, 8 rollouts, AdamW, max prompt 2048 / output 4096, early stop at 60 steps. Training took 41.6h (4x A100 for training + 4x A100 for evaluation).
- **RAG**: BM25 (k1=1.5, b=0.75, Kiwi POS filtering), mContriever (multilingual MS-MARCO, 512 token truncation), and fine-tuned Contriever on LEGIT train (3 epochs, bs 64, lr 1e-4).

## Key Experimental Results

### Main Results

Scores for 12 generative models on LEGIT test/300 (Gemini-2.0-Flash as judge):

| Model | LEGIT Score / 10 | Remarks |
|-------|------------------|---------|
| GPT-4o | **5.71** | Highest, but far from saturated |
| Gemini-1.5-Pro | ~5.4 | Closed-source models clustered around 5+ |
| o3 | ~5.0 | Reasoning model |
| Gemma-3-27B | 4.82 | Strongest open-source |
| Gemma-3-4B | 4.02 | Base model, RL starting point |
| EXAONE-3.5-7.8B| < 4 | Korean-specific model, relatively weak |

Reliability: Attorney vs. Attorney Krippendorff’s $\alpha = \mathbf{0.87}$; GPT/Gemini judge vs. Attorney $\alpha = \mathbf{0.62–0.74}$. LEGIT rubric LLM-LLM $\alpha$ outperformed Likert across the board ("modular > coarse").

### Ablation Study

| Configuration | LEGIT (Gemma-3-4B) | Final | Coverage | Correctness |
|---------------|-------------------|-------|----------|-------------|
| Base | 4.02 | – | – | – |
| + BM25 RAG | 4.40 | ↑ | ↑ | ↑ |
| + Contriever RAG | 4.42 | ↑ | ↑ | ↑ |
| + GT citation RAG | ~5.0 | ↑↑ | ↑↑ | ↑↑ |
| + RL (LEGIT reward) | **4.77** | ↑↑ | **↓** | ↑↑ |
| + RL (final-order reward) | 4.31 | ↑ | ↓↓ | ↑ |

Gemma-3-4B trained with LEGIT rewards nearly matched Gemma-3-27B (4.82) and **outperformed the model trained solely on final-order rewards**, indicating dense rubric rewards are superior to sparse binary rewards for legal reasoning.

Error Attribution (Fig. 7): Parent issue accuracy vs. sub-issue status:

| Sub-issue status | Parent Accuracy |
|------------------|-----------------|
| All covered and correct | Highest (near upper bound) |
| ≥1 covered but wrong (**deduction error**) | Severe drop |
| Sub-issue not covered (**decomposition error**) | Severe drop |

### Key Findings

- **SOTA LLMs are far from saturated**: The peak score of 5.71/10 suggests significant room for improvement in complex civil law.
- **Two major error categories**: Decomposition (missing sub-issues) and deduction (logical errors), both of which propagate errors up the issue tree.
- **RAG vs. RL complementarity**: RAG improves all scores (broad exploration), while RL improves correctness but reduces coverage (the policy learns to "avoid mentioning issues to avoid penalties"). This aligns with Fig. 7: incorrectness is penalized more heavily than omission, leading the policy to skip ambiguous issues.
- **LLM judges are generally lenient**: Compared to attorneys, judges tend to classify "similar but distinct legal concepts" as covered/correct. Smaller models are sometimes stricter, leading to $\alpha$ scores closer to attorneys by coincidence, not superior capability.
- **Rubric consistency > Likert**: Structured prompts remain the foundation of consistency, as even with full text and guidance, Likert $\alpha$ remained lower than LEGIT.

## Highlights & Insights

- **Judgments as natural rubric annotations**: Using the "claims + conclusions" hierarchy as ground truth bypasses the cost of manual rubrics, scaling expert evaluation to 24K instances. This can be adapted to any domain with structured final documents (medical reports, audit summaries, accident investigations).
- **Additive scores as both metrics and rewards**: Sparse rewards are a bottleneck for R1-style RL. Using dense issue-level rewards allowed a 4B model to rival a 27B model, proving rubric rewards are more efficient for structured reasoning.
- **Inter-judge consistency as a core metric**: Using consistency across evaluators as a criterion for rubric quality provides a robust methodology for LLM-as-judge design.
- **RL as risk-aversion**: The weight design (correctness > coverage) caused the model to favor "silence over error." This suggests that if a task requires comprehensiveness, weights must be adjusted or coverage penalties added.
- **Leniency in judges**: Precautions must be taken against "similarity $\neq$ equivalence" confusion in high-precision fields like law and medicine.

## Limitations & Future Work

- **Restricted to Korean law/language**: Generalizability to common law or other languages is an open question, though the tree structure should be universal.
- **Computational cost**: Rubric-based evaluation cost scales with the number of issues, creating a trade-off between precision and API costs.
- **Citation accuracy not evaluated**: Frequent case ID overlaps led the authors to avoid this to prevent excessive false negatives.
- **No manual correction of the full training set**: Sampling only 50 cases showed 92% answerability, but minor errors (missing antecedents, over-specification) likely persist.
- **RL coverage drop**: Needs redesign for applications where comprehensiveness is paramount (e.g., due diligence).

## Related Work & Insights

- **vs. BigGen-Bench (Kim et al. 2025b)**: BigGen uses high-quality specialized manual rubrics but is limited in scale; LEGIT scales to 24K via automation while maintaining expert alignment.
- **vs. CitaLaw / Legal RAG (Zhang et al. 2025)**: While CitaLaw focuses on citation quality, LEGIT uses retrieval to improve reasoning coverage and quantifies the effect via issue trees.
- **vs. DeepSeek-R1 (Guo et al. 2025)**: R1 succeeded in math with sparse binary rewards; LEGIT shows that dense rubric rewards are more effective in expert domains.
- **vs. Rubrics as Rewards (Gunjal et al. 2025)**: Shares the philosophy but adds "automated generation + inter-judge validation" for a reproducible pipeline.
- **Transferable insight**: Moving from "domain document $\to$ auto-rubric $\to$ direct RL reward" can significantly lower costs for medical, compliance, and specialized QA.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Legal issue trees as rubrics represent a clear, scalable new approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage with 12 gen models, 10 judge models, attorney metrics, and RAG/RL ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Definitions and charts are clear, though density requires careful reading.
- **Value**: ⭐⭐⭐⭐⭐ The 24K dataset and the "rubric reward > final-answer reward" evidence provide a significant push for Legal AI and expert reasoning evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2026\] Evaluating Reasoning Models for Queries with Presuppositions](evaluating_reasoning_models_for_queries_with_presuppositions.md)
- [\[ACL 2026\] Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues](are_they_lovers_or_friends_evaluating_llms39_social_reasoning_in_english_and_kor.md)
- [\[ACL 2026\] HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing](howtobench_holistic_evaluation_for_llms_capability_in_human-level_writing_using_.md)
- [\[ACL 2026\] Gated Tree Cross-Attention for Checkpoint-Compatible Syntax Injection in Decoder-Only LLMs](gated_tree_cross-attention_for_checkpoint-compatible_syntax_injection_in_decoder.md)

</div>

<!-- RELATED:END -->
