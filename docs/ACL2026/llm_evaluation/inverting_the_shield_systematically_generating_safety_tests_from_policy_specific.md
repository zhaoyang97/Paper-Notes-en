---
title: >-
  [Paper Note] Inverting the Shield: Systematically Generating Safety Tests from Policy Specifications
description: >-
  [ACL2026][LLM Evaluation][Safety policy specification] POLARIS compiles natural language safety policies into first-order logic specifications, constructs semantic policy graphs…
tags:
  - "ACL2026"
  - "LLM Evaluation"
  - "Safety policy specification"
  - "Formal testing"
  - "Red-teaming evaluation"
  - "First-order logic"
  - "Coverage-driven generation"
date: 2026-05-08
content_hash: fdc205563a2ae63b
---

# Inverting the Shield: Systematically Generating Safety Tests from Policy Specifications

**Conference**: ACL2026  
**arXiv**: [2605.24883](https://arxiv.org/abs/2605.24883)  
**Code**: https://github.com/huac-lxy/POLARIS  
**Area**: LLM Security Evaluation / Specification-driven Testing  
**Keywords**: Safety policy specification, Formal testing, Red-teaming evaluation, First-order logic, Coverage-driven generation  

## TL;DR
POLARIS compiles natural language safety policies into first-order logic specifications, constructs semantic policy graphs, and systematically traverses them to generate test queries. This shifts LLM safety evaluation from heuristic red-teaming to traceable, coverage-guaranteed, and reproducible specification-driven testing.

## Background & Motivation
**Background**: LLM safety evaluation typically follows two routes: static benchmarks such as AdvBench, HarmBench, and SORRY-Bench, or dynamic attack generation driven by automated red-teaming or curiosity. The former facilitates horizontal comparison, while the latter excels at discovering new failure modes.

**Limitations of Prior Work**: Static benchmarks are costly to produce, prone to obsolescence, and may suffer from training data contamination. Although dynamic red-teaming is flexible, most methods rely on heuristic searches that lack systematic coverage guarantees for the safety policy space. They can indicate that a "model failed" but struggle to specify which policy was tested or which combinations remain unexplored.

**Key Challenge**: Safety policies serve as the protective boundary but are not machine-verifiable specifications when existing as natural language. Evaluation starting only from existing attack samples is constrained by the sample distribution; starting from policies themselves requires transforming ambiguous text into traversable and instantiatable structures.

**Goal**: The authors aim to port specification testing principles from software engineering to AI safety evaluation: extracting verifiable logical constraints from policy text, systematically exploring the policy space, instantiating abstract violation patterns into natural language tests, and maintaining traceability from each test back to the original policy clauses.

**Key Insight**: The paper observes that "the shield also defines the attack boundary." Safety policies define boundaries the model must not cross; once these boundaries are formalized, test cases can be inversely generated to cover risk scenarios near these boundaries.

**Core Idea**: A pipeline of "Natural Language Policy $\rightarrow$ First-order Logic Templates $\rightarrow$ Semantic Policy Graph $\rightarrow$ Graph Traversal Instantiation" replaces safety test generation based solely on existing attack samples or LLM free-play.

## Method

### Overall Architecture
POLARIS consists of three phases. The first phase decomposes natural language policies into atomic rules and rewrites them into First-order Logic (FOL) in the form of Abstract Violation Templates (AVTs). The second phase organizes entities, actions, and relationships from all AVTs into a Semantic Policy Graph, expanding implicit associations through semantic merging and LLM-driven link prediction. The third phase performs controlled random walks on the graph to obtain abstract violation paths, which are then grounded into natural language test queries by a generative model.

The input is safety policy text from enterprises or regulators, and the output is a set of safety tests with policy origins, logical paths, and natural language descriptions. Unlike traditional red-teaming, POLARIS does not collect harmful prompts first; it proactively covers the policy space starting from policy clauses.

### Key Designs
1. **Policy-to-Logic Compilation**:

	- **Function**: Transforms ambiguous natural language policies into verifiable logical templates, establishing traceability between test cases and policy clauses.
	- **Mechanism**: Composite policies are first decomposed into atomic rules (e.g., splitting "do not distribute drugs or firearms" into two prohibited items). Entities, actions, and deontic modalities (obligations/prohibitions) are then extracted to form AVTs like $$\forall x,y: \mathcal{P}_{pre}(x,y) \Rightarrow \textsc{Violation}(R_i)$$.
	- **Design Motivation**: Without logicalization, generators merely mimic policy language, making it difficult to prove which rule a query actually covers. AVTs provide explicit "violation conditions" for test generation.

2. **Semantic Policy Graph**:

	- **Function**: Connects dispersed logical templates into a traversable policy space, supporting the discovery of cross-clause and cross-concept combinatorial risks.
	- **Mechanism**: Entities in AVTs are mapped to nodes, and actions/relationships to edges. Synonymous nodes are merged via embedding similarity, and LLM-driven link prediction adds commonsense or causal connections. For instance, a "chemistry lab" in one clause and "precursor chemicals" in another might form a combined risk path through link prediction.
	- **Design Motivation**: Individual policy rules cover isolated scenarios, whereas real safety failures often occur when multiple concepts are spliced. The semantic graph extends evaluation from "rule-by-rule" to "multi-hop violation paths."

3. **Graph-Guided Query Instantiation**:

	- **Function**: Translates abstract paths into natural, contextualized test queries while preserving traceable graph paths and AVT sources.
	- **Mechanism**: The system performs controlled random walks on the augmented graph to sample violation scenario skeletons. A generative model then produces final queries based on the scenario, context, and intent-masking variables.
	- **Design Motivation**: Directly translating logical paths into queries results in mechanical output; real model failures often appear in more natural narrative contexts. The instantiation phase converts "verifiable" into "executable."

### Loss & Training
POLARIS does not train a target LLM but builds a test generation system. Experiments utilize 16 public policies from 9 AI companies plus 4 Chinese regulatory documents to compile the policy knowledge base. Generation costs primarily stem from GPT-4-Turbo API calls. Evaluation utilizes density-weighted Coverage / Novelty, Policy Clause Coverage, and Attack Success Count. Coverage treats generated samples as covering the benchmark if the nearest neighbor distance is below a threshold. Novelty counts the proportion of generated samples not covered by benchmarks, using local density weights to penalize dense, repetitive clusters.

## Key Experimental Results

### Main Results
| Evaluation Dimension | Key Setting | POLARIS Result | Comparison / Interpretation |
|----------|----------|-------------|-------------|
| Policy Clause Coverage | 16 corporate policies + 4 regulatory docs | 100% | Indicates every policy rule can instantiate at least one test query |
| Coverage @ $\tau=0.6$ | Relative to HarmBench | 93.21% | Demonstrates the generated set covers most existing safety benchmark semantic space |
| Novelty @ $\tau=0.6$ | Relative to HarmBench | 35.26% | Retains significant new semantic content while maintaining high coverage |
| Mistral-7B Attack Success | GPT-5-mini Judge | 13,722 | Approx. 4.8x AirBench (2,850) |
| Qwen-7B Attack Success | GPT-5-mini Judge | 11,150 | Approx. 4.9x strongest baseline Curiosity (2,294) |
| Vicuna Attack Success | DeepSeek-R1 Judge | 8,590 | Approx. 5.2x AirBench (1,639) |

### Ablation Study
| Module / Metric | Full POLARIS | Result w/o Module | Explanation |
|-------------|-------------|------------------|------|
| Logical Formalization: Policy Compliance | 92.90% | w/o Logic: 88.90% | Formal constraints reduce generation drift from policy objectives |
| Semantic Graph Traversal: Avg Novelty @ $\tau=0.6$ | 28.00% | w/o Graph: 24.80% | Graph structure helps discover new paths hard to reach via random sampling |
| Policy-to-Logic Quality: Fine-grained score | 9.10 / 10 | N/A | LLM judge considers most logical expressions retain semantic details |
| Policy-to-Logic Quality: Binary Accuracy | 92.06% | N/A | Strict logical correctness still has minor errors; requires filtering |
| Generation Cost | $70.52 for 28,660 queries, 4.86 hours | Marginal cost: $0.94 / 1k queries | Graph construction is a one-time cost; subsequent expansion is inexpensive |

### Key Findings
- POLARIS shows the most significant advantage on modern models like Mistral-7B and Qwen-7B, achieving 4x to 6x the attack success count of strong baselines.
- Static benchmarks were not simply duplicated: while Coverage is high, Novelty remains substantial, proving graph traversal effectively expands the test space.
- Formal logic and semantic graphs are not merely decorative. Removing logic decreases policy compliance, while removing the graph reduces novelty.

## Highlights & Insights
- The primary highlight is the reformulation of LLM safety evaluation as a "specification testing" problem. This perspective shifts evaluation from sample-driven to policy-driven, which is particularly suited for regulatory or corporate compliance scenarios.
- Density-weighted Coverage / Novelty is more reasonable than standard nearest-neighbor coverage, as safety benchmarks often contain many similar attacks that can mislead standard coverage metrics.
- The Semantic Policy Graph provides a reusable intermediate asset. Once the policy graph is constructed, tests can be instantiated for different domains, models, or risk appetites without rewriting prompts from scratch.
- For safety evaluation toolchains, the insight is that future benchmarks should publish not just question sets, but the policy specifications, coverage definitions, and generation trajectories behind them.

## Limitations & Future Work
- The authors explicitly note that generation quality is limited by the quality of input policies. If the policy itself is ambiguous, conflicting, or incomplete, POLARIS can only systematize these defects.
- Current methods primarily handle static single-turn interactions and do not yet cover multi-turn dialogues, tool-calling agents, or stateful risks; these scenarios require incorporating temporal states and action constraints into logical expressions.
- Intermediate steps rely on LLMs for extracting entities, actions, and FOL templates; while validation scores are high, they are not perfectly accurate. Large-scale deployment requires more robust human auditing, formal verification, or consistency filtering.
- Attack success counts emphasize the quantity of discovered failures, but the severity of failures varies. Risk weighting, harm levels, and remediation priorities could be added in future work.

## Related Work & Insights
- **vs Static Safety Benchmarks**: While AdvBench, HarmBench, etc., provide fixed test sets, POLARIS continuously generates tests from policy specifications. The former is easy to reproduce; the latter better handles rapidly changing policies and models.
- **vs Automated Red-teaming**: Methods like curiosity-driven red-teaming rely on exploration heuristics, whereas POLARIS explicitly binds the exploration space to a policy graph, offering superior coverage and traceability.
- **vs Evol-Instruct / MAGPIE**: These methods generate complex instructions to improve model capability; POLARIS generates specification-traceable safety tests, serving a different objective.
- **Insights for Future Research**: Formal specifications can be introduced into multi-turn agent safety, tool-calling permission testing, and corporate internal safety acceptance, upgrading "number of prompts tested" to "percentage of policy states covered."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically porting specification-driven software testing to LLM safety evaluation; problem definition and method combination are highly distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Coverage, attack success, cost, logical validation, and ablations are comprehensive, though multi-turn/agent scenarios are not yet covered.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure with well-defined experimental questions; some main text readability is compressed by large appendix tables.
- Value: ⭐⭐⭐⭐⭐ Directly insightful for safety evaluation, compliance testing, and dynamic benchmark construction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PolicyLLM: Towards Excellent Comprehension of Public Policy for Large Language Models](policyllm_towards_excellent_comprehension_of_public_policy_for_large_language_mo.md)
- [\[ACL 2026\] Stability vs. Manipulability: Evaluating Robustness Under Post-Decision Interaction in LLM Judges](stability_vs_manipulability_evaluating_robustness_under_post-decision_interactio.md)
- [\[ACL 2026\] StratMem-Bench: Evaluating Strategic Memory Use in Virtual Character Conversation Beyond Factual Recall](stratmem-bench_evaluating_strategic_memory_use_in_virtual_character_conversation.md)
- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[ACL 2026\] Question Difficulty Estimation for Large Language Models via Answer Plausibility Scoring](question_difficulty_estimation_for_large_language_models_via_answer_plausibility.md)

</div>

<!-- RELATED:END -->
