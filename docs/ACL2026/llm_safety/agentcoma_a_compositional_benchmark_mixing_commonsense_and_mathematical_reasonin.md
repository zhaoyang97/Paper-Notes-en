---
title: >-
  [Paper Note] AgentCoMa: A Compositional Benchmark Mixing Commonsense and Mathematical Reasoning in Real-World Scenarios
description: >-
  [ACL2026][LLM Safety][Mixed-type compositional reasoning] AgentCoMa constructs an agentic benchmark that forcibly combines commonsense selection with single-step mathematical operations. Evaluation of 61 LLMs reveals tha…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "Mixed-type compositional reasoning"
  - "Commonsense reasoning"
  - "Mathematical reasoning"
  - "Agent evaluation"
  - "Model vulnerability"
date: 2026-05-08
content_hash: 3d72624e8470d041
---

# AgentCoMa: A Compositional Benchmark Mixing Commonsense and Mathematical Reasoning in Real-World Scenarios

**Conference**: ACL2026  
**arXiv**: [2508.19988](https://arxiv.org/abs/2508.19988)  
**Code**: https://agentcoma.github.io  
**Area**: LLM Safety / LLM Reasoning Evaluation  
**Keywords**: Mixed-type compositional reasoning, Commonsense reasoning, Mathematical reasoning, Agent evaluation, Model vulnerability  

## TL;DR
AgentCoMa constructs an agentic benchmark that forcibly combines commonsense selection with single-step mathematical operations. Evaluation of 61 LLMs reveals that while models typically solve both sub-problems independently, the average accuracy drops from 80% (when both sub-steps are solvable independently) to 51% after composition, exposing significant vulnerability in mixed-type compositional reasoning.

## Background & Motivation
**Background**: Current LLMs satisfy various benchmarks for commonsense reasoning (spatial, temporal, social, and causal knowledge) and mathematical reasoning (from K-12 word problems to competitive math). Agentic benchmarks often introduce complexity via tool calls, long task chains, and dynamic environments to measure the comprehensive capabilities of LLM agents.

**Limitations of Prior Work**: While these evaluations are valuable, they struggle to isolate a specific question: when a task requires both commonsense judgment and mathematical calculation, does the model fail due to "composition failure" or is it simply hindered by extra factors like long context, tool overhead, or environmental changes? Excessive noise in benchmarks prevents precise error attribution.

**Key Challenge**: Real-world agent tasks frequently require switching between different reasoning types, such as using commonsense to identify items for room-temperature storage before calculating total prices. However, existing compositional benchmarks mostly combine steps of the same type (e.g., two knowledge retrieval steps or two math steps). High performance in a single reasoning type does not guarantee a model can stably link different types of reasoning.

**Goal**: The authors aim to construct a controlled evaluation environment where each sample can be decomposed into a commonsense sub-problem and a math sub-problem, neither of which is difficult for humans or strong models. If a model fails the combined problem, it can be more clearly attributed to the "cross-type composition" itself rather than step-wise difficulty.

**Key Insight**: The paper selects commonsense and math as complementary reasoning types: the former aligns with fast, intuitive "System 1," while the latter resembles slow, explicit "System 2." Placing both in a realistic agent scenario tests whether a model can effectively mobilize different capabilities within the same reasoning chain.

**Core Idea**: Use "commonsense selection + single-step arithmetic" as an artificially constructed task to isolate mixed-type compositional reasoning in agent scenarios. LLM compositional vulnerability is then located through sub-problem controls, human experiments, and interpretability analysis.

## Method
This paper proposes a benchmark and a diagnostic protocol rather than a new model. The key lies in each sample maintaining the "compositional problem" alongside "two independently evaluable sub-problems," allowing comparison across three input formats: commonsense only, math only, and the two-step composition.

### Overall Architecture
AgentCoMa takes real-world agent task descriptions as input and typically expects a numerical answer. A sample contains four core objects: the compositional problem, the commonsense sub-problem, the math sub-problem, and their respective ground truths.

The compositional problem requires the model to make a choice based on commonsense and then perform an arithmetic operation on the selected objects. For example, in a garage tool organization sample, the model must identify electrical appliances (power drill, extension cords, leaf blower) and calculate their total count. The commonsense sub-problem only asks "which items belong in the waterproof cabinet," while the math sub-problem explicitly states the identified items and only requires addition. This distinguishes "knowledge selection failure" from "composition failure."

The dataset covers 5 agentic scenarios: house working, web shopping, science experiments, smart assistant, and travel agent. The math steps consist only of basic arithmetic. The total scale is 260 samples (80 dev, 180 test). It serves as an evaluation set for pre-trained or instruction-tuned models.

### Key Designs
1. **Decomposable Hybrid Reasoning Samples**:
	- **Function**: Explicitly splits each compositional problem into commonsense and math sub-problems for 3-way assessment.
	- **Mechanism**: The commonsense step determines the targets for calculation; the math step performs a single small-value operation. Sub-problems remove the burden of the opposite reasoning type. If a model solves sub-problems but fails the composition, failure is attributed to compositional overhead.
	- **Design Motivation**: This is more diagnostic than final accuracy. Traditional benchmarks only show *that* a model failed; AgentCoMa determines if it failed at commonsense, arithmetic, or the link between them.

2. **Manual Construction and Multi-round Expert Verification**:
	- **Function**: Ensures realistic scenarios, authentic commonsense requirements, single-step arithmetic, and unambiguous answers.
	- **Mechanism**: Samples are manually written by experts (not LLM-generated) and undergo binary checks, independent solving, answer comparison, and feedback. Any failure leads to rewriting and re-verification. Validation is performed by experts other than the sample authors to minimize bias.
	- **Design Motivation**: The core conclusion relies on "easy steps but hard composition." Ambiguity or hidden multi-step math would break this contrast.

3. **Diagnostic Chain from Behavior to Mechanism**:
	- **Function**: Reports more than just accuracy; explains where the compositionality gap originates.
	- **Mechanism**: Authors compare gaps across AgentCoMa, Bamboogle, and MultiArith. They use Min-K%++ to estimate pattern similarity to training distributions, lookback attention ratio to analyze context utilization, and QRNCA to compare neural activation overlap.
	- **Design Motivation**: Prevents attributing performance drops solely to context length or poor prompting. Diagnosis suggests mixed-type tasks are rare in training distributions, causing models to favor math circuits over commonsense circuits during reasoning.

### Loss & Training
No new model is trained; therefore, no model loss function exists. All LLMs use two-shot chain-of-thought (CoT) prompting and greedy decoding. Numerical answers are extracted via regex and exact-matched. Commonsense sub-problem answers are evaluated by LLM-as-a-judge. 

Human control experiments involved 45 non-expert crowdworkers with high school education and English fluency, solving problems without calculators. Sub-problems and compositional problems from the same sample were given to different individuals to prevent leakage.

## Key Experimental Results

### Main Results
61 LLMs were evaluated, with 16 representative models shown in detail (covering instruction-tuned, SFT reasoning, and RL-reasoning strategies). On average, models solve both sub-steps independently in 80% of cases, but compositional accuracy is only 51%, revealing a 29% average compositionality gap.

| Target / Benchmark | Both Sub-steps Correct | Comp. Acc. | Comp. Gap / Note |
|-------------|----------------|------------|-----------------|
| AgentCoMa (16 Rep. LLMs Avg) | 80% | 51% | 29% average gap; issue is composition, not single-step performance |
| AgentCoMa (Human) | 78.9% | 82.8% | Humans show no composition collapse |
| Bamboogle (LLM Avg) | 53% | 52% | Homogeneous (knowledge+knowledge); negligible gap |
| MultiArith (LLM Avg) | Near perfect | Near perfect | Homogeneous (math+math); gap < 1% |

Strong models are not immune. Llama-3.3-70B-Instruct shows 90.0% sub-step joint success vs 73.3% compositional accuracy. Qwen2.5-14B-Instruct shows 88.9% vs 60.6%. Smaller models collapse more severely (e.g., SimpleRL-8B at 56.7% vs 25.0%).

### Ablation Study
Ablations function as diagnostic experiments to rule out context length, prompting flaws, or same-type complexity.

| Analysis | Key Result | Note |
|--------|----------|------|
| Failure Breakdown | ~0.74 of failures occur in samples where both sub-steps were solvable independently | Failure is specifically due to unreliable combination of reasoning types |
| Self-Ask Prompting | ~27% gap (vs 29% for CoT) | Explicit decomposition prompting only slightly mitigates the issue |
| Lookback Attention | CS: 71.49, Math: 72.20, Comp: 70.75 | Models show lower lookback attention in composition, leading to context hallucinations |
| Neuron Overlap: Llama-3.1-8B | Comp-Math: 39%, Comp-CS: 3% | Composition tasks primarily activate math circuits, not commonsense circuits |
| Neuron Overlap: GeneralReasoner-4B | Comp-Math: 54%, Comp-CS: 10% | Reasoning models show similar bias to instruction-tuned models |

### Key Findings
- AgentCoMa’s gap is significantly larger than same-type benchmarks. MultiArith (math+math) and Bamboogle (knowledge+knowledge) show compositional accuracy close to joint sub-step accuracy, whereas AgentCoMa shows a clear divergence.
- Reasoning optimization (RL or SFT) does not solve this. Reasoning models exhibit the same large gaps as instruction-tuned models, proving that long-chain reasoning capability $\neq$ cross-type compositional capability.
- Training distribution similarity analysis suggests mixed-type problems are rare. Min-K%++ scores indicate compositional problems are less similar to typical training patterns than individual components.
- Mechanism analysis reveals that when faced with composition, models follow math-related neural patterns; commonsense neurons are under-activated, leading to fluent but factually incorrect (contextually disjointed) reasoning.

## Highlights & Insights
- The most valuable design is the triplet structure ("Composition + 2 Sub-problems"). It converts ambiguous failure diagnosis into quantifiable comparisons, proving models aren't "bad at addition" or "bad at shopping," but bad at linking "what to buy" with "how much it costs."
- The difficulty of AgentCoMa is the switch between reasoning types, not absolute complexity. This is vital for agent evaluation; real-world failures often stem from miscombining simple skills in the correct order rather than a lack of high-level planning.
- The paper connects behavioral gaps to training distributions and neural activation. While not causal proof, it makes the "mixed-type tasks are under-learned" hypothesis highly plausible.
- Insights for safety: If models cannot stably combine commonsense and arithmetic in short, tool-free contexts, deploying them as autonomous decision-makers in complex agent scenarios requires rigorous step-wise verification.

## Limitations & Future Work
- AgentCoMa is an intentionally simplified control experiment. Each problem only combines two types and one math operation. It does not cover the long chains, multi-constraints, or multi-turn interactions of real agent tasks.
- The data scale is relatively small (180 test samples, 260 total). While comparable to benchmarks like Bamboogle, larger datasets are needed to subdivide by scenario or arithmetic type.
- Only preliminary exploration of reasoning order was conducted. Reversing the order (Math then CS) requires significant expert annotation to ensure validity.
- Automated evaluation of commonsense sub-problems relies on LLM-as-a-judge. Evaluator bias remains a potential factor in open-ended answer spaces.
- Future work could extend to other hybrid types (e.g., Commonsense + Constrained Planning, Spatial + Arithmetic) and integrate the benchmark into real agent traces.

## Related Work & Insights
- **vs Commonsense Benchmarks**: Traditional benchmarks test if a model *knows* facts; this tests if it can *use* them as a link in a reasoning chain.
- **vs Math Benchmarks**: GSM-style tasks focus on multi-step calculation; AgentCoMa asks if the model can correctly select the targets for calculation using commonsense.
- **vs Compositional Benchmarks**: Most combine the same reasoning type. AgentCoMa identifies vulnerabilities specific to cross-type transitions.
- **vs Agentic Benchmarks**: Sacrifices real-world noise (tools, dynamic environments) for a cleaner diagnostic window into reasoning failures.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Mixed-type composition is a distinct and well-defined problem.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers 61 LLMs and includes mechanistic analysis; however, data scale is modest.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear structure with well-aligned evidence and conclusions.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for agent reliability, highlighting that individual skills of different types do not guarantee successful composition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](../../ICLR2026/llm_safety/measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)
- [\[ICLR 2026\] Moving Beyond Medical Exams: A Clinician-Annotated Fairness Dataset of Real-World Tasks and Ambiguity in Mental Healthcare](../../ICLR2026/llm_safety/moving_beyond_medical_exams_a_clinician-annotated_fairness_dataset_of_real-world.md)
- [\[NeurIPS 2025\] SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications](../../NeurIPS2025/llm_safety/swe-sql_illuminating_llm_pathways_to_solve_user_sql_issues_in_real-world_applica.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)
- [\[ACL 2026\] CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge](curate_continual_unlearning_in_real_time_with_ensured_preservation_of_llm_knowle.md)

</div>

<!-- RELATED:END -->
</div>

## Related Papers

- [\[ICLR 2026\] Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](../../ICLR2026/llm_safety/measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)
- [\[ICLR 2026\] Moving Beyond Medical Exams: A Clinician-Annotated Fairness Dataset of Real-World Tasks and Ambiguity in Mental Healthcare](../../ICLR2026/llm_safety/moving_beyond_medical_exams_a_clinician-annotated_fairness_dataset_of_real-world.md)
- [\[NeurIPS 2025\] SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications](../../NeurIPS2025/llm_safety/swe-sql_illuminating_llm_pathways_to_solve_user_sql_issues_in_real-world_applica.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)
- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)

</div>

<!-- RELATED:END -->
