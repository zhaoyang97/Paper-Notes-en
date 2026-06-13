---
title: >-
  [Paper Note] The Path Not Taken: Duality in Reasoning about Program Execution
description: >-
  [ACL 2026][Code Intelligence][Program execution reasoning] This paper proposes the concept of duality in program execution reasoning. Through the DexBench benchmark (445 paired instances)…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Program execution reasoning"
  - "counterfactual reasoning"
  - "dual-path reasoning"
  - "code coverage"
  - "LLM code understanding"
date: 2026-05-08
content_hash: 2e9712a17d7e30e2
---

# The Path Not Taken: Duality in Reasoning about Program Execution

**Conference**: ACL 2026  
**arXiv**: [2604.20917](https://arxiv.org/abs/2604.20917)  
**Code**: [github.com/sail-ucf/dexbench](https://github.com/sail-ucf/dexbench)  
**Area**: Code Intelligence / Program Reasoning  
**Keywords**: Program execution reasoning, counterfactual reasoning, dual-path reasoning, code coverage, LLM code understanding

## TL;DR
This paper proposes the concept of duality in program execution reasoning. Through the DexBench benchmark (445 paired instances), it jointly evaluates the forward execution reasoning (predicting code coverage for a given input) and backward counterfactual reasoning (inferring input mutations to steer execution flow toward a target branch) of LLMs. It finds that strong performance in a single direction does not translate to success under joint evaluation, revealing deficiencies in the models' causal understanding of programs.

## Background & Motivation

**Background**: LLMs are widely adopted in software engineering tasks (code generation, testing, bug fixing, etc.), but their performance is inconsistent—they can solve complex programming problems yet fail at basic loop reasoning. This may stem from mimicking surface patterns rather than truly understanding programs. Recent work has begun to focus on fine-grained reasoning capability assessments, such as code coverage prediction, input-output mapping, and intermediate state tracking.

**Limitations of Prior Work**: Existing program execution benchmarks examine runtime behavior under only a single test case (i.e., reasoning along one execution path), but a given program may traverse multiple paths based on different inputs. Evaluating with a single test case provides only a narrow perspective of a model's program understanding. Furthermore, benchmarks defined by fixed input-output pairs are susceptible to training data contamination (models might memorize the answers).

**Key Challenge**: Two paths of a program share an execution space and only diverge at branch points due to different program states. Existing evaluations ignore the causal relationship between execution paths—evaluating only whether a model can predict "which path was taken" without assessing "why it was taken" and "how to make it take another path."

**Goal**: To design a joint evaluation framework that simultaneously tests the forward execution reasoning and backward counterfactual reasoning capabilities of LLMs, thereby providing a more robust measurement of their causal understanding of program execution.

**Key Insight**: Two program paths share a common execution space and diverge only at branch conditions. The authors argue that understanding program execution requires understanding both "how the observed behavior occurs" and "under what conditions the execution would steer toward another path."

**Core Idea**: Define the duality of program execution reasoning—forward reasoning predicts the observable behavior of an execution path, while backward reasoning infers input mutations that steer the execution flow toward a counterfactual path. These two combined constitute dual-path reasoning: $\mathcal{R}_{dual} = \mathcal{R}_{exec} \oplus \mathcal{R}_{cf}$.

## Method

### Overall Architecture
DexBench consists of 445 evaluation instances, each composed of a pair of forward and backward reasoning tasks. Given a program $P$ and a test input $I_{exec}$: the forward task requires predicting code coverage (which lines are executed); the backward task requires mutating the input $I_{exec}$ into $I_{cf}$ such that the execution flow covers a target branch that was originally unvisited. Dual-path reasoning is considered successful only if both tasks are successful. Data are sourced from CruxEval (298 instances, simple control flow), HumanEval (100 instances, medium complexity), and PythonSaga (47 instances, deep nesting + recursion).

### Key Designs

1.  **Forward Execution Reasoning ($\mathcal{R}_{exec}$)**:
    - **Function**: Predict statement coverage of a program under a given input (code coverage prediction).
    - **Mechanism**: Given a program $P$ and input $I_{exec}$, the model must predict the set of executed line numbers $\phi(\tau_{exec})$. The success criterion is an exact match—at least one generated candidate answer must correctly predict the code coverage. Ground-truth coverage information is collected using the SlipCover tool.
    - **Design Motivation**: Code coverage prediction requires the model to maintain a representation of the program state and update it according to the semantics of each statement. It is a natural task for evaluating execution reasoning. The difficulty increases with the length of the execution trajectory and the complexity of state transitions (e.g., non-trivial control flow).

2.  **Backward Counterfactual Reasoning ($\mathcal{R}_{cf}$)**:
    - **Function**: Infer how to mutate the original input to steer the program execution flow to cover a specific target branch.
    - **Mechanism**: Given a program $P$, the original input $I_{exec}$, and a counterfactual target (covering a specific unvisited branch $b$), the model must generate a mutated input $I_{cf}$ such that $b \in \phi(\tau_{cf})$. The branch with the largest coverage increment is selected as the counterfactual target. Correctness is verified by actually executing the generated input.
    - **Design Motivation**: Unlike coverage-guided fuzzing, DexBench requires the model to identify necessary input changes through reasoning rather than random mutation. Counterfactual reasoning requires analyzing how input changes affect control flow and program state, making it a key task for testing causal understanding.

3.  **Dual-path Reasoning ($\mathcal{R}_{dual}$)**:
    - **Function**: Jointly evaluate forward and backward reasoning capabilities to provide a robust assessment of program execution understanding.
    - **Mechanism**: $S_{dual} = S_{exec} \wedge S_{cf}$—an instance is successful under dual-path evaluation if and only if the model correctly predicts the original coverage and generates at least one input that covers the target branch. This avoids the narrowness and incompleteness of single-path evaluation.
    - **Design Motivation**: Experiments found that many models succeed in only one direction—predicting coverage but failing to mutate the input (or vice versa)—indicating that single-path evaluation overestimates a model's program understanding.

### Loss & Training
DexBench is an evaluation benchmark and does not involve training. Evaluation utilizes the pass@k metric (k=1 and k=5) with one-shot prompting.

## Key Experimental Results

### Main Results (Dual-path Reasoning pass@5, %)

| Model | CruxEval | HumanEval | PythonSaga |
| :--- | :--- | :--- | :--- |
| Llama-3.2-3B-Inst. | 0.0 | 0.0 | 0.0 |
| Qwen2.5-32B | 57.0 | 31.0 | 6.4 |
| QwQ-32B | 33.2 | 25.0 | 2.1 |
| Gemini 2.5 Flash | 73.8 | 62.0 | 8.5 |
| GPT-5 Mini | 91.6 | 76.0 | 44.7 |
| Claude Sonnet 4 | **95.3** | **79.0** | **70.2** |

### Ablation Study (Input Generation vs. Input Mutation, GPT-5 Mini)

| Setting | CruxEval pass@5 | HumanEval pass@5 | PythonSaga pass@5 |
| :--- | :--- | :--- | :--- |
| Input Gen. (No orig. input) | 97.7 | 99.0 | 95.7 |
| Input Mutation (Ours) | 97.0 | 88.0 | 95.7 |

### Key Findings
- **Success in a single direction does not translate to joint success**: Many models perform well in either execution reasoning or counterfactual reasoning but drop significantly under dual-path evaluation, proving the incompleteness of single-path evaluation.
- **Reasoning post-training is ineffective**: Among open-source models, non-reasoning variants (Qwen2.5-32B, Mistral Small 24B) actually outperformed their reasoning-specialized variants (QwQ-32B, Magistral Small) in dual-path reasoning by 63.9% and 47.9% on average.
- **Scaling effects are non-monotonic**: Qwen2.5-32B consistently outperformed Qwen2.5-72B in dual-path reasoning; all models <10B had nearly zero performance in execution reasoning but showed some ability in counterfactual reasoning.
- **Complexity impacts are significant**: From CruxEval to PythonSaga, dual-path performance dropped significantly (e.g., Gemini 2.5 Flash fell from 73.8% to 8.5%).
- **Input mutation is harder than input generation**: Removing the original input actually improved performance, as input mutation requires reasoning about how input perturbations propagate through intermediate program states.

## Highlights & Insights
- **Duality Framework**: Program execution reasoning naturally possesses a forward/backward dual structure. Joint evaluation reflects true understanding better than separate evaluations. This idea can be generalized to other reasoning tasks—such as "forward solving" vs "backward problem construction" in mathematical reasoning.
- **Counterfactual Reasoning as a Causal Probe**: Unlike random fuzzing, requiring the model to identify necessary input changes through reasoning rather than trial-and-error directly tests causal understanding capabilities.
- **Counter-intuitive Discovery on Reasoning Post-training**: Post-training specialized for reasoning did not improve program execution reasoning; instead, it caused degradation, suggesting that current reasoning enhancement strategies may be overly biased toward specific reasoning patterns.

## Limitations & Future Work
- Only covers the Python language; extension to other languages requires building secure sandbox environments.
- Data are derived from public benchmarks, presenting potential risks of data contamination (though the design of counterfactual reasoning partially mitigates this).
- Selecting the branch with the largest coverage increment as the counterfactual target by default may bias toward easier-to-reason branches.
- Currently uses only code coverage as the observable behavior; future work could extend this to other execution attributes like program output and intermediate states.

## Related Work & Insights
- **vs CruxEval**: CruxEval pairs output prediction with input prediction, but evaluates each direction independently. DexBench jointly evaluates causal reasoning by sharing the execution context.
- **vs R-Eval (IC-Score)**: R-Eval combines IC-Score from multiple execution tasks but remains a single-path evaluation. DexBench introduces the counterfactual dimension.
- **vs CES**: CES evaluates reasoning consistency across multiple inputs but processes paths independently without exploring the causal logic leading to path divergence.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The duality framework is a fresh perspective for evaluating program reasoning and is exquisitely designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ 13 models, three complexity levels, and complete sensitivity analysis, although the data scale is relatively small (445 instances).
- Writing Quality: ⭐⭐⭐⭐⭐ Formal definitions are clear, and experimental analysis is in-depth.
- Value: ⭐⭐⭐⭐ Provides a more robust framework for evaluating code LLMs and reveals counter-intuitive phenomena regarding reasoning post-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Once Upon an Input: Reasoning via Per-Instance Program Synthesis](../../NeurIPS2025/code_intelligence/once_upon_an_input_reasoning_via_per-instance_program_synthesis.md)
- [\[ICML 2026\] BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](../../ICML2026/code_intelligence/boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_.md)
- [\[ACL 2026\] ReCode: Reinforcing Code Generation with Reasoning-Process Rewards](recode_reinforcing_code_generation_with_reasoning-process_rewards.md)
- [\[ACL 2026\] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)

</div>

<!-- RELATED:END -->
