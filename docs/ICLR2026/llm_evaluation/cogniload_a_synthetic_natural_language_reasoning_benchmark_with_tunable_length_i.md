---
title: >-
  [Paper Note] CogniLoad: A Synthetic Natural Language Reasoning Benchmark With Tunable Length, Intrinsic Difficulty, and Distractor Density
description: >-
  [ICLR 2026][LLM Evaluation][Paper Note] CogniLoad is a synthetic natural language reasoning benchmark built on Cognitive Load Theory (CLT). It employs three independent and tunable parameters—intrinsic difficulty $d$, distractor density $\rho$, and task length $N$—to manipulate intrinsic, extraneous, and germane cognitive loads (the latter represented as mai
tags:
  - ICLR 2026
  - LLM Evaluation
date: 2026-05-08
content_hash: c7e793f944f25942
---
# CogniLoad: A Synthetic Natural Language Reasoning Benchmark With Tunable Length, Intrinsic Difficulty, and Distractor Density

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=0Sex2H5Jnn](https://openreview.net/forum?id=0Sex2H5Jnn)  
**Code**: https://cogniload.dk.fo (Project homepage including generation code)  
**Area**: LLM Evaluation / Long-context Reasoning / Synthetic benchmark  
**Keywords**: Cognitive Load Theory, Long-context Reasoning, Logic Puzzles, Controllable Synthetic Data, Failure Attribution

## TL;DR
CogniLoad is a synthetic natural language reasoning benchmark built on Cognitive Load Theory (CLT). It employs three independent and tunable parameters—intrinsic difficulty $d$, distractor density $\rho$, and task length $N$—to manipulate intrinsic, extraneous, and germane cognitive loads (the latter represented as maintenance burden). This allows for precise attribution of long-context reasoning failures to specific dimensions. Evaluating 22 SotA reasoning models revealed that task length is the primary bottleneck and models exhibit a U-shaped response to distractors.

## Background & Motivation
**Background**: Long-context reasoning serves as a critical battlefield for evaluating LLMs, leading to benchmarks such as LongBench, L-Eval, BABILong, RULER, Needle-in-a-Haystack (NIAH), and LogicBench. These benchmarks exert pressure from various angles: increasing context length, deepening reasoning depth, or inserting "needles" (distractors) into massive volumes of irrelevant text.

**Limitations of Prior Work**: A significant issue is that these dimensions are **entangled**. When a model's performance drops on a long-context task, it is unclear whether the failure is due to "context length exceeding capacity," "intrinsic difficulty of single-step reasoning," or "distraction by irrelevant information." For example, LongBench/L-Eval may change length without necessarily altering reasoning depth; LogicBench focuses on intrinsic difficulty but lacks distractors; BABILong couples multi-step reasoning with a fixed distractor ratio. Consequently, **existing benchmarks provide aggregate scores rather than diagnostic signals**.

**Key Challenge**: Failures in long-context reasoning may stem from multiple fundamentally different cognitive mechanisms, yet current evaluations entangle these mechanisms in a single dimension, preventing precise failure attribution. Decoupling requires a controllable generator capable of **orthogonally and independently** adjusting each dimension.

**Goal**: To construct a benchmark that enables independent control of three types of load, scales to arbitrary context lengths, remains resistant to data contamination (via procedural random generation), and provides interpretable capacity thresholds for each model across every dimension.

**Key Insight**: The authors draw from **Cognitive Load Theory (CLT, Sweller 1988)**. CLT categorizes working memory load into three types: Intrinsic Cognitive Load (ICL, from element interactivity), Extraneous Cognitive Load (ECL, from irrelevant elements or poor presentation), and Germane Cognitive Load (GCL, resources dedicated to building and maintaining schemas). The authors argue that the "computational resource pressure" faced by LLMs during reasoning maps directly to these three loads, making CLT a blueprint for benchmark design.

**Core Idea**: Translate the three load dimensions of CLT into three **independently adjustable** parameters in logic-grid puzzles—using intrinsic difficulty $d$ for ICL, needle-to-hay ratio $\rho$ for ECL, and task length $N$ as an operational proxy for GCL. This transforms the vague question of "why long-context reasoning fails" into a diagnostic framework suitable for factorial control.

## Method

### Overall Architecture
CogniLoad consists of a family of **natural language logic-grid puzzles**. Each puzzle describes several "people," each with a set of **variable attributes** (e.g., sock color, glove color, favorite music). The task begins with an initial state, followed by a sequence of **strictly ordered** update statements (e.g., "The person wearing green socks changed their music to electronic"). Finally, a question is asked regarding a specific attribute of a randomly selected "Person of Interest" (PoI). The model must function like a state machine, applying rules sequentially to track the PoI's state vector and report the final answer.

The puzzle difficulty is characterized by three parameters: intrinsic difficulty $d$, total statements $N$, and needle-to-hay ratio $\rho$. These correspond to the three loads of CLT and are designed to be **mutually orthogonal**—modifying one does not affect the "unit intensity" of the others. Statements are transcribed from logical forms to natural language using deterministic string templates (independent of LLM generation), ensuring reproducibility, large-scale sampling, and resistance to training set contamination.

The generation pipeline involves: selecting attributes and characters → initializing distinct states → generating statements (where $\rho$ determines if a step is a "needle" or "hay") → performing validity checks for each statement → posing a question about a PoI attribute after $N$ statements.

### Key Designs

**1. CLT 3D Disentanglement: Decomposing reasoning failure into three independent loads**

This design targets the limitation of existing benchmarks where dimensions are entangled. CLT loads are mapped to puzzle attributes: ICL maps to **intrinsic difficulty $d$**, controlling element interactivity in the reasoning chain; ECL maps to irrelevant elements handled by the **needle-to-hay ratio $\rho$**; and GCL maps to the effort of maintaining a schema during long reasoning. Since GCL is internal to the learner, the authors use **task length $N$** as its operational proxy. Increasing $N$ requires more consecutive updates to the PoI state vector while keeping $d$ and $\rho$ constant, isolating the requirement for "continuous, constructive schema maintenance."

This transforms an aggregate score into a **factorized diagnostic coordinate system**, allowing researchers to determine if a model fails due to length or difficulty by sweeping $N$ while fixing $d$ and $\rho$.

**2. Randomized Logic Puzzle Generation Algorithm: Needle/hay statements + strict validity checks**

To ensure $d, \rho, N$ are controllable and puzzles are "non-trivially solvable," the generation process is rigorous. A set of people $P$ ($|P|=\max(d,2)$), attribute categories $A$ ($|V_c|=\max(d+1,3)$), and a PoI $p^*$ are selected. Initial states ensure all individuals are distinguishable.

For each step $t=1\dots N$, a statement is generated: the probability $P(T_t=\text{needle})=n^t_{\text{needle}}/(N-t)$ determines whether the step updates the PoI (needle) or others (hay). Each statement's logical form is a conditional state transition:

$$\forall p \in P:\ \Big(\bigwedge_{c\in C_t} S_{t-1}(p,c)=v_{c,t}\Big)\ \Rightarrow\ \Big(\bigwedge_{c\in U_t} S_t(p,c)=u_{c,t}\Big),$$

where condition count $k_t$ and update count $m_t$ are sampled from $\mathrm{Uniform}\{1,\dots,d\}$. **Validity checks** ensure "hay" statements do not affect the PoI and do not cause state collapse, while "needle" statements must affect the PoI without making the entire population identical. These constraints ensure distractors are structurally similar to needles rather than obvious noise.

**3. Three orhtogonal parameters: $d$, $N$, $\rho$ managing distinct loads**

**Intrinsic difficulty $d\in\{1,3,5,7,10\}$** scales the state space ($\approx (d+1)^d$), person-attribute interactions, and rule complexity. **Task length $N\in\{20,50,100,250\}$** increases the steps of state transitions without changing per-step complexity. **Needle-to-hay ratio $\rho\in\{5,\dots,95\}\%$** regulates irrelevant content; smaller $\rho$ means more distractors and higher ECL.

**4. Load Sensitivity Regression and Capacity Thresholds: Interpretable performance metrics**

To provide a comparable capacity metric, a binomial GLM (logit link) is fitted for each model:

$$\Pr(Y=1)=\sigma\big(\beta_0+\beta_d\,d+\beta_N\log_{10}N+\beta_\rho\,\rho+\beta_{\rho^2}\,\rho^2\big),$$

where $\beta_d, \beta_N, \beta_\rho$ help quantify sensitivity to ICL, GCL, and ECL. The quadratic term for $\rho$ captures the observed U-shaped response. By solving for $\Pr=0.5$, three thresholds are derived: **ECL50** (max statements handled at 50% accuracy), **NT50** (min needle percentage required), and **ID50** (max intrinsic difficulty). For example, $\mathrm{ECL50}=10^{-(\beta_0+\beta_d\bar d+\beta_\rho\bar\rho+\beta_{\rho^2}\bar\rho^2)/\beta_N}$.

### Example
For a puzzle with $d=3,\ N=20,\ \rho=50\%$: The prompt provides initial states (e.g., "Brent wears green socks..."), followed by 20 ordered update rules (e.g., "1. Green sock wearers switch music to electronic..."), and asks a final question ("What color socks is Brent wearing?"). The model must track the attributes of the PoI (Brent) through 20 steps. Evaluation uses progressive exact match to tolerate minor phrasing shifts while maintaining determinism.

## Key Experimental Results

### Main Results
22 SotA models were evaluated. 13 open-weight models were tested on 100 puzzles per $(d, N, \rho)$ configuration (14,000 total per model). Proprietary models (Gemini-1.5, gpt-4o, DeepSeek-R1-0125) were tested on 10 puzzles per configuration (1,400 total) due to cost.

Trends observed across dimensions:

| Load Dimension | Observations | Representative Data |
|----------|------|----------|
| Intrinsic Difficulty $d$ (ICL) | Accuracy monotonically decreases with $d$. | gpt-4o: $d{=}1$→$d{=}10$ drops from 1.00→0.82; o1-preview: 0.96→0.80. |
| Task Length $N$ (GCL Proxy) | **The dominant bottleneck**; sharpest drops occur at $N{=}20$→$50$. | DS-Llama-70B: 0.89→0.66; $N{=}250$ results in only gpt-4o(0.76) and o1(0.68) exceeding 50%. |
| Needle-to-hay $\rho$ (ECL) | **U-shaped response**, accuracy lowest at $\rho\in[25,50]\%$. | gemini-1.5-flash: 0.38→0.53; gpt-4o remains stable at 0.97→0.89→0.91. |

The U-shape results from two opposing effects: increasing $\rho$ reduces distractors (lowering filter difficulty) but increases PoI state transitions (increasing tracking burden). Models with $\Delta_\rho > 0$ are considered noise-resistant, while $\Delta_\rho < 0$ indicates sensitivity to distractors.

### Capacity Thresholds and Regression
GLM fitting identified model tiers:

| Capacity Tier | Models | Characteristics |
|--------|----------|------|
| Frontier/High | gpt-4o, o1 (ECL50 > 300), gemini-1.5-pro, R1-0125 | Superior long-context handling. |
| Mid-tier | DS-Llama-70B, Qwen2.5-32B, QwQ-32B, Phi-4-reasoning | Strong performance at moderate $N, d$. |
| Low-tier | DS-Qwen-7B, Phi-4-mini, Qwen2.5-1.5B | Fail to reach 50% under average load. |

Regression findings:
- **$\beta_d, \beta_N$ were significantly negative across all models**, confirming increased ICL/GCL always degrades performance.
- **Super-additive "Difficulty × Length" Coupling**: Negative $d\times N$ interaction was significant in 17/22 models, showing that combining difficulty and length is harder than their sum.
- **Difficulty amplifies distractor harm**: Negative $d\times\rho$ interaction suggests high intrinsic complexity makes models more prone to distraction.

### Key Findings
- **Task length is the primary failure driver**: $N=250$ resulted in 0.20–0.30 accuracy for most models.
- **State tracking errors dominate**: Error analysis shows the most common failure is incorrectly attributing attributes to the PoI in the final step, rather than formatting errors.
- **Context overflow is model-specific**: Gemini-1.5 faced context budget hits at $N=250$ due to verbose outputs, while gpt-4o and o1 did not.
- **Format drift persists in small models**: Logical errors and formatting issues increased significantly with $N$ and $d$ for compact models like Phi-4-mini.

## Highlights & Insights
- **Theory-driven benchmark design**: Mapping CLT loads to orthogonal parameters provides a clear theoretical foundation for failure attribution.
- **Orthogonal and procedural generation**: Independent parameters combined with template-based generation solve data contamination and scaling issues.
- **Interpretable capacity thresholds**: Translating accuracy curves into single-number metrics (ECL50/NT50/ID50) allows for effective model ranking beyond aggregate scores.
- **U-shaped response analysis**: Identifying the balance between distractor filtering and state update frequency provides deep insight into model reasoning mechanisms.

## Limitations & Future Work
- **GCL as a proxy**: $N$ acts as an operational proxy for GCL but may also capture serving-related context length limits (e.g., context window overflow).
- **Narrow puzzle scope**: The benchmark is limited to logic-grid state machine puzzles; its generalizability to other reasoning types (math, code) remains to be seen.
- **Sparse proprietary sampling**: Due to costs, proprietary models were sampled less densely, leading to wider confidence intervals.
- **Exact match evaluation**: High sensitivity to formatting may penalize small models that reason correctly but exhibit format drift.

## Related Work & Insights
- **vs Long-context Benchmarks (LongBench, RULER)**: These benchmarks often entangle length and difficulty; CogniLoad allows for clear attribution.
- **vs Logic Benchmarks (LogicBench, ZebraLogic)**: These focus on ICL but lack the length and distractor dimensions of CogniLoad.
- **vs Needle-in-a-Haystack (NIAH)**: NIAH tests simple retrieval; CogniLoad requires complex state tracking with distractors that are structurally similar to needles.
- **vs GSM-$\infty$**: While both use parameters to control noise and difficulty, CogniLoad uniquely ensures these parameters are orthogonal to task length.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses CLT to decouple long-context reasoning into three orthogonal dimensions; a rare cross-disciplinary perspective in benchmark design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated 22 models with tens of thousands of samples, GLM regression, and detailed error analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and methodology; however, high density of CLT terminology and heavy reliance on appendices may be challenging for some readers.
- Value: ⭐⭐⭐⭐⭐ Provides a replicable, anti-contamination diagnostic tool that is highly practical for guiding model improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] BelarusianGLUE: Towards a Natural Language Understanding Benchmark for Belarusian](../../ACL2025/llm_evaluation/belarusian_glue.md)
- [\[ACL 2025\] MisMatched: A Benchmark for Scientific Natural Language Inference](../../ACL2025/llm_evaluation/a_mismatched_benchmark_for_scientific_natural_language_inference.md)
- [\[ICLR 2026\] Reliable Fine-Grained Evaluation of Natural Language Math Proofs](reliable_fine-grained_evaluation_of_natural_language_math_proofs.md)
- [\[ACL 2025\] MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated with Knowledge Guidance](../../ACL2025/llm_evaluation/mdbench_a_synthetic_multi-document_reasoning_benchmark_generated_with_knowledge_.md)
- [\[ICLR 2026\] EIP: Weighted Ranking of LLMs by Quantifying Question Difficulty](eip_weighted_ranking_of_llms_by_quantifying_question_difficulty.md)

</div>

<!-- RELATED:END -->
