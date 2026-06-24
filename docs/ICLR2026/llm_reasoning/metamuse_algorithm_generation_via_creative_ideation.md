---
title: >-
  [Paper Note] MetaMuse: Algorithm Generation via Creative Ideation
description: >-
  [ICLR 2026][Reasoning][Creative Ideation] To address the issue of LLMs being trapped by "availability bias" in classical heuristics (e.g., LRU/LFU) when generating system algorithms, MetaMuse proposes three self-reflection principles: measuring diversity in the performance feedback space, guiding via external stimuli rather than internal randomness, and implementation through waypoint reasoning instead of free-form CoT. This enables LLMs to perform "creative leaps" in discont…
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "Creative Ideation"
  - "Availability Bias"
  - "Self-reflection"
  - "Waypoint Reasoning"
  - "Cache Replacement"
  - "Online Bin Packing"
date: 2026-05-08
content_hash: 4c593dfb179792e6
---

# MetaMuse: Algorithm Generation via Creative Ideation

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=qJsShqQgWw](https://openreview.net/forum?id=qJsShqQgWw)  
**Code**: To be confirmed  
**Area**: LLM Reasoning / Algorithm Generation / System Algorithm Design  
**Keywords**: Creative Ideation, Availability Bias, Self-reflection, Waypoint Reasoning, Cache Replacement, Online Bin Packing  

## TL;DR
To address the issue of LLMs being trapped by "availability bias" in classical heuristics (e.g., LRU/LFU) when generating system algorithms, MetaMuse proposes three self-reflection principles: measuring diversity in the performance feedback space, guiding via external stimuli rather than internal randomness, and implementation through waypoint reasoning instead of free-form CoT. This enables LLMs to perform "creative leaps" in discontinuous solution spaces, reducing cache misses by up to 35.76% and bin usage by up to 30.93% on real cloud provider workloads.

## Background & Motivation

**Background**: System algorithms (e.g., cache replacement, bin packing for job scheduling) directly determine system performance, but manual design is extremely expensive—the authors' experience at a global cloud provider shows that even seemingly simple algorithms require tens of thousands of engineering hours. Consequently, engineers often settle for general heuristics like LRU, LFU, or First-Fit, which are sub-optimal. A natural idea is to use LLMs for automated algorithm generation.

**Limitations of Prior Work**: The solution space for system algorithms is **discontinuous**—minor changes in data structures or control flow can cause drastic non-linear performance shifts, leaving no smooth landscape for searching. The authors observed after repeated sampling of GPT-4o, Llama3.3-70B, and DeepSeek-V3 that LLM-generated solutions cluster heavily around well-known heuristics (LRU/LFU/FIFO). This is due to **availability bias** inherent in next-token prediction, where models favor sequences frequently seen in training data.

**Key Challenge**: While the goal is to use LLMs to "leap" and explore new designs in discontinuous spaces, the LLM’s generative mechanism pulls it toward high-frequency classical solutions. The authors prove that this bias **cannot be solved by adjusting temperature**: increasing temperature only performs monotonic smoothing of the softmax distribution, preserving the relative ranking of candidate tokens. Extreme temperatures lead to incoherence rather than breaking the bias. Furthermore, making the LLM perform mutations on existing solutions often results in repetitive fine-tuning of scoring functions while ignoring other dimensions like data structures or hierarchical architectures.

**Goal**: Formalize algorithm generation as a **sampling process** over the solution space—each step generates a different executable solution, accumulating "diverse and useful" solutions to find the optimal. Achieving this requires a self-reflection framework that examines "what solutions have been generated" to decide "how to generate the next."

**Key Insight**: **External Stimuli-Driven Self-reflection**—MetaMuse actively invokes knowledge the LLM "assumes is irrelevant" to the problem (e.g., random English words), forcing it to map these stimuli to the problem domain. This creates leaps beyond classical solutions, using performance feedback rather than semantic embeddings to truly distinguish solution differences.

## Method

### Overall Architecture
MetaMuse is an iterative creative ideation framework where each iteration outputs an executable solution. An iteration consists of three steps: evaluating the diversity of existing solutions in the **performance feedback space** (§3.1), selecting a set of **external stimuli** (keywords) to guide the ideation direction (§3.2), and using **waypoint reasoning** to develop seemingly unrelated stimuli into executable Python code (§3.3). These steps correspond to three model-agnostic self-reflection principles.

```mermaid
flowchart LR
    A[Generated Solution Set] --> B[1. Evaluate Diversity<br/>Performance Feedback Embedding]
    B --> C[2. External Stimuli Guidance<br/>RSDict / RSDict-SF]
    C --> D[3. Waypoint Reasoning Implementation]
    D --> E1[Attribute Extraction] --> E2[Problem Mapping] --> E3[Solution Construction] --> E4[Code Generation]
    E4 --> F[Executable Solution + Feedback]
    F -.Evaluate hit ratio.-> A
```

### Key Designs

**1. Measuring diversity in the performance feedback space: Replacing "description difference" with "result difference"**. MetaMuse represents each solution as a **feedback embedding** $[p_1, p_2, \dots, p_n]$, where $p_i$ is the performance metric (e.g., cache hit rate) on the $i$-th workload trace; $n=30$ traces are synthesized via libCacheSim using various Zipfian/Weibull distributions. This is because **semantic difference does not imply functional difference**: GPT-4o has generated equivalent LFU logic from two different descriptions ("evict least accessed" vs. "evict lowest priority with priority+1 on hit"). Feedback embeddings provide two benefits: each dimension is a quantitative metric, and the Euclidean distance between embeddings directly reflects solution differences (the two LFUs would have a distance of 0). The metrics are naturally bounded (e.g., 0–100%), facilitating the calculation of "guidance directions."

**2. Guiding ideation with external stimuli rather than internal randomness**. MetaMuse avoids internal model randomness (temperature) and uses **external stimuli** as starting points. These stimuli (English keywords) may be unrelated to the problem, forcing the LLM to associate "probabilistically unrelated" knowledge. Implementation involves 2,899 common words. Selecting stimuli uses two strategies: **RSDict** is a stateless random sampling strategy; **RSDict-SF** utilizes feedback embeddings of historical solutions to calculate a **guidance direction** (target feedback embedding). In exploration, it targets points furthest from existing solutions; in exploitation, it sets dimensions to high values (e.g., 100% hit rate). RSDict-SF models the selection as a prediction problem, using $n$ Gaussian Process Regressions (GPR) to predict feedback embeddings for stimuli sets. GPR uses a dot-product kernel invariant to stimuli order:

$$K(c_i, c_j) \propto \sum_{p=1}^{s}\sum_{q=1}^{s} \phi(o_{i,p})^\top \phi(o_{j,q})$$

where $o_{i,p}$ is a natural language observation sentence mapping the $p$-th stimulus to the domain, and $\phi(\cdot)$ is its SBERT semantic embedding. Given the $3000^s$ combinations, MetaMuse uses **power-of-two random choices** to select the stimuli set closest to the target.

**3. Implementing waypoint reasoning to ground irrelevant stimuli into executable solutions**. Letting an LLM write code directly from keywords often results in the words being used superficially as variable names. MetaMuse sets four **waypoints** as intermediate checkpoints to force structured transformation: **Attribute Extraction** (associating "flower" with "a ring of petals") → **Problem Mapping** (mapping "ring of petals" to a circular buffer) → **Solution Construction** (synthesizing observations into a complete description) → **Code Generation** (translating description to Python). This prevents superficial processing and is a key source of diversity.

## Key Experimental Results

Tests were conducted on two high-value cloud provider problems: Cache Replacement (96 traces: RetrievalAttention / Tencent Block / Aliyun) and Online Bin Packing (288 traces: BPPLIB / Weibull / Gaussian). Each method generated 350 executable solutions, compared against 5 LLM baselines and 16 manual heuristics (21 total).

### Main Results: High Performance Solutions

| Task | Model | vs. LLM Baseline | vs. Manual Heuristics |
|------|------|------------------|-----------------------|
| Cache (90th percentile trace) | GPT-4o | 5.17%–9.89% lower miss | 1.75%–13.03% lower |
| Cache (75th percentile trace) | GPT-4o | 3.62%–6.39% lower | 6.62%–**35.76%** lower |
| Bin Packing (90th percentile) | GPT-4o | 9.25%–9.42% lower usage | 9.25%–20.59% lower |
| Bin Packing (90th percentile) | DeepSeek-V3 | Up to **21.06%** lower | Up to **30.93%** lower |

MetaMuse yields higher performance gains across almost all percentiles and three different LLMs.

### Ablation Study

| Configuration | Distinct solutions (GPT-4o, Cache) |
|------|------|
| RSDict | 149 |
| RSDict + Waypoint Reasoning | 175 |
| RSDict-SF (no Waypoints / noWR) | 152 |
| RSDict-SF + Waypoint Reasoning (Full) | **197** |

- **Stimuli guidance is effective**: On the 90th percentile trace, 67.20% of RSDict solutions and 75.60% of RSDict-SF solutions outperformed Repeated Sampling.
- **Waypoint reasoning is effective**: Removing waypoints significantly reduces distinct solutions (e.g., 197 → 152 for RSDict-SF).

### Key Findings
- **Temperature is not a cure**: Monotonic smoothing preserves token ordering and fails to escape availability bias.
- MetaMuse yields designs humans might not reconsider: MetaMuse-533 uses NSE counters to track how many eviction events an object has survived (favoring long-resident objects to prevent thrashing) and saturated counters to handle bursty traffic.

## Highlights & Insights
- **Grounding "creativity" in measurable feedback**: Defining diversity through performance vectors rather than semantic embeddings avoids "re-skinned LFU" pseudo-diversity.
- **Synergy between external stimuli and waypoint reasoning**: Stimuli enable "jumping out" (breaking bias) while waypoints ensure "landing" (preventing superficial keyword usage).
- **Engineering efficiency**: GPR and power-of-two random choices allow pre-screening of stimuli without generating full solutions, keeping costs at approximately 2 cents per solution.

## Limitations & Future Work
- Validated only on two system tasks; generalizability to complex designs (scheduling, routing) remains unknown.
- Diversity in the feedback space depends on fast simulation environments; without cheap simulators, the method reverts to stateless RSDict.
- Stimuli selection remains somewhat heuristic; technical terms might be misused as simple variable names.

## Related Work & Insights
- **Comparison with Automated Heuristic Design**: Existing methods (FunSearch, ReEvo, etc.) mostly use evolution-based mutation/crossover, which is limited by the initial population and remains susceptible to availability bias. MetaMuse leaps directly in the solution space using external stimuli.
- **Insight**: Availability bias is a fundamental barrier to LLMs in discovery tasks. Grounding diversity in a measurable feedback space and using structured waypoint constraints is a generalizable recipe for generative discovery (e.g., molecule or circuit design).

## Rating
- **Novelty**: ⭐⭐⭐⭐ Formalizing algorithm generation as sampling in discontinuous spaces and using stimuli to break bias is novel and well-supported.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Uses real production workloads and covers 21 baselines. Slightly docked as it only covers two classes of algorithms.
- **Writing Quality**: ⭐⭐⭐⭐ Logic is clear, and the mapping between principles and method steps is robust.
- **Value**: ⭐⭐⭐⭐ Significant gains over manual heuristics in real-world scenarios at a low cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Exposing Weaknesses of Large Reasoning Models through Graph Algorithm Problems](exposing_weaknesses_of_large_reasoning_models_through_graph_algorithm_problems.md)
- [\[ICLR 2026\] Reverse-Engineered Reasoning for Open-Ended Generation](reverse-engineered_reasoning_for_open-ended_generation.md)
- [\[ICLR 2026\] String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation](string_seed_of_thought_prompting_llms_for_distribution-faithful_and_diverse_gene.md)
- [\[ICLR 2026\] Log-Augmented Generation: Scaling Test-Time Reasoning with Reusable Computation](log-augmented_generation_scaling_test-time_reasoning_with_reusable_computation.md)
- [\[ICLR 2026\] Unleashing Scientific Reasoning for Bio-Experimental Protocol Generation via Structured Component-based Reward Mechanism](unleashing_scientific_reasoning_for_bio-experimental_protocol_generation_via_str.md)

</div>

<!-- RELATED:END -->
