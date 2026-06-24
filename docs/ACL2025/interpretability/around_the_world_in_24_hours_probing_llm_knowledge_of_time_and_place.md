---
title: >-
  [Paper Note] Around the World in 24 Hours: Probing LLM Knowledge of Time and Place
description: >-
  [ACL2025][Interpretability][Spatiotemporal reasoning] This paper presents the GeoTemp dataset (320k prompts covering 289 cities and 37 time zones) to evaluate the capability of LLMs in joint temporal and spatial reasoning for the first time. The study finds that models can handle time calculation and geographic knowledge independently, but their performance drops sharply when combining both is required.
tags:
  - "ACL2025"
  - "Interpretability"
  - "Spatiotemporal reasoning"
  - "geographic knowledge"
  - "temporal reasoning"
  - "LLM evaluation"
  - "benchmark"
date: 2026-05-08
content_hash: ce13a8515e2a85ca
---

# Around the World in 24 Hours: Probing LLM Knowledge of Time and Place

**Conference**: ACL2025  
**arXiv**: [2506.03984](https://arxiv.org/abs/2506.03984)  
**Code**: [UhhDS/GeoTemp](https://github.com/UhhDS/GeoTemp)  
**Area**: Interpretability  
**Keywords**: Spatiotemporal reasoning, geographic knowledge, temporal reasoning, LLM evaluation, benchmark  

## TL;DR

This paper presents the GeoTemp dataset (320k prompts covering 289 cities and 37 time zones) to evaluate the capability of LLMs in joint temporal and spatial reasoning for the first time. The study finds that models can handle time calculation and geographic knowledge independently, but their performance drops sharply when combining both is required.

## Background & Motivation

Temporal and spatial reasoning are fundamental capabilities for understanding the world. In globalized working environments, tasks such as cross-border logistics planning and business travel arrangement require incorporating both temporal and geographic information into the reasoning process.

Prior works have separately studied the temporal reasoning capabilities of LLMs (e.g., TempLama, McTaco) and spatial understanding capabilities (e.g., WorldBench), but few have evaluated their ability to perform **joint temporal and spatial reasoning**. The only two related works (TRAM and TOT) either heavily guide the models using a multiple-choice format or use completely synthetic environments lacking real-world location information.

This paper uses **global time zones** as the test scenario to systematically evaluate LLMs' reasoning capabilities under different combinations of temporal and geographic knowledge for the first time.

## Method

### GeoTemp Dataset Construction

#### Step 1: Collecting Time Zones and Locations
- Extract time zone information from the Olson Time Zone Database (OTZD).
- Filter out time zones representing entire regions or abandoned ones.
- Retrieve city-level data (population, latitude, and longitude) via the Opendatasoft API.
- Ultimately select 289 locations covering 37 UTC time zones and 217 ISO country codes.

#### Step 2: Designing Four Task Templates

| Task | Template | Knowledge Required |
|------|----------|--------------------|
| **Verification** | What time is it now in $l_1$? | Basic understanding (echoing time) |
| **TimeTime** | What time will it be in x hours? | Time calculation only |
| **TimePlace** | What time is it now in $l_2$? | Time + Geographic knowledge |
| **TimeTimePlace** | What time will it be in $l_2$ in x hours? | Time calculation + Geographic knowledge |

Each prompt is prefixed with: "Today is {Time&Date} in {$l_1$}"

#### Step 3: Combining the Test Set
- Generate the Cartesian product $l_1 \times l_2$ of all location pairs.
- Apply all task templates to each combination.
- Randomly select times and dates within the year 2023.
- Ultimately generate **332,928** test prompts.

### Dataset Characteristics
- Covers all continents (including Antarctica and several islands).
- 6%+ of the locations have a population $\le 500$ (e.g., Atikokan, Canada).
- 50% of the cities have a population $\le 500,000$.

### Evaluation Protocol

#### Models
Evaluate 8 open-source chat models: Llama2-Chat (7B/13B/70B), Llama3-Instruct (8B/70B), Qwen2-Instruct (1.5B/7B/72B).

#### Instruction Types
- **Neutral**: No extra guidance.
- **CoT**: Appended with "Think step by step."
- **Short**: Appended with "Keep your answer short..."

#### Evaluation Method
Use a custom regex algorithm to extract dates and times from open-ended responses. The accuracy on the validation set is $\ge 98\%$, which is much more efficient than using an LLM-as-judge.

## Key Experimental Results

### Main Results

Performance of the best model, Llama3-70B:
- **Verification**: ~95%
- **TimeTime**: ~99%
- **TimePlace**: 56.1%
- **TimeTimePlace**: 25.4% (Highest accuracy!)

**Key Findings**: Most models perform well on tasks involving only temporal knowledge, but their performance drops significantly when both temporal and geographic information need to be integrated.

### Model Scaling Effects

- Llama3 from 8B to 70B: TimePlace improves by 29.3%, TimeTimePlace improves by 23.0%.
- The improvement from scaling Qwen2 is very limited (TimePlace ultimately reaches only 18.0%).
- The improvement for Llama2 is even less pronounced.
- Scaling alone may not solve the joint spatiotemporal reasoning problem.

### Impact of Instruction Types

- **Short instructions**: Performance improves on simple tasks and decreases on complex tasks.
- **CoT instructions**: Unexpectedly decrease performance on simple tasks (Llama2-70B drops significantly on Verification).
- Qualitative analysis reveals that under CoT, the model attempts to solve problems harder than they actually are, falling into contradictions within its own explanations.

### Geographic Bias Analysis

- No obvious bias toward Western countries is **found**.
- Llama3-70B performs best on African countries but worse on North America and Oceania.
- No clear patterns emerge when aggregating by continent, population, or income level.

### Location Name Perplexity Analysis

| Perplexity Combination | Llama3-70B Accuracy |
|-------------------|-------------------|
| Very Low × Very Low | ~53.9% |
| Very High × Very High | ~29.9% |

The performance gap between low-perplexity and high-perplexity combinations reaches **22.5%**. This indicates that the model's performance is biased toward locations that appear frequently in the training data, rather than being biased toward Western countries.

### Direct Time Zone Knowledge Probing

| Model | Accuracy |
|------|--------|
| Llama3-70B | **90.0%** |
| Llama3-8B | 84.1% |
| Qwen2-72B | 86.3% |
| Qwen2-1.5B | 39.3% |

When **directly asked** about the time zone of a specific location, most models can answer correctly 65%+ of the time (Llama3-70B reaches 90%). However, they fail when required to use this knowledge in combination — **the knowledge exists but cannot be effectively retrieved and combined**.

### Error Analysis (Llama3-70B, 200 error samples)

| Error Type | Proportion |
|---------|------|
| DST/UTC conversion error | 25.3% |
| Time difference calculation error (Correct UTC) | 22.3% |
| Incorrect UTC knowledge for at least one location | **48.2%** |

### Knowledge Injection Experiment

| Setting | Llama3-70B Accuracy |
|------|-------------------|
| Original | ~33.4% |
| Added time zone information | **76.3%** |
| Replaced city name with time zone only | ~65% |

Injecting geographic knowledge dramatically improves performance. However, the "replace" method is inferior to the "add" method, indicating that the model does indeed utilize its geographical knowledge to assist in reasoning.

## Highlights & Insights

1. **Knowledge exists but cannot be combined**: This is the core finding — the models "know" time calculation and time zone knowledge separately, but fail when they need to use both concurrently. This points to a fundamental limitation of LLMs: the retrieval and integration of knowledge in complex reasoning.
2. **CoT is harmful to simple tasks**: This aligns with the findings of Sprague et al. (CoT is mainly effective in mathematical and symbolic reasoning) and may introduce distraction for tasks that do not require deep reasoning.
3. **Perplexity explains performance disparity better than geographic regions**: Contrary to the expected "Western bias", the frequency of location occurrences in the training data is the key factor.
4. **Elegant dataset design**: Through four levels of tasks with progressively increasing complexity, the study precisely pinpointed the failure nodes of the models.

## Limitations & Future Work

1. The regex evaluation algorithm has high accuracy ($\ge 98\%$), but a small amount of noise still exists.
2. Robustness checks are only conducted on a subset.
3. Pre-training data of the models is inaccessible, making the use of perplexity as a proxy for frequency not entirely reliable.
4. Purely geographic tasks (e.g., predicting location) are not covered, but this overlaps with existing work.
5. Only open-source models are evaluated, lacking a comprehensive evaluation of closed-source models like GPT-4 (gpt4o-mini was only tested in robustness experiments).

## Related Work & Insights

- **Temporal knowledge testing**: TimeBank, TempLama, McTaco, etc., focus on different aspects of temporal reasoning.
- **Geographic knowledge testing**: GeoLLM, WorldBench, etc., evaluate LLMs' geographic knowledge and biases.
- **Spatiotemporal knowledge testing**: TRAM (multiple-choice format, limited analysis) and TOT (synthetic data, no real-world locations).

## Rating

⭐⭐⭐⭐ — The research problem is novel and has practical significance, the dataset is elegantly designed, and the analysis is deep and systematic. The core finding (knowledge exists but cannot be combined) has broad implications. The limitations lie in the relatively restricted scope of evaluated models and task types, and the exploration of solutions is not sufficiently deep.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MINED: Probing and Updating with Multimodal Time-Sensitive Knowledge for Large Multimodal Models](../../ACL2026/interpretability/mined_probing_and_updating_with_multimodal_time-sensitive_knowledge_for_large_mu.md)
- [\[NeurIPS 2025\] LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning](../../NeurIPS2025/interpretability/llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)
- [\[NeurIPS 2025\] LLM Probing with Contrastive Eigenproblems: Improving Understanding and Applicability of CCS](../../NeurIPS2025/interpretability/llm_probing_with_contrastive_eigenproblems_improving_understanding_and_applicabi.md)
- [\[ACL 2025\] Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models](degenerate_knowledge_neurons.md)
- [\[ACL 2025\] Probing Subphonemes in Morphology Models](probing_subphonemes_in_morphology_models.md)

</div>

<!-- RELATED:END -->
