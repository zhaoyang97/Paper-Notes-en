---
title: >-
  [Paper Note] Identifying the Achilles' Heel: An Iterative Method for Dynamically Uncovering Factual Errors in Large Language Models
description: >-
  [ACL 2026][LLM Evaluation][Knowledge Graph Testing] HalluHunter is a fully automated LLM factual error testing framework based on Knowledge Graphs (KG). It extracts factual triplets from Wikidata…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Knowledge Graph Testing"
  - "Factual Error Detection"
  - "Adaptive Question Generation"
  - "Multi-hop Reasoning"
  - "Data Contamination"
date: 2026-05-08
content_hash: 17a1698b29478d7b
---

# Identifying the Achilles' Heel: An Iterative Method for Dynamically Uncovering Factual Errors in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2401.00761](https://arxiv.org/abs/2401.00761)  
**Code**: <https://github.com/Mysterchan/HalluHunter>  
**Area**: LLM Evaluation / Factuality / Automated Testing  
**Keywords**: Knowledge Graph Testing, Factual Error Detection, Adaptive Question Generation, Multi-hop Reasoning, Data Contamination

## TL;DR
HalluHunter is a fully automated LLM factual error testing framework based on Knowledge Graphs (KG). It extracts factual triplets from Wikidata, generates three types of questions (Yes/No, Multiple Choice, and Question Answering) using rule-based templates, and supports multi-hop reasoning. Using an "Adaptive Iterative Algorithm" to select the next batch of difficult questions based on entity similarity and relationship accuracy from previous rounds, it reduces the accuracy of 9 mainstream LLMs by 32-42% over 5 iterations, triggering errors in up to 55% of questions and significantly outperforming static benchmarks.

## Background & Motivation

**Background**: Current mainstream LLM factuality evaluation paths include: (1) Static benchmarks (TruthfulQA, SimpleQA, LAMA, PopQA) with manual design or human annotation; (2) Semi-automatic QA generation (PAQ, KQA Pro); (3) Small-scale automatic KG-based evaluation (Head-to-Tail, DyKnow).

**Limitations of Prior Work** (summarized into four points in Table 1):
- **High manual cost**: Benchmarks like TruthfulQA and CommonsenseQA rely on manual design and annotation.
- **Data contamination**: Static evaluation sets are likely to have been included in LLM training data (OpenAI 2023 reports acknowledge GPT-4 was trained on web data), making evaluation results unreliable.
- **Limited coverage**: The LAMA series only tests a limited set of relations like "place of birth"; most benchmarks use only MC (Multiple Choice) formats and are biased toward specific topics.
- **Weak error exposure mechanisms**: Existing benchmarks are "one-shot," lacking mechanisms to target specific model weaknesses based on error patterns.

**Key Challenge**: To thoroughly test LLM factual weaknesses, a framework must simultaneously satisfy: (a) dynamic generation to avoid contamination, (b) extensive coverage across question types, and (c) the ability to locate weak areas based on model feedback. Blind random sampling on KGs addresses (a) and (b) but fails to hit specific weaknesses effectively.

**Goal**: Construct an automated framework to address these four limitations and verify its effectiveness in exposing LLM weaknesses compared to random sampling and existing benchmarks.

**Key Insight**: Treat "finding LLM factual errors" as a **search problem**—the KG is the search space, and weak relations or difficult entities are high-reward areas. Start with a random seed batch of questions and use LLM feedback to adaptively narrow down to "frequently failed relations + structurally similar entities."

**Core Idea**: KG-grounded rule-based generation combined with an adaptive iterative algorithm (selecting the next batch of triplets based on relation accuracy and entity embedding similarity). There is zero LLM intervention in question generation to prevent contamination.

## Method

### Overall Architecture
A four-stage pipeline:

1. **KG Construction**: User provides a topic (e.g., "occupation: emperor") → SPARQL query on Wikidata → Extract (SUBJECT, relation, OBJECT) triplets → Build directed graph $G = (V, E)$. Approx. 500k-600k triplets and 10k-12k entities per domain.
2. **Rule-based Question Generation** (No LLM involvement): Converts triplets into Yes/No, MC, and WH questions using POS + NER. Multi-hop questions are formed by chaining adjacent triplets, e.g., (Michelle Obama, spouse, Barack Obama) + (Barack Obama, educated at, Harvard) → "Where was Michelle Obama's spouse educated at?"
3. **Answer Evaluation**: Yes/No and MC use exact match; WH use sentence-transformer similarity (Table 8 shows this method achieves F1=87% and the highest recall at 97.9% among five methods).
4. **Adaptive Iterative Generation** (Algorithm 1): The core innovation. Calculates a rolling accuracy $R^{(l+1)}$ for each relation based on previous (question, answer, triplet). For each new question, use probability $e=0.2$ to explore (select low-accuracy relations where $R < a=0.4$), or exploit—if the previous answer was wrong, use QuatE embeddings to find top-$k=10$ similar entities for the same relation; if correct, select a new random triplet.

### Key Designs

1. **Rule-based No-LLM Question Generation (Avoids Contamination + Bias)**:
    - **Function**: Deterministically converts KG triplets into 3 question types (Yes/No, MC, WH), ensuring unique verifiable answers.
    - **Mechanism**: Yes/No questions use POS analysis to select auxiliary verbs ("is" for nouns, "does" for verbs) and create balanced "No" samples by replacing the object with an incorrect entity. MC uses NER to select interrogative words, with 1 correct and 3 incorrect options (from the same relation). WH questions strictly use triplets with a "single outgoing edge for the relation" (e.g., (China, capital, Beijing)) to ensure uniqueness. Multi-hop questions use $(s, \{r_1, r_2\}, o)$ chains.
    - **Design Motivation**: (a) Using LLMs for generation introduces self-bias, increases API costs, and risks duplicating training data. (b) Rule-based methods ensure reproducibility and controllability (98.5% semantic accuracy vs. 13% deviation in ChatGPT-generated questions). (c) Balanced Yes/No samples prevent sycophancy bias.

2. **Adaptive Iterative Generation Algorithm (Algorithm 1)**:
    - **Function**: Dynamically focuses on weak relations and similar entities based on LLM feedback, shifting from "broad sampling" to "precision targeting."
    - **Mechanism**: Maintains a relation-accuracy map $R^{(l)}(r)$ and used triplet set $T^{(l)}$. For the next batch: (i) Explore ($e=0.2$): Pick triplets from relations with $R(r) < 0.4$. (ii) Exploit: If the previous answer was incorrect ($c_i = \text{False}$), use QuatE-trained KG embeddings to find the top-10 similar entity set $C$ and query the same relation for subject $\in C$. (iii) If correct, pick a random new triplet.
    - **Design Motivation**: Factual errors are often clustered—if a model misses the atomic mass of Hydrogen, it likely misses Oxygen. Probing similar entities is more efficient than random sampling. Exploration constant $e=0.2$ prevents getting stuck in local minima.

3. **Weighted Coverage Metric (Group Degree Centrality)**:
    - **Function**: Measures whether the algorithm maintains KG coverage while increasing difficulty.
    - **Mechanism**: Treats the "queried entity set" $S$ as a node subset. Calculates normalized group degree centrality $\widehat{C}_{\deg}(S) = |N(S)| / (|V| - |S|) \in [0,1]$, where $N(S)$ is the open neighborhood. Higher values indicate better coverage of KG "hubs."
    - **Design Motivation**: Accuracy drops must not come at the cost of budget concentration in obscure corners. Trial 5 coverage (0.473) outperformed random sampling (0.406), confirming adaptive selection does not sacrifice coverage.

### Loss & Training
This is a **testing framework rather than a training method**; no LLM parameters are updated. The only "training" involves KG embeddings $\mathcal{M}$: QuatE is trained using the PyKEEN framework to generate entity embeddings for similarity search. Hyperparameters: $e=0.2$, $a=0.4$, $k=10$. Each domain and type undergoes 5 iterations of 1000 questions each. Total API cost was approximately $400 USD.

## Key Experimental Results

### Main Results (Median Accuracy after 5 Iterations for 9 LLMs across 3 Domains)

| Trial | Humanity Median Acc | Social Science | STEM |
|-------|---------------------|----------------|------|
| Seed (Trial 0) | 0.712 (0%) | 0.699 (0%) | 0.649 (0%) |
| Trial 1 | 0.542 (−19.5%) | 0.524 (−28.1%) | 0.478 (−24.8%) |
| Trial 2 | 0.516 (−24.1%) | 0.462 (−31.5%) | 0.428 (−31.7%) |
| Trial 3 | 0.492 (−29.2%) | 0.439 (−37.5%) | 0.406 (−36.1%) |
| Trial 5 | **0.462 (−32.7%)** | **0.384 (−40.2%)** | **0.373 (−41.8%)** |

By model: GPT-4o's accuracy in Humanity Yes/No dropped from 84.4% to 65.8%, and MC from 82.9% to 54.1%. WH questions consistently proved most difficult, with all models averaging just 37.4% in Trial 0. Multi-hop: Accuracy drops sharply from 1 to 2 hops (GPT-4o STEM MC dropped from 72.6% to 49.6%) and stabilizes/gradually declines up to 4 hops.

### Ablation Study (Sensitivity of Key Hyperparameters)

| Configuration | Trial 5 Accuracy | Trial 5 Coverage |
|------|------------------|-------------------|
| $e=0.2, a=0.3$ (Aggressive Exploit) | 0.371 | **0.417** (Low) |
| $e=0.2, a=0.4$ (Default) | **0.373** | 0.471 |
| $e=0.2, a=0.5$ (Relaxed Exploit) | 0.450 | 0.468 |
| $e=0.1, a=0.4$ (Less Explore) | 0.430 | 0.460 |
| $e=0.3, a=0.4$ (More Explore) | 0.412 | 0.472 |

**Trial 5 Coverage Comparison**: HalluHunter (0.473) > Random (0.406), proving that the iterative algorithm does not sacrifice KG coverage.

### Key Findings
- **Significant Iterative Effect**: Accuracy in the STEM domain dropped by 41.8% over 5 rounds, exposing far more errors than random sampling; linear regression results (p-value 0.031 for 1-hop, 0.01 for multi-hop) are statistically significant.
- **Difficulty Ranking (STEM > Social Science > Humanity)**: STEM showed the largest drop (−41.8%), while Humanity was the most stable (−32.7%), indicating LLMs are more fragile regarding precise technical knowledge.
- **GPT-4o Blind Spots**: Accuracy for "binding energy" was only 0.258, and "mass excess" 0.237, whereas "genetic association" reached 0.778, suggesting training data bias toward biomedicine.
- **Claude-3.5-Haiku Weakness**: Performance on "prime factor" was only 0.313, while Gemini-2.0 and GPT-4o both exceeded 0.60 on the same subject.
- **WH Questions are the Hardest**: Average accuracy across models was 37.4%; open-ended generation places higher demands on parametric knowledge than selection.
- **Multi-hop Amplification**: The sharpest drop occurs from 1 to 2 hops, suggesting that the initial reasoning step is the primary bottleneck.
- **Coverage Improvement**: Trial 5 coverage (0.473) exceeded Random (0.406), validating the $e=0.2$ exploration mechanism.

## Highlights & Insights
- **Application of Exploit-Explore**: Using KG embeddings to find similar entities for adaptive probing is a brilliant application of search paradigms to LLM testing. It treats "incorrect responses" as a reward signal and KG embeddings as a structural similarity metric.
- **Bypassing the LLM-loop Trap**: Unlike many recent "LLM-as-a-judge" or LLM-generation works that suffer from self-bias, HalluHunter uses rule-based KG generation to ensure results are credible.
- **Error Pattern Analysis**: Locating model-domain weaknesses (e.g., GPT-4o's biological vs. physical knowledge) provides fine-grained diagnostic data for model vendors to improve training sets.
- **Weighted Coverage**: Using group degree centrality offers a more logical measure of coverage than raw entity counts by reflecting real knowledge distribution hubs.
- **Engineering Detail**: The single-outgoing-edge constraint for multi-hop questions is crucial for enabling automated exact-match evaluation for WH questions.

## Limitations & Future Work
- **KG Dependency**: Relies entirely on Wikidata; errors or missing data in the KG propagate to the tests.
- **Diagnostic Only**: Provides no new mitigation methods for the identified hallucinations.
- **Simplistic Multi-hop**: Restricted to 2-4 hop "chains" rather than tree-like or circular structures found in real-world reasoning.
- **Evaluation Noise**: Sentence Transformer F1 (87%) implies some noise in judging WH questions.
- **Embedding Dependency**: Effectiveness relies on QuatE embedding quality; dynamic KGs would require frequent retraining.

## Related Work & Insights
- **vs. Head-to-Tail (2023)**: HalluHunter extends KG factuality testing from single-cloze formats to three question types and multi-hop reasoning, pushing GPT-4o WH accuracy from ~40% down to 10% in Trial 5.
- **vs. AutoDetect / Self-Challenge**: Unlike these LLM-driven adversarial probing methods, HalluHunter remains outside the LLM loop by using structured KGs.
- **Inspiration**: The "Adaptive Iteration + Structured Search + Reward-driven Exploit" paradigm can be transferred to other "Evaluation as Search" scenarios like code bug detection or safety testing.

## Rating
- Novelty: ⭐⭐⭐⭐ KG-grounded automation + adaptive iteration is a strong combination, though individual components (KG QA, QuatE) are established.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Massive scale (9 LLMs × 3 domains × 3 types × 6 trials) with detailed sensitivity and coverage analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative; Table 1 effectively highlights the limitations of 14 related works.
- Value: ⭐⭐⭐⭐ Fully automated and repeatable framework that avoids contamination; provides long-term value for LLM evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ACL 2026\] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)

</div>

<!-- RELATED:END -->
