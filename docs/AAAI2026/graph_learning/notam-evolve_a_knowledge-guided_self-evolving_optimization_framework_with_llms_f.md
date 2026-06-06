---
title: >-
  [Paper Note] NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation
description: >-
  [AAAI 2026][Graph Learning][NOTAM Parsing] This paper proposes NOTAM-Evolve, a self-evolving framework that achieves dynamic knowledge grounding via knowledge graph-enhanced tabular retrieval (KG-TableRAG)…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "NOTAM Parsing"
  - "Knowledge Graph-Enhanced Retrieval"
  - "Large Language Models"
  - "Self-Evolving Optimization"
  - "Aviation Safety"
date: 2026-05-08
content_hash: 8867cb9cd1414d8e
---

# NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation

**Conference**: AAAI 2026
**arXiv**: [2511.07982](https://arxiv.org/abs/2511.07982)  
**Code**: [https://github.com/Estrellajer/NOTAM-Evolve](https://github.com/Estrellajer/NOTAM-Evolve)  
**Area**: Graph Learning / NLP Applications
**Keywords**: NOTAM Parsing, Knowledge Graph-Enhanced Retrieval, Large Language Models, Self-Evolving Optimization, Aviation Safety

## TL;DR

This paper proposes NOTAM-Evolve, a self-evolving framework that achieves dynamic knowledge grounding via knowledge graph-enhanced tabular retrieval (KG-TableRAG), combined with iterative SFT+DPO preference optimization and a multi-view voting inference mechanism. The framework enables a 7B-parameter LLM to autonomously master deep parsing of complex aviation NOTAMs, achieving a 30.4% accuracy improvement over the base LLM.

## Background & Motivation

### Problem Background

**NOTAM (Notice to Airmen)** is an official notice issued by aviation authorities to inform pilots and air traffic controllers of time-sensitive changes to airspace structures, airport facilities, or flight procedures. Over one million NOTAMs are issued globally each year. NOTAMs employ highly compressed telegraphic language with extensive specialized abbreviations and non-standard syntax, and accurate interpretation is critical to flight safety.

### Shallow Parsing vs. Deep Parsing

**Limitations of Prior Work**: Current automated systems (based on regular expressions, traditional NER, etc.) can only perform **Shallow Parsing**—extracting surface-level information without yielding the actionable intelligence required for decision-making.

The authors introduce the concept of **Deep Parsing**, framing it as a dual reasoning challenge:

**Dynamic Knowledge Grounding**: NOTAMs are not self-contained; textual references must be linked to external, continuously updated aviation infrastructure knowledge bases. For example, the airport code ZBAA requires retrieval of the corresponding runway configuration (e.g., RWY 09L).

**Schema-Based Inference**: Deriving the true meaning of a NOTAM requires schema-level reasoning beyond raw text extraction. For instance, "...REDUCED LENGTH OF 300M" provides a raw parameter, but applying ICAO rules is necessary to infer that 300 meters constitutes a "Basic Approach Lighting System (BALS)."

### Core Motivation
- Shallow parsing is insufficient for operational safety requirements.
- Deep parsing requires simultaneous knowledge grounding and rule-based reasoning capabilities.
- LLMs possess strong semantic understanding but require domain knowledge augmentation and iterative optimization to be effective.
- Safety and cost constraints in the aviation domain typically preclude the use of closed-source commercial APIs, necessitating deployable open-source solutions.

## Method

### Overall Architecture

NOTAM-Evolve comprises three core stages:
1. **Knowledge-Grounded Retrieval**: KG-TableRAG grounds predictions in aviation domain knowledge.
2. **Self-Optimizing Model Refinement**: Self-evolution is achieved through iterative SFT+DPO.
3. **Multi-View Inference**: Robust parsing is ensured via a paraphrase-and-vote mechanism.

### Key Designs

#### 1. **KG-TableRAG Knowledge-Grounded Retrieval**: Addressing the Dynamic Knowledge Grounding Challenge

**Design Motivation**: Aviation data is dynamically updated (airport facility status, runway availability, etc., stored in regularly updated tables). Conventional retrieval is insufficient for the aviation domain because the semantics of table columns and data values are often implicit—a query for "runway closure" may fail to retrieve relevant lighting system or navigation aid information.

**Workflow**:
1. The LLM receives the raw NOTAM and generates a Cypher query to search the knowledge graph.
2. The knowledge graph returns structured domain knowledge (e.g., airport–runway membership relations).
3. The graph query results are concatenated with the original query to form an augmented query.
4. The augmented query is used to retrieve the most relevant information from operational tables.
5. The retrieved information is combined with the original NOTAM as the final input to the LLM.

**Key Advantage**: The knowledge graph provides structured real-world knowledge absent from traditional tabular retrieval, bridging the gap created by implicit relationships.

#### 2. **Self-Optimizing Model Refinement**: Self-Evolution via Closed-Loop Learning

This is the core component of the framework, consisting of alternating SFT and DPO phases.

**Initialization**:
- Dataset $\mathcal{D}_0 = \{(x \circ K, Y^*)\}$ is split 8:2 into training/test sets.
- Base model $\pi_0$ (DeepSeek-R1-Distill-Qwen-7B).
- An empty response pool $\mathcal{R}$ is initialized.

**Iterative Optimization Loop** (each iteration $e$):

**Step 1: Generation and Evaluation**
- The current model $\pi_e$ generates responses $\hat{Y}^{(e)}$ on the training set.
- Responses are compared against ground-truth annotations $Y^*$ and labeled as correct or incorrect.
- The error rate for each input is computed over a lookback window $K'$:

$$\xi(x) = \frac{\sum_{k=1}^{K'} \mathbb{I}(\hat{Y}^{(k)} \neq Y^*)}{K'}$$

**Step 2: SFT Phase**
- Correct input–output pairs are extracted from the response pool $\mathcal{R}$ to form the SFT dataset.
- Fine-tuning is performed with standard negative log-likelihood loss:

$$\mathcal{L}_{\text{SFT}}^{(e)} = -\mathbb{E}_{(x,Y^*) \sim \mathcal{D}_{\text{SFT}}^{(e)}} \left[\sum_{i=1}^{m} \log \pi_\theta(Y_i^* \mid x \circ K, Y_{<i}^*)\right]$$

**Step 3: DPO Phase**
- A preference dataset is constructed: positive samples $y^*$ (correct responses) and negative samples $y^-$ (incorrect responses).
- **Dynamic Data Augmentation**: Inputs with high error rates ($\xi(x) \geq \tau$) are used to generate semantically preserved variants $\mathcal{V}_x$.
- **Weighted Curriculum Learning**: Adaptive sampling weights transition progressively from uniform to error-weighted:

$$w_e(x) = (1-\alpha_e) \frac{1}{N} + \alpha_e \frac{\exp(\beta_{\text{weight}} \xi(x))}{\sum_{j=1}^{N} \exp(\beta_{\text{weight}} \xi(x_j))}$$

where $\alpha_e = \min(e/E, 1)$ is the curriculum progress factor.

- DPO loss optimization:

$$\mathcal{L}_{\text{DPO}}^{(e)} = -\mathbb{E} \left[\log \sigma\left(\beta_{\text{DPO}} \log \frac{\pi_\theta(y^*|x)}{\pi_{\text{ref}}(y^*|x)} - \beta_{\text{DPO}} \log \frac{\pi_\theta(y^-|x)}{\pi_{\text{ref}}(y^-|x)}\right)\right]$$

**Convergence Criterion**: Test set accuracy reaches the target threshold $\eta$. In experiments, 3–5 iterations suffice to reach commercial SOTA-level performance.

#### 3. **Multi-View Inference (Paraphrase + Voting)**: Enhancing Inference Stability

**Design Motivation**: Standard parsing paradigms produce inconsistent predictions on edge cases. Although the baseline model demonstrates partial understanding, minor variations in reasoning paths can determine correctness.

**Implementation**:
1. $N=5$ semantically equivalent NOTAM variants are generated via controlled paraphrasing (preserving aviation terminology, temporal/spatial constraints, and safety-critical numerical values).
2. Each variant is processed independently to yield a candidate structured output.
3. Majority voting determines the final prediction:

$$\hat{Y}_{\text{final}} = \arg\max_{Y} \sum_{k=1}^{N} \mathbb{I}(Y = \hat{Y}^{(k)})$$

Paraphrasing mechanisms include: lexical substitution (e.g., "CTAM" ↔ "Controller Advisory Message"), syntactic restructuring (voice transformation), and contextual expansion (ICAO terminology clarification).

### Loss & Training

- Efficient fine-tuning is performed using the Unsloth framework.
- Base model: DeepSeek-R1-Distill-Qwen-7B (7B parameters).
- All experiments are conducted on a single NVIDIA A800-80GB GPU.
- Iterative optimization runs for 3 rounds, with total computation times of 0.58h → 1.5h → 3.2h.
- Three mechanisms suppress the theoretical $O(t^2)$ growth to a practical average growth of 2.3×.

## Key Experimental Results

### Dataset
A newly constructed NOTAM benchmark dataset comprising 10,000 expert-annotated samples with global distribution, divided into four subsets: Light (1,000), Area (4,000), Runway (2,500), and Taxiway (2,500). Inter-annotator agreement: Krippendorff's Alpha = 0.96.

### Main Results

| Model | Light | Area | Runway | Taxiway | AVG |
|-------|-------|------|--------|---------|-----|
| Regex Template Matching | 0.370 | 0.491 | 0.443 | 0.396 | 0.425 |
| UIE | 0.270 | 0.380 | 0.320 | 0.430 | 0.350 |
| Qwen2.5-7B | 0.560 | 0.777 | 0.412 | 0.748 | 0.624 |
| DeepSeek-R1-7B (base) | 0.410 | 0.484 | 0.446 | 0.492 | 0.458 |
| Qwen2.5-7B (SFT) | 0.590 | 0.793 | 0.730 | 0.864 | 0.744 |
| **NOTAM-Evolve** | **0.620** | 0.725 | **0.836** | **0.868** | **0.762** |
| GPT-4o | 0.605 | **0.851** | 0.770 | 0.914 | 0.785 |
| DeepSeek-R1 (full) | **0.725** | **0.871** | 0.792 | **0.924** | **0.828** |

NOTAM-Evolve achieves a **30.4%** improvement over the base model (0.458 → 0.762), with an AVG score approaching GPT-4o.

### Ablation Study

| KG-TableRAG | Multi-View | AVG |
|-------------|-----------|-----|
| ✓ | ✓ | **0.762** |
| ✓ | ✗ | 0.721 |
| ✗ | ✓ | 0.740 |
| ✗ | ✗ | 0.690 |

Per-category performance across iterative self-optimization:

| Category | Iter 1 | Iter 2 | Iter 3 |
|----------|--------|--------|--------|
| Light | 45.0% | 57.5% | 62.0% |
| Taxiway | 64.6% | 80.4% | 86.8% |

### Key Findings

1. **Self-evolution is highly effective**: Three iterations drive improvements from 45%→62% (Light) and 64.6%→86.8% (Taxiway), validating the efficacy of closed-loop learning.
2. **SFT alone can be detrimental**: Applying SFT in isolation to DeepSeek-R1-Distill-Qwen-7B causes a sharp performance drop from 0.458 to **0.212**, as fine-tuning without chain-of-thought reasoning traces degrades reasoning capability.
3. **Multi-view inference contributes most**: Removing Multi-View causes a 4.1% performance drop, exceeding the 2.2% drop from removing KG-TableRAG.
4. **The 7B model approaches commercial large models**: NOTAM-Evolve performance is close to GPT-4o (0.762 vs. 0.785), which is critical for practical deployment under aviation's safety and cost constraints.
5. **Case study validation**: Closure of airport AGGC → correct inference that runway RWY 07R is also closed (due to airport membership), a reasoning step that baseline models fail to complete.

## Highlights & Insights

1. **Clear problem formulation**: Explicitly framing NOTAM parsing as a "deep parsing" dual reasoning challenge, distinct from shallow information extraction—the problem definition itself constitutes a contribution.
2. **Practical self-evolving paradigm**: No large-scale manual annotation of reasoning chains is required; the model learns from its own outputs, significantly reducing domain adaptation costs.
3. **Elegant application of curriculum learning**: The progressive transition of sampling weights from uniform to error-weighted allows the model to first learn from easy examples before tackling hard ones.
4. **Tangible value for aviation safety**: The 7B model supports local deployment, addressing aviation's safety and cost concerns regarding closed-source APIs.
5. **Dataset contribution**: The 10,000-sample expert-annotated NOTAM dataset with high inter-annotator agreement (α=0.96) has independent value for the research community.

## Limitations & Future Work

1. **Computational cost grows with iterations**: Despite three suppression mechanisms, the number of preference pairs in iterative optimization grows approximately as $O(t^2)$.
2. **Inherent difficulty of NOTAM annotation**: Even expert annotations cannot guarantee perfect accuracy, potentially capping achievable performance.
3. **English-only NOTAMs**: Multilingual scenarios are not addressed, though regional NOTAM variants exist in practice.
4. **Knowledge graph requires manual maintenance**: The update frequency and coverage of aviation infrastructure data are not discussed in detail.
5. **Future directions**: LLM-assisted annotation with expert verification, more efficient optimization strategies, multilingual extension, and real-time operational deployment scenarios.

## Related Work & Insights

- **TableRAG** provides foundational tabular retrieval capability; NOTAM-Evolve augments it with a knowledge graph to handle implicit aviation relationships.
- **DPO preference optimization** combined with curriculum learning demonstrates superior domain adaptation performance compared to SFT alone.
- The self-evolving framework concept is transferable to high-precision NLP tasks requiring domain knowledge, such as **medical report parsing and legal document understanding**.
- Multi-view inference via paraphrase-and-voting is a general strategy for improving the stability of LLMs on structured output tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The self-evolving framework is cleverly designed, though individual components (KG retrieval, DPO, voting) are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — New dataset + comprehensive ablation + iteration analysis + case studies + complexity analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem formulation is clear; the Deep vs. Shallow Parsing framework is well-articulated.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses a genuine need in aviation safety, offers a deployable 7B solution, and provides an open-source dataset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Assessing LLMs for Serendipity Discovery in Knowledge Graphs: A Case for Drug Repurposing](assessing_llms_for_serendipity_discovery_in_knowledge_graphs_a_case_for_drug_rep.md)
- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [\[AAAI 2026\] PathMind: A Retrieve-Prioritize-Reason Framework for Knowledge Graph Reasoning with Large Language Models](pathmind_a_retrieve-prioritize-reason_framework_for_knowledge_graph_reasoning_wi.md)
- [\[AAAI 2026\] Self-Adaptive Graph Mixture of Models](self-adaptive_graph_mixture_of_models.md)
- [\[AAAI 2026\] Self-Correction Distillation for Structured Data Question Answering](self-correction_distillation_for_structured_data_question_answering.md)

</div>

<!-- RELATED:END -->
