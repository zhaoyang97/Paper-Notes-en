---
title: >-
  [Paper Note] From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding
description: >-
  [ACL 2025 (Outstanding Paper Award 🏆)][Instruction Synthesis] This paper proposes the "Attributed Grounding" framework, which constructs the SynthQuestions dataset consisting of 1 million diverse and complex instructions through top-down user attribution and bottom-up web-document-based instruction synthesis, enabling trained models to achieve state-of-the-art performance across multiple general benchmarks.
tags:
  - "ACL 2025 (Outstanding Paper Award 🏆)"
  - "Instruction Synthesis"
  - "Data Diversity"
  - "Attributed Generation"
  - "LLM Alignment"
  - "Web Documents"
date: 2026-05-08
content_hash: f1049d4d6940f03c
---

# From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding

**Conference**: ACL 2025 (Outstanding Paper Award 🏆)  
**arXiv**: [2506.03968](https://arxiv.org/abs/2506.03968)  
**Code**: [https://github.com/Ignoramus0817/SynthQuestions](https://github.com/Ignoramus0817/SynthQuestions)  
**Area**: Others  
**Keywords**: Instruction Synthesis, Data Diversity, Attributed Generation, LLM Alignment, Web Documents

## TL;DR

This paper proposes the "Attributed Grounding" framework, which constructs the SynthQuestions dataset consisting of 1 million diverse and complex instructions through top-down user attribution and bottom-up web-document-based instruction synthesis, enabling trained models to achieve state-of-the-art performance across multiple general benchmarks.

## Background & Motivation

**Background**: Large-scale, high-quality instruction data serves as the critical fuel for LLM alignment. Methods such as Self-Instruct and Evol-Instruct have demonstrated that synthetic instruction data can effectively train models, while works like WizardLM enhance instruction complexity through "evolution" strategies.

**Limitations of Prior Work**: Existing instruction synthesis methods face two major challenges: (1) Limited grounding sources—most methods rewrite or expand on a small set of seed instructions, resulting in a narrow distribution of generated instructions that fails to cover the wide spectrum of real user needs; (2) Superficial complexity enhancement—increasing instruction length by mechanically stacking constraints (e.g., "please answer in English" or "limit to 200 words") without truly improving the cognitive complexity or actual utility of the instructions.

**Key Challenge**: Truly valuable instructions should reflect the actual needs of real users in specific scenarios. However, existing synthesis methods lack grounding in real-world contexts, producing instructions that "appear complex but are empty." To achieve the triple objectives of large scale, high diversity, and high complexity, a sufficiently rich source of information is required to drive the generation process.

**Goal**: Create a framework capable of utilizing massive web documents as grounding sources to automatically generate millions of diverse and complex instructions.

**Key Insight**: High-quality, real-world instructions typically feature three core elements: the document (information source), the user (the requester), and the motivation (why the request is made). These three elements form the "attribution triangle" of instructions. By automatically completing these three elements, high-quality instructions can be reversely engineered from web documents.

**Core Idea**: Deconstruct the instruction synthesis process into two steps: first, "attribute" (infer user personas and motivations from existing instructions), and second, "synthesize" (generate instructions starting from web documents combined with user context). The massive supply of web documents guarantees scale and diversity.

## Method

### Overall Architecture

The overall pipeline consists of two symmetric processes. Top-down: Starting from 29K manually collected seed instructions, related documents are retrieved for each instruction via web search. Subsequently, an LLM infers "what kind of user would raise this request under what scenario," constructing an attribution triplet (document, user, motivation). Bottom-up: Documents are sampled from large-scale web corpora (such as FineWeb). Using the attribution triplets as in-context demonstrations, the LLM first conceptualizes a plausible user scenario, and then generates a meaningful instruction based on this scenario and the document content.

### Key Designs

1. **Top-down Attribution**:

    - **Function**: Deconstruct real instructions into document-user-motivation triplets, providing high-quality demonstrations for subsequent synthesis.
    - **Mechanism**: Collect 29K high-quality seed instructions from various open-source instruction datasets. Extract keywords from each instruction and retrieve the top-ranked web document via a search engine to serve as the information source. An LLM is then prompted to generate attribution descriptions, inferring the user persona (e.g., "a computer science graduate student writing a thesis") and their specific motivation (e.g., "needs to understand the complexity analysis of an algorithm to complete the experiment section"). Thus, each seed instruction is mapped to a complete attribution triplet $(d, u, m)$, where $d$ represents the document, $u$ is the user persona, and $m$ is the motivation.
    - **Design Motivation**: This step establishes the context of "why someone would ask this question" for seed instructions, while the generated attribution triplets serve as few-shot demonstrations for downstream synthesis.

2. **Bottom-up Synthesis**:

    - **Function**: Mass-produce diverse and complex instructions starting from massive web documents.
    - **Mechanism**: Randomly sample documents from large-scale web corpora like FineWeb. For each document, sample several attribution triplets from the pool as in-context demonstrations, and prompt an LLM to perform a two-step generation: (1) conceptualize a reasonable user scenario based on the document content (who they are, in what context, with what motivation); (2) generate a meaningful user instruction based on the scenario and document content. The natural diversity of these documents (spanning the entire web) inherently ensures a wide distribution of instructions.
    - **Design Motivation**: Setting web documents rather than seed instructions as the starting point bypasses the bottleneck where "seeds dictate the distribution." The scenario conceptualization step guarantees that generated instructions stem from realistic contexts rather than hollow rewrites.

3. **Quality Assessment and Diversity Filtering**:

    - **Function**: Ensure the baseline quality and distributional diversity of large-scale generated instructions.
    - **Mechanism**: After generating instructions, evaluate their quality using LLM scoring and filter out low-quality entries. Subsequently, apply BERTopic for topic modeling, and select the highest-scoring instructions within each topic to guarantee a uniform distribution of the final dataset across the topic space. Finally, run safety audits using LLaMA-Guard-3-8B to filter out potentially harmful content.
    - **Design Motivation**: Large-scale generation inevitably introduces noise; multi-stage filtering maintains quality and safety while scaling. Topic-based diversity sampling avoids overrepresentation of certain popular domains.

### Loss & Training

Once the dataset is constructed, models are trained using a standard SFT pipeline. Additionally, 100K preference data samples are constructed for subsequent DPO training. ArmoRM-Llama3-8B-v0.1 is used as the reward model to score multiple responses from the SFT model, constructing preference pairs.

## Key Experimental Results

### Main Results

Performance of models trained on SynthQuestions across multiple benchmarks:

| Benchmark | SynthQuestions (SFT) | WizardLM | Self-Instruct | Evol-Instruct | Gain |
|------|---------------------|----------|---------------|---------------|------|
| AlpacaEval 2.0 LC(%) | 32.7 | 24.1 | 18.5 | 27.3 | +5.4 |
| MT-Bench Average | 7.52 | 6.89 | 6.21 | 7.15 | +0.37 |
| Arena-Hard | 28.4 | 21.2 | 15.8 | 24.6 | +3.8 |
| IFEval (Strict) | 54.2 | 46.8 | 40.1 | 49.5 | +4.7 |
| MMLU | 63.8 | 62.1 | 60.5 | 62.9 | +0.9 |

Impact of data scale on performance (SFT + Llama-3-8B):

| Data Volume | AlpacaEval 2.0 LC(%) | MT-Bench |
|-------|---------------------|----------|
| 100K | 26.3 | 7.08 |
| 250K | 28.9 | 7.25 |
| 500K | 31.2 | 7.41 |
| 1M | 32.7 | 7.52 |

### Ablation Study

| Configuration | AlpacaEval 2.0 LC(%) | Description |
|------|---------------------|------|
| Full pipeline | 32.7 | Full attribution + synthesis |
| w/o User attribution | 28.1 | Generates instructions directly from documents without constructing user personas, drops by 4.6% |
| w/o Document grounding | 25.4 | Pure seed rewriting without utilizing web documents, drops by 7.3% |
| w/o Quality filtering | 30.5 | No quality score filtering, drops by 2.2% |
| w/o Diversity sampling | 31.1 | No BERTopic balanced sampling, drops by 1.6% |
| Random document + Direct generation | 24.8 | Generates instructions directly from random documents without attribution |

### Key Findings

- **Document Grounding is the Primary Contributor**: Omitting web documents leads to a steep 7.3% decline in performance, proving that diverse information sources are key drivers for instruction diversity.
- **User Attribution Substantially Increases Complexity**: Integrating user personas elevates the average instruction complexity score by approximately 35%, showing that "conceptualizing users before generating instructions" is far more effective at generating in-depth instructions than "direct generation."
- **Continuous Scaling**: From 100K to 1M, performance climbs continuously without scaling saturation, suggesting that incorporating more web documents can further enhance performance.
- SynthQuestions exhibits stable performance across different base models (Llama-3-8B, Mistral-7B), validating the cross-model transferability of the dataset quality.

## Highlights & Insights

- **"Attribution Triangle" Framework**: The philosophy of decomposing an instruction into three core elements—document, user, and motivation—is elegant. Beyond instruction synthesis, this abstract framework could theoretically transfer to any scenario that demands the generation of "contextualized natural language content," such as user simulation in dialogue systems or question generation in education.
- **Web Documents as an Infinite Grounding Source**: This serves as a vital insight—the internet is inherently the largest knowledge repository. Using documents as the starting point breaks the distributional constraints of seed instructions. Furthermore, as web corpora expand, the dataset can scale boundlessly.
- **Recipient of the ACL 2025 Outstanding Paper Award**: This underscores the widespread academic recognition of this work. The generalizability of the method and the thoroughness of the experiments were highly praised by the program committee.

## Limitations & Future Work

- Instruction synthesis relies heavily on the generation capabilities of the teacher LLM (e.g., GPT-4), meaning the upper bound of synthetic data quality is capped by the teacher model.
- Web documents may contain inaccurate information (factual errors, stale data), and instructions synthesized based on these documents risk propagating errors.
- Currently, the focus is solely on single-turn instructions, without touching upon multi-turn conversational instruction synthesis. In real scenarios, many complex user needs require multi-turn interactions.
- The choice of topic granularity in BERTopic may influence the quality of the final diversity distribution; the selection criteria for this hyperparameter remain somewhat ambiguous.
- Incorporating temporal control is a promising future direction—prioritizing recent documents to generate instructions relevant to the current state of the world.

## Related Work & Insights

- **vs Self-Instruct**: Self-Instruct scales from a few seeds via self-generation, but its distribution is constrained by those seeds. SynthQuestions breaks this ceiling using web documents, yielding greater diversity and scale.
- **vs Evol-Instruct/WizardLM**: Evol-Instruct improves complexity via "evolution" rules (deepening, introducing constraints, etc.), but these rules rely on fixed templates, generating "superficial complexity." Conversely, SynthQuestions' scenario-driven generation yields "semantically complex" instructions.
- **vs LIMA "Less is More"**: LIMA argues that a small volume of high-quality data is sufficient. The scaling experiments of SynthQuestions demonstrate that under guaranteed data quality, larger scale indeed brings continuous improvements.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The "Attributed Grounding" framework elegantly addresses the diversity and complexity challenges of instruction synthesis, featuring a novel perspective and broad inspirational value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluations across multiple benchmarks, scaling experiments, ablation studies, and cross-model validations are provided. Both the dataset and models are fully open-sourced.
- **Writing Quality**: ⭐⭐⭐⭐ The paper is well-structured, and the motivation of the attribution framework is thoroughly explained, though some experimental details could be more granular.
- **Value**: ⭐⭐⭐⭐⭐ Honored with the ACL 2025 Outstanding Paper Award, the dataset and methodology hold immense practical value and community impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Predicting Implicit Arguments in Procedural Video Instructions](implicit_arguments_video_instructions.md)
- [\[ACL 2025\] USDC: A Dataset of User Stance and Dogmatism in Long Conversations](usdc_a_dataset_of_underlineuser_underlinestance_and_underlinedogmatism_in_long_u.md)
- [\[ACL 2025\] Counterspeech the Ultimate Shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning](hippro_counterspeech_gen.md)
- [\[ACL 2025\] KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding](kodcode_a_diverse_challenging_and_verifiable_synthetic_dataset_for_coding.md)
- [\[ACL 2025\] Generating Synthetic Relational Tabular Data via Structural Causal Models](generating_synthetic_relational_tabular_data_via_structural_causal_models.md)

</div>

<!-- RELATED:END -->
