---
title: >-
  [Paper Note] Stochastic Chameleons: Irrelevant Context Hallucinations Reveal Class-Based (Mis)Generalization in LLMs
description: >-
  [ACL 2025][Hallucination Detection][hallucination] Explains the internal mechanisms of irrelevant context hallucinations in LLMs using behavioral analysis and mechanistic interpretability experiments: models construct abstract class representations (e.g., "language") in the early layers, followed by a competition for feature selection between two competing circuits (query-based vs. context-based), where their relative activation strength determines whether the model generaliz…
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "hallucination"
  - "irrelevant context"
  - "class-based generalization"
  - "mechanistic interpretability"
  - "competing circuits"
date: 2026-05-08
content_hash: a06bd4d450a7a21e
---

# Stochastic Chameleons: Irrelevant Context Hallucinations Reveal Class-Based (Mis)Generalization in LLMs

**Conference**: ACL 2025  
**arXiv**: [2505.22630](https://arxiv.org/abs/2505.22630)  
**Code**: [GitHub](https://github.com/ziling-cheng/Irrelevant-Context-Hallucination)  
**Area**: Hallucination Detection  
**Keywords**: hallucination, irrelevant context, class-based generalization, mechanistic interpretability, competing circuits

## TL;DR
Explains the internal mechanisms of irrelevant context hallucinations in LLMs using behavioral analysis and mechanistic interpretability experiments: models construct abstract class representations (e.g., "language") in the early layers, followed by a competition for feature selection between two competing circuits (query-based vs. context-based), where their relative activation strength determines whether the model generalizes correctly or hallucinates.

## Background & Motivation
**Background**: LLMs perform exceptionally well on NLP benchmarks, but the "stochastic parrots" criticism argues that they merely stitch together statistical co-occurrences from training data. However, this claim is too coarse—do model errors follow structured patterns?

**Limitations of Prior Work**: Existing literature recognizes that irrelevant context leads to hallucinations (e.g., generating erroneous outputs when RAG retrieves irrelevant documents). However, these studies remain at the level of behavioral observation, leaving it unclear "why" and "how" the internal mechanisms of models are affected by irrelevant context.

**Key Challenge**: Context sensitivity is the foundational capability of In-Context Learning (ICL) (the model must utilize context), but it is also the root cause of hallucinations—this capability and its risk are two sides of the same coin.

**Goal**: (a) Are irrelevant context hallucinations random or structured? (b) How do LLMs internally process irrelevant contexts and generate hallucinations?

**Key Insight**: Designing controlled QA experiments to systematically vary context relevance (same/different category, relevant/irrelevant), and tracking internal information flow using logit attribution, activation patching, and attention knockout.

**Core Idea**: Irrelevant context hallucinations stem from "class-based misgeneralization"—the model first constructs an abstract class representation, and then the context-based circuit hijacks the feature selection process.

## Method

### Overall Architecture
The input format is C+Q (where an irrelevant context C is prepended to the query Q). The model's predictions are compared under Q-only and C+Q conditions. Behavioral analysis is used to validate the "class-based generalization" hypothesis, followed by mechanistic interpretability to trace internal computational mechanisms. Experiments cover Llama-3 (8B/70B), Mistral v0.3 (7B), and Pythia (6.9B/12B) using 39 factual QA subsets from the ParaRel dataset.

### Key Designs

1. **Class-based (mis)generalization hypothesis**:

    - Function: Proposed a structured explanatory framework for context hallucinations
    - Mechanism: When receiving C+Q, the model first infers the "abstract class" of the answer from Q (e.g., "What is the original language of 'A Secret'?" $\rightarrow$ class is "language"). It then selects features from either C or Q (e.g., C mentions Honda $\rightarrow$ Japan $\rightarrow$ Japanese) and combines the class with the features to generate the answer
    - Design Motivation: Explain why hallucinations are not random—71% of context hallucinations simultaneously satisfy "extracting context features" and "belonging to the correct category"

2. **Logit attribution analysis**:

    - Function: Track the logit contributions of each model layer to the candidate answer tokens
    - Mechanism: Project the residual stream $R_{T,l}$ of the final token onto the vocabulary space to track the logit changes of $C_{\text{cand}}$ (context candidate) and $Q_{\text{cand}}$ (query candidate) across layers
    - Key Findings: Early layers (L1-L16) construct abstract class representations (e.g., "languages" ranks the highest), middle layers (L17-L24) host the competition between the two circuits, and late layers determine the final winner

3. **Activation patching + Attention knockout**:

    - Function: Causal intervention experiments to locate and validate competing circuits
    - Mechanism: Inject Gaussian noise ($\sigma=0.3$) into the context subject/object tokens, restore them layer by layer, and measure the Restoration Effect. The context circuit begins transmitting context features to the final token starting from L17, while the query circuit begins transmitting query features starting from L8
    - Design Motivation: Prove the existence of two independent circuits—their relative strength determines the final output, and L17-L24 is the critical computational window for hallucination generation

### Attention Knockout Flip Experiment
Restricting attention at L17 and L24 to focus only on either the query or the context. In context-dominant cases, blocking the context information flow flips 465/1000 samples to query candidates; in query-dominant cases, blocking the query information flow flips 225/1000 samples—causally validating the competing dual-circuit hypothesis.

## Key Experimental Results

### Main Results: Behavioral Analysis of Contextual Influence (106M Data Points)

| Case | Llama-3 | Mistral | Pythia |
|------|---------|---------|--------|
| No Influence (top-3 are all query candidates) | 47.9% | 48.0% | 39.3% |
| Query-dominant (top-1 is query) | 27.9% | 25.7% | 27.2% |
| Context-dominant (top-1 is context) | 15.1% | 17.0% | 19.2% |
| All context candidates | 10.1% | 10.3% | 14.3% |

### Ablation Study: Attention Knockout (Llama-3)

| Configuration | $C_{\text{cand}}$ Probability | $Q_{\text{cand}}$ Probability | Flip Count |
|------|----------------------|----------------------|--------|
| Original Context-dominant | 25.5 | 8.6 | — |
| Knockout L17+L24 | 13.1 | 14.8 | 465/1000 |
| Original Query-dominant | 6.6 | 35.2 | — |
| Knockout L17+L24 | 11.3 | 26.8 | 225/1000 |

### Key Findings
- **71% of hallucinations align with the class-based generalization hypothesis**: Of the 500 manually annotated samples, 81.6% integrated context features, and 84.4% fell within the correct category.
- **Statistically significant PMI test** ($p=0.001$): The average PMI between contexts and their generated candidates is $\approx 4$.
- **Critical Layers L17-L24**: The context circuit decisively integrates context information into the final token within this interval.
- **Scaling up does not eliminate the phenomenon**: Class-based generalization occurs at similar frequencies in Llama-3 70B and Pythia 12B.

## Highlights & Insights
- **The "class-based misgeneralization" hypothesis** provides a deep structured explanation for hallucinations—not random errors, but organized computational biases, which is much more precise than the "stochastic parrots" argument.
- **The competing dual-circuit model** offers mechanistic-level insights into RAG retrieval quality issues: irrelevant retrieval results from the same category are particularly dangerous because they activate the context circuit.
- **The "Stochastic Chameleon" metaphor** refines the concept of "stochastic parrot"—showing that while models indeed generalize and abstract, their generalization process unreliably depends on contextual cues.

## Limitations & Future Work
- **Controlled scenarios are somewhat artificial**: The factual QA format of ParaRel is simple; hallucination mechanisms in real long-context RAG and multi-turn conversations might be more complex.
- **Limited to factual QA**: The mechanisms behind reasoning and creative hallucinations may differ.
- **No mitigation method proposed**: The mechanism is revealed, but no intervention scheme is provided—future work could explore selectively suppressing the context circuit in layers L17-L24.

## Related Work & Insights
- **vs. Li et al. (ITI)**: ITI found that the "truthfulness direction" can reduce hallucinations via inference-time intervention, whereas this paper provides a more micro-level dual-circuit model.
- **vs. Shi et al. (Context Distraction)**: They observed that irrelevant context degrades mathematical reasoning capabilities, whereas this paper offers the internal computational mechanisms.
- **vs. Meng et al. (ROME)**: ROME localizes factual storage locations using causal tracing, whereas this paper uses similar methods to locate feature selection circuits.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Competing dual-circuit model + class-based generalization hypothesis are entirely novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 106M data points + manual annotation + PMI statistics + three mechanistic interpretability methods
- Writing Quality: ⭐⭐⭐⭐ Clear reasoning chain, intuitive Figure 1
- Value: ⭐⭐⭐⭐⭐ Bears significant theoretical importance for understanding hallucination mechanisms, with direct implications for RAG design

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers](../../NeurIPS2025/hallucination/generalization_or_hallucination_understanding_out-of-context_reasoning_in_transf.md)
- [\[ACL 2025\] HD-NDEs: Neural Differential Equations for Hallucination Detection in LLMs](hd-ndes_neural_differential_equations_for_hallucination_detection_in_llms.md)
- [\[ACL 2025\] ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs](icr_probe_tracking_hidden_state_dynamics_for_reliable_hallucination_detection_in.md)
- [\[ACL 2025\] DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination](drag_distilling_rag_slm.md)
- [\[ICCV 2025\] Why LVLMs Are More Prone to Hallucinations in Longer Responses: The Role of Context](../../ICCV2025/hallucination/why_lvlms_are_more_prone_to_hallucinations_in_longer_responses_the_role_of_conte.md)

</div>

<!-- RELATED:END -->
