---
title: >-
  [Paper Note] Filling the Gap: Is Commonsense Knowledge Generation useful for Natural Language Inference?
description: >-
  [ACL 2026][NLP Understanding][NLI] The paper enables LLMs to autonomously generate natural language "commonsense axioms" that bridge premises and hypotheses. These axioms are filtered via a "factuality judge…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "NLI"
  - "Commonsense Axiom"
  - "SNLI"
  - "ANLI"
  - "Selective Knowledge Injection"
date: 2026-05-08
content_hash: 57e84c183d9a904b
---

# Filling the Gap: Is Commonsense Knowledge Generation useful for Natural Language Inference?

**Conference**: ACL 2026  
**arXiv**: [2507.15100](https://arxiv.org/abs/2507.15100)  
**Code**: None (No link provided in the paper)  
**Area**: Natural Language Inference / Commonsense Reasoning / LLM Evaluation  
**Keywords**: NLI, Commonsense Axiom, SNLI, ANLI, Selective Knowledge Injection

## TL;DR
The paper enables LLMs to autonomously generate natural language "commonsense axioms" that bridge premises and hypotheses. These axioms are filtered via a "factuality judge," and only high-quality ones are injected back into NLI prompts. Results show Llama-3.1-70B and gpt-oss-120b achieve accuracy gains of 1.99-6.88% on SNLI/ANLI, significantly mitigating the "Neutral" safety preference bias.

## Background & Motivation
**Background**: NLI (Recognizing Textual Entailment) requires determining whether a premise Entails, Contradicts, or is Neutral to a hypothesis. Theoretically, this requires inferences made by a "reader with common sense." While LLMs report high accuracy on benchmarks like SNLI/ANLI, studies (Luo 2022, McKenna 2023, Liu 2023) suggest they often rely on surface artifacts rather than genuine premise $\to$ hypothesis logical chains.

**Limitations of Prior Work**: When implicit "bridging knowledge" exists between the premise and hypothesis (e.g., "a woman with a big grin" $\to$ "she is not shot"), models often fail due to missing commonsense knowledge. Existing methods for integrating external knowledge (ExBERT, ERNIE-NLI, e-SNLI extensions) either rely on manual keyword labeling or extraction from structured KGs like ConceptNet/Aristo, which cannot guarantee relevance to the specific P-H pair.

**Key Challenge**: There is a trade-off between "coverage" and "relevance" in commonsense knowledge—KG sources have narrow coverage but high accuracy, while LLM generations have broad coverage but risk hallucinations. For commonsense to be effective, it must be both "freely generated" and "selectively utilized."

**Goal**: (1) Verify whether LLMs can reliably generate commonsense axioms for P-H pairs; (2) Evaluate the impact of injecting these axioms on NLI accuracy.

**Key Insight**: Reconceptualize "commonsense" as natural language "axioms"—bridge rules of world knowledge expressed in a single sentence, rather than formal logical axioms. The LLM first generates axioms, then the same model scores them based on factuality/helpfulness, ultimately injecting only high-scoring ones.

**Core Idea**: A mixed strategy of "selective access" is employed, injecting axioms only when the judge deems them highly factual, allowing the model to revert to vanilla P-H reasoning when generation is unreliable.

## Method

### Overall Architecture
The authors designed a three-stage prompting pipeline: (1) **Axiom Generation Prompt**: Feeds $(P, H)$ to the LLM to generate a natural language axiom $A$; (2) **Axiom Evaluation Prompt**: Feeds $(P, H, A)$ back to the same model to score it across dimensions like factuality, consistency, and helpfulness (adapted from Zheng 2024); (3) **Inference Prompt**: Branches based on the evaluation—baseline mode $(P, H) \to \text{label}$, axiom-injected mode $(P, H, A) \to \text{label}$, or hybrid mode which injects only high-factuality $A$, otherwise falling back to baseline. This hybrid approach represents the selective access strategy.

### Key Designs

1.  **LLM-generated Natural Language Axioms (not KG triples)**:
    - **Function**: Enables the LLM to articulate commonsense rules bridging the premise to the hypothesis (e.g., "a big grin usually implies happiness/safety, which is inconsistent with being shot").
    - **Mechanism**: Prompts the LLM directly to describe the "commonsense bridge enabling the hypothesis to follow from the premise," avoiding the sparsity of discrete KGs like ConceptNet. The model leverages its internalized knowledge to produce axioms highly relevant to the $(P, H)$ context.
    - **Design Motivation**: The authors argue that NLI failures often stem from a "knowledge gap" between $P$ and $H$. Articulating this gap internally is more targeted and controllable than external KG retrieval.

2.  **Factuality-aware Selective Injection (Core of the Hybrid Strategy)**:
    - **Function**: Filters axioms by using the LLM as a judge of factuality, injecting only "highly factual" axioms into the inference prompt.
    - **Mechanism**: The same LLM performs axiom evaluation, labeling them as helpful, factual, or consistent. Only axioms passing a factuality threshold proceed to the injection stage. Formally: $\hat{y} = \text{LLM}(P, H, A) \text{ if } \text{score}(A) \geq \tau \text{ else } \text{LLM}(P, H)$.
    - **Design Motivation**: Pure injection experiments revealed inconsistent quality—generated axioms often contain hallucinations or tautologies. Gating injection ensures it only occurs when the model "believes" the axiom, maximizing the signal-to-noise ratio.

3.  **Mitigating Neutral Bias**:
    - **Function**: Breaks the LLM's tendency to select "Neutral" as a safe fallback when facing uncertainty by providing specific world knowledge.
    - **Mechanism**: When a premise and hypothesis seem tangentially related, models default to "Neutral." High-quality axioms transform "apparent irrelevance" into "commonsense bridge hence entailment/contradiction," pulling the model toward the correct category.
    - **Design Motivation**: It was observed that baselines on adversarial datasets like ANLI had excessive Neutral recall—a self-protection mechanism rather than a true judgment of independence. Axiom injection provides an explicit reason for the model to take a stance.

### Loss & Training
The entire process is training-free. Llama-3.1-70B-Instruct and gpt-oss-120b were utilized in a zero-shot setting across three prompt types (generation, evaluation, inference). Balanced samples of 2000 items each were drawn from SNLI and ANLI.

## Key Experimental Results

### Main Results
Comparison of three strategies on 2000 samples from SNLI/ANLI:

| Dataset | Model | Baseline | Strong Injection | Hybrid (factuality-gated) | Gain |
|---------|-------|----------|------------------|---------------------------|------|
| SNLI | Llama-3.1-70B | ~Base | Fluctuation | +1.99%~+6.88% | Consistent |
| SNLI | gpt-oss-120b | ~Base | Fluctuation | +1.99%~+6.88% | Consistent |
| ANLI | Llama-3.1-70B | ~Base | Fluctuation | +1.99%~+6.88% | Consistent |
| ANLI | gpt-oss-120b | ~Base | Fluctuation | +1.99%~+6.88% | Consistent |

> Note: Detailed numerical tables are in §4/Appendix; the abstract confirms a gain range of [1.99%, 6.88%] across all tested configurations.

Data distribution (Table 1): SNLI (Entail 689 / Contradict 651 / Neutral 660); ANLI (Entail 771 / Contradict 585 / Neutral 644). The balanced classes rule out spurious gains from class priors.

### Ablation Study

| Configuration | Key Observation | Description |
|---------------|-----------------|-------------|
| Baseline (Direct P, H inference) | Reference Point | High Neutral recall |
| Strong injection (Inject all axioms) | Unstable Gain | Performance dropped in some cases due to axiom noise |
| **Hybrid (factuality-gated injection)** | **+1.99~+6.88%** | Selective injection is the primary driver of improvement |

### Key Findings
- The primary effect of axiom injection is shifting the model from a "conservative Neutral bias" toward Entailment/Contradiction, providing a "rational basis for commitment."
- Strong injection (without filtering) often degrades performance, proving that autonomously generated axioms vary wildly in quality; the filtering mechanism is a core contribution rather than just a trick.
- Improvements are more pronounced on ANLI (adversarial) than SNLI, suggesting commonsense bridges are more critical for harder cases.

## Highlights & Insights
- "Using the LLM as its own axiom factuality judge" is a simple but effective self-evaluation strategy—cheaper and more context-aware than retrieving triples from external KGs.
- The failure of NLI is framed as a "knowledge gap" rather than insufficient model capacity. Filling this gap yields pure gains, suggesting a path for "selective augmentation" in other reasoning tasks.
- It validates a counter-intuitive point: feeding more information into a prompt is not always better; information credibility must be assessed first. This mirrors the "noisy retrieval" problem in RAG.

## Limitations & Future Work
- Evaluation was limited to SNLI/ANLI; generalization to MNLI, e-SNLI, or HANS remains untested.
- Factuality scoring relies on the same LLM, creating a potential "self-praising" loop; using an independent model as a judge would increase reliability.
- Lack of comparison with external knowledge baselines like RAG-from-ConceptNet or e-SNLI explanations means the superiority of LLM-axioms over KG-axioms is not definitively established.
- Tested only on 70B+ models; the quality of axiom generation and self-judging in smaller models is unknown.

## Related Work & Insights
- **vs ExBERT (Gajbhiye 2021)**: Both inject commonsense, but ExBERT trains fusion layers with ConceptNet triples; this work uses prompting with factuality filtering. The latter avoids training costs but sacrifices the precision of fine-tuning.
- **vs e-SNLI Explanations (Camburu 2018)**: e-SNLI uses human-annotated explanations; this work uses self-generated axioms, saving annotation costs but requiring quality control.
- **vs Nguyen & Hatua 2024**: They retrieve commonsense from ConceptNet/Google based on e-SNLI keywords; this work uses internalized knowledge, avoiding retrieval failure risks.
- **vs Wei et al. 2024 (CSQA)**: Uses LLMs as commonsense generators for CSQA; this work adapts the idea specifically to NLI with selective access.

## Rating
- Novelty: ⭐⭐⭐ Combinatorial innovation of "LLM-generated commonsense + self-evaluated factuality + selective injection" rather than a revolutionary paradigm.
- Experimental Thoroughness: ⭐⭐⭐ Two datasets × two models × three strategies are informative, but the study lacks smaller models and broader NLI benchmark comparisons.
- Writing Quality: ⭐⭐⭐⭐ Figure 1 clearly illustrates motivation; the distinction of the axiom concept is well-defined.
- Value: ⭐⭐⭐⭐ Provides a cautionary note for "injecting external knowledge via prompts"—injection without a quality gate is often a negative optimization. This principle is readily applicable to RAG and CoT augmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Commonsense Knowledge with Negation: A Resource to Enhance Negation Understanding](commonsense_knowledge_with_negation_a_resource_to_enhance_negation_understanding.md)
- [\[AAAI 2026\] Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives](../../AAAI2026/nlp_understanding/understanding_syllogistic_reasoning_in_llms_from_formal_and_natural_language_per.md)
- [\[ACL 2026\] BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation](boundrl_efficient_structured_text_segmentation_through_reinforced_boundary_gener.md)
- [\[ACL 2026\] Semantic Reranking at Inference Time for Hard Examples in Rhetorical Role Labeling](semantic_reranking_at_inference_time_for_hard_examples_in_rhetorical_role_labeli.md)
- [\[ACL 2026\] Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs](creating_conlangs_to_probe_the_metalinguistic_grammatical_knowledge_of_llms.md)

</div>

<!-- RELATED:END -->
