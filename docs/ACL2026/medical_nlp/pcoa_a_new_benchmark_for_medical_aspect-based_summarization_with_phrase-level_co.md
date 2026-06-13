---
title: >-
  [Paper Note] PCoA: A New Benchmark for Medical Aspect-Based Summarization With Phrase-Level Context Attribution
description: >-
  [ACL2026][Medical NLP][Medical Aspect Summarization] PCoA establishes a medical aspect-level summarization benchmark for Randomized Controlled Trial (RCT) abstracts. It aligns each aspect summary with both supporting sen…
tags:
  - "ACL2026"
  - "Medical NLP"
  - "Medical Aspect Summarization"
  - "Phrase-level Attribution"
  - "RCT Summarization"
  - "Verifiable Summarization"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 494d7dade3cad2b7
---

# PCoA: A New Benchmark for Medical Aspect-Based Summarization With Phrase-Level Context Attribution

**Conference**: ACL2026  
**arXiv**: [2601.03418](https://arxiv.org/abs/2601.03418)  
**Code**: https://github.com/chubohao/PCoA  
**Area**: Medical NLP / Medical Summarization / Attributable Generation  
**Keywords**: Medical Aspect Summarization, Phrase-level Attribution, RCT Summarization, Verifiable Summarization, LLM Evaluation

## TL;DR
PCoA establishes a medical aspect-level summarization benchmark for Randomized Controlled Trial (RCT) abstracts. It aligns each aspect summary with both supporting sentences and contributory phrases, utilizing a three-tier metric system—claim, citation, and phrase—to evaluate the proficiency of LLMs in verifiable medical summarization.

## Background & Motivation
**Background**: Synthesis of medical evidence typically requires reviewing numerous RCT abstracts and extracting key information based on clinical aspects such as Participants, Intervention, Comparator, and Outcomes (PICO). Traditional summarization tasks often produce a single general summary, whereas medical readers frequently focus on specific aspects like treatment regimens, primary endpoints, follow-up duration, or adverse events. Consequently, aspect-level summarization is better suited for clinical evidence comparison and systematic review compilation than general summarization.

**Limitations of Prior Work**: While LLMs can generate fluent medical summaries, fluency does not equate to reliability in high-stakes medical scenarios. If a sentence in a summary lack a clear pointer to its source in the original text, readers must still manually retrieve evidence from the long-form text. Existing context attribution methods often provide citations only at the document, paragraph, or sentence level. This granularity is too coarse to verify whether specific values, drug names, dosages, and endpoints in the summary are accurately derived from the source.

**Key Challenge**: Medical summarization must simultaneously satisfy the objectives of "information compression" and "auditable evidence." Stronger compression often leads to loss of provenance, while coarse citations make it difficult for readers to quickly locate specific facts within the summary. This paper argues that truly usable medical aspect summaries should identify not just "what was written," but also "which sentence and which phrases contributed to each piece of key information."

**Goal**: The authors decompose the task into three interlinked outputs: given an RCT abstract and a medical aspect, the system must generate an aspect summary, identify the supporting sentences from the source, and highlight the phrases within those sentences that directly contribute to the summary content. Consequently, the evaluation target shifts from the summary text alone to a joint assessment of summary quality, sentence-level citation quality, and phrase-level evidence alignment.

**Key Insight**: The paper selects RCT abstracts related to melanoma as the source for annotation, as RCTs represent the core evidence form in evidence-based medicine, featuring relatively stable abstract structures and public availability. Based on the PICO framework and physician interviews, the authors define 16 medical aspects. This benchmark covers classic PICO information as well as often-overlooked clinical reporting aspects such as funding, registration, and blinding.

**Core Idea**: By employing a tripartite annotation of "aspect summary + supporting sentences + contributory phrases," the work replaces coarse-grained evaluations of generated summaries, advancing the verifiability of medical summarization to the phrase level.

## Method
PCoA is essentially a combination of a new task, a new dataset, and a new evaluation framework. It defines the medical aspect summarization task as a generation problem with an evidence chain: models cannot simply output a polished summary; they must also specify which source sentences support the summary and which phrases in those sentences directly contributed to the content. This design is highly suitable for medical NLP, where medical facts consist of specific entities, values, dosages, durations, and endpoints—phrase-level evidence aligns more closely with the verification needs of clinical readers than paragraph-level citations.

### Overall Architecture
The work is divided into four stages.

The first stage is source data filtering. The authors retrieved melanoma-related RCT abstracts from PubMed, initially identifying 607 papers and finally retaining 152. Filtering criteria included publication within the last decade, English language, RCT type, focus on melanoma, and publication in JCR Q1 or Q2 journals. The study utilizes abstracts rather than full texts because abstracts are publicly accessible and typically contain information on the primary clinical aspects.

The second stage is the definition of the medical aspect system. Based on physician interviews and the PICO framework, the authors defined 16 aspects: Objective, Participants, Intervention, Comparator, Outcomes, Findings, Medicines, Treatment Duration, Primary Endpoints, Secondary Endpoints, Follow-Up Duration, Adverse Events, Randomization, Blinding, Funding, and Registration. Each aspect has minimum reporting requirements; for instance, Intervention must include the administration regimen, and Outcomes must include endpoints and values. This step provides annotators with a clear definition of "complete information" and establishes consistent semantic boundaries for evaluation.

The third stage is expert annotation. Two medical students completed a three-step annotation process using a custom online system. They first assigned one or more clinical aspects to each source sentence (allowing for sentences with no associated aspects). Then, based on the sentences corresponding to an aspect, they wrote the aspect summary. Finally, they highlighted contributory phrases from the selected sentences, requiring these phrases to enter the summary in their original or variant forms. The final dataset contains 1,799 aspect-level instances, each including a summary, cited sentences, and contributory phrases.

The fourth stage is model evaluation. Given a document $d=[c_1,c_2,\cdots,c_n]$ and a target aspect $a$, the system outputs a generated summary $sum'$, a set of cited sentences $\mathcal{C}'$, and a set of contributory phrases $\mathcal{P}'$. The evaluation framework measures whether the summary covers key facts, whether the cited sentences truly support the summary, and whether the phrases originate from the cited sentences and correspond to the summary content. The authors utilized Mistral-Large-2411 for claim decomposition, TRUE for entailment judgment, and NLTK for phrase tokenization.

### Key Designs
1.  **Phrase-Level Context Attribution Dataset**:
    - **Function**: Explicitly connects each medical aspect summary to supporting sentences and contributory phrases, creating a finer evidence structure than standard summarization datasets.
    - **Mechanism**: Conducts sentence-level aspect annotation for each RCT abstract, writes aspect summaries based on those sentences, and identifies the phrases within the sentences that were incorporated into the summary.
    - **Design Motivation**: Errors in medical summarization often occur at the level of specific phrases, such as drug dosages, follow-up months, endpoint names, or registration numbers; coarse document-level or paragraph-level citations are inefficient for locating these facts.

2.  **Decoupled Three-Tier Evaluation Metrics**:
    - **Function**: Separately evaluates the summary, citation, and phrase components rather than conflating all issues into a single ROUGE score or overall rating.
    - **Mechanism**: The summary layer uses claim recall and claim precision by decomposing reference and generated summaries into atomic subclaims and then using NLI for mutual entailment checks. The sentence layer uses sentence recall and precision, requiring cited sentences to be in the reference set and to support at least one claim in the generated summary. The phrase layer uses phrase recall and precision, requiring contributory phrases to exist within the valid intersection of reference phrases, generated phrases, cited sentences, and the generated summary.
    - **Design Motivation**: A model might generate a complete summary with inaccurate citations or cite the correct sentences but fail to extract key phrases; decoupled evaluation exposes these distinct failure modes.

3.  **Unified Comparison of Three Attribution Strategies**:
    - **Function**: Compares intrinsic, prior, and post-hoc context attribution approaches using identical data and metrics.
    - **Mechanism**: The intrinsic strategy generates the summary, citations, and phrases in a single pass. The prior strategy first retrieves relevant sentences and contributory phrases, then generates the summary based solely on this evidence. The post-hoc strategy generates the summary first, then retrieves citations and phrases for it.
    - **Design Motivation**: These strategies represent different engineering paths for attributable generation. PCoA provides not just a leaderboard but answers to the methodological question of whether evidence should be bound before, during, or after generation.

### Loss & Training
No specialized models were trained; the experimental focus was on zero-shot benchmarking. All LLMs used a shared prompt template. Under the intrinsic setting, they outputted the summary, sentences, and key phrases directly from the RCT abstract and target aspect. The prior setting was split into two steps: prompting the model to find relevant sentences/phrases first, and then prompting it to write the summary based on that evidence. The post-hoc setting was also split: writing the summary first, then checking for supporting sentences/phrases. Temperature was set to 0.7. The authors evaluated LLaMA3.1-70B-Instruct, Mistral-Large-2411, DeepSeek-V3-0324, and GPT-4o via commercial APIs, with a total evaluation cost of approximately $23.6.

## Key Experimental Results

### Main Results
The main experiment compares four LLMs under the intrinsic context attribution setting. Metrics are grouped into three sets: C-R/C-P/C-F1 for summary claim quality, S-R/S-P/S-F1 for cited sentence quality, and P-R/P-P/P-F1 for contributory phrase quality.

| Model | C-F1 | S-F1 | P-F1 | Main Observation |
| :--- | :--- | :--- | :--- | :--- |
| LLaMA3.1-70B | 0.605 | 0.650 | 0.538 | High recall, but lower precision |
| DeepSeek-V3 | 0.659 | 0.672 | 0.539 | Best at summary layer, joint best at citation layer |
| Mistral-Large | 0.651 | 0.655 | 0.574 | Best at phrase layer, stronger C-P and P-P |
| GPT-4o | 0.621 | 0.672 | 0.539 | Joint best at citation layer, stable format following |

The results indicate that no single model leads across all three tiers. DeepSeek-V3 achieved a claim recall of 0.757, suggesting it is better at covering facts from the reference summary. Mistral-Large performed best at the phrase layer with a phrase precision of 0.522 and a P-F1 of 0.574. Generally, recall was higher than precision for all models, indicating a tendency toward over-generation, over-citation, and over-extraction, leading to redundancy or partial inaccuracies.

Regarding template compliance, DeepSeek-V3 and GPT-4o achieved 100% compliance. LLaMA3.1 had 33 format deviations out of 1,799 outputs, requiring manual correction. Mistral-Large had only 1,609 template-compliant outputs, with 99 invalid responses excluded. This highlights that format stability is a critical component of deployability in structured medical tasks.

### Ablation Study
Rather than traditional module ablation, the paper used the comparison of three context attribution strategies as the core experimental analysis. This comparison addresses whether evidence is more reliable when filtered before, during, or after the summary generation.

| Attribution Strategy | C-F1 | S-F1 | P-F1 | Description |
| :--- | :--- | :--- | :--- | :--- |
| Intrinsic | 0.630 | 0.670 | 0.540 | Simultaneous generation of summary, citations, and phrases |
| Prior | 0.660 | 0.700 | 0.610 | Identify sentences/phrases first, then write summary |
| Post-hoc | 0.620 | 0.580 | 0.480 | Write summary first, then retrieve evidence |

The prior strategy showed the most significant advantages in the citation and phrase layers, reaching an S-F1 of 0.700 and a P-F1 of 0.610. The intrinsic strategy achieved an S-F1 of 0.670 and a P-F1 of 0.540, showing that simultaneous citation is feasible but prone to including sentences that do not actually support the summary. The post-hoc strategy was the weakest, with S-F1 at 0.580 and P-F1 at 0.480. This is likely because if the initial summary contains irrelevant or over-inferred information, the subsequent evidence retrieval is biased by these erroneous claims.

### Key Findings
- **High Data Quality**: Manual evaluation checked completeness and conciseness for summaries, citations, and phrases, with average scores across 16 aspects exceeding 4.6/5. Annotation consistency was strong (within-one rate of 97.4%, exact match rate of 92.1%, MAE of 0.109).
- **Performance Drops with Complex Contexts**: As the number of subclaims, citations, or contributory phrases in the reference increased, DeepSeek-V3's metrics declined, indicating that current LLMs remain unstable when handling medical summaries with multiple facts and pieces of evidence.
- **Difficulty Varies by Medical Aspect**: Intervention and Outcomes are harder due to high information density. Blinding, Funding, and Registration are typically short and explicit, making them easier for models to extract correctly.
- **Findings is a Major Bottleneck**: Performance for the Findings aspect was significantly lower (DeepSeek-V3: C-F1 0.382, S-F1 0.205, P-F1 0.092), as synthesized conclusions are harder to attribute accurately than structured fields.
- **Evidence-First is More Robust**: In case studies, the prior strategy successfully locked onto specific sentences and dosage phrases before generating the intervention summary, achieving 1.0 across F1 metrics. In contrast, the post-hoc strategy introduced irrelevant information about safety and tolerability, "polluting" the evidence chain.

## Highlights & Insights
- **Phrase-Level Verifiability**: While most attribution benchmarks focus on citation relevance, PCoA requires key phrases in cited sentences to align with the summary content, mimicking how doctors verify specific facts.
- **Diagnostic Evaluation Framework**: The three-tier metrics (claim, sentence, phrase) distinguish between "wrong summary," "wrong citation," and "unaligned phrase." These failures carry different risks in clinical applications.
- **Engineering Insights for Prior Attribution**: Generating summaries after retrieving evidence essentially constrains the LLM's generative space to a verifiable context, which is more effective at controlling hallucinations than post-hoc citations.
- **Clinically Relevant Aspects**: Including Funding, Registration, and Blinding alongside PICO makes the benchmark suitable for assessing information needed to evaluate the credibility of an RCT.
- **Explicit Reporting of Format Compliance**: A common pain point in structured output tasks is non-compliance; this paper’s transparent reporting and handling of invalid responses enhance the reproducibility of experimental details.

## Limitations & Future Work
- **Limited Scope**: PCoA currently only covers melanoma-related RCT abstracts. The generalizability to other diseases, observational studies, or full-text clinical papers remains to be verified.
- **Abstract-Only Constraint**: While abstracts are structured and accessible, many details are only found in the full-text results and methods sections. Future work could extend to full-text evidence attribution.
- **Reliance on Automatic Evaluators**: Claim decomposition and entailment checks depend on Mistral-Large-2411 and TRUE; errors in these evaluators may propagate to the final metrics.
- **Surface-Level Phrase Matching**: Phrase metrics rely on token overlap (similar to ROUGE-1) and do not fully account for paraphrasing, word order changes, or semantic equivalence.
- **Annotation Scale**: While 1,799 instances are sufficient for fine-grained evaluation, training robust models would require larger, multi-disease, and multi-lingual datasets.
- **Future Directions**: Prior attribution could be developed into a RAG pipeline where specialized models first lock onto aspect-specific evidence, followed by constrained generation by medical LLMs, incorporating semantic-level phrase matching for evaluation.

## Related Work & Insights
- **vs PICO Extraction**: Traditional PICO extraction identifies fields like Intervention or Outcome; PCoA goes further by requiring natural language summaries supported by sentence and phrase-level evidence, closer to how clinical readers consume information.
- **vs FactPICO**: FactPICO focuses on the factuality of RCT abstracts. PCoA differentiates itself by refining fact-checking to citations and contributory phrases, assessing both factuality and traceability.
- **vs DocLens**: PCoA adopts the claim-level recall/precision idea from DocLens but applies it to medical aspect-based summarization and adds sentence and phrase-level attribution metrics.
- **vs Intrinsic Self-Citation**: While self-citation allows simultaneous generation, PCoA results show it still produces invalid citations. The prior route (filtering evidence before generation) is safer for high-risk medical scenarios.
- **Insight**: For clinical summarization systems, the pipeline should not just optimize for final summary scores but should treat evidence selection, phrase extraction, generation, and verification as observable modules to ensure errors can be localized and corrected.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Proposes a phrase-level context attribution benchmark for medical aspect summarization, offering finer task definitions than standard summarization or citation tasks.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Data quality, model comparison, aspect-wise analysis, and strategy comparisons are comprehensive, though limited by disease scope and lack of model training.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure, thorough definitions of methods and metrics, and direct conclusions.
- Value: ⭐⭐⭐⭐⭐ High reference value for verifiable medical summarization, RAG evaluation, and citation-aware generation; particularly useful as a diagnostic benchmark for high-risk summarization systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CT-Flow: Orchestrating CT Interpretation Workflow with Model Context Protocol Servers](ct-flow_orchestrating_ct_interpretation_workflow_with_model_context_protocol_ser.md)
- [\[ACL 2026\] RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction](ra-rrg_multimodal_retrieval-augmented_radiology_report_generation_with_key_phras.md)
- [\[NeurIPS 2025\] Document Summarization with Conformal Importance Guarantees](../../NeurIPS2025/medical_nlp/document_summarization_with_conformal_importance_guarantees.md)
- [\[ACL 2026\] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)
- [\[ACL 2026\] CT-FineBench: A Diagnostic Fidelity Benchmark for Fine-Grained Evaluation of CT Report Generation](ct-finebench_a_diagnostic_fidelity_benchmark_for_fine-grained_evaluation_of_ct_r.md)

</div>

<!-- RELATED:END -->
