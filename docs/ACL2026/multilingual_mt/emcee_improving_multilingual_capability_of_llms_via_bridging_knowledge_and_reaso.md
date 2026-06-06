---
title: >-
  [Paper Note] EMCEE: Improving Multilingual Capability of LLMs via Bridging Knowledge and Reasoning with Extracted Synthetic Multilingual Context
description: >-
  [ACL2026][Multilingual & Machine Translation][multilingual prompting] EMCEE enables LLMs to extract synthetic multilingual context related to non-English queries from their own parameters…
tags:
  - "ACL2026"
  - "Multilingual & Machine Translation"
  - "multilingual prompting"
  - "synthetic context"
  - "LLM-as-a-Judge"
  - "low-resource languages"
  - "cultural knowledge"
date: 2026-05-08
content_hash: fb4cfec194e1e673
---

# EMCEE: Improving Multilingual Capability of LLMs via Bridging Knowledge and Reasoning with Extracted Synthetic Multilingual Context

**Conference**: ACL2026  
**arXiv**: [2503.05846](https://arxiv.org/abs/2503.05846)  
**Code**: https://github.com/hamin2065/EMCEE  
**Area**: Multilingual LLM / Prompting  
**Keywords**: multilingual prompting, synthetic context, LLM-as-a-Judge, low-resource languages, cultural knowledge

## TL;DR
EMCEE enables LLMs to extract synthetic multilingual context related to non-English queries from their own parameters, then merges context-enriched answers with CoT reasoning answers using an LLM-as-a-Judge approach. This significantly improves performance on low-resource languages across four multilingual tasks.

## Background & Motivation
**Background**: LLMs exhibit strong performance on English tasks, but often degrade on non-English queries due to English-centric pre-training corpora. Common remedies include translating queries into English, using English instructions for CoT, or integrating external retrieval to supplement background knowledge.

**Limitations of Prior Work**: Translation and English CoT are effective for reasoning-heavy problems like mathematics or natural sciences but frequently lose local context in knowledge-intensive questions involving linguistics, social sciences, or cultural common sense. External RAG depends on specific retrievers and external corpora; retrieved content might not align with the cultural nuances of the query.

**Key Challenge**: Multilingual queries typically involve two types of requirements: abstract reasoning and language/cultural/national background. A single path struggles to cover both categories simultaneously. Furthermore, routing between paths based on the query alone often leads to errors due to insufficient information in the initial query.

**Goal**: To construct a prompting framework that requires no external retrieval or additional training, allowing LLMs to generate both "context-enriched answers" and "reasoning-enhanced answers" and dynamically select the more appropriate output.

**Key Insight**: The authors observe that LLM parameters likely store a wealth of linguistic and cultural knowledge that is not explicitly activated during direct answering. Instead of translating all non-English questions, it is more effective to first require the model to "extract" relevant background knowledge in text form.

**Core Idea**: Extract synthetic multilingual context, then merge it with reasoning; the name EMCEE is derived from **E**xtracting synthetic **M**ultilingual **C**ontext and **E**ntirely **E**rging (Merging).

## Method
EMCEE is a pure prompting pipeline. It does not update model parameters or call external knowledge bases. Instead, it runs multiple LLM inferences: one to extract query-relevant context, one for standard CoT reasoning, and a final one for judging/merging. The core value lies not just in "spending more tokens," but in ensuring the two candidate answers originate from different information sources: one emphasizing cultural and linguistic background, the other emphasizing general reasoning.

### Overall Architecture
The input is a non-English native query. The first path directs the LLM to extract 3 to 5 sentences of synthetic context relevant to the query based on English instructions; this context can include cultural, historical, domain-specific, or local linguistic knowledge. This context is then prepended to the native query to produce a context-enriched response. The second path utilizes English CoT instructions to generate a reasoning-focused response without additional context. The third step passes both responses to an LLM-as-a-Judge, which evaluates their suitability regarding linguistic background, cultural context, and reasoning adequacy to select or synthesize the final answer.

### Key Designs
1.  **Synthetic Multilingual Context Extraction**:
    *   **Function**: Explicitly transforms implicit linguistic, cultural, or domain knowledge within the LLM into short-text context.
    *   **Mechanism**: Uses English instructions on the native query to ask the model to extract the background knowledge necessary to answer the question, usually limited to 3-5 sentences, demonstrated via few-shot examples. The extracted content comes from latent knowledge in model parameters rather than web retrieval.
    *   **Design Motivation**: The key to many low-resource language problems lies not in the length of the reasoning chain but in knowing local vocabulary, cultural entities, or social norms. Explicit extraction brings this information into the context window, reducing the probability of omission during generation.

2.  **Reasoning-Focused CoT Path**:
    *   **Function**: Maintains the advantages of English CoT in handling math, natural science, and commonsense reasoning.
    *   **Mechanism**: Parallel generation of a CoT answer without synthetic context, allowing the model to solve the problem using its existing reasoning capabilities. This ensures the system is not forced into the knowledge extraction path for problems requiring only logical inference.
    *   **Design Motivation**: Multilingual tasks are heterogeneous. Context extraction alone may offer limited help for reasoning problems or introduce irrelevant background; CoT alone cannot compensate for missing low-resource cultural knowledge. The parallel paths preserve the benefits of both.

3.  **LLM-as-a-Judge Merging**:
    *   **Function**: Performs dynamic selection between candidate answers to avoid hard routing errors based solely on the query.
    *   **Mechanism**: The judge examines both the query and the specific content of the context-enriched and reasoning-focused responses to determine which is more culturally appropriate and which is more logically sound. For example, in a Javanese question, Eng-CoT might incorrectly associate "pagupon" with a chicken coop, while the extraction path correctly identifies its relation to pigeons/doves; the judge then selects the correct option.
    *   **Design Motivation**: If the model decides between extraction or reasoning based only on the query, it might misjudge the question type before seeing the potential knowledge output. Comparing two fully generated answers is more stable as the judge has more evidence.

### Loss & Training
EMCEE has no training loss or parameter fine-tuning. In experiments, API model temperatures are set to 0.0, and open-source models like Llama use greedy decoding to minimize randomness. The default main model is GPT-4o-mini, evaluated on M3-Exam, MKQA, XNLI, and XCOPA. Accuracy is used for M3-Exam/XNLI/XCOPA, and span-level F1 for MKQA. Results are reported separately for high-resource and low-resource languages based on Native-Basic performance.

## Key Experimental Results

### Main Results
The main experiments compare multiple multilingual prompting baselines using GPT-4o-mini. The table below highlights the All/Low resource trends; EMCEE achieved the highest or tied for highest scores across all metrics.

| Method | M3-Exam All | M3-Exam Low | MKQA All | MKQA Low | XNLI All | XNLI Low | XCOPA All | XCOPA Low |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native-Basic | 65.2 | 57.7 | 44.1 | 38.5 | 66.2 | 58.4 | 79.3 | 61.4 |
| Eng-CoT | 74.6 | 67.3 | 49.4 | 49.3 | 73.2 | 72.7 | 90.5 | 83.8 |
| XLT | 70.4 | 63.8 | 51.1 | 51.5 | 72.6 | 71.0 | 91.1 | 85.4 |
| RAG (Eng) | 72.1 | 63.9 | 44.7 | 44.5 | 70.4 | 69.7 | 87.9 | 80.6 |
| EMCEE (Route) | 76.2 | 69.2 | 50.8 | 49.8 | 73.1 | 72.3 | 90.5 | 83.8 |
| EMCEE | 77.4 | 71.5 | 52.3 | 52.4 | 74.3 | 73.9 | 92.0 | 86.2 |

The average relative improvement of EMCEE over Native-Basic was 16.4%, reaching 31.7% for low-resource languages. Specific relative gains for low-resource tasks include: M3-Exam 23.7%, MKQA 36.1%, XNLI 27.7%, and XCOPA 40.4%.

### Ablation Study
Ablation on M3-Exam decomposes the CoT, ExT (Extraction), and MeR (Merging) components. ExT alone approaches Eng-CoT, but the full EMCEE pipeline provides the largest gain for low-resource languages.

| Configuration | CoT | ExT | MeR | All / High / Low |
| :--- | :--- | :--- | :--- | :--- |
| Native-Basic | ✗ | ✗ | ✗ | 65.2 / 72.7 / 57.7 |
| Eng-CoT | ✓ | ✗ | ✗ | 74.6 / 81.8 / 67.3 |
| Extraction only | ✗ | ✓ | ✗ | 74.7 / 82.0 / 67.5 |
| CoT + MeR variant | ✓ | ✗ | ✓ | 75.2 / 83.4 / 67.1 |
| EMCEE | ✓ | ✓ | ✓ | 77.4 / 83.3 / 71.5 |

### Analysis
| Experiment | Comparison | EMCEE Result | Key Insight |
| :--- | :--- | :--- | :--- |
| GPT-4o M3-Exam | Native-Basic 78.1 | 85.7 | 8.9% relative gain |
| Claude-Haiku M3-Exam | Native-Basic 67.4 | 75.6 | 10.8% relative gain |
| Llama-3.1-8B M3-Exam | Native-Basic 49.8 | 56.9 | XLT/CoT performed weaker on this model |
| GlobalOpinionQA | Native-Basic 65.3 | 69.0 | Low-resource countries rose from 53.7 to 60.4 |
| Aya-8B | Native-Basic 46.0 | 49.8 | Average gains on multilingual-specialized models |
| Qwen3-8B w/o Think | Native-Basic 37.8 | 67.3 | Extraction is more critical than think-mode |
| Cost | 3x Eng-CoT: $0.149 | EMCEE: $0.140 | EMCEE has higher input tokens but lower total cost and output tokens |

### Key Findings
*   The benefits of EMCEE are concentrated in low-resource languages and cultural knowledge tasks, rather than simply being a result of increased reasoning iterations.
*   RAG (Native/Eng) performed worse than EMCEE in several tasks, suggesting that external retrieved content may be less effective than query-aligned context extracted internally.
*   EMCEE (Route) is weaker than full EMCEE, supporting the idea that comparing generated candidates is superior to routing based only on the query.
*   Failure cases reveal a pattern: for globally known entities, extraction might mistakenly apply local cultural knowledge (e.g., misassociating "Wake Me Up Before You Go-Go" with a Japanese singer instead of the band Wham!).

## Highlights & Insights
*   The paper decomposes multilingual prompting into "knowledge elicitation" and "reasoning selection" rather than focusing on prompt tuning for translation or language selection.
*   Synthetic context serves a unique role: it is not an external fact-base, but a mechanism to make the model explicitly state background it knows but might overlook during direct generation.
*   The superiority of merging over routing is of practical value. In complex queries, it is difficult to determine if a problem requires knowledge or reasoning beforehand, but much easier to spot contextual inconsistencies when comparing two candidate explanations.
*   The cost analysis refutes the notion that EMCEE is stronger simply via "more calls," as 3x Eng-CoT + Merge was both more expensive and less accurate.

## Limitations & Future Work
*   Multiple LLM inferences increase computational cost and latency; despite being more efficient than repeated CoT, it remains more expensive than single-turn prompting.
*   The extraction step carries the risk of irrelevant contextualization, which can mislead the model on global or general knowledge questions.
*   The method depends entirely on internal knowledge; if the model lacks knowledge of a specific low-resource culture, synthetic context may result in confident hallucinations.
*   Integration with RAG could mitigate knowledge gaps but would change the "pure prompting" nature and require stricter retrieval quality control.
*   For open-ended subjective questions, the cultural bias of the judge model remains a factor requiring more granular evaluation across demographic groups.

## Related Work & Insights
*   **vs XLT**: XLT improves tasks by translating to English and reasoning in English; EMCEE retains the native query while extracting language/cultural context.
*   **vs Machine Translation**: MT can improve basic understanding but often misses local semantics; EMCEE generates context around the original query to minimize translation loss.
*   **vs RAG**: RAG retrieves external passages; quality depends on the retriever. EMCEE extracts context internally—lighter and more query-aligned, though bounded by internal knowledge.
*   **vs multi-agent debate**: EMCEE's merge functions by comparing candidates from different information sources rather than iterating through consensus, a design applicable to specialized domains like medical QA or cross-cultural recommendations.

## Rating
*   **Novelty**: ⭐⭐⭐⭐☆ The combination of synthetic context extraction and LLM-as-a-Judge merging is clear and insightful without requiring complex architecture changes.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers four benchmarks, low/high resource splits, cross-model analysis, cost, failure modes, and extensive appendices.
*   **Writing Quality**: ⭐⭐⭐⭐☆ Intuitive examples, comprehensive tables, and honest discussion of method boundaries.
*   **Value**: ⭐⭐⭐⭐⭐ Highly practical for multilingual LLM applications, especially in scenarios lacking external retrieval resources but requiring cultural nuance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?](why_do_multilingual_reasoning_gaps_emerge_in_reasoning_language_models.md)
- [\[ACL 2026\] DFKI-MLT at SemEval-2026 TASK 7: Steering Multilingual Models Towards Cultural Knowledge](dfki-mlt_at_semeval-2026_task_7_steering_multilingual_models_towards_cultural_kn.md)
- [\[ACL 2026\] Prosody as Supervision: Bridging the Non-Verbal–Verbal for Multilingual Speech Emotion Recognition](prosody_as_supervision_bridging_the_non-verbal--verbal_for_multilingual_speech_e.md)
- [\[ACL 2026\] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)
- [\[AAAI 2026\] Bridging the Multilingual Safety Divide: Efficient, Culturally-Aware Alignment for Global South Languages](../../AAAI2026/multilingual_mt/bridging_the_multilingual_safety_divide_efficient_culturally-aware_alignment_for.md)

</div>

<!-- RELATED:END -->
