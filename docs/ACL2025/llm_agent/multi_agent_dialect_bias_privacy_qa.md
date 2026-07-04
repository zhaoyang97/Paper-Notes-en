---
title: >-
  [Paper Note] A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems
description: >-
  [ACL 2025][LLM Agent][Multi-Agent Collaboration] A dual-agent iterative collaborative framework is constructed, comprising a Dialect Agent (for dialect translation + review) and a Privacy Policy Agent (for domain-specific answering). By injecting dialectological linguistic knowledge via prompt engineering, this framework simultaneously improves the overall accuracy of privacy policy QA and cross-dialect fairness without any retraining.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Multi-Agent Collaboration"
  - "Dialect Bias"
  - "Privacy Policy QA"
  - "Fairness"
  - "Zero-Training"
date: 2026-05-08
content_hash: 9b9f31723a1fe812
---

# A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems

**Conference**: ACL 2025  
**arXiv**: [2506.02998](https://arxiv.org/abs/2506.02998)  
**Code**: None  
**Area**: LLM Agent  
**Keywords**: Multi-Agent Collaboration, Dialect Bias, Privacy Policy QA, Fairness, Zero-Training

## TL;DR
A dual-agent iterative collaborative framework is constructed, comprising a Dialect Agent (for dialect translation + review) and a Privacy Policy Agent (for domain-specific answering). By injecting dialectological linguistic knowledge via prompt engineering, this framework simultaneously improves the overall accuracy of privacy policy QA and cross-dialect fairness without any retraining.

## Background & Motivation

**Background**: Privacy policy QA systems help users extract key information from long privacy terms. Existing systems rely on LLMs like GPT-4o-mini, Llama 3.1, and DeepSeek-R1 to answer privacy-related questions using zero/few-shot prompting.

**Limitations of Prior Work**: The capability of LLMs to process non-standard English dialects (AAVE, Jamaican English, Welsh English, Indigenous English, etc.) is significantly weaker than standard American English (SAE). This bias is particularly dangerous in the privacy domain—marginalized communities are inherently more vulnerable to data collection and privacy violations (EPIC explicitly notes that communities of color are particularly harmed by surveillance, law enforcement, and algorithmic bias). If their dialects lead to poorer performance in QA systems, it means those who need privacy information most receive the worst service.

**Key Challenge**: Traditional methods for mitigating dialect bias (e.g., DADA, TADA) require dialect-specific training data and fine-tuning. However, collecting such data in sensitive domains like privacy is difficult, expensive, and ethically risky. The core problem is: how to adjust LLM prompting strategies to be fair to all dialect groups without collecting dialect training data?

**Goal**: Minimize the cross-dialect performance gap, defined as $\Delta(f) = \max_{d_i, d_j \in \mathcal{D}} |\Phi_{d_i}(f) - \Phi_{d_j}(f)|$, while maintaining overall accuracy.

**Key Insight**: Drawing inspiration from Human-Centered Design principles, systems should adapt to the linguistic backgrounds of users rather than forcing users to adapt to systems. This is achieved by utilizing the multi-dialect knowledge inherent in LLMs, activated through structured role-playing prompts.

**Core Idea**: Deconstruct the dialect bias mitigation problem into a multi-agent collaborative pipeline of "dialect translation $\rightarrow$ domain answering $\rightarrow$ intent review," replacing large-scale dialect fine-tuning with minimal injection of dialect background knowledge.

## Method

### Overall Architecture
Input: A user query $q_d$ in any English dialect + a privacy policy text segment $p$; Output: An accurate and fair answer $A$. The system consists of two roles, Dialect Agent and Privacy Policy Agent, which reach consensus through up to two rounds of iterative dialogue. The entire process requires no model training or fine-tuning.

### Key Designs

1. **Dialect Agent (Dialect Translator + Intent Guardian)**:

    - **Function**: (Step 1) Translates dialect queries into SAE while preserving the original intent and cultural nuances; (Step 2b/2c) Reviews whether the Privacy Agent's answer is faithful to the original intent of the user's dialect, and provides feedback for revision if unsatisfied.
    - **Mechanism**: Injects a concise linguistic profile of the target dialect (phonological features, grammatical rules, specific vocabulary, cultural background) into the prompt. For instance, Indian English is injected with descriptions of retroflex consonants and grammatical patterns, while Jamaican English is injected with non-rhotic pronunciation and unique verb structures. The translation quality is measured at BLEU = 46.5 and ROUGE-L = 80.5, with zero hallucinations across 500+ samples.
    - **Design Motivation**: LLMs are most thoroughly trained on SAE, and direct understanding of dialects can lead to semantic misunderstandings. Explicit translation allows the downstream Privacy Agent to always work in its most competent language. The Dialect Agent also serves as a reviewer, forming a "translation-review" closed loop to ensure that any information loss during translation is repaired through iteration.

2. **Privacy Policy Agent (Domain Expert)**:

    - **Function**: (Step 2a) Generates answers and reasoning grounds based on SAE queries and privacy policy texts; (Step 2b) Refines answers upon receiving feedback from the Dialect Agent.
    - **Mechanism**: Prompted as a domain expert in privacy policies, understanding standard classification taxonomies of data practices (First Party Collection, Third Party Sharing, Data Retention, User Choice/Control, etc.) to extract information accurately from policy texts.
    - **Design Motivation**: Separation of concerns—allowing the Privacy Agent to focus on the professionalism of legal text comprehension without simultaneously dealing with the complexity of dialect understanding. This division of labor keeps each agent working within its domain of expertise.

3. **Iterative Collaboration and Conflict Resolution Mechanism**:

    - **Function**: Triggers a correction loop of up to 2 rounds if the Dialect Agent finds the Privacy Agent's answer inconsistent with the dialect user's original intent.
    - **Mechanism**: The Dialect Agent receives the original dialect query, the policy text, and the Privacy Agent's answer along with its reasoning. If it determines that the answer deviates from the intent (e.g., ignoring actual concerns behind dialect-specific colloquial expressions), it provides specific feedback for the Privacy Agent to reconsider.
    - **Design Motivation**: Experiments show that iteration is indispensable. Under zero-shot settings, moving from Initial to Final increased the PrivacyQA F1 from 0.53 to 0.59. In 22.99% (zero-shot) to 31.75% (few-shot) of cases, the Dialect Agent overrode the initial answer of the Privacy Agent, with 63-72% of these overrides being correct.

### Loss & Training
No training process. In the few-shot setup, each agent uses 8 exemplars covering various dialects, query types, and policy scenarios. The LLM generation temperature is set to 0.3 (except for the Self-Consistency baseline, which uses 0.5).

## Key Experimental Results

### Main Results
Evaluated on PrivacyQA (35 mobile app policies, 1,750 questions, sentence selection task, F1 metric) and PolicyQA (115 website policies, 25,017 questions, span extraction task, token-F1 metric). Multi-VALUE was used to convert questions into 50 dialects, reporting results for the 5 weakest dialects.

| Method | SAE | RAAVE | Jamaican | Indigenous | Welsh | Average | Max Gap↓ |
|------|-----|-------|--------|--------|--------|------|-----------|
| GPT-4o-mini Zero | .394 | .344 | .332 | .329 | .312 | .335 | .093 |
| GPT-4o-mini Few | .605 | .573 | .562 | .555 | .547 | .565 | .058 |
| **MA-zero (ours)** | .601 | .588 | .578 | .587 | .592 | **.587** | **.025** |
| **MA-few (ours)** | **.611** | **.595** | **.596** | **.602** | **.592** | **.598** | **.019** |
| Llama3.1 Zero | .469 | .349 | .370 | .325 | .356 | .368 | .144 |
| Llama3.1 MA-few | .555 | .525 | .523 | .529 | .522 | .530 | .033 |
| DeepSeek-R1 MA-zero | .582 | .579 | .583 | .579 | .566 | .577 | **.017** |

### Ablation Study

| Configuration | PrivacyQA Initial → Final F1 | Description |
|------|----------------------|------|
| Zero-shot iteration | 0.53 → 0.59 (+11%) | Iterative collaboration yields significant gains |
| Few-shot iteration | 0.58 → 0.61 (+5%) | Exemplars reduce the gain from iteration |
| With dialect knowledge | 0.577 → 0.597 | Dialect knowledge primarily aids initial translation |
| Without dialect knowledge | 0.521 → 0.589 | Initial performance is poor without knowledge, but iteration partially compensates |
| Dialect Agent override rate | zero: 22.99%, few: 31.75% | 63-72% of overrides are beneficial, 19-24% are harmful |

### Key Findings
- **Zero-shot Multi-agent Outperforms Few-shot Baseline**: GPT-4o-mini MA-zero (0.587) > few-shot baseline (0.565), demonstrating that structured agent collaboration is more effective than simply stacking exemplars. This finding suggests that agent design is a better investment in scenarios lacking labeled data.
- **Maximum Gap Reduced by 80%**: From 0.093 to 0.019, nearly eliminating the performance gap across dialects.
- **SAE Also Benefits**: The framework not only helps marginalized dialects but also improves the performance of standard English (GPT-4o-mini: 0.394 → 0.601), showing that the iterative review mechanism itself improves answer quality.
- **Diminishing Marginal Utility of Dialect Knowledge**: While dialect knowledge improves initial translation accuracy (0.577 vs 0.521), the gap narrows from 0.056 to 0.007 after iteration, suggesting that the agent collaboration mechanism possesses self-correction capabilities.
- **Interesting Phenomenon in DeepSeek-R1**: It performs best on Hong Kong English rather than standard American English, implying that different LLMs exhibit different dialect preferences.

## Highlights & Insights
- **Zero-Training Fairness Intervention**: Achieved entirely through prompt engineering without any dialect training data. This paradigm can be deployed out-of-the-box in any NLP system serving diverse user groups (medical QA, legal consulting, educational assistance). The core insight is: rather than changing the model, make the model understand the user.
- **Dual-role Design of "Translation + Review"**: The same Dialect Agent performs both translation and review, leveraging a key insight—the translator understands the original intent best and is thus most qualified to judge whether the response is faithful. This "creator = reviewer" paradigm is globally applicable in agent system design.
- **Solid Quantitative Analysis**: In addition to reporting average metrics, the paper meticulously analyzes agent override rates (22-32%), override accuracy (63-72%), and translation quality (BLEU/ROUGE), providing rich insights into the mechanics of the framework.

## Limitations & Future Work
- **Limitations of Synthetic Dialect Data**: Multi-VALUE relies on rule-based dialect conversion, which may not fully reflect real-world dialect features like vocabulary creation, code-switching, and context dependency. Future work should validate on real-world dialect user data.
- **Lack of Dialect Detection**: The framework assumes that the dialect used by the user is known a priori. A real-world deployment would require an automatic dialect detection module, which introduces new privacy and ethical concerns.
- **Computational Cost**: Dual-agent + up to 2 rounds of iteration means 3-5 LLM calls per query, significantly increasing latency and cost.
- **Limited to English Dialects**: Whether this is applicable to Chinese dialects (e.g., Cantonese/Hokkien), Spanish variants, or Arabic dialects remains unverified.
- **Implicit Assumption of SAE as "Standard"**: Translating all dialects into SAE may reinforce linguistic hegemony. Future work should explore methods to directly enhance the multi-dialect comprehension capabilities of LLMs.

## Related Work & Insights
- **vs DADA/TADA (2023)**: These methods require dialect-specific training data and model adaptation modules. The proposed agent method can be deployed with zero training, though DADA/TADA theoretically learn deeper dialectal features; both have distinct advantages in data-abundant versus data-scarce scenarios.
- **vs LongAgent (2024)**: Also a multi-agent QA system, LongAgent resolves long-document partitioning issues (spatial dimension), whereas this work addresses linguistic diversity (user dimension). The two can be combined to form a "long-document + multi-dialect" solution.
- **vs Multi-VALUE (2023)**: Multi-VALUE provides a dialect simulation and evaluation framework but offers no mitigation solutions. This work proposes the first training-free dialect bias mitigation method evaluated on its framework.

## Rating
- Novelty: ⭐⭐⭐⭐ Modeling dialect fairness as agent collaboration is a novel angle, though the dual-agent architecture itself is not highly complex.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three LLMs, two datasets, 50 dialects, with comprehensive ablation and override rate analyses, though the dialect data is synthetic.
- Writing Quality: ⭐⭐⭐⭐ The motivation narrative exhibits strong social care, and the experimental analysis is thorough, though the prompt section is somewhat lengthy.
- Value: ⭐⭐⭐⭐ The intersection of fairness and privacy is of urgent practical importance, and the framework can be deployed immediately.
---
title: >-
  [Paper Notes] A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems
description: >-
  [ACL 2025][LLM Agent][Multi-Agent] Proposes a dual-agent framework (Dialect Agent + Privacy Policy Agent) that uses dialect-aware translation and iterative collaboration to eliminate performance gaps across English dialects in privacy policy QA systems, without retraining or dialect-specific fine-tuning, reducing the maximum inter-dialect performance gap by up to 82% on PrivacyQA and PolicyQA.
tags:
  - ACL 2025
  - LLM Agent
  - Multi-Agent
  - Dialect Bias
  - Privacy Policy QA
  - Fairness
  - LLM Collaboration
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration](mam_modular_multi-agent_framework_for_multi-modal_medical_diagnosis_via_role-spe.md)
- [\[ACL 2025\] MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection](mind_a_multi-agent_framework_for_zero-shot_harmful_meme_detection.md)
- [\[ACL 2025\] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)
- [\[ACL 2025\] Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks](agents_under_siege_breaking_pragmatic_multi-agent_llm_systems_with_optimized_pro.md)
- [\[ACL 2025\] Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](table_critic_multi_agent.md)

</div>

<!-- RELATED:END -->
