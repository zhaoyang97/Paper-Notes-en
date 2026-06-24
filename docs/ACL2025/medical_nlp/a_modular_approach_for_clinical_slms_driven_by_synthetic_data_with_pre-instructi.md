---
title: >-
  [Paper Note] A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment
description: >-
  [ACL 2025][Medical LLM][Clinical Small Language Models] This paper proposes a modular framework to efficiently adapt Small Language Models (SLMs) into clinical domain models. It includes pre-instruction tuning for domain experts (training multiple expert models on medical corpora), model merging (combining multiple experts into a unified MediPhi), and clinical-task alignment based on 2.5 million synthetic instructions (MediFlow). Ultimately, the 3.8B-parameter MediPhi outperf…
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "Clinical Small Language Models"
  - "Synthetic Data"
  - "Pre-Instruction Tuning"
  - "Model Merging"
  - "Clinical-Tasks Alignment"
date: 2026-05-08
content_hash: b2bbc30471fb3659
---

# A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment

**Conference**: ACL 2025  
**arXiv**: [2505.10717](https://arxiv.org/abs/2505.10717)  
**Code**: None  
**Area**: Medical NLP  
**Keywords**: Clinical Small Language Models, Synthetic Data, Pre-Instruction Tuning, Model Merging, Clinical-Tasks Alignment

## TL;DR

This paper proposes a modular framework to efficiently adapt Small Language Models (SLMs) into clinical domain models. It includes pre-instruction tuning for domain experts (training multiple expert models on medical corpora), model merging (combining multiple experts into a unified MediPhi), and clinical-task alignment based on 2.5 million synthetic instructions (MediFlow). Ultimately, the 3.8B-parameter MediPhi outperforms GPT-4 on several clinical tasks.

## Background & Motivation

**Background**: Large language models like GPT-4 have demonstrated powerful capabilities in clinical scenarios, but their high computational cost and latency limit practical deployment in healthcare institutions. Small Language Models (SLMs, e.g., at the 3-4B parameter level) offer advantages in cost and latency, but their limited model capacity makes domain adaptation more difficult. Model adaptation in the medical domain faces two unique challenges: first, the specialized and diverse nature of medical knowledge (ranging from radiology reports to ICD coding, clinical guidelines, and drug information); second, the extreme scarcity and highly sensitive nature of clinical data.

**Limitations of Prior Work**: Existing clinical LLM adaptation methods typically adopt a "one-size-fits-all" strategy—fine-tuning uniformly on a mixed medical corpus. This approach performs poorly for today's SLMs, whose limited capacity makes it difficult to master all types of medical knowledge simultaneously in a single training run. Furthermore, high-quality annotated clinical data is extremely hard to obtain, and existing clinical NLP benchmarks have limited coverage.

**Key Challenge**: The conflict between the "smallness" of SLMs and the "breadth" of clinical tasks—how to enable a model with only 3.8B parameters to simultaneously handle over a dozen tasks of different natures, such as named entity recognition, radiology report generation, ICD coding, and clinical Q&A.

**Goal**: (1) To design a systematic framework for adapting SLMs into clinical models; (2) To construct a widely covered clinical evaluation benchmark; (3) To build a large-scale, high-quality synthetic clinical instruction dataset.

**Key Insight**: Drawing inspiration from the modular concept of "expert models + merging"—first training multiple expert SLMs for different medical subdomains respectively, then integrating them into a unified model via model merging techniques, and finally aligning them at the task level using synthetic data to achieve "divide and conquer, merge into one."

**Core Idea**: A three-stage pipeline comprising training domain expert models through pre-instruction tuning, unifying knowledge through model merging, and aligning tasks using synthetic data, adapting the 3.8B Phi-3.5-mini into the clinical model MediPhi that outperforms GPT-4.

## Method

### Overall Architecture

The construction of MediPhi involves three stages: (1) Pre-instruction tuning stage—continuously pre-training Phi-3.5-mini on different types of medical corpora, such as PMC (PubMed Central), medical guidelines, and MedWiki, to obtain multiple domain-expert models; (2) Model merging stage—using model merging techniques (such as TIES-Merging or DARE) to merge the parameters of multiple expert models into a unified MediPhi base model; (3) Clinical-tasks alignment stage—instruction fine-tuning and preference alignment of MediPhi using the MediFlow synthetic dataset through SFT and DPO.

### Key Designs

1. **Pre-Instruction Tuning**:

    - **Function**: To inject different types of medical domain knowledge on top of a general SLM, resulting in multiple complementary expert models.
    - **Mechanism**: Select multiple representative medical text corpora—PMC academic papers (covering the latest research knowledge), medical guidelines and textbooks (covering standardized clinical knowledge), and MedWiki (covering layperson-friendly translations of medical concepts), etc. For each corpus, independently perform continuous pre-training (next token prediction) on Phi-3.5-mini. This step does not use instruction-formatted data, hence the term "pre-instruction" tuning. Each expert model possesses deeper knowledge within its corresponding domain, though its performance may decline in other domains.
    - **Design Motivation**: The capacity limit of SLMs means that training on mixed corpora leads to "catastrophic interference" where knowledge from different domains conflicts. Training expert models separately allows each model to fully absorb a single type of knowledge, bypassing capacity bottlenecks.

2. **Model Merging**:

    - **Function**: To merge the parameters of multiple expert models into a single unified model, preserving the strengths of each expert.
    - **Mechanism**: Employs merging tactics in the parameter space, such as TIES-Merging (identifying key parameter changes of each expert relative to the base model, then merging these variations) or DARE (randomly dropping a portion of parameter differences to reduce conflicts). Without additional training, the merged MediPhi maintains performance close to that of each expert in their respective domains. Experiments show that the merged model's performance on the CLUE+ benchmark is no lower than the performance of any individual expert in its strongest domain.
    - **Design Motivation**: At inference time, only one model needs to be loaded instead of multiple experts, significantly reducing deployment complexity. Model merging acts as a "free lunch"—bringing multi-domain capabilities without increasing training or inference costs.

3. **MediFlow Synthetic Dataset and Clinical-Tasks Alignment**:

    - **Function**: To provide large-scale, high-quality, and widely covered clinical instruction data for final task alignment.
    - **Mechanism**: Constructs the MediFlow dataset, consisting of 2.5 million synthetic instructions spanning 14 medical NLP tasks (such as named entity recognition, relation extraction, text classification, summarization, Q&A, ICD coding, etc.) and 98 fine-grained document types (radiology reports, discharge summaries, progress notes, etc.), with support for JSON-formatted outputs. Data generation workflow: Starting from real clinical texts, GPT-4 is leveraged to generate high-quality instruction-output pairs, which then undergo quality filtering and deduplication. During the alignment phase, SFT is used to fine-tune MediPhi on MediFlow, followed by DPO to further align preferences. The CLUE+ benchmark doubles the size of the original CLUE benchmark, covering a wider range of clinical tasks and scenarios.
    - **Design Motivation**: Real annotated clinical data is extremely hard to obtain, making large-scale synthetic data the key to breaking data bottlenecks. The coverage of 14 tasks and 98 document types ensures the model's generalizability in real-world clinical scenarios.

### Loss & Training

Pre-instruction tuning utilizes the standard next-token prediction loss. The SFT phase uses the standard cross-entropy loss to train on instruction-output pairs. The DPO phase employs the direct preference optimization loss to further align model outputs with desired preferences. The three stages are executed serially, with the learning rate and training epochs adjusted independently for each stage.

## Key Experimental Results

### Main Results

| Task Category | Metric | MediPhi-SFT | Phi-3.5-mini | GPT-4-0125 | Gain vs Baseline |
|----------|------|-------------|--------------|------------|------------|
| Medical Entity Recognition | F1 | +64.3% | baseline | - | vs Phi-3.5-mini |
| Radiology Reports | BLEU/ROUGE | +49.5% | baseline | - | vs Phi-3.5-mini |
| ICD-10 Coding | Accuracy | +44.0% | baseline | GPT-4-14% | Outperforms GPT-4 by 14% |
| CLUE+ Comprehensive | Average | +18.9% | MediPhi-merge | - | SFT+DPO vs Merged |
| Clinical Q&A | Accuracy | 73.2% | 58.1% | 71.5% | MediPhi > GPT-4 |

### Ablation Study

| Configuration | CLUE+ Average | Explanation |
|------|-------------|------|
| Full MediPhi (SFT+DPO) | Optimal | Full three-stage pipeline |
| Merged Only (No SFT/DPO) | Moderate | Already possesses good domain knowledge |
| Single Expert (PMC) | Lower than merged | Only skilled in academic knowledge |
| Single Expert (Guidelines) | Lower than merged | Only skilled in clinical guideline knowledge |
| Direct SFT (No pre-instruction tuning) | Lower than full | Lacks domain foundation knowledge |
| SFT without DPO | Lower than full | DPO further enhances alignment |
| Original Phi-3.5-mini | Lowest | General-purpose model without medical adaptation |

### Key Findings

- During the pre-instruction tuning stage, each expert model showed significant improvements in its corresponding domain (medical entities +64.3%, radiology reports +49.5%), demonstrating that continuous pre-training on domain corpora is highly effective for SLMs.
- Model merging successfully preserved the strengths of each expert model, and the merged MediPhi performed no worse than the best expert in cross-domain evaluations, proving the viability of the "divide-and-conquer + merge" strategy.
- MediPhi outpaced GPT-4-0125 by 14% on the ICD-10 coding task, proving that targeted adaptation of a 3.8B SLM can defeat massive general-purpose models on specific clinical tasks.
- DPO alignment on MediFlow yielded an average improvement of 18.9% over SFT alone, indicating that preference alignment is crucial for the output quality of clinical NLP.

## Highlights & Insights

- The three-stage pipeline design of "pre-instruction tuning $\to$ model merging $\to$ task alignment" is elegant and generalizable. This paradigm can be adopted in any domain adaptation scenario—injecting domain-specific knowledge separately, merging them, and finally aligning them to tasks.
- The 2.5 million synthetic instructions dataset MediFlow, covering 14 tasks and 98 document types, is one of the most important resource contributions of this work. This systematic approach to constructing synthetic data can be extended to other data-scarce domains.
- The successful application of model merging as a "free lunch" in clinical SLMs is encouraging—unifying the capabilities of multiple experts without requiring additional training costs.

## Limitations & Future Work

- The pre-instruction tuning phase requires training a complete model independently for each corpus type, causing training costs to scale linearly when expanding to more subdomains.
- The quality of MediFlow synthetic data is bounded by the teacher model (GPT-4) used to generate it, potentially introducing bias.
- Currently, the approach is only validated on Phi-3.5-mini (3.8B). Whether the effects remain consistent when scaled to larger or smaller models remains unknown.
- Although CLUE+ expands coverage, it remains primarily English-focused; multilingual clinical scenarios have not been addressed.
- Deployment validation in real hospital environments (such as handling noisy EMRs, integrating with HIS systems, etc.) has not yet been conducted.

## Related Work & Insights

- **vs Med-PaLM**: Med-PaLM utilizes a much larger model (540B), whose computational cost is far higher than MediPhi (3.8B). MediPhi demonstrates that through careful domain adaptation, an SLM can achieve domain performance close to or even exceeding that of large models.
- **vs BioMistral/OpenBioLLM**: These medical LLMs generally undergo unified fine-tuning on mixed medical corpora. MediPhi's modular expert training + merging strategy is better suited for SLM capacity limits.
- **vs PMC-LLaMA**: PMC-LLaMA is trained solely on PMC literature, limiting its coverage; MediPhi achieves more comprehensive clinical capabilities through multi-source corpora experts and MediFlow alignment.

## Rating

- Novelty: ⭐⭐⭐⭐ High system-level novelty in the modular framework; the three-stage paradigm using "pre-instruction tuning + merging + alignment" is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on the expanded CLUE+ benchmark, covering a variety of clinical tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the framework and detailed implementation of the dataset construction process.
- Value: ⭐⭐⭐⭐⭐ An actionable clinical SLM solution geared toward practical deployment; the MediFlow dataset and the CLUE+ benchmark are both substantial resource contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RedactX: An LLM-Powered Framework for Automatic Clinical Data De-Identification](redactor_an_llm-powered_framework_for_automatic_clinical_data_de-identification.md)
- [\[ACL 2026\] CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](../../ACL2026/medical_nlp/cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)
- [\[ACL 2025\] Aligning AI Research with the Needs of Clinical Coding Workflows: Eight Recommendations Based on US Data Analysis and Critical Review](clinical_coding_eight_recommendations.md)
- [\[ACL 2026\] PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment](../../ACL2026/medical_nlp/principlismqa_a_philosophy-grounded_approach_to_assessing_llm-human_clinical_med.md)
- [\[ACL 2025\] ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents](reflectool_clinical_agent.md)

</div>

<!-- RELATED:END -->
