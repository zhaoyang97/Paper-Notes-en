---
title: >-
  [Paper Note] Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised RL Approach
description: >-
  [ACL 2026][Medical NLP][Medical Reasoning] This paper proposes the MedSSR framework, which efficiently enhances the medical reasoning capabilities of LLMs through controllable data synthesis injected with rare disease kn…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Medical Reasoning"
  - "Rare Diseases"
  - "Data Synthesis"
  - "Semi-supervised Reinforcement Learning"
  - "GRPO"
date: 2026-05-08
content_hash: d1a96a7d1f0cde19
---

# Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised RL Approach

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.11547](https://arxiv.org/abs/2604.11547)  
**Code**: [https://github.com/tdlhl/MedSSR](https://github.com/tdlhl/MedSSR)  
**Area**: Medical Imaging  
**Keywords**: Medical Reasoning, Rare Diseases, Data Synthesis, Semi-supervised Reinforcement Learning, GRPO

## TL;DR
This paper proposes the MedSSR framework, which efficiently enhances the medical reasoning capabilities of LLMs through controllable data synthesis injected with rare disease knowledge and a semi-supervised training paradigm of "self-supervised RL → supervised RL." It achieves a maximum improvement of +5.93% on rare disease tasks, breaking the existing +3% improvement ceiling.

## Background & Motivation

**Background**: The development of LLMs in medical reasoning is limited by the scarcity of high-quality reasoning data. Existing methods primarily initialize policy models by distilling Chain-of-Thought (CoT) reasoning chains from large closed-source models like GPT-4o, followed by RL training.

**Limitations of Prior Work**: (1) Only 22% of questions in existing medical benchmarks are reasoning-intensive, with only 3% involving rare diseases; (2) Distilling long reasoning chains from closed-source models is expensive; (3) Existing methods fail to exceed a +3% improvement cap on rare diseases even with fully supervised GRPO; (4) Privacy constraints and expertise requirements make acquiring complex medical reasoning data challenging.

**Key Challenge**: Rare disease data is extremely scarce, and the data distribution of existing methods is limited by available annotated data, leading to a low improvement ceiling. Additionally, synthetic data may contain factual errors, which are unacceptable in medical scenarios.

**Goal**: Efficiently improve LLM performance across a wide range of medical reasoning tasks, including rare diseases, without relying on expensive reasoning chain distillation.

**Key Insight**: (1) Synthesize questions only (rather than long reasoning chains) to significantly reduce generation costs; (2) Inject rare disease knowledge to control the distribution of synthetic data; (3) Generate pseudo-labels using the policy model itself to avoid reliance on external models.

**Core Idea**: Synthesize distribution-controllable medical reasoning questions (via rare disease knowledge injection), generate pseudo-labels through majority voting of the model itself, and then execute a curriculum training of "self-supervised RL → supervised RL."

## Method

### Overall Architecture
MedSSR consists of two synergistic components: (1) A medical knowledge-enhanced data synthesis pipeline—synthesizing new questions from seed questions, controlling the rare disease proportion via threshold $\alpha$, and generating pseudo-labels using the policy model itself; (2) A semi-supervised RL training strategy—first performing self-supervised RL (intrinsic learning) on pseudo-labeled synthetic data, followed by supervised RL (extrinsic learning) on human-annotated real data.

### Key Designs

1.  **Knowledge-enhanced Data Synthesis**:
    - **Function**: Generate distribution-controllable medical reasoning questions, specifically increasing the proportion of rare disease problems.
    - **Mechanism**: Given two seed questions $\{x_1^s, x_2^s\}$, GPT-4.1 is used to synthesize new questions. The rare disease proportion is controlled by threshold $\alpha$: sampling $\rho \sim \text{Uniform}(0,1)$, if $\rho < \alpha$, an entity $e$ is selected from a rare disease list, and MedCPT is used to retrieve top-k related medical documents $\mathcal{C}(e)$ for injection into the synthesis prompt. Only questions are synthesized (no reasoning chains), keeping the API token cost per sample much lower than distillation methods.
    - **Design Motivation**: Direct synthesis of reasoning chains is costly and error-prone. Synthesizing only questions allows the policy model to use its own reasoning capability for answers, avoiding dependency on external model reasoning quality. Knowledge injection ensures the medical accuracy of synthetic questions.

2.  **Pseudo-label Generation and Quality Control**:
    - **Function**: Generate reliable answer labels for synthetic questions for RL training.
    - **Mechanism**: The policy model (base model) samples multiple answers offline for each synthetic question, and the most consistent answer is selected as the pseudo-label via majority voting. Pseudo-labels with confidence below a threshold are discarded.
    - **Design Motivation**: Labeling with external models may introduce distribution mismatch (reward hacking). Self-labeling ensures data matches the model's learning trajectory. Majority voting provides natural quality filtering.

3.  **Semi-supervised RL Training Strategy**:
    - **Function**: Effectively utilize the complementary advantages of synthetic and real data to achieve curriculum learning from intrinsic to extrinsic.
    - **Mechanism**: A two-stage curriculum: (a) Self-supervised RL: training with GRPO on pseudo-labeled synthetic data to let the model learn from its own knowledge and reasoning (intrinsic learning), expanding knowledge coverage especially for rare diseases; (b) Supervised RL: training with GRPO on human-annotated real data (extrinsic learning) to calibrate and consolidate reasoning capabilities.
    - **Design Motivation**: Direct supervision on synthetic data may be unstable due to pseudo-label noise. The curriculum design of self-supervised exploration followed by supervised refinement allows the model to learn broadly before precise calibration.

### Loss & Training
Optimized using GRPO, with the validation reward $r(y, y') = \mathbb{I}[\text{ans}(y') = y]$. KL divergence constrains deviations from the reference policy. Validated on Qwen3-8B and Llama-3.1-8B-Instruct.

## Key Experimental Results

### Main Results

| Method | General Med. Gain | Rare Disease Gain | API Token Cost per Sample |
| :--- | :--- | :--- | :--- |
| HuatuoGPT-O1 | Medium | <3% | High (Long CoT) |
| MedReason | Medium | <3% | High |
| Fully Supervised GRPO | Medium | <3% | Low |
| **Ours** (Llama) | **+3.91%** | **+5.93%** | Low (Questions only) |
| **Ours** (Qwen3) | Significant | Breaks 3% Cap | Low |

### Ablation Study

| Configuration | General | Rare Disease | Description |
| :--- | :--- | :--- | :--- |
| Full MedSSR | Optimal | Optimal | Complete framework |
| w/o Knowledge Injection | Decrease | Significant Decrease | Insufficient rare disease data |
| w/o Self-supervised RL | Decrease | Decrease | Lack of synthetic data coverage |
| w/o Pseudo-label Filter | Decrease | Decrease | Noise labels affect training |
| Single-stage Mixed | Lower than 2-stage | Lower than 2-stage | Necessity of curriculum design |

### Key Findings
- MedSSR is the first method to break the +3% improvement ceiling on rare disease tasks, reaching +5.93%.
- Synthesizing only questions (without reasoning chains) effectively enhances reasoning capabilities while significantly reducing costs.
- The two-stage curriculum of semi-supervised RL outperforms single-stage mixed training, validating the "broad then precise" strategy.
- Title $\alpha$ for rare disease knowledge injection provides precise control over data distribution.
- Comprehensively outperforms existing methods across 10 medical benchmarks.

## Highlights & Insights
- **Synthesizing Questions, Not Answers**: Ingeniously simplifies high-cost "question + reasoning chain" synthesis into low-cost "question-only" synthesis, leveraging the policy model's own reasoning for answers. This drastically reduces dependence on closed-source APIs.
- **Self-bootstrapping via Pseudo-labels**: Using the model's own majority voting for pseudo-labels is an elegant bootstrapping strategy that ensures training data aligns with model capabilities.
- **Controllable Distribution Synthesis**: The $\alpha$ threshold allows for precise control over the proportion of rare disease data, providing a direct tool to address long-tail distributions in medicine.

## Limitations & Future Work
- Pseudo-label quality depends on the policy model's own capability—if the model is entirely ignorant of a rare disease, pseudo-labels may be unreliable.
- The coverage of the rare disease knowledge base may be limited; uncovered diseases remain difficult for question generation.
- Validated only on 8B-scale models; effectiveness on larger models is unknown.
- The diversity of synthetic questions is limited by the quality and quantity of seed questions.

## Related Work & Insights
- **vs HuatuoGPT-O1**: Distills GPT-4o reasoning chains + SFT + RL; high cost with limited rare disease improvement. MedSSR synthesizes only questions, lowering cost with significant rare disease gains.
- **vs MedReason**: Uses Knowledge Graphs to improve factual accuracy in CoT generation but still relies on long-chain distillation. MedSSR ensures accuracy at the synthesis stage via knowledge injection.
- **vs Self-Instruct**: A general self-instruction synthesis method; MedSSR adds knowledge retrieval and distribution control specifically for the medical domain.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of "question synthesis + self-pseudo-labeling + semi-supervised RL" is a novel and efficient paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 medical benchmarks, two base models, comprehensive ablation and comparison.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, precise problem definition (the 3% ceiling for rare diseases).
- Value: ⭐⭐⭐⭐⭐ Provides a practical and efficient solution for data scarcity in medical LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Ryze: Evidence-Enriched Data Synthesis from Biomedical Papers](ryze_evidence-enriched_data_synthesis_from_biomedical_papers.md)
- [\[ACL 2026\] MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning](multidx_a_multi-source_knowledge_integration_framework_towards_diagnostic_reason.md)
- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](../../CVPR2026/medical_nlp/towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)
- [\[ACL 2026\] PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment](principlismqa_a_philosophy-grounded_approach_to_assessing_llm-human_clinical_med.md)
- [\[ACL 2026\] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)

</div>

<!-- RELATED:END -->
