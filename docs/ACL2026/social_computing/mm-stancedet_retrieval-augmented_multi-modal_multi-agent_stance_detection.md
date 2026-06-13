---
title: >-
  [Paper Note] MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection
description: >-
  [ACL 2026][Social Computing][Multi-modal Stance Detection] The authors reformulate multi-modal stance detection as a 4-stage multi-agent pipeline: CLIP retrieval of similar samples providing few-shot CoT…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Multi-modal Stance Detection"
  - "Retrieval Augmentation"
  - "Multi-agent Debate"
  - "Self-reflection"
  - "Cross-modal Conflict"
date: 2026-05-08
content_hash: ede2fad7475d6c54
---

# MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection

**Conference**: ACL 2026  
**arXiv**: [2604.27934](https://arxiv.org/abs/2604.27934)  
**Code**: https://github.com/luweihai/MM-StanceDet  
**Area**: Multi-modal Stance Detection / Multi-agent / RAG  
**Keywords**: Multi-modal Stance Detection, Retrieval Augmentation, Multi-agent Debate, Self-reflection, Cross-modal Conflict

## TL;DR
The authors reformulate multi-modal stance detection as a 4-stage multi-agent pipeline: CLIP retrieval of similar samples providing few-shot CoT, three expert agents (text/image/cross-modal conflict) for analysis, three debater agents (pro/con/neutral) for multi-perspective argumentation, and a final adjudicator agent performing self-reflection to output the label. This approach outperforms strong baselines including GPT-4V, TMPT, and MV-Debate in both in-target and zero-shot settings across five datasets.

## Background & Motivation

**Background**: Stance detection has evolved from text-only (BERT/RoBERTa) to multi-modal (MSD). Existing work mainly falls into two categories: (1) simple feature fusion of independent encoders like BERT+ViT; (2) using prompt tuning (TMPT) to adapt pre-trained VLMs for stance feature extraction. Recently, using MLLMs like GPT-4V/Qwen-VL as zero-shot judges has emerged as a new direction.

**Limitations of Prior Work**: The authors identify three core bottlenecks: (1) **Contextual Grounding Void**: MLLMs tend to misjudge nuanced multi-modal signals when lacking concrete in-domain samples; (2) **Cross-Modal Interpretation Ambiguity**: When image and text signals conflict or complement each other, MLLMs often hallucinate or ignore the conflict (Zhang et al. 2024c empirically showed GPT-4V's defects in cross-modal consistency); (3) **Single-Pass Reasoning Fragility**: Directly prompting an LLM for a stance in a single-shot manner lacks a structured process for exploring alternative interpretations, leading to persistent errors.

**Key Challenge**: While single-pass "emergent reasoning" works in simple scenarios, it has low fault tolerance for sarcasm, conflict, and cross-modal nuances. Achieving stability requires explicitly modeling human-like decision processes involving "analysis-debate-reflection."

**Goal**: To construct a stance detection framework capable of robustly handling conflicting multi-modal signals purely through prompt orchestration, without fine-tuning any models.

**Key Insight**: Combine RAG (providing concrete examples), specialized agents (focusing on individual modalities), debate (forcing exploration of three stances), and self-reflection (preventing single-pass errors) into a holistic framework to address each bottleneck.

**Core Idea**: Transform "stance judgment" from a single-shot decision into a structured reasoning process via a 4-stage pipeline of multi-agent + RAG + debate + reflection.

## Method

### Overall Architecture
MM-StanceDet processes each input $x = (I, T, K)$ (image/text/target) through four sequential stages: (1) **Retrieval Augmentation** retrieves top-$k$ similar samples and their pre-generated CoT from a vector database $\mathcal{D}$; (2) **Multimodal Analysis** where Text-Agent, Image-Agent, and Modality-Conflict-Agent respectively output $A_{\text{text}}$, $A_{\text{image}}$, and $A_{\text{conflict}}$; (3) **Reasoning-Enhanced Debate** where three debater agents (support/oppose/neutral) receive all analyses and construct arguments for their respective stances; (4) **Self-Reflection and Adjudication** where a judge agent synthesizes the three arguments, original analyses, and critical reflection to produce the final label $\hat{y} \in \{-1, 0, 1\}$ and justification $J_{\text{final}}$. All agents use GPT-4o-mini by default; retrieval utilizes a CLIP encoder with ANN, with default $k=3$ and 3 rounds of debate.

### Key Designs

1.  **Retrieval Augmentation with Pre-generated CoT**:
    *   **Function**: Provides "concrete and comparable" few-shot samples for subsequent reasoning, mitigating LLM blind spots regarding domain-specific cues.
    *   **Mechanism**: CLIP encodes training samples $(I_j, T_j)$ into vectors $\mathbf{v}_j$ stored in a database. Each entry includes the label $y_j$ and an offline MLLM-generated CoT reasoning $C_j$ (explaining "why this sample holds this stance, focusing on image-text alignment and target relationship"). At query time, ANN retrieves $\mathcal{E}_{\text{retrieved}} = \text{ANN}(\mathbf{v}, \mathcal{D}, k)$.
    *   **Design Motivation**: Unlike traditional RAG retrieving raw text, CoT provides both "similar contexts" and "reasoning paradigms." CLIP's joint embedding leverages both modalities for retrieval, aligning closer to "multi-modal contextual similarity" than text-only search.

2.  **Three Expert Multimodal Analysis Agents**:
    *   **Function**: Decomposes multi-modal input into three complementary perspectives to prevent single-shot MLLMs from conflating information.
    *   **Mechanism**: Text-Agent $\mathcal{A}_{\text{text}}(T, K) \to A_{\text{text}}$ extracts keywords, sentiment, implicit sarcasm, and relevance to $K$; Image-Agent $\mathcal{A}_{\text{image}}(I, K) \to A_{\text{image}}$ describes visual objects, context, emotions, color symbolism, and symbolic elements; Modality-Conflict-Agent $\mathcal{A}_{\text{conflict}}(I, T, K, \mathcal{E}_{\text{retrieved}}) \to A_{\text{conflict}}$ specifically detects image-text conflict/complementarity, explicitly referencing retrieved CoT samples.
    *   **Design Motivation**: Assigning specific tasks to each agent avoids prompt overload. The Modality-Conflict-Agent is crucial—it transforms "cross-modal consistency" from an implicit assumption into an explicit output, significantly improving detection of sarcasm and cross-modal inconsistencies.

3.  **Debate + Self-Reflection Adjudication**:
    *   **Function**: Explores all stance possibilities through role-playing debate and identifies hidden flaws via self-reflection.
    *   **Mechanism**: Three debater agents representing support/oppose/neutral construct arguments $\text{Arg}_s = \mathcal{A}_s(I, T, K, A_{\text{text}}, A_{\text{image}}, A_{\text{conflict}})$. The judge agent then performs "critical self-assessment," checking for internal consistency and missed signals from $A_{\text{conflict}}$, drawing on ideas from Self-Refine and Reflexion. Output: $\hat{y}, J_{\text{final}} = \mathcal{A}_{\text{judge}}(\text{Arg}_{\text{support}}, \text{Arg}_{\text{oppose}}, \text{Arg}_{\text{neutral}}, x, A_{\text{text}}, A_{\text{image}}, A_{\text{conflict}})$.
    *   **Design Motivation**: Forced debate prevents the LLM from committing to a high-confidence incorrect stance. Self-reflection acts as a safety net, allowing the judge to transcend the trap where the "most persuasive" argument might not be the "most accurate."

### Loss & Training
This is a purely prompt-based framework. **It involves no model training**, only the orchestration of agents (prompt design, retrieval parameters, and debate rounds). All agents share a single LLM backbone (GPT-4o-mini by default; robustness experiments with other MLLMs confirm consistency).

## Key Experimental Results

### Main Results: In-target Macro F1 (%) Excerpts (5 datasets × 12 targets)

| Method | MTSE-DT | MWTWT-AC | MWTWT-AH | MRUC-RUS | MRUC-UKR | MTWQ-MOC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BERT | 48.25 | 63.05 | 59.24 | 41.25 | 46.80 | 57.77 |
| TMPT | 55.41 | 67.25 | 62.92 | 43.56 | 59.24 | 55.68 |
| GPT-4 + CoT | 69.12 | 70.10 | 72.05 | 42.03 | 54.21 | 58.48 |
| GPT-4 Vision | 70.46 | 57.47 | 57.90 | 44.83 | 56.42 | 66.72 |
| MV-Debate | 69.45 | 69.87 | 72.31 | 41.89 | 54.55 | 58.71 |
| BridgeTower | 68.53 | 67.92 | 65.44 | 43.26 | 58.19 | 68.06 |
| **Ours** | **70.12** | **71.93** | 66.50 | **48.34** | **64.02** | **68.13** |

Zero-shot gains are more significant: on MRUC-RUS, performance increases from GPT-4V's 42.09 to 45.37 (+3.3); on MRUC-UKR, from 47.00 to 55.25 (+8.2).

### Ablation Study and Agent Contribution Analysis

| Configuration | MTSE-DT | MWTWT-AC |
| :--- | :--- | :--- |
| Text Analysis Agent only | 67.52 | 63.30 |
| Image Analysis Agent only | 42.34 | 57.09 |
| Modality Conflict Agent only | 55.10 | 63.51 |
| Text + Image Analysis | 68.91 | 68.37 |
| Full MM-StanceDet | **70.12** | **71.93** |
| w/o Retrieval Augmentation (RA) | Significant drop | Significant drop |
| w/o Multimodal Analysis (MA) | **Largest drop** | **Largest drop** |
| w/o Reasoning-Enhanced Debate (RED)| Moderate drop | Moderate drop |
| w/o Self-Reflection (SRA) | Small drop | Small drop |

Noise robustness: Replacing 50% of top-3 retrieved samples with random ones on MTSE-DT only dropped performance from 70.12 to 68.92 (-1.2), proving debate and reflection can mitigate noise.

### Key Findings
*   **Multimodal Analysis is the largest contributor**: Removing the MA stage caused the sharpest decline, suggesting "expert specialization" is significantly stronger than letting a single LLM process all modalities at once.
*   **Strong Robustness in Retrieval Augmentation**: 50% retrieval noise only resulted in a 1.2 F1 point drop, due to the debate stage actively questioning relevance and self-reflection filtering errors.
*   **Significant Advantage in Zero-shot**: The +8.2 point Gain over GPT-4V on MRUC-UKR highlights that structured reasoning is particularly effective for OOD/low-resource scenarios.
*   **Highest Gains on Small, High-Conflict Datasets**: MRUC, with the highest cross-modal conflict rate (15%) and small sample size, saw the most significant improvement with MM-StanceDet.
*   **Backbone Agnostic**: The framework performs consistently across different MLLMs, indicating the multi-agent orchestration itself is effective.

## Highlights & Insights
*   **Clear Mapping of Components to Bottlenecks**: RA addresses grounding void, MA addresses modality ambiguity, and Debate+Reflection addresses single-pass fragility. This "problem-driven modularity" makes the method highly interpretable.
*   **Pre-generated CoT as Retrieval Units**: Unlike traditional RAG, entries include CoT reasoning templates, transferring "how to reason" as knowledge. This "analyist notes + case file" approach is reusable for complex judgment tasks (e.g., medical diagnosis).
*   **Explicit Output of Conflict Signals**: Making "modality consistency" a first-class output rather than an implicit judgment allows downstream modules to reason explicitly about it, a design directly applicable to fake news or meme detection.
*   **3-Round Debate is the Sweet Spot**: Beyond 3 rounds, gains diminish while costs double. Empirical sensitivity analysis provides clear engineering guidance.

## Limitations & Future Work
*   (1) High overhead: 4 stages × multi-agent reasoning costs ~4.8K tokens and 27s latency per sample, suitable for offline moderation rather than real-time processing. (2) Reliance on backbone: Biases or hallucinations in the backbone propagate to the final judgment. (3) Vector DB quality: Maintaining high-quality CoT libraries for low-resource languages or new domains is challenging.
*   **Observation**: Agents sharing the same backbone might create an "echo chamber." Using different models for different debaters could enhance robustness. Similarly, the judge's self-evaluation might suffer from self-bias.
*   **Future Directions**: Replace retrieval with a mixture (CoT-retrieval + counterfactual samples); introduce process reward models for real-time scoring of agent outputs.

## Related Work & Insights
*   **vs TMPT**: TMPT uses prompt tuning requiring parameter training; MM-StanceDet is training-free and significantly outperforms it (70.12 vs 55.41 on MTSE-DT).
*   **vs GPT-4 Vision + CoT**: Single MLLM with CoT is competitive in-target but fails in zero-shot settings; structured debate + reflection ensures stability.
*   **vs MV-Debate**: MV-Debate focuses on text; this work proves multi-agent debate combined with multi-modal analysis provides significant incremental value.
*   **Insight**: Any task requiring "cross-modal consistency judgment" (fake news, memes, sentiment) can adopt this 4-stage template. Trading 27s of latency for stability is a worthwhile trade-off for many offline analysis scenarios.

## Rating
*   Novelty: ⭐⭐⭐ (Combining existing multi-agent/RAG/debate patterns into MSD; Modality-Conflict-Agent is the distinct detail).
*   Experimental Thoroughness: ⭐⭐⭐⭐ (5 datasets, in-target/zero-shot, ablation, noise, and sensitivity analyses).
*   Writing Quality: ⭐⭐⭐⭐ (Clear mapping between problems and designs; insightful appendix on conflict rates).
*   Value: ⭐⭐⭐⭐ (Provides a strong training-free baseline for MSD; immediately applicable to social media monitoring/moderation).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition](../../AAAI2026/social_computing/cross-modal_prompting_for_balanced_incomplete_multi-modal_emotion_recognition.md)
- [\[ICML 2026\] MIND: Multi-Rationale Integrated Discriminative Reasoning Framework for Multi-Modal Fake News](../../ICML2026/social_computing/mind_multi-rationale_integrated_discriminative_reasoning_framework_for_multi-mod.md)
- [\[AAAI 2026\] T2Agent: A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search](../../AAAI2026/social_computing/t2agent_a_tool-augmented_multimodal_misinformation_detection_agent_with_monte_ca.md)
- [\[AAAI 2026\] Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering](../../AAAI2026/social_computing/multi-modal_dynamic_proxy_learning_for_personalized_multiple_clustering.md)
- [\[ACL 2026\] Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation](beyond_the_crowd_llm-augmented_community_notes_for_governing_health_misinformati.md)

</div>

<!-- RELATED:END -->
