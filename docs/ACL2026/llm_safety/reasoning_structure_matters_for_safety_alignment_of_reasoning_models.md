---
title: >-
  [Paper Note] Reasoning Structure Matters for Safety Alignment of Reasoning Models
description: >-
  [ACL 2026][LLM Safety][Large Reasoning Models] The paper points out that the safety issues in Large Reasoning Models (LRMs) stem from a reasoning structure of "first understanding the problem…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Large Reasoning Models"
  - "Safety Alignment"
  - "Reasoning Structure"
  - "AltTrain"
  - "SFT"
date: 2026-05-08
content_hash: 8fd8e2ee0f809491
---

# Reasoning Structure Matters for Safety Alignment of Reasoning Models

**Conference**: ACL 2026  
**arXiv**: [2604.18946](https://arxiv.org/abs/2604.18946)  
**Code**: https://github.com/yeonjun-in/R1-Alt  
**Area**: LLM Reasoning / LLM Safety Alignment / Post-training  
**Keywords**: Large Reasoning Models, Safety Alignment, Reasoning Structure, AltTrain, SFT

## TL;DR
The paper points out that the safety issues in Large Reasoning Models (LRMs) stem from a reasoning structure of "first understanding the problem, then solving with full effort." It proposes AltTrain, which uses 1K SFT samples to modify the reasoning structure to "problem understanding → harmfulness assessment → conditional reasoning," significantly reducing harmful responses while largely preserving reasoning capabilities.

## Background & Motivation
**Background**: LRMs such as R1 and the o-series have achieved significant improvements in mathematics, code, and complex logical tasks through long-chain reasoning. Their training trajectories typically encourage the model to understand the problem first, followed by multi-step solving, checking, and correction.

**Limitations of Prior Work**: This same reasoning capability becomes a risk when facing malicious requests. Existing research has found that models after reasoning post-training may be more prone to solving towards the user's goal than standard instruction models, even if they recognize the request is problematic.

**Key Challenge**: Safety models need to stop assisting on harmful requests while fully utilizing reasoning capabilities for benign requests. Simple refusal training harms capability, and simply prompting the model to judge intent is insufficient to change its underlying reasoning inertia.

**Goal**: To explain why LRMs continue to solve tasks even after identifying risks, and to design a lightweight post-training method that explicitly changes the model's reasoning structure instead of just adding refusal templates to the output layer.

**Key Insight**: The authors argue that the root cause is not that the model "does not know it is harmful," but that the training-formed reasoning structure excessively prioritizes task solving. As long as the structure remains "problem understanding → solution reasoning," the model tends to treat any request as a problem to be completed.

**Core Idea**: The key to safety alignment is rewriting the reasoning process, causing the model to insert a harmfulness assessment before formal solving and choose to refuse or continue reasoning based on the assessment result.

## Method
The contribution of AltTrain lies in its restraint: it does not design complex RL or train a reward model. Instead, it constructs 1K SFT samples with a fixed reasoning structure, teaching the LRM to first understand the problem, then judge if the request is harmful, and finally perform conditional reasoning within its internal chain. This structure preserves the "problem understanding" familiar to the original LRM while adding a safety decision point, thereby reducing distribution shift.

### Overall Architecture
The training data AltTrain-1K comes from the SafeChain dataset, containing approximately 900 harmful queries and 100 benign queries. Each sample response consists of a reasoning chain and a final answer, with the reasoning chain carried by the model's original `<think>` template.

For each query, AltTrain sequentially collects three parts: first, **problem understanding**, extracted from the first sentence of the R1 original reasoning trajectory to maintain the familiar starting structure; second, **harmfulness assessment**, where an LLM like GPT-4o provides a one-sentence judgment on the request's harmfulness with a reason; third, **conditional reasoning**, where the model immediately ends further solving and refuses if harmful, or continues the remainder of the R1 original reasoning chain if benign.

### Key Designs
1.  **Problem Understanding Preserves Original Structure**:
    *   **Function**: Minimizes the damage of safety training on original reasoning capabilities.
    *   **Mechanism**: The authors analyzed 1,000 R1 reasoning trajectories in SafeChain and found 985 included "problem understanding" in the first paragraph. Thus, AltTrain retains the first sentence as a structural anchor.
    *   **Design Motivation**: Starting directly from a harmfulness assessment would cause the model to deviate from its learned reasoning distribution, potentially leading to performance degradation in math or code tasks.

2.  **Query-level Harmfulness Assessment**:
    *   **Function**: Forces the model to explicitly check user intent before solving.
    *   **Mechanism**: Every reasoning chain inserts a high-level harmfulness judgment after problem understanding. This judgment does not depend on the specific attack unfolding but determines if the current request should be assisted.
    *   **Design Motivation**: Preliminary analysis showed that LRMs can almost always identify harmful queries, but there is a disconnect between identification and behavior. Placing the assessment into a fixed structure ensures the identification truly controls subsequent reasoning.

3.  **Conditional Reasoning and Lightweight SFT**:
    *   **Function**: Adopts different reasoning paths for harmful and benign requests.
    *   **Mechanism**: If the assessment is harmful, the model stops task solving and provides a safety refusal; if benign, it continues complete reasoning. Training requires only 1K samples and takes approximately 60 minutes for an 8B model on a single A6000.
    *   **Design Motivation**: Safety alignment does not require retraining all capabilities; the key is providing the model with a stable decision structure. Once the structure is learned, small-scale data can generalize across model sizes and attack scenarios.

### Loss & Training
AltTrain uses standard supervised fine-tuning without RL, DPO, or reward models. The training objective is to maximize the likelihood of the structured response. The paper emphasizes data and token efficiency: R1-Alt averages 167 tokens per training sample and 69 tokens per inference query, shorter than SafeChain and STAR-1. The authors believe efficiency comes from the structure itself rather than simple text truncation, as removing any key step leads to significant degradation.

## Key Experimental Results

### Main Results
The main experiments evaluate harmful response rates, over-refusal rates, and reasoning capabilities across several R1/S1 backbones. Representative average metrics are selected below (Harmfulness and Over-refusal: lower is better; Reasoning: higher is better).

| Backbone | Method | Data Volume | Harmful Avg. ↓ | Over-refusal ↓ | Reasoning Avg. ↑ | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R1-7B | No train | - | 82.2 | 0.0 | 72.6 | Original model is strong but high-risk |
| R1-7B | R1-Alt | 1K | 14.3 | 31.6 | 69.5 | Risk drops greatly; over-refusal rises |
| R1-8B | No train | - | 83.5 | 0.4 | 58.1 | Original harmful rate is very high |
| R1-8B | R1-Alt | 1K | 4.8 | 14.0 | 59.8 | Good balance of safety and reasoning |
| R1-32B | No train | - | 82.5 | 0.0 | 76.0 | Large models share structure issues |
| R1-32B | R1-Alt | 1K | 3.7 | 11.2 | 78.0 | Safety improves significantly; capability rises slightly |
| S1-14B | No train | - | 89.7 | 0.0 | 65.9 | S1 series is also high-risk |
| S1-14B | S1-Alt | 1K | 6.7 | 14.0 | 64.8 | Method is effective across backbones |

The authors also checked QA, multilingual, and summarization capabilities; R1-Alt largely retains the original model's performance.

| Method | NQ ↑ | CMMLU ↑ | CNN ROUGE ↑ | Note |
| :--- | :--- | :--- | :--- | :--- |
| No train | 71.7% | 61.8% | 12.3 | Original LRM |
| SafeChain | 73.9% | 59.7% | 13.8 | General capability is preserved after safety training |
| STAR-1 | 72.3% | 59.0% | 14.3 | Higher summarization metrics |
| R1-Alt | 72.0% | 60.5% | 13.6 | Overall close to the original model |

### Ablation Study
Structural ablation shows all three steps are important. Removing HA makes safety significantly worse; removing PU or CR harms reasoning or increases over-refusal.

| Variant | R1-8B Harmful Avg. ↓ | R1-8B Over-refusal ↓ | R1-8B Reasoning Avg. ↑ | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| w/o PU | 1.6 | 14.0 | 56.3 | Strong safety but reasoning drops; original anchors help |
| w/o HA | 15.6 | 19.8 | 59.4 | Safety deteriorates significantly without judgment |
| w/o CR | 4.1 | 18.0 | 59.3 | Missing conditional branches affects calibration |
| CR Rephrase | 5.6 | 10.4 | 59.9 | Still effective with new phrasing; not template memorization |
| R1-Alt | 4.8 | 14.0 | 59.8 | Three-step structure achieves balance |

Data volume analysis indicates that over-refusal issues in AltTrain can be mitigated by expanding data.

| Training Data | R1-8B Harmful Avg. ↓ | R1-8B Over-refusal ↓ | R1-8B Reasoning Avg. ↑ | Observation |
| :--- | :--- | :--- | :--- | :--- |
| AltTrain-0.5K | 4.6 | 22.0 | 60.4 | Higher over-refusal with less data |
| AltTrain-1K | 4.8 | 14.0 | 59.8 | Default setting |
| AltTrain-3K | 4.7 | 2.4 | 58.7 | Significant drop in over-refusal; capability maintained |

### Key Findings
*   The problem with LRMs is not a complete failure to recognize harmful intent, but rather continuing to push forward along the solving structure after recognition.
*   Safety alignment requires changing the **reasoning structure**, rather than merely adding refusal rules at the output layer.
*   1K small data samples are sufficient for the model to learn the structure; expanding to 3K can significantly reduce over-refusal.
*   AltTrain is effective across R1 and S1 from 1.5B to 32B scales, indicating strong transferability of structural signals.
*   There is no systematic decline in reasoning ability; some backbones even show slight improvements, suggesting safety structure and task solving need not be sacrificed for each other.

## Highlights & Insights
*   The most valuable insight is that "safety failure comes from the reasoning structure." This is deeper than simply stating models need more safety data and explains why explicit intention analysis prompts are insufficient.
*   The AltTrain design is lightweight and practical for engineering. It transforms a complex RL alignment problem into SFT with a fixed reasoning format, making it low-cost and easy to reproduce.
*   Retaining "problem understanding" is a subtle but effective design. The authors avoid crudely inserting safety judgments and instead modify the existing LRM structure, reducing distributional conflict.
*   The result that over-refusal decreases with data expansion is important, suggesting that the primary issue with small-scale structural training is a lack of coverage rather than an inherent tendency toward over-conservatism.

## Limitations & Future Work
*   This paper only discusses text-based LRMs; whether reasoning structures can transfer to multimodal reasoning models remains an open question.
*   It is unclear how models using continuous space CoT, implicit reasoning, or those that do not expose reasoning tokens would adopt AltTrain.
*   AltTrain relies on the construction and labeling of harmful/benign samples; sample distribution affects the balance between over-refusal and missed refusals.
*   While evaluation covers standard red-teaming and multi-turn attacks, risks in real deployment such as strategic induction, long-context memory, and tool calls require further verification.
*   Sensitive content must be filtered before the data is released, and subsequent reproduction experiments may be affected by data access policies.

## Related Work & Insights
*   **vs SafeChain**: SafeChain uses filtered R1 trajectories for training but retains the original problem-solving structure, failing to reach the root cause. AltTrain directly modifies the structure.
*   **vs Intention Analysis**: IA uses prompts to have the model analyze intent first, but this does not necessarily change internal reasoning inertia. AltTrain solidifies intent analysis into the trajectory structure via SFT.
*   **vs DirectRefusal / STAR-1**: These methods reduce risk but tend to increase over-refusal or harm reasoning. AltTrain achieves a better balance through conditional reasoning.
*   **Insight**: Alignment for reasoning models may need to shift further from "what answers to reward" to "what reasoning processes to train."

## Rating
*   **Novelty**: ⭐⭐⭐⭐⭐ Explains LRM safety failures from a reasoning structure perspective with clear experimental support.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers multiple backbones, scales, tasks, and ablations; could be extended to more realistic deployment agent scenarios.
*   **Writing Quality**: ⭐⭐⭐⭐☆ The method is concise and tables are persuasive; the main table is large and requires patient comparison.
*   **Value**: ⭐⭐⭐⭐⭐ Provides direct inspiration for LRM safety alignment, low-cost post-training, and reasoning workflow design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)
- [\[ACL 2026\] AutoRAN: Automated Hijacking of Safety Reasoning in Large Reasoning Models](autoran_automated_hijacking_of_safety_reasoning_in_large_reasoning_models.md)
- [\[ACL 2026\] When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)
- [\[ACL 2026\] How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study](how_should_we_enhance_the_safety_of_large_reasoning_models_an_empirical_study.md)
- [\[ACL 2026\] SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging](safemerge_preserving_safety_alignment_in_fine-tuned_large_language_models_via_se.md)

</div>

<!-- RELATED:END -->
