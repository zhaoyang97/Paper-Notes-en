---
title: >-
  [Paper Note] Reasoning Structure Matters for Safety Alignment of Reasoning Models
description: >-
  [ACL 2026][LLM Safety][Large Reasoning Models] The paper identifies that the safety issues in large reasoning models (LRMs) stem from a reasoning structure of "understanding the problem first, then solving with full effort." It proposes AltTrain, which uses 1K SFT data samples to reshape the reasoning structure into "problem understanding → harmfulness assessment → conditional reasoning," significantly reducing harmful responses while largely preserving reasoning capabilities…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Large Reasoning Models"
  - "Safety Alignment"
  - "Reasoning Structure"
  - "AltTrain"
  - "SFT"
date: 2026-05-08
content_hash: 4919b7d371eff8e4
---

# Reasoning Structure Matters for Safety Alignment of Reasoning Models

**Conference**: ACL 2026  
**arXiv**: [2604.18946](https://arxiv.org/abs/2604.18946)  
**Code**: https://github.com/yeonjun-in/R1-Alt  
**Area**: LLM Reasoning / LLM Safety Alignment / Post-training  
**Keywords**: Large Reasoning Models, Safety Alignment, Reasoning Structure, AltTrain, SFT

## TL;DR
The paper identifies that the safety issues in large reasoning models (LRMs) stem from a reasoning structure of "understanding the problem first, then solving with full effort." It proposes AltTrain, which uses 1K SFT data samples to reshape the reasoning structure into "problem understanding → harmfulness assessment → conditional reasoning," significantly reducing harmful responses while largely preserving reasoning capabilities.

## Background & Motivation
**Background**: Large reasoning models such as R1 and the o1 series have achieved significant improvements in mathematics, code, and complex logical tasks through long-chain-of-thought (CoT). Their training trajectories typically encourage the model to understand the problem first, followed by multi-step solving, checking, and self-correction.

**Limitations of Prior Work**: This same reasoning capability becomes a risk when facing malicious requests. Existing research finds that models after reasoning post-training may be more prone to following user goals deeply than standard instruction models; even if they recognize a request is problematic, they may still proceed to complete the task.

**Key Challenge**: Safety models need to stop assisting with harmful requests while fully utilizing reasoning capabilities for benign ones. Simple refusal training damages capability, while simple prompting for intent judgment is insufficient to change the underlying reasoning inertia.

**Goal**: To explain why LRMs continue solving after identifying risks and to design a lightweight post-training method that explicitly alters the model's reasoning structure, rather than just adding refusal templates to the output layer.

**Key Insight**: The authors argue that the root cause is not that the model "does not know it is harmful," but that the trained reasoning structure excessively prioritizes task-solving. As long as the structure remains "problem understanding → solution reasoning," the model tends to treat any request as a problem to be solved.

**Core Idea**: The key to safety alignment is rewriting the reasoning process, allowing the model to insert a harmfulness assessment before formal solving and then choose to refuse or continue reasoning based on that assessment.

## Method
The contribution of AltTrain lies in its restraint: it does not design complex RL or train a reward model. Instead, it constructs 1K SFT samples with a fixed reasoning structure, teaching the LRM to first understand the problem, then judge the harmfulness, and finally perform conditional reasoning within its internal thought process. This structure retains the "problem understanding" familiar to the original LRM while adding a safety decision point, thereby minimizing distribution shift.

### Overall Architecture
The training data AltTrain-1K comes from the SafeChain dataset, containing approximately 900 harmful queries and 100 benign queries. Each response consists of a reasoning chain (carried in the model's original "think" template) and a final answer.

For each query, AltTrain sequentially collects three parts: first, Problem Understanding (PU), extracted from the first sentence of the R1 original reasoning trajectory to maintain the familiar opening structure; second, Harmfulness Assessment (HA), where an LLM like GPT-4o provides a one-sentence judgment on whether the request is harmful with a reason; and third, Conditional Reasoning (CR). If the request is harmful, further solving ends immediately with a refusal; if benign, it continues with the remainder of the R1 original reasoning chain. The assembled samples undergo lightweight SFT to obtain the aligned R1-Alt.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Query (SafeChain: ~900 Harmful + 100 Benign)"] --> B["Problem Understanding (PU)<br/>Retain R1 trajectory first sentence as anchor"]
    B --> C["Harmfulness Assessment (HA)<br/>GPT-4o one-sentence judgment"]
    subgraph CR["Conditional Reasoning (CR)"]
        direction TB
        D["Judged Harmful: Immediate termination + Safety refusal"]
        E["Judged Benign: Continue remaining R1 reasoning chain"]
    end
    C -->|Harmful| D
    C -->|Benign| E
    CR --> F["Assemble Reasoning Chain (think template) + Final Answer → AltTrain-1K"]
    F --> G["Lightweight SFT: 1K samples, no RL / reward → R1-Alt"]
```

### Key Designs
**1. Preserving the Original Structure in PU: Attaching safety judgment after the familiar opening rather than starting anew**

Forcing the model to perform harmfulness assessment immediately can push it away from the reasoning distribution learned during training, dragging down performance on normal tasks like math and code. AltTrain's approach is restrained: authors found that 985 out of 1,000 R1 trajectories in SafeChain naturally include problem understanding in the first paragraph. Thus, they retain the first sentence of the R1 original trajectory as a structural anchor. The reasoning still starts with "understanding the problem," and safety logic is grafted on rather than replacing it, keeping distribution shift to a minimum—removing PU dropped the reasoning score to 56.3 in ablations, proving this anchor protects capability.

**2. Query-level Harmfulness Assessment: Inserting an explicit intent check before solving**

The preliminary analysis reveals a counter-intuitive fact: LRMs can almost always identify harmful queries, but there is a disconnect between "identification" and "behavior"—they recognize the problem but follow the solving structure anyway. AltTrain forces a high-level harmfulness judgment ("Harmful/Benign + Reason") after problem understanding. This judgment focuses only on whether to assist, regardless of the specific attack type. Embedding this into the reasoning structure ensures the identification results actually "take over" the subsequent flow rather than being overridden by solving inertia.

**3. Conditional Reasoning and Lightweight SFT: Allowing assessment results to branch into two reasoning paths**

Once assessed, the result must be actionable: if harmful, the model terminates solving and provides a refusal; if benign, it completes the task using the original R1 reasoning chain. This step welds "assessment" to "behavior," mending the prior disconnect. This structure requires only 1K SFT samples; an 8B model can be trained on a single A6000 in about 60 minutes. Small data suffices because the model learns a stable decision structure rather than specific responses to categories of attacks, allowing it to generalize across model sizes and unseen scenarios.

### Loss & Training
AltTrain utilizes standard supervised fine-tuning (SFT) without RL, DPO, or a reward model. The objective is to maximize the likelihood of the structured response. The paper emphasizes data and token efficiency: R1-Alt training samples average 167 tokens, and reasoning averages 69 tokens per query, which is shorter than SafeChain and STAR-1. The authors believe efficiency stems from the structure itself rather than mere text shortening, as deleting any key step leads to significant degradation.

## Key Experimental Results

### Main Results
The main experiment evaluates harmful response rates, over-refusal rates, and reasoning capabilities across several R1/S1 backbones. Average metrics for representative models are selected below. Lower Harmfulness is better, lower Over-refusal is better, and higher Reasoning is better.

| Backbone | Method | Data Volume | Harmful Avg. ↓ | Over-refusal ↓ | Reasoning Avg. ↑ | Observation |
|----------|------|--------|----------------|----------------|------------------|------|
| R1-7B | No train | - | 82.2 | 0.0 | 72.6 | Original is strong but high-risk |
| R1-7B | R1-Alt | 1K | 14.3 | 31.6 | 69.5 | Risk dropped, over-refusal rose |
| R1-8B | No train | - | 83.5 | 0.4 | 58.1 | High original harmful response rate |
| R1-8B | R1-Alt | 1K | 4.8 | 14.0 | 59.8 | Good balance of safety and reasoning |
| R1-32B | No train | - | 82.5 | 0.0 | 76.0 | Large models share structure issues |
| R1-32B | R1-Alt | 1K | 3.7 | 11.2 | 78.0 | Safety improved, capability slightly rose |
| S1-14B | No train | - | 89.7 | 0.0 | 65.9 | S1 series also high-risk |
| S1-14B | S1-Alt | 1K | 6.7 | 14.0 | 64.8 | Method effective across backbones |

Authors also checked QA, multilingual, and summarization capabilities; R1-Alt largely retained the performance of original models.

| Method | NQ ↑ | CMMLU ↑ | CNN ROUGE ↑ | Note |
|------|------|---------|-------------|------|
| No train | 71.7% | 61.8% | 12.3 | Original LRM |
| SafeChain | 73.9% | 59.7% | 13.8 | General capability preserved after safety training |
| STAR-1 | 72.3% | 59.0% | 14.3 | Higher summarization metrics |
| R1-Alt | 72.0% | 60.5% | 13.6 | Overall close to original model |

### Ablation Study
Structure ablations show that all three steps are essential. Removing HA significantly worsens safety; removing PU or CR hurts reasoning or increases over-refusal.

| Variant | R1-8B Harmful Avg. ↓ | R1-8B Over-refusal ↓ | R1-8B Reasoning Avg. ↑ | Explanation |
|------|----------------------|----------------------|------------------------|------|
| w/o PU | 1.6 | 14.0 | 56.3 | Safe but reasoning drops; anchor is useful |
| w/o HA | 15.6 | 19.8 | 59.4 | Safety worsens without harmfulness judgment |
| w/o CR | 4.1 | 18.0 | 59.3 | Missing conditional branch affects calibration |
| CR Rephrase | 5.6 | 10.4 | 59.9 | Still effective; not template memorization |
| R1-Alt | 4.8 | 14.0 | 59.8 | Three-step structure achieves balance |

Data volume analysis shows that over-refusal in AltTrain can be mitigated by scaling data.

| Training Data | R1-8B Harmful Avg. ↓ | R1-8B Over-refusal ↓ | R1-8B Reasoning Avg. ↑ | Observation |
|----------|----------------------|----------------------|------------------------|------|
| AltTrain-0.5K | 4.6 | 22.0 | 60.4 | Higher over-refusal with less data |
| AltTrain-1K | 4.8 | 14.0 | 59.8 | Default setting |
| AltTrain-3K | 4.7 | 2.4 | 58.7 | Significant drop in over-refusal |

### Key Findings
- The problem with LRMs is not a failure to identify harmful intent, but continuing to solve along the reasoning structure after identification.
- Safety alignment requires changing the reasoning structure rather than just adding output-layer refusal rules.
- 1K small data samples are sufficient for the model to learn the structure; expanding to 3K significantly reduces over-refusal.
- AltTrain is effective across R1 and S1 from 1.5B to 32B scales, suggesting strong transferability of structural signals.
- Reasoning capability did not systematically decrease, and even improved in some backbones, indicating that safety structures and task solving do not need to be mutually sacrificed.

## Highlights & Insights
- The most valuable insight is that "safety failure comes from reasoning structure." This is deeper than saying models need more safety data and explains why explicit intent analysis prompts are insufficient.
- AltTrain's design is lightweight and fits engineering practices. It converts a complex RL alignment problem into a fixed-format SFT, which is low-cost and easy to reproduce.
- Retaining problem understanding is a nuanced step. Instead of crudely inserting safety judgments, the authors adapted the existing LRM structure to reduce distributional conflict.
- The result that over-refusal drops with data scaling is important, indicating that the main issue with small-data structural training is coverage rather than the structure itself being inherently over-conservative.

## Limitations & Future Work
- This paper only discusses text-based LRMs; whether reasoning structures can transfer to multimodal reasoning models remains an open question.
- It is unclear how AltTrain would be adopted for models using continuous space CoT, implicit reasoning, or those that do not expose reasoning tokens.
- AltTrain relies on the construction and labeling of harmful/benign samples; sample distribution affects the balance of over-refusal and missed detections.
- While evaluations covered standard red-teaming and multi-turn attacks, risks in real deployment—such as strategic induction, long-context memory, and tool-calling—require further verification.
- Sensitive content filtering is needed before data release; subsequent reproduction may be affected by data access policies.

## Related Work & Insights
- **vs SafeChain**: SafeChain uses filtered R1 trajectories for training but retains the original problem-solving structure, making it difficult to address the root cause. AltTrain changes the structure directly.
- **vs Intention Analysis**: IA uses prompts to have the model analyze intent, but does not necessarily change internal reasoning inertia. AltTrain solidifies intent analysis into the trajectory structure via SFT.
- **vs DirectRefusal / STAR-1**: These methods reduce risk but prone to increasing over-refusal or damaging reasoning. AltTrain is more balanced via conditional reasoning.
- **Insight**: For alignment of reasoning models, one might need to shift from "what answers to reward" to "what reasoning processes to train."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Explains LRM safety failure from reasoning structure; clear perspective with experimental support.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple backbones, scales, tasks, and ablations; could be extended to more realistic deployment agent scenarios.
- Writing Quality: ⭐⭐⭐⭐☆ Method is concise, and tables are persuasive; the main table is large and requires patient comparison.
- Value: ⭐⭐⭐⭐⭐ Provides direct inspiration for LRM safety alignment, low-cost post-training, and reasoning process design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)
- [\[ACL 2026\] AutoRAN: Automated Hijacking of Safety Reasoning in Large Reasoning Models](autoran_automated_hijacking_of_safety_reasoning_in_large_reasoning_models.md)
- [\[ICLR 2026\] AdvChain: Adversarial Chain-of-Thought Tuning for Robust Safety Alignment of Large Reasoning Models](../../ICLR2026/llm_safety/advchain_adversarial_chain-of-thought_tuning_for_robust_safety_alignment_of_larg.md)
- [\[ACL 2026\] When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)
- [\[ACL 2026\] How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study](how_should_we_enhance_the_safety_of_large_reasoning_models_an_empirical_study.md)

</div>

<!-- RELATED:END -->
