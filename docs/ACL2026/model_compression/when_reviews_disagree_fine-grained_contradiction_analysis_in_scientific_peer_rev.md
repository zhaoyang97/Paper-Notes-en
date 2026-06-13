---
title: >-
  [Paper Note] When Reviews Disagree: Fine-Grained Contradiction Analysis in Scientific Peer Reviews
description: >-
  [ACL2026][Model Compression][Peer Review] This paper advances reviewer disagreement analysis from sentence-pair binary classification to evidence extraction and intensity scoring on full reviews…
tags:
  - "ACL2026"
  - "Model Compression"
  - "Peer Review"
  - "Contradiction Detection"
  - "Intensity Scoring"
  - "Multi-Agent Deliberation"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: f84d905b67195d33
---

# When Reviews Disagree: Fine-Grained Contradiction Analysis in Scientific Peer Reviews

**Conference**: ACL2026  
**arXiv**: [2605.10171](https://arxiv.org/abs/2605.10171)  
**Code**: https://github.com/sandeep82945/Contradiction-Intensity.git  
**Area**: model_compression  
**Keywords**: Peer Review, Contradiction Detection, Intensity Scoring, Multi-Agent Deliberation, Knowledge Distillation

## TL;DR
This paper advances reviewer disagreement analysis from sentence-pair binary classification to evidence extraction and intensity scoring on full reviews, utilizing the IMPACT multi-agent teacher to distill a TIDE small model capable of deployment with a single forward pass.

## Background & Motivation
**Background**: Disagreements in scientific peer reviews are the most time-consuming parts for Area Chairs (ACs) and editors when making decisions. Existing computational methods mostly frame reviewer disagreement as natural language inference (NLI) or binary contradiction detection, such as determining contradiction/non-contradiction between two sentences.

**Limitations of Prior Work**: Reviewer contradictions are not always explicit sentence-level conflicts. Two reviewers may give differing judgments on aspects like novelty, soundness, clarity, or meaningful comparisons, and these judgments are often scattered across multiple paragraphs of a full review. Binary sentence-pair models lose review-level discourse and cannot inform ACs whether a conflict is mild, moderate, or severe.

**Key Challenge**: Review assistance systems must be granular enough to provide evidence, aspects, and intensity, yet efficient enough to avoid expensive multi-agent deliberation for every call. A clear trade-off exists between high-quality reasoning and low-latency deployment.

**Goal**: The paper proposes a new fine-grained task: given two full peer reviews, output contradiction evidence pairs, their corresponding evaluation dimensions, intensity levels, and explanations. It constructs the RevCI expert-annotated dataset, designs the high-quality multi-agent framework IMPACT, and distills it into a more affordable TIDE small model.

**Key Insight**: Instead of starting from "whether sentences contradict," the authors start from the actual workflow of an AC: first finding potentially conflicting evidence by aspect, then tasking multiple agents with independent judgment and debate on intensity, and finally outputting via an adjudicator. This aligns the model output with the "evidence + severity + reasoning" actually required by editors.

**Core Idea**: Utilize a task-customized multi-agent deliberation framework to generate high-quality, interpretable contradiction intensity judgments, then use teacher-student distillation to let small models learn this evidence-grounded intensity reasoning, achieving a balance between quality and deployment cost.

## Method

### Overall Architecture
The paper first constructs the RevCI dataset. Based on the ASAP-Review source used in ContraSciView, it covers 8,582 paper reviews from ICLR 2017-2020 and NeurIPS 2016-2019. Reviews for the same paper are paired, resulting in approximately 28K pairs; as explicit contradictions are rare, GPT-4o mini is used for screening before expert re-annotation. RevCI contains 800 review pairs, with 352 containing at least one contradiction and 448 serving as negative examples.

Mechanistically, there are two layers. The first is IMPACT, an inference-time multi-agent framework. It takes two full reviews, extracts candidate evidence by aspect, and two intensity agents score and explain them independently. If they disagree, a Disagreement Orchestrator organizes a structured discussion, an Adjudication Agent rules based on the discussion trajectory, and finally, a Contradiction Validity Gate filters invalid contradictions.

The second layer is TIDE. Given IMPACT's high latency, the authors use IMPACT-P to generate synthetic contradiction annotations on ~2,000 additional ICLR 2021-2023 review pairs, mapping full review pairs to structured outputs. Meta-Llama-3-8B-Instruct is then fine-tuned via LoRA. TIDE only requires a single forward pass during testing to output evidence, intensity, and explanations.

### Key Designs
1. **Aspect-Conditioned Evidence Agent (ACEA)**:

    - **Function**: Identifies potential contradiction evidence pairs with high recall on full reviews and organizes them by evaluation dimensions.
    - **Mechanism**: Given a set of aspects (e.g., Motivation, Clarity, Soundness, Substance, Originality, Meaningful Comparison), ACEA extracts candidate span pairs from both reviews for each aspect. Formally, this can be viewed as $\mathcal{E}_{a_m}^{(i,j)}=f_{ACEA}(r_i,r_j,a_m)$, aggregating candidate pairs into an aspect-specific evidence pool.
    - **Design Motivation**: Without aspects, models easily miss implicit or scattered conflicts in long reviews; however, broad extraction leads to false positives. Prompting the model to specifically "find novelty conflicts" or "find clarity conflicts" improves recall and constrains intensity scoring within a clearer semantic frame.

2. **Deliberative Intensity Agents + Disagreement Orchestrator**:

    - **Function**: Enables two intensity judgment agents to independently score and explain the same evidence pair, initiating structured deliberation if they disagree.
    - **Mechanism**: Each DIA predicts $\alpha\in\{0,1,2,3\}$ and an explanation, where 0 represents an invalid contradiction and 1-3 indicate increasing intensity. Consensus is accepted immediately; if they disagree, the DO requires them to maintain their original scores while supplementing evidence, clarifying rubrics, or responding to the other's reasoning to avoid lazy convergence.
    - **Design Motivation**: Standard multi-agent debates often lead to conformity or unprincipled consensus. "Score-locking" prevents agents from simply changing votes, forcing them to surface the evidence behind their judgments for the adjudicator, which is more suitable for intensity judgment than simply "discussing until consensus."

3. **Teacher-Student Distillation from IMPACT to TIDE**:

    - **Function**: Compresses high-latency multi-agent deliberation into a single-model, single-forward deployment form.
    - **Mechanism**: IMPACT acts as the teacher, generating structured instances $c_j=(e_j,\alpha_j^*,\rho_j)$ for additional review pairs, including evidence pairs, adjudicated intensity, and explanations. TIDE learns $p_\theta(\{c_j\}|r_i,r_j)$ via SFT, updating only LoRA adapter parameters.
    - **Design Motivation**: AC tools must be scalable. IMPACT is suitable for high-value reviews or offline labeling, while daily batch pre-screening is better served by TIDE. This design converts "slow but accurate" multi-agent reasoning into a "fast and sufficient" small-model capability.

### Loss & Training
IMPACT does not train models but fixes temperature to 0 during inference, disabling nucleus and top-k sampling, and uses a fixed seed for reproducibility; duplicate contradictions are removed via a ROUGE-L threshold of 0.9. TIDE uses Meta-Llama-3-8B-Instruct with LoRA injected into attention and FFN projection layers, trained for 5 epochs using AdamW, a learning rate of $5\times10^{-5}$, a cosine schedule, a warmup ratio of 0.03, updating only LoRA adapters while the base model remains frozen.

## Key Experimental Results

### Main Results
Evaluation metrics include review-pair level FNR/FPR, and Cohen's $\kappa$, Spearman $\rho$, and Kendall $\tau$ on matched evidence pairs. Lower FNR/FPR and higher intensity consistency are preferred. Evidence matching uses ROUGE-L and Hungarian matching to avoid issues where simple counts cannot handle variable-length evidence sets.

| Category / Method | FNR ↓ | FPR ↓ | $\kappa$ ↑ | $\rho$ ↑ | $\tau$ ↑ | Note |
|--------|------|------|------|------|------|------|
| GPT-5.2 CoT | 0.2935 | 0.3012 | 0.2612 | 0.3679 | 0.3043 | Strong single-model baseline, but limited consistency |
| CourtEval | 0.2520 | 0.2590 | 0.2860 | 0.4100 | 0.3490 | Strongest general multi-agent baseline |
| IMPACT-OA | 0.2390 | 0.2287 | 0.3270 | 0.4783 | 0.4421 | Open-source version, already exceeding CourtEval |
| IMPACT-P | 0.1901 | 0.1613 | 0.3862 | 0.6193 | 0.5826 | Best performance, showing task-customized deliberation is effective |
| TIDE | 0.3771 | 0.3048 | 0.2202 | 0.3793 | 0.3549 | Single forward pass, high efficiency, consistency exceeds some LLMs |

### Ablation Study
The authors conducted ablations on both IMPACT and TIDE to verify the roles of aspect conditioning, intensity examples, intensity scoring, validity gate, multi-agent discussion, fine-tuning, and intensity reasoning supervision.

| Configuration | Key Metrics | Note |
|------|---------|------|
| w/o ACEA / w/o Deliberation | FNR 0.2969, FPR 0.3661 | Basic settings result in many missed detections and false positives |
| ACEA only | FNR 0.1092, FPR 0.5120 | Aspect conditioning significantly reduces misses but introduces more candidate false positives |
| IS + IEx | FNR 0.3293, FPR 0.3346, $\rho$ 0.5134 | Intensity examples help the model understand the 1-3 point rubric |
| ACEA + IEx + IS + CVG | FNR 0.1953, FPR 0.2614 | Validity gate suppresses false positives brought by ACEA |
| Full IMPACT | FNR 0.1901, FPR 0.1613, $\rho$ 0.6193 | DO, DIA, and Adjudicator significantly reduce FPR and improve consistency |
| TIDE full | FNR 0.3771, FPR 0.3048, $\rho$ 0.3793 | Joint training of fine-tuning + intensity scoring + reasoning achieves best results |

### Key Findings
- Compared to the strongest general multi-agent baseline CourtEval, IMPACT-P reduces average detection error by 31.2% and improves average consistency by 52.0%; IMPACT-OA also improves these by 8.5% and 19.4% respectively, indicating gains are not solely from stronger closed-source models.
- The number of discussion rounds is not "the more the better." The composite score improves significantly from 1 round (0.3608) to 3 rounds (0.4068) and 4 rounds, but gains almost saturate after 5 rounds, with a slight drop at 6 rounds, making $D=4$ a reasonable operating point.
- While TIDE does not outperform IMPACT overall, it successfully compresses evidence-grounded intensity reasoning into an 8B model with a single forward pass, suitable for large-scale pre-screening or low-cost editorial assistance.

## Highlights & Insights
- The task definition closely aligns with real-world AC workflows. It provides evidence, aspects, intensity, and explanations, allowing humans to quickly judge which disagreements warrant further discussion.
- The "score-locking" multi-agent deliberation design is ingenious. It prevents agents from yielding for the sake of consensus and shifts the goal to "exposing reasons for disagreement," which is highly transferable to other evaluation-style tasks.
- TIDE represents a natural model compression path: using a high-quality, multi-step, interpretable teacher to produce training signals and distilling that ability into a smaller model. This paradigm can be transferred to peer review quality checks, rebuttal handling, and factual conflict detection in long documents.

## Limitations & Future Work
- RevCI consists of only 800 review pairs. While the cost of expert annotation is understandably high, the data scale still limits generalization. Subtler contradictions might be underestimated due to LLM pre-screening.
- The experiments focus on Computer Science reviews (ICLR/NeurIPS) and six high-frequency aspects. Review styles, evaluation dimensions, and conflict expressions may differ across disciplines; cross-domain validation is still needed.
- IMPACT can be updated with new aspects via ACEA prompt changes, but TIDE requires retraining to adapt to new aspects. Future work could consider aspect-description conditional training for more open evaluation dimensions.

## Related Work & Insights
- **vs ContraSciView**: ContraSciView models reviewer disagreement as binary contradiction detection on isolated sentence pairs; this work handles full reviews, outputting evidence sets and intensity levels, better suiting the actual decision-making needs of ACs.
- **vs General NLI Models**: NLI models excel at standard premise-hypothesis judgments, but contradictions in peer reviews often involve hedging, technical assumptions, and differences in evaluation scales; IMPACT handles this pragmatic information better through aspect conditioning and full-context reasoning.
- **vs General Multi-agent Evaluation Frameworks**: Self-Refine, Debate, ChatEval, and CourtEval use relatively general discussion/adjudication workflows; IMPACT's advantage lies in specialized designs for reviewer contradiction tasks (ACEA, score-locking, CVG, and intensity adjudication), meaning improvements stem from task structure rather than simple agent counts.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The task definition and score-locking deliberation are highly innovative; the TIDE distillation path is natural yet practical.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Main results, IMPACT/TIDE ablations, discussion rounds, and human error analyses are comprehensive, though dataset scale and domain coverage are limited.
- Writing Quality: ⭐⭐⭐⭐☆ Method modules are clear, metric definitions are meticulous, and while some tables are dense, they support the conclusions.
- Value: ⭐⭐⭐⭐☆ Highly valuable for review assistance and long-document contradiction detection; provides a paradigm for compressing multi-agent teachers into small models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](a_layer-wise_analysis_of_supervised_fine-tuning.md)
- [\[CVPR 2026\] DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry Estimation](../../CVPR2026/model_compression/dage_dual-stream_architecture_for_efficient_and_fine-grained_geometry_estimation.md)
- [\[ICML 2026\] FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models](../../ICML2026/model_compression/frism_fine-grained_reasoning_injection_via_subspace-level_model_merging_for_visi.md)
- [\[ICLR 2026\] Paper Copilot: Tracking the Evolution of Peer Review in AI Conferences](../../ICLR2026/model_compression/paper_copilot_tracking_the_evolution_of_peer_review_in_ai_conferences.md)
- [\[ACL 2026\] Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis](analytical_ffn-to-moe_restructuring_via_activation_pattern_analysis.md)

</div>

<!-- RELATED:END -->
