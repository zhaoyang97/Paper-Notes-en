---
title: >-
  [Paper Note] CyberJurors: A Multi-Agent Simulation Task for E-Commerce Disputes Verdict
description: >-
  [ICML 2026][Multimodal VLM][E-commerce Dispute Verdict] The authors formalize the real-world crowdsourced jury mechanism of e-commerce platforms into the E-commerce Dispute Verdicts (EDV) task. They construct VerdictBenc…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "E-commerce Dispute Verdict"
  - "Crowdsourced Jury"
  - "Multi-agent Simulation"
  - "IV-CoT"
  - "Precedent Constraints"
date: 2026-05-08
content_hash: b69351550ee77481
---

# CyberJurors: A Multi-Agent Simulation Task for E-Commerce Disputes Verdict

**Conference**: ICML 2026  
**arXiv**: [2605.28369](https://arxiv.org/abs/2605.28369)  
**Code**: https://github.com/YanhuiS/CyberJurors  
**Area**: Multi-Agent / Legal & E-commerce AI / Multi-modal Reasoning  
**Keywords**: E-commerce Dispute Verdict, Crowdsourced Jury, Multi-agent Simulation, IV-CoT, Precedent Constraints  

## TL;DR
The authors formalize the real-world crowdsourced jury mechanism of e-commerce platforms into the E-commerce Dispute Verdicts (EDV) task. They construct VerdictBench, the first multi-modal benchmark containing ground-truth votes from 17 jurors (6,000 cases, text/image/video/multi-turn), and propose CyberJurors. This system utilizes a four-stage Individual Verdict Chain-of-Thought (IV-CoT) for fine-grained evidence localization by single jurors and a Jury Consensus Verdict (JCV) that introduces historical precedents via Stare Decisis for collective consensus. On VerdictBench, the accuracy improves by +9.48%, +9.38%, and +6.19% compared to the strongest LLMs, MLLMs, and courtroom simulators, respectively.

## Background & Motivation
**Background**: To efficiently process massive transaction disputes, e-commerce platforms have introduced "crowdsourced jury" mechanisms where buyers and sellers submit multi-round multi-modal evidence (chat history, images, videos), and 17 volunteer jurors decide the outcome. The scalability of this mechanism is bottlenecked by the days required to recruit 17 jurors. Recently, multi-agent systems (e.g., ChatEval, AgentCourt) have demonstrated potential in legal judgment tasks.

**Limitations of Prior Work**: Directly migrating multi-agent courtroom simulations from the legal domain to e-commerce dispute verdicts is unfeasible for two reasons:

**Key Challenge**: (1) Evidence in e-commerce disputes is **redundant, multi-round, and cross-modal** (alternating between questioning, rebuttal, and clarification); key clues are often buried in vast amounts of evidence. Existing methods are limited to pure text reasoning, and even with MLLMs, inputs are passive and one-time, failing to capture fine-grained visual clues from redundant evidence (e.g., a flashing 2% battery indicator in a video). This leads to a counter-intuitive phenomenon: MLLMs actually perform worse than text-only LLMs on EDV. (2) Unlike formal courts that rely on rigid legal statutes, e-commerce verdicts rely on **flexible, platform-specific trading conventions** lacking explicit guidance, causing existing models to exhibit inherent generation biases that undermine fairness and interpretability.

**Goal**: (a) Propose the EDV task as a rigorously evaluated task and provide a multi-modal benchmark; (b) Design a multi-agent system that simultaneously addresses "fine-grained evidence localization" and "collective consensus + fair verdict."

**Key Insight**: The authors draw inspiration from the "Stare Decisis" principle in common law—historical precedents can provide normative references for current verdicts. They also transform traditional one-time MLLM reasoning into an iterative, active evidence sampling process of "select-and-perceive."

**Core Idea**: Use IV-CoT to split individual juror reasoning into four stages: "focus extraction → active evidence selection/perception → adversarial analysis → final verdict." Use JCV to inject precedent constraints during multi-round voting, ensuring the 17-juror simulation is both accurate and aligned with the real voting distribution.

## Method

### Overall Architecture
**Dataset VerdictBench**: 6,000 cases across five categories (Appliances, Clothing, Food, Digital, Others), preserving transaction metadata, multi-round multi-modal evidence, and ground-truth votes from 17 jurors. Each case averages 14 images and 0.9 videos; the seller win rate is 62.6% (due to familiarity with fulfillment rules). The dataset is stratified by category × difficulty (margin of 17 votes) into a 3:1:2 split for train/val/test.

**Model CyberJurors**: Modeled as a directed social network $\boldsymbol{G}=\langle\boldsymbol{A},\boldsymbol{E}\rangle$, where $\boldsymbol{A}=\{a_1,...,a_N\}$ represents $N$ heterogeneous jurors, and $e_{k,j}\in\boldsymbol{E}$ denotes $a_k$ following $a_j$. Given a case $\boldsymbol{D}=\{d,\bm{e}_1^b,\bm{e}_1^s,...\}$ ($d$ is metadata, $\bm{e}_i^b=\{\bm{T}_i^b,\bm{I}_i^b,\bm{V}_i^b\}$ is the buyer's $i$-th round of multi-modal evidence, symmetric for the seller), JCV simulates $T$ rounds of discussion: in each round, jurors first receive the Previous Collective Verdict Summary and Verdict Precedent Base, then use IV-CoT to generate an individual verdict $\hat y_{k,t}$ and justification $J_{k,t}$. Finally, a majority vote yields the ultimate verdict, with the summary serving as an interpretable justification.

### Key Designs

1.  **Individual Verdict Chain-of-Thought (IV-CoT)**—Four-stage reasoning + "Select-Perceive" iteration:
    *   **Function**: Enables a single juror to actively locate key clues from redundant multi-modal evidence and explicitizes the causal logic chain.
    *   **Mechanism**: Stage I (Focus Extraction) $\boldsymbol{O}_{\text{I}}:\{F,F^b,F^s\}=\mathcal{F}_{\text{extract}}(d,\bm{T}^b,\bm{T}^s)$ extracts the dispute focus and core claims. Stage II (Clue Localization) is the core innovation, replacing traditional one-time multi-modal understanding with a "Select-Perceive" iteration—in each round, $\bm{e}^b_{*,t}=\mathcal{F}_{\text{select}}(\boldsymbol{O}_{\text{I}},\bm{T}^b-\bm{T}^b_{select})$ selects the most likely piece of evidence containing key clues, followed by fine-grained perception $\{K_t^b,A_t^b\}=\mathcal{F}_{\text{perceive}}(\boldsymbol{O}_{\text{I}},\bm{e}^b_{*,t},\boldsymbol{O}_{t-1}^b)$ on that piece, looping up to $T_{max}$ rounds. Stage III (Adversarial Analysis) $\{\Delta,\Delta^b,\Delta^s\}=\mathcal{F}_{\text{analyze}}(\boldsymbol{O}_{\text{I}},\boldsymbol{O}_{\text{II}})$ identifies logical conflicts. Stage IV (Final Verdict) $\{\hat y_k,J_k\}=\mathcal{F}_{\text{judge}}(\boldsymbol{O}_{\text{I}},\boldsymbol{O}_{\text{II}},\boldsymbol{O}_{\text{III}})$ produces the judgment and traceable reasons.
    *   **Design Motivation**: Passive one-time perception in MLLMs survives poorly in ultra-long contexts where key visual clues are submerged. "Select-Perceive" iteration compresses context into one evidence segment per round, bypassing context window bottlenecks and explicitly recording the "focus → evidence" causal chain in $\boldsymbol{O}_{\text{II}}$ for subsequent analysis. Buyer and seller iterations are performed independently to avoid cross-interference.

2.  **Jury Consensus Verdict (JCV)**—Heterogeneous jurors + collective summary $T$-round social simulation:
    *   **Function**: Mitigates inherent biases of single models through multi-round multi-juror discussions and uses social network structure as an information flow constraint.
    *   **Mechanism**: Each juror $a_k=\{\boldsymbol{P}_k,\boldsymbol{M}_k\}$ consists of a persona $\boldsymbol{P}_k$ and memory $\boldsymbol{M}_k$. Decisions in round $t$ depend on the case, persona, memory, neighbor set $\boldsymbol{R}_{k,t}=\{J_{j,t-1}\mid e_{k,j}\in\boldsymbol{E}\}$, and global collective summary $\boldsymbol{S}_t=\mathcal{F}_{\text{sum}}(d,\bigcup_j J_{j,t-1})$, i.e., $\hat y_{k,t},J_{k,t}=\mathcal{F}_{\text{judge}}(\boldsymbol{D},\boldsymbol{P}_k,\boldsymbol{M}_k,\boldsymbol{R}_{k,t},\boldsymbol{S}_t)$. Majority voting $\hat y=\mathbb{I}(\hat y^s>\hat y^b)$ determines the final result.
    *   **Design Motivation**: Single LLMs repeat training data biases (e.g., systematic bias towards sellers) when explicit regulations are absent. Allowing $N$ heterogeneous jurors to exchange opinions locally via $\boldsymbol{G}$ with macro-guidance from global summaries retains independent reasoning while driving outlier jurors to reconsider extreme positions, enhancing decision stability.

3.  **Verdict Precedent**—Injecting Stare Decisis into agent memory:
    *   **Function**: Provides jurors with normative and traceable verdict grounds using historical precedents, filling the gap of "non-codified laws."
    *   **Mechanism**: Construct a Precedent Base $\boldsymbol{B}=\langle\boldsymbol{H},\boldsymbol{N}\rangle$, where $\boldsymbol{H}$ are historical cases and $\boldsymbol{N}$ are explicit verdict guidelines distilled from them. For each new case, semantic retrieval $\boldsymbol{H}_{\text{guide}},\boldsymbol{N}_{\text{guide}}=\mathcal{F}_{\text{retrieve}}(\boldsymbol{D},\boldsymbol{B})$ finds the most relevant precedents; then, the top-$K$ guidelines per juror are selected based on persona matching $\boldsymbol{M}_k=\{\text{Rank}(\Phi(\boldsymbol{P}_k,\boldsymbol{N}_j))\le K\mid \boldsymbol{N}_j\in\boldsymbol{N}_{\text{guide}}\}$ and stored in memory.
    *   **Design Motivation**: Precedents replace "statutes" and are **personalized**—different jurors receive different subsets of norms, simulating natural variations in member focus within a real jury while ensuring the group centers on the same precedent for fairness and interpretability.

### Loss & Training
The framework is inference-only, using Gemini-2.5-Flash-Lite-Nothinking as the backbone. Parameters: $T_{max}=3$ (IV-CoT iteration rounds), $T=3$ (JCV discussion rounds), $K=3$ (top-3 precedents per juror), early-stop threshold $\delta=0.8$, and video sampling at 30 frames. $\boldsymbol{G}$ initialization follows existing social simulation work.

## Key Experimental Results

### Main Results
Comparing five baseline categories on the VerdictBench test set (Closed/Open LLMs, Closed/Open MLLMs, Court simulators):

| Category | Method | Acc ↑ | Weig. F1 ↑ | Macro F1 ↑ | MAE ↓ | RMSE ↓ | Token ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Closed LLM | GPT-5.2-Chat | 0.6344 | 0.6309 | 0.6340 | - | - | 158k |
| Closed LLM | DeepSeek-V3 | 0.6080 | 0.6042 | 0.6075 | - | - | 114k |
| Open LLM | Dolphin3.0-R1-24B | 0.4929 | 0.4568 | 0.4790 | - | - | 145k |
| Closed MLLM | Gemini-3-Pro | 0.6354 | 0.6378 | 0.6351 | - | - | 4.37M |
| Closed MLLM | Claude-Opus-4.5 | 0.5910 | 0.5901 | 0.5910 | - | - | 2.70M |
| Closed MLLM | GPT-5.2 | 0.4833 | 0.4907 | 0.4798 | - | - | 1.63M |
| Open MLLM | Qwen3-VL-235B | 0.4843 | 0.4748 | 0.4825 | - | - | 2.99M |
| Court Sim | ChatEval | 0.6589 | 0.6645 | 0.6525 | - | - | 0.92M |
| Court Sim | AgentCourt | 0.6673 | 0.6644 | 0.6383 | - | - | 75.28M |
| **Ours** | **CyberJurors** | **0.7292** | **0.7258** | **0.7037** | **4.7312** | **6.3724** | 62.33M |

CyberJurors shows a 6.19% Acc gain and 0.0613 Weig. F1 gain over the runner-up (AgentCourt). It is the only model providing MAE/RMSE aligned with the 17-vote distribution. A counter-intuitive finding: average MLLM Acc (52.98%) < average text LLM Acc (55.78%), confirming that "passive one-time perception" in MLLMs is insufficient for locating visual clues in long contexts.

### Ablation Study
Gradual activation of CyberJurors modules on the validation set:

| Configuration | Acc ↑ | Weig. F1 ↑ | Macro F1 ↑ | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | 0.5416 | 0.5433 | 0.5385 | Direct Gemini-2.5-Flash-Lite verdict |
| + Rules | 0.5876 | 0.5887 | 0.5875 | Adding verdict rule prompt: +4.60% |
| + SR-CoT | 0.6406 | 0.6416 | 0.6810 | Four-stage with single-step selection: +5.30% |
| + IV-CoT | 0.6734 | 0.6788 | 0.6757 | Adding Select-Perceive iteration: +3.28% |
| + Jury | 0.7018 | 0.7043 | 0.6868 | Adding multi-juror simulation: +2.84% |
| + Precedent | **0.7252** | 0.7196 | 0.6980 | Adding precedent constraints: +2.34% |

### Key Findings
- The "Select-Perceive" iteration provides a 3.28% Acc gain over single-step selection, marking it as the most significant design within IV-CoT. This proves that **active multi-round sampling** outperforms one-time injection for redundant multi-modal evidence.
- Adding Jury simulation (+2.84%) and Precedents (+2.34%) yields comparable gains, proving both "collective consensus" and injecting normative constraints into individual memory are effective.
- Token consumption (62.33M) is significantly higher than single LLMs but lower than AgentCourt (75M). CyberJurors exchanges fewer tokens for higher accuracy, serving as a "pruned" redesign of courtroom simulation.

## Highlights & Insights
- **Real-world Value of Task Definition**: EDV directly addresses operational scenarios on e-commerce platforms. The 17-vote ground truth provides both labels and difficulty (vote margin), allowing benchmarks to evaluate accuracy and alignment with social consensus distributions.
- **Evidence of "Passive MLLM < Active LLM"**: In settings with long context and redundant multi-modal evidence, simple MLLM stacking is detrimental. This serves as a warning for "multi-round multi-modal agent" tasks: visual encoding must be coupled with active retrieval or iterative perception.
- **Stare Decisis as a Normative Source for Multi-Agents**: Injecting legal principles as memory rather than hard constraints makes "precedent → personalized guidance" a novel alignment mechanism. This is naturally transferable to content moderation or compliance auditing tasks where codified laws are absent.

## Limitations & Future Work
- Single backbone (Gemini-2.5-Flash-Lite): JCV heterogeneity effectiveness across different models remains unverified.
- Imbalance in VerdictBench (seller 62.6% vs. buyer 37.4%) might cause CyberJurors to inherit "pro-seller" bias—fairness metrics split by direction were not provided.
- Retrieval quality of the precedent base directly affects $\boldsymbol{M}_k$; fallback mechanisms for rare categories or cold starts are not discussed.
- Video sampling at 30 frames may miss critical frames for event-based evidence; event-aware sampling is a future direction.

## Related Work & Insights
- **vs ChatEval / AgentCourt**: These focus on "general debate / legal courtroom simulation" with text-based inputs. This work shifts to e-commerce multi-modal multi-round scenarios, introducing active evidence sampling and personalized precedents to make the simulator paradigm work for "long redundant multi-modal evidence."
- **vs Pure MLLM End-to-End Reasoning**: Feeding 14 images + 0.9 videos directly to MLLMs performs worse than LLMs. CyberJurors' Select-Perceive provides an engineering template for decomposing "long input" into "goal-driven local perception."
- **vs Chain-of-Thought**: While standard CoT is a single line of reasoning, IV-CoT decouples "evidence selection-perception" into Stage II, creating a dual-layer chain of evidence and logic, which is more effective when "evidence selection" is the primary challenge.

## Rating
- Novelty: ⭐⭐⭐⭐ Formalizing e-commerce crowdsourced juries as EDV, the first 17-vote multi-modal benchmark, and using precedents as agent memory are all novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 14 baselines across 5 categories, 5-step ablation, token comparison. Lacks multi-backbone and fairness breakdown.
- Writing Quality: ⭐⭐⭐⭐ Clear task motivation; IV-CoT stages and JCV are formalized with formulas and symbol definitions.
- Value: ⭐⭐⭐⭐⭐ Provides both a high-quality benchmark and a method deployable as a commercial auxiliary system; highly practical for academia and industry.

## Rating
- Novelty: To be evaluated
- Experimental Thoroughness: To be evaluated
- Writing Quality: To be evaluated
- Value: To be evaluated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning in E-commerce](../../ACL2026/multimodal_vlm/afmrl_attribute-enhanced_fine-grained_multi-modal_representation_learning_in_e-c.md)
- [\[CVPR 2026\] VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments](../../CVPR2026/multimodal_vlm/vs_bench_evaluating_vlms_for_strategic_abilities_in_multi_agent_environments.md)
- [\[ACL 2026\] From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision-Language Models](../../ACL2026/multimodal_vlm/from_heads_to_neurons_causal_attribution_and_steering_in_multi-task_vision-langu.md)
- [\[ACL 2026\] MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems](../../ACL2026/multimodal_vlm/moneta_multimodal_industry_classification_through_geographic_information_with_mu.md)
- [\[AAAI 2026\] Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models](../../AAAI2026/multimodal_vlm/concept-rulenet_grounded_multi-agent_neurosymbolic_reasoning.md)

</div>

<!-- RELATED:END -->
