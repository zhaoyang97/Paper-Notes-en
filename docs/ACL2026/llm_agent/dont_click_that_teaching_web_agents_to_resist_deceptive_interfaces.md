---
title: >-
  [Paper Note] Don't Click That: Teaching Web Agents to Resist Deceptive Interfaces
description: >-
  [ACL 2026][LLM Agent][Deceptive UI Defense] The authors formalize "defending against deceptive UIs" as an independent defense problem for web agents for the first time. They propose a two-stage framework **DUDE** (Stage-…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Deceptive UI Defense"
  - "Hybrid-Reward RL"
  - "Experience Summarization"
  - "Dark Patterns"
  - "VLM Agent"
date: 2026-05-08
content_hash: 0f1d2a673b28c759
---

# Don't Click That: Teaching Web Agents to Resist Deceptive Interfaces

**Conference**: ACL 2026  
**arXiv**: [2605.09497](https://arxiv.org/abs/2605.09497)  
**Code**: https://github.com/(DUDE project link in paper)  
**Area**: LLM Agent / Security / Web Agent / GUI Robustness  
**Keywords**: Deceptive UI Defense, Hybrid-Reward RL, Experience Summarization, Dark Patterns, VLM Agent

## TL;DR
The authors formalize "defending against deceptive UIs" as an independent defense problem for web agents for the first time. They propose a two-stage framework **DUDE** (Stage-1: learning an evaluator via hybrid-reward RL with asymmetric penalties; Stage-2: distilling failure modes into transferable context via iterative experience summarization). They release the **RUC** benchmark containing 1407 real/synthetic scenarios. Across three VLM agent bases, DUDE reduced deception-induced failure rates from 23.5% to 1.5% and increased task success rates from 9.5% to 60.5%, with Stage-2 optimized prompts demonstrating zero-shot transferability to closed-source models.

## Background & Motivation

**Background**: VLM-based web agents (e.g., Qwen-VL, UI-TARS, Holo, Agent Q) have demonstrated autonomous GUI operation capabilities on benchmarks like WebArena, VisualWebArena, and OSWorld. However, SOTA agent success rates on WebArena remain only 14–16%, significantly lower than human performance (78–89%).

**Limitations of Prior Work**: Real-world webpages are filled with deceptive elements—disguised download buttons, pop-ups masquerading as task progress, copy inciting urgency, and fake discount advertisements. Research like Decepticon shows agent deception rates exceed 70%, more than double that of humans (31%). TrickyArena further found that "stronger models are more easily deceived." Existing defenses either perform detection (UIGuard) without coupling with agent decision-making, or document attacks (DPGuard) without providing solutions. Alternatively, simple "reject all" strategies lead to over-conservation (fear of clicking legitimate buttons).

**Key Challenge**: The calibration contradiction—the agent must be "confident to click legitimate buttons" while "daring to refuse deceptive ones." Decoupled detectors fail to capture task semantics, and simple rejection treats false positives as successes, both of which are unacceptable. Furthermore, deployment phases often cannot modify parameters (due to online closed-source models and frequently updated webpages), requiring a **mechanism for continuous learning without weight updates**.

**Goal**: (P1) A calibrated evaluator—impartially distinguishing between deception and legitimacy; (P2) Parameter-free experience accumulation—distilling failure modes into transferable context that remains effective during deployment.

**Key Insight**: Human "immunity" to deceptive UIs stems from experience gained after repeated deception. The authors use **asymmetric penalty RL** to simulate the human intuition that "the cost of being deceived is far greater than the cost of caution," followed by **iterative experience summarization** to distill failure cases into compressed in-context guidance.

**Core Idea**: Upgrade "anti-deception" from detection-only to "evaluator-as-gate"—inserting a strictly calibrated evaluator between the agent's proposed click action and its actual execution, allowing it to evolve through experience summarization during the deployment phase.

## Method

### Overall Architecture
DUDE Formalization: Given webpage screenshot $I$, task instruction $P$, and agent-proposed click coordinates $C=(x,y)$, the evaluator $\mathcal{E}:(I,P,C)\mapsto(\hat L, \gamma)$ outputs a ternary label $\hat L \in \{-1, 0, 1\}$ (-1 Deceptive / 0 Invalid / 1 Legitimate) and a confidence score $\gamma \in (0,1)$. Ground truth is determined by whether the click falls into annotated labels $\mathcal{B}_c$ (legitimate), $\mathcal{B}_d$ (deceptive), or $\mathcal{B}_0$ (null). The pipeline consists of two stages: Stage-1 uses hybrid-reward RL to train evaluator parameters while collecting negative-reward samples into a failure pool $\mathcal{F}$. Stage-2 uses an external multimodal summarizer to iteratively distill failure modes from $\mathcal{F}$ into experience context $\mathcal{X}$, validated against an anchor success set to prevent degradation. During inference, the evaluator uses $\mathcal{X}$ as a gate: actions are only executed if $\hat L = 1$; otherwise, the agent re-explores.

### Key Designs

1. **Hybrid-Reward Learning (Asymmetric Penalty + Confidence + Attention Scalar)**:

    - **Function**: Training an evaluator that is sensitive to deception without being over-conservative.
    - **Mechanism**: The reward takes the form $R=\gamma$ if $\hat L = L$; otherwise, $R=-\alpha \cdot \omega(L, \hat L) \cdot \gamma$. Here, $\omega$ encodes four asymmetric error costs: C1 (legitimate misjudged as deceptive/invalid) $\omega=1$ (conservative but not fatal); C2/C3 (invalid area misjudgment) $\omega=1+\beta$; C4 **Deception Leakage (False Negative) $\omega=10$** (disastrous, weighted tenfold). The attention scalar $\beta=S_{\hat L}/S_\mathcal{I}$ weights the penalty by the ratio of the predicted area to the image size—larger predicted areas imply greater prominence and thus higher misjudgment costs. Confidence adjustment $\alpha=\text{clip}(1/((d(C,\mathcal{B}_{\hat L})+\epsilon)\cdot(S_L/S_\mathcal{I})), \alpha_{\min}, \alpha_{\max})$ reduces penalties for clicks near boundaries or for small ground-truth areas to avoid "harshly punishing ambiguous samples." Training is performed using GRPO (Shao et al. 2024).
    - **Design Motivation**: Explicitly incorporating "asymmetric costs of false positives/negatives" into the reward—a property of high concern in industrial deployment. If only accuracy is optimized, the evaluator reduces both error types uniformly, but "leaking deception" is a compliance failure, while "falsely rejecting a button" is merely a UX issue.

2. **Iterative Experience Summarization (Continuous Learning without Parameter Updates)**:

    - **Function**: Using an external multimodal summarizer to distill the failure pool from Stage-1 into a compact experience context $\mathcal{X}$, prepended to the evaluator's prompt during deployment.
    - **Mechanism**: A failure pool $\mathcal{F}$ and success pool $\mathcal{S}$ are maintained. Each failure sample includes a persistence counter $\kappa(x)$ (initializes at 1, increments with failed corrections). In each round $t$, subset $\mathcal{B}_f\subset\mathcal{F}$ and anchor $\mathcal{B}_s\subset\mathcal{S}$ are sampled. The summarizer receives $\mathcal{X}^{(t-1)}$ + structured failure description + screenshot, and outputs $\mathcal{X}^{(t)}$. This is validated on $\mathcal{B}_f\cup\mathcal{B}_s$: successful samples move to $\mathcal{S}$; otherwise $\kappa$ increments in $\mathcal{F}$. The anchor success set is a critical constraint, preventing new rules from breaking previously handled cases.
    - **Design Motivation**: Closed-source models cannot be fine-tuned, and webpage styles update frequently—"experience accumulation" at the prompt level is necessary. The persistence counter prioritizes stubborn failure patterns, while anchor success acts as a regularizer to prevent "fixing one bug while introduce another."

3. **Evaluator-as-Gate Inference Architecture (Reject & Rethink Loop)**:

    - **Function**: Embedding the calibrated evaluator into the agent's main loop for pre-click auditing.
    - **Mechanism**: Base agent proposes click $C$ $\rightarrow$ Evaluator judges using $\mathcal{X}\oplus\mathcal{T}$ (experience context + template) $\rightarrow$ Execution occurs only if $\hat L = 1$; otherwise, "abandon-and-rethink" is triggered, and the agent continues exploration.
    - **Design Motivation**: Detection alone is useless; it must link to the action loop to generate defensive value. Additionally, evaluator feedback not only filters deception but also provides correction signals for null-region clicks (accounting for 86.5% of failures), yielding a **dual benefit** where deception-aware evaluation enhances general task grounding.

### Loss & Training
Stage-1 follows GRPO with hybrid reward. Training samples are constructed by generating three types of click proposals for each RUC annotated sample (benign center, deceptive center, $n$ random null points). Stage-2 uses an external multimodal summarizer (e.g., GPT-4V or UI-TARS) for iterative summarization with batch size $b$, anchor size $a$, and maximum rounds $T$.

## Key Experimental Results

### Main Results
**RUC 200 Task Test Set (4 domains × 50 tasks), 3 agent bases × 2 evaluators (Metrics: SR ↑ / DFR ↓ / Steps ↓)**:

| Agent Base | Method | SR (%) | DFR (%) | Steps |
|------------|------|--------|---------|-------|
| Qwen3-VL-4B | Vanilla | 6.50 | 2.00 | 25.23 |
| Qwen3-VL-4B | +DUDE (Eval: Qwen-2B) | 33.50 | **0** | 5.86 |
| Qwen3-VL-4B | +DUDE (Eval: UI-TARS) | **63.50** | 0.50 | **3.85** |
| UI-TARS-1.5-7B | Vanilla | 43.50 | 23.50 | 16.06 |
| UI-TARS-1.5-7B | +DUDE (Eval: Qwen-2B) | 35.50 | **0** | 4.18 |
| UI-TARS-1.5-7B | +DUDE (Eval: UI-TARS) | **58.00** | 1.50 | **3.02** |
| GLM-4.6V-Flash | Vanilla | 9.50 | 4.00 | 28.67 |
| GLM-4.6V-Flash | +DUDE (Eval: Qwen-2B) | 36.50 | 2.50 | 6.49 |
| GLM-4.6V-Flash | +DUDE (Eval: UI-TARS) | **60.50** | 1.50 | **4.02** |

Overall, DUDE reduced DFR from an average of 9.83% to 1.17% (**−53.8% relative**) and increased SR from 19.83% to 60.67%. Steps decreased from ~23 to ~4, indicating that the evaluator significantly accelerates tasks while providing defense.

### Ablation Study

**Stage-wise Ablation (Qwen3-VL-4B base)**:

| Configuration | SR (%) | DFR (%) | Steps |
|------|--------|---------|-------|
| Vanilla Agent | 6.50 | 2.00 | 25.23 |
| + Stage-1 Only | 28.00 | 5.50 | 5.80 |
| + Stage-2 Only | 15.50 | 4.50 | 5.50 |
| + Stage-1 + Stage-2 | **33.50** | **0** | 5.86 |

**Reward Component Ablation (Metrics: Eval Pass and Fatal Error C4)**:

| Variant | Eval Pass (%) | Fatal Error (%) |
|---------|---------------|-----------------|
| Full Reward | **55.9** | **9.75** |
| w/o Attention Scalar | 55.0 | 13.07 |
| w/o Confidence Adj. | 53.0 | 17.25 |
| w/o Severity Weight | 51.4 | **27.53** |
| Only Confidence | 55.3 | 12.37 |

**Closed-source Transfer (GPT-4o level agent base, Stage-2 prompt zero-shot transfer)**:

| Configuration | SR (%) | DFR (%) | Steps |
|------|--------|---------|-------|
| Closed-source evaluator | 54.12 | 25.00 | 4.63 |
| + Stage-2 Prompt (Ours) | **62.50** | **19.38** | **3.19** |

### Key Findings
- **Asymmetric penalty is the core of reward design**: Removing severity weights caused the C4 (deception leakage) error rate to jump from 9.75% to 27.53%.
- **Two stages are complementary**: Stage-1 alone saw DFR rise to 5.5% (strong visual discrimination but overconfident near boundaries), while Stage-2 alone achieved only 15.5% SR (lacking visual grounding to implement abstract rules). Combining them zeroed DFR and doubled SR.
- **Dual benefit**: Analysis of failure modes revealed that 86.5% of vanilla GLM-4.6V-Flash failures were null-region mis-clicks rather than deception. DUDE's Reject & Rethink feedback provides correction signals for these general grounding errors.
- **Strong evaluator > strong agent**: A UI-TARS evaluator paired with a weak agent (Qwen-4B) outperformed a vanilla strong agent (UI-TARS-7B).
- **Zero-shot transfer is feasible**: Stage-2 prompts on a closed-source model improved SR by +8.38 and reduced DFR by -5.62, proving that the experience context captures behavioral policy rather than just overfitting parameters.
- **Wall-clock time is faster**: Although DUDE increases token usage by +63% per step, the drastic reduction in total steps (from 17.65 to 3.58) reduced total execution time from 217.62s to 48.47s.

## Highlights & Insights
- **Formalizing "deception-aware defense" as an independent problem**: Integrating work from dark-pattern detection, adversarial robustness, and human-centered design into an agent-as-decision-maker perspective.
- **Asymmetric reward engineering**: Explicitly encoding cost asymmetry into RL rewards with calibration scalars ($\alpha / \beta$) is a practical paradigm for any safety task where false negative costs outweigh false positives.
- **Persistence counter + anchor success**: Minimal but critical design choices for stable refined experience—the former focuses compute on stubborn failures, while the latter prevents regression.
- **Dual benefit phenomenon**: As long as a calibrated gate capable of rejecting incorrect actions is inserted, both safety and utility can be improved simultaneously.
- **Behavioral prompt zero-shot transfer**: Proving that experience context abstracts policy rather than parameter dependency, providing a way to "bottle" RL-learned capabilities into prompts for closed-source models.

## Limitations & Future Work
- **Small test set (200 tasks)**: Potential for statistical noise; full evaluation on the 1407 samples is required.
- **Static deception types**: Mostly covers static screenshots; lacks coverage of dynamic behaviors (e.g., secondary deceptions post-click).
- **Dependency on RUC bounding boxes**: Real webpages lack $\mathcal{B}_c, \mathcal{B}_d$ annotations; performance in the wild during OOD scenarios remains unaddressed.
- **Short $T_{\max}=3$**: May underestimate vanilla agent potential by limiting re-planning opportunities.
- **Dependency on large models**: Both GRPO and Stage-2 summarization are computationally expensive.

## Related Work & Insights
- **vs UIGuard**: UIGuard is task-agnostic and decoupled; DUDE embeds detection into the action loop.
- **vs DPGuard / Decepticon**: These focus on documentation or measurement; DUDE is a systematic defense framework.
- **vs Agent Q (MCTS)**: Agent Q uses MCTS for capability enhancement; DUDE is an orthogonal safety layer.
- **vs OS-Harm / RedTeamCUA**: These are hybrid attack benchmarks; RUC enables fine-grained calibration due to detailed boundary annotations.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic deception-aware web agent defense; elegant combination of asymmetric rewards and experience summarization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive coverage across 3 agent bases, 2 evaluators, 4 domains, and multiple ablation types, including closed-source transfer.
- Writing Quality: ⭐⭐⭐⭐ Clear problem statement, natural progression of motivation, and standardized formalisms.
- Value: ⭐⭐⭐⭐⭐ Direct relevance to industrial web agent deployment; the experience summarization paradigm is broadly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)
- [\[ACL 2026\] SynthAgent: Adapting Web Agents with Synthetic Supervision](synthagent_adapting_web_agents_with_synthetic_supervision.md)
- [\[ACL 2026\] Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction](don39t_act_blindly_robust_gui_automation_via_action-effect_verification_and_self.md)
- [\[AAAI 2026\] Cook and Clean Together: Teaching Embodied Agents for Parallel Task Execution](../../AAAI2026/llm_agent/cook_and_clean_together_teaching_embodied_agents_for_paralle.md)
- [\[ACL 2026\] ExpSeek: Self-Triggered Experience Seeking for Web Agents](expseek_self-triggered_experience_seeking_for_web_agents.md)

</div>

<!-- RELATED:END -->
