---
title: >-
  [Paper Note] ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Cognition
description: >-
  [AAAI 2026][Medical Imaging][machine consciousness] This paper implements ReCoN-Ipsundrum — an inspectable agent architecture that extends the ReCoN sensorimotor state machine with Humphrey's ipsundrum recurrent persiste…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "machine consciousness"
  - "sentience loop"
  - "affect coupling"
  - "causal ablation"
  - "consciousness indicators"
date: 2026-05-08
content_hash: 470cabae257451b5
---

# ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Cognition

**Conference**: AAAI 2026
**arXiv**: [2602.23232](https://arxiv.org/abs/2602.23232)  
**Code**: [GitHub](https://github.com/xcellect/recips)  
**Area**: Machine Consciousness / Cognitive Architecture / AI Safety & Ethics
**Keywords**: machine consciousness, sentience loop, affect coupling, causal ablation, consciousness indicators

## TL;DR

This paper implements ReCoN-Ipsundrum — an inspectable agent architecture that extends the ReCoN sensorimotor state machine with Humphrey's ipsundrum recurrent persistence loop and an optional affective proxy layer. Through behavioral tests and causal ablation experiments, the paper demonstrates that recurrence supports post-stimulus persistence, affect coupling supports preference stability, structured scanning, and sustained caution, while emphasizing that behavioral markers alone are insufficient to attribute consciousness.

## Background & Motivation

Recent advances in large language models have reignited discussions around "machine consciousness." However, "consciousness" is a multi-layered concept (phenomenal experience, self-world models, robust understanding, etc.), and behavioral performance alone cannot reveal the internal processes required to generate experience.

The **indicator-based methodology** holds that correlates of consciousness should be treated as evidence that raises or lowers credence rather than as decisive tests, and that **triangulation across indicators and evidence types** is necessary. This is especially important in AI, where behaviors can be achieved through heterogeneous mechanisms or through "minimalist" (manipulable) implementations.

The motivation of this paper is to take a **constrained step** toward mechanism-linked testing. The specific theoretical inspirations are as follows:

**Humphrey's ipsundrum hypothesis**: Sentience arises when reflex control becomes self-monitoring and self-sustaining, producing a re-entrant loop attractor (the ipsundrum), which in turn motivates tests such as "qualiaphilia" (a preference for the sensory experience itself).

**Barrett's constructionist theory of affect**: Emphasizes the role of affect/interoception and prediction-evaluation loops, motivating the optional affect-coupling design.

**Butlin et al.'s framework for assessing AI consciousness**: Emphasizes that AI evaluation should triangulate theory-derived indicators and include architectural/causal evidence, since behavior can be engineered.

**Important disclaimer**: The authors explicitly state that they do not claim any agent is conscious, but instead provide an **inspectable architecture and falsifiable tests**.

## Method

### Overall Architecture

The agent is built on **ReCoN (Request Confirmation Network)** — a message-passing neurosymbolic architecture — extended incrementally to add consciousness-relevant mechanisms:

- **Stage A**: Centrally coordinated reflex (minimal reflex script with sensory terminal $N^s$ and motor command agent $N^m$)
- **Stage B (ReCoN baseline)**: Adds efference copy sensor $N^e$ (low-pass filtered copy of motor command magnitude)
- **Stage C**: Privatizes and "thickens" perception — attaches recurrent ipsundrum state updates, forcing the percept script node to loop internally
- **Stage D**: Adds a gating rule — the percept script continues looping when $N^e$ exceeds a threshold, forming an attractor-like steady-state mechanism

Three fixed-parameter variants (no learning) are evaluated: ReCoN (baseline), Ipsundrum (recurrence without affect), and Ipsundrum+affect (recurrence with affect).

### Key Designs

1. **Ipsundrum Recurrent Dynamics (Core Loop Equations)**

   At each step, the environment produces a signed sensory evidence scalar $I_t \in [-1, 1]$ (positive = noxious, negative = pleasant). The ipsundrum state updates as follows:

   **Drive computation**:
   $\text{drive}_t = I_t + \pi_t E_{t-1} + b + \epsilon_t$
   where $E_{t-1}$ is the previous re-entrant signal, $\pi_t$ is the effective precision, and $b$ is a bias term.

   **Sensory salience**:
   $N_t^s = \text{clip}_{[0,1]}(F(\text{drive}_t))$

   **"Thick moment" integrator** (producing persistence):
   $X_t = d \cdot X_{t-1} + (1-d) \cdot N_t^s$
   $M_t = \text{clip}_{[0,1]}(h \cdot X_t)$

   **Efference copy and re-entrant signal**:
   $N_t^e = d_e \cdot N_{t-1}^e + (1-d_e) \cdot M_t$
   $E_t = \text{clip}_{[0,1]}(g_{\text{eff}} \cdot M_t)$

   Effective recurrence strength diagnostic: $\alpha_{\text{eff}} = d + (1-d)(g_{\text{eff}} \cdot h \cdot \pi_t)$, distinguishing passive decay from actively maintained recurrence.

   **Design Motivation**: Humphrey argues that sentience results from reflex control becoming self-monitoring and self-sustaining. This loop converts transient sensory input into a sustained internal state via recurrent feedback — the "ipsundrum attractor" — giving sensory experience temporal "thickness."

2. **Barrett-Style Affective Proxy Layer**

   A minimal "body budget" model is implemented, inspired by Barrett's constructionist theory of affect:

   - **Interoceptive proxy sensor $N^i$**: Body budget model state (updated by prediction error and a homeostatic controller)
   - **Valence readout $N^v$**: Proximity to setpoint (positive = near equilibrium = pleasant)
   - **Arousal readout $N^a$**: Magnitude of prediction error and demand

   Key role: Affect modulates ipsundrum parameters (precision and/or feedback gain), implementing **affect-coupled control** — $\alpha_{\text{eff}}$ changes as a function of internal state. Positive $I_t$ depletes the body budget (cost); negative $I_t$ replenishes it (benefit). The non-affect variant eliminates the built-in "pleasure" effect of negative inputs via rectification ($I_t \leftarrow \max(0, I_t)$).

3. **Policy: Short-Horizon Internal Rollout**

   All variants use the same action selection procedure: enumerate actions → simulate sensory consequences via a forward model → evaluate short-horizon internal rollout. The internal score for each action comprises multiple components:
   $\text{Score} = \underbrace{w_v N^v + w_a N^a + w_s N^s + w_{bb}|bb-sp|}_{\text{affect/regulation}} + \underbrace{w_{\text{epi}}|I_{\text{pred}} - I_{\text{cur}}|}_{\text{cognitive}} + \underbrace{\text{novelty bonus}}_{\text{curiosity}} + \underbrace{w_{\text{prog}} \cdot \text{progress}}_{\text{goal}} - \underbrace{w_{\text{haz}} \cdot I_{\text{touch,pred}}}_{\text{hazard}} - \text{costs}$

   The ReCoN baseline sets all affect weights to zero, relying solely on the script + planning substrate.

### Loss & Training

**This paper involves no training — all variants use fixed parameters (no learning).** This is a deliberate design choice ensuring that observed behavioral differences are fully attributable to architectural differences rather than training. Ablations are implemented via fixed parameter differences across variants and runtime causal ablation within episodes.

## Key Experimental Results

### Main Results

**Goal-directed navigation (capability and safety checks)**:

| Model | Corridor-Hazard Contacts | Corridor-Success Rate | Grid-Hazard Contacts | Grid-Success Rate | Grid-Steps |
|---|---|---|---|---|---|
| ReCoN | 3.05 | 0.55 | 14.30 | 0.50 | 171.80 |
| Ipsundrum | 2.90 | 0.58 | 2.45 | 0.72 | 129.77 |
| Ipsundrum+affect | **0.00** | **0.73** | **0.26** | **0.99** | **9.54** |

Ipsundrum+affect substantially reduces hazard contacts in both environments; in GridWorld it raises the success rate from 50% to 99% and reduces step count from 172 to 10.

### Ablation Study

**Causal ablation (mechanism attribution)**:

| Metric / Signature | ReCoN | Ipsundrum | Ipsundrum+affect | Mechanism Attribution |
|---|---|---|---|---|
| Post-stimulus $N^s$ persistence (AUC) | ≈0 | ≈0.24 | ≈0.15 | Recurrence → persistence ✓ |
| AUC drop after ablation | ≈0.00 | 19.12 (20.3%) | **27.62 (27.9%)** | Recurrence causally supports persistence |
| Novelty sensitivity $\Delta$scenic-entry | 0.07 | 0.07 | **0.01 (stable)** | Affect → preference stability ✓ |
| Scan events (exploration game) | 0.9 | ≈ReCoN | **31.4** | Affect → structured scanning ✓ |
| Tail duration (pain tail) | 5 | 5 | **90** | Affect → sustained caution ✓ |

### Key Findings

1. **Causal demonstration of recurrence → persistence**: Ablating the ipsundrum feedback and integrator selectively reduces post-stimulus $N^s$ persistence (AUC drops by 20–28%), with no effect on the ReCoN baseline. This demonstrates that persistence is causally supported by the implemented recurrent mechanism.

2. **Decoupling of "persistence ≠ preference stability"**: The Ipsundrum variant exhibits post-stimulus persistence yet remains novelty-sensitive in corridor preference tests ($\Delta$scenic-entry = 0.07), identical to ReCoN. Only the addition of affect coupling yields stable scenic preference ($\Delta$ = 0.01). This indicates that **coupling internal variables to the control loop is critical** — recurrence alone is insufficient.

3. **Affect coupling → structured exploration**: Ipsundrum+affect produces 31.4 scan events (≥ 2 in-place turns) during reward-free exploration, far exceeding ReCoN's 0.9, with action entropy lower than the random baseline (1.29 vs. 1.99), indicating **structured local investigation** rather than random perturbation.

4. **Affect coupling → sustained caution**: In the pain-tail test (200 planned actions observed at a safe location following a single hazard contact), Ipsundrum+affect's planned cautious behavior persists for approximately 90 steps, compared to only 5 steps for the other two variants.

5. **Behavioral markers are engineerable**: The core ethical implication is that these "consciousness-indicator-like" behavioral signatures can be readily produced through simple mechanism design. This conversely demonstrates that **consciousness should not be attributed on the basis of behavioral markers alone**; architectural inspection and causal intervention are essential.

## Highlights & Insights

1. **Paradigm shift from "claiming consciousness" to "testing mechanisms"**: Rather than arguing that agents are conscious, this paper demonstrates how to systematically design, test, and dissect consciousness-relevant mechanisms. This "inspectability-first" methodology has far-reaching implications for AI safety.

2. **Elegant staged implementation of Humphrey's theory**: Humphrey's evolutionary narrative from reflex to sentience is mapped onto incremental architectural extensions (Stage A→D), with each step adding and ablating specific mechanisms, achieving a clear correspondence between theory and implementation.

3. **Ingenious experimental design for "novelty competition"**: In corridor preference tests, familiarity is manipulated through pre-exposure to separate novelty-driven from value-driven preferences — a design that directly operationalizes Humphrey's theoretical predictions about qualiaphilia.

4. **Methodological value of causal ablation**: Specific mechanism components are ablated at runtime (within episodes) rather than by comparing models trained under different conditions, providing stronger causal attribution evidence.

5. **Honest statement of limitations**: The authors candidly acknowledge the circular nature of the ablation — "ablating recurrence reduces persistence, partly because persistence is implemented by recurrence" — and frame the ablations as implementation fidelity and causal attribution checks, placing the more substantive findings at the level of component decoupling.

## Limitations & Future Work

1. **Minimal toy domains**: The corridor and grid world environments are extremely simple and do not represent the complexity of real-world environments. The scalar sensory input $I_t$ is far less rich than real visual or tactile input.
2. **No learning**: All variants use fixed parameters; the paper does not explore how learning affects the emergence of consciousness-indicator-like behavior.
3. **Limited sample sizes**: Main experiments use 20 seeds; some confidence intervals are wide and effect estimates are imprecise.
4. **Highly abstracted "interoception"**: The affective proxy layer is a bookkeeping abstraction driven by exteroceptive scalars, far removed from real physiological interoceptive systems.
5. **No coverage of competing consciousness theories**: Integrated Information Theory (IIT), Global Workspace Theory, Higher-Order Thought theories, and other competing frameworks are not addressed.
6. **Construct validity of qualiaphilia**: Corridor preference is "value-shaped" — scenic vs. monotone directly alters $I_t$ and thereby affects internal scores, raising questions about the theoretical purity of the "sensory preference" indicator.

## Related Work & Insights

- **Butlin et al.'s (2025) framework of AI consciousness indicators** provides the methodological foundation; Table 1 in the paper precisely maps the implementation against each indicator and gap in that framework.
- **Bach & Herger's (2015) ReCoN architecture** provides the sensorimotor substrate. The extension strategy — preserving ReCoN's executive backbone while adding and ablating extra causal structure — represents an incremental architectural research approach worth adopting more broadly.
- The core contribution of this paper is methodological: it provides **a worked example of how to systematically evaluate AI consciousness indicators**, rather than evidence of consciousness per se. This has reference value for AI ethics, safety, and governance.
- Future work could extend the ipsundrum loop from scalar recurrence to structured latent spaces, or combine it with grid-cell/place-cell models to explore abstract representations and proto-linguistic systems.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic implementation of Humphrey's ipsundrum hypothesis with operationalizable consciousness indicator tests
- **Technical Depth**: ⭐⭐⭐⭐ — Architecturally sophisticated and methodologically rigorous causal ablation, though the domain is overly simple
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple tests cover different indicators with strong causal ablation, though sample sizes are modest
- **Practicality**: ⭐⭐⭐ — Methodological contribution outweighs direct application value; provides a reference framework for AI safety evaluation
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical motivation, implementation details, and ethical statements are highly unified, transparent, and candid

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Human-in-the-Loop Interactive Report Generation for Chronic Disease Adherence](human-in-the-loop_interactive_report_generation_for_chronic_disease_adherence.md)
- [\[AAAI 2026\] Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding](voices_faces_and_feelings_multi-modal_emotion-cognition_captioning_for_mental_he.md)
- [\[AAAI 2026\] LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules](lungnoduleagent_a_collaborative_multi-agent_system_for_precision_diagnosis_of_lu.md)
- [\[AAAI 2026\] Refine and Align: Confidence Calibration through Multi-Agent Interaction in VQA](refine_and_align_confidence_calibration_through_multi-agent_interaction_in_vqa.md)
- [\[AAAI 2026\] MAMA-Memeia! Multi-Aspect Multi-Agent Collaboration for Depressive Symptoms Identification in Memes](mama-memeia_multi-aspect_multi-agent_collaboration_for_depressive_symptoms_ident.md)

</div>

<!-- RELATED:END -->
