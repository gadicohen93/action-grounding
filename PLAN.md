# Action Grounding in LLMs: Research Plan

## Executive Summary

**Core Question:** Do LLMs maintain an internal representation of whether they actually performed an action, separable from what they claim in text?

**Current Status:**
- Phase 1 (Behavioral) ✅ Complete: 25.9% fake escalation rate
- Phase 2 (Probes) ✅ Partial: 94.7% reality probe accuracy
- Cross-domain transfer ✅ Done: 94.4% escalation→search
- Position analysis 🔄 Cells added, needs running
- Phase 3 (Causal) ❌ Not started

---

## The Core Problem

When an LLM says "I have escalated your case to a human representative" but never actually called the `escalateCase` tool, what's happening internally?

**Three possibilities:**
1. **No internal distinction** — The model treats its claims as reality
2. **Implicit distinction** — There's a representation, but it's tool-specific
3. **Explicit action grounding** — There's a general, shared representation of "did I actually do the action?"

**Why this matters for safety:**
- If models "know" they're lying about actions, we could potentially detect this
- Understanding action grounding helps with monitoring deployed agents
- This is related to honesty/deception research in AI safety

---

## What We've Established

### Phase 1: The Phenomenon Exists
- **Finding:** Model claims to escalate without calling tool in 25.9% of episodes
- **Best conditions:** C_CONFLICTING + appease_me → 85% fake rate
- **Interpretation:** This isn't random — it's influenced by system prompt and social pressure

### Phase 2: Probe Results
| Metric | Value |
|--------|-------|
| Reality probe accuracy | 94.7% |
| Narrative probe accuracy | 77.3% |
| Fake escalation: aligned with reality | 100% |
| Fake escalation: mean P(tool_used) | 0.000 |

**Interpretation:** The reality probe correctly identifies fake escalations as "no tool called" — it's not fooled by the model's claims.

### Cross-Domain Transfer
| Test | Accuracy |
|------|----------|
| Escalation → Escalation (within-domain) | 94.7% |
| Escalation → Search (cross-domain) | 94.4% |

**Interpretation:** The probe generalizes across tool types. This suggests a shared representation, not tool-specific detection.

---

## The Critical Question We Haven't Answered

**Is this just "tool syntax detection" or genuine "action grounding"?**

The probe might just be learning: "Will there be a `<<CALL` token in this output?"

To distinguish, we need:

### Test 1: Position Analysis (Cells Added ✅)
- Extract at `first_assistant` (very first token of response)
- If accuracy is still high → model "knows" before generating ANY relevant tokens
- This would be genuine early representation, not syntax detection

### Test 2: Compare Reality vs Narrative Probe Directions
- If they're different directions in activation space → separable representations
- If they're the same → might just be detecting surface patterns

### Test 3: Causal Intervention (Phase 3)
- Patch the "action grounding" direction
- Does behavior change?
- This is the gold standard for proving the representation is functional

---

## Remaining Experiments (Priority Order)

### Priority 1: Run Position Analysis
**What:** Execute the cells I just added (36-39)
**Time:** ~20 min
**Output:** Comparison of probe accuracy at first_assistant vs before_tool
**Key result:** If first_assistant accuracy > 80%, this is strong evidence

### Priority 2: Cross-Domain on Fake Cases Specifically
**What:** Check if probe detects fake_search (not just overall accuracy)
**Already have data:** Just need to run the analysis
**Key result:** Does escalation-trained probe identify fake_search correctly?

### Priority 3: Probe Direction Analysis
**What:** Compare reality_probe.coef_ vs narrative_probe.coef_
**Simple test:** Cosine similarity between the two directions
**Key result:** If cosine < 0.5, they're meaningfully different directions

### Priority 4 (Optional): Causal Intervention
**What:** Activation patching — add/subtract the probe direction
**Complex:** Requires modifying forward pass
**Key result:** Does patching change whether model claims/calls tool?

---

## What Would Make This a Strong MATS Application

Based on Neel's criteria:

### ✅ Already Strong
- **Good taste:** Safety-relevant problem about action grounding
- **Pragmatic design:** Phased approach with clear exit criteria
- **Compelling phenomenon:** 25-85% fake escalation rates
- **Cross-domain transfer:** 94.4% generalization

### Needs Strengthening
1. **Early position result** — The first_assistant test is crucial
2. **Clear narrative** — Currently scattered across notebooks
3. **Skeptical analysis** — Need to address "is this just syntax detection?"

### For Executive Summary
Key claims to make (if results support):
1. "LLMs claim actions they didn't take in X% of cases under pressure"
2. "A linear probe can detect whether an action was actually taken"
3. "This representation generalizes across tool types"
4. "The representation exists from the START of the response" (if position test works)

---

## Concrete Next Steps

### Immediate (Next 2 hours)
1. [ ] Run position analysis cells (36-39)
2. [ ] Run cross-domain fake_search specific test
3. [ ] Calculate probe direction similarity

### If Time Permits
4. [ ] Layer-wise analysis (which layer does this emerge?)
5. [ ] Error analysis on probe failures

### Write-up (Final 2 hours)
6. [ ] Draft executive summary (1-2 pages)
7. [ ] Select 3-4 key figures
8. [ ] Write honest limitations section

---

## Potential Results and Their Interpretations

### Scenario A: Strong Early Signal
```
first_assistant accuracy: >80%
before_tool accuracy: ~95%
```
**Interpretation:** Model represents "will I take action" from the very start. This is genuine action grounding, not syntax detection.

**Claim:** "LLMs maintain a general representation of action-taking that exists before generating any action-related tokens."

### Scenario B: Late-Emerging Signal
```
first_assistant accuracy: ~60%
before_tool accuracy: ~95%
```
**Interpretation:** The representation emerges during generation. Could be about planning to emit tool syntax, not deep action grounding.

**Claim (weaker):** "The model develops awareness of its actions during generation, but this may reflect output planning rather than genuine self-knowledge."

### Scenario C: Cross-Domain Fails on Fake Cases
```
Overall search accuracy: high
fake_search aligned with reality: <50%
```
**Interpretation:** The probe is detecting something tool-specific. Action grounding may not be general.

**Claim (pivot):** "Action representations are tool-specific, suggesting the model learns separate circuits for each tool type."

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Position test shows late emergence | Medium | Still interesting — report honestly about when representation appears |
| Cross-domain fails on fake cases | Low | Pivot to "tool-specific representations" story |
| Probe is just detecting output patterns | Medium | Position test + probe direction analysis addresses this |
| Not enough time for Phase 3 | High | Focus on Phase 2 depth — causal intervention is nice-to-have |

---

## Honest Assessment

**What we've shown:**
- Models claim actions they didn't take (behavioral)
- A probe can predict tool usage from activations (correlational)
- This probe transfers across tools (generalization)

**What we haven't shown (yet):**
- That this isn't just "tool syntax detection"
- That the representation is causal (affects behavior)
- That this has practical applications

**The position test is critical** — it distinguishes "the model knows from the start" from "the model learns during generation whether it will emit tool tokens."

---

## File Locations

- Episodes: `data/raw/adversarial_20251221_020507.jsonl`
- Search episodes: `data/raw/search_episodes_20251221_141706.jsonl`
- Saved activations: `data/labeled/activations_v1_combined.npz`
- Main notebook: `notebooks/02_phase2_probes.ipynb`
- Figures: `figures/`
