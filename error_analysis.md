# ERROR_ANALYSIS.md
### ArvyaX ML Internship — Failure Case Analysis
**Model version:** predictions_train.csv (post-fix pipeline)  
**Dataset:** 1,200 training predictions  
**Uncertain cases:** 439 / 1200 (36.6%)  
**Mean confidence:** 0.703

---

## Case 1 — Three Words. A Lot Assumed.

**ID 486 | overwhelmed | intensity 3 | confidence 0.26 | uncertain YES**
`uncertainty_reasons: short_text (3 words); low_class_confidence (0.26)`
`what_to_do: movement | when_to_do: within_15_min`

The user wrote three words. We do not know what those three words were, but the model was confident enough to label them "overwhelmed" and prescribe physical movement within 15 minutes. The confidence was 0.26 — well below any reasonable threshold — and the uncertain flag was correctly set.

But here is the real problem: we are still *prescribing* something. The hedge message softens the tone, but the system still commits to an action. For a 3-word input from someone labeled overwhelmed, the correct response might be to ask a follow-up question rather than to recommend anything at all.

**Why it failed:** TF-IDF features built from 3 words produce a nearly empty vector. A few words happen to match high-weight overwhelmed tokens from training, and the model fires. It is not wrong to flag this — it did — but the downstream action should acknowledge that we essentially know nothing.

**What to try:** For inputs under 5 words, replace the action recommendation with a gentle check-in message. Something like: "You haven't shared much today — and that's okay. How are you feeling right now?" This is a better user experience and more honest about the model's limitations.

---

## Case 2 — Calm on the Surface, Turbulent Underneath

**ID 28 | calm | intensity 4 | confidence 0.55 | uncertain YES**
`uncertainty_reasons: state_intensity_conflict (calm,4)`
`what_to_do: rest | when_to_do: tonight`

The text classifier says calm. The intensity regressor says 4 out of 5. These two models are looking at different things — the classifier is reading the words, the regressor is reading the metadata (sleep, stress, energy levels) — and they are telling opposite stories.

**Why it failed:** The text genuinely sounds calm; the physiological signals are genuinely elevated. This is actually a real phenomenon — someone can write calmly while experiencing physical stress — but the system cannot tell the difference between that and a model error.

**What to try:** When state and intensity conflict, consider predicting a third possibility: "surface calm, underlying tension." This could map to a `sound_therapy` or `journaling` action — something that invites reflection rather than assuming either reading is correct.

---

## Case 3 — Neutral With a 4? Something Is Off.

**ID 132 | neutral | intensity 4 | confidence 0.55 | uncertain YES**
`uncertainty_reasons: state_intensity_conflict (neutral,4)`
`what_to_do: rest | when_to_do: tonight`

The word "neutral" implies nothing is particularly strong — yet intensity 4 says something strong is being felt. The model detected this conflict, flagged it, and sent the user to rest tonight with a hedged message.

**Why it failed:** The post-prediction consistency check correctly identifies the contradiction but the decision layer still uses the predicted intensity=4 to set the action, which picks `rest` via the evening baseline. The state and the timing do not talk to each other after the conflict is detected.

**What to try:** When a state-intensity conflict is detected, clip the effective intensity to the midpoint of the valid range for that state before passing it to the decision engine. For neutral, valid range is (1, 3), so intensity 4 becomes 3. This keeps the action more appropriate to the state.

---

## Case 4 — Sitting Right on the Fence

**ID 9 | restless | intensity 4 | confidence 0.77 | uncertain YES**
`uncertainty_reasons: borderline_intensity (3.56)`
`what_to_do: movement | when_to_do: now`

The raw intensity prediction was 3.56 — mathematically right between 3 and 4. Rounding took it to 4, and that single rounding decision changed the action from something mild (like `grounding/within_15_min`) to `movement/now`.

**Why it failed:** Intensity is a continuous spectrum but we force it into integers. Any value between 3.4 and 4.5 becomes 4. For a user near 3.5, whether they get intensity 3 or 4 determines a meaningfully different recommendation. The decision engine treats integers as if they have sharp boundaries, but the underlying signal does not.

**What to try:** Instead of a single integer, pass the raw float to the decision engine and use soft thresholds. Actions can stay the same at 3.0 and change gradually toward 4.0. Alternatively, create a "borderline" action path for raw predictions within 0.4 of any integer boundary.

---

## Case 5 — Focused, Apparently, Based on Almost Nothing

**ID 498 | focused | intensity 4 | confidence 0.36 | uncertain YES**
`uncertainty_reasons: short_text (3 words); low_class_confidence (0.36)`
`what_to_do: deep_work | when_to_do: within_15_min`

Three words landed the model on focused with intensity 4 and a recommendation to do deep work within 15 minutes. The confidence was 0.36 and the flag was set — but the action is still prescriptive.

Telling someone to do deep focused work based on a 3-word input is almost certainly wrong. Even if they are focused, we have not verified it. Even if the hedge message softens the delivery, the core recommendation is fragile.


**Why it failed:** Short text inputs produce high-variance predictions. The confidence cap for short texts (max 0.45) is working correctly. But the decision layer still maps that capped confidence to a real action, just with softer timing.

**What to try:** For uncertain_flag=1 combined with short text, consider replacing the action entirely with `pause/now` and a message that invites the user to share more before receiving a recommendation.

---

## Case 6 — Everything Is Uncertain at Once

**ID 100 | mixed | intensity 2 | confidence 0.078 | uncertain YES**
`uncertainty_reasons: borderline_intensity (2.43); low_class_confidence (0.078)`
`what_to_do: pause | when_to_do: now`

Confidence of 0.078. The model essentially has no idea what state this person is in. The intensity raw prediction was 2.43 — borderline between 2 and 3. And the class confidence across all emotional states was nearly uniform — the model was almost guessing.

This is the lowest-confidence case in the entire dataset. The system correctly routed to `pause/now` — the safest possible action — and added an uncertainty hedge.

The message generated was: "Today seems to have a few layers to it. I'm not entirely certain what's going on, but if I had to offer one thing — Before deciding anything, just pause. Sit with where you are for a minute. Nothing drastic — just a small nudge in the right direction."

That is actually a reasonable response for a 0.078 confidence scenario. The system behaved well here.

**Why it still matters:** Even though the outcome is acceptable, a confidence of 0.078 is a signal that the input was either extremely short, very ambiguous, or contained contradictory signals across features. Understanding why this specific input scored so low could reveal a class of inputs the model is systematically unprepared for. 70 cases have confidence below 0.45 — these deserve their own response path, not just a softer version of the standard one.

---

## Case 7 — Focused, but Told to Sleep

**ID 148 | focused | intensity 3 | confidence 0.82 | uncertain NO**
`uncertainty_reasons: none`
`what_to_do: rest | when_to_do: tonight`

This is a silent failure — no uncertain flag, reasonably high confidence, and yet the recommendation makes no sense. A focused person at intensity 3 is being told to prioritize sleep tonight.

There are 29 such cases, none of them flagged as uncertain because there is no feature-level signal of uncertainty — the prediction itself is confident, just logically wrong.

A focused person in the evening should perhaps journal, do light planning for tomorrow, or wind down intentionally — not simply be told to sleep. Sleep is always a safe default for evening, but it is the wrong default for someone whose emotional state is active engagement.

**Why it failed:** The fix was incomplete. The logic checks `timing == "now"` before redirecting, but the evening baseline produces `rest/tonight` directly from the time-of-day layer, bypassing the focused-state guard entirely.

---

## Case 8 — Movement for Someone Who Is Overwhelmed?

**ID 5 | overwhelmed | intensity 3 | confidence 0.80 | uncertain NO**
`uncertainty_reasons: none`
`what_to_do: movement | when_to_do: within_15_min`

The decision engine routes overwhelmed + moderate stress + adequate energy to `movement/within_15_min` via the stress-by-energy matrix. The rule was designed for restless or anxious states where physical discharge helps. For overwhelmed states, it is less clear.

For an anxious or restless person, physical movement is often grounding. For an overwhelmed person, overwhelm often comes from cognitive overload — too many things, not enough capacity. Telling someone to go for a walk when they feel buried under obligations can feel dismissive rather than helpful.

This is not a clean-cut failure. Movement can genuinely help overwhelmed users by breaking the cognitive loop. That reads with appropriate empathy.

**Why it matters:** At intensity 3, this is defensible. The escalation rules correctly catch intensity >= 4 for overwhelmed. But at intensity 3 with the stress matrix firing, the overwhelmed state gets the same treatment as an anxious state, which may not always be appropriate.

**What to try:** Add overwhelmed as a soft exception to the stress-energy matrix. When overwhelmed + stress >= 4 + intensity == 3, prefer `box_breathing` over `movement`. Movement stays valid as a secondary suggestion in the message, but the primary prescription should be calming rather than activating.

---

## Case 9 — The Model's Absolute Worst Guess

**ID 100 | mixed | intensity 2 | confidence 0.078 | uncertain YES**

When the model returns 0.078 confidence, the softmax probabilities across all state classes are nearly flat. The model is not choosing "mixed" because the text sounds mixed — it is choosing "mixed" because it cannot distinguish between any of the options and "mixed" is the label the training data happened to assign to similarly indecisive inputs.

This is honest in one sense. Mixed is the right label when everything is ambiguous. But it is correct for the wrong reason.

**Why it happened:** When a model predicts "mixed" with near-random confidence, the word "mixed" is not carrying information — it is carrying uncertainty dressed up as a label. The user would be better served by a system that says "I could not read this clearly" than one that maps their input to a state and proceeds with a plan.

**What to try:**Consider adding a confidence floor below which the system enters a completely different branch: not a hedged recommendation, but a genuine acknowledgment that the model does not have enough signal, paired with an invitation to share more.

---

## Case 10 — Two Words, Told to Do Deep Work

**ID 732 | focused | intensity 4 | confidence 0.33 | uncertain YES**
`uncertainty_reasons: borderline_intensity (3.59); short_text (2 words); low_class_confidence (0.33)`
`what_to_do: deep_work | when_to_do: within_15_min`

Two words. The model predicted focused at intensity 4 and recommended 60-90 minutes of deep focused work within 15 minutes. Three separate uncertainty signals fired simultaneously — borderline intensity, short text, and low confidence — and the system still committed to a work prescription.

The hedge message was: "There's good focus coming through. Your signals are a bit nuanced today, so this is more of a nudge than a prescription — In a bit, clear your notifications and block out 60-90 minutes of deep focus."

Read that carefully. We are calling for 60-90 minutes of deep work as a "nudge," based on two words, with three uncertainty flags active and confidence of 0.33. The language is softer but the advice is still substantial.

**Why it happened:** The uncertainty flag softened the timing from `now` to `within_15_min`, but did not change the action. With three simultaneous uncertainty signals, the action itself should have been reconsidered. The system treats each uncertainty signal as a timing modifier, not as evidence that the underlying prediction might be wrong.

**What to try:** Add a multi-signal gate. When two or more uncertainty signals fire simultaneously, default to `pause/now` regardless of the state prediction. Reserve specific prescriptions like `deep_work` for cases where the model has one clear signal and reasonable confidence.

---

## What These 10 Cases Tell Us About the System

Reading across all 10 cases, three root causes keep appearing.

**The two models do not know about each other.** The state classifier and intensity regressor were trained separately. They sometimes return combinations that are semantically impossible — calm at intensity 4, neutral at intensity 4. The conflict detection catches these correctly. But after the flag is set, the decision engine still uses the raw intensity to determine the action, which partly defeats the point.

**Short inputs break the feature pipeline more than we account for.** 115 out of 1,200 training samples came from very short text. TF-IDF was not designed for 2-word inputs. The confidence cap helps, but the system still commits to an action category when the honest answer is that there is not enough signal to say anything meaningful.

**The decision engine was designed for clear cases.** It handles the extremes well — clearly anxious, genuinely tired, confidently focused. What it struggles with is the messy middle: moderate stress, moderate energy, borderline intensity, mild uncertainty across several dimensions at once. Most real users will sit in this messy middle on most days.

The system is good at knowing when it is uncertain. The gap is in what it does with that uncertainty. Right now, uncertain cases get a softer message — but they still get a prescription. The next iteration should explore a genuinely different response path for multi-signal uncertain cases: one that asks rather than tells.

---