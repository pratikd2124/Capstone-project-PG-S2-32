# Capstone-project-PG-S2-32

## Dataset Variables Scoring Guide

This document summarises the main survey variables, scoring rules, and interpretation categories used in the Resilience Survey dataset.

---

## 1. Children’s Hope Scale (CHS)

**Variables:**  
`chs1`, `chs2`, `chs3`, `chs4`, `chs5`, `chs6`

**Scoring:**  
- Sum all six items.
- If any item is missing, the total score should also be missing.

**Interpretation:**  
Higher scores indicate greater hope.

**Categories:**
- Low: 6–18
- Mid: 19–27
- High: 28–36

---

## 2. Depression Symptoms (PHQ-2)

**Variables:**  
`ph3`, `ph4`

**Scoring:**  
- Reverse-score both items.
- Reverse scoring:
  - 1 → 4
  - 2 → 3
  - 3 → 2
  - 4 → 1
- Sum the two reverse-scored values.
- If either item is missing, the final score should also be missing.

**Interpretation:**  
Higher scores indicate worse depression symptoms.

**Categories:**
- Low: 2–3
- Mid: 4–5
- High: 6–8

---

## 3. Anxiety Symptoms (GAD-2)

**Variables:**  
`ph1`, `ph2`

**Scoring:**  
- Reverse-score both items.
- Reverse scoring:
  - 1 → 4
  - 2 → 3
  - 3 → 2
  - 4 → 1
- Sum the two reverse-scored values.
- If either item is missing, the final score should also be missing.

**Interpretation:**  
Higher scores indicate worse anxiety symptoms.

**Categories:**
- Low: 2–3
- Mid: 4–5
- High: 6–8

---

## 4. Life Satisfaction

**Variable:**  
`cantril`

**Scoring:**  
Use the raw score directly.

**Interpretation:**  
Higher scores indicate greater life satisfaction.

**Categories:**
- Struggling: 1–4
- Doing OK: 5–6
- Thriving: 7–8

---

## 5. Avoidance Coping / Disengagement

**Variables:**  
`cop1`, `cop2`, `cop3`, `cop4`

**Scoring:**  
- Reverse-score all four items.
- Reverse scoring:
  - 1 → 4
  - 2 → 3
  - 3 → 2
  - 4 → 1
- Sum all reverse-scored items.
- If any item is missing, the final score should also be missing.

**Interpretation:**  
Higher scores indicate greater avoidance coping / disengagement.

**Categories:**
- Low: 4–8
- Mid: 9–10
- High: 11–16

---

## 6. Connection to Nature

**Variable:**  
`iins`

**Scoring:**  
Use the raw score directly.

**Interpretation:**  
Higher scores indicate stronger connection to nature.

**Categories:**
- Low: 1–2
- Mid: 3
- High: 4–5

---

## 7. School Engagement

**Variables:**  
`ry16`, `ry17`, `ry18`, `ry20`, `ry28`

**Scoring:**  
- Calculate the mean of all items.
- If any item is missing, the final score should also be missing.

**Interpretation:**  
Higher scores indicate greater school engagement.

**Categories:**
- Never or rarely: < 2
- Sometimes: 2 to < 3
- Often: 3 to < 4
- Always / Almost Always: 4

---

## 8. Friendship

**Variables:**  
`ry6`, `ry9`, `ry10`, `ry29`, `k12`

**Scoring:**  
- Calculate the mean of all items.
- If any item is missing, the final score should also be missing.

**Interpretation:**  
Higher scores indicate stronger friendships.

**Categories:**
- Never or rarely: < 2
- Sometimes: 2 to < 3
- Often: 3 to < 4
- Always / Almost Always: 4

---

## 9. Family Support

**Variables:**  
`ry2`, `ry5`, `ry7`, `ry8`, `ry13`, `ry27`

**Scoring:**  
- Calculate the mean of all items.
- If any item is missing, the final score should also be missing.

**Interpretation:**  
Higher scores indicate stronger support.

**Categories:**
- Never or rarely: < 2
- Sometimes: 2 to < 3
- Often: 3 to < 4
- Always / Almost Always: 4

---

## 10. Experience of Being Bullied

**Variables:**  
`ry37`, `ry38`, `ry39`

**Scoring:**  
- Calculate the mean of all items.
- If any item is missing, the final score should also be missing.

**Interpretation:**  
Higher scores indicate less frequent experience of bullying.

**Categories:**
- Daily: < 2
- Weekly: 2 to < 3
- Monthly: 3 to < 4
- Never: 4

---

## 11. Social Media

**Variables:**  
`sm3`, `sm4`

### `sm3`

Represents daily social media usage duration.

**Categories:**
- Never: 1
- Up to 2 hours each day: 2
- 2–4 hours each day: 3
- More than 4 hours each day: 4

### `sm4`

Used to calculate the proportion of students selecting each response option.

**Note:**  
Interpretation categories for `sm4` are not fully defined.

---

## 12. Healthy Behaviours

### Healthy Eating

**Variable:**  
`ry35`

**Scoring:**  
Use the raw score.

**Interpretation:**  
Higher scores indicate more frequent healthy eating.

---

### Adequate Sleep

**Variable:**  
`sun3`

**Scoring:**  
Use the raw score.

**Interpretation:**  
Higher scores indicate more frequent adequate sleep.

---

### Physical Activity

**Variable:**  
`ry36`

**Scoring:**  
Use the raw score.

**Interpretation:**  
Higher scores indicate more frequent physical activity.

---

## 13. Gender

**Variable:**  
`gender`

**Categories:**
- `M` = Male
- `F` = Female
- `O` = Other
- `P` = Prefer not to say / Not reported

For grouped reporting:
- Male = `M`
- Female = `F`
- Other / Not reported = `O`, `P`

---

## 14. Socioeconomic Status (SES)

**Variable:**  
`postcode`

For datasets from 2025 onward, the postcode represents the **school postcode**.

The postcode should be matched with the Australian Bureau of Statistics SEIFA data using the **Index of Relative Socio-economic Advantage and Disadvantage (IRSAD)**.

**Interpretation:**  
Higher SEIFA deciles indicate higher socioeconomic status.

**Categories:**
- Low SES: Deciles 1–3
- Mid SES: Deciles 4–7
- High SES: Deciles 8–10

---


3. Compare derived scores with the existing GemFinder implementation.
4. Prepare inputs for Stage 1 baseline reproduction.
5. Support later minimal-effects testing and robustness analysis.
