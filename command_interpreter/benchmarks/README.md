# Model Performance Tester

```bash
cd command_interpreter/benchmarks
pip install -r benchmark-requirements.txt
python model_performance_tester.py
```

# Summary of results
## Fine-tuned model inferenced locally
### Regular dataset

**Test Summary: `LOCAL_FINETUNED`**

| Metric | Value |
| --- | --- |
| Total Cases | 230 |
| Passed | 230 (100.0%) |
| Failed | 0 |
| Average Execution Time | 3.25s |
| Average Input Tokens | 58 |
| Average Output Tokens | 70 |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| 3 | 90 | 90 | 0 | 100.0% |
| 4 | 112 | 112 | 0 | 100.0% |
| 5 | 18 | 18 | 0 | 100.0% |
| 6 | 10 | 10 | 0 | 100.0% |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 110 | 110 | 0 | 100.0% |
| Task B: Take an object from a placement, and perform an action | 40 | 40 | 0 | 100.0% |
| Task C: Speak or answer a question | 140 | 140 | 0 | 100.0% |

### Enriched and reordered commands

**Test Summary: `LOCAL_FINETUNED`**

| Metric | Value |
| --- | --- |
| Total Cases | 230 |
| Passed | 222 (96.5%) |
| Failed | 8 |
| Average Execution Time | 3.12s |
| Average Input Tokens | 60 |
| Average Output Tokens | 70 |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| 3 | 90 | 90 | 0 | 100.0% |
| 4 | 112 | 106 | 6 | 94.6% |
| 5 | 18 | 16 | 2 | 88.9% |
| 6 | 10 | 10 | 0 | 100.0% |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 110 | 103 | 7 | 93.6% |
| Task B: Take an object from a placement, and perform an action | 40 | 38 | 2 | 95.0% |
| Task C: Speak or answer a question | 140 | 135 | 5 | 96.4% |

## Gemini 2.5 Pro
### Regular dataset

**Test Summary: `GEMINI_PRO_2_5`**

| Metric | Value |
| --- | --- |
| Total Cases | 230 |
| Passed | 225 (97.8%) |
| Failed | 5 |
| Average Execution Time | 8.63s |
| Average Input Tokens | 1810 |
| Average Output Tokens | 752 |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| 3 | 90 | 89 | 1 | 98.9% |
| 4 | 112 | 108 | 4 | 96.4% |
| 5 | 18 | 18 | 0 | 100.0% |
| 6 | 10 | 10 | 0 | 100.0% |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 110 | 106 | 4 | 96.4% |
| Task B: Take an object from a placement, and perform an action | 40 | 39 | 1 | 97.5% |
| Task C: Speak or answer a question | 140 | 137 | 3 | 97.9% |

### Enriched and reordered commands

**Test Summary: `GEMINI_PRO_2_5`**

| Metric | Value |
| --- | --- |
| Total Cases | 230 |
| Passed | 208 (90.4%) |
| Failed | 22 |
| Average Execution Time | 9.24s |
| Average Input Tokens | 1811 |
| Average Output Tokens | 853 |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| 3 | 90 | 84 | 6 | 93.3% |
| 4 | 112 | 98 | 14 | 87.5% |
| 5 | 18 | 18 | 0 | 100.0% |
| 6 | 10 | 8 | 2 | 80.0% |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 110 | 94 | 16 | 85.5% |
| Task B: Take an object from a placement, and perform an action | 40 | 39 | 1 | 97.5% |
| Task C: Speak or answer a question | 140 | 129 | 11 | 92.1% |


# Results execution

### Not grounded

**Test Summary: `LOCAL_FINETUNED`**

| Metric | Value |
| --- | --- |
| Total Cases | 230 |
| Passed | 64 (27.8%) |
| Failed | 166 |
| Average Execution Score | 0.649 |
| Average Execution Time | 0.54s |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate | Avg Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 3 | 90 | 24 | 66 | 26.7% | 0.651 |
| 4 | 112 | 27 | 85 | 24.1% | 0.629 |
| 5 | 18 | 11 | 7 | 61.1% | 0.761 |
| 6 | 10 | 2 | 8 | 20.0% | 0.669 |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate | Avg Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 110 | 22 | 88 | 20.0% | 0.594 |
| Task B: Take an object from a placement, and perform an action | 40 | 15 | 25 | 37.5% | 0.583 |
| Task C: Speak or answer a question | 140 | 42 | 98 | 30.0% | 0.692 |

### Grounded

**Test Summary: `LOCAL_FINETUNED`**

| Metric | Value |
| --- | --- |
| Total Cases | 230 |
| Passed | 111 (48.3%) |
| Failed | 119 |
| Average Execution Score | 0.758 |
| Average Execution Time | 0.55s |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate | Avg Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 3 | 90 | 34 | 56 | 37.8% | 0.739 |
| 4 | 112 | 56 | 56 | 50.0% | 0.759 |
| 5 | 18 | 15 | 3 | 83.3% | 0.851 |
| 6 | 10 | 6 | 4 | 60.0% | 0.752 |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate | Avg Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 110 | 39 | 71 | 35.5% | 0.713 |
| Task B: Take an object from a placement, and perform an action | 40 | 27 | 13 | 67.5% | 0.716 |
| Task C: Speak or answer a question | 140 | 74 | 66 | 52.9% | 0.789 |