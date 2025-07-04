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

## Deepseek R1 Distill Llama 8B

### Regular dataset

**Test Summary: `DEEPSEEK_R1_DISTILL_LLAMA_8B`**

| Metric | Value |
| --- | --- |
| Total Cases | 115 |
| Passed | 36 (31.3%) |
| Failed | 79 |
| Average Execution Time | 13.17s |
| Average Input Tokens | 1599 |
| Average Output Tokens | 678 |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| 3 | 45 | 13 | 32 | 28.9% |
| 4 | 56 | 20 | 36 | 35.7% |
| 5 | 9 | 3 | 6 | 33.3% |
| 6 | 5 | 0 | 5 | 0.0% |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 55 | 15 | 40 | 27.3% |
| Task B: Take an object from a placement, and perform an action | 20 | 12 | 8 | 60.0% |
| Task C: Speak or answer a question | 70 | 22 | 48 | 31.4% |

### Enriched and reordered commands

**Test Summary: `DEEPSEEK_R1_DISTILL_LLAMA_8B`**

| Metric | Value |
| --- | --- |
| Total Cases | 115 |
| Passed | 34 (29.6%) |
| Failed | 81 |
| Average Execution Time | 16.82s |
| Average Input Tokens | 1601 |
| Average Output Tokens | 876 |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| 3 | 45 | 18 | 27 | 40.0% |
| 4 | 56 | 13 | 43 | 23.2% |
| 5 | 9 | 2 | 7 | 22.2% |
| 6 | 5 | 1 | 4 | 20.0% |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 55 | 18 | 37 | 32.7% |
| Task B: Take an object from a placement, and perform an action | 20 | 12 | 8 | 60.0% |
| Task C: Speak or answer a question | 70 | 17 | 53 | 24.3% |

## Gemini 2.5 Flash Lite

### Regular dataset

**Test Summary: `GEMINI_FLASH_LITE_2_5`**

| Metric | Value |
| --- | --- |
| Total Cases | 115 |
| Passed | 87 (75.7%) |
| Failed | 28 |
| Average Execution Time | 1.04s |
| Average Input Tokens | 1809 |
| Average Output Tokens | 125 |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| 3 | 45 | 34 | 11 | 75.6% |
| 4 | 56 | 43 | 13 | 76.8% |
| 5 | 9 | 5 | 4 | 55.6% |
| 6 | 5 | 5 | 0 | 100.0% |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 55 | 37 | 18 | 67.3% |
| Task B: Take an object from a placement, and perform an action | 20 | 12 | 8 | 60.0% |
| Task C: Speak or answer a question | 70 | 57 | 13 | 81.4% |

### Enriched and reordered commands

**Test Summary: `GEMINI_FLASH_LITE_2_5`**

| Metric | Value |
| --- | --- |
| Total Cases | 115 |
| Passed | 81 (70.4%) |
| Failed | 34 |
| Average Execution Time | 0.98s |
| Average Input Tokens | 1811 |
| Average Output Tokens | 120 |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| 3 | 45 | 41 | 4 | 91.1% |
| 4 | 56 | 36 | 20 | 64.3% |
| 5 | 9 | 2 | 7 | 22.2% |
| 6 | 5 | 2 | 3 | 40.0% |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 55 | 42 | 13 | 76.4% |
| Task B: Take an object from a placement, and perform an action | 20 | 12 | 8 | 60.0% |
| Task C: Speak or answer a question | 70 | 49 | 21 | 70.0% |

# Structured outputs

## Gemini 2.5 Flash

### BAML (Schema Aligned Parsing)

**Test Summary: `GEMINI_FLASH_2_5`**

| Metric | Value |
| --- | --- |
| Total Cases | 115 |
| Passed | 108 (93.9%) |
| Failed | 7 |
| Average Execution Time | 1.32s |
| Average Input Tokens | 1809 |
| Average Output Tokens | 129 |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| 3 | 45 | 42 | 3 | 93.3% |
| 4 | 56 | 54 | 2 | 96.4% |
| 5 | 9 | 8 | 1 | 88.9% |
| 6 | 5 | 4 | 1 | 80.0% |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 55 | 51 | 4 | 92.7% |
| Task B: Take an object from a placement, and perform an action | 20 | 19 | 1 | 95.0% |
| Task C: Speak or answer a question | 70 | 68 | 2 | 97.1% |

### JSON mode

**Test Summary: `GEMINI_FLASH_2_5`**

| Metric | Value |
| --- | --- |
| Total Cases | 115 |
| Passed | 92 (80.0%) |
| Failed | 23 |
| Average Execution Time | 3.83s |
| Average Input Tokens | 0 |
| Average Output Tokens | 0 |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| 3 | 45 | 40 | 5 | 88.9% |
| 4 | 56 | 47 | 9 | 83.9% |
| 5 | 9 | 4 | 5 | 44.4% |
| 6 | 5 | 1 | 4 | 20.0% |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 55 | 50 | 5 | 90.9% |
| Task B: Take an object from a placement, and perform an action | 20 | 17 | 3 | 85.0% |
| Task C: Speak or answer a question | 70 | 55 | 15 | 78.6% |

### Prompting

**Test Summary: `GEMINI_FLASH_2_5`**

| Metric | Value |
| --- | --- |
| Total Cases | 115 |
| Passed | 0 (0.0%) |
| Failed | 115 |
| Average Execution Time | 0.78s |
| Average Input Tokens | 0 |
| Average Output Tokens | 0 |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| 3 | 45 | 0 | 45 | 0.0% |
| 4 | 56 | 0 | 56 | 0.0% |
| 5 | 9 | 0 | 9 | 0.0% |
| 6 | 5 | 0 | 5 | 0.0% |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 55 | 0 | 55 | 0.0% |
| Task B: Take an object from a placement, and perform an action | 20 | 0 | 20 | 0.0% |
| Task C: Speak or answer a question | 70 | 0 | 70 | 0.0% |

## Gemini 2.5 Flash Lite

### BAML (Schema Aligned Parsing)

**Test Summary: `GEMINI_FLASH_LITE_2_5`**

| Metric | Value |
| --- | --- |
| Total Cases | 115 |
| Passed | 87 (75.7%) |
| Failed | 28 |
| Average Execution Time | 1.04s |
| Average Input Tokens | 1809 |
| Average Output Tokens | 125 |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| 3 | 45 | 34 | 11 | 75.6% |
| 4 | 56 | 43 | 13 | 76.8% |
| 5 | 9 | 5 | 4 | 55.6% |
| 6 | 5 | 5 | 0 | 100.0% |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 55 | 37 | 18 | 67.3% |
| Task B: Take an object from a placement, and perform an action | 20 | 12 | 8 | 60.0% |
| Task C: Speak or answer a question | 70 | 57 | 13 | 81.4% |

### JSON mode

**Test Summary: `GEMINI_FLASH_LITE_2_5`**

| Metric | Value |
| --- | --- |
| Total Cases | 115 |
| Passed | 74 (64.3%) |
| Failed | 41 |
| Average Execution Time | 2.12s |
| Average Input Tokens | 0 |
| Average Output Tokens | 0 |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| 3 | 45 | 30 | 15 | 66.7% |
| 4 | 56 | 42 | 14 | 75.0% |
| 5 | 9 | 0 | 9 | 0.0% |
| 6 | 5 | 2 | 3 | 40.0% |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 55 | 36 | 19 | 65.5% |
| Task B: Take an object from a placement, and perform an action | 20 | 9 | 11 | 45.0% |
| Task C: Speak or answer a question | 70 | 46 | 24 | 65.7% |

### Prompting

**Test Summary: `GEMINI_FLASH_LITE_2_5`**

| Metric | Value |
| --- | --- |
| Total Cases | 115 |
| Passed | 10 (8.7%) |
| Failed | 105 |
| Average Execution Time | 0.57s |
| Average Input Tokens | 0 |
| Average Output Tokens | 0 |

**Results by Command Count**

| Commands | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| 3 | 45 | 3 | 42 | 6.7% |
| 4 | 56 | 7 | 49 | 12.5% |
| 5 | 9 | 0 | 9 | 0.0% |
| 6 | 5 | 0 | 5 | 0.0% |

**Results by Task Type**

| Task Type | Total | Passed | Failed | Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| Task A: Navigate to a location, look for a person, and follow | 55 | 1 | 54 | 1.8% |
| Task B: Take an object from a placement, and perform an action | 20 | 0 | 20 | 0.0% |
| Task C: Speak or answer a question | 70 | 9 | 61 | 12.9% |

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