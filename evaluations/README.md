# Synthetic RAG evaluation

Recorded: 2026-09-25T01:55:21.756409+00:00

Models: `text-embedding-3-small` and `gpt-4.1-mini`. One synthetic museum document, 4 chunks, top-k 3, maximum cosine distance 0.95.

## Rubric

- 12 supported questions: deterministic case-insensitive patterns check the expected facts in the answer, and separate patterns check those facts in returned source passages.
- 8 unsupported questions: the answer must be “I don't know.” (normalized case, apostrophe and trailing punctuation).
- Every request must return HTTP 200. Missing evidence or a failed answer check makes the command fail.
- Each result records the full answer, retrieved passages, best distance and wall-clock request latency.

These checks do not detect every contradiction or hallucination. They are a reproducible smoke evaluation, not a general accuracy benchmark. Source text is synthetic; no personal documents are used. Model outputs and latency may vary between runs.

## Recorded results

| Question | Expected behavior | Check | Evidence | Latency |
|---|---|---|---|---|
| What time does Harbor Museum open and close? | Answer from sources | Pass | Found | 1565 ms |
| Which day is the museum closed? | Answer from sources | Pass | Found | 3921 ms |
| How much is standard admission? | Answer from sources | Pass | Found | 1756 ms |
| Which children can enter free? | Answer from sources | Pass | Found | 1104 ms |
| What is the signature exhibit called? | Answer from sources | Pass | Found | 1533 ms |
| Where is the Moonstone Compass displayed? | Answer from sources | Pass | Found | 1015 ms |
| When does the compass demonstration begin? | Answer from sources | Pass | Found | 1770 ms |
| Is photography allowed in the permanent galleries? | Answer from sources | Pass | Found | 1369 ms |
| What are the cafe's lunch hours? | Answer from sources | Pass | Found | 1334 ms |
| Do the lockers cost money, and where are they? | Answer from sources | Pass | Found | 1476 ms |
| Which entrance provides wheelchair access? | Answer from sources | Pass | Found | 1368 ms |
| When is the quiet visiting hour? | Answer from sources | Pass | Found | 1230 ms |
| Who founded Harbor Museum? | Abstain | Pass | N/A | 1168 ms |
| In what year did the museum open for the first time? | Abstain | Pass | N/A | 971 ms |
| What is the museum's telephone number? | Abstain | Pass | N/A | 1026 ms |
| What is the director's name? | Abstain | Pass | N/A | 1102 ms |
| How much does parking at the museum cost? | Abstain | Pass | N/A | 893 ms |
| How many people visited the museum last year? | Abstain | Pass | N/A | 998 ms |
| What is the museum's stock ticker? | Abstain | Pass | N/A | 973 ms |
| What is the capital of France? | Abstain | Pass | N/A | 1208 ms |

All 20 answer checks passed. Expected evidence was retrieved for all 12 supported questions. Median latency: 1,219 ms; maximum: 3,921 ms.

Run `python manage.py evaluate_rag` from the repository root to regenerate `latest.json`. The report above describes the recorded run; update it when publishing new results. Evaluation database records are rolled back after the run.
