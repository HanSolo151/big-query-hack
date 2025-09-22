# 🧠 DevOps SaaS Agent System

An intelligent, multi-agent system for **automated incident resolution**, **semantic log search**, and **proactive outage detection**—powered by **Gemini AI**, **BigQuery**, and **BigFrames**.

---

## 📌 Overview

### 🚀 Project Goals

- Automate **incident triage and resolution** in DevOps / SaaS environments.
- Enable **semantic search** over logs, stack traces, alerts, and screenshots.
- Detect **emerging system-wide issues** before they escalate.
- Incorporate user feedback for **continual learning and improvement**.

---

## 🧠 Architecture

### 🎯 Orchestrator Agent (The Brain)

- Interprets user prompts and routes tasks to appropriate sub-agents.
- Coordinates embedding, search, summarization, explainability, and proactive logic.
- Ensures end-to-end flow from prompt → retrieval → recommendation → feedback.

---

## 🔹 Sub-Agents

### 1. **Embedding Agent**
- Generates vector embeddings via `TextEmbeddingGenerator`.
- Supports **text, logs, alerts, error codes, and multimodal inputs**.
- Builds and maintains a **vector index** (`create_vector_index`) for semantic retrieval.

### 2. **Search Agent**
- Executes `vector_search()` via **BigFrames / BigQuery**.
- Retrieves **top-K semantically similar incidents/tickets**.
- Returns associated metadata (timestamp, service, severity).

### 3. **Resolution Agent**
- Uses `GeminiTextGenerator` to summarize solutions.
- Generates **concise runbooks** or **multi-step action plans**.
- Can merge solutions across multiple past incidents.

### 4. **Explainability Agent**
- Adds **confidence scores and supporting evidence**.
- Example output:
  > "91% similar to Incident #456 (Jan 2024), 84% match with #372. Fix: DB connection pool reset."

### 5. **Multimodal Agent** *(Differentiator)*
- Handles unstructured data (screenshots, PDFs, logs).
- Leverages BigQuery **Object Tables + multimodal embeddings**.
- Example: user uploads Grafana dashboard → finds visually similar past incidents.

### 6. **Feedback Integration Agent** *(Learning Layer)*
- Accepts feedback: 👍 / 👎 or text-based comments.
- Feedback influences:
  - Embedding weights.
  - Ranking logic for resolutions.
  - Training set for future responses.

### 7. **Proactive Agent** *(Proactive Intelligence)*
- Continuously scans real-time incidents/logs/alerts.
- Clusters unresolved incidents with similar embeddings.
- Flags **emerging root causes / global outages** early.

---

## ⚙️ System Flow

### Example Prompt:
> "A customer reports payment failures after using PayPal checkout. How do we fix this?"

### Step-by-step Flow:

1. **Orchestrator** receives prompt.
2. → `Embedding Agent` vectorizes query.
3. → `Search Agent` retrieves top incidents.
4. → `Resolution Agent` summarizes fix.
5. → `Explainability Agent` provides confidence + evidence.
6. → `Feedback Agent` waits for user input.
7. → `Proactive Agent` checks for larger trend.

---

### 🧾 Final Output (Sample):

- ✅ **Recommended Fix**:  
  "Reinitialize PayPal token and clear session cache."

- 📌 **Supporting Evidence**:  
  Ticket #8931 (89% match), Ticket #7522 (85% match).

- 📊 **Confidence**:  
  High (avg similarity 87%).

- ⚠️ **Proactive Alert**:  
  "12 new tickets in the last hour mention PayPal checkout failures → possible incident."

- 🙋 **Feedback Prompt**:  
  "Did this solve your issue? [Yes / No] (or describe why not)"

---

## 🔍 Use Cases

| Use Case                        | Description                                                 |
| ------------------------------ | ----------------------------------------------------------- |
| 🔎 **Incident Triage**         | Auto-find resolutions for new or recurring system outages.  |
| 🧠 **Semantic Log Search**     | Match stack traces, logs, screenshots with past fixes.      |
| ⚠️ **Proactive Detection**     | Spot emerging issues before they become outages.            |
| 🧪 **Multimodal Analysis**     | Analyze screenshots, PDFs, log dumps using embeddings.      |
| 📈 **Continuous Learning**     | Improve recommendations based on user feedback.             |

---

## 🛠️ Stack

- **Gemini API** – for natural language understanding and summarization.
- **BigQuery + BigFrames** – for high-scale log storage and vector search.
- **BigQuery Object Tables** – for multimodal data (images, logs, PDFs).
- **Custom Python Agents** – for orchestration and agent interaction.

---

## 🔐 Credentials

**BigQuery Email (Service Account):**
