# Semiconductor Industry Analysis AI Agent

A sophisticated AI agent for semiconductor industry analysis using **ReAct (Reason-Act-Observe)** pattern with **RAG (Retrieval-Augmented Generation)**. The agent autonomously selects between document retrieval, industry-wide analysis, and real-time web search to provide comprehensive insights into the chip industry.



## 🚀 What it does

This AI agent specializes in semiconductor industry analysis with three core capabilities:

1. **📊 Company-Specific Analysis**: Retrieves and analyzes information from individual company 10-K filings
2. **🏭 Industry-Wide Comparative Analysis**: Performs cross-company analysis across major semiconductor firms
3. **🔍 Real-Time Market Intelligence**: Searches the web for current news, stock prices, and market developments

The agent uses a **ReAct pattern** to autonomously decide which tools to use based on your query, providing both historical insights from SEC filings and current market information.

## 🛠️ Technology Stack

- **🤖 AI Models**: [Upstage](https://upstage.ai/) for chat and embedding models
- **🗄️ Vector Database**: [Pinecone](https://pinecone.io/) for document storage and semantic search
- **🌐 Web Search**: [Tavily Search API](https://tavily.com/) for real-time information retrieval
- **🔄 Agent Framework**: [LangGraph](https://github.com/langchain-ai/langgraph) for ReAct agent orchestration
- **📚 Pattern**: ReAct (Reason-Act-Observe) + RAG (Retrieval-Augmented Generation)

## 📈 Data Sources

The agent analyzes **10-K SEC filings** from major semiconductor companies:

| Company | Document | Focus Areas |
|---------|----------|-------------|
| **NVIDIA** | `nvidia_10k.pdf` | AI/GPU computing, data center, gaming |
| **AMD** | `amd_10k.pdf` | Processors, graphics, data center |
| **Intel** | `intel_10k.pdf` | Processors, foundry services, enterprise |
| **Broadcom** | `broadcom_10k.pdf` | Networking, wireless, infrastructure |

*Documents located in `documents_pending/` directory*

## 🎯 Agent Capabilities

### 🔧 Autonomous Tool Selection
The agent intelligently chooses between three tools:

1. **Document Retrieval Tool**: For queries about specific companies or detailed financial data
2. **Industry Analysis Tool**: For comparative analysis across multiple semiconductor companies  
3. **Web Search Tool**: For current market information, recent news, and real-time data

### 💡 Smart Query Understanding
- Detects company names and applies appropriate filtering
- Balances results across multiple companies for comparative queries
- Combines historical 10-K data with current market intelligence

## 📝 Sample Questions

### Basic Company Queries
- "What is NVIDIA's revenue for fiscal year 2024?"
- "What are AMD's main risk factors?"
- "How much did Intel spend on R&D?"

### Industry Analysis 
- "Compare R&D spending across NVIDIA, AMD, Intel, and Broadcom"
- "How do profit margins compare across the semiconductor industry?"
- "Which companies are most exposed to AI/datacenter markets?"

### Real-Time Market Intelligence
- "What is NVIDIA's current stock price and recent performance?"
- "What recent news exists about AI chip demand in 2024?"
- "Any recent regulatory developments affecting semiconductors?"

### Hybrid Analysis
- "How has NVIDIA's financial performance compared to competitors, and what does recent market news say about their outlook?"
- "What do 10-K filings say about AI opportunities, and what is the current market reaction?"

📄 **See `sample_test_questions.md` for 40+ comprehensive test questions across all categories.**

## ⚙️ Setup

### 1. Environment Variables

Create a `.env` file with the following:

```bash
# Upstage AI Models
UPSTAGE_API_KEY=your-upstage-api-key

# Pinecone Vector Database  
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-index-name

# Tavily Search API
TAVILY_API_KEY=your-tavily-api-key
```

### 2. Get API Keys

#### Upstage AI
1. Sign up at [Upstage Console](https://console.upstage.ai/)
2. Create an API key for chat and embedding models
3. Add to `.env` file as `UPSTAGE_API_KEY`

#### Pinecone
1. Sign up at [Pinecone](https://pinecone.io/)
2. Create a serverless index with:
   - **Dimension**: 4096 (for Upstage embeddings)
   - **Metric**: cosine
   - **Cloud**: AWS (recommended)
3. Add API key and index name to `.env` file

#### Tavily Search
1. Sign up at [Tavily](https://tavily.com/)
2. Get your API key from the dashboard
3. Add to `.env` file as `TAVILY_API_KEY`

### 3. Install Dependencies

```bash
pip install -e .
```

## 🚀 Usage

### Start the LangGraph Server

```bash
langgraph dev --port 2024
```

## 📊 Architecture

```
User Query → Generate Query → Retrieve Documents → Agent Reasoning → [Tool Selection] → Final Response

Tools Available:
├── Document Retrieval (company-specific)
├── Industry Analysis (cross-company comparison)  
└── Web Search (real-time information)
```

The ReAct agent autonomously chooses which tools to use based on query analysis, ensuring optimal information gathering for each request.

---
