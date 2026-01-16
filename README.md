# Onboarding Multi-Agent Service

A Python service that analyzes onboarding Q&A responses through two parallel LLM agents using LangGraph.

## Features

- **Parallel Agent Processing**: Two LLM agents run simultaneously for faster analysis
- **InsightAgent**: Extracts natural language summary and key phrases
- **TraitAgent**: Generates personality trait scores (-1 to 1) with reasoning
- **Robust Error Handling**: Automatic retry with exponential backoff
- **Input Validation**: Comprehensive validation with Pydantic
- **Async Architecture**: Non-blocking LLM calls for high throughput

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Set the OpenAI API key in `.env`:
```
OPENAI_API_KEY=sk-proj-...
```

## Testing

### Run Unit Tests
```bash
python -m pytest tests/ -v
```

### Start the Server
```bash
python -m src.main
```

Server runs at `http://localhost:8000`

### Test the API

#### Valid Request (Should succeed)
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d @test_valid_baseline.json
```

Expected: 200 OK with insights and traits

#### Invalid Requests (Should fail with 422)

**Empty answer:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d @test_invalid_empty.json
```

**Too short answer:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d @test_invalid_short.json
```

### Check Health
```bash
curl http://localhost:8000/
```

## Project Structure
```
├── src/
│   ├── models.py          # Pydantic models with validation
│   ├── agents.py          # InsightAgent & TraitAgent (sync + async)
│   ├── graph.py           # LangGraph workflow with parallel execution
│   └── main.py            # FastAPI application
├── tests/
│   └── test_models.py     # Unit tests for data models
├── test_*.json            # Test case files
├── requirements.txt       # Python dependencies
├── .env                   # OpenAI API key
└── README.md             # This file
```

## Architecture
```
Request → FastAPI → LangGraph → [InsightAgent + TraitAgent] → Response
                                      ↓ parallel ↓
                                   OpenAI GPT-4o-mini
```

Both agents run in parallel, cutting response time in half.

## Example Output

**Input:**
```json
{
  "user_id": "user-123",
  "question": "What are you looking for in your dating life?",
  "answer": "I'm ready to find a serious partner and I'm tired of casual dating."
}
```

**Output:**
```json
{
  "user_id": "user-123",
  "insight": {
    "summary": "User seeks a committed relationship and is done with casual dating.",
    "keywords": ["serious partner", "committed relationship", "tired of casual"]
  },
  "traits": [
    {
      "name": "relationship_goal_readiness",
      "score": 0.9,
      "reason": "Explicitly states readiness for serious partner."
    },
    {
      "name": "openness_to_commitment",
      "score": 0.85,
      "reason": "Clear desire to move beyond casual dating."
    }
  ]
}
```

## Future Enhancements

*Given more time, would consider the below improvements:*

### Performance & Scalability
- **Caching**: LRU cache for repeated questions 
- **Rate Limiting**: Prevent abuse with per-IP limits
- **Database**: Store analysis history for analytics
- **Redis**: Distributed cache for multi-instance deployment

### Testing & Quality
- **Agent Unit Tests**: Mock LLM responses for comprehensive testing
- **Load Testing**: Verify performance under high traffic
- **Coverage**: Increase test coverage to 90%+

### Features
- **Third Agent**: Generate follow-up questions based on answers
- **Batch Processing**: Analyze multiple Q&As in single request
- **Webhooks**: Callback URLs for long-running requests

### Production Readiness
- **Monitoring**: Prometheus metrics for observability
- **Logging**: Structured logging with ELK stack
- **Authentication**: JWT tokens or API keys
- **Docker**: Containerized deployment
- **CI/CD**: Automated testing and deployment pipeline

### Prompt Engineering
- **Few-shot Examples**: Include examples in prompts for better quality
- **Chain-of-Thought**: Make agents explain their reasoning step-by-step
- **Prompt Templates**: Dynamic prompts based on question type
