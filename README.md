# Sephira LLM Backend

LLM-powered backend API for sentiment data analysis and visualization with OpenAI integration.

## Features

- **Chat Interface**: Interactive LLM-powered queries about sentiment data
- **Chart Generation**: Automated chart creation with Sephira branding, watermarks, and legal disclaimers
- **Data Protection**: Guardrails preventing data extraction, reverse engineering, and unethical use
- **Workflow Integration**: Multi-turn conversation support with session management

## Architecture

- **FastAPI**: Modern async web framework
- **OpenAI API**: GPT-4 for natural language processing
- **Pandas**: Data manipulation and querying
- **Matplotlib**: Chart generation with custom branding

## Project Structure

```
Sephira/
├── app.py                    # FastAPI application entry point
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies
├── services/
│   ├── data_service.py      # CSV data loading and querying
│   ├── guardrail_service.py # Query validation and filtering
│   ├── chart_service.py     # Chart generation with branding
│   └── llm_service.py       # OpenAI API integration
├── utils/
│   ├── prompt_templates.py  # System prompts and templates
│   └── validators.py        # Input/output validation
├── static/
│   └── charts/              # Generated chart storage
└── all_indexes_beta.csv     # Sentiment data
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Set the following environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export DATA_CSV_PATH="all_indexes_beta.csv"  # Optional, defaults to all_indexes_beta.csv
export CHART_OUTPUT_DIR="static/charts"      # Optional, defaults to static/charts
```

Or create a `.env` file:

```
OPENAI_API_KEY=your-openai-api-key-here
DATA_CSV_PATH=all_indexes_beta.csv
CHART_OUTPUT_DIR=static/charts
OPENAI_MODEL=gpt-4
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. Run the Server

```bash
python app.py
```

Or with uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### POST `/api/chat`

Chat endpoint for LLM interactions.

**Request:**
```json
{
  "message": "Show me sentiment trends for France over the last 5 years",
  "session_id": "optional-session-id",
  "conversation_history": []
}
```

**Response:**
```json
{
  "response": "LLM response text",
  "chart_request": {
    "needs_chart": true,
    "countries": ["France"],
    "date_range": {"start": "2020-01-01", "end": "2025-01-01"},
    "chart_type": "time_series",
    "title": "France Sentiment Trends"
  },
  "session_id": "session-id",
  "blocked": false
}
```

### POST `/api/generate-chart`

Generate a branded chart.

**Request:**
```json
{
  "countries": ["France", "Germany"],
  "date_range": {"start": "2020-01-01", "end": "2024-12-31"},
  "chart_type": "time_series",
  "title": "France vs Germany Sentiment Comparison"
}
```

**Response:**
```json
{
  "chart_url": "/static/charts/chart_France_Germany_20200101_20241231_20250101_123456.png",
  "base64_image": "base64-encoded-image-data",
  "filename": "chart_France_Germany_20200101_20241231_20250101_123456.png"
}
```

### POST `/api/query-data`

Direct data query endpoint.

**Request:**
```json
{
  "query": "What are the sentiment trends for Italy in 2023?",
  "parameters": {}
}
```

### GET `/api/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "data_loaded": true,
  "countries_count": 32,
  "date_range": {"start": "1970-01-01", "end": "2025-09-30"}
}
```

## Guardrails

The system includes multiple layers of protection:

1. **Data Extraction Prevention**: Blocks requests for bulk data, complete datasets, or raw data exports
2. **Reverse Engineering Prevention**: Blocks queries about data collection methods, algorithms, or methodologies
3. **Unethical Use Prevention**: Blocks queries that could be used for manipulation, discrimination, or harm
4. **Rate Limiting**: Prevents excessive requests and potential scraping

## Testing

Run basic tests:

```bash
python test_basic.py
```

This tests:
- Module imports
- Data service functionality
- Guardrail service validation
- Validator functions

## Security Notes

- **API Keys**: Never commit API keys to the repository. Use environment variables.
- **Data Protection**: The system is configured to exclude data from OpenAI training when possible. Verify OpenAI API settings.
- **Input Validation**: All user inputs are sanitized and validated.
- **Output Sanitization**: LLM responses are sanitized to prevent accidental data leaks.

## Configuration

Key configuration options in `config.py`:

- `OPENAI_API_KEY`: OpenAI API key (required)
- `OPENAI_MODEL`: Model to use (default: "gpt-4")
- `RATE_LIMIT_REQUESTS`: Maximum requests per window (default: 100)
- `RATE_LIMIT_WINDOW`: Time window in seconds (default: 60)
- `ENABLE_GUARDRAILS`: Enable/disable guardrails (default: True)
- `MAX_QUERY_LENGTH`: Maximum query length (default: 2000)

## Chart Branding

All generated charts include:
- Sephira color scheme matching the website design
- Copyright watermark (bottom-right)
- Footer with data source attribution and legal disclaimer
- High-resolution output (300 DPI by default)

## License

Copyright © Sephira. All rights reserved.
