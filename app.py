"""
FastAPI application for Sephira LLM backend.
Provides chat, chart generation, and data query endpoints.
"""

import logging
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uvicorn

from config import Config, config
from services.data_service import DataService
from services.guardrail_service import GuardrailService
from services.chart_service import ChartService
from services.llm_service import LLMService
from utils.validators import (
    validate_chart_request,
    validate_query_length,
    sanitize_input,
    validate_session_id
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sephira LLM API",
    description="LLM-powered backend for sentiment data analysis and visualization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (lazy loading in production)
data_service: Optional[DataService] = None
guardrail_service: Optional[GuardrailService] = None
chart_service: Optional[ChartService] = None
llm_service: Optional[LLMService] = None

# In-memory session storage (use Redis/DB in production)
sessions: Dict[str, Dict[str, Any]] = {}


def get_data_service() -> DataService:
    """Get or create DataService instance."""
    global data_service
    if data_service is None:
        data_service = DataService(Config.DATA_CSV_PATH)
    return data_service


def get_guardrail_service() -> GuardrailService:
    """Get or create GuardrailService instance."""
    global guardrail_service
    if guardrail_service is None:
        guardrail_service = GuardrailService()
    return guardrail_service


def get_chart_service() -> ChartService:
    """Get or create ChartService instance."""
    global chart_service
    if chart_service is None:
        chart_service = ChartService(Config.CHART_OUTPUT_DIR, Config.CHART_DPI)
    return chart_service


def get_llm_service() -> LLMService:
    """Get or create LLMService instance."""
    global llm_service
    if llm_service is None:
        llm_service = LLMService(
            get_data_service(),
            get_guardrail_service()
        )
    return llm_service


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str] = Field(None, description="Session identifier")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None, description="Previous conversation turns"
    )


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="LLM response text")
    chart_request: Optional[Dict[str, Any]] = Field(None, description="Chart request if needed")
    session_id: str = Field(..., description="Session identifier")
    blocked: bool = Field(False, description="Whether query was blocked")
    error: Optional[str] = Field(None, description="Error type if any")


class ChartRequest(BaseModel):
    """Chart generation request model."""
    countries: List[str] = Field(..., min_items=1, max_items=10, description="Country names")
    date_range: Dict[str, str] = Field(
        ..., description="Date range with 'start' and 'end' keys (YYYY-MM-DD)"
    )
    chart_type: str = Field("time_series", description="Chart type: time_series, comparison, regional")
    title: str = Field("Sentiment Trends", max_length=200, description="Chart title")


class ChartResponse(BaseModel):
    """Chart generation response model."""
    chart_url: str = Field(..., description="URL path to generated chart")
    base64_image: str = Field(..., description="Base64-encoded chart image")
    filename: str = Field(..., description="Generated chart filename")


class DataQueryRequest(BaseModel):
    """Data query request model."""
    query: str = Field(..., min_length=1, max_length=2000, description="Query string")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Query parameters")


class DataQueryResponse(BaseModel):
    """Data query response model."""
    results: Dict[str, Any] = Field(..., description="Query results")
    summary: str = Field(..., description="Text summary")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    data_loaded: bool = Field(..., description="Whether data is loaded")
    countries_count: int = Field(..., description="Number of available countries")
    date_range: Dict[str, str] = Field(..., description="Available date range")


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # region agent log
    try:
        with open('/Users/tanayj/Sephira/.cursor/debug.log', 'a') as f:
            import json, time
            f.write(json.dumps({"location":"app.py:210","message":"startup_event() called","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B","data":{"api_key_length":len(Config.OPENAI_API_KEY)}})+'\n')
    except: pass
    # endregion
    try:
        Config.validate()
        # region agent log
        try:
            with open('/Users/tanayj/Sephira/.cursor/debug.log', 'a') as f:
                import json, time
                f.write(json.dumps({"location":"app.py:218","message":"Config.validate() succeeded in startup","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B"})+'\n')
        except: pass
        # endregion
        logger.info("Configuration validated successfully")
        
        # Pre-load services
        get_data_service()
        get_guardrail_service()
        logger.info("Services initialized successfully")
        
    except Exception as e:
        # region agent log
        try:
            with open('/Users/tanayj/Sephira/.cursor/debug.log', 'a') as f:
                import json, time
                f.write(json.dumps({"location":"app.py:231","message":"startup_event() error","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B","data":{"error":str(e),"error_type":type(e).__name__}})+'\n')
        except: pass
        # endregion
        logger.error(f"Error during startup: {e}")
        raise


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        ds = get_data_service()
        countries = ds.get_countries()
        date_range = ds.get_date_range()
        
        return HealthResponse(
            status="healthy",
            data_loaded=True,
            countries_count=len(countries),
            date_range={"start": date_range[0], "end": date_range[1]}
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            data_loaded=False,
            countries_count=0,
            date_range={"start": "", "end": ""}
        )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for LLM interactions."""
    try:
        # Sanitize input
        sanitized_message = sanitize_input(request.message)
        
        # Validate query length
        if not validate_query_length(sanitized_message, Config.MAX_QUERY_LENGTH):
            raise HTTPException(
                status_code=400,
                detail=f"Query exceeds maximum length of {Config.MAX_QUERY_LENGTH} characters"
            )
        
        # Validate/generate session ID
        session_id = validate_session_id(request.session_id)
        
        # Get conversation history from session if available
        conversation_history = request.conversation_history
        if not conversation_history and session_id in sessions:
            conversation_history = sessions[session_id].get("history", [])
        
        # Get LLM service
        llm = get_llm_service()
        
        # Process query
        result = llm.process_query(
            sanitized_message,
            conversation_history=conversation_history,
            session_id=session_id
        )
        
        # Update session history
        if session_id not in sessions:
            sessions[session_id] = {"history": [], "created_at": datetime.now()}
        
        sessions[session_id]["history"].append({
            "user": sanitized_message,
            "assistant": result["response"],
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 20 turns per session
        if len(sessions[session_id]["history"]) > 20:
            sessions[session_id]["history"] = sessions[session_id]["history"][-20:]
        
        return ChatResponse(
            response=result["response"],
            chart_request=result.get("chart_request"),
            session_id=session_id,
            blocked=result.get("blocked", False),
            error=result.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-chart", response_model=ChartResponse)
async def generate_chart(request: ChartRequest):
    """Chart generation endpoint."""
    try:
        # Get services
        ds = get_data_service()
        cs = get_chart_service()
        
        # Validate chart request
        available_countries = ds.get_countries()
        date_range = ds.get_date_range()
        
        validated_request = validate_chart_request(
            request.dict(),
            available_countries,
            date_range
        )
        
        # Get data for countries
        data = ds.get_multiple_countries_data(
            validated_request["countries"],
            validated_request["date_range"]["start"],
            validated_request["date_range"]["end"]
        )
        
        if data.empty:
            raise HTTPException(
                status_code=404,
                detail="No data available for the specified countries and date range"
            )
        
        # Convert DataFrame to dict for chart service
        data_dict = data.to_dict(orient='records')
        
        # Generate chart
        chart_result = cs.generate_chart(
            data,
            validated_request["countries"],
            validated_request["date_range"],
            validated_request["chart_type"],
            validated_request["title"]
        )
        
        return ChartResponse(
            chart_url=chart_result["chart_url"],
            base64_image=chart_result["base64_image"],
            filename=chart_result["filename"]
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error in chart generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query-data", response_model=DataQueryResponse)
async def query_data(request: DataQueryRequest):
    """Direct data query endpoint."""
    try:
        # Sanitize input
        sanitized_query = sanitize_input(request.query)
        
        # Validate query length
        if not validate_query_length(sanitized_query, Config.MAX_QUERY_LENGTH):
            raise HTTPException(
                status_code=400,
                detail=f"Query exceeds maximum length of {Config.MAX_QUERY_LENGTH} characters"
            )
        
        # Check guardrails
        if Config.ENABLE_GUARDRAILS:
            guardrail = get_guardrail_service()
            is_allowed, rejection_reason, category = guardrail.validate_query(sanitized_query)
            
            if not is_allowed:
                raise HTTPException(
                    status_code=403,
                    detail=rejection_reason
                )
        
        # Get services
        ds = get_data_service()
        llm = get_llm_service()
        
        # Get data summary for query
        data_summary = llm.get_data_summary_for_query(sanitized_query)
        
        # Use LLM to process query with data context
        from utils.prompt_templates import get_data_query_prompt
        prompt = get_data_query_prompt(sanitized_query, data_summary)
        
        messages = [
            {"role": "system", "content": llm.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = llm.client.chat.completions.create(
            model=llm.model,
            messages=messages,
            temperature=llm.temperature
        )
        
        summary = response.choices[0].message.content
        
        # Get actual data results (aggregated)
        results = {
            "query": sanitized_query,
            "data_summary": data_summary
        }
        
        return DataQueryResponse(
            results=results,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in data query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.API_DEBUG
    )

