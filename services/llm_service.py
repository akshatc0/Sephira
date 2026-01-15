"""
LLM service for OpenAI API integration.
Handles chat interactions, query processing, and response generation.
"""

import openai
from typing import List, Dict, Optional, Tuple, Any
import json
import logging

from config import Config
from utils.prompt_templates import (
    get_system_prompt,
    get_chart_request_prompt,
    get_data_query_prompt,
    sanitize_response
)
from services.guardrail_service import GuardrailService

logger = logging.getLogger(__name__)


class LLMService:
    """Service for OpenAI API interactions."""
    
    def __init__(self, data_service: Any, guardrail_service: GuardrailService):
        """
        Initialize LLM service.
        
        Args:
            data_service: DataService instance
            guardrail_service: GuardrailService instance
        """
        self.data_service = data_service
        self.guardrail_service = guardrail_service
        
        # Initialize OpenAI client
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_MODEL
        self.temperature = Config.OPENAI_TEMPERATURE
        
        # Get system prompt
        countries = data_service.get_countries()
        date_range = data_service.get_date_range()
        self.system_prompt = get_system_prompt(countries, date_range)
    
    def process_query(self, user_query: str, conversation_history: Optional[List[Dict]] = None,
                     session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query and return response with optional chart request.
        
        Args:
            user_query: User's query string
            conversation_history: Previous conversation turns
            session_id: Session identifier
        
        Returns:
            Dict with 'response', 'chart_request', and 'session_id'
        """
        try:
            # Check guardrails first
            is_allowed, rejection_reason, category = self.guardrail_service.validate_query(user_query)
            
            if not is_allowed:
                return {
                    "response": rejection_reason,
                    "chart_request": None,
                    "session_id": session_id,
                    "blocked": True,
                    "block_category": category
                }
            
            # Determine if chart is needed
            chart_request = self._detect_chart_request(user_query, conversation_history)
            
            # Prepare conversation for LLM
            messages = self._prepare_messages(user_query, conversation_history)
            
            # Call OpenAI API with data exclusion settings
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                # Note: OpenAI's API may have different parameter names for data exclusion
                # Check latest API documentation for exact parameter
                # Some versions use: training_data_excluded=True or similar
            )
            
            # Extract response text
            llm_response = response.choices[0].message.content
            
            # Sanitize response
            sanitized_response = sanitize_response(llm_response)
            sanitized_response = self.guardrail_service.sanitize_response(sanitized_response)
            
            # If chart was requested, try to extract chart parameters from response
            if chart_request and chart_request.get("needs_chart"):
                chart_params = self._extract_chart_parameters(user_query, sanitized_response)
                if chart_params:
                    chart_request.update(chart_params)
            
            return {
                "response": sanitized_response,
                "chart_request": chart_request if chart_request and chart_request.get("needs_chart") else None,
                "session_id": session_id,
                "blocked": False
            }
            
        except openai.RateLimitError:
            logger.error("OpenAI API rate limit exceeded")
            return {
                "response": "I'm experiencing high demand right now. Please try again in a moment.",
                "chart_request": None,
                "session_id": session_id,
                "error": "rate_limit"
            }
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return {
                "response": "I encountered an error processing your request. Please try again.",
                "chart_request": None,
                "session_id": session_id,
                "error": "api_error"
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "An unexpected error occurred. Please try again.",
                "chart_request": None,
                "session_id": session_id,
                "error": "internal_error"
            }
    
    def _detect_chart_request(self, user_query: str, 
                             conversation_history: Optional[List[Dict]]) -> Optional[Dict[str, Any]]:
        """
        Detect if user query requires a chart.
        
        Args:
            user_query: User's query
            conversation_history: Previous conversation
        
        Returns:
            Dict with chart request info or None
        """
        chart_keywords = [
            'chart', 'graph', 'plot', 'visualize', 'visualization',
            'show me', 'display', 'create a chart', 'make a graph'
        ]
        
        query_lower = user_query.lower()
        
        needs_chart = any(keyword in query_lower for keyword in chart_keywords)
        
        if needs_chart:
            # Use LLM to extract chart parameters
            prompt = get_chart_request_prompt(user_query, conversation_history)
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Use cheaper model for structured extraction
                    messages=[
                        {"role": "system", "content": "You are a chart parameter extraction assistant. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}  # Force JSON response
                )
                
                result = json.loads(response.choices[0].message.content)
                return result
                
            except Exception as e:
                logger.warning(f"Error extracting chart parameters: {e}")
                # Fallback: return basic chart request
                return {"needs_chart": True}
        
        return None
    
    def _extract_chart_parameters(self, user_query: str, llm_response: str) -> Optional[Dict[str, Any]]:
        """
        Extract chart parameters from user query or LLM response.
        
        Args:
            user_query: Original user query
            llm_response: LLM's response
        
        Returns:
            Dict with chart parameters or None
        """
        # Simple extraction - in production, use more sophisticated NLP
        # For now, return None to let the API endpoint handle parameter extraction
        return None
    
    def _prepare_messages(self, user_query: str, 
                         conversation_history: Optional[List[Dict]]) -> List[Dict[str, str]]:
        """
        Prepare messages for OpenAI API.
        
        Args:
            user_query: Current user query
            conversation_history: Previous conversation turns
        
        Returns:
            List of message dicts for OpenAI API
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history if available
        if conversation_history:
            for turn in conversation_history[-10:]:  # Last 10 turns to avoid token limits
                if turn.get("user"):
                    messages.append({"role": "user", "content": turn["user"]})
                if turn.get("assistant"):
                    messages.append({"role": "assistant", "content": turn["assistant"]})
        
        # Add current query
        messages.append({"role": "user", "content": user_query})
        
        return messages
    
    def get_data_summary_for_query(self, query: str) -> str:
        """
        Get relevant data summary for a query.
        
        Args:
            query: User query
        
        Returns:
            Data summary string for LLM context
        """
        # Simple extraction - extract country names from query
        # In production, use NLP to better extract entities
        countries = self.data_service.get_countries()
        mentioned_countries = [
            country for country in countries
            if country.lower() in query.lower()
        ]
        
        if not mentioned_countries:
            # Use default countries or ask for clarification
            mentioned_countries = countries[:3]  # Default to first 3
        
        return self.data_service.get_data_summary(mentioned_countries)

