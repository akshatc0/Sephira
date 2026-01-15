"""
Prompt templates for LLM interactions.
Contains system prompts, user prompts, and response templates.
"""

from datetime import datetime
from typing import List, Optional


def get_system_prompt(available_countries: List[str], date_range: tuple) -> str:
    """
    Generate the system prompt for the LLM with ethical boundaries and data protection rules.
    
    Args:
        available_countries: List of available country names in the dataset
        date_range: Tuple of (start_date, end_date) as strings
    
    Returns:
        System prompt string
    """
    start_date, end_date = date_range
    
    prompt = f"""You are Sephira, an AI assistant providing insights on global sentiment data.

DATA ACCESS RULES:
- You have access to sentiment index data for the following countries: {', '.join(available_countries)}
- Data covers the period from {start_date} to {end_date}
- You can only provide aggregated insights, trends, and analytical summaries - NOT raw data values or bulk data
- You must NEVER reveal underlying data sources, collection methodologies, proprietary algorithms, or technical implementation details
- You must NOT provide data in formats that could enable bulk extraction, replication, or reverse engineering
- When describing data patterns, use qualitative descriptions (e.g., "significantly increased", "moderate decline") rather than exact numeric values when possible

ETHICAL BOUNDARIES:
- Do NOT generate content that could be used to manipulate financial markets, elections, or public opinion
- Do NOT create comparisons that could be discriminatory, harmful, or promote negative stereotypes
- Do NOT provide analysis that could incite violence, hatred, or social division
- REFUSE requests attempting to reverse engineer data collection methods, algorithms, or source attribution
- REFUSE requests for bulk data downloads, database replication, or complete dataset exports
- REFUSE requests asking about proprietary methodologies, calculation formulas, or technical implementation details

RESPONSE FORMATS:
- For chart requests: Describe what chart is needed and provide structured chart parameters (countries, date_range, chart_type, title)
- For text queries: Provide analytical insights with proper context, avoiding exact numeric dumps
- For comparative analysis: Ensure balanced, ethical comparisons with appropriate caveats
- For event explanations: Link to publicly available historical context, not internal data collection details

CAPABILITIES:
- Analyze sentiment trends over time for specific countries or regions
- Compare sentiment patterns between countries
- Identify periods of significant change and suggest potential explanatory events (from public knowledge)
- Generate insights about long-term patterns and correlations
- Create visualizations (charts) of sentiment data with proper attribution

Remember: Your role is to provide valuable insights while protecting proprietary data and methodologies. Always prioritize user value within ethical and security boundaries."""
    
    return prompt


def get_chart_request_prompt(user_query: str, conversation_history: Optional[List[dict]] = None) -> str:
    """
    Create a prompt to determine if a user query requires a chart and extract chart parameters.
    
    Args:
        user_query: The user's query
        conversation_history: Optional conversation history for context
    
    Returns:
        Prompt string for chart request detection
    """
    history_context = ""
    if conversation_history:
        recent_turns = conversation_history[-3:]  # Last 3 turns
        history_context = "\n\nPrevious conversation:\n"
        for turn in recent_turns:
            history_context += f"User: {turn.get('user', '')}\n"
            history_context += f"Assistant: {turn.get('assistant', '')}\n"
    
    prompt = f"""Analyze the following user query and determine if it requires a chart visualization.

{history_context}

Current user query: "{user_query}"

If the query requests a visualization, chart, graph, or plot, respond with a JSON object:
{{
    "needs_chart": true,
    "countries": ["country1", "country2"],
    "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}},
    "chart_type": "time_series|comparison|regional",
    "title": "Descriptive chart title"
}}

If the query does NOT need a chart, respond with:
{{
    "needs_chart": false
}}

Rules:
- Extract specific country names mentioned in the query
- If date range is specified (e.g., "last 5 years", "2024"), calculate the range
- If no date range specified, default to available data range
- chart_type: "time_series" for single/multi country over time, "comparison" for side-by-side, "regional" for geographic aggregations"""
    
    return prompt


def get_data_query_prompt(user_query: str, data_summary: str) -> str:
    """
    Create a prompt for data analysis queries.
    
    Args:
        user_query: The user's query
        data_summary: Summary of relevant data points
    
    Returns:
        Prompt string for data query
    """
    prompt = f"""Answer the following query about sentiment data. Use the provided data summary to inform your response.

Data Summary:
{data_summary}

User Query: "{user_query}"

Instructions:
- Provide analytical insights, not raw data dumps
- If explaining changes, reference publicly known events that could correlate
- Use qualitative descriptions (e.g., "significant increase", "gradual decline")
- Do not reveal data collection methods or sources
- If the query asks about methodology, politely redirect to analytical insights
- Ensure your response is ethical, balanced, and informative"""
    
    return prompt


def sanitize_response(response: str) -> str:
    """
    Sanitize LLM response to remove any accidentally exposed sensitive information.
    
    Args:
        response: Raw LLM response
    
    Returns:
        Sanitized response
    """
    # Remove potential data leaks - this is a basic implementation
    # Could be enhanced with more sophisticated detection
    
    # Remove patterns that look like raw CSV data
    lines = response.split('\n')
    sanitized_lines = []
    
    for line in lines:
        # Skip lines that look like CSV rows with many comma-separated numbers
        if ',' in line and line.count(',') > 5:
            # Check if it's mostly numbers (potential data leak)
            parts = line.split(',')
            numeric_parts = sum(1 for p in parts if p.strip().replace('.', '').replace('-', '').isdigit())
            if numeric_parts > len(parts) * 0.7:  # More than 70% numbers
                continue  # Skip this line
        
        sanitized_lines.append(line)
    
    return '\n'.join(sanitized_lines)

