"""
Validation utilities for requests, responses, and data.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import re


def validate_country(country: str, available_countries: List[str]) -> bool:
    """
    Validate that a country name exists in the dataset.
    
    Args:
        country: Country name to validate
        available_countries: List of valid country names
    
    Returns:
        True if valid, False otherwise
    """
    # Case-insensitive matching with normalization
    country_normalized = country.strip().lower()
    available_normalized = [c.lower() for c in available_countries]
    
    return country_normalized in available_normalized


def validate_date_range(start_date: Optional[str], end_date: Optional[str], 
                       data_start: str, data_end: str) -> tuple:
    """
    Validate and normalize date range.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        data_start: Earliest available date
        data_end: Latest available date
    
    Returns:
        Tuple of (validated_start, validated_end)
    
    Raises:
        ValueError: If dates are invalid or out of range
    """
    if not start_date:
        start_date = data_start
    if not end_date:
        end_date = data_end
    
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        data_start_dt = datetime.strptime(data_start, "%Y-%m-%d")
        data_end_dt = datetime.strptime(data_end, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")
    
    if start_dt > end_dt:
        raise ValueError("Start date must be before end date")
    
    if start_dt < data_start_dt or end_dt > data_end_dt:
        raise ValueError(f"Date range must be between {data_start} and {data_end}")
    
    return start_date, end_date


def validate_chart_request(request: Dict[str, Any], available_countries: List[str],
                          date_range: tuple) -> Dict[str, Any]:
    """
    Validate chart generation request parameters.
    
    Args:
        request: Chart request dictionary
        available_countries: List of valid countries
        date_range: Tuple of (data_start, data_end)
    
    Returns:
        Validated and normalized request dictionary
    
    Raises:
        ValueError: If validation fails
    """
    # Validate countries
    countries = request.get("countries", [])
    if not countries:
        raise ValueError("At least one country must be specified")
    
    if not isinstance(countries, list):
        raise ValueError("Countries must be a list")
    
    if len(countries) > 10:  # Reasonable limit
        raise ValueError("Maximum 10 countries allowed per chart")
    
    validated_countries = []
    for country in countries:
        if not validate_country(country, available_countries):
            raise ValueError(f"Invalid country: {country}")
        validated_countries.append(country)
    
    # Validate date range
    date_range_dict = request.get("date_range", {})
    start_date = date_range_dict.get("start")
    end_date = date_range_dict.get("end")
    
    data_start, data_end = date_range
    validated_start, validated_end = validate_date_range(
        start_date, end_date, data_start, data_end
    )
    
    # Validate chart type
    chart_type = request.get("chart_type", "time_series")
    valid_types = ["time_series", "comparison", "regional"]
    if chart_type not in valid_types:
        chart_type = "time_series"  # Default
    
    # Validate title
    title = request.get("title", "Sentiment Trends")
    if len(title) > 200:
        title = title[:200]
    
    return {
        "countries": validated_countries,
        "date_range": {"start": validated_start, "end": validated_end},
        "chart_type": chart_type,
        "title": title
    }


def validate_query_length(query: str, max_length: int = 2000) -> bool:
    """
    Validate that a query is within length limits.
    
    Args:
        query: Query string
        max_length: Maximum allowed length
    
    Returns:
        True if valid, False otherwise
    """
    return len(query) <= max_length


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        text: Input text
    
    Returns:
        Sanitized text
    """
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # Limit length
    if len(text) > 5000:
        text = text[:5000]
    
    return text.strip()


def validate_session_id(session_id: Optional[str]) -> str:
    """
    Validate or generate a session ID.
    
    Args:
        session_id: Optional existing session ID
    
    Returns:
        Valid session ID
    """
    if session_id and re.match(r'^[a-zA-Z0-9_-]{1,100}$', session_id):
        return session_id
    
    # Generate a simple session ID if invalid
    import uuid
    return str(uuid.uuid4())

