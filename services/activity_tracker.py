"""
Activity tracking service for user analytics.
Tracks query types, countries, chart vs text requests, and use intensity.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import re
import logging

logger = logging.getLogger(__name__)


class ActivityTracker:
    """Service for tracking user activity and analytics."""
    
    def __init__(self):
        """Initialize activity tracker with in-memory storage."""
        # In-memory storage (use database in production)
        self.queries: List[Dict[str, Any]] = []
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Aggregated statistics
        self.country_mentions: Dict[str, int] = defaultdict(int)
        self.query_types: Dict[str, int] = defaultdict(int)  # "chart", "text"
        self.daily_usage: Dict[str, int] = defaultdict(int)
        self.blocked_queries: Dict[str, int] = defaultdict(int)  # block category -> count
    
    def track_query(self, session_id: str, query: str, query_type: str,
                   countries: List[str], blocked: bool = False,
                   block_category: Optional[str] = None,
                   chart_requested: bool = False) -> None:
        """
        Track a user query with all relevant metadata.
        
        Args:
            session_id: Session identifier
            query: User query text
            query_type: "chart" or "text"
            countries: List of countries mentioned or queried
            blocked: Whether query was blocked
            block_category: Category if blocked (data_extraction, etc.)
            chart_requested: Whether a chart was requested
        """
        today = datetime.now().date().isoformat()
        
        # Track query
        query_record = {
            "session_id": session_id,
            "query": query[:200],  # Truncate for storage
            "query_type": query_type,
            "countries": countries,
            "blocked": blocked,
            "block_category": block_category,
            "chart_requested": chart_requested,
            "timestamp": datetime.now().isoformat(),
            "date": today
        }
        self.queries.append(query_record)
        
        # Update aggregated statistics
        self.daily_usage[today] += 1
        
        if blocked and block_category:
            self.blocked_queries[block_category] += 1
        else:
            self.query_types[query_type] += 1
            
            for country in countries:
                self.country_mentions[country] += 1
        
        # Update session statistics
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "created_at": datetime.now().isoformat(),
                "query_count": 0,
                "chart_requests": 0,
                "text_queries": 0,
                "countries_queried": set(),
                "last_activity": datetime.now().isoformat()
            }
        
        self.sessions[session_id]["query_count"] += 1
        self.sessions[session_id]["last_activity"] = datetime.now().isoformat()
        
        if chart_requested:
            self.sessions[session_id]["chart_requests"] += 1
        else:
            self.sessions[session_id]["text_queries"] += 1
        
        self.sessions[session_id]["countries_queried"].update(countries)
        
        logger.debug(f"Tracked query: session={session_id}, type={query_type}, countries={countries}")
    
    def extract_countries_from_query(self, query: str, available_countries: List[str]) -> List[str]:
        """
        Extract country names mentioned in a query.
        
        Args:
            query: User query text
            available_countries: List of available countries
        
        Returns:
            List of countries mentioned in query
        """
        query_lower = query.lower()
        mentioned = []
        
        for country in available_countries:
            if country.lower() in query_lower:
                mentioned.append(country)
        
        return mentioned
    
    def get_daily_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get daily usage statistics for the last N days.
        
        Args:
            days: Number of days to include
        
        Returns:
            Dictionary with daily statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        daily_stats = {}
        for date_str, count in self.daily_usage.items():
            date_obj = datetime.fromisoformat(date_str).date()
            if date_obj >= cutoff_date.date():
                daily_stats[date_str] = count
        
        return daily_stats
    
    def get_country_statistics(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Get most queried countries.
        
        Args:
            top_n: Number of top countries to return
        
        Returns:
            List of dicts with country and query count
        """
        sorted_countries = sorted(
            self.country_mentions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return [
            {"country": country, "query_count": count}
            for country, count in sorted_countries
        ]
    
    def get_query_type_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on query types (chart vs text).
        
        Returns:
            Dictionary with query type statistics
        """
        total = sum(self.query_types.values())
        
        return {
            "chart_requests": self.query_types.get("chart", 0),
            "text_queries": self.query_types.get("text", 0),
            "total_queries": total,
            "chart_percentage": round(self.query_types.get("chart", 0) / total * 100, 2) if total > 0 else 0,
            "text_percentage": round(self.query_types.get("text", 0) / total * 100, 2) if total > 0 else 0
        }
    
    def get_blocked_query_statistics(self) -> Dict[str, int]:
        """
        Get statistics on blocked queries by category.
        
        Returns:
            Dictionary with block category counts
        """
        return dict(self.blocked_queries)
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on user sessions.
        
        Returns:
            Dictionary with session statistics
        """
        if not self.sessions:
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "average_queries_per_session": 0
            }
        
        total_queries = sum(s["query_count"] for s in self.sessions.values())
        active_sessions = len([
            s for s in self.sessions.values()
            if (datetime.now() - datetime.fromisoformat(s["last_activity"])).total_seconds() < 3600
        ])
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "average_queries_per_session": round(total_queries / len(self.sessions), 2) if self.sessions else 0,
            "total_queries": total_queries
        }
    
    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics summary.
        
        Returns:
            Dictionary with all analytics data
        """
        return {
            "daily_usage": self.get_daily_statistics(days=30),
            "top_countries": self.get_country_statistics(top_n=20),
            "query_types": self.get_query_type_statistics(),
            "blocked_queries": self.get_blocked_query_statistics(),
            "session_statistics": self.get_session_statistics(),
            "generated_at": datetime.now().isoformat()
        }
    
    def save_use_case(self, query: str, response: str, countries: List[str],
                     query_type: str, notes: Optional[str] = None) -> None:
        """
        Save an interesting use case for documentation.
        
        Args:
            query: User query
            response: LLM response
            countries: Countries involved
            query_type: Type of query
            notes: Optional notes about why this is interesting
        """
        use_case = {
            "query": query[:500],
            "response": response[:1000],
            "countries": countries,
            "query_type": query_type,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }
        
        # Append to a use cases file or in-memory list
        # In production, use a database
        logger.info(f"Use case saved: {query[:50]}...")
        
        # For now, we'll log it and could write to a file
        # In production, this should go to a database or document store
        # TODO: Implement persistent storage for use cases

