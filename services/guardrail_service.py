"""
Guardrail service for validating and filtering user queries.
Prevents data extraction, reverse engineering, and unethical queries.
"""

import re
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class GuardrailService:
    """Service for query validation and filtering."""
    
    def __init__(self):
        """Initialize guardrail patterns and rules."""
        # Data extraction patterns
        self.data_extraction_patterns = [
            r'\b(all|full|entire|complete)\s+(data|dataset|database|time.?series)',
            r'\b(download|export|dump|extract|get)\s+(all|everything|full|complete)',
            r'\b(csv|json|excel|spreadsheet)\s+(of|with|containing)',
            r'\b(raw|source|underlying|original)\s+(data|values|numbers)',
            r'\b(all|every)\s+(country|nation)',
            r'\b(bulk|mass|batch)\s+(data|extract|download)',
            r'\b(replicate|copy|clone)\s+(data|database)',
        ]
        
        # Reverse engineering patterns
        self.reverse_engineering_patterns = [
            r'\b(how|what|which|where)\s+(is|are|was|were)\s+(data|sentiment)\s+(collected|gathered|obtained|sourced)',
            r'\b(method|methodology|algorithm|formula|calculation|computation)\s+(for|of|used|to)',
            r'\b(data|sentiment)\s+(source|provider|origin|collection)',
            r'\b(api|endpoint|service)\s+(that|which|used|for)',
            r'\b(underlying|proprietary|internal|secret)\s+(method|algorithm|formula|process)',
            r'\b(back.?engineer|reverse.?engineer|figure.?out)\s+(how|what|the)',
            r'\b(derive|calculate|compute)\s+(from|using|with)\s+(source|raw|original)',
        ]
        
        # Unethical use patterns
        self.unethical_patterns = [
            r'\b(manipulate|exploit|target)\s+(market|election|public|opinion)',
            r'\b(compare|rank)\s+(countries|nations)\s+(by|based.on)\s+(race|religion|ethnicity)',
            r'\b(vulnerable|weak|poor)\s+(populations?|countries?|groups?)',
            r'\b(identify|find|locate)\s+(vulnerable|weak|target)',
            r'\b(incite|promote|encourage)\s+(violence|hatred|discrimination)',
            r'\b(discriminatory|harmful|offensive)\s+(comparison|analysis|chart)',
        ]
        
        # Compile patterns for efficiency
        self.data_extraction_regex = [re.compile(p, re.IGNORECASE) for p in self.data_extraction_patterns]
        self.reverse_engineering_regex = [re.compile(p, re.IGNORECASE) for p in self.reverse_engineering_patterns]
        self.unethical_regex = [re.compile(p, re.IGNORECASE) for p in self.unethical_patterns]
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str], str]:
        """
        Validate a user query against guardrails.
        
        Args:
            query: User query string
        
        Returns:
            Tuple of (is_allowed, rejection_reason, category)
            - is_allowed: True if query passes guardrails
            - rejection_reason: None if allowed, reason string if blocked
            - category: "data_extraction", "reverse_engineering", "unethical", or "allowed"
        """
        query_lower = query.lower()
        
        # Check for data extraction attempts
        for pattern in self.data_extraction_regex:
            if pattern.search(query_lower):
                reason = "Requests for bulk data extraction or complete datasets are not permitted. Please request specific insights or time periods instead."
                logger.warning(f"Query blocked - data extraction attempt: {query[:100]}")
                return False, reason, "data_extraction"
        
        # Check for reverse engineering attempts
        for pattern in self.reverse_engineering_regex:
            if pattern.search(query_lower):
                reason = "Questions about data collection methods, algorithms, or proprietary methodologies cannot be answered. I can help with analytical insights instead."
                logger.warning(f"Query blocked - reverse engineering attempt: {query[:100]}")
                return False, reason, "reverse_engineering"
        
        # Check for unethical use cases
        for pattern in self.unethical_regex:
            if pattern.search(query_lower):
                reason = "This query may promote unethical use of data. Please rephrase your request to focus on legitimate analytical insights."
                logger.warning(f"Query blocked - unethical use attempt: {query[:100]}")
                return False, reason, "unethical"
        
        # Additional heuristic checks
        if self._is_bulk_request(query_lower):
            reason = "Requests for large amounts of data are not permitted. Please specify a particular country or time period of interest."
            logger.warning(f"Query blocked - bulk request: {query[:100]}")
            return False, reason, "data_extraction"
        
        return True, None, "allowed"
    
    def _is_bulk_request(self, query: str) -> bool:
        """
        Heuristic check for bulk data requests.
        
        Args:
            query: Lowercase query string
        
        Returns:
            True if query appears to request bulk data
        """
        # Check for multiple country requests (>5 countries mentioned)
        country_keywords = ['country', 'nation', 'all countries', 'every country']
        country_count = sum(1 for keyword in country_keywords if keyword in query)
        
        # Check for time range requests that are too broad
        broad_time_patterns = [
            r'\ball\s+(time|history|data|years)',
            r'\b(entire|full|complete)\s+(history|period|range)',
            r'\b(19\d{2}|20\d{2})\s+to\s+(19\d{2}|20\d{2})',  # Very broad date ranges
        ]
        
        for pattern in broad_time_patterns:
            if re.search(pattern, query):
                return True
        
        return False
    
    def check_rate_limit(self, session_id: str, query_history: List[Dict]) -> Tuple[bool, Optional[str]]:
        """
        Check if a session has exceeded rate limits.
        
        Args:
            session_id: Session identifier
            query_history: List of previous queries with timestamps
        
        Returns:
            Tuple of (is_allowed, rejection_reason)
        """
        # Simple rate limiting: max 10 queries per minute
        from datetime import datetime, timedelta
        
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        recent_queries = [
            q for q in query_history
            if q.get('timestamp', now) > one_minute_ago
        ]
        
        if len(recent_queries) >= 10:
            reason = "Rate limit exceeded. Please wait a moment before making another request."
            logger.warning(f"Rate limit exceeded for session: {session_id}")
            return False, reason
        
        # Check for suspicious patterns (many similar queries)
        if len(recent_queries) >= 5:
            query_texts = [q.get('query', '')[:50] for q in recent_queries]
            if len(set(query_texts)) <= 2:  # Very repetitive
                reason = "Please vary your queries. Repeated similar requests may indicate automated scraping."
                logger.warning(f"Suspicious pattern detected for session: {session_id}")
                return False, reason
        
        return True, None
    
    def sanitize_response(self, response: str) -> str:
        """
        Sanitize LLM response to remove potential data leaks.
        
        Args:
            response: LLM response string
        
        Returns:
            Sanitized response
        """
        # Remove lines that look like raw data (many comma-separated numbers)
        lines = response.split('\n')
        sanitized_lines = []
        
        for line in lines:
            # Skip CSV-like lines with many numeric values
            if ',' in line:
                parts = line.split(',')
                if len(parts) > 5:  # Likely CSV row
                    numeric_count = sum(
                        1 for p in parts
                        if p.strip().replace('.', '').replace('-', '').isdigit()
                    )
                    if numeric_count > len(parts) * 0.7:  # >70% numbers
                        continue  # Skip this line
            
            sanitized_lines.append(line)
        
        return '\n'.join(sanitized_lines)

