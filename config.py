"""
Configuration module for Sephira LLM backend.
Handles environment variables, API keys, and application settings.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Application configuration loaded from environment variables."""
    
    # OpenAI API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    
    # Data Configuration
    DATA_CSV_PATH: Path = Path(os.getenv("DATA_CSV_PATH", "all_indexes_beta.csv"))
    
    # Chart Configuration
    CHART_OUTPUT_DIR: Path = Path(os.getenv("CHART_OUTPUT_DIR", "static/charts"))
    CHART_DPI: int = int(os.getenv("CHART_DPI", "300"))
    CHART_FORMAT: str = os.getenv("CHART_FORMAT", "png")
    
    # Sephira Design System Colors
    COLOR_BG_PRIMARY: str = "#0A0D1C"
    COLOR_BG_SECONDARY: str = "#0F152F"
    COLOR_TEXT_PRIMARY: str = "#FFFFFF"
    COLOR_TEXT_SECONDARY: str = "rgba(255, 255, 255, 0.62)"
    COLOR_CARD_GRADIENT_TOP: str = "#121834"
    COLOR_CARD_GRADIENT_BOTTOM: str = "#090D20"
    
    # Typography
    FONT_FAMILY: str = "Helvetica Neue, Helvetica, Arial, sans-serif"
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_DEBUG: bool = os.getenv("API_DEBUG", "False").lower() == "true"
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60")) 
    
    # Guardrail Configuration
    ENABLE_GUARDRAILS: bool = os.getenv("ENABLE_GUARDRAILS", "True").lower() == "true"
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "2000"))
    
    # Session Management
    SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600")) 
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        if not cls.DATA_CSV_PATH.exists():
            raise FileNotFoundError(f"Data file not found: {cls.DATA_CSV_PATH}")
        
        # Create chart output directory if it doesn't exist
        cls.CHART_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        return True


# Global config instance
config = Config()

