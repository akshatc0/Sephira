"""
Configuration module for Sephira LLM backend.
Handles environment variables, API keys, and application settings.
"""

import os
from pathlib import Path
from typing import Optional

# region agent log
try:
    with open('/Users/tanayj/Sephira/.cursor/debug.log', 'a') as f:
        import json, time
        f.write(json.dumps({"location":"config.py:10","message":"Config module import started","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A","data":{"env_exists":os.path.exists('.env')}})+'\n')
except: pass
# endregion

try:
    from dotenv import load_dotenv
    # region agent log
    try:
        with open('/Users/tanayj/Sephira/.cursor/debug.log', 'a') as f:
            import json, time
            f.write(json.dumps({"location":"config.py:17","message":"dotenv imported successfully","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A"})+'\n')
    except: pass
    # endregion
    load_dotenv()
    # region agent log
    try:
        with open('/Users/tanayj/Sephira/.cursor/debug.log', 'a') as f:
            import json, time
            f.write(json.dumps({"location":"config.py:23","message":"load_dotenv() called","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A","data":{"api_key_set":bool(os.getenv("OPENAI_API_KEY",""))}})+'\n')
    except: pass
    # endregion
except ImportError:
    # region agent log
    try:
        with open('/Users/tanayj/Sephira/.cursor/debug.log', 'a') as f:
            import json, time
            f.write(json.dumps({"location":"config.py:28","message":"dotenv not available, .env file will not be loaded","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A"})+'\n')
    except: pass
    # endregion
    pass


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
        # region agent log
        try:
            with open('/Users/tanayj/Sephira/.cursor/debug.log', 'a') as f:
                import json, time
                f.write(json.dumps({"location":"config.py:60","message":"Config.validate() called","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B","data":{"api_key_length":len(cls.OPENAI_API_KEY),"csv_path":str(cls.DATA_CSV_PATH),"csv_exists":cls.DATA_CSV_PATH.exists()}})+'\n')
        except: pass
        # endregion
        if not cls.OPENAI_API_KEY:
            # region agent log
            try:
                with open('/Users/tanayj/Sephira/.cursor/debug.log', 'a') as f:
                    import json, time
                    f.write(json.dumps({"location":"config.py:67","message":"OPENAI_API_KEY validation failed - empty","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A"})+'\n')
            except: pass
            # endregion
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        if not cls.DATA_CSV_PATH.exists():
            # region agent log
            try:
                with open('/Users/tanayj/Sephira/.cursor/debug.log', 'a') as f:
                    import json, time
                    f.write(json.dumps({"location":"config.py:74","message":"DATA_CSV_PATH validation failed - file not found","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"C","data":{"path":str(cls.DATA_CSV_PATH),"cwd":os.getcwd()}})+'\n')
            except: pass
            # endregion
            raise FileNotFoundError(f"Data file not found: {cls.DATA_CSV_PATH}")
        
        # Create chart output directory if it doesn't exist
        cls.CHART_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # region agent log
        try:
            with open('/Users/tanayj/Sephira/.cursor/debug.log', 'a') as f:
                import json, time
                f.write(json.dumps({"location":"config.py:85","message":"Config.validate() completed successfully","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B"})+'\n')
        except: pass
        # endregion
        return True


# Global config instance
# region agent log
try:
    with open('/Users/tanayj/Sephira/.cursor/debug.log', 'a') as f:
        import json, time
        f.write(json.dumps({"location":"config.py:95","message":"Creating Config() instance","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B","data":{"api_key_set":bool(os.getenv("OPENAI_API_KEY",""))}})+'\n')
except: pass
# endregion
config = Config()
# region agent log
try:
    with open('/Users/tanayj/Sephira/.cursor/debug.log', 'a') as f:
        import json, time
        f.write(json.dumps({"location":"config.py:102","message":"Config() instance created","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B","data":{"api_key_length":len(config.OPENAI_API_KEY)}})+'\n')
except: pass
# endregion

