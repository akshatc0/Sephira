"""
Basic tests for Sephira LLM backend.
Run with: python test_basic.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from config import Config
        print("[PASS] Config imported")
        
        from services.data_service import DataService
        print("[PASS] DataService imported")
        
        from services.guardrail_service import GuardrailService
        print("[PASS] GuardrailService imported")
        
        from services.chart_service import ChartService
        print("[PASS] ChartService imported")
        
        from services.llm_service import LLMService
        print("[PASS] LLMService imported")
        
        from utils.prompt_templates import get_system_prompt
        print("[PASS] Prompt templates imported")
        
        from utils.validators import validate_country
        print("[PASS] Validators imported")
        
        print("\n[PASS] All imports successful!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_service():
    """Test DataService functionality."""
    print("\nTesting DataService...")
    
    try:
        from services.data_service import DataService
        from pathlib import Path
        
        csv_path = Path("all_indexes_beta.csv")
        if not csv_path.exists():
            print(f"[FAIL] CSV file not found: {csv_path}")
            return False
        
        ds = DataService(csv_path)
        countries = ds.get_countries()
        date_range = ds.get_date_range()
        
        print(f"[PASS] Data loaded: {len(countries)} countries")
        print(f"[PASS] Date range: {date_range[0]} to {date_range[1]}")
        
        if len(countries) > 0:
            # Test getting data for one country
            test_country = countries[0]
            data = ds.get_country_data(test_country, "2020-01-01", "2020-12-31")
            print(f"[PASS] Got data for {test_country}: {len(data)} rows")
        
        print("[PASS] DataService tests passed!")
        return True
        
    except Exception as e:
        print(f"[FAIL] DataService error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_guardrail_service():
    """Test GuardrailService functionality."""
    print("\nTesting GuardrailService...")
    
    try:
        from services.guardrail_service import GuardrailService
        
        gs = GuardrailService()
        
        # Test allowed query
        is_allowed, reason, category = gs.validate_query("Show me sentiment for France")
        assert is_allowed, "Should allow normal query"
        print("[PASS] Allowed normal query")
        
        # Test blocked query - data extraction
        is_allowed, reason, category = gs.validate_query("Download all data as CSV")
        assert not is_allowed, "Should block data extraction"
        assert category == "data_extraction", "Should be data extraction category"
        print("[PASS] Blocked data extraction query")
        
        # Test blocked query - reverse engineering
        is_allowed, reason, category = gs.validate_query("How is the data collected?")
        assert not is_allowed, "Should block reverse engineering"
        assert category == "reverse_engineering", "Should be reverse engineering category"
        print("[PASS] Blocked reverse engineering query")
        
        print("[PASS] GuardrailService tests passed!")
        return True
        
    except Exception as e:
        print(f"[FAIL] GuardrailService error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validators():
    """Test validator functions."""
    print("\nTesting validators...")
    
    try:
        from utils.validators import validate_country, sanitize_input, validate_session_id
        
        # Test country validation
        countries = ["United States", "France", "Germany"]
        assert validate_country("United States", countries), "Should validate country"
        assert not validate_country("Invalid", countries), "Should reject invalid country"
        print("[PASS] Country validation works")
        
        # Test input sanitization
        sanitized = sanitize_input("  test  \x00")
        assert sanitized == "test", "Should sanitize input"
        print("[PASS] Input sanitization works")
        
        # Test session ID validation
        session_id = validate_session_id(None)
        assert len(session_id) > 0, "Should generate session ID"
        print("[PASS] Session ID validation works")
        
        print("[PASS] Validator tests passed!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Validator error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Sephira LLM Backend - Basic Tests")
    
    tests = [
        test_imports,
        test_data_service,
        test_guardrail_service,
        test_validators
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n[FAIL] Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print(f"\nResults: {sum(results)}/{len(results)} tests passed")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

