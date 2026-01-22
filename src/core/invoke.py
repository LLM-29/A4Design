import time
from typing import Any

from src.core.logger import Logger

def safe_invoke(runnable: Any, input_data: Any, **kwargs) -> Any:
    """
    Invoke a runnable (LLM or chain) with retry logic.
    
    Args:
        runnable: The runnable to invoke
        input_data: Input data for the runnable
        **kwargs: Additional keyword arguments
        
    Returns:
        Result from the runnable
        
    Raises:
        Exception: If all retries fail
    """
    max_retries = 3
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return runnable.invoke(input_data, **kwargs)
        except Exception as e:
            last_exception = e
            Logger.log_warning(
                f"LLM call failed (attempt {attempt+1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
    
    Logger.log_error(f"Max retries reached for LLM call: {last_exception}")
    raise last_exception

