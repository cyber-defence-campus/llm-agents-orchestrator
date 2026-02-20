import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agent_framework.audit")

def log_command(command: str, agent_id: str, job_id: Optional[str] = None):
    """
    Logs a command execution to an operation-specific audit file.
    """
    operation_id = job_id or os.getenv("PLATFORM_SESSION_ID")
    
    if not operation_id:
        logger.warning(f"No operation_id found for auditing command: {command[:50]}...")
        operation_id = "unknown"

    try:
        # Determine log directory - prioritize /app/logs if it exists
        log_base = Path("/app/logs") if Path("/app/logs").exists() else Path("logs")
        audit_dir = log_base / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        audit_file = audit_dir / f"operation_{operation_id}.log"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format: [2024-05-20 12:00:00] [agent_xyz] command
        log_entry = f"[{timestamp}] [{agent_id}] {command}\n"
        
        with open(audit_file, "a") as f:
            f.write(log_entry)
            
    except Exception as e:
        logger.error(f"Failed to write to audit log: {e}")
