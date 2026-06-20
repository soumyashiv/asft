import logging
from typing import List, Callable, Optional
from datetime import datetime

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from asft.core.settings import get_settings
from asft.db.database import get_db
from asft.db.models import User

logger = logging.getLogger(__name__)

settings = get_settings()
security = HTTPBearer()

ALGORITHM = "HS256"

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Validate JWT token and retrieve the active user."""
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError as e:
        logger.warning("JWT validation failed: %s", e)
        raise credentials_exception
        
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
        
    return user


class RoleChecker:
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: User = Depends(get_current_user)):
        """Check if the user has any of the allowed roles."""
        user_roles = [role.name for role in user.roles]
        
        # 'Admin' always bypasses role checks
        if "Admin" in user_roles:
            return user
            
        for role in self.allowed_roles:
            if role in user_roles:
                return user
                
        logger.warning("User %s attempted to access resource requiring roles: %s", user.username, self.allowed_roles)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation not permitted"
        )

# Pre-defined dependencies
require_admin = RoleChecker(["Admin"])
require_researcher = RoleChecker(["Admin", "Researcher"])
require_agent = RoleChecker(["Admin", "Researcher", "Agent"])
require_readonly = RoleChecker(["Admin", "Researcher", "Agent", "ReadOnly"])
