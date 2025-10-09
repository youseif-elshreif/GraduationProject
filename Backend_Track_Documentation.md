# Backend Track Documentation - IDS-AI System

## Table of Contents
1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [API Design](#api-design)
6. [Database Integration](#database-integration)
7. [Authentication & Authorization](#authentication--authorization)
8. [Real-time Communication](#real-time-communication)
9. [AI Model Integration](#ai-model-integration)
10. [Network Actions](#network-actions)
11. [Security Implementation](#security-implementation)
12. [Performance Optimization](#performance-optimization)
13. [Testing Strategy](#testing-strategy)
14. [Deployment & DevOps](#deployment--devops)
15. [Monitoring & Logging](#monitoring--logging)

## Overview

The backend serves as the central orchestrator of the IDS-AI system, responsible for processing AI detection results, managing user authentication, providing REST APIs, handling real-time communications, and executing network security actions. Built with FastAPI for high performance and scalability.

### Core Responsibilities
- **AI Integration**: Receive and process real-time attack detection results
- **API Gateway**: Provide comprehensive REST API for frontend and external systems
- **Authentication Service**: JWT-based user authentication and role management
- **Real-time Communication**: WebSocket connections for instant alerts and updates
- **Network Actions**: Execute security responses (IP blocking, port management)
- **Data Management**: Store and retrieve flows, alerts, and user actions
- **Audit Logging**: Comprehensive logging for security and compliance

## Architecture Design

### Microservices Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Service    │───▶│  Backend Core   │◀───│  Frontend App   │
│   (Port 8001)   │    │   (Port 8000)   │    │   (Port 3000)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Database      │◀───│   Redis Cache   │    │  Network Tools  │
│  (PostgreSQL)   │    │   (Port 6379)   │    │  (iptables/fw)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Service Layer Architecture

```python
# Core Service Layers
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                     │
├─────────────────────────────────────────────────────────────┤
│                   Business Logic Layer                      │
├─────────────────────────────────────────────────────────────┤
│                    Data Access Layer                        │
├─────────────────────────────────────────────────────────────┤
│                  External Services Layer                    │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Core Technologies
```python
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.7
redis==5.0.1
celery==5.3.4
python-socketio==5.10.0
python-multipart==0.0.6
python-jose==3.3.0
passlib==1.7.4
bcrypt==4.0.1
pydantic==2.5.0
pydantic-settings==2.1.0
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
```

### Additional Dependencies
```python
# Development & Production Tools
gunicorn==21.2.0          # Production WSGI server
prometheus-client==0.19.0  # Metrics collection
structlog==23.2.0         # Structured logging
typer==0.9.0              # CLI commands
python-dotenv==1.0.0      # Environment management
cryptography==41.0.8      # Encryption utilities
aiofiles==23.2.1          # Async file operations
httpx==0.25.2             # Async HTTP client
```

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration management
│   ├── database.py             # Database connection and setup
│   ├── dependencies.py         # Dependency injection
│   │
│   ├── api/                    # API route handlers
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py         # Authentication endpoints
│   │   │   ├── alerts.py       # Alert management endpoints
│   │   │   ├── network.py      # Network action endpoints
│   │   │   ├── users.py        # User management endpoints
│   │   │   ├── stats.py        # Statistics endpoints
│   │   │   └── inference.py    # AI inference endpoints
│   │   └── deps.py             # API dependencies
│   │
│   ├── core/                   # Core business logic
│   │   ├── __init__.py
│   │   ├── auth.py             # Authentication logic
│   │   ├── security.py         # Security utilities
│   │   ├── config.py           # Configuration models
│   │   └── exceptions.py       # Custom exceptions
│   │
│   ├── services/               # Business services
│   │   ├── __init__.py
│   │   ├── alert_service.py    # Alert processing logic
│   │   ├── network_service.py  # Network action service
│   │   ├── user_service.py     # User management service
│   │   ├── stats_service.py    # Statistics service
│   │   ├── ai_service.py       # AI integration service
│   │   └── notification_service.py # Real-time notifications
│   │
│   ├── models/                 # Database models
│   │   ├── __init__.py
│   │   ├── user.py             # User models
│   │   ├── alert.py            # Alert models
│   │   ├── flow.py             # Network flow models
│   │   ├── action.py           # Network action models
│   │   └── base.py             # Base model classes
��   │
│   ├── schemas/                # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── user.py             # User schemas
│   │   ├── alert.py            # Alert schemas
│   │   ├── auth.py             # Authentication schemas
│   │   ├── network.py          # Network action schemas
│   │   └── common.py           # Common schemas
│   │
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py           # Logging configuration
│   │   ├── cache.py            # Redis cache utilities
│   │   ├── validators.py       # Custom validators
│   │   └── helpers.py          # Helper functions
│   │
│   └── workers/                # Background tasks
│       ├── __init__.py
│       ├── celery_app.py       # Celery configuration
│       ├── alert_processor.py  # Alert processing tasks
│       └── network_tasks.py    # Network action tasks
│
├── migrations/                 # Database migrations
├── tests/                      # Test files
├── scripts/                    # Utility scripts
├── docker/                     # Docker configurations
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
└── README.md                   # Project documentation
```

## API Design

### FastAPI Application Setup

```python
# app/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
import socketio
from contextlib import asynccontextmanager

from app.api.v1 import auth, alerts, network, users, stats, inference
from app.core.config import settings
from app.database import engine, create_tables
from app.utils.logger import setup_logging

# Socket.IO setup
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=settings.CORS_ORIGINS,
    logger=True,
    engineio_logger=True
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    await create_tables()
    print("🚀 Backend server started successfully")
    yield
    # Shutdown
    print("👋 Backend server shutting down")

app = FastAPI(
    title="IDS-AI Backend API",
    description="Backend API for AI-powered Intrusion Detection System",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Socket.IO integration
socket_app = socketio.ASGIApp(sio, app)

# API Routes
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["alerts"])
app.include_router(network.router, prefix="/api/v1/network", tags=["network"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(stats.router, prefix="/api/v1/stats", tags=["statistics"])
app.include_router(inference.router, prefix="/api/v1/inference", tags=["ai-inference"])

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT
    }

@app.get("/")
async def root():
    return {"message": "IDS-AI Backend API", "docs": "/docs"}
```

### Configuration Management

```python
# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List, Optional
import secrets

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    JWT_SECRET_KEY: str = secrets.token_urlsafe(32)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = "postgresql://user:pass@localhost:5432/ids_db"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    
    # AI Service
    AI_SERVICE_URL: str = "http://localhost:8001"
    AI_SERVICE_TIMEOUT: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Network Actions
    ENABLE_NETWORK_ACTIONS: bool = True
    IPTABLES_PATH: str = "/sbin/iptables"
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### Authentication & Authorization

```python
# app/core/auth.py
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings
from app.models.user import User
from app.services.user_service import UserService

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class AuthService:
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        try:
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

# Dependency for getting current user
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user_service: UserService = Depends()
) -> User:
    token = credentials.credentials
    payload = AuthService.verify_token(token)
    
    user_id: int = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    user = await user_service.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user

# Role-based access control
def require_role(required_role: str):
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != required_role and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker

# Permission-based access control
def require_permission(permission: str):
    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        user_permissions = {
            "admin": ["view_alerts", "take_actions", "manage_users", "view_stats", "system_config"],
            "security": ["view_alerts", "take_actions", "view_stats"],
            "viewer": ["view_alerts", "view_stats"],
        }
        
        if permission not in user_permissions.get(current_user.role, []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    return permission_checker
```

### API Endpoints Implementation

```python
# app/api/v1/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.auth import AuthService
from app.schemas.auth import Token, LoginRequest, RefreshTokenRequest
from app.schemas.user import UserResponse
from app.services.user_service import UserService
from app.database import get_db

router = APIRouter()

@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    db: Session = Depends(get_db),
    user_service: UserService = Depends()
):
    user = await user_service.authenticate_user(login_data.email, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    access_token = AuthService.create_access_token(data={"sub": str(user.id)})
    refresh_token = AuthService.create_refresh_token(data={"sub": str(user.id)})
    
    # Update last login
    await user_service.update_last_login(user.id)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": UserResponse.from_orm(user)
    }

@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    user_service: UserService = Depends()
):
    payload = AuthService.verify_token(refresh_data.refresh_token)
    
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )
    
    user_id = payload.get("sub")
    user = await user_service.get_user_by_id(int(user_id))
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    new_access_token = AuthService.create_access_token(data={"sub": str(user.id)})
    
    return {
        "access_token": new_access_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    # In a more sophisticated implementation, you might want to blacklist the token
    return {"message": "Successfully logged out"}

# app/api/v1/alerts.py
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Optional
from sqlalchemy.orm import Session

from app.schemas.alert import AlertResponse, AlertCreate, AlertUpdate, AlertFilters
from app.services.alert_service import AlertService
from app.core.auth import get_current_user, require_permission
from app.models.user import User

router = APIRouter()

@router.get("/", response_model=List[AlertResponse])
async def get_alerts(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    severity: Optional[str] = Query(None),
    attack_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    time_range: Optional[str] = Query("24h"),
    current_user: User = Depends(require_permission("view_alerts")),
    alert_service: AlertService = Depends()
):
    filters = AlertFilters(
        severity=severity,
        attack_type=attack_type,
        status=status,
        time_range=time_range
    )
    
    alerts = await alert_service.get_alerts(skip=skip, limit=limit, filters=filters)
    return alerts

@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int,
    current_user: User = Depends(require_permission("view_alerts")),
    alert_service: AlertService = Depends()
):
    alert = await alert_service.get_alert_by_id(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert

@router.post("/{alert_id}/resolve")
async def resolve_alert(
    alert_id: int,
    resolution_data: dict,
    current_user: User = Depends(require_permission("take_actions")),
    alert_service: AlertService = Depends()
):
    result = await alert_service.resolve_alert(
        alert_id=alert_id,
        resolved_by=current_user.id,
        resolution=resolution_data.get("resolution"),
        notes=resolution_data.get("notes")
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return {"message": "Alert resolved successfully"}

# app/api/v1/network.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List

from app.schemas.network import BlockIPRequest, BlockPortRequest, NetworkActionResponse
from app.services.network_service import NetworkService
from app.core.auth import require_permission
from app.models.user import User

router = APIRouter()

@router.post("/block-ip", response_model=NetworkActionResponse)
async def block_ip(
    block_request: BlockIPRequest,
    current_user: User = Depends(require_permission("take_actions")),
    network_service: NetworkService = Depends()
):
    try:
        result = await network_service.block_ip(
            ip=block_request.ip,
            duration=block_request.duration,
            reason=block_request.reason,
            user_id=current_user.id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/block-port", response_model=NetworkActionResponse)
async def block_port(
    block_request: BlockPortRequest,
    current_user: User = Depends(require_permission("take_actions")),
    network_service: NetworkService = Depends()
):
    try:
        result = await network_service.block_port(
            port=block_request.port,
            protocol=block_request.protocol,
            duration=block_request.duration,
            reason=block_request.reason,
            user_id=current_user.id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/unblock/{action_id}")
async def unblock(
    action_id: int,
    current_user: User = Depends(require_permission("take_actions")),
    network_service: NetworkService = Depends()
):
    try:
        result = await network_service.unblock(action_id, current_user.id)
        return {"message": "Successfully removed block", "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/actions", response_model=List[NetworkActionResponse])
async def get_network_actions(
    active_only: bool = True,
    current_user: User = Depends(require_permission("view_alerts")),
    network_service: NetworkService = Depends()
):
    actions = await network_service.get_actions(active_only=active_only)
    return actions
```

### Real-time Communication (Socket.IO)

```python
# app/services/notification_service.py
import socketio
from typing import Dict, Any, List
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class NotificationService:
    def __init__(self, sio: socketio.AsyncServer):
        self.sio = sio
        self.connected_users: Dict[str, Dict[str, Any]] = {}
    
    async def user_connected(self, sid: str, user_id: int, user_role: str):
        """Register a user connection"""
        self.connected_users[sid] = {
            "user_id": user_id,
            "user_role": user_role,
            "connected_at": datetime.utcnow()
        }
        logger.info(f"User {user_id} connected with session {sid}")
    
    async def user_disconnected(self, sid: str):
        """Handle user disconnection"""
        if sid in self.connected_users:
            user_data = self.connected_users.pop(sid)
            logger.info(f"User {user_data['user_id']} disconnected")
    
    async def broadcast_attack_detected(self, alert_data: Dict[str, Any]):
        """Broadcast new attack detection to all connected users"""
        await self.sio.emit("attack_detected", alert_data)
        logger.info(f"Broadcasted attack detection: {alert_data['attack_type']}")
    
    async def broadcast_alert_resolved(self, alert_data: Dict[str, Any]):
        """Broadcast alert resolution to all connected users"""
        await self.sio.emit("alert_resolved", alert_data)
        logger.info(f"Broadcasted alert resolution: {alert_data['alert_id']}")
    
    async def broadcast_ip_blocked(self, action_data: Dict[str, Any]):
        """Broadcast IP blocking action to all connected users"""
        await self.sio.emit("ip_blocked", action_data)
        logger.info(f"Broadcasted IP block: {action_data['ip_address']}")
    
    async def broadcast_system_status(self, status_data: Dict[str, Any]):
        """Broadcast system status updates"""
        await self.sio.emit("system_status", status_data)
    
    async def send_to_user_role(self, role: str, event: str, data: Dict[str, Any]):
        """Send message to all users with specific role"""
        target_sessions = [
            sid for sid, user_data in self.connected_users.items()
            if user_data["user_role"] == role or user_data["user_role"] == "admin"
        ]
        
        for sid in target_sessions:
            await self.sio.emit(event, data, room=sid)

# Socket.IO Event Handlers
@sio.event
async def connect(sid, environ, auth):
    """Handle client connection"""
    try:
        # Verify JWT token from auth
        token = auth.get("token") if auth else None
        if not token:
            await sio.disconnect(sid)
            return False
        
        # Verify and decode token
        payload = AuthService.verify_token(token)
        user_id = payload.get("sub")
        
        # Get user details
        user_service = UserService()
        user = await user_service.get_user_by_id(int(user_id))
        
        if not user:
            await sio.disconnect(sid)
            return False
        
        # Register connection
        notification_service = NotificationService(sio)
        await notification_service.user_connected(sid, user.id, user.role)
        
        logger.info(f"User {user.email} connected via WebSocket")
        return True
        
    except Exception as e:
        logger.error(f"Socket connection error: {e}")
        await sio.disconnect(sid)
        return False

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    notification_service = NotificationService(sio)
    await notification_service.user_disconnected(sid)
```

### AI Model Integration

```python
# app/services/ai_service.py
import httpx
import asyncio
from typing import Dict, Any, Optional
from app.core.config import settings
from app.utils.logger import get_logger
from app.services.alert_service import AlertService
from app.services.notification_service import NotificationService

logger = get_logger(__name__)

class AIService:
    def __init__(self):
        self.ai_service_url = settings.AI_SERVICE_URL
        self.timeout = settings.AI_SERVICE_TIMEOUT
        self.client = httpx.AsyncClient(timeout=self.timeout)
    
    async def process_inference_result(self, inference_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process AI inference result and create alert if attack detected"""
        try:
            prediction = inference_data.get("prediction")
            
            if prediction == "attack":
                # Create alert in database
                alert_service = AlertService()
                alert = await alert_service.create_alert_from_inference(inference_data)
                
                # Broadcast real-time notification
                notification_service = NotificationService()
                await notification_service.broadcast_attack_detected({
                    "alert_id": alert.id,
                    "attack_type": alert.attack_type,
                    "risk_level": alert.risk_level,
                    "src_ip": alert.src_ip,
                    "dst_ip": alert.dst_ip,
                    "confidence": alert.prediction_confidence,
                    "timestamp": alert.created_at.isoformat()
                })
                
                return {"status": "alert_created", "alert_id": alert.id}
            else:
                # Log normal traffic (optional, for statistics)
                await self._log_normal_traffic(inference_data)
                return {"status": "normal_traffic"}
                
        except Exception as e:
            logger.error(f"Error processing inference result: {e}")
            return None
    
    async def _log_normal_traffic(self, flow_data: Dict[str, Any]):
        """Log normal traffic for statistics (optional)"""
        # Implementation for logging normal traffic statistics
        pass
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get AI model health status"""
        try:
            response = await self.client.get(f"{self.ai_service_url}/health")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get AI model status: {e}")
            return {"status": "error", "message": str(e)}
    
    async def retrain_model(self, training_data_path: str) -> Dict[str, Any]:
        """Trigger model retraining (future enhancement)"""
        try:
            response = await self.client.post(
                f"{self.ai_service_url}/retrain",
                json={"data_path": training_data_path}
            )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to trigger model retraining: {e}")
            return {"status": "error", "message": str(e)}

# AI Inference Endpoint
# app/api/v1/inference.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.schemas.ai import InferenceRequest, InferenceResponse
from app.services.ai_service import AIService

router = APIRouter()

@router.post("/process", response_model=InferenceResponse)
async def process_inference(
    inference_data: InferenceRequest,
    background_tasks: BackgroundTasks,
    ai_service: AIService = Depends()
):
    """Receive inference result from AI model"""
    try:
        # Process in background to avoid blocking AI service
        background_tasks.add_task(
            ai_service.process_inference_result,
            inference_data.dict()
        )
        
        return {"status": "received", "message": "Inference data received and processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-status")
async def get_model_status(ai_service: AIService = Depends()):
    """Get AI model health status"""
    status = await ai_service.get_model_status()
    return status
```

### Database Models

```python
# app/models/base.py
from sqlalchemy import Column, Integer, DateTime, func
from sqlalchemy.ext.declarative import declarative_base, declared_attr

Base = declarative_base()

class BaseModel(Base):
    __abstract__ = True
    
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# app/models/user.py
from sqlalchemy import Column, String, Boolean, DateTime, Enum
from sqlalchemy.orm import relationship
import enum

from app.models.base import BaseModel

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    SECURITY = "security"
    VIEWER = "viewer"

class User(BaseModel):
    __tablename__ = "users"
    
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    alerts_resolved = relationship("Alert", back_populates="resolved_by_user")
    actions = relationship("Action", back_populates="user")

# app/models/alert.py
from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
import enum

from app.models.base import BaseModel

class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(str, enum.Enum):
    ACTIVE = "active"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"

class Alert(BaseModel):
    __tablename__ = "alerts"
    
    flow_id = Column(String(100), ForeignKey("flows.flow_id"), nullable=False)
    attack_type_id = Column(Integer, ForeignKey("attack_types.id"), nullable=False)
    prediction_confidence = Column(Float, nullable=False)
    risk_level = Column(Enum(RiskLevel), nullable=False)
    src_ip = Column(String(45), nullable=False, index=True)
    dst_ip = Column(String(45), nullable=False, index=True)
    src_port = Column(Integer, nullable=True)
    dst_port = Column(Integer, nullable=True)
    protocol = Column(String(10), nullable=True)
    attack_details = Column(JSON, nullable=True)
    status = Column(Enum(AlertStatus), default=AlertStatus.ACTIVE, nullable=False)
    resolved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolution_notes = Column(String(1000), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    flow = relationship("Flow", back_populates="alerts")
    attack_type = relationship("AttackType", back_populates="alerts")
    resolved_by_user = relationship("User", back_populates="alerts_resolved")
    actions = relationship("Action", back_populates="alert")

# app/models/flow.py
from sqlalchemy import Column, String, Integer, BigInteger, JSON, DateTime, Index
from sqlalchemy.orm import relationship

from app.models.base import BaseModel

class Flow(BaseModel):
    __tablename__ = "flows"
    
    flow_id = Column(String(100), unique=True, nullable=False, index=True)
    src_ip = Column(String(45), nullable=False, index=True)
    dst_ip = Column(String(45), nullable=False, index=True)
    src_port = Column(Integer, nullable=False)
    dst_port = Column(Integer, nullable=False)
    protocol = Column(String(10), nullable=False)
    flow_duration = Column(BigInteger, nullable=True)
    total_fwd_packets = Column(Integer, nullable=True)
    total_bwd_packets = Column(Integer, nullable=True)
    features = Column(JSON, nullable=False)  # Processed 60 features
    raw_features = Column(JSON, nullable=True)  # Original 80 features
    processed_at = Column(DateTime(timezone=True), nullable=False)
    
    # Relationships
    alerts = relationship("Alert", back_populates="flow")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_flows_timestamp', 'created_at'),
        Index('idx_flows_src_dst', 'src_ip', 'dst_ip'),
    )
```

### Business Logic Services

```python
# app/services/alert_service.py
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from datetime import datetime, timedelta

from app.models.alert import Alert, AlertStatus, RiskLevel
from app.models.flow import Flow
from app.schemas.alert import AlertCreate, AlertFilters
from app.database import get_db
from app.utils.logger import get_logger

logger = get_logger(__name__)

class AlertService:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db
    
    async def create_alert_from_inference(self, inference_data: Dict[str, Any]) -> Alert:
        """Create alert from AI inference result"""
        try:
            # Map inference data to alert fields
            alert_data = {
                "flow_id": inference_data.get("flow_id"),
                "attack_type": inference_data.get("attack_type"),
                "prediction_confidence": inference_data.get("confidence"),
                "risk_level": self._map_risk_level(inference_data.get("confidence")),
                "src_ip": inference_data.get("src_ip"),
                "dst_ip": inference_data.get("dst_ip"),
                "src_port": inference_data.get("src_port"),
                "dst_port": inference_data.get("dst_port"),
                "protocol": inference_data.get("protocol"),
                "attack_details": inference_data.get("details", {}),
                "status": AlertStatus.ACTIVE
            }
            
            alert = Alert(**alert_data)
            self.db.add(alert)
            self.db.commit()
            self.db.refresh(alert)
            
            logger.info(f"Created alert {alert.id} for {alert.attack_type} attack")
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            self.db.rollback()
            raise
    
    async def get_alerts(
        self, 
        skip: int = 0, 
        limit: int = 100, 
        filters: Optional[AlertFilters] = None
    ) -> List[Alert]:
        """Get alerts with optional filtering"""
        query = self.db.query(Alert)
        
        if filters:
            if filters.severity and filters.severity != "all":
                query = query.filter(Alert.risk_level == filters.severity)
            
            if filters.attack_type and filters.attack_type != "all":
                query = query.filter(Alert.attack_type == filters.attack_type)
            
            if filters.status and filters.status != "all":
                query = query.filter(Alert.status == filters.status)
            
            if filters.time_range:
                time_delta = self._parse_time_range(filters.time_range)
                start_time = datetime.utcnow() - time_delta
                query = query.filter(Alert.created_at >= start_time)
        
        return query.order_by(Alert.created_at.desc()).offset(skip).limit(limit).all()
    
    async def get_alert_by_id(self, alert_id: int) -> Optional[Alert]:
        """Get alert by ID"""
        return self.db.query(Alert).filter(Alert.id == alert_id).first()
    
    async def resolve_alert(
        self, 
        alert_id: int, 
        resolved_by: int, 
        resolution: str, 
        notes: Optional[str] = None
    ) -> bool:
        """Resolve an alert"""
        try:
            alert = self.db.query(Alert).filter(Alert.id == alert_id).first()
            if not alert:
                return False
            
            alert.status = AlertStatus.RESOLVED
            alert.resolved_by = resolved_by
            alert.resolution_notes = notes
            alert.resolved_at = datetime.utcnow()
            
            self.db.commit()
            logger.info(f"Alert {alert_id} resolved by user {resolved_by}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            self.db.rollback()
            return False
    
    def _map_risk_level(self, confidence: float) -> RiskLevel:
        """Map confidence score to risk level"""
        if confidence >= 0.9:
            return RiskLevel.CRITICAL
        elif confidence >= 0.7:
            return RiskLevel.HIGH
        elif confidence >= 0.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _parse_time_range(self, time_range: str) -> timedelta:
        """Parse time range string to timedelta"""
        time_mappings = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }
        return time_mappings.get(time_range, timedelta(days=1))

# app/services/network_service.py
import subprocess
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from app.models.action import Action, ActionType, ActionStatus
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class NetworkService:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db
        self.iptables_path = settings.IPTABLES_PATH
    
    async def block_ip(
        self, 
        ip: str, 
        duration: int, 
        reason: str, 
        user_id: int
    ) -> Dict[str, Any]:
        """Block IP address using iptables"""
        try:
            if not settings.ENABLE_NETWORK_ACTIONS:
                raise Exception("Network actions are disabled")
            
            # Execute iptables command
            command = [
                self.iptables_path,
                "-I", "INPUT",
                "-s", ip,
                "-j", "DROP"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Create action record
                expires_at = datetime.utcnow() + timedelta(seconds=duration)
                action = Action(
                    alert_id=None,  # Can be set if blocking from specific alert
                    user_id=user_id,
                    action_type=ActionType.BLOCK_IP,
                    target_value=ip,
                    parameters={"duration": duration, "reason": reason},
                    status=ActionStatus.EXECUTED,
                    executed_at=datetime.utcnow(),
                    expires_at=expires_at
                )
                
                self.db.add(action)
                self.db.commit()
                
                # Schedule automatic unblock
                asyncio.create_task(self._schedule_unblock(action.id, duration))
                
                logger.info(f"Successfully blocked IP {ip} for {duration} seconds")
                return {
                    "success": True,
                    "action_id": action.id,
                    "message": f"IP {ip} blocked successfully",
                    "expires_at": expires_at.isoformat()
                }
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"Failed to block IP {ip}: {error_msg}")
                raise Exception(f"iptables command failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Error blocking IP {ip}: {e}")
            raise
    
    async def unblock_ip(self, ip: str, user_id: int) -> Dict[str, Any]:
        """Unblock IP address"""
        try:
            # Execute iptables command to remove rule
            command = [
                self.iptables_path,
                "-D", "INPUT",
                "-s", ip,
                "-j", "DROP"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Update action record
                action = self.db.query(Action).filter(
                    and_(
                        Action.target_value == ip,
                        Action.action_type == ActionType.BLOCK_IP,
                        Action.status == ActionStatus.EXECUTED
                    )
                ).first()
                
                if action:
                    action.status = ActionStatus.CANCELLED
                    self.db.commit()
                
                logger.info(f"Successfully unblocked IP {ip}")
                return {"success": True, "message": f"IP {ip} unblocked successfully"}
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise Exception(f"iptables command failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Error unblocking IP {ip}: {e}")
            raise
    
    async def _schedule_unblock(self, action_id: int, delay_seconds: int):
        """Schedule automatic unblock after specified delay"""
        await asyncio.sleep(delay_seconds)
        
        try:
            action = self.db.query(Action).filter(Action.id == action_id).first()
            if action and action.status == ActionStatus.EXECUTED:
                await self.unblock_ip(action.target_value, action.user_id)
                logger.info(f"Automatically unblocked {action.target_value} after {delay_seconds} seconds")
        except Exception as e:
            logger.error(f"Failed to automatically unblock action {action_id}: {e}")
```

### Performance Optimization

```python
# app/utils/cache.py
import redis
import json
import pickle
from typing import Any, Optional, Union
from app.core.config import settings

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)

class CacheService:
    @staticmethod
    async def get(key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = redis_client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception:
            return None
    
    @staticmethod
    async def set(key: str, value: Any, expire: int = settings.REDIS_CACHE_TTL) -> bool:
        """Set value in cache with expiration"""
        try:
            serialized_value = pickle.dumps(value)
            return redis_client.setex(key, expire, serialized_value)
        except Exception:
            return False
    
    @staticmethod
    async def delete(key: str) -> bool:
        """Delete key from cache"""
        try:
            return redis_client.delete(key) > 0
        except Exception:
            return False
    
    @staticmethod
    async def get_json(key: str) -> Optional[dict]:
        """Get JSON value from cache"""
        try:
            value = redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception:
            return None
    
    @staticmethod
    async def set_json(key: str, value: dict, expire: int = settings.REDIS_CACHE_TTL) -> bool:
        """Set JSON value in cache"""
        try:
            json_value = json.dumps(value)
            return redis_client.setex(key, expire, json_value)
        except Exception:
            return False

# Caching decorator
def cache_result(expire: int = 300, key_prefix: str = ""):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = await CacheService.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await CacheService.set(cache_key, result, expire)
            return result
        return wrapper
    return decorator

# Usage example
@cache_result(expire=600, key_prefix="alerts")
async def get_alert_statistics(time_range: str) -> Dict[str, Any]:
    # Expensive database query
    pass
```

### Background Tasks with Celery

```python
# app/workers/celery_app.py
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "ids_backend",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.workers.alert_processor", "app.workers.network_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "app.workers.alert_processor.*": {"queue": "alerts"},
        "app.workers.network_tasks.*": {"queue": "network"},
    },
)

# app/workers/alert_processor.py
from celery import current_app as celery_app
from app.services.alert_service import AlertService
from app.services.notification_service import NotificationService

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3})
def process_bulk_alerts(self, alert_data_list):
    """Process multiple alerts in background"""
    try:
        alert_service = AlertService()
        results = []
        
        for alert_data in alert_data_list:
            result = alert_service.create_alert_from_inference(alert_data)
            results.append(result.id)
        
        return {"processed": len(results), "alert_ids": results}
    except Exception as e:
        logger.error(f"Failed to process bulk alerts: {e}")
        raise

@celery_app.task
def cleanup_expired_actions():
    """Clean up expired network actions"""
    try:
        network_service = NetworkService()
        expired_actions = network_service.get_expired_actions()
        
        for action in expired_actions:
            if action.action_type == "block_ip":
                network_service.unblock_ip(action.target_value, action.user_id)
        
        return {"cleaned_up": len(expired_actions)}
    except Exception as e:
        logger.error(f"Failed to cleanup expired actions: {e}")
        raise
```

This comprehensive backend documentation provides all the necessary details for implementing the server-side application of the IDS-AI system, including API design, authentication, real-time communication, AI integration, database management, and performance optimization strategies.