# IDS Integrated with AI Model (Semi-IPS) - Documentation

## 1. Project Overview

The **IDS integrated with AI model (Semi-IPS)** is an advanced network security system that combines artificial intelligence with intrusion detection capabilities to provide real-time network threat monitoring and response. The system is designed for local area network (LAN) deployment, capturing and analyzing network traffic flows at the router layer through SPAN/mirror ports to ensure maximum security and privacy.

### Problem Statement

Modern networks face increasingly sophisticated cyber attacks that traditional signature-based intrusion detection systems cannot effectively identify. Organizations require intelligent, real-time monitoring solutions that can detect both known and unknown attack patterns while providing immediate response capabilities.

### Key Features

- **AI-Powered Detection**: Machine learning model analyzes network flow features to identify attack patterns
- **Real-time Socket Alerts**: Instant WebSocket/TCP notifications for immediate threat response
- **Local-only Deployment**: Operates entirely within the LAN for enhanced security
- **Interactive Dashboard**: Web-based interface for visualization, analytics, and network control
- **Role-based Access Control**: Three-tier permission system (Admin/Security/Viewer)
- **Automated Response**: Semi-IPS functionality for network action automation

### System Architecture

The system follows a layered architecture: **Network Capture → AI Analysis → Backend Processing → Database Storage → Frontend Visualization → User Actions**. This flow ensures comprehensive threat detection from packet capture to user response, with real-time socket communication enabling instant alerts across all system components.

## 2. Business Model & Use Case

### Target Organizations

- **Enterprise Networks**: Companies requiring comprehensive internal network monitoring
- **Critical Infrastructure**: Organizations with high-security requirements
- **Educational Institutions**: Universities and schools protecting campus networks
- **Healthcare Systems**: Medical facilities safeguarding patient data networks
- **Financial Services**: Banks and financial institutions requiring real-time threat detection

### User Personas

#### Admin

- **Responsibilities**: Full system control, user management, system configuration
- **Use Cases**: Configure security policies, manage user accounts, oversee system health
- **Value**: Complete visibility and control over network security infrastructure

#### Security Officer

- **Responsibilities**: Monitor threats, investigate attacks, take defensive actions
- **Use Cases**: Respond to alerts, block malicious IPs, analyze attack patterns
- **Value**: Real-time threat response capabilities with detailed attack intelligence

#### Viewer

- **Responsibilities**: Monitor network status and view security reports
- **Use Cases**: Track network health, view attack statistics, generate reports
- **Value**: Network visibility without operational responsibilities

### Business Benefits

- **Reduced Response Time**: Real-time alerts enable immediate threat response
- **Enhanced Security Posture**: AI-driven detection identifies previously unknown threats
- **Operational Efficiency**: Automated detection reduces manual monitoring overhead
- **Compliance Support**: Comprehensive logging and reporting for regulatory requirements
- **Cost Effectiveness**: Local deployment eliminates cloud dependencies and reduces operational costs

### Future Enhancements

- **Full IPS Integration**: Automated blocking and network response capabilities
- **Cloud Dashboard Option**: Remote monitoring capabilities for distributed organizations
- **AI Model Retraining**: Continuous learning from new threat patterns
- **Advanced Analytics**: Predictive threat modeling and trend analysis

## 3. System Design

### Logical Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Network       │───▶│   AI Pipeline    │───▶│   Backend API   │
│   Capture       │    │   (80→60 features)│    │   (Flask/FastAPI)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │◀───│   Database       │◀───│   Socket        │
│   Dashboard     │    │   (PostgreSQL)   │    │   Communication │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Network Layer Implementation

The system operates at the network infrastructure level, connecting to router SPAN (Switched Port Analyzer) or mirror ports to capture network traffic. This approach ensures:

- **Passive Monitoring**: No impact on network performance
- **Complete Visibility**: Access to all network traffic flows
- **Security Isolation**: Monitoring system operates independently of production network
- **Scalability**: Can monitor multiple network segments simultaneously

### Data Flow Pipeline

1. **Traffic Capture**: Raw packets captured from SPAN/mirror ports
2. **Flow Extraction**: Packets processed into network flow records
3. **Feature Engineering**: ~80 statistical and behavioral features extracted per flow
4. **Preprocessing**: Feature scaling, normalization, and reduction to ~60 optimal features
5. **AI Detection**: Machine learning model classifies flows as normal or attack
6. **Real-time Alerting**: Detected attacks immediately sent via socket to backend
7. **Database Storage**: Flow data, attack details, and metadata stored in PostgreSQL
8. **Dashboard Updates**: Frontend receives real-time updates via WebSocket connections
9. **User Response**: Security personnel can take immediate action through the interface

### Socket Integration Architecture

The system implements a dual-socket architecture:

- **AI-to-Backend Socket**: High-throughput connection for streaming detection results
- **Backend-to-Frontend Socket**: Real-time dashboard updates and alert notifications

## 4. AI Model Pipeline

### Input Features (80 Initial Features)

The AI model processes comprehensive network flow characteristics:

```
Flow Duration Features:
- flow_duration, flow_bytes_s, flow_packets_s, flow_iat_mean, flow_iat_std
- flow_iat_max, flow_iat_min, fwd_iat_mean, fwd_iat_std, fwd_iat_max
- fwd_iat_min, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min

Packet Size Features:
- fwd_pkt_len_max, fwd_pkt_len_min, fwd_pkt_len_mean, fwd_pkt_len_std
- bwd_pkt_len_max, bwd_pkt_len_min, bwd_pkt_len_mean, bwd_pkt_len_std
- pkt_len_max, pkt_len_min, pkt_len_mean, pkt_len_std, pkt_len_var

Flow Statistics:
- tot_fwd_pkts, tot_bwd_pkts, totlen_fwd_pkts, totlen_bwd_pkts
- fwd_pkt_len_mean, bwd_pkt_len_mean, flow_pkts_s, flow_bytes_s

Behavioral Features:
- down_up_ratio, pkt_size_avg, fwd_seg_size_avg, bwd_seg_size_avg
- fwd_byts_b_avg, fwd_pkts_b_avg, fwd_blk_rate_avg, bwd_byts_b_avg
- bwd_pkts_b_avg, bwd_blk_rate_avg

TCP Flag Features:
- fwd_psh_flags, bwd_psh_flags, fwd_urg_flags, bwd_urg_flags
- fwd_header_len, bwd_header_len, fwd_pkts_s, bwd_pkts_s

Advanced Statistical Features:
- min_seg_size_forward, act_data_pkt_fwd, min_pkt_len, max_pkt_len
- ece_flag_cnt, rst_flag_cnt, psh_flag_cnt, ack_flag_cnt
- urg_flag_cnt, cwe_flag_count, syn_flag_cnt, fin_flag_cnt
```

### Preprocessing Pipeline

1. **Data Validation**: Remove invalid or incomplete flow records
2. **Feature Scaling**: StandardScaler normalization for numerical stability
3. **Feature Selection**: Correlation analysis and recursive feature elimination reduces 80→60 features
4. **Outlier Detection**: Statistical methods identify and handle anomalous values
5. **Data Balancing**: Ensure representative samples for both attack and normal traffic

### Model Architecture

- **Algorithm**: Ensemble method combining Random Forest and Gradient Boosting
- **Input Layer**: 60 preprocessed features
- **Training Data**: Labeled dataset with various attack types (DDoS, Port Scan, Brute Force, etc.)
- **Validation**: K-fold cross-validation with stratified sampling
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

### Output Classification

```python
{
    "prediction": "attack",  # "normal" or "attack"
    "attack_type": "DDoS",   # specific attack classification
    "confidence": 0.95,      # model confidence score
    "risk_level": "high",    # "low", "medium", "high", "critical"
    "flow_id": "flow_12345", # unique flow identifier
    "timestamp": "2025-10-09T14:30:45Z"
}
```

### Real-time Integration

Detected attacks are immediately transmitted via socket connection to the backend, ensuring zero-delay alerting and enabling rapid response capabilities.

## 5. Backend Design

### Core Responsibilities

- **AI Integration**: Receive and process real-time attack detection results
- **Data Management**: Store flow data, attacks, user actions, and system logs
- **API Services**: Provide RESTful endpoints for frontend and external integrations
- **Authentication**: JWT-based user authentication and role-based authorization
- **Socket Management**: Real-time communication with AI pipeline and frontend
- **Action Execution**: Network control operations (IP blocking, port management)

### Framework Selection

**FastAPI** (Recommended) for high-performance async operations and automatic API documentation, with **Flask** as alternative for simpler deployment scenarios.

### API Endpoints

| Endpoint                   | Method | Description                    | Request Body                                            | Response Example                                          |
| -------------------------- | ------ | ------------------------------ | ------------------------------------------------------- | --------------------------------------------------------- |
| `/api/auth/login`          | POST   | User authentication            | `{"email": "user@domain.com", "password": "secure123"}` | `{"access_token": "jwt_token", "user_role": "admin"}`     |
| `/api/auth/logout`         | POST   | User logout                    | `{"token": "jwt_token"}`                                | `{"message": "Logged out successfully"}`                  |
| `/api/inference`           | POST   | Receive AI detection results   | `{"flow_data": {...}, "prediction": "attack"}`          | `{"status": "processed", "alert_id": 12345}`              |
| `/api/alerts`              | GET    | Retrieve recent alerts         | `?limit=50&severity=high`                               | `[{"id": 1, "type": "DDoS", "timestamp": "..."}]`         |
| `/api/alerts/{id}`         | GET    | Get specific alert details     | -                                                       | `{"id": 1, "details": {...}, "flow_data": {...}}`         |
| `/api/alerts/{id}/resolve` | POST   | Mark alert as resolved         | `{"resolution": "blocked_ip", "notes": "..."}`          | `{"status": "resolved", "resolved_by": "admin"}`          |
| `/api/network/block-ip`    | POST   | Block malicious IP address     | `{"ip": "192.168.1.100", "duration": 3600}`             | `{"success": true, "blocked_until": "..."}`               |
| `/api/network/block-port`  | POST   | Block specific port            | `{"port": 8080, "protocol": "tcp"}`                     | `{"success": true, "rule_id": "rule_001"}`                |
| `/api/network/unblock`     | DELETE | Remove network blocks          | `{"target": "192.168.1.100", "type": "ip"}`             | `{"success": true, "removed_rules": [...]}`               |
| `/api/users`               | GET    | List system users (Admin only) | -                                                       | `[{"id": 1, "username": "admin", "role": "admin"}]`       |
| `/api/users`               | POST   | Create new user (Admin only)   | `{"username": "newuser", "role": "viewer"}`             | `{"id": 3, "username": "newuser", "created": true}`       |
| `/api/users/{id}`          | PUT    | Update user details            | `{"role": "security", "active": true}`                  | `{"updated": true, "user": {...}}`                        |
| `/api/stats/dashboard`     | GET    | Dashboard statistics           | `{"timerange": "24h", "filters": {...}}`                | `{"total_flows": 10000, "attacks": 23, "charts": [...]}`  |
| `/api/stats/attacks`       | GET    | Attack statistics              | `{"period": "week", "group_by": "type"}`                | `{"ddos": 12, "port_scan": 8, "trends": [...]}`           |
| `/api/flows`               | GET    | Network flow data              | `{"limit": 100, "filter": "suspicious"}`                | `[{"flow_id": "f1", "src_ip": "...", "features": {...}}]` |

### Database Integration

- **Connection Pool**: PostgreSQL connection pooling for optimal performance
- **ORM**: SQLAlchemy for database operations and model definitions
- **Migrations**: Alembic for database schema version control
- **Backup Strategy**: Automated daily backups with point-in-time recovery

### Security Implementation

- **JWT Authentication**: Secure token-based authentication with refresh tokens
- **Rate Limiting**: API endpoint protection against abuse
- **Input Validation**: Pydantic models for request/response validation
- **CORS Configuration**: Controlled cross-origin resource sharing
- **Logging**: Comprehensive audit trails for all user actions

## 6. Frontend Design

### Technology Stack

**React 18** with TypeScript for type safety, **Tailwind CSS** for styling, **Chart.js** for data visualization, and **Socket.io-client** for real-time communication.

### Application Pages

#### Login Page

- **Purpose**: User authentication and role-based access control
- **Features**: Email/password login, "Remember Me" option, password reset
- **API Integration**: `POST /api/auth/login`
- **Security**: Input validation, secure token storage, brute force protection

#### Dashboard/Overview

- **Purpose**: High-level network security status and key metrics
- **Components**:
  - Real-time attack counter and severity distribution
  - Network traffic volume charts (last 24h/7d/30d)
  - Top attack types and source IPs
  - System health indicators
- **API Integration**: `GET /api/stats/dashboard`
- **Socket Events**: Live updates for attack counts and system status

#### Real-time Alerts System

- **Implementation**: Persistent notification panel with sound alerts
- **Features**: Alert severity colors, auto-dismiss options, action buttons
- **Socket Integration**:

```javascript
const socket = io("ws://localhost:8000");
socket.on("attack_detected", (data) => {
  showAlertPopup({
    type: data.attack_type,
    severity: data.risk_level,
    source_ip: data.src_ip,
    timestamp: data.timestamp,
  });
  playAlertSound(data.risk_level);
});
```

#### Attacks Page (Current)

- **Purpose**: Active attack monitoring and immediate response
- **Features**:
  - Real-time attack table with filtering (type, severity, source)
  - Sortable columns (timestamp, risk level, confidence)
  - Quick action buttons (block IP, investigate, resolve)
  - Bulk operations for multiple attacks
- **API Integration**: `GET /api/alerts`, `POST /api/network/block-ip`
- **Auto-refresh**: 5-second intervals with socket updates

#### Attack History

- **Purpose**: Historical attack analysis and trend identification
- **Features**:
  - Date range filtering and export functionality
  - Attack pattern visualization and statistics
  - Detailed flow information and forensic data
  - Resolution tracking and notes
- **API Integration**: `GET /api/alerts?historical=true`
- **Performance**: Pagination and lazy loading for large datasets

#### Network Actions

- **Purpose**: Proactive network security management
- **Features**:
  - IP blocking/unblocking interface
  - Port management controls
  - Active rules display and management
  - Scheduled actions and automation rules
- **API Integration**: `POST /api/network/block-ip`, `POST /api/network/block-port`
- **Permissions**: Security and Admin roles only

#### User Management (Admin Only)

- **Purpose**: System user administration
- **Features**:
  - User creation, modification, and deactivation
  - Role assignment and permission management
  - Activity logging and audit trails
  - Password policy enforcement
- **API Integration**: `GET /api/users`, `POST /api/users`, `PUT /api/users/{id}`

### Socket Communication Example

```javascript
import io from "socket.io-client";

class AlertManager {
  constructor() {
    this.socket = io("ws://localhost:8000", {
      transports: ["websocket"],
      upgrade: false,
    });

    this.setupEventHandlers();
  }

  setupEventHandlers() {
    this.socket.on("attack_detected", this.handleNewAttack);
    this.socket.on("alert_resolved", this.handleAlertResolution);
    this.socket.on("ip_blocked", this.handleIPBlock);
    this.socket.on("system_status", this.updateSystemStatus);
  }

  handleNewAttack = (data) => {
    this.showNotification({
      title: `${data.attack_type} Attack Detected`,
      message: `Source: ${data.src_ip}`,
      severity: data.risk_level,
      actions: ["Block IP", "Investigate", "Dismiss"],
    });
  };
}
```

## 7. Database Design

### Core Tables

```sql
-- User Management
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'viewer',
    active BOOLEAN DEFAULT true,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE roles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    permissions JSON NOT NULL
);

-- Attack Classification
CREATE TABLE attack_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    category VARCHAR(50) NOT NULL,
    severity_level INTEGER NOT NULL, -- 1=low, 2=medium, 3=high, 4=critical
    description TEXT,
    mitigation_steps TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Network Flow Data
CREATE TABLE flows (
    id SERIAL PRIMARY KEY,
    flow_id VARCHAR(100) UNIQUE NOT NULL,
    src_ip INET NOT NULL,
    dst_ip INET NOT NULL,
    src_port INTEGER NOT NULL,
    dst_port INTEGER NOT NULL,
    protocol VARCHAR(10) NOT NULL,
    flow_duration BIGINT,
    total_fwd_packets INTEGER,
    total_bwd_packets INTEGER,
    features JSON NOT NULL, -- All 60 processed features
    raw_features JSON, -- Original 80 features for analysis
    timestamp TIMESTAMP DEFAULT NOW(),
    processed_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_flows_timestamp ON flows(timestamp);
CREATE INDEX idx_flows_src_ip ON flows(src_ip);
CREATE INDEX idx_flows_dst_ip ON flows(dst_ip);

-- Attack Alerts
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    flow_id VARCHAR(100) REFERENCES flows(flow_id),
    attack_type_id INTEGER REFERENCES attack_types(id),
    prediction_confidence DECIMAL(5,4) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    src_ip INET NOT NULL,
    dst_ip INET NOT NULL,
    src_port INTEGER,
    dst_port INTEGER,
    protocol VARCHAR(10),
    attack_details JSON,
    status VARCHAR(20) DEFAULT 'active', -- active, investigating, resolved, false_positive
    resolved_by INTEGER REFERENCES users(id),
    resolution_notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP
);

CREATE INDEX idx_alerts_timestamp ON alerts(created_at);
CREATE INDEX idx_alerts_status ON alerts(status);
CREATE INDEX idx_alerts_risk ON alerts(risk_level);

-- Security Actions
CREATE TABLE actions (
    id SERIAL PRIMARY KEY,
    alert_id INTEGER REFERENCES alerts(id),
    user_id INTEGER REFERENCES users(id) NOT NULL,
    action_type VARCHAR(50) NOT NULL, -- block_ip, block_port, unblock_ip, investigate
    target_value VARCHAR(255) NOT NULL, -- IP address, port number, etc.
    parameters JSON, -- Additional action parameters
    status VARCHAR(20) DEFAULT 'pending', -- pending, executed, failed, cancelled
    executed_at TIMESTAMP,
    expires_at TIMESTAMP, -- For temporary blocks
    created_at TIMESTAMP DEFAULT NOW(),
    notes TEXT
);

-- System Logs
CREATE TABLE system_logs (
    id SERIAL PRIMARY KEY,
    log_level VARCHAR(20) NOT NULL, -- debug, info, warning, error, critical
    component VARCHAR(50) NOT NULL, -- ai_model, backend, frontend, database
    event_type VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    metadata JSON,
    user_id INTEGER REFERENCES users(id),
    ip_address INET,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_logs_timestamp ON system_logs(timestamp);
CREATE INDEX idx_logs_level ON system_logs(log_level);
```

### Data Relationships

- **Users** have **Roles** with specific permissions
- **Flows** generate **Alerts** when attacks are detected
- **Alerts** trigger **Actions** performed by users
- **All activities** are logged in **System Logs** for audit trails

### Performance Optimizations

- **Partitioning**: Time-based partitioning for flows and alerts tables
- **Indexing**: Strategic indexes on frequently queried columns
- **Archiving**: Automated archiving of old data to maintain performance
- **Connection Pooling**: Optimized database connection management

## 8. Socket Communication (Realtime)

### Event Types

#### attack_detected

Emitted when the AI model identifies a new attack

```json
{
  "event": "attack_detected",
  "data": {
    "alert_id": 12345,
    "flow_id": "flow_67890",
    "attack_type": "DDoS",
    "risk_level": "high",
    "confidence": 0.95,
    "src_ip": "192.168.1.100",
    "dst_ip": "10.0.0.50",
    "src_port": 54321,
    "dst_port": 80,
    "protocol": "TCP",
    "timestamp": "2025-10-09T14:30:45.123Z",
    "features": {
      "flow_duration": 1500,
      "total_packets": 100,
      "bytes_per_second": 50000
    }
  }
}
```

#### alert_resolved

Emitted when a security analyst resolves an alert

```json
{
  "event": "alert_resolved",
  "data": {
    "alert_id": 12345,
    "resolved_by": "john.doe",
    "resolution": "false_positive",
    "notes": "Verified as legitimate traffic from partner organization",
    "resolved_at": "2025-10-09T14:45:30.456Z"
  }
}
```

#### ip_blocked

Emitted when an IP address is blocked through network actions

```json
{
  "event": "ip_blocked",
  "data": {
    "action_id": 789,
    "ip_address": "192.168.1.100",
    "blocked_by": "security.officer",
    "reason": "DDoS attack source",
    "duration": 3600,
    "expires_at": "2025-10-09T15:30:45.789Z",
    "timestamp": "2025-10-09T14:30:45.789Z"
  }
}
```

#### system_status

Periodic system health and statistics updates

```json
{
  "event": "system_status",
  "data": {
    "ai_model_status": "active",
    "flows_per_second": 150,
    "active_alerts": 12,
    "blocked_ips": 5,
    "system_load": 0.65,
    "memory_usage": 0.45,
    "last_update": "2025-10-09T14:30:00.000Z"
  }
}
```

### Socket Implementation Benefits

- **Zero Latency Alerts**: Immediate notification of security events
- **Real-time Dashboard Updates**: Live statistics without page refresh
- **Collaborative Response**: Multiple users see actions in real-time
- **System Monitoring**: Continuous health status updates
- **Reduced Server Load**: Eliminates constant polling from frontend

### Connection Management

- **Automatic Reconnection**: Client handles connection drops gracefully
- **Authentication**: Socket connections require valid JWT tokens
- **Room-based Broadcasting**: Role-based event filtering
- **Rate Limiting**: Protection against socket abuse

## 9. Roles & Permissions

| Role     | Can View Attacks | Can Take Actions | Manage Users | View Stats | System Config | Export Data |
| -------- | ---------------- | ---------------- | ------------ | ---------- | ------------- | ----------- |
| Admin    | ✅               | ✅               | ✅           | ✅         | ✅            | ✅          |
| Security | ✅               | ✅               | ❌           | ✅         | ❌            | ✅          |
| Viewer   | ✅               | ❌               | ❌           | ✅         | ❌            | ❌          |

### Detailed Permissions

#### Admin Role

- **Full System Access**: Complete control over all system functions
- **User Management**: Create, modify, and deactivate user accounts
- **System Configuration**: Modify AI model parameters, network settings
- **Advanced Analytics**: Access to all historical data and reports
- **Security Actions**: All blocking/unblocking capabilities
- **Audit Access**: View all system logs and user activities

#### Security Role

- **Threat Response**: Primary role for handling security incidents
- **Investigation Tools**: Access to detailed attack analysis and flow data
- **Network Actions**: IP/port blocking capabilities with approval workflows
- **Alert Management**: Resolve, escalate, and annotate security alerts
- **Reporting**: Generate security reports and trend analysis
- **Limited Administration**: Cannot modify users or system settings

#### Viewer Role

- **Read-Only Access**: Monitor security status without modification rights
- **Dashboard Viewing**: Access to real-time and historical dashboards
- **Alert Monitoring**: View active and resolved alerts
- **Basic Reporting**: Generate standard security reports
- **No Actions**: Cannot block IPs, modify settings, or resolve alerts

### Permission Enforcement

- **Frontend**: UI elements hidden/disabled based on user role
- **Backend**: API endpoints validate user permissions before execution
- **Database**: Row-level security policies enforce data access controls
- **Audit Trail**: All permission checks logged for compliance

## 10. Tech Stack

### Frontend Technologies

- **React 18**: Modern component-based UI framework with hooks
- **TypeScript**: Type-safe JavaScript for enhanced code quality
- **Tailwind CSS**: Utility-first CSS framework for rapid styling
- **Chart.js**: Interactive charts and data visualization
- **Socket.io-client**: Real-time WebSocket communication
- **React Router**: Client-side routing and navigation
- **Axios**: HTTP client for API communication
- **React Query**: Server state management and caching

### Backend Technologies

- **FastAPI**: High-performance async Python web framework
- **Socket.io**: Real-time bidirectional event-based communication
- **SQLAlchemy**: Python SQL toolkit and Object-Relational Mapping
- **Alembic**: Database migration tool for SQLAlchemy
- **JWT**: JSON Web Tokens for secure authentication
- **Pydantic**: Data validation using Python type annotations
- **Uvicorn**: ASGI server for FastAPI applications

### Database

- **PostgreSQL 14+**: Advanced open-source relational database
- **pgAdmin**: Database administration and management tool
- **Connection Pooling**: PgBouncer for optimized database connections

### AI/ML Stack

- **Python 3.9+**: Primary language for AI pipeline
- **Scikit-learn**: Machine learning library for model training
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing library
- **Joblib**: Model serialization and parallel processing
- **CICFlowMeter**: Network flow feature extraction tool

### Infrastructure & Deployment

- **Docker**: Containerization for consistent deployment
- **Docker Compose**: Multi-container application orchestration
- **Nginx**: Reverse proxy and load balancer
- **Redis**: In-memory data structure store for caching
- **Grafana**: Monitoring and observability dashboards
- **Prometheus**: Metrics collection and monitoring

### Development Tools

- **Git**: Version control system
- **VS Code**: Primary development environment
- **ESLint**: JavaScript/TypeScript code linting
- **Prettier**: Code formatting tool
- **Jest**: JavaScript testing framework
- **Pytest**: Python testing framework

### Security Tools

- **SSL/TLS**: Encrypted communication protocols
- **Fail2ban**: Intrusion prevention software
- **Firewall**: Network access control
- **Backup Tools**: Automated database backup solutions

## 11. Timeline (12-Week Plan)

### Phase 1: Research & Foundation (Weeks 1-2)

**Week 1: Project Setup & Research**

- Literature review of IDS/IPS systems and AI applications
- Dataset acquisition and analysis (CICIDS2017, NSL-KDD)
- Technology stack finalization and environment setup
- Team role assignments and communication protocols

**Week 2: Data Analysis & Feature Engineering**

- Exploratory data analysis of network flow datasets
- Feature importance analysis and selection methodology
- Data preprocessing pipeline design
- Initial AI model architecture planning

### Phase 2: AI Model Development (Weeks 3-4)

**Week 3: Model Training & Validation**

- Implement feature extraction pipeline (80→60 features)
- Train multiple ML models (Random Forest, SVM, Neural Networks)
- Cross-validation and hyperparameter tuning
- Model performance evaluation and comparison

**Week 4: Model Optimization & Integration**

- Final model selection and optimization
- Real-time inference pipeline development
- Socket communication setup for AI-to-Backend
- Model serialization and deployment preparation

### Phase 3: Backend Development (Weeks 5-6)

**Week 5: Core Backend Infrastructure**

- FastAPI application structure and configuration
- Database schema implementation with PostgreSQL
- User authentication and JWT token management
- Basic API endpoints for user management

**Week 6: Advanced Backend Features**

- Socket.io integration for real-time communication
- Alert processing and storage systems
- Network action implementation (IP/port blocking)
- API documentation and testing setup

### Phase 4: Frontend Development (Weeks 7-8)

**Week 7: Core UI Components**

- React application setup with TypeScript
- Authentication system and routing
- Dashboard layout and basic components
- Socket.io client integration

**Week 8: Advanced Frontend Features**

- Real-time alert system implementation
- Data visualization with Chart.js
- Attack management and network action interfaces
- User management interface (Admin only)

### Phase 5: Integration & Testing (Weeks 9-10)

**Week 9: System Integration**

- End-to-end system integration testing
- AI model to frontend data flow validation
- Socket communication testing and optimization
- Database performance testing and optimization

**Week 10: Comprehensive Testing**

- Unit testing for all components
- Integration testing for API endpoints
- Load testing for real-time socket connections
- Security testing and vulnerability assessment

### Phase 6: Documentation & Deployment (Weeks 11-12)

**Week 11: Documentation & Demo Preparation**

- Complete technical documentation
- User manual and installation guides
- Demo scenario preparation
- Performance benchmarking and optimization

**Week 12: Final Review & Deployment**

- Final code review and quality assurance
- Docker containerization and deployment scripts
- System deployment on target hardware
- Final presentation and project handover

### Team Responsibilities

- **AI Specialist**: Model development, feature engineering, performance optimization
- **Backend Developer**: API development, database design, socket implementation
- **Frontend Developer**: UI/UX design, dashboard development, real-time integration
- **DevOps Engineer**: Deployment, monitoring, security implementation
- **Project Manager**: Timeline coordination, documentation, testing oversight

## 12. Testing Plan

### Unit Testing Strategy

- **Backend API Tests**: Pytest framework for all endpoint testing
- **Frontend Component Tests**: Jest and React Testing Library
- **AI Model Tests**: Validation of prediction accuracy and performance
- **Database Tests**: SQLAlchemy model validation and query optimization
- **Coverage Target**: Minimum 80% code coverage across all components

### Integration Testing

- **API Integration**: End-to-end API workflow testing
- **Socket Communication**: Real-time message flow validation
- **Database Integration**: Data consistency and transaction testing
- **AI Pipeline Integration**: Feature extraction to prediction workflow

### Performance Testing

- **Load Testing**: Socket connection handling under high traffic
- **Stress Testing**: System behavior under maximum capacity
- **Latency Testing**: Real-time alert delivery performance
- **Database Performance**: Query optimization and connection pooling

### Security Testing

- **Authentication Testing**: JWT token validation and expiration
- **Authorization Testing**: Role-based access control validation
- **Input Validation**: SQL injection and XSS protection
- **Network Security**: SSL/TLS configuration and data encryption

### AI Model Testing

- **Accuracy Testing**: Precision, recall, and F1-score validation
- **False Positive Analysis**: Minimizing incorrect attack classifications
- **Real-time Performance**: Inference speed and throughput testing
- **Model Drift Detection**: Monitoring for performance degradation

### User Acceptance Testing

- **Role-based Testing**: Functionality validation for each user role
- **Usability Testing**: Interface design and user experience validation
- **Scenario Testing**: Complete attack detection and response workflows
- **Documentation Testing**: Installation and user guide validation

## 13. How to Run Locally

### Prerequisites

- Docker and Docker Compose installed
- Python 3.9+ for development
- Node.js 16+ for frontend development
- PostgreSQL 14+ (or use Docker container)

### Quick Start with Docker Compose

1. **Clone Repository and Setup Environment**

```bash
git clone <repository-url>
cd ids-ai-system
cp .env.example .env
# Edit .env file with your configuration
```

2. **Start All Services**

```bash
docker-compose up -d
```

This will start:

- PostgreSQL database (port 5432)
- Redis cache (port 6379)
- Backend API (port 8000)
- Frontend application (port 3000)
- AI model server (port 8001)

### Manual Setup (Development Mode)

#### 1. Database Setup

```bash
# Start PostgreSQL container
docker run -d --name ids-postgres \
  -e POSTGRES_DB=ids_database \
  -e POSTGRES_USER=ids_user \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 postgres:14

# Apply database migrations
cd backend
alembic upgrade head
```

#### 2. AI Model Server

```bash
cd ai-pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download or train the model
python train_model.py

# Start AI inference server
python ai_server.py
# Server runs on http://localhost:8001
```

#### 3. Backend API

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://ids_user:secure_password@localhost:5432/ids_database"
export JWT_SECRET="your-secret-key"
export AI_MODEL_URL="http://localhost:8001"

# Start backend server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# API available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

#### 4. Frontend Application

```bash
cd frontend
npm install

# Set environment variables
echo "REACT_APP_API_URL=http://localhost:8000" > .env
echo "REACT_APP_SOCKET_URL=ws://localhost:8000" >> .env

# Start development server
npm start
# Application available at http://localhost:3000
```

#### 5. Socket Communication Test

```bash
# Test AI to Backend socket connection
cd tests
python test_ai_socket.py

# Test Backend to Frontend socket connection
node test_frontend_socket.js
```

### Viewing the System in Action

1. **Access Dashboard**: Navigate to http://localhost:3000
2. **Default Login**:
   - Username: admin@ids.local
   - Password: admin123
3. **Generate Test Traffic**: Use provided scripts to simulate network flows

```bash
cd test-data
python generate_test_flows.py --attack-rate 0.1 --duration 300
```

4. **Monitor Alerts**: Watch real-time alerts appear in the dashboard
5. **Test Actions**: Block IPs and observe network action execution

### Monitoring System Health

- **Backend API Status**: http://localhost:8000/health
- **AI Model Status**: http://localhost:8001/health
- **Database Connection**: Check via backend health endpoint
- **Socket Connections**: Monitor WebSocket connections in browser dev tools

### Troubleshooting Common Issues

- **Port conflicts**: Modify docker-compose.yml port mappings
- **Database connection errors**: Verify PostgreSQL container status
- **AI model loading errors**: Check model file paths and permissions
- **Socket connection failures**: Verify firewall settings and port availability

## 14. Future Enhancements

### Automated Intrusion Prevention (Full IPS)

- **Automatic IP Blocking**: Real-time blocking without manual intervention
- **Dynamic Firewall Rules**: Automated rule creation and management
- **Threat Response Playbooks**: Predefined response actions for different attack types
- **Integration with Network Hardware**: Direct communication with routers and switches

### Advanced AI Capabilities

- **Online Learning**: Continuous model improvement from new attack patterns
- **Ensemble Models**: Multiple specialized models for different attack types
- **Anomaly Detection**: Unsupervised learning for zero-day attack detection
- **Behavioral Analysis**: User and entity behavior analytics (UEBA)

### Enhanced Visualization and Analytics

- **3D Network Topology**: Interactive network visualization
- **Attack Path Analysis**: Visual representation of attack progression
- **Predictive Analytics**: Forecasting potential security threats
- **Custom Dashboard Builder**: User-configurable dashboard layouts

### Scalability Improvements

- **Distributed Processing**: Multi-node deployment for large networks
- **Cloud Integration**: Hybrid on-premises and cloud deployment options
- **High Availability**: Redundant systems with automatic failover
- **Performance Optimization**: Enhanced processing for high-volume networks

### Integration Capabilities

- **SIEM Integration**: Export to popular SIEM platforms (Splunk, ELK Stack)
- **Threat Intelligence Feeds**: Integration with external threat databases
- **API Ecosystem**: Comprehensive REST and GraphQL APIs
- **Mobile Applications**: iOS and Android apps for remote monitoring

### Advanced Security Features

- **Multi-factor Authentication**: Enhanced user security
- **Certificate Management**: Automated SSL/TLS certificate handling
- **Encryption at Rest**: Database and file system encryption
- **Compliance Reporting**: Automated compliance report generation

### Operational Enhancements

- **Automated Backup and Recovery**: Comprehensive data protection
- **Health Monitoring**: Advanced system health and performance monitoring
- **Update Management**: Automated security updates and patch management
- **Documentation Portal**: Integrated help system and knowledge base

### Research and Development Areas

- **Quantum-Safe Cryptography**: Future-proof security implementations
- **Edge Computing**: Processing at network edge devices
- **5G Network Security**: Specialized detection for 5G infrastructure
- **IoT Device Protection**: Specialized monitoring for IoT networks

These enhancements represent the natural evolution of the IDS-AI system, transforming it from a detection system into a comprehensive network security platform capable of protecting modern, complex network infrastructures.
