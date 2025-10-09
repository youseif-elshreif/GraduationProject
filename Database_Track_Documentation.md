# Database Track Documentation - IDS-AI System

## Table of Contents
1. [Overview](#overview)
2. [Database Design Philosophy](#database-design-philosophy)
3. [Schema Architecture](#schema-architecture)
4. [Core Tables](#core-tables)
5. [Relationships & Constraints](#relationships--constraints)
6. [Indexes & Performance](#indexes--performance)
7. [Data Partitioning](#data-partitioning)
8. [Security & Access Control](#security--access-control)
9. [Backup & Recovery](#backup--recovery)
10. [Monitoring & Maintenance](#monitoring--maintenance)
11. [Migration Strategy](#migration-strategy)
12. [Performance Tuning](#performance-tuning)
13. [Data Archival](#data-archival)
14. [Compliance & Auditing](#compliance--auditing)

## Overview

The database layer serves as the persistent storage foundation for the IDS-AI system, designed to handle high-volume network flow data, real-time attack alerts, user management, and comprehensive audit trails. PostgreSQL was chosen for its advanced features, JSON support, and excellent performance characteristics.

### Key Requirements
- **High Throughput**: Handle thousands of network flows per second
- **Real-time Queries**: Sub-second response times for alert retrieval
- **Data Integrity**: ACID compliance for critical security data
- **Scalability**: Support for growing data volumes and concurrent users
- **Security**: Row-level security and comprehensive access controls
- **Compliance**: Audit trails and data retention policies

## Database Design Philosophy

### ACID Compliance
- **Atomicity**: All database operations are atomic, ensuring data consistency
- **Consistency**: Database constraints maintain data integrity
- **Isolation**: Concurrent operations don't interfere with each other
- **Durability**: Committed data survives system failures

### Normalization Strategy
- **3NF Compliance**: Eliminate data redundancy while maintaining performance
- **Selective Denormalization**: Strategic denormalization for query performance
- **JSON Storage**: Use JSON columns for flexible, evolving data structures

### Performance-First Design
- **Strategic Indexing**: Indexes optimized for common query patterns
- **Partitioning**: Table partitioning for large datasets
- **Connection Pooling**: Efficient database connection management

## Schema Architecture

### Database Structure Overview
```sql
-- Database: ids_ai_system
-- Version: PostgreSQL 14+
-- Encoding: UTF8
-- Collation: en_US.UTF-8

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Custom Types
CREATE TYPE user_role AS ENUM ('admin', 'security', 'viewer');
CREATE TYPE alert_status AS ENUM ('active', 'investigating', 'resolved', 'false_positive');
CREATE TYPE risk_level AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE action_type AS ENUM ('block_ip', 'block_port', 'unblock_ip', 'unblock_port', 'investigate');
CREATE TYPE action_status AS ENUM ('pending', 'executed', 'failed', 'cancelled', 'expired');
CREATE TYPE log_level AS ENUM ('debug', 'info', 'warning', 'error', 'critical');
```

## Core Tables

### User Management Tables

```sql
-- Users table for authentication and authorization
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role user_role NOT NULL DEFAULT 'viewer',
    active BOOLEAN NOT NULL DEFAULT true,
    last_login TIMESTAMP WITH TIME ZONE,
    login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT users_email_check CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT users_username_check CHECK (length(username) >= 3),
    CONSTRAINT users_password_check CHECK (length(password_hash) >= 60)
);

-- User sessions for JWT token management
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL,
    refresh_token_hash TEXT,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_user_sessions_user_id ON user_sessions(user_id),
    INDEX idx_user_sessions_expires ON user_sessions(expires_at),
    INDEX idx_user_sessions_token ON user_sessions(token_hash)
);

-- Role permissions mapping
CREATE TABLE role_permissions (
    id SERIAL PRIMARY KEY,
    role user_role NOT NULL,
    permission VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint
    UNIQUE(role, permission)
);

-- Insert default permissions
INSERT INTO role_permissions (role, permission) VALUES
    ('admin', 'view_alerts'),
    ('admin', 'take_actions'),
    ('admin', 'manage_users'),
    ('admin', 'view_stats'),
    ('admin', 'system_config'),
    ('security', 'view_alerts'),
    ('security', 'take_actions'),
    ('security', 'view_stats'),
    ('viewer', 'view_alerts'),
    ('viewer', 'view_stats');
```

### Attack Classification Tables

```sql
-- Attack types and categories
CREATE TABLE attack_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    category VARCHAR(50) NOT NULL,
    severity_level INTEGER NOT NULL CHECK (severity_level BETWEEN 1 AND 4),
    description TEXT,
    mitigation_steps TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_attack_types_category ON attack_types(category),
    INDEX idx_attack_types_severity ON attack_types(severity_level)
);

-- Insert common attack types
INSERT INTO attack_types (name, category, severity_level, description) VALUES
    ('DDoS', 'Network', 4, 'Distributed Denial of Service attack'),
    ('Port Scan', 'Reconnaissance', 2, 'Network port scanning activity'),
    ('Brute Force', 'Authentication', 3, 'Password brute force attack'),
    ('SQL Injection', 'Web Application', 4, 'SQL injection attack attempt'),
    ('Malware C&C', 'Malware', 4, 'Malware command and control communication'),
    ('Data Exfiltration', 'Data Breach', 4, 'Unauthorized data transfer'),
    ('Web Attack', 'Web Application', 3, 'General web application attack'),
    ('Infiltration', 'Network', 4, 'Network infiltration attempt'),
    ('Botnet', 'Malware', 3, 'Botnet communication detected');
```

### Network Flow Tables

```sql
-- Network flows (partitioned by date)
CREATE TABLE flows (
    id BIGSERIAL,
    flow_id VARCHAR(100) NOT NULL,
    src_ip INET NOT NULL,
    dst_ip INET NOT NULL,
    src_port INTEGER NOT NULL CHECK (src_port BETWEEN 0 AND 65535),
    dst_port INTEGER NOT NULL CHECK (dst_port BETWEEN 0 AND 65535),
    protocol VARCHAR(10) NOT NULL,
    flow_duration BIGINT,
    total_fwd_packets INTEGER,
    total_bwd_packets INTEGER,
    total_length_fwd_packets BIGINT,
    total_length_bwd_packets BIGINT,
    
    -- Feature data (JSON for flexibility)
    features JSONB NOT NULL,
    raw_features JSONB,
    
    -- Metadata
    captured_at TIMESTAMP WITH TIME ZONE NOT NULL,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT flows_protocol_check CHECK (protocol IN ('TCP', 'UDP', 'ICMP', 'OTHER')),
    CONSTRAINT flows_flow_id_check CHECK (length(flow_id) > 0)
) PARTITION BY RANGE (created_at);

-- Create partitions for current and future months
DO $$
DECLARE
    start_date DATE;
    end_date DATE;
    table_name TEXT;
BEGIN
    -- Create partitions for current month and next 12 months
    FOR i IN 0..12 LOOP
        start_date := date_trunc('month', CURRENT_DATE) + (i || ' months')::INTERVAL;
        end_date := start_date + '1 month'::INTERVAL;
        table_name := 'flows_' || to_char(start_date, 'YYYY_MM');
        
        EXECUTE format('CREATE TABLE %I PARTITION OF flows 
                       FOR VALUES FROM (%L) TO (%L)',
                       table_name, start_date, end_date);
                       
        -- Create indexes on each partition
        EXECUTE format('CREATE INDEX %I ON %I (src_ip, dst_ip)', 
                       'idx_' || table_name || '_src_dst', table_name);
        EXECUTE format('CREATE INDEX %I ON %I (captured_at)', 
                       'idx_' || table_name || '_captured', table_name);
        EXECUTE format('CREATE INDEX %I ON %I USING GIN (features)', 
                       'idx_' || table_name || '_features', table_name);
    END LOOP;
END $$;

-- Flows summary table for quick statistics
CREATE TABLE flows_summary (
    date DATE PRIMARY KEY,
    total_flows BIGINT DEFAULT 0,
    total_attacks BIGINT DEFAULT 0,
    unique_src_ips INTEGER DEFAULT 0,
    unique_dst_ips INTEGER DEFAULT 0,
    top_protocols JSONB,
    top_ports JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Alert Management Tables

```sql
-- Security alerts (partitioned by date)
CREATE TABLE alerts (
    id BIGSERIAL,
    flow_id VARCHAR(100) NOT NULL,
    attack_type_id INTEGER NOT NULL REFERENCES attack_types(id),
    prediction_confidence DECIMAL(5,4) NOT NULL CHECK (prediction_confidence BETWEEN 0 AND 1),
    risk_level risk_level NOT NULL,
    
    -- Network details
    src_ip INET NOT NULL,
    dst_ip INET NOT NULL,
    src_port INTEGER CHECK (src_port BETWEEN 0 AND 65535),
    dst_port INTEGER CHECK (dst_port BETWEEN 0 AND 65535),
    protocol VARCHAR(10),
    
    -- Alert details
    attack_details JSONB,
    status alert_status NOT NULL DEFAULT 'active',
    
    -- Resolution details
    resolved_by INTEGER REFERENCES users(id),
    resolution_notes TEXT,
    resolved_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    detected_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT alerts_confidence_check CHECK (prediction_confidence > 0),
    CONSTRAINT alerts_resolution_check CHECK (
        (status IN ('resolved', 'false_positive') AND resolved_by IS NOT NULL AND resolved_at IS NOT NULL) OR
        (status NOT IN ('resolved', 'false_positive'))
    )
) PARTITION BY RANGE (created_at);

-- Create alert partitions
DO $$
DECLARE
    start_date DATE;
    end_date DATE;
    table_name TEXT;
BEGIN
    FOR i IN 0..12 LOOP
        start_date := date_trunc('month', CURRENT_DATE) + (i || ' months')::INTERVAL;
        end_date := start_date + '1 month'::INTERVAL;
        table_name := 'alerts_' || to_char(start_date, 'YYYY_MM');
        
        EXECUTE format('CREATE TABLE %I PARTITION OF alerts 
                       FOR VALUES FROM (%L) TO (%L)',
                       table_name, start_date, end_date);
                       
        -- Partition-specific indexes
        EXECUTE format('CREATE INDEX %I ON %I (status, risk_level)', 
                       'idx_' || table_name || '_status_risk', table_name);
        EXECUTE format('CREATE INDEX %I ON %I (src_ip)', 
                       'idx_' || table_name || '_src_ip', table_name);
        EXECUTE format('CREATE INDEX %I ON %I (attack_type_id)', 
                       'idx_' || table_name || '_attack_type', table_name);
        EXECUTE format('CREATE INDEX %I ON %I (detected_at)', 
                       'idx_' || table_name || '_detected', table_name);
    END LOOP;
END $$;

-- Alert aggregations for dashboards
CREATE TABLE alert_aggregations (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    hour INTEGER CHECK (hour BETWEEN 0 AND 23),
    attack_type_id INTEGER REFERENCES attack_types(id),
    risk_level risk_level,
    count BIGINT DEFAULT 0,
    avg_confidence DECIMAL(5,4),
    unique_sources INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint to prevent duplicates
    UNIQUE(date, hour, attack_type_id, risk_level)
);
```

### Network Action Tables

```sql
-- Network security actions
CREATE TABLE actions (
    id BIGSERIAL PRIMARY KEY,
    alert_id BIGINT,  -- Can be NULL for manual actions
    user_id INTEGER NOT NULL REFERENCES users(id),
    action_type action_type NOT NULL,
    target_value VARCHAR(255) NOT NULL,
    
    -- Action parameters (JSON for flexibility)
    parameters JSONB,
    
    -- Execution details
    status action_status NOT NULL DEFAULT 'pending',
    executed_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    
    -- Metadata
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT actions_target_check CHECK (length(target_value) > 0),
    CONSTRAINT actions_execution_check CHECK (
        (status = 'executed' AND executed_at IS NOT NULL) OR
        (status != 'executed')
    )
);

-- Blocked IPs view for quick access
CREATE VIEW blocked_ips AS
SELECT DISTINCT
    target_value as ip_address,
    user_id,
    executed_at as blocked_at,
    expires_at,
    parameters->>'reason' as reason,
    CASE 
        WHEN expires_at IS NULL THEN true
        WHEN expires_at > NOW() THEN true
        ELSE false
    END as is_active
FROM actions
WHERE action_type = 'block_ip' 
  AND status = 'executed'
  AND (expires_at IS NULL OR expires_at > NOW());

-- Action history for auditing
CREATE TABLE action_history (
    id BIGSERIAL PRIMARY KEY,
    action_id BIGINT NOT NULL REFERENCES actions(id),
    old_status action_status,
    new_status action_status NOT NULL,
    changed_by INTEGER REFERENCES users(id),
    change_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### System Logging Tables

```sql
-- Comprehensive system logs
CREATE TABLE system_logs (
    id BIGSERIAL PRIMARY KEY,
    log_level log_level NOT NULL,
    component VARCHAR(50) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    
    -- Context data
    metadata JSONB,
    user_id INTEGER REFERENCES users(id),
    session_id UUID,
    ip_address INET,
    user_agent TEXT,
    
    -- Request/Response data for API logs
    request_id UUID,
    request_method VARCHAR(10),
    request_path VARCHAR(500),
    response_status INTEGER,
    response_time_ms INTEGER,
    
    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create log partitions (daily partitions for high volume)
DO $$
DECLARE
    start_date DATE;
    end_date DATE;
    table_name TEXT;
BEGIN
    FOR i IN 0..90 LOOP  -- 90 days of partitions
        start_date := CURRENT_DATE + (i || ' days')::INTERVAL;
        end_date := start_date + '1 day'::INTERVAL;
        table_name := 'system_logs_' || to_char(start_date, 'YYYY_MM_DD');
        
        EXECUTE format('CREATE TABLE %I PARTITION OF system_logs 
                       FOR VALUES FROM (%L) TO (%L)',
                       table_name, start_date, end_date);
                       
        -- Partition indexes
        EXECUTE format('CREATE INDEX %I ON %I (log_level, component)', 
                       'idx_' || table_name || '_level_comp', table_name);
        EXECUTE format('CREATE INDEX %I ON %I (user_id)', 
                       'idx_' || table_name || '_user', table_name);
        EXECUTE format('CREATE INDEX %I ON %I USING GIN (metadata)', 
                       'idx_' || table_name || '_metadata', table_name);
    END LOOP;
END $$;

-- Security events for compliance
CREATE TABLE security_events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    severity INTEGER NOT NULL CHECK (severity BETWEEN 1 AND 5),
    user_id INTEGER REFERENCES users(id),
    ip_address INET,
    description TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Index for quick security queries
    INDEX idx_security_events_type_severity ON security_events(event_type, severity),
    INDEX idx_security_events_user ON security_events(user_id),
    INDEX idx_security_events_created ON security_events(created_at)
);
```

## Relationships & Constraints

### Primary Relationships

```sql
-- Flow to Alert relationship (one-to-many)
ALTER TABLE alerts ADD CONSTRAINT fk_alerts_flow_id 
    FOREIGN KEY (flow_id) REFERENCES flows(flow_id) 
    ON DELETE RESTRICT;

-- Alert to Action relationship (one-to-many)
ALTER TABLE actions ADD CONSTRAINT fk_actions_alert_id 
    FOREIGN KEY (alert_id) REFERENCES alerts(id) 
    ON DELETE SET NULL;

-- User relationships
ALTER TABLE alerts ADD CONSTRAINT fk_alerts_resolved_by 
    FOREIGN KEY (resolved_by) REFERENCES users(id) 
    ON DELETE SET NULL;

ALTER TABLE actions ADD CONSTRAINT fk_actions_user_id 
    FOREIGN KEY (user_id) REFERENCES users(id) 
    ON DELETE RESTRICT;

-- Attack type relationship
ALTER TABLE alerts ADD CONSTRAINT fk_alerts_attack_type 
    FOREIGN KEY (attack_type_id) REFERENCES attack_types(id) 
    ON DELETE RESTRICT;
```

### Data Integrity Constraints

```sql
-- Check constraints for data validation
ALTER TABLE alerts ADD CONSTRAINT check_alert_timestamps 
    CHECK (resolved_at IS NULL OR resolved_at >= created_at);

ALTER TABLE actions ADD CONSTRAINT check_action_expiry 
    CHECK (expires_at IS NULL OR expires_at > created_at);

ALTER TABLE flows ADD CONSTRAINT check_flow_duration 
    CHECK (flow_duration IS NULL OR flow_duration >= 0);

-- Unique constraints
ALTER TABLE flows ADD CONSTRAINT unique_flow_id_date 
    UNIQUE (flow_id, created_at);

-- Partial unique indexes for performance
CREATE UNIQUE INDEX idx_users_email_active 
    ON users(email) WHERE active = true;

CREATE UNIQUE INDEX idx_users_username_active 
    ON users(username) WHERE active = true;
```

## Indexes & Performance

### Strategic Indexing

```sql
-- High-frequency query indexes
CREATE INDEX CONCURRENTLY idx_alerts_active_recent 
    ON alerts(created_at DESC, status) 
    WHERE status = 'active';

CREATE INDEX CONCURRENTLY idx_alerts_risk_level_time 
    ON alerts(risk_level, created_at DESC)
    WHERE status IN ('active', 'investigating');

CREATE INDEX CONCURRENTLY idx_flows_time_src 
    ON flows(created_at DESC, src_ip);

CREATE INDEX CONCURRENTLY idx_actions_active_expires 
    ON actions(expires_at, status) 
    WHERE status = 'executed' AND expires_at IS NOT NULL;

-- Composite indexes for complex queries
CREATE INDEX CONCURRENTLY idx_alerts_composite_dashboard 
    ON alerts(status, risk_level, attack_type_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_flows_network_analysis 
    ON flows(src_ip, dst_ip, protocol, created_at DESC);

-- GIN indexes for JSON queries
CREATE INDEX CONCURRENTLY idx_flows_features_gin 
    ON flows USING GIN (features);

CREATE INDEX CONCURRENTLY idx_alerts_details_gin 
    ON alerts USING GIN (attack_details);

CREATE INDEX CONCURRENTLY idx_actions_params_gin 
    ON actions USING GIN (parameters);

-- Text search indexes
CREATE INDEX CONCURRENTLY idx_system_logs_message_search 
    ON system_logs USING GIN (to_tsvector('english', message));

-- Partial indexes for common filters
CREATE INDEX CONCURRENTLY idx_alerts_critical_active 
    ON alerts(created_at DESC) 
    WHERE risk_level = 'critical' AND status = 'active';

CREATE INDEX CONCURRENTLY idx_users_active_by_role 
    ON users(role, created_at) 
    WHERE active = true;
```

### Performance Monitoring Views

```sql
-- Index usage statistics
CREATE VIEW index_usage_stats AS
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation,
    most_common_vals,
    most_common_freqs
FROM pg_stats 
WHERE schemaname = 'public'
ORDER BY tablename, attname;

-- Query performance monitoring
CREATE VIEW slow_queries AS
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    min_time,
    max_time,
    stddev_time
FROM pg_stat_statements 
WHERE mean_time > 100  -- Queries taking more than 100ms on average
ORDER BY mean_time DESC;

-- Table size monitoring
CREATE VIEW table_sizes AS
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## Data Partitioning

### Automated Partition Management

```sql
-- Function to create monthly partitions
CREATE OR REPLACE FUNCTION create_monthly_partitions(
    table_name TEXT,
    months_ahead INTEGER DEFAULT 3
) RETURNS VOID AS $$
DECLARE
    start_date DATE;
    end_date DATE;
    partition_name TEXT;
BEGIN
    FOR i IN 0..months_ahead LOOP
        start_date := date_trunc('month', CURRENT_DATE) + (i || ' months')::INTERVAL;
        end_date := start_date + '1 month'::INTERVAL;
        partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
        
        -- Check if partition already exists
        IF NOT EXISTS (
            SELECT 1 FROM pg_tables 
            WHERE tablename = partition_name
        ) THEN
            EXECUTE format('CREATE TABLE %I PARTITION OF %I 
                           FOR VALUES FROM (%L) TO (%L)',
                           partition_name, table_name, start_date, end_date);
            
            RAISE NOTICE 'Created partition: %', partition_name;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to drop old partitions
CREATE OR REPLACE FUNCTION drop_old_partitions(
    table_name TEXT,
    retention_months INTEGER DEFAULT 12
) RETURNS VOID AS $$
DECLARE
    cutoff_date DATE;
    partition_record RECORD;
BEGIN
    cutoff_date := date_trunc('month', CURRENT_DATE) - (retention_months || ' months')::INTERVAL;
    
    FOR partition_record IN
        SELECT tablename 
        FROM pg_tables 
        WHERE tablename LIKE table_name || '_%'
          AND tablename < table_name || '_' || to_char(cutoff_date, 'YYYY_MM')
    LOOP
        EXECUTE format('DROP TABLE IF EXISTS %I', partition_record.tablename);
        RAISE NOTICE 'Dropped old partition: %', partition_record.tablename;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Automated partition maintenance (scheduled via cron or pg_cron)
SELECT create_monthly_partitions('flows', 6);
SELECT create_monthly_partitions('alerts', 6);
SELECT drop_old_partitions('flows', 12);
SELECT drop_old_partitions('alerts', 24);  -- Keep alerts longer for analysis
```

## Security & Access Control

### Row Level Security (RLS)

```sql
-- Enable RLS on sensitive tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_logs ENABLE ROW LEVEL SECURITY;

-- Users can only see their own profile (except admins)
CREATE POLICY user_profile_access ON users
    FOR ALL TO app_user
    USING (
        id = current_setting('app.current_user_id')::INTEGER OR
        current_setting('app.current_user_role') = 'admin'
    );

-- Alert access based on user role
CREATE POLICY alert_access ON alerts
    FOR SELECT TO app_user
    USING (
        current_setting('app.current_user_role') IN ('admin', 'security', 'viewer')
    );

-- Action access - users can see their own actions, admins see all
CREATE POLICY action_access ON actions
    FOR ALL TO app_user
    USING (
        user_id = current_setting('app.current_user_id')::INTEGER OR
        current_setting('app.current_user_role') = 'admin'
    );

-- System logs - restricted access
CREATE POLICY system_log_access ON system_logs
    FOR SELECT TO app_user
    USING (
        current_setting('app.current_user_role') = 'admin' OR
        (current_setting('app.current_user_role') = 'security' AND log_level != 'debug')
    );
```

### Database Roles & Permissions

```sql
-- Application roles
CREATE ROLE app_admin;
CREATE ROLE app_security;
CREATE ROLE app_viewer;
CREATE ROLE app_service;  -- For backend service connections

-- Grant permissions to roles
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_admin;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_admin;

GRANT SELECT, INSERT, UPDATE ON alerts, actions, flows TO app_security;
GRANT SELECT ON users, attack_types, system_logs TO app_security;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_security;

GRANT SELECT ON alerts, flows, attack_types TO app_viewer;

GRANT SELECT, INSERT, UPDATE ON flows, alerts, actions, system_logs TO app_service;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_service;

-- Create application users
CREATE USER ids_admin WITH PASSWORD 'secure_admin_password';
CREATE USER ids_backend WITH PASSWORD 'secure_backend_password';
CREATE USER ids_readonly WITH PASSWORD 'secure_readonly_password';

-- Grant roles to users
GRANT app_admin TO ids_admin;
GRANT app_service TO ids_backend;
GRANT app_viewer TO ids_readonly;

-- Connection limits
ALTER USER ids_backend CONNECTION LIMIT 50;
ALTER USER ids_admin CONNECTION LIMIT 10;
ALTER USER ids_readonly CONNECTION LIMIT 20;
```

### Encryption & Data Protection

```sql
-- Encrypt sensitive data using pgcrypto
CREATE OR REPLACE FUNCTION encrypt_sensitive_data(data TEXT) 
RETURNS TEXT AS $$
BEGIN
    RETURN encode(pgp_sym_encrypt(data, current_setting('app.encryption_key')), 'base64');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION decrypt_sensitive_data(encrypted_data TEXT) 
RETURNS TEXT AS $$
BEGIN
    RETURN pgp_sym_decrypt(decode(encrypted_data, 'base64'), current_setting('app.encryption_key'));
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Audit trigger for sensitive tables
CREATE OR REPLACE FUNCTION audit_trigger() 
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO system_logs (log_level, component, event_type, message, metadata, user_id)
    VALUES (
        'info',
        'database',
        TG_OP || '_' || TG_TABLE_NAME,
        'Database operation performed',
        jsonb_build_object(
            'table', TG_TABLE_NAME,
            'operation', TG_OP,
            'old_values', to_jsonb(OLD),
            'new_values', to_jsonb(NEW)
        ),
        current_setting('app.current_user_id', true)::INTEGER
    );
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Apply audit triggers to sensitive tables
CREATE TRIGGER audit_users AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION audit_trigger();

CREATE TRIGGER audit_actions AFTER INSERT OR UPDATE OR DELETE ON actions
    FOR EACH ROW EXECUTE FUNCTION audit_trigger();
```

## Backup & Recovery

### Backup Strategy

```sql
-- Full backup script (run via cron)
-- #!/bin/bash
-- BACKUP_DIR="/opt/backups/ids_db"
-- DATE=$(date +%Y%m%d_%H%M%S)
-- 
-- # Full database backup
-- pg_dump -h localhost -U postgres -d ids_ai_system \
--     -f "$BACKUP_DIR/full_backup_$DATE.sql" \
--     --verbose --create --clean
-- 
-- # Compress backup
-- gzip "$BACKUP_DIR/full_backup_$DATE.sql"
-- 
-- # Remove backups older than 30 days
-- find $BACKUP_DIR -name "full_backup_*.sql.gz" -mtime +30 -delete

-- Point-in-time recovery setup
ALTER SYSTEM SET wal_level = 'replica';
ALTER SYSTEM SET archive_mode = 'on';
ALTER SYSTEM SET archive_command = 'cp %p /opt/backups/wal_archive/%f';
ALTER SYSTEM SET max_wal_senders = 3;
ALTER SYSTEM SET wal_keep_segments = 32;

-- Backup validation function
CREATE OR REPLACE FUNCTION validate_backup_integrity() 
RETURNS TABLE(
    table_name TEXT,
    row_count BIGINT,
    last_updated TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.table_name::TEXT,
        t.row_count,
        t.last_updated
    FROM (
        SELECT 'users' as table_name, COUNT(*) as row_count, MAX(updated_at) as last_updated FROM users
        UNION ALL
        SELECT 'flows', COUNT(*), MAX(created_at) FROM flows
        UNION ALL
        SELECT 'alerts', COUNT(*), MAX(created_at) FROM alerts
        UNION ALL
        SELECT 'actions', COUNT(*), MAX(created_at) FROM actions
    ) t;
END;
$$ LANGUAGE plpgsql;
```

### Recovery Procedures

```sql
-- Recovery testing function
CREATE OR REPLACE FUNCTION test_recovery_point(
    target_time TIMESTAMP WITH TIME ZONE
) RETURNS TABLE(
    test_result TEXT,
    data_consistency BOOLEAN,
    missing_records INTEGER
) AS $$
DECLARE
    consistency_check BOOLEAN DEFAULT true;
    missing_count INTEGER DEFAULT 0;
BEGIN
    -- Test data consistency at recovery point
    SELECT COUNT(*) INTO missing_count
    FROM alerts a
    LEFT JOIN flows f ON a.flow_id = f.flow_id
    WHERE f.flow_id IS NULL
      AND a.created_at <= target_time;
    
    consistency_check := (missing_count = 0);
    
    RETURN QUERY SELECT 
        'Recovery point validation'::TEXT,
        consistency_check,
        missing_count;
END;
$$ LANGUAGE plpgsql;
```

## Monitoring & Maintenance

### Performance Monitoring

```sql
-- Database health monitoring views
CREATE VIEW db_health_summary AS
SELECT 
    'Database Size' as metric,
    pg_size_pretty(pg_database_size(current_database())) as value
UNION ALL
SELECT 
    'Active Connections',
    COUNT(*)::TEXT
FROM pg_stat_activity
WHERE state = 'active'
UNION ALL
SELECT 
    'Cache Hit Ratio',
    ROUND(
        100 * sum(blks_hit) / (sum(blks_hit) + sum(blks_read))::NUMERIC, 
        2
    )::TEXT || '%'
FROM pg_stat_database
WHERE datname = current_database()
UNION ALL
SELECT 
    'Deadlocks',
    deadlocks::TEXT
FROM pg_stat_database
WHERE datname = current_database();

-- Connection monitoring
CREATE VIEW connection_stats AS
SELECT 
    usename,
    application_name,
    client_addr,
    state,
    COUNT(*) as connection_count,
    MAX(backend_start) as oldest_connection
FROM pg_stat_activity
WHERE pid != pg_backend_pid()
GROUP BY usename, application_name, client_addr, state
ORDER BY connection_count DESC;

-- Lock monitoring
CREATE VIEW lock_monitoring AS
SELECT 
    l.mode,
    l.granted,
    l.relation::regclass as table_name,
    a.usename,
    a.query,
    a.query_start
FROM pg_locks l
JOIN pg_stat_activity a ON l.pid = a.pid
WHERE l.relation IS NOT NULL
ORDER BY l.granted, a.query_start;
```

### Automated Maintenance

```sql
-- Automated VACUUM and ANALYZE
CREATE OR REPLACE FUNCTION maintenance_vacuum_analyze() 
RETURNS VOID AS $$
DECLARE
    table_record RECORD;
BEGIN
    FOR table_record IN
        SELECT schemaname, tablename
        FROM pg_tables
        WHERE schemaname = 'public'
          AND tablename NOT LIKE '%_partition_%'
    LOOP
        EXECUTE format('VACUUM ANALYZE %I.%I', 
                      table_record.schemaname, 
                      table_record.tablename);
        
        RAISE NOTICE 'Vacuumed and analyzed: %.%', 
                     table_record.schemaname, 
                     table_record.tablename;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Statistics update for query planner
CREATE OR REPLACE FUNCTION update_table_statistics() 
RETURNS VOID AS $$
BEGIN
    -- Update statistics for critical tables
    ANALYZE flows;
    ANALYZE alerts;
    ANALYZE actions;
    ANALYZE users;
    
    -- Update pg_stat_statements
    SELECT pg_stat_statements_reset();
    
    RAISE NOTICE 'Statistics updated successfully';
END;
$$ LANGUAGE plpgsql;

-- Clean up expired data
CREATE OR REPLACE FUNCTION cleanup_expired_data() 
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
    temp_count INTEGER;
BEGIN
    -- Clean expired user sessions
    DELETE FROM user_sessions 
    WHERE expires_at < NOW();
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;
    
    -- Clean expired actions
    UPDATE actions 
    SET status = 'expired' 
    WHERE status = 'executed' 
      AND expires_at IS NOT NULL 
      AND expires_at < NOW();
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;
    
    -- Archive old system logs (move to archive table)
    INSERT INTO system_logs_archive
    SELECT * FROM system_logs 
    WHERE created_at < NOW() - INTERVAL '90 days';
    
    DELETE FROM system_logs 
    WHERE created_at < NOW() - INTERVAL '90 days';
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
```

## Migration Strategy

### Schema Versioning

```sql
-- Schema version tracking
CREATE TABLE schema_migrations (
    version VARCHAR(20) PRIMARY KEY,
    description TEXT NOT NULL,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    execution_time_ms INTEGER,
    checksum TEXT
);

-- Migration execution function
CREATE OR REPLACE FUNCTION execute_migration(
    version_number VARCHAR(20),
    migration_description TEXT,
    migration_sql TEXT
) RETURNS VOID AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    execution_time INTEGER;
    sql_checksum TEXT;
BEGIN
    -- Check if migration already executed
    IF EXISTS (SELECT 1 FROM schema_migrations WHERE version = version_number) THEN
        RAISE EXCEPTION 'Migration % already executed', version_number;
    END IF;
    
    start_time := clock_timestamp();
    sql_checksum := md5(migration_sql);
    
    -- Execute migration
    EXECUTE migration_sql;
    
    end_time := clock_timestamp();
    execution_time := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    -- Record migration
    INSERT INTO schema_migrations (version, description, execution_time_ms, checksum)
    VALUES (version_number, migration_description, execution_time, sql_checksum);
    
    RAISE NOTICE 'Migration % completed in %ms', version_number, execution_time;
END;
$$ LANGUAGE plpgsql;
```

### Safe Migration Procedures

```sql
-- Example migration with rollback capability
DO $$
BEGIN
    -- Start transaction for atomic migration
    BEGIN
        -- Migration: Add new column to alerts table
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'alerts' AND column_name = 'severity_score'
        ) THEN
            ALTER TABLE alerts ADD COLUMN severity_score INTEGER;
            UPDATE alerts SET severity_score = 
                CASE risk_level
                    WHEN 'low' THEN 1
                    WHEN 'medium' THEN 2
                    WHEN 'high' THEN 3
                    WHEN 'critical' THEN 4
                END;
            ALTER TABLE alerts ALTER COLUMN severity_score SET NOT NULL;
            
            -- Record migration
            INSERT INTO schema_migrations (version, description)
            VALUES ('2025.001', 'Add severity_score column to alerts');
        END IF;
        
    EXCEPTION WHEN OTHERS THEN
        -- Rollback on error
        RAISE NOTICE 'Migration failed: %', SQLERRM;
        RAISE;
    END;
END $$;
```

## Performance Tuning

### Query Optimization

```sql
-- Materialized views for expensive queries
CREATE MATERIALIZED VIEW mv_alert_dashboard_stats AS
SELECT 
    date_trunc('hour', created_at) as hour,
    risk_level,
    attack_type_id,
    COUNT(*) as alert_count,
    AVG(prediction_confidence) as avg_confidence,
    COUNT(DISTINCT src_ip) as unique_sources
FROM alerts
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY date_trunc('hour', created_at), risk_level, attack_type_id;

CREATE UNIQUE INDEX ON mv_alert_dashboard_stats (hour, risk_level, attack_type_id);

-- Refresh materialized view (scheduled every 15 minutes)
CREATE OR REPLACE FUNCTION refresh_dashboard_stats() 
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_alert_dashboard_stats;
    RAISE NOTICE 'Dashboard stats refreshed at %', NOW();
END;
$$ LANGUAGE plpgsql;

-- Connection pooling configuration
-- postgresql.conf optimizations:
-- max_connections = 200
-- shared_buffers = 256MB
-- effective_cache_size = 1GB
-- work_mem = 4MB
-- maintenance_work_mem = 64MB
-- checkpoint_completion_target = 0.9
-- wal_buffers = 16MB
-- default_statistics_target = 100
```

### Database Configuration Tuning

```sql
-- Performance tuning recommendations
SELECT 
    'shared_buffers' as setting,
    '25% of RAM' as recommended_value,
    current_setting('shared_buffers') as current_value
UNION ALL
SELECT 
    'effective_cache_size',
    '75% of RAM',
    current_setting('effective_cache_size')
UNION ALL
SELECT 
    'work_mem',
    '4MB per connection',
    current_setting('work_mem')
UNION ALL
SELECT 
    'maintenance_work_mem',
    '10% of RAM (max 2GB)',
    current_setting('maintenance_work_mem');

-- Automated statistics collection
CREATE OR REPLACE FUNCTION collect_performance_stats() 
RETURNS TABLE(
    metric_name TEXT,
    metric_value NUMERIC,
    recorded_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'queries_per_second'::TEXT,
        (SELECT SUM(calls) FROM pg_stat_statements)::NUMERIC / 
        EXTRACT(EPOCH FROM (NOW() - pg_postmaster_start_time())),
        NOW()
    UNION ALL
    SELECT 
        'cache_hit_ratio',
        ROUND(100 * SUM(blks_hit) / NULLIF(SUM(blks_hit) + SUM(blks_read), 0), 2),
        NOW()
    FROM pg_stat_database;
END;
$$ LANGUAGE plpgsql;
```

This comprehensive database documentation provides a complete foundation for implementing, maintaining, and optimizing the PostgreSQL database for the IDS-AI system, ensuring high performance, security, and scalability.