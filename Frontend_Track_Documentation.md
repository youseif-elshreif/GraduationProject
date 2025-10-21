# Frontend Track Documentation - IDS-AI System

## Table of Contents

1. [Overview](#overview)
2. [Technology Stack](#technology-stack)
3. [Project Structure](#project-structure)
4. [Component Architecture](#component-architecture)
5. [State Management](#state-management)
6. [Real-time Communication](#real-time-communication)
7. [UI/UX Design System](#uiux-design-system)
8. [Page Components](#page-components)
9. [Security Implementation](#security-implementation)
10. [Performance Optimization](#performance-optimization)
11. [Testing Strategy](#testing-strategy)
12. [Development Setup](#development-setup)
13. [Build & Deployment](#build--deployment)
14. [Troubleshooting](#troubleshooting)

## Overview

The frontend application serves as the primary interface for the IDS-AI system, providing real-time monitoring, attack visualization, and network management capabilities. Built with React 18 and TypeScript, it offers a responsive, secure, and intuitive dashboard for security professionals.

### Key Features

- **Real-time Attack Monitoring**: Live attack detection with instant notifications
- **Interactive Dashboards**: Comprehensive security analytics and visualizations
- **Role-based Access Control**: Different interfaces for Admin, Security, and Viewer roles
- **Network Action Controls**: IP/port blocking and security response management
- **Responsive Design**: Mobile-friendly interface for on-the-go monitoring

## Technology Stack

### Core Technologies

```json
{
  "framework": "React 18.2.0",
  "language": "TypeScript 4.9+",
  "styling": "Tailwind CSS 3.3+",
  "bundler": "Vite 4.4+",
  "state_management": "Zustand 4.4+",
  "routing": "React Router 6.8+",
  "http_client": "Axios 1.4+",
  "websocket": "Socket.io-client 4.7+",
  "charts": "Chart.js 4.3+ with react-chartjs-2",
  "forms": "React Hook Form 7.45+",
  "validation": "Zod 3.22+",
  "notifications": "React Hot Toast 2.4+",
  "icons": "Lucide React 0.263+",
  "animations": "Framer Motion 10.12+"
}
```

### Development Tools

```json
{
  "linting": "ESLint 8.45+",
  "formatting": "Prettier 3.0+",
  "testing": "Vitest 0.34+ with React Testing Library",
  "e2e_testing": "Playwright 1.36+",
  "type_checking": "TypeScript",
  "dev_server": "Vite Dev Server",
  "package_manager": "npm 9+"
}
```

## Project Structure

```
frontend/
├── public/
│   ├── index.html
│   ├── favicon.ico
│   └── manifest.json
├── src/
│   ├── components/           # Reusable UI components
│   │   ├── ui/              # Base UI components
│   │   ├── charts/          # Chart components
│   │   ├── forms/           # Form components
│   │   └── layout/          # Layout components
│   ├── pages/               # Page components
│   │   ├── auth/            # Authentication pages
│   │   ├── dashboard/       # Dashboard pages
│   │   ├── alerts/          # Alert management pages
│   │   ├── network/         # Network action pages
│   │   └── admin/           # Admin pages
│   ├── hooks/               # Custom React hooks
│   ├── services/            # API and external services
│   ├── stores/              # State management stores
│   ├── types/               # TypeScript type definitions
│   ├── utils/               # Utility functions
│   ├── constants/           # Application constants
│   ├── styles/              # Global styles and themes
│   └── assets/              # Static assets
├── tests/                   # Test files
├── docs/                    # Component documentation
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── vite.config.ts
└── README.md
```

## Component Architecture

### Core Components Hierarchy

```typescript
// App.tsx - Root component
interface AppProps {}

// Layout Components
interface DashboardLayoutProps {
  children: React.ReactNode;
  user: User;
}

interface HeaderProps {
  user: User;
  onLogout: () => void;
}

interface SidebarProps {
  currentPage: string;
  userRole: UserRole;
}

// Page Components
interface DashboardPageProps {
  timeRange: TimeRange;
  filters: DashboardFilters;
}

interface AlertsPageProps {
  alerts: Alert[];
  onAlertAction: (alertId: string, action: AlertAction) => void;
}

// UI Components
interface AlertCardProps {
  alert: Alert;
  onAction: (action: AlertAction) => void;
  showActions: boolean;
}

interface ChartProps {
  data: ChartData;
  type: ChartType;
  options?: ChartOptions;
}
```

### Component Design Patterns

#### 1. Container/Presentational Pattern

```typescript
// Container Component
const AlertsContainer: React.FC = () => {
  const { alerts, loading, error } = useAlerts();
  const { blockIP, resolveAlert } = useNetworkActions();

  const handleAlertAction = (alertId: string, action: AlertAction) => {
    switch (action.type) {
      case "BLOCK_IP":
        blockIP(action.payload.ip);
        break;
      case "RESOLVE":
        resolveAlert(alertId, action.payload.resolution);
        break;
    }
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage error={error} />;

  return (
    <AlertsPresentation alerts={alerts} onAlertAction={handleAlertAction} />
  );
};

// Presentational Component
const AlertsPresentation: React.FC<AlertsPresentationProps> = ({
  alerts,
  onAlertAction,
}) => {
  return (
    <div className="space-y-4">
      {alerts.map((alert) => (
        <AlertCard
          key={alert.id}
          alert={alert}
          onAction={(action) => onAlertAction(alert.id, action)}
          showActions={true}
        />
      ))}
    </div>
  );
};
```

#### 2. Custom Hooks Pattern

```typescript
// useAlerts hook
export const useAlerts = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        setLoading(true);
        const response = await alertsService.getAlerts();
        setAlerts(response.data);
      } catch (err) {
        setError(err as Error);
      } finally {
        setLoading(false);
      }
    };

    fetchAlerts();
  }, []);

  // Real-time updates via socket
  useSocket("attack_detected", (newAlert: Alert) => {
    setAlerts((prev) => [newAlert, ...prev]);
  });

  return { alerts, loading, error, setAlerts };
};
```

## State Management

### Zustand Store Architecture

```typescript
// stores/authStore.ts
interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  token: localStorage.getItem("auth_token"),
  isAuthenticated: false,

  login: async (credentials) => {
    const response = await authService.login(credentials);
    const { user, token } = response.data;

    localStorage.setItem("auth_token", token);
    set({ user, token, isAuthenticated: true });
  },

  logout: () => {
    localStorage.removeItem("auth_token");
    set({ user: null, token: null, isAuthenticated: false });
  },

  refreshToken: async () => {
    const { token } = get();
    if (!token) return;

    try {
      const response = await authService.refreshToken(token);
      const newToken = response.data.token;

      localStorage.setItem("auth_token", newToken);
      set({ token: newToken });
    } catch (error) {
      get().logout();
    }
  },
}));

// stores/alertsStore.ts
interface AlertsState {
  alerts: Alert[];
  activeAlerts: Alert[];
  filters: AlertFilters;
  setAlerts: (alerts: Alert[]) => void;
  addAlert: (alert: Alert) => void;
  updateAlert: (id: string, updates: Partial<Alert>) => void;
  setFilters: (filters: AlertFilters) => void;
}

export const useAlertsStore = create<AlertsState>((set, get) => ({
  alerts: [],
  activeAlerts: [],
  filters: {
    severity: "all",
    type: "all",
    timeRange: "24h",
  },

  setAlerts: (alerts) => {
    const activeAlerts = alerts.filter((alert) => alert.status === "active");
    set({ alerts, activeAlerts });
  },

  addAlert: (alert) => {
    set((state) => ({
      alerts: [alert, ...state.alerts],
      activeAlerts:
        alert.status === "active"
          ? [alert, ...state.activeAlerts]
          : state.activeAlerts,
    }));
  },

  updateAlert: (id, updates) => {
    set((state) => ({
      alerts: state.alerts.map((alert) =>
        alert.id === id ? { ...alert, ...updates } : alert
      ),
      activeAlerts: state.activeAlerts.map((alert) =>
        alert.id === id ? { ...alert, ...updates } : alert
      ),
    }));
  },

  setFilters: (filters) => set({ filters }),
}));
```

## Real-time Communication

### Socket.io Integration

```typescript
// services/socketService.ts
class SocketService {
  private socket: Socket | null = null;
  private eventHandlers: Map<string, Function[]> = new Map();

  connect(token: string) {
    this.socket = io(process.env.REACT_APP_SOCKET_URL!, {
      auth: { token },
      transports: ["websocket"],
    });

    this.socket.on("connect", () => {
      console.log("Socket connected:", this.socket?.id);
    });

    this.socket.on("disconnect", () => {
      console.log("Socket disconnected");
    });

    // Set up event listeners
    this.setupEventHandlers();
  }

  private setupEventHandlers() {
    if (!this.socket) return;

    this.socket.on("attack_detected", (data: Alert) => {
      this.emit("attack_detected", data);
    });

    this.socket.on("alert_resolved", (data: AlertResolution) => {
      this.emit("alert_resolved", data);
    });

    this.socket.on("ip_blocked", (data: IPBlockEvent) => {
      this.emit("ip_blocked", data);
    });

    this.socket.on("system_status", (data: SystemStatus) => {
      this.emit("system_status", data);
    });
  }

  on(event: string, handler: Function) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event)!.push(handler);
  }

  off(event: string, handler: Function) {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  private emit(event: string, data: any) {
    const handlers = this.eventHandlers.get(event) || [];
    handlers.forEach((handler) => handler(data));
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.eventHandlers.clear();
  }
}

export const socketService = new SocketService();

// hooks/useSocket.ts
export const useSocket = (event: string, handler: Function) => {
  useEffect(() => {
    socketService.on(event, handler);

    return () => {
      socketService.off(event, handler);
    };
  }, [event, handler]);
};
```

### Real-time Alert System

```typescript
// components/alerts/AlertNotificationSystem.tsx
export const AlertNotificationSystem: React.FC = () => {
  const addAlert = useAlertsStore((state) => state.addAlert);
  const [notifications, setNotifications] = useState<AlertNotification[]>([]);

  useSocket("attack_detected", (alert: Alert) => {
    // Add to store
    addAlert(alert);

    // Show notification
    const notification: AlertNotification = {
      id: alert.id,
      title: `${alert.attack_type} Attack Detected`,
      message: `Source: ${alert.src_ip} → Target: ${alert.dst_ip}`,
      severity: alert.risk_level,
      timestamp: new Date(),
      actions: [
        { label: "Block IP", action: () => blockIP(alert.src_ip) },
        { label: "Investigate", action: () => navigateToAlert(alert.id) },
        { label: "Dismiss", action: () => dismissNotification(alert.id) },
      ],
    };

    setNotifications((prev) => [notification, ...prev.slice(0, 4)]);

    // Play alert sound
    playAlertSound(alert.risk_level);

    // Show toast notification
    toast.custom(
      (t) => (
        <AlertToast
          notification={notification}
          onClose={() => toast.dismiss(t.id)}
        />
      ),
      {
        duration: alert.risk_level === "critical" ? Infinity : 10000,
      }
    );
  });

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2">
      {notifications.map((notification) => (
        <AlertNotificationCard
          key={notification.id}
          notification={notification}
          onDismiss={(id) =>
            setNotifications((prev) => prev.filter((n) => n.id !== id))
          }
        />
      ))}
    </div>
  );
};
```

## UI/UX Design System

### Color Palette

```typescript
// styles/colors.ts
export const colors = {
  // Brand Colors
  primary: {
    50: "#eff6ff",
    100: "#dbeafe",
    500: "#3b82f6",
    600: "#2563eb",
    700: "#1d4ed8",
    900: "#1e3a8a",
  },

  // Semantic Colors
  success: {
    50: "#f0fdf4",
    100: "#dcfce7",
    500: "#22c55e",
    600: "#16a34a",
    700: "#15803d",
  },

  warning: {
    50: "#fffbeb",
    100: "#fef3c7",
    500: "#f59e0b",
    600: "#d97706",
    700: "#b45309",
  },

  danger: {
    50: "#fef2f2",
    100: "#fee2e2",
    500: "#ef4444",
    600: "#dc2626",
    700: "#b91c1c",
  },

  // Risk Level Colors
  risk: {
    low: "#22c55e", // Green
    medium: "#f59e0b", // Yellow
    high: "#f97316", // Orange
    critical: "#dc2626", // Red
  },

  // Neutral Colors
  gray: {
    50: "#f9fafb",
    100: "#f3f4f6",
    200: "#e5e7eb",
    300: "#d1d5db",
    400: "#9ca3af",
    500: "#6b7280",
    600: "#4b5563",
    700: "#374151",
    800: "#1f2937",
    900: "#111827",
  },
};
```

### Component Library

```typescript
// components/ui/Button.tsx
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "danger" | "ghost";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  variant = "primary",
  size = "md",
  loading = false,
  leftIcon,
  rightIcon,
  children,
  className = "",
  disabled,
  ...props
}) => {
  const baseClasses =
    "inline-flex items-center justify-center font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors duration-200";

  const variantClasses = {
    primary:
      "bg-primary-600 text-white hover:bg-primary-700 focus:ring-primary-500",
    secondary:
      "bg-gray-200 text-gray-900 hover:bg-gray-300 focus:ring-gray-500",
    danger:
      "bg-danger-600 text-white hover:bg-danger-700 focus:ring-danger-500",
    ghost: "text-gray-700 hover:bg-gray-100 focus:ring-gray-500",
  };

  const sizeClasses = {
    sm: "px-3 py-1.5 text-sm",
    md: "px-4 py-2 text-sm",
    lg: "px-6 py-3 text-base",
  };

  return (
    <button
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
      {!loading && leftIcon && <span className="mr-2">{leftIcon}</span>}
      {children}
      {!loading && rightIcon && <span className="ml-2">{rightIcon}</span>}
    </button>
  );
};

// components/ui/Card.tsx
interface CardProps {
  children: React.ReactNode;
  className?: string;
  padding?: "none" | "sm" | "md" | "lg";
  shadow?: "none" | "sm" | "md" | "lg";
}

export const Card: React.FC<CardProps> = ({
  children,
  className = "",
  padding = "md",
  shadow = "md",
}) => {
  const paddingClasses = {
    none: "",
    sm: "p-3",
    md: "p-4",
    lg: "p-6",
  };

  const shadowClasses = {
    none: "",
    sm: "shadow-sm",
    md: "shadow-md",
    lg: "shadow-lg",
  };

  return (
    <div
      className={`bg-white rounded-lg border border-gray-200 ${paddingClasses[padding]} ${shadowClasses[shadow]} ${className}`}
    >
      {children}
    </div>
  );
};
```

## Page Components

### Dashboard Page

```typescript
// pages/dashboard/DashboardPage.tsx
import { useState, useEffect } from "react";
import { useSocket } from "@/hooks/useSocket";
import { dashboardService } from "@/services/dashboardService";

interface DashboardStats {
  // KPI Cards
  activeThreats: number;
  activeThreatsChange: number;
  blockedIPs: number;
  blockedIPsToday: number;
  blockedPorts: number;
  topTargetedPort: number;
  responseTime: number;
  responseTimeStatus: "good" | "medium" | "slow";

  // Alerts Timeline
  alertsTimeline: {
    timestamp: string;
    critical: number;
    high: number;
    medium: number;
    low: number;
  }[];

  // Attack Types
  attackTypes: {
    type: string;
    count: number;
    percentage: number;
    priority: { critical: number; high: number; medium: number; low: number };
  }[];

  // Severity Over Time
  severityOverTime: {
    timestamp: string;
    high: number;
    medium: number;
    low: number;
  }[];

  // Network Topology
  networkTopology: {
    nodes: NetworkNode[];
    connections: NetworkConnection[];
    threats: ActiveThreat[];
  };

  // Top Protocols
  topProtocols: {
    protocol: string;
    connections: number;
    percentage: number;
    bandwidth: string;
    attacks: number;
    status: "normal" | "suspicious" | "high";
  }[];

  // Blocked IPs
  recentBlockedIPs: {
    ip: string;
    blockTime: string;
    reason: string;
    duration: "temporary" | "permanent";
    durationValue?: string;
    status: "active" | "expired";
  }[];

  // Blocked Ports
  blockedPortsList: {
    port: number;
    protocol: "TCP" | "UDP";
    reason: string;
    attacksBlocked: number;
    status: "active" | "inactive";
  }[];

  // Live Alerts
  liveAlerts: {
    id: string;
    severity: "critical" | "high" | "medium" | "low";
    type: string;
    timestamp: string;
    sourceIP: string;
    sourceLocation: string;
    targetIP: string;
    description: string;
  }[];

  // System Health
  systemHealth: {
    cpu: number;
    memory: number;
    storage: number;
    networkIn: string;
    networkOut: string;
    databaseStatus: "connected" | "disconnected";
    databaseResponseTime: number;
    aiEngineStatus: "active" | "inactive";
    detectionRate: number;
    processingSpeed: number;
    uptime: string;
  };
}

export const DashboardPage: React.FC = () => {
  const [timeRange, setTimeRange] = useState<TimeRange>("24h");
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [showAllAlertsModal, setShowAllAlertsModal] = useState(false);
  const [isPaused, setIsPaused] = useState(false);

  useEffect(() => {
    const fetchDashboardStats = async () => {
      try {
        setLoading(true);
        const response = await dashboardService.getStats(timeRange);
        setStats(response.data);
      } catch (error) {
        console.error("Failed to fetch dashboard stats:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardStats();

    // Refresh data every 30 seconds
    const interval = setInterval(fetchDashboardStats, 30000);
    return () => clearInterval(interval);
  }, [timeRange]);

  // Real-time updates via WebSocket
  useSocket("new_alert", (alert: Alert) => {
    if (!isPaused) {
      setStats((prev) =>
        prev
          ? {
              ...prev,
              activeThreats: prev.activeThreats + 1,
              liveAlerts: [alert, ...prev.liveAlerts.slice(0, 14)],
            }
          : null
      );
    }
  });

  useSocket("system_health", (health: SystemHealth) => {
    setStats((prev) => (prev ? { ...prev, systemHealth: health } : null));
  });

  useSocket("blocked_ip", (data: BlockedIP) => {
    setStats((prev) =>
      prev
        ? {
            ...prev,
            blockedIPs: prev.blockedIPs + 1,
            recentBlockedIPs: [data, ...prev.recentBlockedIPs.slice(0, 9)],
          }
        : null
    );
  });

  if (loading) return <DashboardSkeleton />;
  if (!stats) return <ErrorMessage message="Failed to load dashboard data" />;

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-900">
            Security Dashboard
          </h1>
          <TimeRangeSelector value={timeRange} onChange={setTimeRange} />
        </div>

        {/* KPI Cards - 4 في صف واحد */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="Active Threats"
            value={stats.activeThreats}
            change={stats.activeThreatsChange}
            icon={<ShieldAlert className="w-6 h-6" />}
            color="danger"
            sparkline={stats.alertsTimeline.map((d) => d.critical + d.high)}
            onClick={() => navigate("/threats?status=active")}
          />
          <KPICard
            title="Blocked IPs"
            value={stats.blockedIPs}
            subtitle={`${stats.blockedIPsToday} today`}
            icon={<Ban className="w-6 h-6" />}
            color="warning"
            linkText="View All"
            onLinkClick={() => navigate("/blocked-ips")}
          />
          <KPICard
            title="Blocked Ports"
            value={stats.blockedPorts}
            subtitle={`Port ${stats.topTargetedPort} most targeted`}
            icon={<Server className="w-6 h-6" />}
            color="info"
            linkText="Manage"
            onLinkClick={() => navigate("/port-management")}
          />
          <KPICard
            title="Response Time"
            value={`${stats.responseTime}ms`}
            icon={<Timer className="w-6 h-6" />}
            color={
              stats.responseTimeStatus === "good"
                ? "success"
                : stats.responseTimeStatus === "medium"
                ? "warning"
                : "danger"
            }
            subtitle={
              stats.responseTimeStatus === "good"
                ? "Excellent"
                : stats.responseTimeStatus === "medium"
                ? "Good"
                : "Slow"
            }
          />
        </div>

        {/* Row 1: Alerts Timeline & Attack Types Distribution */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2">
            <CardHeader>
              <h3 className="text-lg font-semibold">Alerts Over Time</h3>
            </CardHeader>
            <CardBody>
              <AlertsTimelineChart data={stats.alertsTimeline} height={300} />
            </CardBody>
            <CardFooter>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowAllAlertsModal(true)}
              >
                View All Alerts
              </Button>
            </CardFooter>
          </Card>

          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">
                Attack Types Distribution
              </h3>
            </CardHeader>
            <CardBody>
              <AttackTypesDonutChart data={stats.attackTypes} height={250} />
              <div className="mt-4 space-y-2">
                <h4 className="text-sm font-medium text-gray-600">
                  Priority Breakdown
                </h4>
                {stats.attackTypes[0]?.priority && (
                  <PriorityBreakdown priority={stats.attackTypes[0].priority} />
                )}
              </div>
            </CardBody>
          </Card>
        </div>

        {/* Row 2: Severity Levels & Network Threat Map */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">
                Severity Levels Over Time
              </h3>
            </CardHeader>
            <CardBody>
              <SeverityLevelsChart data={stats.severityOverTime} height={300} />
            </CardBody>
          </Card>

          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">
                Real-time Network Threat Map
              </h3>
            </CardHeader>
            <CardBody>
              <NetworkTopologyViewer
                topology={stats.networkTopology}
                height={300}
                interactive
              />
            </CardBody>
          </Card>
        </div>

        {/* Row 3: Network Security & Top Protocols */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">
                Network Security Overview
              </h3>
            </CardHeader>
            <CardBody>
              <NetworkSecurityOverview
                data={stats.networkSecurity}
                height={300}
              />
            </CardBody>
          </Card>

          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">Top Network Protocols</h3>
            </CardHeader>
            <CardBody>
              <TopProtocolsChart data={stats.topProtocols} height={300} />
            </CardBody>
          </Card>
        </div>

        {/* Row 4: Blocked IPs & Blocked Ports */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader className="flex justify-between">
              <div>
                <h3 className="text-lg font-semibold">Blocked IP Addresses</h3>
                <p className="text-sm text-gray-500">
                  {stats.blockedIPs} total
                </p>
              </div>
              <Button size="sm" onClick={() => navigate("/blocked-ips/add")}>
                Add IP
              </Button>
            </CardHeader>
            <CardBody>
              <BlockedIPsTable
                data={stats.recentBlockedIPs}
                compact
                autoRefresh
              />
            </CardBody>
            <CardFooter>
              <Button
                variant="outline"
                onClick={() => navigate("/blocked-ips")}
              >
                View All Blocked IPs
              </Button>
            </CardFooter>
          </Card>

          <Card>
            <CardHeader className="flex justify-between">
              <div>
                <h3 className="text-lg font-semibold">Blocked Ports</h3>
                <p className="text-sm text-gray-500">
                  {stats.blockedPorts} total
                </p>
              </div>
              <Button size="sm" onClick={() => navigate("/ports/add")}>
                Add Port
              </Button>
            </CardHeader>
            <CardBody>
              <BlockedPortsTable data={stats.blockedPortsList} compact />
            </CardBody>
          </Card>
        </div>
      </div>

      {/* Right Sidebar - Sticky */}
      <aside className="w-96 border-l border-gray-200 overflow-y-auto p-4 space-y-6">
        {/* Live Alerts Feed */}
        <Card>
          <CardHeader className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Live Alerts</h3>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setIsPaused(!isPaused)}
              >
                {isPaused ? <Play size={16} /> : <Pause size={16} />}
              </Button>
            </div>
          </CardHeader>
          <CardBody className="space-y-2 max-h-96 overflow-y-auto">
            {stats.liveAlerts.map((alert) => (
              <AlertCard
                key={alert.id}
                alert={alert}
                compact
                onBlock={(ip) => handleBlockIP(ip)}
                onInvestigate={(id) => navigate(`/threats/${id}`)}
                onDismiss={(id) => handleDismissAlert(id)}
              />
            ))}
          </CardBody>
          <CardFooter>
            <Button
              variant="outline"
              fullWidth
              onClick={() => navigate("/alerts")}
            >
              View All Alerts
            </Button>
          </CardFooter>
        </Card>

        {/* System Health Monitor */}
        <Card>
          <CardHeader>
            <h3 className="text-lg font-semibold">System Health</h3>
          </CardHeader>
          <CardBody className="space-y-4">
            <MetricBar
              label="CPU Usage"
              value={stats.systemHealth.cpu}
              status={
                stats.systemHealth.cpu < 70
                  ? "good"
                  : stats.systemHealth.cpu < 85
                  ? "warning"
                  : "danger"
              }
            />
            <MetricBar
              label="Memory"
              value={stats.systemHealth.memory}
              status={
                stats.systemHealth.memory < 70
                  ? "good"
                  : stats.systemHealth.memory < 85
                  ? "warning"
                  : "danger"
              }
            />
            <MetricBar
              label="Storage"
              value={stats.systemHealth.storage}
              status={
                stats.systemHealth.storage < 70
                  ? "good"
                  : stats.systemHealth.storage < 85
                  ? "warning"
                  : "danger"
              }
            />

            <Separator />

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Network</span>
                <span className="font-medium">
                  ↓ {stats.systemHealth.networkIn} / ↑{" "}
                  {stats.systemHealth.networkOut}
                </span>
              </div>

              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Database</span>
                <Badge
                  variant={
                    stats.systemHealth.databaseStatus === "connected"
                      ? "success"
                      : "danger"
                  }
                >
                  {stats.systemHealth.databaseStatus}
                </Badge>
              </div>

              <div className="flex justify-between text-sm">
                <span className="text-gray-600">AI Engine</span>
                <Badge
                  variant={
                    stats.systemHealth.aiEngineStatus === "active"
                      ? "success"
                      : "danger"
                  }
                >
                  {stats.systemHealth.aiEngineStatus}
                </Badge>
              </div>

              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Detection Rate</span>
                <span className="font-medium">
                  {stats.systemHealth.detectionRate}%
                </span>
              </div>

              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Processing</span>
                <span className="font-medium">
                  {stats.systemHealth.processingSpeed} flows/s
                </span>
              </div>

              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Uptime</span>
                <span className="font-medium">{stats.systemHealth.uptime}</span>
              </div>
            </div>
          </CardBody>
        </Card>
      </aside>

      {/* All Alerts Modal */}
      {showAllAlertsModal && (
        <AllAlertsModal
          timeRange={timeRange}
          onClose={() => setShowAllAlertsModal(false)}
        />
      )}
    </div>
  );
};
```

### Alerts Management Page

```typescript
// pages/alerts/AlertsPage.tsx
export const AlertsPage: React.FC = () => {
  const { alerts, loading, error } = useAlerts();
  const [filters, setFilters] = useState<AlertFilters>({
    severity: "all",
    type: "all",
    status: "active",
    timeRange: "24h",
  });
  const [selectedAlerts, setSelectedAlerts] = useState<string[]>([]);
  const { blockIP, resolveAlert } = useNetworkActions();

  const filteredAlerts = useMemo(() => {
    return alerts.filter((alert) => {
      if (filters.severity !== "all" && alert.risk_level !== filters.severity)
        return false;
      if (filters.type !== "all" && alert.attack_type !== filters.type)
        return false;
      if (filters.status !== "all" && alert.status !== filters.status)
        return false;
      return true;
    });
  }, [alerts, filters]);

  const handleBulkAction = async (action: BulkAction) => {
    switch (action.type) {
      case "BLOCK_IPS":
        await Promise.all(
          selectedAlerts.map((alertId) => {
            const alert = alerts.find((a) => a.id === alertId);
            return alert ? blockIP(alert.src_ip) : Promise.resolve();
          })
        );
        break;
      case "RESOLVE_ALL":
        await Promise.all(
          selectedAlerts.map((alertId) =>
            resolveAlert(alertId, action.resolution)
          )
        );
        break;
    }
    setSelectedAlerts([]);
  };

  if (loading) return <AlertsPageSkeleton />;
  if (error) return <ErrorMessage error={error} />;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Security Alerts</h1>
        <div className="flex space-x-3">
          <AlertsExportButton alerts={filteredAlerts} />
          <Button onClick={() => window.location.reload()}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card padding="sm">
        <AlertsFilters filters={filters} onChange={setFilters} />
      </Card>

      {/* Bulk Actions */}
      {selectedAlerts.length > 0 && (
        <Card padding="sm">
          <BulkActionsBar
            selectedCount={selectedAlerts.length}
            onAction={handleBulkAction}
            onClear={() => setSelectedAlerts([])}
          />
        </Card>
      )}

      {/* Alerts Table */}
      <Card padding="none">
        <AlertsTable
          alerts={filteredAlerts}
          selectedAlerts={selectedAlerts}
          onSelectionChange={setSelectedAlerts}
          onAlertAction={async (alertId, action) => {
            const alert = alerts.find((a) => a.id === alertId);
            if (!alert) return;

            switch (action.type) {
              case "BLOCK_IP":
                await blockIP(alert.src_ip);
                break;
              case "RESOLVE":
                await resolveAlert(alertId, action.resolution);
                break;
            }
          }}
        />
      </Card>
    </div>
  );
};
```

## Security Implementation

### Authentication & Authorization

```typescript
// utils/auth.ts
export const setAuthHeader = (token: string) => {
  axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;
};

export const removeAuthHeader = () => {
  delete axios.defaults.headers.common["Authorization"];
};

// components/auth/ProtectedRoute.tsx
interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: UserRole;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredRole,
}) => {
  const { isAuthenticated, user } = useAuthStore();
  const location = useLocation();

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (requiredRole && user?.role !== requiredRole && user?.role !== "admin") {
    return <Navigate to="/unauthorized" replace />;
  }

  return <>{children}</>;
};

// hooks/usePermissions.ts
export const usePermissions = () => {
  const user = useAuthStore((state) => state.user);

  const can = (permission: Permission): boolean => {
    if (!user) return false;

    const rolePermissions: Record<UserRole, Permission[]> = {
      admin: [
        "view_alerts",
        "take_actions",
        "manage_users",
        "view_stats",
        "system_config",
      ],
      security: ["view_alerts", "take_actions", "view_stats"],
      viewer: ["view_alerts", "view_stats"],
    };

    return rolePermissions[user.role]?.includes(permission) || false;
  };

  return { can, user };
};
```

### Input Validation & Sanitization

```typescript
// utils/validation.ts
import { z } from "zod";

export const loginSchema = z.object({
  email: z.string().email("Invalid email address"),
  password: z.string().min(6, "Password must be at least 6 characters"),
});

export const ipAddressSchema = z
  .string()
  .regex(
    /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/,
    "Invalid IP address format"
  );

export const blockIPSchema = z.object({
  ip: ipAddressSchema,
  duration: z
    .number()
    .min(60, "Duration must be at least 1 minute")
    .max(86400, "Duration cannot exceed 24 hours"),
  reason: z
    .string()
    .min(1, "Reason is required")
    .max(500, "Reason cannot exceed 500 characters"),
});

// hooks/useValidatedForm.ts
export const useValidatedForm = <T extends z.ZodRawShape>(
  schema: z.ZodObject<T>,
  onSubmit: (data: z.infer<typeof schema>) => void | Promise<void>
) => {
  const form = useForm<z.infer<typeof schema>>({
    resolver: zodResolver(schema),
  });

  const handleSubmit = form.handleSubmit(async (data) => {
    try {
      await onSubmit(data);
    } catch (error) {
      console.error("Form submission error:", error);
    }
  });

  return { ...form, handleSubmit };
};
```

## Performance Optimization

### Code Splitting & Lazy Loading

```typescript
// App.tsx
import { lazy, Suspense } from "react";

const DashboardPage = lazy(() => import("./pages/dashboard/DashboardPage"));
const AlertsPage = lazy(() => import("./pages/alerts/AlertsPage"));
const NetworkPage = lazy(() => import("./pages/network/NetworkPage"));
const AdminPage = lazy(() => import("./pages/admin/AdminPage"));

export const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <DashboardLayout />
            </ProtectedRoute>
          }
        >
          <Route
            index
            element={
              <Suspense fallback={<PageSkeleton />}>
                <DashboardPage />
              </Suspense>
            }
          />
          <Route
            path="alerts"
            element={
              <Suspense fallback={<PageSkeleton />}>
                <AlertsPage />
              </Suspense>
            }
          />
          <Route
            path="network"
            element={
              <Suspense fallback={<PageSkeleton />}>
                <NetworkPage />
              </Suspense>
            }
          />
          <Route
            path="admin"
            element={
              <ProtectedRoute requiredRole="admin">
                <Suspense fallback={<PageSkeleton />}>
                  <AdminPage />
                </Suspense>
              </ProtectedRoute>
            }
          />
        </Route>
      </Routes>
    </Router>
  );
};
```

### Memoization & Optimization

```typescript
// hooks/useOptimizedAlerts.ts
export const useOptimizedAlerts = (filters: AlertFilters) => {
  const { alerts } = useAlertsStore();

  const filteredAlerts = useMemo(() => {
    return alerts.filter((alert) => {
      if (filters.severity !== "all" && alert.risk_level !== filters.severity)
        return false;
      if (filters.type !== "all" && alert.attack_type !== filters.type)
        return false;
      if (filters.status !== "all" && alert.status !== filters.status)
        return false;
      return true;
    });
  }, [alerts, filters]);

  const groupedAlerts = useMemo(() => {
    return filteredAlerts.reduce((acc, alert) => {
      const key = alert.attack_type;
      if (!acc[key]) acc[key] = [];
      acc[key].push(alert);
      return acc;
    }, {} as Record<string, Alert[]>);
  }, [filteredAlerts]);

  return { filteredAlerts, groupedAlerts };
};

// components/charts/OptimizedChart.tsx
export const OptimizedChart: React.FC<ChartProps> = memo(
  ({ data, options, type }) => {
    const chartRef = useRef<Chart>(null);

    useEffect(() => {
      if (chartRef.current) {
        chartRef.current.update("none"); // Update without animation for performance
      }
    }, [data]);

    return (
      <Chart
        ref={chartRef}
        type={type}
        data={data}
        options={{
          ...options,
          animation: {
            duration: 0, // Disable animations for better performance
          },
          responsive: true,
          maintainAspectRatio: false,
        }}
      />
    );
  }
);
```

## Testing Strategy

### Unit Testing

```typescript
// tests/components/AlertCard.test.tsx
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { AlertCard } from "../components/alerts/AlertCard";
import { mockAlert } from "../__mocks__/alerts";

describe("AlertCard", () => {
  const mockOnAction = jest.fn();

  beforeEach(() => {
    mockOnAction.mockClear();
  });

  it("renders alert information correctly", () => {
    render(
      <AlertCard alert={mockAlert} onAction={mockOnAction} showActions={true} />
    );

    expect(screen.getByText(mockAlert.attack_type)).toBeInTheDocument();
    expect(screen.getByText(mockAlert.src_ip)).toBeInTheDocument();
    expect(screen.getByText(mockAlert.risk_level)).toBeInTheDocument();
  });

  it("calls onAction when block IP button is clicked", async () => {
    render(
      <AlertCard alert={mockAlert} onAction={mockOnAction} showActions={true} />
    );

    const blockButton = screen.getByText("Block IP");
    fireEvent.click(blockButton);

    await waitFor(() => {
      expect(mockOnAction).toHaveBeenCalledWith({
        type: "BLOCK_IP",
        payload: { ip: mockAlert.src_ip },
      });
    });
  });

  it("does not show action buttons when showActions is false", () => {
    render(
      <AlertCard
        alert={mockAlert}
        onAction={mockOnAction}
        showActions={false}
      />
    );

    expect(screen.queryByText("Block IP")).not.toBeInTheDocument();
    expect(screen.queryByText("Resolve")).not.toBeInTheDocument();
  });
});

// tests/hooks/useAlerts.test.ts
import { renderHook, waitFor } from "@testing-library/react";
import { useAlerts } from "../hooks/useAlerts";
import { alertsService } from "../services/alertsService";

jest.mock("../services/alertsService");

describe("useAlerts", () => {
  it("fetches alerts on mount", async () => {
    const mockAlerts = [mockAlert];
    (alertsService.getAlerts as jest.Mock).mockResolvedValue({
      data: mockAlerts,
    });

    const { result } = renderHook(() => useAlerts());

    expect(result.current.loading).toBe(true);

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
      expect(result.current.alerts).toEqual(mockAlerts);
    });
  });

  it("handles fetch errors gracefully", async () => {
    const mockError = new Error("Failed to fetch");
    (alertsService.getAlerts as jest.Mock).mockRejectedValue(mockError);

    const { result } = renderHook(() => useAlerts());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toEqual(mockError);
    });
  });
});
```

### Integration Testing

```typescript
// tests/integration/AlertsPage.test.tsx
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { AlertsPage } from "../pages/alerts/AlertsPage";
import { TestWrapper } from "../__mocks__/TestWrapper";
import * as alertsService from "../services/alertsService";
import * as networkService from "../services/networkService";

jest.mock("../services/alertsService");
jest.mock("../services/networkService");

describe("AlertsPage Integration", () => {
  it("displays alerts and allows blocking IPs", async () => {
    const mockAlerts = [mockAlert];
    jest.spyOn(alertsService, "getAlerts").mockResolvedValue({
      data: mockAlerts,
    });
    jest.spyOn(networkService, "blockIP").mockResolvedValue({
      success: true,
    });

    render(
      <TestWrapper>
        <AlertsPage />
      </TestWrapper>
    );

    // Wait for alerts to load
    await waitFor(() => {
      expect(screen.getByText(mockAlert.attack_type)).toBeInTheDocument();
    });

    // Click block IP button
    const blockButton = screen.getByText("Block IP");
    fireEvent.click(blockButton);

    // Verify network service was called
    await waitFor(() => {
      expect(networkService.blockIP).toHaveBeenCalledWith(mockAlert.src_ip);
    });
  });
});
```

## Development Setup

### Environment Configuration

```bash
# .env.local
REACT_APP_API_URL=http://localhost:8000
REACT_APP_SOCKET_URL=ws://localhost:8000
REACT_APP_ENVIRONMENT=development
REACT_APP_VERSION=1.0.0
```

### Package.json Scripts

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:ui": "vitest --ui",
    "test:coverage": "vitest --coverage",
    "lint": "eslint src --ext .ts,.tsx",
    "lint:fix": "eslint src --ext .ts,.tsx --fix",
    "format": "prettier --write src/**/*.{ts,tsx}",
    "type-check": "tsc --noEmit",
    "analyze": "npm run build && npx vite-bundle-analyzer dist"
  }
}
```

### Development Workflow

1. **Setup Development Environment**

```bash
npm install
npm run dev
```

2. **Code Quality Checks**

```bash
npm run lint
npm run type-check
npm run test
```

3. **Pre-commit Hooks** (using Husky)

```bash
# .husky/pre-commit
#!/bin/sh
npm run lint
npm run type-check
npm run test --run
```

## Build & Deployment

### Production Build

```typescript
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  build: {
    target: "es2020",
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["react", "react-dom"],
          charts: ["chart.js", "react-chartjs-2"],
          socket: ["socket.io-client"],
        },
      },
    },
  },
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM node:18-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

```nginx
# nginx.conf
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # Handle React Router
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API proxy
    location /api {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebSocket proxy
    location /socket.io {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Troubleshooting

### Common Issues & Solutions

#### 1. Socket Connection Issues

```typescript
// Check connection status
const [connected, setConnected] = useState(false);

useEffect(() => {
  socketService.socket?.on("connect", () => setConnected(true));
  socketService.socket?.on("disconnect", () => setConnected(false));
}, []);

// Display connection status
{
  !connected && (
    <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
      Connection lost. Attempting to reconnect...
    </div>
  );
}
```

#### 2. Performance Issues with Large Alert Lists

```typescript
// Implement virtualization for large lists
import { FixedSizeList as List } from "react-window";

const VirtualizedAlertsList: React.FC<{ alerts: Alert[] }> = ({ alerts }) => {
  const Row = ({
    index,
    style,
  }: {
    index: number;
    style: React.CSSProperties;
  }) => (
    <div style={style}>
      <AlertCard alert={alerts[index]} />
    </div>
  );

  return (
    <List height={600} itemCount={alerts.length} itemSize={120} width="100%">
      {Row}
    </List>
  );
};
```

#### 3. Memory Leaks with Socket Listeners

```typescript
// Proper cleanup in useEffect
useEffect(() => {
  const handleAlert = (alert: Alert) => {
    // Handle alert
  };

  socketService.on("attack_detected", handleAlert);

  return () => {
    socketService.off("attack_detected", handleAlert);
  };
}, []); // Empty dependency array is important
```

#### 4. State Synchronization Issues

```typescript
// Use optimistic updates with rollback
const blockIPOptimistic = async (ip: string) => {
  // Optimistic update
  updateAlertStatus(ip, "blocking");

  try {
    await networkService.blockIP(ip);
    updateAlertStatus(ip, "blocked");
  } catch (error) {
    // Rollback on error
    updateAlertStatus(ip, "active");
    showError("Failed to block IP");
  }
};
```

This comprehensive frontend documentation provides all the necessary details for implementing the client-side application of the IDS-AI system, including architecture, components, security, performance optimization, and deployment strategies.
