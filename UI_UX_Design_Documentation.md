# Design Requirements Documentation - IDS-AI System

## Table of Contents

1. [Project Overview & Design Requirements](#project-overview--design-requirements)
2. [User Personas & Use Cases](#user-personas--use-cases)
3. [Page-by-Page Design Requirements](#page-by-page-design-requirements)
4. [Component Design Requirements](#component-design-requirements)
5. [Data Visualization Requirements](#data-visualization-requirements)
6. [User Interface Flow Requirements](#user-interface-flow-requirements)
7. [Visual Design Requirements](#visual-design-requirements)
8. [Responsive Design Requirements](#responsive-design-requirements)
9. [Accessibility Requirements](#accessibility-requirements)
10. [Animation & Interaction Requirements](#animation--interaction-requirements)
11. [Theme Requirements](#theme-requirements)
12. [Technical Design Constraints](#technical-design-constraints)

## Project Overview & Design Requirements

### What is IDS-AI System?

**IDS-AI** هو نظام كشف التسلل الذكي اللي بيحلل network traffic في الوقت الفعلي ويكشف الهجمات السيبرانية باستخدام الذكاء الاصطناعي.

### Main Design Goals

المطلوب من الـ designer يعمل interface يحقق الأهداف دي:

1. **Security Control Center Look**: الشكل يبقى زي command center للأمان
2. **Real-time Monitoring**: كل حاجة تظهر live updates
3. **Professional Enterprise Feel**: يناسب الشركات الكبيرة والمؤسسات
4. **Quick Response**: المستخدم يقدر يتعامل مع التهديدات بسرعة
5. **Clear Data Presentation**: البيانات المعقدة تتعرض بشكل مفهوم

### System Functions للـ Designer

النظام ده بيعمل:

- **Network Monitoring**: مراقبة حركة الشبكة 24/7
- **Threat Detection**: كشف الهجمات السيبرانية
- **Alert Management**: إدارة التنبيهات والإنذارات
- **Incident Response**: الرد على الحوادث الأمنية
- **Reporting**: تقارير أمنية مفصلة
- **System Management**: إدارة النظام والمستخدمين

## User Personas & Use Cases

### Primary Users اللي هيستخدموا النظام

#### 1. Security Analyst (محلل الأمان)

**الشخصية**:

- عمره 28-45 سنة، خبرة في الأمان السيبراني
- بيشتغل في shift system (24/7 monitoring)
- محتاج يشوف التهديدات بسرعة ويتعامل معاها

**المطلوب في الـ Design**:

- Dashboard واضح يظهر أهم التهديدات فوق
- Alert system يلفت الانتباه للحاجات المهمة
- Quick actions للرد على التهديدات
- Details view لكل تهديد مع كل المعلومات

#### 2. System Administrator (مدير النظام)

**الشخصية**:

- مسؤول عن النظام كله وإعداداته
- محتاج يشوف performance وصحة النظام
- بيعمل maintenance ويدير المستخدمين

**المطلوب في الـ Design**:

- System health dashboard
- Configuration pages سهلة الاستخدام
- User management interface
- System logs وmonitoring tools

#### 3. IT Manager (مدير تقنية المعلومات)

**الشخصية**:

- محتاج تقارير high-level للإدارة
- مهتم بالـ KPIs والمؤشرات العامة
- مش متخصص تقني عميق

**المطلوب في الـ Design**:

- Executive dashboard مع charts وgraphs
- Summary reports واضحة ومختصرة
- Trend analysis وcomparison charts
- Printable reports للمشاركة

### Use Cases رئيسية

#### Use Case 1: Daily Monitoring

المستخدم يفتح النظام الصبح ويشوف:

- إيه اللي حصل بالليل
- التهديدات النشطة دلوقتي
- حالة النظام العامة
- أهم الإحصائيات

#### Use Case 2: Threat Response

لما يحصل تهديد جديد:

- Alert يظهر فوراً
- تفاصيل التهديد تكون واضحة
- Actions متاحة للرد السريع
- Timeline للأحداث المرتبطة

#### Use Case 3: Investigation

لما المحلل يحتاج يحقق في حادثة:

- Search في كل البيانات
- Filter وsort للنتائج
- Deep dive في الـ details
- Export البيانات للتحليل

## Page-by-Page Design Requirements

### 1. Login Page Requirements

**الوصف**: صفحة تسجيل الدخول للنظام

**المطلوب في الـ Design**:

- **Logo**: شعار IDS-AI في الأعلى، واضح ومميز
- **Login Form**:
  - Username field مع icon
  - Password field مع show/hide button
  - Remember me checkbox
  - Login button كبير وواضح
- **Background**: خلفية تعبر عن الأمان السيبراني (مثل network patterns أو cyber grid)
- **Security Feel**: الصفحة تدي إحساس بالأمان والمهنية
- **Responsive**: تشتغل على الموبايل والتابلت
- **Loading State**: loading animation لما المستخدم يضغط login

**العناصر المطلوبة**:

- Company branding area
- Login form container
- Error messages area
- Footer مع copyright وlinks

### 2. Main Dashboard Requirements

**الوصف**: الصفحة الرئيسية اللي بتظهر overview عام للنظام

**المطلوب في الـ Design**:

#### Header Section:

- **Navigation Bar**: قائمة أفقية بالصفحات الرئيسية
- **Search Bar**: بحث global في النظام
- **Notifications**: جرس التنبيهات مع عدد الإشعارات الجديدة
- **User Menu**: صورة المستخدم مع dropdown menu
- **System Status**: مؤشر حالة النظام (green/yellow/red)

#### Main Content Area:

- **KPI Cards Section**: 4-6 cards تظهر أهم المؤشرات:

  - عدد التهديدات النشطة
  - حركة الشبكة الحالية
  - عدد الـ alerts الجديدة
  - حالة النظام العامة
  - زمن الاستجابة
  - عدد الـ IPs المحظورة

- **Real-time Threat Map**: خريطة أو network diagram يظهر:

  - التهديدات الحية
  - اتجاه الهجمات
  - الـ nodes المتأثرة
  - Traffic flow

- **Live Alerts Panel**: قائمة بآخر التنبيهات:

  - نوع التهديد
  - وقت الحدوث
  - مصدر التهديد
  - مستوى الخطورة
  - Quick actions

- **Traffic Analysis Chart**: جراف يظهر:
  - حركة الشبكة عبر الزمن
  - التهديدات المكتشفة
  - المقارنة مع الفترات السابقة

#### Sidebar (Optional):

- **Quick Stats**: إحصائيات سريعة
- **System Health**: صحة مكونات النظام
- **Recent Activity**: آخر الأنشطة

### 3. Threat Detection Page Requirements

**الوصف**: صفحة مراقبة التهديدات في الوقت الفعلي

**المطلوب في الـ Design**:

#### Top Section:

- **Filter Bar**: فلاتر للبحث والتصفية:
  - نوع التهديد (DDoS, Port Scan, etc.)
  - مستوى الخطورة
  - المصدر
  - الفترة الزمنية
- **Bulk Actions**: إجراءات جماعية على التهديدات المختارة

#### Main Content:

- **Threats Table**: جدول بكل التهديدات:
  - Timestamp
  - Source IP
  - Target IP
  - Threat Type
  - Severity Level
  - Status
  - Actions (Block, Investigate, Dismiss)
- **Pagination**: للتنقل بين الصفحات
- **Real-time Updates**: التحديث التلقائي للبيانات

#### Right Sidebar:

- **Threat Statistics**: إحصائيات التهديدات
- **Top Attackers**: أكثر IPs مهاجمة
- **Attack Patterns**: أنماط الهجمات

### 4. Alert Details Modal Requirements

**الوصف**: نافذة منبثقة تظهر تفاصيل التهديد

**المطلوب في الـ Design**:

- **Header**: نوع التهديد ومستوى الخطورة
- **Tabs**:
  - **Overview**: معلومات أساسية
  - **Network Details**: تفاصيل الشبكة
  - **Timeline**: الأحداث المرتبطة
  - **Response**: إجراءات الرد
- **Action Buttons**:
  - Block IP
  - Create Firewall Rule
  - Escalate
  - Mark as Resolved
- **Close Button**: إغلاق النافذة

### 5. Network Analysis Page Requirements

**الوصف**: صفحة تحليل حركة الشبكة

**المطلوب في الـ Design**:

#### Top Controls:

- **Time Range Selector**: اختيار الفترة الزمنية
- **Visualization Type**: نوع العرض (Chart, Table, Map)
- **Metrics Selector**: المؤشرات المطلوب عرضها

#### Main Area:

- **Interactive Charts**: جرافات تفاعلية:
  - Line charts للاتجاهات
  - Bar charts للمقارنات
  - Pie charts للتوزيعات
  - Heatmaps للكثافة
- **Network Topology**: رسم بياني للشبكة
- **Data Table**: جدول بالبيانات التفصيلية

### 6. Reports Page Requirements

**الوصف**: صفحة التقارير والتحليلات

**المطلوب في الـ Design**:

#### Report Builder:

- **Report Type Selection**: نوع التقرير
- **Date Range**: الفترة الزمنية
- **Filters**: فلاتر التقرير
- **Format Options**: PDF, Excel, CSV

#### Reports Gallery:

- **Predefined Reports**: تقارير جاهزة
- **Custom Reports**: تقارير مخصصة
- **Scheduled Reports**: تقارير مجدولة

#### Report Preview:

- **Charts and Graphs**: الرسوم البيانية
- **Summary Tables**: جداول ملخصة
- **Export Options**: خيارات التصدير

### 7. System Management Page Requirements

**الوصف**: صفحة إدارة النظام والإعدادات

**المطلوب في الـ Design**:

#### Left Navigation:

- **System Configuration**
- **User Management**
- **Network Settings**
- **Alert Rules**
- **Maintenance**

#### Main Content Area (يتغير حسب القسم المختار):

- **Forms**: نماذج الإعدادات
- **Tables**: قوائم المستخدمين/القواعد
- **Status Panels**: حالة النظام
- **Action Buttons**: أزرار الإجراءات

### 8. User Profile Page Requirements

**الوصف**: صفحة الملف الشخصي للمستخدم

**المطلوب في الـ Design**:

- **Profile Information**: معلومات المستخدم
- **Change Password**: تغيير كلمة المرور
- **Notification Preferences**: تفضيلات التنبيهات
- **Theme Settings**: إعدادات المظهر
- **Activity Log**: سجل النشاطات

## Component Design Requirements

### 1. Navigation Components

#### Main Navigation Bar

**المطلوب**:

- **Logo Area**: شعار IDS-AI على الشمال
- **Navigation Links**: روابط الصفحات الرئيسية
  - Dashboard
  - Threats
  - Network
  - Reports
  - System
- **Search Bar**: بحث global مع autocomplete
- **Right Side**: notifications, user menu, system status

**Visual Requirements**:

- خلفية داكنة أو متدرجة
- أيقونات واضحة لكل link
- Active state مميز للصفحة الحالية
- Hover effects للـ interactivity

#### Sidebar Navigation (Alternative)

**المطلوب**:

- **Collapsible**: قابل للطي والفرد
- **Icons + Text**: أيقونة مع نص لكل عنصر
- **Grouping**: تجميع العناصر حسب الوظيفة
- **Badge Indicators**: عرض أرقام التنبيهات

### 2. Data Display Components

#### KPI Cards

**المطلوب لكل Card**:

- **Main Number**: الرقم الرئيسي (كبير وواضح)
- **Label**: وصف المؤشر
- **Trend Indicator**: سهم أو رمز يوضح الاتجاه
- **Change Value**: قيمة التغيير (+/-)
- **Time Period**: الفترة الزمنية
- **Background Color**: لون يعبر عن الحالة
- **Icon**: أيقونة معبرة عن نوع المؤشر

**أمثلة على الـ KPIs المطلوبة**:

- Active Threats (التهديدات النشطة)
- Blocked IPs (الـ IPs المحظورة)
- Network Traffic (حركة الشبكة)
- Response Time (زمن الاستجابة)
- System Uptime (وقت تشغيل النظام)
- Detection Rate (معدل الكشف)

#### Alert Cards

**المطلوب لكل Alert**:

- **Severity Indicator**: شريط ملون يوضح مستوى الخطورة
- **Timestamp**: وقت حدوث التهديد
- **Threat Type**: نوع التهديد (DDoS, Port Scan, etc.)
- **Source Info**: معلومات المصدر (IP, Location)
- **Target Info**: معلومات الهدف
- **Status Badge**: حالة التهديد (New, In Progress, Resolved)
- **Quick Actions**: أزرار سريعة (Block, Investigate, Dismiss)

#### Data Tables

**المطلوب**:

- **Header Row**: أسماء الأعمدة مع إمكانية الترتيب
- **Sortable Columns**: ترتيب البيانات
- **Filterable**: فلترة حسب القيم
- **Selectable Rows**: اختيار عدة صفوف
- **Pagination**: تقسيم الصفحات
- **Search**: بحث في الجدول
- **Export Options**: تصدير البيانات
- **Row Actions**: إجراءات لكل صف

### 3. Input Components

#### Search Components

**Global Search Bar**:

- **Search Icon**: أيقونة البحث
- **Placeholder Text**: نص توضيحي
- **Autocomplete**: اقتراحات أثناء الكتابة
- **Recent Searches**: البحثات الأخيرة
- **Advanced Filters**: فلاتر متقدمة

#### Filter Components

**Filter Bar**:

- **Dropdown Filters**: قوائم منسدلة للفلترة
- **Date Range Picker**: اختيار الفترة الزمنية
- **Clear Filters**: مسح جميع الفلاتر
- **Active Filter Tags**: عرض الفلاتر النشطة
- **Save Filter Set**: حفظ مجموعة فلاتر

#### Form Components

**Input Fields**:

- **Text Inputs**: حقول النص العادية
- **IP Address Input**: حقل خاص بعناوين IP
- **Port Number Input**: حقل أرقام المنافذ
- **Dropdown Selects**: قوائم الاختيار
- **Checkboxes**: مربعات الاختيار
- **Radio Buttons**: أزرار الخيار الواحد
- **Date/Time Pickers**: اختيار التاريخ والوقت

### 4. Status & Alert Components

#### Status Indicators

**System Status**:

- **Healthy**: أخضر، كل شيء يعمل بشكل طبيعي
- **Warning**: أصفر، تحذير أو مشكلة بسيطة
- **Critical**: أحمر، مشكلة خطيرة تحتاج تدخل
- **Offline**: رمادي، النظام غير متاح

**Threat Severity**:

- **Critical**: أحمر غامق، تهديد خطير جداً
- **High**: أحمر، تهديد عالي
- **Medium**: برتقالي، تهديد متوسط
- **Low**: أزرق، تهديد منخفض
- **Info**: رمادي، معلومات عامة

#### Notification Components

**Alert Badges**:

- **Number Badge**: عدد التنبيهات الجديدة
- **Pulsing Effect**: تأثير نبضات للتهديدات الجديدة
- **Color Coding**: ألوان مختلفة حسب الأهمية

**Toast Notifications**:

- **Success Messages**: رسائل النجاح
- **Error Messages**: رسائل الخطأ
- **Warning Messages**: رسائل التحذير
- **Info Messages**: رسائل إعلامية

### 5. Action Components

#### Buttons

**Primary Actions**:

- **Investigate**: فحص التهديد
- **Block IP**: حظر عنوان IP
- **Create Rule**: إنشاء قاعدة جديدة
- **Generate Report**: إنشاء تقرير

**Secondary Actions**:

- **Dismiss**: تجاهل التنبيه
- **Snooze**: تأجيل التنبيه
- **Mark as Read**: وضع علامة كمقروء
- **Export**: تصدير البيانات

**Danger Actions**:

- **Delete**: حذف
- **Reset**: إعادة تعيين
- **Force Stop**: إيقاف إجباري

#### Dropdown Menus

**Context Menus**:

- **Quick Actions**: إجراءات سريعة للتهديدات
- **Export Options**: خيارات التصدير المختلفة
- **View Options**: خيارات عرض البيانات

### 6. Modal & Dialog Components

#### Alert Detail Modal

**المطلوب**:

- **Modal Header**: عنوان التهديد ومستوى الخطورة
- **Close Button**: زر الإغلاق (X)
- **Tab Navigation**: تبويبات للمعلومات المختلفة
- **Content Area**: منطقة المحتوى الرئيسي
- **Action Footer**: أزرار الإجراءات في الأسفل

#### Confirmation Dialogs

**المطلوب**:

- **Warning Icon**: أيقونة تحذير
- **Clear Message**: رسالة واضحة عن الإجراء
- **Cancel Button**: زر الإلغاء
- **Confirm Button**: زر التأكيد (ملون حسب نوع الإجراء)

## Data Visualization Requirements

### 1. Charts & Graphs المطلوبة

#### Real-time Line Charts

**الغرض**: عرض البيانات المتغيرة عبر الزمن
**المطلوب في الـ Design**:

- **Multi-line Support**: عدة خطوط في نفس الجراف
- **Color Coding**: ألوان مختلفة لكل خط
- **Interactive Tooltips**: معلومات عند hover
- **Zoom & Pan**: تكبير وتحريك الجراف
- **Time Axis**: محور زمني واضح
- **Legend**: مفتاح الألوان
- **Threshold Lines**: خطوط العتبات الحرجة

**أمثلة على البيانات**:

- Network traffic over time
- Threat detection rate
- System performance metrics
- Response time trends

#### Bar Charts & Histograms

**الغرض**: مقارنة البيانات والتوزيعات
**المطلوب**:

- **Vertical & Horizontal**: أعمدة رأسية وأفقية
- **Grouped Bars**: أعمدة مجمعة للمقارنة
- **Color Gradients**: تدرجات لونية حسب القيمة
- **Value Labels**: عرض القيم على الأعمدة
- **Interactive Selection**: اختيار الأعمدة

**أمثلة**:

- Top attacking countries
- Most targeted ports
- Threat types distribution
- Monthly incident counts

#### Pie & Donut Charts

**الغرض**: عرض النسب والتوزيعات
**المطلوب**:

- **Percentage Labels**: عرض النسب المئوية
- **Legend with Colors**: مفتاح ملون
- **Interactive Slices**: أجزاء تفاعلية
- **Center Information**: معلومات في الوسط (للـ donut)
- **Explosion Effect**: تفريق الأجزاء عند الضغط

**أمثلة**:

- Attack types breakdown
- Traffic distribution by protocol
- System resource usage
- Alert severity distribution

#### Heatmaps

**الغرض**: عرض كثافة البيانات والأنماط
**المطلوب**:

- **Color Scale**: تدرج لوني للقيم
- **Grid Layout**: تخطيط شبكي منظم
- **Hover Details**: تفاصيل عند hover
- **Scale Legend**: مقياس الألوان
- **Time-based**: محاور زمنية

**أمثلة**:

- Attack intensity by time/day
- Network activity patterns
- Geographic threat distribution
- Port scanning patterns

### 2. Network Visualization Requirements

#### Network Topology Diagram

**الغرض**: عرض بنية الشبكة والاتصالات
**المطلوب في الـ Design**:

- **Node Types**: أنواع مختلفة من العقد
  - Servers (مربعات)
  - Workstations (دوائر)
  - Routers (معين)
  - Firewalls (درع)
  - Switches (مستطيل)
- **Connection Lines**: خطوط الاتصال
  - Normal traffic (خط أزرق)
  - Suspicious traffic (خط برتقالي)
  - Blocked traffic (خط أحمر منقط)
- **Interactive Features**:
  - Click على node لعرض التفاصيل
  - Hover لعرض معلومات سريعة
  - Zoom in/out
  - Pan للحركة
- **Status Indicators**:
  - Health status لكل node
  - Traffic intensity
  - Active threats

#### Geographic Map

**الغرض**: عرض الهجمات حسب الموقع الجغرافي
**المطلوب**:

- **World Map**: خريطة العالم
- **Attack Origins**: نقاط أصل الهجمات
- **Attack Paths**: خطوط مسار الهجمات
- **Intensity Indicators**: مؤشرات الكثافة
- **Country Information**: معلومات عن كل دولة
- **Zoom Controls**: تحكم في التكبير

### 3. Real-time Data Display

#### Live Data Feeds

**المطلوب**:

- **Auto-refresh**: تحديث تلقائي للبيانات
- **Smooth Transitions**: انتقالات سلسة للتغييرات
- **New Data Highlighting**: تمييز البيانات الجديدة
- **Pause/Resume**: إيقاف/استئناف التحديث
- **Timestamp Display**: عرض آخر تحديث

#### Progress Indicators

**المطلوب**:

- **Progress Bars**: أشرطة التقدم
- **Circular Progress**: دوائر التقدم
- **Animated Counters**: عدادات متحركة
- **Loading States**: حالات التحميل

### 4. Data Table Requirements

#### Advanced Data Tables

**المطلوب**:

- **Column Sorting**: ترتيب الأعمدة
- **Multi-column Filtering**: فلترة متعددة
- **Search Functionality**: وظيفة البحث
- **Row Selection**: اختيار الصفوف
- **Bulk Actions**: إجراءات جماعية
- **Export Options**: خيارات التصدير
- **Pagination**: تقسيم الصفحات
- **Row Details**: تفاصيل الصف القابلة للتوسيع

#### Mobile-friendly Tables

**المطلوب**:

- **Card Layout**: تخطيط بطاقات للموبايل
- **Swipe Actions**: إجراءات بالسحب
- **Collapsible Columns**: أعمدة قابلة للطي
- **Touch-friendly**: مناسب للمس

## User Interface Flow Requirements

### 1. User Journey Flows

#### Login Flow

**الخطوات المطلوبة**:

1. **Landing على Login Page**
2. **Enter Credentials** (Username/Password)
3. **Loading State** (أثناء التحقق)
4. **Success**: Redirect لـ Dashboard
5. **Error**: Error message مع إمكانية إعادة المحاولة

**التصميم المطلوب**:

- Loading spinner أثناء التحقق
- Error states واضحة
- Success feedback قبل الانتقال

#### Daily Monitoring Flow

**السيناريو**: محلل الأمان يبدأ يومه

1. **Dashboard Overview** (5 ثوان)
   - Quick scan للأرقام الرئيسية
   - System status check
2. **Alert Review** (2-3 دقائق)
   - مراجعة الـ alerts الجديدة
   - ترتيب حسب الأولوية
3. **Threat Investigation** (حسب الحاجة)
   - فتح تفاصيل التهديد
   - تحليل البيانات
   - اتخاذ إجراء

#### Threat Response Flow

**السيناريو**: ظهور تهديد جديد

1. **Alert Notification** (فوري)
   - Push notification
   - Audio alert (optional)
   - Visual indicator
2. **Quick Assessment** (30 ثانية)
   - Alert summary
   - Severity level
   - Initial details
3. **Detailed Investigation** (2-5 دقائق)
   - Full threat details
   - Related events
   - Context information
4. **Response Action** (1-2 دقائق)
   - Choose response
   - Execute action
   - Confirm completion

### 2. Navigation Flow Requirements

#### Primary Navigation Structure

```
Main Navigation:
├── Dashboard (Home)
├── Threats (Real-time monitoring)
├── Network (Analysis & monitoring)
├── Reports (Analytics & reports)
├── System (Configuration & management)
└── User Menu
    ├── Profile
    ├── Settings
    ├── Help
    └── Logout
```

#### Page-to-Page Flow

**Dashboard → Threats**:

- Click on threat count → Filter applied automatically
- Click on alert → Direct to alert details
- "View All" button → Full threats page

**Threats → Investigation**:

- Click on threat row → Modal with details
- "Investigate" button → Deep analysis view
- "Block IP" → Confirmation dialog

#### Modal Flow Requirements

**Alert Detail Modal**:

1. **Open**: Click على alert
2. **Navigate**: Tabs للمعلومات المختلفة
3. **Action**: اختيار إجراء من الأزرار
4. **Confirmation**: تأكيد الإجراء إذا لزم الأمر
5. **Close**: العودة للصفحة الأساسية

### 3. Interactive Element Requirements

#### Hover States

**المطلوب للـ Designer**:

- **Cards**: رفع Card بـ shadow عند hover
- **Buttons**: تغيير اللون أو إضافة تأثير
- **Table Rows**: تمييز الصف عند hover
- **Charts**: إظهار tooltip مع التفاصيل
- **Navigation Items**: تمييز العنصر

#### Click States

**المطلوب**:

- **Button Press**: تأثير الضغط
- **Loading States**: spinner أو skeleton
- **Success States**: تأكيد نجاح العملية
- **Error States**: رسالة خطأ واضحة

#### Focus States

**للـ Keyboard Navigation**:

- **Visible Focus Ring**: إطار واضح حول العنصر المحدد
- **Skip Links**: روابط للتخطي للمحتوى الرئيسي
- **Tab Order**: ترتيب منطقي للتنقل

### 4. Form Interaction Requirements

#### Input Field Behaviors

**المطلوب**:

- **Focus State**: تمييز الحقل النشط
- **Validation**: فحص المدخلات في الوقت الفعلي
- **Error Display**: عرض الأخطاء بوضوح
- **Success Indication**: تأكيد صحة المدخل
- **Placeholder Text**: نص توضيحي مفيد

#### Search Behavior

**المطلوب**:

- **Auto-complete**: اقتراحات أثناء الكتابة
- **Search History**: البحثات السابقة
- **No Results State**: رسالة عند عدم وجود نتائج
- **Loading State**: مؤشر التحميل أثناء البحث

### 5. Notification & Alert Requirements

#### Toast Notifications

**المطلوب**:

- **Position**: أعلى يمين الشاشة
- **Auto-dismiss**: اختفاء تلقائي بعد 5 ثوان
- **Manual Close**: زر إغلاق يدوي
- **Stacking**: تراكم الإشعارات المتعددة
- **Types**:
  - Success (أخضر)
  - Error (أحمر)
  - Warning (برتقالي)
  - Info (أزرق)

#### System Alerts

**للتهديدات الحرجة**:

- **Modal Alert**: نافذة منبثقة للتهديدات الخطيرة
- **Sound Alert**: تنبيه صوتي (optional)
- **Blinking Indicator**: مؤشر وامض
- **Persistent Display**: عرض مستمر حتى المعالجة

### 6. Loading & Error States

#### Loading States

**المطلوب**:

- **Page Loading**: Skeleton screens للصفحات
- **Data Loading**: Spinners للبيانات
- **Action Loading**: Loading على الأزرار
- **Progressive Loading**: تحميل تدريجي للبيانات الكبيرة

#### Error States

**المطلوب**:

- **Network Errors**: رسالة خطأ الشبكة
- **404 Pages**: صفحة غير موجودة
- **403 Access Denied**: رفض الوصول
- **500 Server Error**: خطأ الخادم
- **No Data**: عدم وجود بيانات للعرض

#### Empty States

**المطلوب**:

- **Illustration**: رسمة توضيحية
- **Helpful Message**: رسالة مفيدة للمستخدم
- **Action Button**: زر لبدء إجراء
- **Context**: شرح سبب عدم وجود البيانات

## Visual Design Requirements

### 1. Color Scheme Requirements

#### Primary Colors للنظام

**المطلوب من الـ Designer**:

- **Brand Blue**: `#1e40af` - اللون الأساسي للبراند
- **Security Green**: `#059669` - للحالات الآمنة والإيجابية
- **Warning Orange**: `#d97706` - للتحذيرات والانتباه
- **Danger Red**: `#dc2626` - للتهديدات والأخطار
- **Info Cyan**: `#0891b2` - للمعلومات المحايدة

#### Status Colors (مهم جداً)

**للتهديدات**:

- **Critical**: `#b91c1c` (أحمر غامق) - تهديدات خطيرة جداً
- **High**: `#ea580c` (برتقالي أحمر) - تهديدات عالية
- **Medium**: `#ca8a04` (أصفر داكن) - تهديدات متوسطة
- **Low**: `#2563eb` (أزرق) - تهديدات منخفضة
- **Safe**: `#16a34a` (أخضر) - حالة آمنة

#### Background Colors

**Light Theme**:

- **Primary Background**: `#ffffff` (أبيض)
- **Secondary Background**: `#f8fafc` (رمادي فاتح جداً)
- **Card Background**: `#ffffff` مع shadow
- **Sidebar Background**: `#f1f5f9` (رمادي فاتح)

**Dark Theme**:

- **Primary Background**: `#0f172a` (أزرق داكن جداً)
- **Secondary Background**: `#1e293b` (أزرق داكن)
- **Card Background**: `#1e293b` مع border
- **Sidebar Background**: `#334155` (رمادي أزرق)

### 2. Typography Requirements

#### Font Selection

**المطلوب**:

- **Primary Font**: Inter أو system font للواجهة العامة
- **Monospace Font**: JetBrains Mono للـ IPs والكود والبيانات التقنية
- **Arabic Support**: إذا كان النظام يدعم العربية

#### Text Hierarchy

**المطلوب للـ Designer**:

- **Page Titles**: 32px، bold، لون داكن
- **Section Headers**: 24px، semi-bold
- **Card Titles**: 18px، medium
- **Body Text**: 16px، regular، مقروء
- **Labels**: 14px، medium، للتسميات
- **Captions**: 12px، regular، للنصوص الثانوية

#### Technical Text

**IP Addresses والبيانات التقنية**:

- **Font**: Monospace (JetBrains Mono)
- **Size**: 14px
- **Weight**: Medium
- **Color**: مميز عن النص العادي

### 3. Iconography Requirements

#### System Icons المطلوبة

**Security Icons**:

- Shield (للحماية والأمان)
- Alert Triangle (للتحذيرات)
- Eye (للمراقبة)
- Lock (للأمان)
- Key (للمصادقة)
- Firewall (للحماية)

**Network Icons**:

- Network/Globe (للشبكة)
- Server (للخوادم)
- Router (للموجهات)
- Computer (للأجهزة)
- Connection Lines (للاتصالات)

**Action Icons**:

- Block/Stop (للحظر)
- Play/Pause (للتشغيل/الإيقاف)
- Search (للبحث)
- Filter (للفلترة)
- Export (للتصدير)
- Settings (للإعدادات)

**Navigation Icons**:

- Dashboard/Grid (للوحة الرئيسية)
- Chart/Analytics (للتقارير)
- Bell (للإشعارات)
- User (للملف الشخصي)
- Menu/Hamburger (للقائمة)

#### Icon Style Requirements

**المطلوب**:

- **Style**: Outline أو Filled (موحد في كل النظام)
- **Size**: 16px, 20px, 24px حسب الاستخدام
- **Stroke Width**: 1.5px للـ outline icons
- **Color**: يتبع نظام الألوان
- **Hover State**: تغيير لون عند hover

### 4. Spacing & Layout

#### Grid System

**المطلوب للـ Designer**:

- **Base Unit**: 8px (كل المسافات مضاعفات الـ 8)
- **Container Max Width**: 1440px للشاشات الكبيرة
- **Margins**: 24px على الجانبين
- **Gutter**: 24px بين الأعمدة

#### Component Spacing

**Cards**:

- **Padding**: 24px داخلي
- **Margin**: 16px بين الـ cards
- **Gap**: 24px في الـ grid

**Buttons**:

- **Padding**: 12px horizontal, 8px vertical للـ medium
- **Gap**: 12px بين الأزرار
- **Icon Gap**: 8px بين الأيقونة والنص

#### Page Layout

**Header Height**: 64px ثابت
**Sidebar Width**: 240px قابل للطي إلى 64px
**Content Padding**: 24px من كل الجهات
**Footer Height**: 48px للـ copyright

### 5. Visual Effects

#### Shadows & Depth

**Card Shadows**:

- **Default**: خفيف `0 2px 4px rgba(0,0,0,0.1)`
- **Hover**: متوسط `0 4px 12px rgba(0,0,0,0.15)`
- **Modal**: قوي `0 8px 32px rgba(0,0,0,0.3)`

#### Border Radius

**المطلوب**:

- **Cards**: 8px
- **Buttons**: 6px
- **Input Fields**: 4px
- **Badges**: 16px (pill shape)
- **Avatars**: 50% (دائري)

#### Animations & Transitions

**Hover Effects**:

- **Duration**: 200ms
- **Easing**: ease-out
- **Properties**: color, background, shadow, transform

**Page Transitions**:

- **Duration**: 300ms
- **Type**: Fade أو slide
- **Loading**: Skeleton screens

### 6. Brand Identity

#### Logo Requirements

**المطلوب للـ Designer**:

- **Logo Design**: شعار IDS-AI معبر عن الأمان السيبراني
- **Logo Variations**:
  - Full logo مع النص
  - Icon only للمساحات الضيقة
  - Monochrome version للخلفيات الملونة
- **Logo Placement**: أعلى يسار الواجهة
- **Logo Size**: 120px width للـ full logo

#### Brand Voice

**الطابع المطلوب**:

- **Professional**: مهني ومناسب للشركات
- **Trustworthy**: يوحي بالثقة والأمان
- **Modern**: عصري ومتطور
- **Alert**: يوحي باليقظة والانتباه

## Responsive Design Requirements

### 1. Breakpoints المطلوبة

#### Screen Sizes

**المطلوب للـ Designer**:

- **Mobile**: 320px - 768px
- **Tablet**: 768px - 1024px
- **Desktop**: 1024px - 1440px
- **Large Desktop**: 1440px+

#### Design Adaptations

**Mobile (< 768px)**:

- **Navigation**: Bottom tab bar بدلاً من sidebar
- **Cards**: Full width, stacked vertically
- **Tables**: Card layout بدلاً من table
- **Charts**: Simplified مع touch controls
- **Modals**: Full screen على الموبايل

**Tablet (768px - 1024px)**:

- **Navigation**: Collapsible sidebar
- **Grid**: 2-column layout للـ cards
- **Charts**: Medium size مع simplified legends
- **Touch Targets**: 44px minimum للـ touch

**Desktop (1024px+)**:

- **Navigation**: Full sidebar (240px)
- **Grid**: 3-4 column layout
- **Charts**: Full features مع tooltips
- **Multi-panel**: Side-by-side layouts

### 2. Component Responsive Behavior

#### Navigation Adaptation

**Mobile**:

- **Bottom Navigation**: 5 main tabs في الأسفل
- **Search**: Full screen overlay
- **User Menu**: Full screen modal

**Tablet**:

- **Collapsible Sidebar**: يفتح/يقفل بـ hamburger menu
- **Overlay**: Sidebar يفتح فوق المحتوى

**Desktop**:

- **Fixed Sidebar**: ثابت على الجانب
- **Breadcrumbs**: مسار التنقل أعلى الصفحة

#### Data Display Adaptation

**Mobile Tables → Cards**:

```
Desktop Table:
| Time | Source IP | Type | Severity | Actions |

Mobile Cards:
┌─────────────────────────────┐
│ 10:30 AM        [Critical]  │
│ Port Scan                   │
│ From: 192.168.1.100        │
│ [Block] [Investigate]       │
└─────────────────────────────┘
```

#### Chart Adaptations

**Mobile Charts**:

- **Height**: 200px maximum
- **Legend**: Bottom placement
- **Interactions**: Touch & swipe
- **Tooltips**: Full screen على tap

**Desktop Charts**:

- **Height**: 400px+
- **Legend**: Right side
- **Interactions**: Hover & click
- **Tooltips**: Contextual pop-ups

### 3. Touch & Gesture Requirements

#### Touch Targets

**المطلوب**:

- **Minimum Size**: 44px × 44px للـ touch targets
- **Spacing**: 8px minimum بين العناصر القابلة للنقر
- **Visual Feedback**: Immediate response للمس

#### Gestures

**المطلوب في التصميم**:

- **Swipe**: للتنقل بين التبويبات
- **Pull to Refresh**: لتحديث البيانات
- **Pinch to Zoom**: للخرائط والجرافات
- **Long Press**: لإظهار context menu

## Accessibility Requirements

### 1. Visual Accessibility

#### Color & Contrast

**المطلوب للـ Designer**:

- **Contrast Ratio**: 4.5:1 minimum للنص العادي
- **Large Text**: 3:1 minimum للنص الكبير (18px+)
- **Color Independence**: لا تعتمد على اللون فقط لنقل المعلومات
- **Status Indicators**: استخدم أيقونات مع الألوان

#### Text Accessibility

**المطلوب**:

- **Font Size**: 16px minimum للنص الأساسي
- **Line Height**: 1.5 minimum للقراءة السهلة
- **Text Spacing**: مسافات كافية بين الكلمات والأحرف
- **Contrast**: نص واضح على خلفيات متباينة

### 2. Keyboard Navigation

#### Focus Management

**المطلوب في التصميم**:

- **Visible Focus**: إطار واضح حول العنصر المحدد
- **Logical Tab Order**: ترتيب منطقي للتنقل
- **Skip Links**: روابط للتخطي للمحتوى الرئيسي
- **Focus Trap**: في الـ modals والـ dialogs

#### Keyboard Shortcuts

**المطلوب**:

- **Global Search**: Ctrl+K أو Cmd+K
- **Navigation**: Arrow keys للقوائم
- **Actions**: Enter للتأكيد، Escape للإلغاء
- **Close Modals**: Escape key

### 3. Screen Reader Support

#### Semantic HTML

**المطلوب للـ Developer**:

- **Headings**: استخدام H1, H2, H3 بترتيب منطقي
- **Landmarks**: main, nav, aside, footer
- **Lists**: ul, ol للقوائم
- **Tables**: proper table headers

#### ARIA Labels

**المطلوب**:

- **Button Labels**: وصف واضح لكل زر
- **Form Labels**: تسميات للحقول
- **Status Updates**: إعلام بالتغييرات
- **Live Regions**: للتحديثات الحية

### 4. Motor Accessibility

#### Large Touch Targets

**للأجهزة اللمسية**:

- **Minimum Size**: 44px × 44px
- **Spacing**: 8px بين العناصر
- **Click Areas**: مساحة كليك كبيرة

#### Alternative Input Methods

**المطلوب**:

- **Voice Control**: دعم voice commands
- **Switch Navigation**: للمستخدمين ذوي الإعاقات الحركية
- **Sticky Hover**: تجنب hover-only interactions

## Animation & Interaction Requirements

### 1. Animation Principles

#### Purpose-Driven Animation

**المطلوب للـ Designer**:

- **Feedback Animation**: تأكيد الإجراءات
- **Loading Animation**: مؤشرات التحميل
- **Transition Animation**: انتقالات سلسة بين الحالات
- **Attention Animation**: لفت الانتباه للتهديدات

#### Performance Guidelines

**المطلوب**:

- **Duration**: 150-300ms للتفاعلات السريعة
- **Easing**: Natural easing functions
- **60 FPS**: انتقالات سلسة
- **Reduced Motion**: احترام تفضيلات المستخدم

### 2. Micro-interactions

#### Button Interactions

**المطلوب**:

- **Hover State**: تغيير لون أو رفع
- **Active State**: ضغط بصري
- **Loading State**: spinner في الزر
- **Success State**: تأكيد نجاح العملية

#### Data Updates

**للبيانات الحية**:

- **New Data Highlight**: تمييز البيانات الجديدة
- **Smooth Transitions**: انتقالات سلسة للقيم
- **Pulse Animation**: للتنبيهات الحرجة
- **Count Animation**: عدادات متحركة

### 3. Loading States

#### Skeleton Screens

**المطلوب**:

- **Content Shape**: شكل يحاكي المحتوى الحقيقي
- **Animation**: تأثير shimmer أو pulse
- **Progressive Loading**: تحميل تدريجي
- **Placeholder Text**: نصوص مؤقتة

#### Progress Indicators

**المطلوب**:

- **Determinant**: progress bar مع نسبة مئوية
- **Indeterminant**: spinner للعمليات غير المحددة
- **Step Indicators**: للعمليات متعددة المراحل

## Theme Requirements

### 1. Dark/Light Theme Support

#### Theme Toggle

**المطلوب في التصميم**:

- **Toggle Switch**: في الـ header أو settings
- **System Preference**: اتباع إعدادات النظام
- **Smooth Transition**: انتقال سلس بين الثيمات
- **Persistence**: حفظ اختيار المستخدم

#### Dark Theme Adaptations

**المطلوب**:

- **Background Colors**: خلفيات داكنة مريحة للعين
- **Text Contrast**: نص واضح على الخلفيات الداكنة
- **Color Adjustments**: تعديل ألوان البراند للثيم الداكن
- **Image Handling**: معالجة الصور والأيقونات

### 2. Theme Consistency

#### Color Variables

**المطلوب للـ Developer**:

- **CSS Custom Properties**: متغيرات للألوان
- **Semantic Naming**: أسماء معبرة (--text-primary)
- **Theme Switching**: تغيير المتغيرات حسب الثيم

#### Component Adaptation

**المطلوب**:

- **Status Colors**: نفس المعنى في الثيمين
- **Readability**: وضوح في كل الثيمات
- **Brand Consistency**: الحفاظ على هوية البراند

## Technical Design Constraints

### 1. Performance Requirements

#### Loading Performance

**المطلوب**:

- **First Paint**: < 1.5 ثانية
- **Interactive**: < 3 ثواني
- **Image Optimization**: WebP format مع fallback
- **Code Splitting**: تحميل تدريجي للكود

#### Runtime Performance

**المطلوب**:

- **60 FPS**: animations سلسة
- **Memory Usage**: استخدام ذاكرة محدود
- **Real-time Updates**: < 100ms للبيانات الحية

### 2. Browser Support

#### Compatibility

**المطلوب**:

- **Modern Browsers**: Chrome, Firefox, Safari, Edge
- **Mobile Browsers**: iOS Safari, Chrome Mobile
- **Fallbacks**: للمتصفحات القديمة
- **Progressive Enhancement**: العمل بدون JavaScript

### 3. Development Constraints

#### Framework Requirements

**المطلوب معرفته للـ Designer**:

- **React Components**: كومبوننتات قابلة للإعادة
- **CSS Framework**: Tailwind CSS أو similar
- **Icon Library**: React Icons أو similar
- **Chart Library**: Chart.js أو D3.js

#### Design System

**للتسليم**:

- **Figma Components**: مكتبة كومبوننتات في Figma
- **Design Tokens**: قيم ثابتة للألوان والمسافات
- **Style Guide**: دليل الاستخدام
- **Prototypes**: نماذج تفاعلية

هذا الـ documentation شامل لكل ما يحتاجه الـ Designer لفهم المشروع والبدء في التصميم. كل قسم يوضح بالتفصيل ما هو مطلوب في كل جزء من النظام.

#### Status Indicators

```jsx
// Status Badge
<StatusBadge status="critical" size="sm">
  High Risk
</StatusBadge>

// Status Colors
- critical: Red background, white text
- high: Orange background, white text
- medium: Yellow background, dark text
- low: Blue background, white text
- safe: Green background, white text
- unknown: Gray background, white text

// Status Icons
<StatusIcon status="critical" />  // Red shield with X
<StatusIcon status="safe" />      // Green shield with check
```

#### Data Display Components

```jsx
// Metric Card
<MetricCard
  title="Active Threats"
  value="23"
  change="+5"
  trend="up"
  status="warning"
/>

// Data Table
<DataTable
  columns={threatColumns}
  data={threatData}
  sortable={true}
  filterable={true}
  pagination={true}
/>

// Progress Indicator
<ProgressBar
  value={75}
  max={100}
  status="warning"
  showValue={true}
/>
```

### Complex Components

#### Alert Card

```jsx
<AlertCard
  severity="high"
  timestamp="2024-10-09T10:30:00Z"
  title="Suspicious Port Scanning Detected"
  description="Multiple connection attempts from 192.168.1.100"
  source="192.168.1.100"
  destination="192.168.1.50"
  actions={[
    { label: "Investigate", action: "investigate" },
    { label: "Block IP", action: "block", variant: "danger" },
    { label: "Dismiss", action: "dismiss", variant: "ghost" },
  ]}
/>
```

#### Network Topology Viewer

```jsx
<NetworkTopology
  nodes={networkNodes}
  connections={networkConnections}
  threats={activeThreats}
  interactive={true}
  zoomable={true}
  filters={["internal", "external", "threats"]}
/>
```

#### Real-time Chart

```jsx
<RealtimeChart
  type="line"
  data={trafficData}
  timeRange="1h"
  metrics={["inbound", "outbound", "threats"]}
  colors={chartColors}
  interactive={true}
  annotations={threatAnnotations}
/>
```

## Layout & Navigation

### Primary Navigation

```jsx
// Sidebar Navigation
<Sidebar collapsed={false}>
  <NavItem icon="dashboard" label="Dashboard" active />
  <NavItem icon="shield" label="Threat Detection" />
  <NavItem icon="network" label="Network Analysis" />
  <NavItem icon="chart" label="Reports" />
  <NavItem icon="settings" label="System" />
</Sidebar>

// Navigation States
- active: Current page (blue background, white text)
- hover: Hover state (light blue background)
- default: Normal state (transparent)
```

### Header Layout

```jsx
<Header>
  <HeaderLeft>
    <Logo />
    <PageTitle />
  </HeaderLeft>

  <HeaderCenter>
    <GlobalSearch placeholder="Search threats, IPs, events..." />
  </HeaderCenter>

  <HeaderRight>
    <NotificationBell count={5} />
    <SystemStatus status="healthy" />
    <UserMenu user={currentUser} />
  </HeaderRight>
</Header>
```

### Content Layout Patterns

#### Dashboard Grid Layout

```css
.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 24px;
  padding: 24px;
}

.metric-cards {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px;
}

.main-chart {
  grid-column: 1 / 9;
  min-height: 400px;
}

.threat-feed {
  grid-column: 9 / -1;
  max-height: 600px;
  overflow-y: auto;
}
```

#### Full-width Analysis Layout

```css
.analysis-layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.analysis-header {
  flex: 0 0 auto;
  padding: 16px 24px;
  border-bottom: 1px solid var(--border-light);
}

.analysis-content {
  flex: 1 1 auto;
  display: flex;
  overflow: hidden;
}

.analysis-main {
  flex: 1 1 auto;
  padding: 24px;
  overflow-y: auto;
}

.analysis-sidebar {
  flex: 0 0 320px;
  border-left: 1px solid var(--border-light);
  padding: 24px;
  overflow-y: auto;
}
```

## Dashboard Design

### Overview Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Header: Logo | Search | Notifications | User Menu           │
├─────────────────────────────────────────────────────────────┤
│ Quick Metrics Cards                                         │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│ │ Active  │ │Blocked  │ │Network  │ │Critical │ │Response │ │
│ │Threats  │ │ IPs     │ │Traffic  │ │Alerts   │ │  Time   │ │
│ │   23    │ │   156   │ │ 2.3GB/s │ │    5    │ │  1.2s   │ │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Main Content Area                                           │
│ ┌─────────────────────────────────┐ ┌─────────────────────┐ │
│ │                                 │ │                     │ │
│ │     Real-time Threat Map        │ │    Live Alerts      │ │
│ │                                 │ │                     │ │
│ │  [Network Topology View]        │ │  • Port scan from   │ │
│ │                                 │ │    192.168.1.100    │ │
│ │                                 │ │  • DDoS attempt     │ │
│ │                                 │ │    detected         │ │
│ │                                 │ │  • Suspicious       │ │
│ │                                 │ │    traffic spike    │ │
│ └─────────────────────────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────┐ ┌─────────────────────┐ │
│ │                                 │ │                     │ │
│ │    Traffic Analysis Chart       │ │   System Health     │ │
│ │                                 │ │                     │ │
│ │   [Line chart showing traffic   │ │  ┌─ CPU: 45%        │ │
│ │    patterns, threat levels]     │ │  ├─ Memory: 62%     │ │
│ │                                 │ │  ├─ Storage: 78%    │ │
│ │                                 │ │  └─ Network: 89%    │ │
│ │                                 │ │                     │ │
│ └─────────────────────────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Metric Cards Design

```jsx
<MetricCard className="metric-card">
  <div className="metric-header">
    <Icon name="shield-alert" className="metric-icon danger" />
    <span className="metric-label">Active Threats</span>
  </div>

  <div className="metric-value">
    <span className="value">23</span>
    <TrendIndicator value="+5" period="last hour" />
  </div>

  <div className="metric-footer">
    <Sparkline data={hourlyThreatData} />
    <Link href="/threats" className="metric-link">
      View Details →
    </Link>
  </div>
</MetricCard>
```

## Data Visualization

### Chart Types & Usage

#### Real-time Line Charts

```jsx
// Network Traffic Chart
<LineChart
  data={trafficData}
  xAxis="timestamp"
  yAxis="bytes_per_second"
  series={[
    { key: "inbound", color: "#3b82f6", label: "Inbound Traffic" },
    { key: "outbound", color: "#10b981", label: "Outbound Traffic" },
    { key: "threats", color: "#ef4444", label: "Threat Events" },
  ]}
  realtime={true}
  updateInterval={1000}
  timeWindow="30m"
  thresholds={[
    { value: 1000000, label: "High Traffic", color: "orange" },
    { value: 5000000, label: "Critical", color: "red" },
  ]}
/>
```

#### Threat Distribution

```jsx
// Donut Chart for Attack Types
<DonutChart
  data={attackTypeData}
  centerLabel="Total Threats"
  centerValue={totalThreats}
  colors={threatColors}
  showPercentages={true}
  interactive={true}
  legend="bottom"
/>
```

#### Network Topology

```jsx
// Force-directed Graph
<NetworkGraph
  nodes={networkNodes}
  edges={networkConnections}
  nodeTypes={["server", "workstation", "router", "firewall"]}
  edgeTypes={["normal", "suspicious", "blocked"]}
  clustering={true}
  physics={true}
  filters={["internal", "external", "dmz"]}
  onNodeClick={handleNodeDetails}
  onEdgeClick={handleConnectionDetails}
/>
```

#### Heatmaps

```jsx
// Threat Activity Heatmap
<Heatmap
  data={threatHeatmapData}
  xAxis="hour_of_day"
  yAxis="day_of_week"
  colorScale="red"
  tooltip={threatTooltip}
  responsive={true}
/>
```

### Color Coding for Data

#### Threat Severity Colors

```css
.severity-critical {
  color: #b91c1c;
  background: #fef2f2;
}
.severity-high {
  color: #ea580c;
  background: #fff7ed;
}
.severity-medium {
  color: #ca8a04;
  background: #fffbeb;
}
.severity-low {
  color: #2563eb;
  background: #eff6ff;
}
.severity-info {
  color: #0891b2;
  background: #f0fdfa;
}
```

#### Network Status Colors

```css
.status-healthy {
  color: #059669;
}
.status-warning {
  color: #d97706;
}
.status-critical {
  color: #dc2626;
}
.status-offline {
  color: #6b7280;
}
.status-unknown {
  color: #8b5cf6;
}
```

## User Flows & Interactions

### Threat Investigation Flow

```
1. Alert Notification
   ↓
2. Alert Details View
   ├─ Basic Info Display
   ├─ Source/Destination Analysis
   └─ Related Events
   ↓
3. Investigation Actions
   ├─ Deep Packet Analysis
   ├─ Historical Correlation
   ├─ Threat Intelligence Lookup
   └─ Network Flow Analysis
   ↓
4. Response Actions
   ├─ Block IP/Port
   ├─ Create Firewall Rule
   ├─ Escalate to Team
   └─ Document Findings
   ↓
5. Resolution
   ├─ Mark as Resolved
   ├─ Add to Whitelist
   └─ Generate Report
```

### Interactive Elements

#### Alert Actions Menu

```jsx
<DropdownMenu trigger={<Button variant="ghost">Actions</Button>}>
  <MenuItem icon="search" onClick={investigate}>
    Investigate
  </MenuItem>
  <MenuItem icon="block" onClick={blockIP} variant="danger">
    Block IP Address
  </MenuItem>
  <MenuItem icon="flag" onClick={escalate}>
    Escalate to Team
  </MenuItem>
  <MenuItem icon="check" onClick={resolve}>
    Mark as Resolved
  </MenuItem>
  <MenuDivider />
  <MenuItem icon="download" onClick={exportData}>
    Export Details
  </MenuItem>
</DropdownMenu>
```

#### Search with Autocomplete

```jsx
<SearchInput
  placeholder="Search by IP, domain, alert ID..."
  suggestions={searchSuggestions}
  onSearch={handleSearch}
  filters={[
    { label: "IP Address", value: "ip" },
    { label: "Domain", value: "domain" },
    { label: "Alert ID", value: "alert_id" },
    { label: "Event Type", value: "event_type" },
  ]}
  recentSearches={recentSearches}
/>
```

### Modal & Dialog Patterns

#### Alert Detail Modal

```jsx
<Modal
  size="xl"
  title="Alert Details"
  subtitle="Suspicious Port Scanning Activity"
  onClose={closeModal}
>
  <ModalBody>
    <Tabs defaultTab="overview">
      <Tab id="overview" label="Overview">
        <AlertOverview alert={alertData} />
      </Tab>
      <Tab id="timeline" label="Timeline">
        <EventTimeline events={alertEvents} />
      </Tab>
      <Tab id="network" label="Network">
        <NetworkAnalysis source={alertData.source} />
      </Tab>
      <Tab id="response" label="Response">
        <ResponseActions alert={alertData} />
      </Tab>
    </Tabs>
  </ModalBody>

  <ModalFooter>
    <Button variant="ghost" onClick={closeModal}>
      Close
    </Button>
    <Button variant="danger" onClick={blockThreat}>
      Block Threat
    </Button>
    <Button variant="primary" onClick={investigateFurther}>
      Investigate Further
    </Button>
  </ModalFooter>
</Modal>
```

## Responsive Design

### Breakpoint System

```css
/* Mobile First Breakpoints */
--breakpoint-sm: 640px; /* Small devices */
--breakpoint-md: 768px; /* Medium devices */
--breakpoint-lg: 1024px; /* Large devices */
--breakpoint-xl: 1280px; /* Extra large */
--breakpoint-2xl: 1536px; /* 2X Extra large */
```

### Responsive Layout Patterns

#### Desktop Layout (1024px+)

- Full sidebar navigation (240px width)
- Multi-column dashboard grid
- Large data tables with all columns
- Detailed chart tooltips and legends

#### Tablet Layout (768px - 1023px)

- Collapsible sidebar navigation
- 2-column dashboard grid
- Scrollable data tables
- Simplified chart legends

#### Mobile Layout (< 768px)

- Bottom navigation bar
- Single-column layout
- Card-based data presentation
- Touch-optimized interactions

### Responsive Components

```jsx
// Responsive Navigation
<Navigation
  variant={screenSize >= 1024 ? 'sidebar' : 'bottom'}
  collapsed={screenSize < 768}
/>

// Responsive Data Table
<DataTable
  columns={screenSize >= 768 ? allColumns : mobileColumns}
  layout={screenSize >= 768 ? 'table' : 'cards'}
  pagination={screenSize >= 768 ? 'full' : 'simple'}
/>

// Responsive Chart
<Chart
  height={screenSize >= 768 ? 400 : 200}
  legend={screenSize >= 768 ? 'right' : 'bottom'}
  tooltip={screenSize >= 768 ? 'detailed' : 'minimal'}
/>
```

## Accessibility Guidelines

### WCAG 2.1 Compliance

#### Color & Contrast

- **AA Standard**: 4.5:1 contrast ratio for normal text
- **AAA Standard**: 7:1 contrast ratio for important text
- **Color Independence**: Never rely on color alone to convey information
- **Status Indicators**: Always include icons with color coding

#### Keyboard Navigation

```jsx
// Focus Management
<Button
  onKeyDown={handleKeyDown}
  tabIndex={0}
  aria-label="Investigate threat from IP 192.168.1.100"
>
  Investigate
</Button>

// Skip Links
<SkipLink href="#main-content">
  Skip to main content
</SkipLink>

// Focus Trap in Modals
<Modal trapFocus restoreFocus>
  {/* Modal content */}
</Modal>
```

#### Screen Reader Support

```jsx
// Semantic HTML
<main role="main" aria-labelledby="page-title">
  <h1 id="page-title">Security Dashboard</h1>
  {/* Content */}
</main>

// ARIA Labels
<button
  aria-label="Block IP address 192.168.1.100"
  aria-describedby="block-ip-description"
>
  Block IP
</button>

// Live Regions for Updates
<div
  aria-live="polite"
  aria-atomic="true"
  className="sr-only"
>
  {alertMessage}
</div>
```

#### Data Tables

```jsx
<Table
  role="table"
  aria-label="Active threats table"
  caption="List of currently active security threats"
>
  <thead>
    <tr role="row">
      <th scope="col" aria-sort="ascending">
        Timestamp
      </th>
      <th scope="col">Source IP</th>
      <th scope="col">Threat Type</th>
      <th scope="col">Severity</th>
    </tr>
  </thead>
  <tbody>
    {threats.map((threat) => (
      <tr key={threat.id} role="row">
        <td>{threat.timestamp}</td>
        <td>
          <code>{threat.sourceIp}</code>
        </td>
        <td>{threat.type}</td>
        <td>
          <StatusBadge
            status={threat.severity}
            aria-label={`Severity: ${threat.severity}`}
          >
            {threat.severity}
          </StatusBadge>
        </td>
      </tr>
    ))}
  </tbody>
</Table>
```

## Animation & Micro-interactions

### Animation Principles

- **Purposeful**: Animations should guide attention and provide feedback
- **Fast**: Duration between 150-300ms for most interactions
- **Smooth**: Use easing functions for natural movement
- **Respectful**: Respect user preferences for reduced motion

### Loading States

```css
/* Skeleton Loading */
.skeleton {
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: skeleton-loading 1.5s infinite;
}

@keyframes skeleton-loading {
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: -200% 0;
  }
}

/* Pulse for Real-time Data */
.pulse-danger {
  animation: pulse-red 2s infinite;
}

@keyframes pulse-red {
  0%,
  100% {
    box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
  }
  50% {
    box-shadow: 0 0 0 10px rgba(239, 68, 68, 0);
  }
}
```

### Hover Effects

```css
/* Button Hover */
.button {
  transition: all 0.2s ease;
}

.button:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

/* Card Hover */
.card {
  transition: all 0.3s ease;
}

.card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}
```

### Page Transitions

```jsx
// Route Transition
<PageTransition>
  <Routes>
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/threats" element={<Threats />} />
  </Routes>
</PageTransition>

// Staggered List Animation
<AnimatedList
  items={alerts}
  staggerDelay={50}
  animation="slideInUp"
/>
```

## Dark/Light Theme

### Theme Toggle Implementation

```jsx
<ThemeToggle
  currentTheme={theme}
  onThemeChange={setTheme}
  position="header"
/>

// Theme Provider
<ThemeProvider theme={theme}>
  <App />
</ThemeProvider>
```

### Theme-aware Components

```css
/* CSS Custom Properties for Theming */
[data-theme="light"] {
  --bg-primary: #ffffff;
  --text-primary: #0f172a;
  --border-color: #e2e8f0;
}

[data-theme="dark"] {
  --bg-primary: #0f172a;
  --text-primary: #f1f5f9;
  --border-color: #475569;
}

/* Component Styles */
.card {
  background-color: var(--bg-primary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
}
```

### Dark Theme Considerations

- **Reduced Brightness**: Lower overall luminance for eye comfort
- **Increased Contrast**: Higher contrast for critical elements
- **Color Adaptation**: Adjust brand colors for dark backgrounds
- **Status Colors**: Maintain semantic meaning in both themes

## Implementation Guidelines

### Code Organization

```
src/
├── components/
│   ├── ui/              # Base UI components
│   ├── charts/          # Data visualization
│   ├── forms/           # Form components
│   └── layout/          # Layout components
├── styles/
│   ├── globals.css      # Global styles
│   ├── components.css   # Component styles
│   └── themes.css       # Theme definitions
├── hooks/
│   ├── useTheme.js      # Theme management
│   ├── useRealtimeData.js # Real-time updates
│   └── useKeyboard.js   # Keyboard navigation
└── utils/
    ├── colors.js        # Color utilities
    ├── formatting.js    # Data formatting
    └── accessibility.js # A11y helpers
```

### Component Standards

```jsx
// Component Template
export const ComponentName = ({
  children,
  variant = "default",
  size = "md",
  className,
  ...props
}) => {
  const classes = cn("base-styles", variants[variant], sizes[size], className);

  return (
    <element className={classes} {...props}>
      {children}
    </element>
  );
};

// TypeScript Interface
interface ComponentProps {
  children?: React.ReactNode;
  variant?: "default" | "primary" | "danger";
  size?: "sm" | "md" | "lg";
  className?: string;
}
```

### Performance Guidelines

- **Lazy Loading**: Load components only when needed
- **Virtual Scrolling**: For large data lists
- **Memoization**: Prevent unnecessary re-renders
- **Code Splitting**: Split by routes and features
- **Image Optimization**: Compress and serve appropriate formats

### Testing Guidelines

```jsx
// Component Testing
test("renders alert with correct severity", () => {
  render(<Alert severity="high" message="Test alert" />);

  expect(screen.getByRole("alert")).toHaveClass("alert-high");
  expect(screen.getByText("Test alert")).toBeInTheDocument();
});

// Accessibility Testing
test("alert is accessible to screen readers", () => {
  render(<Alert severity="high" message="Security threat detected" />);

  expect(screen.getByRole("alert")).toHaveAttribute("aria-live", "polite");
  expect(screen.getByLabelText(/security threat/i)).toBeInTheDocument();
});
```

### Documentation Standards

- **Component Storybook**: Visual component documentation
- **Usage Examples**: Code examples for each component
- **Accessibility Notes**: A11y requirements and implementation
- **Design Tokens**: Document all design system values
- **Change Log**: Track component updates and breaking changes

This comprehensive UI/UX documentation provides your design and development teams with clear guidelines for creating a consistent, accessible, and professional interface for the IDS-AI system. The documentation covers everything from basic design tokens to complex interaction patterns, ensuring a cohesive user experience across all system components.
