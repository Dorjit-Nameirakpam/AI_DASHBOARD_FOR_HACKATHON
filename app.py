import streamlit as st
import cv2
import numpy as np
import time
import requests
from math import radians, sin, cos, sqrt, atan2
import tempfile
import os

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="SafeDrive AI - Neurodivergent Driver Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .alert-critical {
        background-color: #dc3545;
        color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #a71d2a;
        font-size: 20px;
        font-weight: bold;
        animation: pulse 1s infinite;
    }
    .alert-warning {
        background-color: #fd7e14;
        color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #dc6502;
        font-size: 18px;
        font-weight: bold;
    }
    .alert-safe {
        background-color: #28a745;
        color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1e7e34;
        font-size: 16px;
    }
    .rule-card {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .guideline-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
        padding: 20px;
        margin: 15px 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .traffic-signal-box {
        background: #2c3e50;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .signal-light {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        display: inline-block;
        margin: 10px auto;
        border: 4px solid #34495e;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
    }
    .signal-light.active {
        box-shadow: 0 0 30px currentColor, inset 0 0 20px currentColor;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0.3; }
    }
    .step-number {
        background: #667eea;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: inline-block;
        text-align: center;
        line-height: 30px;
        font-weight: bold;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DRIVING GUIDELINES DATABASE
# -----------------------------
DRIVING_GUIDELINES = {
    "traffic_signals": {
        "RED": {
            "meaning": "STOP - Do Not Move",
            "steps": [
                "Press brake pedal firmly",
                "Come to complete stop behind white line",
                "Keep foot on brake",
                "Wait and watch the signal",
                "Do NOT move until light turns GREEN"
            ],
            "duration": "Wait as long as needed",
            "warning": "Never run a red light - it's dangerous and illegal"
        },
        "YELLOW": {
            "meaning": "CAUTION - Prepare to Stop",
            "steps": [
                "Start slowing down immediately",
                "If very close to intersection, you may proceed carefully",
                "If far from intersection, prepare to stop",
                "Check mirrors for vehicles behind you",
                "Red light is coming next"
            ],
            "duration": "Usually 3-5 seconds",
            "warning": "Do not speed up to beat the light"
        },
        "GREEN": {
            "meaning": "GO - Proceed with Caution",
            "steps": [
                "Look left, right, then left again",
                "Check for pedestrians crossing",
                "Check for vehicles still in intersection",
                "If clear, gently press accelerator",
                "Proceed at safe speed"
            ],
            "duration": "Until light changes",
            "warning": "Green means you CAN go, not that you MUST go fast"
        }
    },
    "basic_rules": [
        {
            "category": "Lane Discipline",
            "icon": "🛣",
            "rules": [
                "Always stay in your lane - don't drift",
                "Use turn signals 3-5 seconds before changing lanes",
                "Check mirrors and blind spots before lane change",
                "Never cross solid white or yellow lines"
            ]
        },
        {
            "category": "Speed Management",
            "icon": "⚡",
            "rules": [
                "Follow posted speed limit signs",
                "Reduce speed in rain, fog, or bad weather",
                "Slow down in residential areas (schools, hospitals)",
                "Maintain steady speed - avoid sudden acceleration"
            ]
        },
        {
            "category": "Safe Distance",
            "icon": "📏",
            "rules": [
                "Keep 3-second following distance from vehicle ahead",
                "In rain: increase to 5-6 seconds",
                "On highway: maintain even more distance",
                "If someone tailgates you, let them pass safely"
            ]
        },
        {
            "category": "Intersections",
            "icon": "✖",
            "rules": [
                "Slow down when approaching intersections",
                "Look all directions even with green light",
                "Never block intersection - wait if traffic ahead",
                "Yield to vehicles already in roundabout"
            ]
        },
        {
            "category": "Pedestrian Safety",
            "icon": "🚶",
            "rules": [
                "Always stop for pedestrians at crosswalks",
                "Watch for people near bus stops",
                "Be extra careful near schools during drop-off/pick-up",
                "Give pedestrians right of way"
            ]
        },
        {
            "category": "Emergency Response",
            "icon": "🚨",
            "rules": [
                "Pull over for ambulance/police with sirens",
                "Move to rightmost lane safely",
                "Never block emergency vehicles",
                "Wait until they pass completely"
            ]
        }
    ],
    "sensory_tips": [
        {
            "title": "Managing Overwhelm",
            "tips": [
                "Take planned breaks every 30-45 minutes on long drives",
                "Use sunglasses to reduce glare",
                "Keep cabin temperature comfortable",
                "Play calming music at low volume (optional)"
            ]
        },
        {
            "title": "Focus Strategies",
            "tips": [
                "Scan ahead 10-12 seconds down the road",
                "Check mirrors every 5-8 seconds",
                "Narrate what you see (helps maintain focus)",
                "If feeling overwhelmed, find safe spot to pull over"
            ]
        },
        {
            "title": "Routine Building",
            "tips": [
                "Follow same route when possible for familiarity",
                "Drive during less busy times initially",
                "Practice in calm areas before busy roads",
                "Create pre-drive checklist (mirrors, seatbelt, signals)"
            ]
        }
    ]
}

# -----------------------------
# FUNCTIONS
# -----------------------------
def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    R = 6371.0
    lat1_r, lon1_r = radians(lat1), radians(lon1)
    lat2_r, lon2_r = radians(lat2), radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = sin(dlat/2)*2 + cos(lat1_r)*cos(lat2_r)*sin(dlon/2)*2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def detect_traffic_light(frame):
    """Enhanced traffic light detection with better accuracy."""
    # Take upper half of frame where signals are usually located
    height = frame.shape[0]
    roi = frame[0:height//2, :]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for traffic lights (optimized)
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])
    
    yellow_lower = np.array([15, 150, 150])
    yellow_upper = np.array([35, 255, 255])
    
    green_lower = np.array([35, 100, 100])
    green_upper = np.array([85, 255, 255])
    
    # Create masks
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5,5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
    # Count pixels
    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)
    
    # Determine dominant color with confidence threshold
    threshold = 150
    max_pixels = max(red_pixels, yellow_pixels, green_pixels)
    
    if max_pixels < threshold:
        return None
        
    if red_pixels == max_pixels:
        return "RED"
    elif yellow_pixels == max_pixels:
        return "YELLOW"
    elif green_pixels == max_pixels:
        return "GREEN"
    
    return None

def detect_objects(frame, prev_frame, focal_constant=800.0, min_area=800):
    """Improved motion detection with distance estimation."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if prev_frame is None:
        return [], gray

    diff = cv2.absdiff(prev_frame, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        assumed_obj_height_m = 1.6
        pixel_height = max(h, 1)
        distance_m = (focal_constant * assumed_obj_height_m) / pixel_height
        distance_m = float(min(max(distance_m, 0.5), 200.0))
        objects.append({'bbox': (x, y, w, h), 'distance_m': distance_m})
    return objects, gray

def get_weather_data(lat, lon, api_key="e9d2782525e73ec7d9283cb5f57052d0"):
    """Get weather from OpenWeatherMap."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

def check_alerts(objects, speed, traffic_light):
    """Return comprehensive alert based on objects, speed, and traffic signals."""
    # Traffic light has HIGHEST priority
    if traffic_light == "RED":
        return {
            'priority': 'CRITICAL',
            'message': '🛑 RED LIGHT DETECTED',
            'instruction': 'STOP IMMEDIATELY - Press brake pedal and wait for GREEN',
            'color': 'red'
        }
    elif traffic_light == "YELLOW":
        return {
            'priority': 'WARNING',
            'message': '🟡 YELLOW LIGHT DETECTED',
            'instruction': 'SLOW DOWN - Prepare to stop, red light coming next',
            'color': 'orange'
        }
    
    # Object proximity alerts
    if objects:
        min_distance_m = min(obj['distance_m'] for obj in objects)
        speed_m_s = max(0.1, speed / 3.6)
        ttc = min_distance_m / speed_m_s if speed_m_s > 0.2 else float('inf')
        
        if ttc < 1.5:
            return {
                'priority': 'CRITICAL',
                'message': '⚠ COLLISION WARNING',
                'instruction': f'BRAKE NOW - Vehicle only {min_distance_m:.1f}m ahead',
                'color': 'red'
            }
        elif ttc < 3.0:
            return {
                'priority': 'WARNING',
                'message': '⚠ TOO CLOSE',
                'instruction': f'Slow down - Maintain safe {min_distance_m:.1f}m distance',
                'color': 'orange'
            }
    
    # All clear
    if traffic_light == "GREEN":
        return {
            'priority': 'SAFE',
            'message': '✅ GREEN LIGHT - PROCEED',
            'instruction': 'Check surroundings, then drive at safe speed',
            'color': 'green'
        }
    
    return {
        'priority': 'SAFE',
        'message': '✅ ROAD CLEAR',
        'instruction': 'Drive safely, maintain speed and distance',
        'color': 'green'
    }

def get_signal_guidance(traffic_light):
    """Get detailed guidance for current traffic signal."""
    if traffic_light and traffic_light in DRIVING_GUIDELINES["traffic_signals"]:
        return DRIVING_GUIDELINES["traffic_signals"][traffic_light]
    return None

# -----------------------------
# SESSION STATE
# -----------------------------
if 'demo_running' not in st.session_state:
    st.session_state.demo_running = False
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'last_weather_update' not in st.session_state:
    st.session_state.last_weather_update = 0
if 'show_guidelines' not in st.session_state:
    st.session_state.show_guidelines = False

# -----------------------------
# SIDEBAR CONFIG
# -----------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/000000/car.png", width=80)
st.sidebar.title("🚗 SafeDrive AI")
st.sidebar.markdown("---")

# Pre-defined safe routes
SAFE_ROUTES = {
    "Lalbagh Botanical Garden Area": {
        "lat": 12.9507,
        "lon": 77.5848,
        "max_speed": 40,
        "description": "🌳 Low traffic, smooth roads, minimal potholes",
        "hazards": "Low risk - Peaceful area"
    },
    "Cubbon Park - MG Road": {
        "lat": 12.9762,
        "lon": 77.5929,
        "max_speed": 50,
        "description": "🏞 Wide roads, moderate traffic, well-maintained",
        "hazards": "Medium risk - Watch for pedestrians"
    },
    "Nice Road (Airport Highway)": {
        "lat": 13.1986,
        "lon": 77.7066,
        "max_speed": 80,
        "description": "🛣 Well-maintained highway, fast-moving traffic",
        "hazards": "⚠ HIGH RISK - Heavy traffic, high speeds, stay focused"
    },
    "Tumkur Road (NH-48)": {
        "lat": 13.0358,
        "lon": 77.5265,
        "max_speed": 80,
        "description": "🚛 Major highway, trucks and buses present",
        "hazards": "⚠ HIGH RISK - Large vehicles, potholes possible"
    },
    "Hosur Road (NH-44)": {
        "lat": 12.9099,
        "lon": 77.6388,
        "max_speed": 80,
        "description": "🏭 Industrial highway, connects to Electronic City",
        "hazards": "⚠ HIGH RISK - Very busy, multiple lanes"
    }
}

st.sidebar.header("📍 Route Selection")
selected_route = st.sidebar.selectbox(
    "Choose Your Route",
    options=list(SAFE_ROUTES.keys())
)

route_info = SAFE_ROUTES[selected_route]
st.sidebar.info(f"{route_info['description']}\n\n{route_info['hazards']}\n\n🚗 *Max Safe Speed:* {route_info['max_speed']} km/h")

st.sidebar.markdown("---")
st.sidebar.header("⚙ Settings")

speed = st.sidebar.slider("Current Speed (km/h)", 0, route_info['max_speed'], min(40, route_info['max_speed']))

latitude = route_info['lat']
longitude = route_info['lon']

video_mode = st.sidebar.radio("📹 Video Source", ["Demo Video", "Webcam"])
video_file = None
if video_mode == "Demo Video":
    video_file = st.sidebar.file_uploader("Upload video file", type=['mp4','avi','mov'])

st.sidebar.markdown("---")

# Guidelines toggle
if st.sidebar.button("📖 View Complete Driving Guidelines", use_container_width=True):
    st.session_state.show_guidelines = not st.session_state.show_guidelines

st.sidebar.markdown("---")
st.sidebar.header("🌤 Weather")
if st.sidebar.button("🔄 Update Weather"):
    now = time.time()
    if now - st.session_state.last_weather_update > 30:
        weather = get_weather_data(latitude, longitude)
        if weather:
            st.session_state.weather_data = weather
            st.session_state.last_weather_update = now
            st.sidebar.success("✅ Weather updated!")
        else:
            st.sidebar.error("❌ Could not fetch weather")
    else:
        st.sidebar.warning("⏳ Wait 30 seconds")

if st.session_state.weather_data:
    wd = st.session_state.weather_data
    st.sidebar.metric("Temperature", f"{wd['main']['temp']:.1f}°C")
    st.sidebar.write(f"*Conditions:* {wd['weather'][0]['description'].title()}")
    st.sidebar.write(f"*Humidity:* {wd['main']['humidity']}%")

# -----------------------------
# MAIN UI
# -----------------------------
st.markdown("<div class='main-header'><h1>🚗 SafeDrive AI Dashboard</h1><p>Neurodivergent-Friendly Driver Assistance System</p></div>", unsafe_allow_html=True)

# Show complete guidelines if toggled
if st.session_state.show_guidelines:
    st.markdown("## 📖 Complete Driving Guidelines")
    
    # Traffic Signals Section
    st.markdown("### 🚦 Traffic Signal Rules")
    sig_col1, sig_col2, sig_col3 = st.columns(3)
    
    for idx, (color, col) in enumerate(zip(["RED", "YELLOW", "GREEN"], [sig_col1, sig_col2, sig_col3])):
        guide = DRIVING_GUIDELINES["traffic_signals"][color]
        with col:
            st.markdown(f"""
            <div class='guideline-card'>
                <h3 style='color: {"#dc3545" if color=="RED" else "#ffc107" if color=="YELLOW" else "#28a745"};'>
                    {color} LIGHT
                </h3>
                <p><strong>{guide['meaning']}</strong></p>
                <p><em>Duration: {guide['duration']}</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("*What to do:*")
            for i, step in enumerate(guide['steps'], 1):
                st.markdown(f"{i}.** {step}")
            
            st.warning(f"⚠ {guide['warning']}")
    
    st.markdown("---")
    
    # Basic Rules Section
    st.markdown("### 📋 Essential Driving Rules")
    for rule_cat in DRIVING_GUIDELINES["basic_rules"]:
        with st.expander(f"{rule_cat['icon']} {rule_cat['category']}", expanded=False):
            for rule in rule_cat['rules']:
                st.markdown(f"✓ {rule}")
    
    st.markdown("---")
    
    # Sensory Tips Section
    st.markdown("### 🧠 Neurodivergent-Friendly Tips")
    for tip_section in DRIVING_GUIDELINES["sensory_tips"]:
        with st.expander(f"💡 {tip_section['title']}", expanded=False):
            for tip in tip_section['tips']:
                st.markdown(f"• {tip}")
    
    st.markdown("---")

# Control buttons
col_btn1, col_btn2, col_btn3 = st.columns([1,1,3])
with col_btn1:
    start = st.button("▶ Start System", use_container_width=True, type="primary")
with col_btn2:
    stop = st.button("⏹ Stop System", use_container_width=True)

if start:
    st.session_state.demo_running = True
if stop:
    st.session_state.demo_running = False

st.markdown("---")

# Main dashboard layout
col_video, col_right = st.columns([2, 1])

with col_video:
    st.subheader("📹 Live Camera Feed")
    video_placeholder = st.empty()

with col_right:
    st.subheader("🚦 Traffic Signal Status")
    traffic_signal_placeholder = st.empty()
    
    st.markdown("---")
    st.subheader("🚨 Current Alert")
    alert_placeholder = st.empty()

# Signal guidance section
st.markdown("---")
signal_guidance_placeholder = st.empty()

# Metrics row
st.markdown("---")
st.subheader("📊 Real-Time Metrics")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    speed_metric = st.empty()
with metric_col2:
    objects_metric = st.empty()
with metric_col3:
    distance_metric = st.empty()
with metric_col4:
    status_metric = st.empty()

# -----------------------------
# MAIN PROCESSING LOOP
# -----------------------------
prev_frame = None
if st.session_state.demo_running:
    cap = None
    temp_file = None
    
    try:
        if video_mode == "Webcam":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not open webcam")
                st.session_state.demo_running = False
        else:
            if video_file is None:
                st.warning("⚠ Upload a demo video first")
                st.session_state.demo_running = False
            else:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(video_file.read())
                temp_file.close()
                cap = cv2.VideoCapture(temp_file.name)
                
                if not cap.isOpened():
                    st.error("❌ Could not open video")
                    st.session_state.demo_running = False

        if cap and cap.isOpened():
            while st.session_state.demo_running:
                ret, frame = cap.read()
                if not ret:
                    st.info("✅ Video ended")
                    break
                    
                frame = cv2.resize(frame, (640, 360))

                # Detect traffic lights
                traffic_light = detect_traffic_light(frame)
                
                # Detect objects
                objects, prev_frame = detect_objects(frame, prev_frame)
                
                # Get alerts
                alert = check_alerts(objects, speed, traffic_light)
                
                # Get signal guidance
                guidance = get_signal_guidance(traffic_light)

                # Draw bounding boxes
                for obj in objects:
                    x, y, w, h = obj['bbox']
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(frame, f"{obj['distance_m']:.1f}m", (x,y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Display traffic light indicator on frame
                if traffic_light:
                    color_map = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0)}
                    cv2.circle(frame, (30, 30), 20, color_map[traffic_light], -1)
                    cv2.circle(frame, (30, 30), 20, (255, 255, 255), 3)
                    cv2.putText(frame, traffic_light, (60, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                # Update video feed
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Traffic signal visualization
                signal_html = """
                <div class='traffic-signal-box'>
                """
                
                for color in ["RED", "YELLOW", "GREEN"]:
                    is_active = (traffic_light == color)
                    color_hex = {"RED": "#dc3545", "YELLOW": "#ffc107", "GREEN": "#28a745"}[color]
                    opacity = "1.0" if is_active else "0.2"
                    
                    signal_html += f"""
                    <div style='margin: 10px 0;'>
                        <div class='signal-light {"active" if is_active else ""}' 
                             style='background-color: {color_hex}; opacity: {opacity};'></div>
                        <p style='color: white; margin-top: 5px; font-weight: {"bold" if is_active else "normal"};'>
                            {color}
                        </p>
                    </div>
                    """
                
                if not traffic_light:
                    signal_html += "<p style='color: #bbb; margin-top: 20px;'>No signal detected</p>"
                
                signal_html += "</div>"
                traffic_signal_placeholder.markdown(signal_html, unsafe_allow_html=True)

                # Display alert
                if alert:
                    alert_bg = {"red": "#dc3545", "orange": "#fd7e14", "green": "#28a745"}[alert['color']]
                    alert_placeholder.markdown(
                        f"""<div style='background-color:{alert_bg}; padding:20px; border-radius:10px; color:white;'>
                        <div style='font-size:22px; font-weight:bold; margin-bottom:10px;'>{alert['message']}</div>
                        <div style='font-size:16px;'>{alert['instruction']}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                # Display signal guidance
                if guidance:
                    guidance_html = f"""
                    <div class='guideline-card'>
                        <h3>🚦 {traffic_light} LIGHT - What To Do:</h3>
                        <p style='font-size:18px; font-weight:bold; margin:10px 0;'>{guidance['meaning']}</p>
                        <div style='margin-top:15px;'>
                    """
                    
                    for i, step in enumerate(guidance['steps'], 1):
                        guidance_html += f"""
                        <div style='margin: 10px 0;'>
                            <span class='step-number'>{i}</span>
                            <span style='font-size:16px;'>{step}</span>
                        </div>
                        """
                    
                    guidance_html += f"""
                        </div>
                        <div style='background:#fff3cd; padding:10px; border-radius:5px; margin-top:15px; color:#856404;'>
                            <strong>⚠ Remember:</strong> {guidance['warning']}
                        </div>
                        <p style='margin-top:10px; font-style:italic;'>Duration: {guidance['duration']}</p>
                    </div>
                    """
                    signal_guidance_placeholder.markdown(guidance_html, unsafe_allow_html=True)
                else:
                    signal_guidance_placeholder.empty()

                # Update metrics
                speed_metric.metric("🚗 Speed", f"{speed} km/h")
                objects_metric.metric("👁 Objects", len(objects))
                
                min_dist = min([obj['distance_m'] for obj in objects]) if objects else 0
                distance_metric.metric("📏 Nearest", f"{min_dist:.1f}m" if objects else "Clear")
                
                status_color = {"CRITICAL": "🔴", "WARNING": "🟠", "SAFE": "🟢"}
                status_metric.metric("Status", f"{status_color.get(alert['priority'], '🔵')} {alert['priority']}")

                time.sleep(0.03)
                
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    finally:
        if cap:
            cap.release()
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        st.session_state.demo_running = False
        st.success("✅ System stopped")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666; padding:20px;'>
    <p><strong>SafeDrive AI</strong> - Designed for neurodivergent drivers</p>
    <p style='font-size:12px;'>✓ Clear step-by-step instructions • ✓ Visual traffic signals • ✓ Real-time guidance</p>
    <p style='font-size:11px; margin-top:10px;'>Always follow local traffic laws. This system assists but does not replace driver responsibility.</p>
</div>
""", unsafe_allow_html=True)