#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Streamlit Frontend
Replicates all PyQt6 app functionalities using Streamlit and the Neptune API
"""

import streamlit as st
import requests
import cv2
import numpy as np
import time
import io
from pathlib import Path
from datetime import datetime
import base64
from PIL import Image
import os

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Page configuration
st.set_page_config(
    page_title="Neptune - Aquatic Surveillance",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (similar to PyQt6 styling)
st.markdown("""
<style>
    .main {
        background-color: #2b2b2b;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stat-box {
        background-color: #3b3b3b;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #555;
        color: white;
    }
    .danger-alert {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0.5; }
    }
    .underwater-alert {
        background-color: #ff9800;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    h1, h2, h3 {
        color: #00D4FF !important;
    }
    .stMetric {
        background-color: #3b3b3b;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def process_video_file(video_path, conf_threshold):
    """Process video through API"""
    try:
        with open(video_path, 'rb') as f:
            files = {'video': f}
            data = {
                'conf_threshold': conf_threshold,
                'track_persons': True,
                'detect_drowning': True,
                'analyze_water': True
            }
            response = requests.post(
                f"{API_BASE_URL}/video/upload",
                files=files,
                data=data,
                timeout=30
            )
            
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error uploading video: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def get_session_status(session_id):
    """Get video processing status"""
    try:
        response = requests.get(f"{API_BASE_URL}/video/session/{session_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_session_results(session_id):
    """Get video processing results"""
    try:
        response = requests.get(f"{API_BASE_URL}/video/session/{session_id}/results", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error getting results: {e}")
        return None


def render_video_player(video_path, results, conf_threshold, show_water, frame_idx):
    """Render video with detections overlay"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Cannot open video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set to specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return
    
    # Get detections for this frame
    frame_detections = results['detections_per_frame'].get(str(frame_idx), [])
    
    # Draw detections
    for det in frame_detections:
        x1 = int(det['center_x'] - det['width'] / 2)
        y1 = int(det['center_y'] - det['height'] / 2)
        x2 = int(det['center_x'] + det['width'] / 2)
        y2 = int(det['center_y'] + det['height'] / 2)
        
        # Color based on status
        if det['status'] == 'danger':
            color = (0, 0, 255)  # Red
        elif det['status'] == 'underwater':
            color = (255, 165, 0)  # Orange
        else:
            color = (0, 255, 0)  # Green
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw text
        label = f"ID:{det['track_id']} "
        if det['status'] == 'danger':
            label += f"DANGER {det['underwater_duration']:.1f}s"
        elif det['status'] == 'underwater':
            label += f"Underwater {det['frames_underwater']}"
        else:
            label += "Surface"
        
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw danger score
        score_text = f"Score: {det['dangerosity_score']}"
        cv2.putText(frame, score_text, (x1, y2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw water zone if requested
    if show_water and results.get('water_zone') and results['water_zone'].get('detected'):
        water_polygon = results['water_zone'].get('polygon')
        if water_polygon:
            pts = np.array(water_polygon, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
    
    # Convert BGR to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame_rgb, fps, total_frames


def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'frame_idx' not in st.session_state:
        st.session_state.frame_idx = 0
    if 'playing' not in st.session_state:
        st.session_state.playing = False
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    
    # Header
    st.title("🌊 Neptune - Aquatic Surveillance System")
    st.markdown("**Real-time person detection and drowning alert system**")
    
    # Check API status
    api_status = check_api_health()
    
    if not api_status:
        st.error("⚠️ API is not running! Please start the API server first.")
        st.code("cd api && ./start.sh", language="bash")
        st.stop()
    
    st.success("✅ API Connected")
    
    # Sidebar - Control Panel
    with st.sidebar:
        st.header("🎮 Control Panel")
        
        # File Section
        st.subheader("📁 Video File")
        
        video_source = st.radio("Video Source:", ["Upload File", "Enter Path"])
        
        if video_source == "Upload File":
            uploaded_file = st.file_uploader("Choose a video file", 
                                            type=['mp4', 'avi', 'mov', 'mkv'])
            if uploaded_file:
                # Save temporarily
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.session_state.video_path = temp_path
        else:
            video_path_input = st.text_input(
                "Video Path:",
                value="/home/achambaz/neptune/G-EIP-700-REN-7-1-eip-adrien.picot/app/video/rozel-15fps-fullhd.mp4",
                help="Enter full path to video file"
            )
            if Path(video_path_input).exists():
                st.session_state.video_path = video_path_input
            else:
                st.warning("File does not exist")
        
        # Load Video Button
        if st.button("🎬 Load Video", type="primary", disabled=not st.session_state.video_path):
            if st.session_state.video_path:
                st.session_state.processing = True
                st.session_state.results = None
                st.session_state.frame_idx = 0
        
        st.divider()
        
        # Configuration Section
        st.subheader("⚙️ Configuration")
        
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Detection confidence threshold"
        )
        
        underwater_threshold = st.number_input(
            "Underwater Frames",
            min_value=1,
            max_value=30,
            value=5,
            help="Frames to consider person underwater"
        )
        
        danger_threshold = st.number_input(
            "Danger Time (seconds)",
            min_value=1.0,
            max_value=30.0,
            value=5.0,
            step=0.5,
            help="Time underwater to trigger danger alert"
        )
        
        st.divider()
        
        # Display Options
        st.subheader("👁️ Display Options")
        
        show_water = st.checkbox("Show Water Detection", value=True)
        show_tracks = st.checkbox("Show Person Tracks", value=True)
        show_minimap = st.checkbox("Show Minimap", value=True)
        
        st.divider()
        
        # Stats Section
        st.subheader("📊 Statistics")
        if st.session_state.results:
            results = st.session_state.results
            st.metric("Total Frames", results['total_frames'])
            st.metric("Processed Frames", results['processed_frames'])
            st.metric("Alerts", len(results.get('alerts', [])))
            st.metric("Processing Time", f"{results['processing_time_seconds']:.2f}s")
    
    # Main Content Area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("📹 Video Display")
        
        # Processing status
        if st.session_state.processing and not st.session_state.results:
            with st.spinner("Processing video..."):
                # Upload and process
                session_data = process_video_file(
                    st.session_state.video_path,
                    conf_threshold
                )
                
                if session_data:
                    st.session_state.session_id = session_data['session_id']
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    while True:
                        status = get_session_status(st.session_state.session_id)
                        
                        if not status:
                            st.error("Failed to get status")
                            break
                        
                        progress = status['processed_frames'] / status['total_frames']
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {status['processed_frames']}/{status['total_frames']} frames")
                        
                        if status['status'] == 'completed':
                            st.session_state.results = get_session_results(st.session_state.session_id)
                            st.session_state.processing = False
                            st.success("✅ Processing complete!")
                            break
                        elif status['status'] == 'failed':
                            st.error("❌ Processing failed")
                            st.session_state.processing = False
                            break
                        
                        time.sleep(0.5)
                    
                    st.rerun()
        
        # Video player
        if st.session_state.results:
            results = st.session_state.results
            
            # Render current frame
            video_container = st.empty()
            
            frame_data = render_video_player(
                st.session_state.video_path,
                results,
                conf_threshold,
                show_water,
                st.session_state.frame_idx
            )
            
            if frame_data:
                frame_rgb, fps, total_frames = frame_data
                video_container.image(frame_rgb, use_container_width=True)
                
                # Playback controls
                st.subheader("⏯️ Playback Controls")
                
                col_play1, col_play2, col_play3, col_play4 = st.columns(4)
                
                with col_play1:
                    if st.button("⏮️ Start"):
                        st.session_state.frame_idx = 0
                        st.rerun()
                
                with col_play2:
                    if st.button("⏸️ Pause"):
                        st.session_state.playing = False
                
                with col_play3:
                    if st.button("▶️ Play"):
                        st.session_state.playing = True
                        st.rerun()
                
                with col_play4:
                    if st.button("⏭️ End"):
                        st.session_state.frame_idx = total_frames - 1
                        st.rerun()
                
                # Frame slider
                new_frame_idx = st.slider(
                    "Frame",
                    min_value=0,
                    max_value=total_frames - 1,
                    value=st.session_state.frame_idx,
                    key="frame_slider"
                )
                
                if new_frame_idx != st.session_state.frame_idx:
                    st.session_state.frame_idx = new_frame_idx
                    st.rerun()
                
                # Current time display
                current_time = st.session_state.frame_idx / fps
                duration = total_frames / fps
                st.text(f"Time: {current_time:.2f}s / {duration:.2f}s | Frame: {st.session_state.frame_idx}/{total_frames}")
        
        else:
            # Placeholder
            st.info("👆 Load a video to start surveillance")
            st.image("https://via.placeholder.com/800x450/2b2b2b/00D4FF?text=Neptune+Video+Display", 
                    use_container_width=True)
    
    with col2:
        st.header("🚨 Alerts")
        
        if st.session_state.results:
            alerts = st.session_state.results.get('alerts', [])
            
            if alerts:
                # Current frame alerts
                current_alerts = [a for a in alerts 
                                if abs(a['frame'] - st.session_state.frame_idx) < 30]
                
                if current_alerts:
                    for alert in current_alerts:
                        st.markdown(f"""
                        <div class="danger-alert">
                        🚨 DANGER ALERT<br>
                        Person ID: {alert['track_id']}<br>
                        Frame: {alert['frame']}<br>
                        Duration: {alert['duration']:.1f}s<br>
                        {alert['message']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.audio("data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvHZizYIGV+y6OqkUR0JU6rm8MOTUxATXr7w2J1UGBNbv+HqnlgaFFrB4O+4aSkjVsXr71NnKyNezuvo5qjf5qnU5qrt5uXo5O...") # Placeholder for alert sound
                
                # All alerts list
                st.subheader("Alert History")
                for alert in alerts[-10:]:  # Last 10 alerts
                    with st.expander(f"Frame {alert['frame']} - Person {alert['track_id']}"):
                        st.write(f"**Time:** {alert['timestamp']:.2f}s")
                        st.write(f"**Duration underwater:** {alert['duration']:.1f}s")
                        st.write(f"**Message:** {alert['message']}")
            else:
                st.success("✅ No danger alerts")
        
        else:
            st.info("No alerts yet")
        
        st.divider()
        
        # Current detections
        st.subheader("👥 Current Detections")
        
        if st.session_state.results:
            frame_detections = st.session_state.results['detections_per_frame'].get(
                str(st.session_state.frame_idx), []
            )
            
            if frame_detections:
                for det in frame_detections:
                    status_color = {
                        'surface': '🟢',
                        'underwater': '🟠',
                        'danger': '🔴'
                    }.get(det['status'], '⚪')
                    
                    with st.expander(f"{status_color} Person {det['track_id']}"):
                        st.write(f"**Status:** {det['status']}")
                        st.write(f"**Danger Score:** {det['dangerosity_score']}/100")
                        st.write(f"**Frames Underwater:** {det['frames_underwater']}")
                        st.write(f"**Duration:** {det['underwater_duration']:.1f}s")
                        st.write(f"**Distance from shore:** {det['distance_from_shore']:.2f}")
                        
                        # Progress bar for danger score
                        danger_pct = det['dangerosity_score'] / 100
                        st.progress(danger_pct)
            else:
                st.info("No persons detected in current frame")
        else:
            st.info("Load video to see detections")
    
    # Footer
    st.divider()
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.caption("🌊 Neptune Aquatic Surveillance")
    with col_f2:
        st.caption(f"API: {API_BASE_URL}")
    with col_f3:
        if api_status:
            st.caption("✅ Connected")
        else:
            st.caption("❌ Disconnected")


if __name__ == "__main__":
    main()
