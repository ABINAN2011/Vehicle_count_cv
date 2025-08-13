import streamlit as st
import tempfile
import pandas as pd
import plotly.graph_objs as go
from car_counter import vehicle_counter  


st.sidebar.title("Settings")

uploaded_video = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov'])
conf_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
frame_skip = st.sidebar.slider("Frame Skip (for speed)", 1, 10, 1)
start_processing = st.sidebar.button("Start Processing")


st.title(" Vehicle Counting System (YOLOv8 + SORT)")

if uploaded_video is None:
    st.info("Please upload a video file in the sidebar to start.")

if uploaded_video is not None and start_processing:
    
    temp_video_file = tempfile.NamedTemporaryFile(delete=False)
    temp_video_file.write(uploaded_video.read())
    temp_video_file.flush()

    output_video_path = "processed_output.mp4"


    video_placeholder = st.empty()
    chart_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    vehicle_counts = []


    for i, (frame, counts) in enumerate(vehicle_counter(
        temp_video_file.name,
        output_video_path,
        conf_threshold=conf_threshold,
        frame_skip=frame_skip
    )):

        video_placeholder.image(frame, channels="BGR")

        vehicle_counts = counts.copy()

 
        progress = i / max(len(vehicle_counts), 1)
        progress_bar.progress(min(progress, 1.0))


        if i % 5 == 0 or progress >= 1.0:
            df = pd.DataFrame({
                'Frame': list(range(len(vehicle_counts))),
                'Vehicle Count': vehicle_counts
            })

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Frame'],
                y=df['Vehicle Count'],
                mode='lines+markers',
                line=dict(color='blue')
            ))
            fig.update_layout(
                title='Vehicle Count Over Time',
                xaxis_title='Frame Number',
                yaxis_title='Count',
                height=400
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

        status_text.text(f"Processing frame {i + 1}")

    progress_bar.empty()
    status_text.success("✅ Video Processing Complete!")


    with open(output_video_path, 'rb') as vid_file:
        st.download_button(
            label=" Download Processed Video",
            data=vid_file,
            file_name="vehicle_count_output.mp4",
            mime="video/mp4"
        )

else:
    st.info("Configure the settings and click 'Start Processing' in the sidebar to begin.")
