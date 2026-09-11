import os
import sys
import tempfile

import cv2
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from main import create_pipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Surveillance",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SIMPLE CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 30px;
    font-weight: 600;
    margin-bottom: 5px;
}

.subtitle {
    color: #666;
    font-size: 16px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 21px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 12px;
}

.video-box {
    max-width: 700px;
    margin: 0 auto;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Smart Surveillance System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Hệ thống giám sát thông minh sử dụng Detection, Tracking, '
    'ReID, Motion, Anomaly Detection và AI Agent.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System")

    st.write("Các module đang sử dụng:")

    st.write("• Person Detection")
    st.write("• Object Tracking")
    st.write("• Person ReID")
    st.write("• Motion Segmentation")
    st.write("• Anomaly Detection")
    st.write("• AI Agent")

    st.divider()

    st.caption(
        "Upload video → Run System → Xem kết quả."
    )


# ============================================================
# INPUT VIDEO
# ============================================================

st.markdown(
    '<div class="section-title">1. Video đầu vào</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Chọn video cần phân tích",
    type=["mp4", "avi", "mov", "mkv"],
    label_visibility="collapsed"
)


if uploaded_file is None:

    st.info(
        "Chưa có video. Hãy chọn một file video để bắt đầu."
    )


else:

    st.success(
        f"Đã chọn: {uploaded_file.name}"
    )


    # ========================================================
    # SAVE TEMP VIDEO
    # ========================================================

    temp_input = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1]
    )

    temp_input.write(
        uploaded_file.getbuffer()
    )

    temp_input.close()


    # ========================================================
    # INPUT VIDEO
    # ========================================================

    st.markdown(
        '<div class="video-box">',
        unsafe_allow_html=True
    )

    st.video(uploaded_file)

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()


    # ========================================================
    # RUN BUTTON
    # ========================================================

    run_button = st.button(
        "RUN SMART SURVEILLANCE",
        type="primary",
        use_container_width=True
    )


    if run_button:

        # ====================================================
        # CREATE PIPELINE
        # ====================================================

        with st.spinner(
            "Đang khởi tạo hệ thống..."
        ):

            try:

                pipeline, agent = create_pipeline()

            except Exception as e:

                st.error(
                    f"Không thể khởi tạo hệ thống: {e}"
                )

                try:
                    os.unlink(temp_input.name)
                except Exception:
                    pass

                st.stop()


        st.success(
            "Hệ thống đã sẵn sàng."
        )


        # ====================================================
        # OPEN VIDEO
        # ====================================================

        cap = cv2.VideoCapture(
            temp_input.name
        )

        if not cap.isOpened():

            st.error(
                "Không thể mở video."
            )

            os.unlink(temp_input.name)

            st.stop()


        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25.0


        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )


        # ====================================================
        # VIDEO INFORMATION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '2. Thông tin video'
            '</div>',
            unsafe_allow_html=True
        )


        info1, info2, info3, info4 = st.columns(4)

        info1.metric(
            "Width",
            width
        )

        info2.metric(
            "Height",
            height
        )

        info3.metric(
            "FPS",
            f"{fps:.2f}"
        )

        info4.metric(
            "Frames",
            total_frames
        )


        st.divider()


        # ====================================================
        # OUTPUT SETUP
        # ====================================================

        output_dir = os.path.join(
            PROJECT_ROOT,
            "outputs"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )


        output_video = os.path.join(
            output_dir,
            "streamlit_smart_surveillance.mp4"
        )


        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )


        writer = cv2.VideoWriter(
            output_video,
            fourcc,
            fps,
            (width, height)
        )


        if not writer.isOpened():

            st.error(
                "Không thể tạo video output."
            )

            cap.release()

            os.unlink(temp_input.name)

            st.stop()


        # ====================================================
        # PROCESSING
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '3. Đang phân tích'
            '</div>',
            unsafe_allow_html=True
        )


        progress = st.progress(0)

        status_text = st.empty()

        frame_placeholder = st.empty()


        normal_count = 0
        suspicious_count = 0
        abnormal_count = 0
        critical_count = 0

        max_anomaly = 0.0

        frame_id = 0

        last_agent_result = None


        # ====================================================
        # PROCESS EACH FRAME
        # ====================================================

        while True:

            ret, frame = cap.read()

            if not ret:
                break


            try:

                # ------------------------------------------------
                # PIPELINE
                # ------------------------------------------------

                context = pipeline.process_frame(
                    frame,
                    frame_id
                )


                # ------------------------------------------------
                # AI AGENT
                # ------------------------------------------------

                agent_result = agent.analyze(

                    detections=len(
                        context.detections
                    ),

                    tracks=len(
                        context.tracks
                    ),

                    reid_count=len(
                        context.reid_features
                    ),

                    motion_count=len(
                        context.motion_regions
                    ),

                    anomaly_score=context.anomaly_score
                )


                last_agent_result = agent_result

                status = agent_result["status"]


                # ------------------------------------------------
                # COUNT STATUS
                # ------------------------------------------------

                if status == "NORMAL":

                    normal_count += 1

                elif status == "SUSPICIOUS":

                    suspicious_count += 1

                elif status == "ABNORMAL":

                    abnormal_count += 1

                elif status == "CRITICAL":

                    critical_count += 1


                # ------------------------------------------------
                # MAX ANOMALY
                # ------------------------------------------------

                if context.anomaly_score > max_anomaly:

                    max_anomaly = (
                        context.anomaly_score
                    )


                # ------------------------------------------------
                # OUTPUT FRAME
                # ------------------------------------------------

                output_frame = (
                    context.annotated_image
                )

                if output_frame is None:

                    output_frame = frame


                writer.write(
                    output_frame
                )


                # ------------------------------------------------
                # SHOW FRAME
                # ------------------------------------------------

                if frame_id % 10 == 0:

                    display_frame = cv2.cvtColor(
                        output_frame,
                        cv2.COLOR_BGR2RGB
                    )

                    frame_placeholder.image(
                        display_frame,
                        caption=(
                            f"Frame {frame_id} "
                            f"| AI Agent: {status}"
                        ),
                        width=700
                    )


                # ------------------------------------------------
                # PROGRESS
                # ------------------------------------------------

                if total_frames > 0:

                    percent = (
                        (frame_id + 1)
                        / total_frames
                    )

                    progress.progress(
                        min(percent, 1.0)
                    )


                status_text.text(
                    f"Đang xử lý: "
                    f"{frame_id + 1}/{total_frames}"
                )


            except Exception as e:

                st.error(
                    f"Lỗi tại frame "
                    f"{frame_id}: {e}"
                )

                break


            frame_id += 1


        # ====================================================
        # RELEASE
        # ====================================================

        cap.release()

        writer.release()


        try:

            os.unlink(
                temp_input.name
            )

        except Exception:

            pass


        progress.progress(1.0)

        status_text.success(
            f"Hoàn thành {frame_id} frame."
        )


        st.divider()


        # ====================================================
        # AI AGENT RESULT
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '4. AI Agent'
            '</div>',
            unsafe_allow_html=True
        )


        if last_agent_result is not None:

            final_status = (
                last_agent_result["status"]
            )

            final_action = (
                last_agent_result["action"]
            )

            reasons = (
                last_agent_result["reasons"]
            )


            result_col1, result_col2 = st.columns(2)


            with result_col1:

                st.metric(
                    "Status",
                    final_status
                )


            with result_col2:

                st.metric(
                    "Action",
                    final_action
                )


            if reasons:

                st.write(
                    "**Reason:** "
                    + ", ".join(reasons)
                )


        # ====================================================
        # STATISTICS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '5. Thống kê'
            '</div>',
            unsafe_allow_html=True
        )


        stat1, stat2, stat3, stat4 = st.columns(4)


        stat1.metric(
            "Normal",
            normal_count
        )

        stat2.metric(
            "Suspicious",
            suspicious_count
        )

        stat3.metric(
            "Abnormal",
            abnormal_count
        )

        stat4.metric(
            "Critical",
            critical_count
        )


        st.write(
            f"**Maximum Anomaly Score:** "
            f"{max_anomaly:.6f}"
        )


        st.divider()


        # ====================================================
        # RESULT VIDEO
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '6. Video kết quả'
            '</div>',
            unsafe_allow_html=True
        )


        if os.path.exists(output_video):

            st.success(
                "Video phân tích đã được tạo."
            )


            st.markdown(
                '<div class="video-box">',
                unsafe_allow_html=True
            )

            st.video(
                output_video
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


            with open(
                output_video,
                "rb"
            ) as video_file:

                st.download_button(

                    label="Download Result Video",

                    data=video_file.read(),

                    file_name=(
                        "smart_surveillance_result.mp4"
                    ),

                    mime="video/mp4",

                    use_container_width=True
                )


        else:

            st.warning(
                "Không tìm thấy video output."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Surveillance System | "
    "Detection • Tracking • ReID • Motion • "
    "Anomaly Detection • AI Agent"
)