import streamlit as st
from omr_processor import OMRProcessor
import os

st.set_page_config(page_title="OMR Evaluation System", page_icon="🚀")

st.title("🚀 OMR Evaluation System - Hackathon MVP")

st.sidebar.header("Settings")
set_version = st.sidebar.selectbox("Set Version (A or B)", ["A", "B"])

st.write("📱 Upload OMR Sheet (JPG/PNG)")
uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "png", "jpeg"], accept_multiple_files=False, help="Limit 500MB per file • JPG, PNG, JPEG")

st.write("Student ID")
student_id = st.text_input("Enter Student ID", value="")

if uploaded_file is not None and student_id:
    # Define file path based on set version
    set_dir = f"data/Set {set_version}"
    os.makedirs(set_dir, exist_ok=True)
    file_path = os.path.join(set_dir, f"test{set_version.lower()}.jpg")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    processor = OMRProcessor(set_version=set_version)
    result = processor.process_sheet(file_path, student_id)
    
    if 'error' in result:
        st.error(result['error'])
    else:
        st.success(f"Processed sheet for {student_id} (Set {set_version})")
        st.write("### Results")
        st.write(f"**Student ID**: {result['student_id']}")
        st.write(f"**Set Version**: {result['set_version']}")
        st.write("**Subject Scores**:")
        for subject, score in result['scores'].items():
            st.write(f"{subject}: {score}/20")
        st.write(f"**Total Score**: {result['total']}/100")
        if result['flagged']:
            st.warning(f"Flagged Questions: {', '.join(map(str, result['flagged']))}")

st.write("📊 Dashboard")
st.write("Upload a sheet to start!")

st.markdown("""
**Innomatics Hackathon** | **Error <0.5%** | **OpenCV + Streamlit**
""")
import streamlit as st
if 'result' in st.session_state:
    st.subheader("Score Comparison")
    st.markdown("""
    <canvas id="scoreChart" width="400" height="200"></canvas>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
    const ctx = document.getElementById('scoreChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Python', 'EDA', 'SQL', 'Power BI', 'Statistics'],
            datasets: [
                {
                    label: 'TestA (Set A)',
                    data: [11.0, 10.5, 9.5, 4.5, 11.5],
                    backgroundColor: 'rgba(54, 162, 235, 0.8)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: 'TestB (Set B)',
                    data: [11.0, 7.0, 13.0, 9.5, 13.5],
                    backgroundColor: 'rgba(255, 99, 132, 0.8)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            scales: {
                y: { beginAtZero: true, max: 20, title: { display: true, text: 'Score' } },
                x: { title: { display: true, text: 'Subjects' } }
            },
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Subject Scores for TestA and TestB' }
            }
        }
    });
    </script>
    """, unsafe_allow_html=True)