import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="AI-Powered Resume Screening",
    page_icon="📄",
    layout="wide"
)

# API configuration
API_BASE_URL = "http://localhost:8000"  # Update if your FastAPI runs elsewhere

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'current_job' not in st.session_state:
    st.session_state.current_job = None

# Helper functions
def make_api_request(endpoint, method="GET", data=None, requires_auth=True, form_data=False):
    """Make API request to backend."""
    headers = {}
    if requires_auth and st.session_state.access_token:
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"

    url = f"{API_BASE_URL}{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=data)
        elif method == "POST":
            if form_data:
                response = requests.post(url, headers=headers, data=data)
            else:
                response = requests.post(url, headers=headers, json=data)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            return {"error": f"Unsupported method: {method}"}

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

# Login/Register sidebar
with st.sidebar:
    st.title("🔐 Authentication")

    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("Login"):
                response = make_api_request(
                    "/auth/token",
                    method="POST",
                    data={"username": username, "password": password},
                    requires_auth=False,
                    form_data=True
                )

                if response and "access_token" in response:
                    st.session_state.logged_in = True
                    st.session_state.access_token = response["access_token"]
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Login failed")

        with tab2:
            reg_username = st.text_input("Username", key="reg_username")
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            is_admin = st.checkbox("Admin User", key="reg_admin")

            if st.button("Register"):
                response = make_api_request(
                    "/auth/register",
                    method="POST",
                    data={
                        "username": reg_username,
                        "email": reg_email,
                        "password": reg_password,
                        "is_admin": is_admin
                    },
                    requires_auth=False
                )

                if response:
                    st.success("Registration successful! Please login.")
    else:
        st.success(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.access_token = None
            st.session_state.username = None
            st.rerun()

# Main application
st.title("📄 AI-Powered Resume Screening & Candidate Ranking System")

if not st.session_state.logged_in:
    st.info("Please login or register to use the system.")
    st.stop()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "💼 Job Management",
    "📁 Resume Processing",
    "🎯 Candidate Ranking",
    "📈 Reports"
])

# Dashboard Tab
with tab1:
    st.header("Dashboard Overview")

    if st.button("Refresh Data", key="refresh_dashboard"):
        st.rerun()

    # Get jobs for stats
    jobs = make_api_request("/jobs", data={"active_only": True})

    if jobs:
        col1, col2, col3, col4 = st.columns(4)

        active_jobs = len(jobs)

        with col1:
            st.metric("Active Jobs", active_jobs)

        with col2:
            # Get total candidates across all jobs
            total_candidates = 0
            for job in jobs:
                stats = make_api_request(f"/jobs/{job['id']}/stats")
                if stats:
                    total_candidates += stats.get("total_candidates", 0)
            st.metric("Total Candidates", total_candidates)

        with col3:
            # Get total batches
            total_batches = 0
            for job in jobs:
                batches = make_api_request(f"/jobs/{job['id']}/batches")
                if batches:
                    total_batches += len(batches)
            st.metric("Total Batches", total_batches)

        with col4:
            # Get average score
            avg_scores = []
            for job in jobs:
                stats = make_api_request(f"/jobs/{job['id']}/stats")
                if stats and stats.get("average_score", 0) > 0:
                    avg_scores.append(stats["average_score"])

            overall_avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0
            st.metric("Average Score", f"{overall_avg:.2%}")

        # Recent activity
        st.subheader("Recent Activity")

        # Get recent batches
        recent_batches = []
        for job in jobs[:3]:  # Check first 3 jobs
            batches = make_api_request(f"/jobs/{job['id']}/batches", data={"limit": 2})
            if batches:
                for batch in batches:
                    recent_batches.append({
                        "Job": job["title"],
                        "Batch": batch["name"],
                        "Files": batch["file_count"],
                        "Status": batch["status"],
                        "Created": batch["created_at"][:10]
                    })

        if recent_batches:
            df_recent = pd.DataFrame(recent_batches)
            st.dataframe(df_recent, use_container_width=True)
        else:
            st.info("No recent activity found.")
    else:
        st.info("No active jobs found. Create a job to get started.")

# Job Management Tab
with tab2:
    st.header("Job Requirement Management")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Create New Job")

        with st.form("create_job_form"):
            job_title = st.text_input("Job Title*")
            job_description = st.text_area("Job Description")

            col_a, col_b = st.columns(2)
            with col_a:
                min_cgpa = st.number_input("Minimum CGPA", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
                min_experience = st.number_input("Minimum Experience (years)", min_value=0.0, value=0.0, step=0.5)

            with col_b:
                required_skills = st.text_area("Required Skills (comma-separated)")
                preferred_skills = st.text_area("Preferred Skills (comma-separated)")

            required_certs = st.text_area("Required Certifications (comma-separated)")
            education_reqs = st.text_area("Education Requirements (comma-separated)")

            submitted = st.form_submit_button("Create Job")

            if submitted and job_title:
                job_data = {
                    "title": job_title,
                    "description": job_description,
                    "min_cgpa": min_cgpa if min_cgpa > 0 else None,
                    "min_experience_years": min_experience if min_experience > 0 else None,
                    "required_skills": [s.strip() for s in required_skills.split(",") if s.strip()],
                    "preferred_skills": [s.strip() for s in preferred_skills.split(",") if s.strip()],
                    "required_certifications": [c.strip() for c in required_certs.split(",") if c.strip()],
                    "education_requirements": [e.strip() for e in education_reqs.split(",") if e.strip()]
                }

                response = make_api_request("/jobs", method="POST", data=job_data)
                if response:
                    st.success(f"Job '{job_title}' created successfully!")
                    st.rerun()

    with col2:
        st.subheader("Active Jobs")

        jobs = make_api_request("/jobs", data={"active_only": True})
        if jobs:
            for job in jobs:
                with st.expander(f"{job['title']}"):
                    st.write(f"**Description:** {job.get('description', 'No description')}")
                    st.write(f"**CGPA Requirement:** {job.get('min_cgpa', 'Not specified')}")
                    st.write(f"**Experience Requirement:** {job.get('min_experience_years', 'Not specified')} years")

                    col_x, col_y = st.columns(2)
                    with col_x:
                        if st.button("Select", key=f"select_{job['id']}"):
                            st.session_state.current_job = job
                            st.success(f"Selected job: {job['title']}")

                    with col_y:
                        if st.button("Delete", key=f"delete_{job['id']}"):
                            response = make_api_request(f"/jobs/{job['id']}", method="DELETE")
                            if response is None:  # DELETE returns 204 No Content
                                st.success(f"Job '{job['title']}' deactivated")
                                st.rerun()
        else:
            st.info("No jobs created yet.")

# Resume Processing Tab
with tab3:
    st.header("Resume Processing")

    # Select job first
    jobs = make_api_request("/jobs", data={"active_only": True})
    if not jobs:
        st.warning("Please create a job first before processing resumes.")
        st.stop()

    selected_job = st.selectbox(
        "Select Job for Processing",
        options=[job['title'] for job in jobs],
        key="job_select_resume"
    )

    current_job = next(job for job in jobs if job['title'] == selected_job)
    st.session_state.current_job = current_job

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Create Batch")

        batch_name = st.text_input("Batch Name", value=f"Batch_{datetime.now().strftime('%Y%m%d_%H%M')}")

        if st.button("Create Batch", key="create_batch"):
            batch_data = {
                "name": batch_name,
                "job_id": current_job["id"]
            }

            response = make_api_request("/resumes/batch", method="POST", data=batch_data)
            if response:
                st.session_state.current_batch = response
                st.success(f"Batch '{batch_name}' created!")

    with col2:
        st.subheader("Upload Resumes")

        if 'current_batch' not in st.session_state:
            st.info("Create a batch first to upload resumes.")
        else:
            uploaded_files = st.file_uploader(
                "Upload Resumes (PDF/DOCX)",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                key="file_uploader"
            )

            if uploaded_files and st.button("Upload Files", key="upload_files"):
                # Note: File upload would require proper handling with FastAPI
                st.info("File upload functionality requires backend integration.")
                st.info(f"Ready to upload {len(uploaded_files)} files to batch {st.session_state.current_batch['name']}")

    # Batch Management
    st.subheader("Manage Batches")

    batches = make_api_request(f"/jobs/{current_job['id']}/batches")
    if batches:
        for batch in batches:
            with st.expander(f"📦 {batch['name']} - {batch['status'].upper()}"):
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.write(f"**Files:** {batch['file_count']}")
                    st.write(f"**Created:** {batch['created_at'][:10]}")

                with col_b:
                    if batch['status'] == 'pending':
                        if st.button("Process", key=f"process_{batch['id']}"):
                            response = make_api_request(
                                f"/resumes/batch/{batch['id']}/process",
                                method="POST",
                                data={"anonymize": True, "score_immediately": True}
                            )
                            if response:
                                st.success("Batch processing started!")
                                st.rerun()

                with col_c:
                    if st.button("Delete", key=f"del_batch_{batch['id']}"):
                        response = make_api_request(f"/resumes/batch/{batch['id']}", method="DELETE")
                        if response is None:
                            st.success("Batch deleted!")
                            st.rerun()

# Candidate Ranking Tab
with tab4:
    st.header("Candidate Ranking & Scoring")

    if not st.session_state.current_job:
        st.warning("Please select a job first.")
        st.stop()

    current_job = st.session_state.current_job

    # Get candidates
    candidates = make_api_request(
        f"/jobs/{current_job['id']}/candidates",
        data={"limit": 50}
    )

    if candidates:
        st.subheader(f"Top Candidates for '{current_job['title']}'")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_score = st.slider("Minimum Score", 0.0, 1.0, 0.0, 0.1, key="min_score_filter")
        with col2:
            show_shortlisted = st.checkbox("Show Shortlisted Only", key="shortlisted_filter")
        with col3:
            export_format = st.selectbox("Export Format", ["JSON", "CSV", "Excel"], key="export_format")

        # Filter candidates
        filtered_candidates = [
            c for c in candidates
            if c['overall_score'] >= min_score and
            (not show_shortlisted or c['is_shortlisted'])
        ]

        # Display candidates
        for candidate in filtered_candidates[:20]:  # Show top 20
            with st.expander(f"🏆 Rank {candidate['rank']}: Score {candidate['overall_score']:.2%}"):
                col_a, col_b = st.columns(2)

                with col_a:
                    st.write(f"**Resume:** {candidate['resume']['filename']}")
                    st.write(f"**Experience:** {candidate['resume']['experience_years']} years")
                    st.write(f"**CGPA:** {candidate['resume']['cgpa'] or 'Not specified'}")
                    st.write(f"**Skills:** {', '.join(candidate['resume']['skills'][:5])}")

                    if candidate['is_shortlisted']:
                        st.success("✅ Shortlisted")
                    else:
                        if st.button("Shortlist", key=f"shortlist_{candidate['id']}"):
                            response = make_api_request(
                                f"/jobs/{current_job['id']}/candidates/{candidate['id']}/shortlist",
                                method="POST"
                            )
                            if response:
                                st.success("Candidate shortlisted!")
                                st.rerun()

                with col_b:
                    # Score breakdown visualization
                    scores = {
                        "Skills": candidate['skills_score'],
                        "Experience": candidate['experience_score'],
                        "Education": candidate['education_score'],
                        "Certifications": candidate['certification_score'],
                        "CGPA": candidate['cgpa_score']
                    }

                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(scores.keys()),
                            y=list(scores.values()),
                            marker_color='lightblue'
                        )
                    ])
                    fig.update_layout(
                        title="Score Breakdown",
                        yaxis_range=[0, 1],
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Export button
        if st.button(f"Export {len(filtered_candidates)} Candidates"):
            st.info(f"Export to {export_format} would be implemented here.")

    else:
        st.info("No candidates found for this job. Process some resumes first.")

# Reports Tab
with tab5:
    st.header("Reports & Analytics")

    if not st.session_state.current_job:
        st.warning("Please select a job first.")
        st.stop()

    current_job = st.session_state.current_job

    report_type = st.selectbox(
        "Select Report Type",
        ["Candidate Report", "Fairness Analysis", "Job Summary", "Full Export"]
    )

    if report_type == "Candidate Report":
        st.subheader("Candidate Explainability Report")

        candidates = make_api_request(f"/jobs/{current_job['id']}/candidates", data={"limit": 10})

        if candidates:
            selected_candidate = st.selectbox(
                "Select Candidate",
                options=[f"Rank {c['rank']}: {c['resume']['filename']} (Score: {c['overall_score']:.2%})"
                        for c in candidates],
                key="candidate_select"
            )

            candidate_idx = int(selected_candidate.split(":")[0].replace("Rank ", "")) - 1
            candidate = candidates[candidate_idx]

            # Display report
            col1, col2 = st.columns(2)

            with col1:
                st.write("### 📊 Score Breakdown")
                scores_df = pd.DataFrame({
                    "Category": ["Overall", "Skills", "Experience", "Education", "Certifications", "CGPA"],
                    "Score": [
                        candidate['overall_score'],
                        candidate['skills_score'],
                        candidate['experience_score'],
                        candidate['education_score'],
                        candidate['certification_score'],
                        candidate['cgpa_score']
                    ]
                })

                fig = px.bar(
                    scores_df,
                    x="Category",
                    y="Score",
                    title="Candidate Scores",
                    range_y=[0, 1]
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.write("### ✅ Strengths")
                for strength in candidate['strengths'][:5]:
                    st.success(f"• {strength}")

                st.write("### ⚠️ Areas for Improvement")
                for weakness in candidate['weaknesses'][:5]:
                    st.warning(f"• {weakness}")

                st.write("### 📋 Missing Requirements")
                for missing in candidate['missing_requirements'][:5]:
                    st.error(f"• {missing}")

    elif report_type == "Fairness Analysis":
        st.subheader("Fairness & Bias Analysis")

        if st.button("Generate Fairness Report"):
            with st.spinner("Analyzing fairness metrics..."):
                response = make_api_request(
                    f"/reports/job/{current_job['id']}/fairness",
                    method="POST"
                )

                if response:
                    st.success("Fairness report generated!")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Candidates", response['total_candidates'])

                    with col2:
                        st.metric("Potential Bias Cases", response['candidates_with_bias'])

                    with col3:
                        st.metric("Average Bias Score", f"{response['average_bias_score']:.2%}")

                    # Anonymization impact
                    st.write("### 🔍 Anonymization Impact")
                    impact = response['anonymization_impact']

                    impact_df = pd.DataFrame({
                        "Impact Type": ["Positive", "Negative", "Neutral"],
                        "Count": [
                            impact.get('positive_impact_count', 0),
                            impact.get('negative_impact_count', 0),
                            impact.get('no_impact_count', 0)
                        ]
                    })

                    fig = px.pie(
                        impact_df,
                        values="Count",
                        names="Impact Type",
                        title="Score Changes After Anonymization"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Job Summary":
        st.subheader("Job Summary Report")

        response = make_api_request(f"/reports/job/{current_job['id']}/summary")

        if response:
            col1, col2 = st.columns(2)

            with col1:
                st.write("### 📈 Statistics")
                stats = response['statistics']

                metrics_df = pd.DataFrame({
                    "Metric": ["Total Candidates", "Total Batches", "Average Score", "Shortlist Rate"],
                    "Value": [
                        stats['total_candidates'],
                        stats['total_batches'],
                        f"{stats['average_score']:.2%}",
                        f"{stats['shortlist_rate']:.2%}"
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            with col2:
                st.write("### 🎯 Score Distribution")

                if response['score_distribution']:
                    dist_df = pd.DataFrame(
                        list(response['score_distribution'].items()),
                        columns=["Score Range", "Count"]
                    )

                    fig = px.bar(
                        dist_df,
                        x="Score Range",
                        y="Count",
                        title="Candidate Score Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    elif report_type == "Full Export":
        st.subheader("Full Job Export")

        export_format = st.selectbox("Export Format", ["Excel", "JSON"], key="full_export_format")

        if st.button("Generate Full Report"):
            with st.spinner(f"Generating {export_format} report..."):
                response = make_api_request(
                    f"/reports/export/job/{current_job['id']}/full",
                    data={"format": export_format.lower()}
                )

                if response:
                    st.success(f"Report generated successfully!")
                    st.info("In a real implementation, this would download the file.")

# Footer
st.divider()
st.caption("AI-Powered Resume Screening System v1.0 • Built with FastAPI & Streamlit")
