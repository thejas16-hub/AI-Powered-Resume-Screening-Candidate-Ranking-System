from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    jobs = relationship("Job", back_populates="creator")


class Job(Base):
    """Job requirement definition model."""
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text)

    # Core requirements
    min_cgpa = Column(Float, nullable=True)
    min_experience_years = Column(Float, nullable=True)
    required_skills = Column(JSON, default=list)  # List of skills
    required_languages = Column(JSON, default=list)  # List of programming languages
    required_certifications = Column(JSON, default=list)  # List of certifications
    preferred_skills = Column(JSON, default=list)  # List of preferred skills
    education_requirements = Column(JSON, default=list)  # List of degree requirements

    # Job metadata
    is_active = Column(Boolean, default=True)
    creator_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    creator = relationship("User", back_populates="jobs")
    resume_batches = relationship("ResumeBatch", back_populates="job")
    candidate_scores = relationship("CandidateScore", back_populates="job")


class ResumeBatch(Base):
    """Batch of uploaded resumes for a specific job."""
    __tablename__ = "resume_batches"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    job_id = Column(Integer, ForeignKey("jobs.id"))
    upload_path = Column(String)  # Path to uploaded files
    file_count = Column(Integer, default=0)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    # Relationships
    job = relationship("Job", back_populates="resume_batches")
    resumes = relationship("Resume", back_populates="batch")


class Resume(Base):
    """Individual resume model after parsing."""
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("resume_batches.id"))
    filename = Column(String, nullable=False)
    original_path = Column(String)  # Original uploaded file path
    anonymized_path = Column(String)  # Path to anonymized version

    # Extracted data (anonymized)
    skills = Column(JSON, default=list)
    languages = Column(JSON, default=list)
    certifications = Column(JSON, default=list)
    education = Column(JSON, default=list)  # List of education entries
    experience_years = Column(Float, default=0)
    cgpa = Column(Float, nullable=True)

    # Processing metadata
    is_anonymized = Column(Boolean, default=False)
    parsed_at = Column(DateTime, nullable=True)
    anonymized_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    batch = relationship("ResumeBatch", back_populates="resumes")
    candidate_scores = relationship("CandidateScore", back_populates="resume")


class CandidateScore(Base):
    """Scoring results for each candidate per job."""
    __tablename__ = "candidate_scores"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"))
    resume_id = Column(Integer, ForeignKey("resumes.id"))

    # Score breakdown
    overall_score = Column(Float, default=0)
    skills_score = Column(Float, default=0)
    experience_score = Column(Float, default=0)
    education_score = Column(Float, default=0)
    certification_score = Column(Float, default=0)
    cgpa_score = Column(Float, default=0)

    # Semantic matching scores
    semantic_similarity = Column(Float, default=0)
    job_fit_confidence = Column(Float, default=0)

    # Rule-based matching
    missing_requirements = Column(JSON, default=list)
    matched_requirements = Column(JSON, default=list)
    strengths = Column(JSON, default=list)
    weaknesses = Column(JSON, default=list)

    # Ranking
    rank = Column(Integer, nullable=True)
    is_shortlisted = Column(Boolean, default=False)
    shortlisted_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    job = relationship("Job", back_populates="candidate_scores")
    resume = relationship("Resume", back_populates="candidate_scores")
    fairness_reports = relationship("FairnessReport", back_populates="candidate_score")


class FairnessReport(Base):
    """Fairness analysis reports for each candidate."""
    __tablename__ = "fairness_reports"

    id = Column(Integer, primary_key=True, index=True)
    candidate_score_id = Column(Integer, ForeignKey("candidate_scores.id"))

    # Bias detection metrics
    has_potential_bias = Column(Boolean, default=False)
    bias_types = Column(JSON, default=list)
    bias_score = Column(Float, default=0)

    # Anonymization impact
    score_before_anonymization = Column(Float, nullable=True)
    score_after_anonymization = Column(Float, nullable=True)
    score_delta = Column(Float, nullable=True)

    # Demographic parity (if available)
    demographic_group = Column(String, nullable=True)
    group_fairness_score = Column(Float, nullable=True)

    report_data = Column(JSON, default=dict)  # Detailed report data
    generated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    candidate_score = relationship("CandidateScore", back_populates="fairness_reports")
