"""Pydantic request and response schemas."""
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime


# Shared schemas
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: Optional[str] = None


# User schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr
    is_admin: Optional[bool] = False


class UserCreate(UserBase):
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Job schemas
class JobBase(BaseModel):
    title: str
    description: Optional[str] = None
    min_cgpa: Optional[float] = Field(None, ge=0.0, le=10.0)
    min_experience_years: Optional[float] = Field(None, ge=0.0)
    required_skills: List[str] = []
    required_languages: List[str] = []
    required_certifications: List[str] = []
    preferred_skills: List[str] = []
    education_requirements: List[str] = []


class JobCreate(JobBase):
    pass


class JobUpdate(JobBase):
    title: Optional[str] = None
    is_active: Optional[bool] = None


class Job(JobBase):
    id: int
    is_active: bool
    creator_id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Resume batch schemas
class ResumeBatchBase(BaseModel):
    name: str
    job_id: int


class ResumeBatchCreate(ResumeBatchBase):
    pass


class ResumeBatch(ResumeBatchBase):
    id: int
    upload_path: Optional[str] = None
    file_count: int
    status: str
    created_at: datetime
    processed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# Resume schemas
class ResumeBase(BaseModel):
    filename: str
    skills: List[str] = []
    languages: List[str] = []
    certifications: List[str] = []
    education: List[Dict[str, Any]] = []
    experience_years: float = 0.0
    cgpa: Optional[float] = None


class ResumeCreate(ResumeBase):
    batch_id: int
    original_path: str


class Resume(ResumeBase):
    id: int
    batch_id: int
    original_path: Optional[str] = None
    anonymized_path: Optional[str] = None
    is_anonymized: bool
    parsed_at: Optional[datetime] = None
    anonymized_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# Scoring schemas
class ScoreBreakdown(BaseModel):
    skills_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    certification_score: float = 0.0
    cgpa_score: float = 0.0
    semantic_similarity: float = 0.0
    job_fit_confidence: float = 0.0


class CandidateScoreBase(BaseModel):
    overall_score: float = 0.0
    rank: Optional[int] = None
    is_shortlisted: bool = False


class CandidateScoreCreate(CandidateScoreBase):
    job_id: int
    resume_id: int
    score_breakdown: ScoreBreakdown
    missing_requirements: List[str] = []
    matched_requirements: List[str] = []
    strengths: List[str] = []
    weaknesses: List[str] = []


class CandidateScore(CandidateScoreBase):
    id: int
    job_id: int
    resume_id: int
    missing_requirements: List[str] = []
    matched_requirements: List[str] = []
    strengths: List[str] = []
    weaknesses: List[str] = []
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CandidateScoreWithResume(CandidateScore):
    resume: Optional[Resume] = None


# Fairness report schemas
class FairnessReportBase(BaseModel):
    has_potential_bias: bool = False
    bias_types: List[str] = []
    bias_score: float = 0.0
    score_before_anonymization: Optional[float] = None
    score_after_anonymization: Optional[float] = None
    score_delta: Optional[float] = None
    demographic_group: Optional[str] = None
    group_fairness_score: Optional[float] = None


class FairnessReportCreate(FairnessReportBase):
    candidate_score_id: int
    report_data: Dict[str, Any] = {}


class FairnessReport(FairnessReportBase):
    id: int
    candidate_score_id: int
    report_data: Dict[str, Any] = {}
    generated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# API response schemas
class BatchProcessingResponse(BaseModel):
    batch_id: int
    total_files: int
    processed_files: int
    status: str
    estimated_completion_time: Optional[str] = None


class ScoringResponse(BaseModel):
    job_id: int
    candidate_count: int
    scoring_completed: bool
    top_candidates: List[CandidateScoreWithResume] = []
    average_score: float = 0.0


class FairnessAnalysisResponse(BaseModel):
    total_candidates: int
    candidates_with_bias: int
    average_bias_score: float
    anonymization_impact: Dict[str, float] = {}
    reports: List[FairnessReport] = []


# File upload schemas
class FileUploadResponse(BaseModel):
    filename: str
    size: int
    content_type: str
    upload_id: str


class BatchUploadResponse(BaseModel):
    batch_id: int
    total_files: int
    uploaded_files: List[FileUploadResponse] = []
    errors: List[Dict[str, str]] = []