import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import pandas as pd
import logging

from app.auth import get_current_active_user, get_current_admin_user
from app.database import get_db
from app.models import User, Job, ResumeBatch, CandidateScore, FairnessReport
from app.schemas import (
    CandidateScoreWithResume, FairnessReport as FairnessReportSchema,
    FairnessAnalysisResponse, ScoringResponse
)
from app.services.fairness import fairness_analyzer as fairness_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/job/{job_id}/candidates", response_model=List[CandidateScoreWithResume])
def get_candidate_report(
    job_id: int,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    min_score: Optional[float] = Query(None, ge=0, le=1, description="Minimum overall score"),
    format: str = Query("json", regex="^(json|csv|excel)$", description="Output format"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed candidate report for a job.
    """
    # Check if job exists
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    # Check if user can view this job
    if not job.is_active and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Job is not active"
        )

    # Build query
    query = (
        db.query(CandidateScore)
        .filter(CandidateScore.job_id == job_id)
        .join(CandidateScore.resume)
        .join(CandidateScore.job)
    )

    # Apply filters
    if min_score is not None:
        query = query.filter(CandidateScore.overall_score >= min_score)

    # Get candidate scores
    candidate_scores = (
        query.order_by(CandidateScore.rank.asc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    # Convert to response format
    results = []
    for score in candidate_scores:
        result = CandidateScoreWithResume(
            **score.__dict__,
            resume=score.resume
        )
        results.append(result)

    # Return in requested format
    if format == "json":
        return results
    elif format in ["csv", "excel"]:
        return export_candidate_report(results, format, job.title)


def export_candidate_report(
    candidates: List[CandidateScoreWithResume],
    format: str,
    job_title: str
):
    """
    Export candidate report to CSV or Excel.
    """
    # Prepare data for export
    data = []
    for candidate in candidates:
        row = {
            "rank": candidate.rank,
            "overall_score": candidate.overall_score,
            "skills_score": candidate.skills_score,
            "experience_score": candidate.experience_score,
            "education_score": candidate.education_score,
            "certification_score": candidate.certification_score,
            "cgpa_score": candidate.cgpa_score,
            "semantic_similarity": candidate.semantic_similarity,
            "job_fit_confidence": candidate.job_fit_confidence,
            "is_shortlisted": candidate.is_shortlisted,
            "filename": candidate.resume.filename if candidate.resume else "",
            "experience_years": candidate.resume.experience_years if candidate.resume else 0,
            "cgpa": candidate.resume.cgpa if candidate.resume else None,
            "skills_count": len(candidate.resume.skills) if candidate.resume else 0,
            "certifications_count": len(candidate.resume.certifications) if candidate.resume else 0,
        }
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as tmp_file:
        tmp_path = tmp_file.name

    # Export based on format
    if format == "csv":
        df.to_csv(tmp_path, index=False)
        content_type = "text/csv"
        filename = f"candidate_report_{job_title.replace(' ', '_')}.csv"
    else:  # excel
        df.to_excel(tmp_path, index=False, sheet_name="Candidates")
        content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"candidate_report_{job_title.replace(' ', '_')}.xlsx"

    return FileResponse(
        path=tmp_path,
        media_type=content_type,
        filename=filename
    )


@router.get("/candidate/{candidate_id}/explainability")
def get_candidate_explainability_report(
    candidate_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get explainability report for a specific candidate.
    """
    # Get candidate score
    candidate_score = db.query(CandidateScore).filter(CandidateScore.id == candidate_id).first()
    if not candidate_score:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate score not found"
        )

    # Get job and check access
    job = db.query(Job).filter(Job.id == candidate_score.job_id).first()
    if not job.is_active and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Job is not active"
        )

    # Get resume
    resume = candidate_score.resume

    # Prepare explainability report
    report = {
        "candidate_info": {
            "candidate_id": candidate_id,
            "rank": candidate_score.rank,
            "overall_score": candidate_score.overall_score,
            "is_shortlisted": candidate_score.is_shortlisted,
            "resume_filename": resume.filename if resume else "Unknown"
        },
        "score_breakdown": {
            "skills_score": candidate_score.skills_score,
            "experience_score": candidate_score.experience_score,
            "education_score": candidate_score.education_score,
            "certification_score": candidate_score.certification_score,
            "cgpa_score": candidate_score.cgpa_score,
            "semantic_similarity": candidate_score.semantic_similarity,
            "job_fit_confidence": candidate_score.job_fit_confidence
        },
        "strengths": candidate_score.strengths,
        "weaknesses": candidate_score.weaknesses,
        "matched_requirements": candidate_score.matched_requirements,
        "missing_requirements": candidate_score.missing_requirements,
        "job_requirements": {
            "title": job.title,
            "min_cgpa": job.min_cgpa,
            "min_experience_years": job.min_experience_years,
            "required_skills": job.required_skills,
            "preferred_skills": job.preferred_skills,
            "required_certifications": job.required_certifications,
            "education_requirements": job.education_requirements
        },
        "candidate_qualifications": {
            "experience_years": resume.experience_years if resume else 0,
            "cgpa": resume.cgpa if resume else None,
            "skills": resume.skills if resume else [],
            "certifications": resume.certifications if resume else [],
            "languages": resume.languages if resume else [],
            "education": resume.education if resume else []
        },
        "interpretation": generate_score_interpretation(candidate_score)
    }

    return report


def generate_score_interpretation(candidate_score: CandidateScore) -> Dict[str, Any]:
    """
    Generate human-readable interpretation of the score.
    """
    overall_score = candidate_score.overall_score

    if overall_score >= 0.8:
        rating = "Excellent Fit"
        description = "Candidate strongly matches job requirements with high scores across all dimensions."
    elif overall_score >= 0.6:
        rating = "Good Fit"
        description = "Candidate matches most job requirements with solid qualifications."
    elif overall_score >= 0.4:
        rating = "Moderate Fit"
        description = "Candidate has some matching qualifications but may lack certain requirements."
    else:
        rating = "Poor Fit"
        description = "Candidate has limited match with job requirements."

    # Key factors analysis
    key_factors = []
    if candidate_score.skills_score >= 0.8:
        key_factors.append("Strong skills match")
    if candidate_score.experience_score >= 0.8:
        key_factors.append("Sufficient or above required experience")
    if candidate_score.semantic_similarity >= 0.7:
        key_factors.append("High semantic alignment with job description")

    if candidate_score.missing_requirements:
        improvement_areas = candidate_score.missing_requirements[:3]  # Top 3 areas
    else:
        improvement_areas = []

    return {
        "rating": rating,
        "description": description,
        "key_factors": key_factors,
        "improvement_areas": improvement_areas,
        "recommendation": "Consider for interview" if overall_score >= 0.6 else "Review carefully"
    }


@router.post("/job/{job_id}/fairness", response_model=FairnessAnalysisResponse)
def generate_fairness_report(
    job_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Generate fairness analysis report for a job.
    """
    # Check if job exists
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    # Get all candidate scores for this job
    candidate_scores = (
        db.query(CandidateScore)
        .filter(CandidateScore.job_id == job_id)
        .all()
    )

    if not candidate_scores:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No candidate scores found for this job"
        )

    # Get resume texts (simplified - in reality would need actual text storage)
    resume_texts_before = {}
    resume_texts_after = {}

    for score in candidate_scores:
        resume = score.resume
        if resume:
            # Simplified: using filename as placeholder for text
            # In real implementation, would extract text from original/anonymized files
            resume_texts_before[resume.id] = f"Resume: {resume.filename}"
            resume_texts_after[resume.id] = f"Resume: {resume.filename} (Anonymized)"

    # Perform fairness analysis
    try:
        analysis_results = fairness_analyzer.analyze_candidate_fairness(
            candidate_scores,
            resume_texts_before,
            resume_texts_after
        )

        # Generate and save fairness reports for each candidate
        fairness_reports = []
        for score in candidate_scores:
            report_data = fairness_analyzer.generate_fairness_report(
                analysis_results,
                score.id
            )

            # Save to database
            db_report = FairnessReport(
                candidate_score_id=score.id,
                has_potential_bias=report_data.has_potential_bias,
                bias_types=report_data.bias_types,
                bias_score=report_data.bias_score,
                score_before_anonymization=report_data.score_before_anonymization,
                score_after_anonymization=report_data.score_after_anonymization,
                score_delta=report_data.score_delta,
                demographic_group=report_data.demographic_group,
                group_fairness_score=report_data.group_fairness_score,
                report_data=report_data.report_data
            )
            db.add(db_report)
            fairness_reports.append(db_report)

        db.commit()

        # Count candidates with bias
        candidates_with_bias = sum(1 for report in fairness_reports if report.has_potential_bias)

        # Calculate average bias score
        bias_scores = [report.bias_score for report in fairness_reports if report.bias_score is not None]
        avg_bias_score = sum(bias_scores) / len(bias_scores) if bias_scores else 0.0

        # Calculate anonymization impact
        anonymization_deltas = [
            report.score_delta for report in fairness_reports
            if report.score_delta is not None
        ]
        anonymization_impact = {
            "average_delta": sum(anonymization_deltas) / len(anonymization_deltas) if anonymization_deltas else 0.0,
            "positive_impact_count": sum(1 for delta in anonymization_deltas if delta > 0),
            "negative_impact_count": sum(1 for delta in anonymization_deltas if delta < 0),
            "no_impact_count": sum(1 for delta in anonymization_deltas if delta == 0)
        }

        return FairnessAnalysisResponse(
            total_candidates=len(candidate_scores),
            candidates_with_bias=candidates_with_bias,
            average_bias_score=avg_bias_score,
            anonymization_impact=anonymization_impact,
            reports=[FairnessReportSchema(**report.__dict__) for report in fairness_reports]
        )

    except Exception as e:
        logger.error(f"Failed to generate fairness report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate fairness report: {e}"
        )


@router.get("/fairness/{report_id}", response_model=FairnessReportSchema)
def get_fairness_report(
    report_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific fairness report.
    """
    report = db.query(FairnessReport).filter(FairnessReport.id == report_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fairness report not found"
        )

    # Check if user can access this report
    candidate_score = report.candidate_score
    if candidate_score:
        job = candidate_score.job
        if not job.is_active and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Job is not active"
            )

    return report


@router.get("/job/{job_id}/summary")
def get_job_summary_report(
    job_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get summary report for a job.
    """
    # Check if job exists
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    # Check if user can view this job
    if not job.is_active and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Job is not active"
        )

    # Get statistics
    candidate_count = (
        db.query(CandidateScore)
        .filter(CandidateScore.job_id == job_id)
        .count()
    )

    batch_count = (
        db.query(ResumeBatch)
        .filter(ResumeBatch.job_id == job_id)
        .count()
    )

    avg_score_result = (
        db.query(db.func.avg(CandidateScore.overall_score))
        .filter(CandidateScore.job_id == job_id)
        .scalar()
    )
    avg_score = float(avg_score_result) if avg_score_result else 0.0

    shortlisted_count = (
        db.query(CandidateScore)
        .filter(
            CandidateScore.job_id == job_id,
            CandidateScore.is_shortlisted == True
        )
        .count()
    )

    # Get score distribution
    score_distribution = {}
    for i in range(0, 11):
        min_score = i / 10
        max_score = (i + 1) / 10
        count = (
            db.query(CandidateScore)
            .filter(
                CandidateScore.job_id == job_id,
                CandidateScore.overall_score >= min_score,
                CandidateScore.overall_score < max_score
            )
            .count()
        )
        if count > 0:
            score_distribution[f"{min_score:.1f}-{max_score:.1f}"] = count

    # Get top skills from candidates
    all_skills = []
    candidate_scores = (
        db.query(CandidateScore)
        .filter(CandidateScore.job_id == job_id)
        .join(CandidateScore.resume)
        .limit(100)
        .all()
    )

    for score in candidate_scores:
        if score.resume and score.resume.skills:
            all_skills.extend(score.resume.skills)

    # Count skill frequency
    skill_counts = {}
    for skill in all_skills:
        skill_counts[skill] = skill_counts.get(skill, 0) + 1

    top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "job_info": {
            "id": job.id,
            "title": job.title,
            "description": job.description,
            "is_active": job.is_active
        },
        "statistics": {
            "total_candidates": candidate_count,
            "total_batches": batch_count,
            "average_score": avg_score,
            "shortlisted_candidates": shortlisted_count,
            "shortlist_rate": shortlisted_count / max(candidate_count, 1)
        },
        "score_distribution": score_distribution,
        "top_candidate_skills": [{"skill": skill, "count": count} for skill, count in top_skills],
        "requirements_summary": {
            "min_cgpa": job.min_cgpa,
            "min_experience": job.min_experience_years,
            "required_skills_count": len(job.required_skills),
            "preferred_skills_count": len(job.preferred_skills),
            "certification_requirements": len(job.required_certifications)
        }
    }


@router.get("/export/job/{job_id}/full")
def export_full_job_report(
    job_id: int,
    format: str = Query("excel", regex="^(excel|json)$", description="Export format"),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Export full job report including candidates, scores, and fairness analysis.
    """
    # Check if job exists
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    # Get all data
    candidate_scores = (
        db.query(CandidateScore)
        .filter(CandidateScore.job_id == job_id)
        .join(CandidateScore.resume)
        .order_by(CandidateScore.rank.asc())
        .all()
    )

    batches = db.query(ResumeBatch).filter(ResumeBatch.job_id == job_id).all()

    # Prepare comprehensive report
    report_data = {
        "job_info": {
            "id": job.id,
            "title": job.title,
            "description": job.description,
            "requirements": {
                "min_cgpa": job.min_cgpa,
                "min_experience_years": job.min_experience_years,
                "required_skills": job.required_skills,
                "preferred_skills": job.preferred_skills,
                "required_certifications": job.required_certifications,
                "education_requirements": job.education_requirements
            }
        },
        "batches": [
            {
                "id": batch.id,
                "name": batch.name,
                "file_count": batch.file_count,
                "status": batch.status,
                "created_at": batch.created_at.isoformat() if batch.created_at else None
            }
            for batch in batches
        ],
        "candidates": [
            {
                "rank": score.rank,
                "overall_score": score.overall_score,
                "score_breakdown": {
                    "skills": score.skills_score,
                    "experience": score.experience_score,
                    "education": score.education_score,
                    "certifications": score.certification_score,
                    "cgpa": score.cgpa_score,
                    "semantic_similarity": score.semantic_similarity,
                    "job_fit_confidence": score.job_fit_confidence
                },
                "is_shortlisted": score.is_shortlisted,
                "resume_info": {
                    "filename": score.resume.filename if score.resume else None,
                    "experience_years": score.resume.experience_years if score.resume else None,
                    "cgpa": score.resume.cgpa if score.resume else None,
                    "skills": score.resume.skills if score.resume else [],
                    "certifications": score.resume.certifications if score.resume else []
                },
                "strengths": score.strengths,
                "weaknesses": score.weaknesses,
                "missing_requirements": score.missing_requirements,
                "matched_requirements": score.matched_requirements
            }
            for score in candidate_scores
        ],
        "summary_statistics": {
            "total_candidates": len(candidate_scores),
            "average_score": sum(score.overall_score for score in candidate_scores) / max(len(candidate_scores), 1),
            "shortlisted_count": sum(1 for score in candidate_scores if score.is_shortlisted),
            "score_distribution": {}
        }
    }

    # Add score distribution
    for i in range(0, 11):
        min_score = i / 10
        max_score = (i + 1) / 10
        count = sum(1 for score in candidate_scores if min_score <= score.overall_score < max_score)
        if count > 0:
            report_data["summary_statistics"]["score_distribution"][f"{min_score:.1f}-{max_score:.1f}"] = count

    # Export based on format
    if format == "json":
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as tmp_file:
            json.dump(report_data, tmp_file, indent=2, default=str)
            tmp_path = tmp_file.name

        return FileResponse(
            path=tmp_path,
            media_type="application/json",
            filename=f"job_{job_id}_full_report.json"
        )
    else:  # excel
        # Create Excel with multiple sheets
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_path = tmp_file.name

        with pd.ExcelWriter(tmp_path, engine='openpyxl') as writer:
            # Job info sheet
            job_info_data = {
                "Field": ["Title", "Description", "Min CGPA", "Min Experience (years)"],
                "Value": [
                    job.title,
                    job.description or "",
                    job.min_cgpa or "N/A",
                    job.min_experience_years or "N/A"
                ]
            }
            pd.DataFrame(job_info_data).to_excel(writer, sheet_name="Job Info", index=False)

            # Candidates sheet
            candidates_data = []
            for score in candidate_scores:
                candidates_data.append({
                    "Rank": score.rank,
                    "Overall Score": score.overall_score,
                    "Skills Score": score.skills_score,
                    "Experience Score": score.experience_score,
                    "Education Score": score.education_score,
                    "Certification Score": score.certification_score,
                    "CGPA Score": score.cgpa_score,
                    "Semantic Similarity": score.semantic_similarity,
                    "Job Fit Confidence": score.job_fit_confidence,
                    "Shortlisted": "Yes" if score.is_shortlisted else "No",
                    "Resume File": score.resume.filename if score.resume else "",
                    "Experience Years": score.resume.experience_years if score.resume else 0,
                    "CGPA": score.resume.cgpa if score.resume else None
                })
            pd.DataFrame(candidates_data).to_excel(writer, sheet_name="Candidates", index=False)

            # Requirements sheet
            requirements_data = {
                "Requirement Type": ["Required Skills", "Preferred Skills", "Required Certifications", "Education Requirements"],
                "Items": [
                    ", ".join(job.required_skills),
                    ", ".join(job.preferred_skills),
                    ", ".join(job.required_certifications),
                    ", ".join(job.education_requirements)
                ]
            }
            pd.DataFrame(requirements_data).to_excel(writer, sheet_name="Requirements", index=False)

        return FileResponse(
            path=tmp_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"job_{job_id}_full_report.xlsx"
        )
