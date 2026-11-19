"""Candidate scoring and ranking with semantic matching."""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import logging

from app.models import Job, Resume, CandidateScore
from app.schemas import ScoreBreakdown

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Weights for different scoring components."""
    skills: float = 0.3
    experience: float = 0.25
    education: float = 0.2
    certifications: float = 0.15
    cgpa: float = 0.1
    semantic_similarity: float = 0.3  # Applied separately


class CandidateScorer:
    """Scoring engine for matching candidates to job requirements."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the scoring engine with Sentence-BERT model.

        Args:
            model_name: Name of the Sentence-BERT model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded Sentence-BERT model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load Sentence-BERT model: {e}")
            raise

        # Default scoring weights
        self.weights = ScoringWeights()

        # Skill normalization factors
        self.skill_importance = {
            'required': 1.0,
            'preferred': 0.6,
            'bonus': 0.3
        }

    def prepare_job_embeddings(self, job: Job) -> Dict[str, Any]:
        """
        Prepare job embeddings for semantic matching.

        Returns:
            Dictionary containing job embeddings and metadata
        """
        # Create text representations for different job aspects
        job_texts = {
            'title': job.title,
            'description': job.description or "",
            'required_skills': " ".join(job.required_skills),
            'preferred_skills': " ".join(job.preferred_skills),
            'requirements': f"CGPA: {job.min_cgpa or 'N/A'}, "
                          f"Experience: {job.min_experience_years or 'N/A'} years, "
                          f"Education: {', '.join(job.education_requirements)}"
        }

        # Combine all job text for overall embedding
        combined_job_text = (
            f"Job Title: {job_texts['title']}. "
            f"Description: {job_texts['description']}. "
            f"Required Skills: {job_texts['required_skills']}. "
            f"Preferred Skills: {job_texts['preferred_skills']}. "
            f"Requirements: {job_texts['requirements']}"
        )

        # Generate embeddings
        embeddings = {}
        for key, text in job_texts.items():
            if text.strip():
                embeddings[key] = self.model.encode(text)

        embeddings['combined'] = self.model.encode(combined_job_text)

        return {
            'embeddings': embeddings,
            'texts': job_texts,
            'job_data': {
                'min_cgpa': job.min_cgpa,
                'min_experience': job.min_experience_years,
                'required_skills': job.required_skills,
                'preferred_skills': job.preferred_skills,
                'education_requirements': job.education_requirements,
                'required_certifications': job.required_certifications,
                'required_languages': job.required_languages
            }
        }

    def prepare_resume_embeddings(self, resume: Resume) -> Dict[str, Any]:
        """
        Prepare resume embeddings for semantic matching.

        Returns:
            Dictionary containing resume embeddings and metadata
        """
        # Create text representations for different resume aspects
        resume_texts = {
            'skills': " ".join(resume.skills),
            'languages': " ".join(resume.languages),
            'certifications': " ".join(resume.certifications),
            'education': self._format_education_text(resume.education),
            'experience': f"{resume.experience_years} years of experience",
            'summary': self._create_resume_summary(resume)
        }

        # Combine all resume text for overall embedding
        combined_resume_text = (
            f"Skills: {resume_texts['skills']}. "
            f"Languages: {resume_texts['languages']}. "
            f"Certifications: {resume_texts['certifications']}. "
            f"Education: {resume_texts['education']}. "
            f"Experience: {resume_texts['experience']}"
        )

        # Generate embeddings
        embeddings = {}
        for key, text in resume_texts.items():
            if text.strip():
                embeddings[key] = self.model.encode(text)

        embeddings['combined'] = self.model.encode(combined_resume_text)

        return {
            'embeddings': embeddings,
            'texts': resume_texts,
            'resume_data': {
                'skills': resume.skills,
                'languages': resume.languages,
                'certifications': resume.certifications,
                'education': resume.education,
                'experience_years': resume.experience_years,
                'cgpa': resume.cgpa
            }
        }

    def _format_education_text(self, education: List[Dict[str, Any]]) -> str:
        """Format education data into text."""
        if not education:
            return ""

        education_texts = []
        for edu in education:
            parts = []
            if edu.get('degree_type'):
                parts.append(edu['degree_type'])
            if edu.get('field'):
                parts.append(f"in {edu['field']}")
            if edu.get('university'):
                parts.append(f"from {edu['university']}")
            if edu.get('cgpa'):
                parts.append(f"CGPA: {edu['cgpa']}")
            if parts:
                education_texts.append(" ".join(parts))

        return ". ".join(education_texts)

    def _create_resume_summary(self, resume: Resume) -> str:
        """Create a comprehensive summary of the resume."""
        summary_parts = []

        if resume.skills:
            summary_parts.append(f"Skills: {', '.join(resume.skills[:5])}")

        if resume.experience_years > 0:
            summary_parts.append(f"{resume.experience_years} years of experience")

        if resume.education:
            # Get highest education
            highest_edu = max(resume.education, key=lambda x: self._get_education_level(x.get('degree_type', '')))
            edu_text = highest_edu.get('degree_type', '')
            if highest_edu.get('field'):
                edu_text += f" in {highest_edu['field']}"
            summary_parts.append(edu_text)

        if resume.cgpa:
            summary_parts.append(f"CGPA: {resume.cgpa}")

        return ". ".join(summary_parts)

    def _get_education_level(self, degree_type: str) -> int:
        """Get education level for ranking."""
        degree_type = degree_type.lower()
        if 'phd' in degree_type:
            return 5
        elif 'master' in degree_type:
            return 4
        elif 'bachelor' in degree_type:
            return 3
        elif 'diploma' in degree_type:
            return 2
        else:
            return 1

    def calculate_rule_based_scores(
        self,
        job_data: Dict[str, Any],
        resume_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate rule-based matching scores.

        Returns:
            Dictionary with individual scores and matched/missing items
        """
        scores = {
            'skills_score': 0.0,
            'experience_score': 0.0,
            'education_score': 0.0,
            'certification_score': 0.0,
            'cgpa_score': 0.0,
            'missing_requirements': [],
            'matched_requirements': [],
            'strengths': [],
            'weaknesses': []
        }

        # 1. Skills matching
        required_skills = set(job_data['required_skills'])
        preferred_skills = set(job_data['preferred_skills'])
        candidate_skills = set(resume_data['skills'])

        # Match required skills
        matched_required = required_skills.intersection(candidate_skills)
        required_match_ratio = len(matched_required) / max(len(required_skills), 1)

        # Match preferred skills
        matched_preferred = preferred_skills.intersection(candidate_skills)
        preferred_match_ratio = len(matched_preferred) / max(len(preferred_skills), 1)

        # Calculate skills score (weighted combination)
        scores['skills_score'] = (
            required_match_ratio * self.skill_importance['required'] +
            preferred_match_ratio * self.skill_importance['preferred']
        )

        # Track matched and missing skills
        if matched_required:
            scores['matched_requirements'].extend([
                f"Required skill: {skill}" for skill in matched_required
            ])
        if matched_preferred:
            scores['strengths'].extend([
                f"Preferred skill: {skill}" for skill in matched_preferred
            ])
        missing_required = required_skills - candidate_skills
        if missing_required:
            scores['missing_requirements'].extend([
                f"Missing required skill: {skill}" for skill in missing_required
            ])

        # 2. Experience matching
        min_experience = job_data['min_experience']
        candidate_experience = resume_data['experience_years']

        if min_experience is not None:
            if candidate_experience >= min_experience:
                scores['experience_score'] = 1.0
                scores['matched_requirements'].append(
                    f"Experience: {candidate_experience} years (meets minimum {min_experience} years)"
                )
            else:
                experience_ratio = candidate_experience / max(min_experience, 1)
                scores['experience_score'] = min(experience_ratio, 1.0)
                scores['weaknesses'].append(
                    f"Insufficient experience: {candidate_experience} years "
                    f"(minimum {min_experience} years required)"
                )
        else:
            # No experience requirement specified
            scores['experience_score'] = 0.5  # Neutral score

        # 3. Education matching
        education_requirements = set(job_data['education_requirements'])
        candidate_education = set()

        # Extract degree types from candidate education
        for edu in resume_data['education']:
            if degree_type := edu.get('degree_type'):
                candidate_education.add(degree_type.lower())

        if education_requirements:
            # Check if candidate has any of the required education levels
            candidate_levels = [self._get_education_level(deg) for deg in candidate_education]
            if candidate_levels:
                max_candidate_level = max(candidate_levels)

                # Check requirements
                requirement_satisfied = False
                for req in education_requirements:
                    req_level = self._get_education_level(req)
                    if max_candidate_level >= req_level:
                        requirement_satisfied = True
                        scores['matched_requirements'].append(
                            f"Education: meets requirement for {req}"
                        )
                        break

                if requirement_satisfied:
                    scores['education_score'] = 1.0
                else:
                    scores['education_score'] = 0.3  # Partial credit for having education
                    scores['missing_requirements'].append(
                        f"Missing education requirement: {', '.join(education_requirements)}"
                    )
        else:
            # No specific education requirement
            scores['education_score'] = 0.5  # Neutral score

        # 4. Certification matching
        required_certs = set(job_data['required_certifications'])
        candidate_certs = set(resume_data['certifications'])

        if required_certs:
            matched_certs = required_certs.intersection(candidate_certs)
            cert_match_ratio = len(matched_certs) / max(len(required_certs), 1)

            scores['certification_score'] = cert_match_ratio

            if matched_certs:
                scores['matched_requirements'].extend([
                    f"Certification: {cert}" for cert in matched_certs
                ])
            missing_certs = required_certs - candidate_certs
            if missing_certs:
                scores['missing_requirements'].extend([
                    f"Missing certification: {cert}" for cert in missing_certs
                ])
        else:
            # No certification requirements
            scores['certification_score'] = 0.5  # Neutral score

        # 5. CGPA matching
        min_cgpa = job_data['min_cgpa']
        candidate_cgpa = resume_data['cgpa']

        if min_cgpa is not None and candidate_cgpa is not None:
            if candidate_cgpa >= min_cgpa:
                scores['cgpa_score'] = 1.0
                scores['matched_requirements'].append(
                    f"CGPA: {candidate_cgpa} (meets minimum {min_cgpa})"
                )
            else:
                cgpa_ratio = candidate_cgpa / max(min_cgpa, 1)
                scores['cgpa_score'] = min(cgpa_ratio, 1.0)
                scores['weaknesses'].append(
                    f"CGPA below requirement: {candidate_cgpa} "
                    f"(minimum {min_cgpa} required)"
                )
        elif candidate_cgpa is not None:
            # No CGPA requirement, but candidate has CGPA
            scores['cgpa_score'] = candidate_cgpa / 10.0  # Normalize to 0-1 scale
        else:
            # No CGPA information
            scores['cgpa_score'] = 0.0

        return scores

    def calculate_semantic_similarity(
        self,
        job_embeddings: Dict[str, Any],
        resume_embeddings: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate semantic similarity scores using cosine similarity.

        Returns:
            Dictionary with similarity scores for different aspects
        """
        similarities = {}

        # Compare each aspect
        aspects = ['combined', 'skills', 'description', 'requirements']
        for aspect in aspects:
            if (aspect in job_embeddings['embeddings'] and
                aspect in resume_embeddings['embeddings']):
                job_emb = job_embeddings['embeddings'][aspect].reshape(1, -1)
                resume_emb = resume_embeddings['embeddings'][aspect].reshape(1, -1)

                similarity = cosine_similarity(job_emb, resume_emb)[0][0]
                similarities[f"{aspect}_similarity"] = float(similarity)

        # Overall similarity (weighted average)
        if similarities:
            # Give more weight to combined and skills similarities
            weights = {
                'combined_similarity': 0.4,
                'skills_similarity': 0.3,
                'description_similarity': 0.2,
                'requirements_similarity': 0.1
            }

            overall_similarity = 0.0
            total_weight = 0.0

            for key, similarity in similarities.items():
                weight = weights.get(key, 0.1)
                overall_similarity += similarity * weight
                total_weight += weight

            if total_weight > 0:
                overall_similarity /= total_weight
            similarities['overall_similarity'] = overall_similarity

        return similarities

    def calculate_job_fit_confidence(
        self,
        rule_based_scores: Dict[str, float],
        semantic_similarities: Dict[str, float]
    ) -> float:
        """
        Calculate overall job fit confidence score.

        Combines rule-based matching with semantic similarity.
        """
        # Calculate weighted rule-based score
        rule_based_score = (
            rule_based_scores['skills_score'] * self.weights.skills +
            rule_based_scores['experience_score'] * self.weights.experience +
            rule_based_scores['education_score'] * self.weights.education +
            rule_based_scores['certification_score'] * self.weights.certifications +
            rule_based_scores['cgpa_score'] * self.weights.cgpa
        )

        # Get semantic similarity
        semantic_sim = semantic_similarities.get('overall_similarity', 0.0)

        # Combined score (weighted average)
        combined_score = (
            rule_based_score * (1 - self.weights.semantic_similarity) +
            semantic_sim * self.weights.semantic_similarity
        )

        return combined_score

    def score_candidate(
        self,
        job: Job,
        resume: Resume
    ) -> Dict[str, Any]:
        """
        Score a candidate against job requirements.

        Returns:
            Complete scoring results including breakdown
        """
        # Prepare embeddings
        job_embeddings = self.prepare_job_embeddings(job)
        resume_embeddings = self.prepare_resume_embeddings(resume)

        # Calculate scores
        rule_based_scores = self.calculate_rule_based_scores(
            job_embeddings['job_data'],
            resume_embeddings['resume_data']
        )

        semantic_similarities = self.calculate_semantic_similarity(
            job_embeddings,
            resume_embeddings
        )

        # Calculate overall scores
        overall_score = self.calculate_job_fit_confidence(
            rule_based_scores,
            semantic_similarities
        )

        # Identify strengths and weaknesses
        strengths = rule_based_scores.get('strengths', [])
        weaknesses = rule_based_scores.get('weaknesses', [])

        # Add semantic similarity insights
        if semantic_similarities.get('overall_similarity', 0) > 0.7:
            strengths.append("High semantic match with job description")
        elif semantic_similarities.get('overall_similarity', 0) < 0.3:
            weaknesses.append("Low semantic match with job description")

        # Prepare result
        result = {
            'overall_score': float(overall_score),
            'score_breakdown': ScoreBreakdown(
                skills_score=rule_based_scores['skills_score'],
                experience_score=rule_based_scores['experience_score'],
                education_score=rule_based_scores['education_score'],
                certification_score=rule_based_scores['certification_score'],
                cgpa_score=rule_based_scores['cgpa_score'],
                semantic_similarity=semantic_similarities.get('overall_similarity', 0.0),
                job_fit_confidence=float(overall_score)
            ),
            'missing_requirements': rule_based_scores['missing_requirements'],
            'matched_requirements': rule_based_scores['matched_requirements'],
            'strengths': strengths,
            'weaknesses': weaknesses,
            'semantic_similarities': semantic_similarities,
            'rule_based_scores': {
                'skills': rule_based_scores['skills_score'],
                'experience': rule_based_scores['experience_score'],
                'education': rule_based_scores['education_score'],
                'certifications': rule_based_scores['certification_score'],
                'cgpa': rule_based_scores['cgpa_score']
            }
        }

        return result

    def rank_candidates(
        self,
        job: Job,
        resumes: List[Resume],
        candidate_scores: Optional[List[CandidateScore]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank multiple candidates for a job.

        Args:
            job: Job requirements
            resumes: List of candidate resumes
            candidate_scores: Optional pre-calculated scores

        Returns:
            List of ranked candidate results
        """
        ranked_results = []

        for resume in resumes:
            # Calculate or use existing scores
            if candidate_scores:
                # Find matching score
                score = next(
                    (cs for cs in candidate_scores if cs.resume_id == resume.id),
                    None
                )
                if score:
                    result = {
                        'resume_id': resume.id,
                        'overall_score': score.overall_score,
                        'rank': score.rank,
                        'is_shortlisted': score.is_shortlisted,
                        'score_breakdown': ScoreBreakdown(
                            skills_score=score.skills_score,
                            experience_score=score.experience_score,
                            education_score=score.education_score,
                            certification_score=score.certification_score,
                            cgpa_score=score.cgpa_score,
                            semantic_similarity=score.semantic_similarity,
                            job_fit_confidence=score.job_fit_confidence
                        ),
                        'missing_requirements': score.missing_requirements,
                        'matched_requirements': score.matched_requirements,
                        'strengths': score.strengths,
                        'weaknesses': score.weaknesses
                    }
                else:
                    # Calculate new score
                    result = self.score_candidate(job, resume)
                    result['resume_id'] = resume.id
            else:
                # Calculate new score
                result = self.score_candidate(job, resume)
                result['resume_id'] = resume.id

            ranked_results.append(result)

        # Sort by overall score (descending)
        ranked_results.sort(key=lambda x: x['overall_score'], reverse=True)

        # Assign ranks
        for i, result in enumerate(ranked_results):
            result['rank'] = i + 1

        return ranked_results


# Global scorer instance
scorer = CandidateScorer()
