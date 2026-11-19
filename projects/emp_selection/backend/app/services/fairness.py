"""Fairness analysis for candidate evaluation."""
import re
import statistics
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import logging
from collections import defaultdict

from app.models import CandidateScore, FairnessReport
from app.schemas import FairnessReportCreate

logger = logging.getLogger(__name__)


@dataclass
class BiasThresholds:
    """Thresholds for bias detection."""
    demographic_disparity: float = 0.2  # 20% difference between groups
    anonymization_impact: float = 0.15  # 15% score change due to anonymization
    individual_bias_score: float = 0.7  # Score above which bias is flagged


@dataclass
class BiasWeights:
    """Weights for different bias types."""
    demographic_disparity: float = 0.4
    anonymization_impact: float = 0.3
    text_pattern_bias: float = 0.3


class FairnessAnalyzer:
    """Analyzer for detecting bias in candidate evaluation."""

    def __init__(self):
        """Initialize the fairness analyzer."""
        self.thresholds = BiasThresholds()
        self.weights = BiasWeights()

        # Patterns that might indicate demographic information
        self.demographic_patterns = {
            'gender': [
                r'\b(?:he|him|his|she|her|hers)\b',
                r'\b(?:mr\.|mrs\.|ms\.|miss)\b',
                r'\b(?:father|mother|husband|wife|son|daughter)\b',
            ],
            'ethnicity': [
                r'\b(?:christian|muslim|hindu|jewish|buddhist)\b',
                r'\b(?:african|asian|european|american|middle eastern)\b',
                r'\b(?:native|indigenous|aboriginal)\b',
            ],
            'nationality': [
                r'\b(?:citizen|passport|visa|immigrant|expat)\b',
                r'\b(?:born in|nationality|country of origin)\b',
            ],
            'age': [
                r'\b(?:years? old|age:\s*\d+|born\s+\d{4})\b',
                r'\b(?:fresh graduate|recent graduate|early career)\b',
                r'\b(?:senior|experienced|veteran|seasoned)\b',
            ],
            'disability': [
                r'\b(?:disability|disabled|handicap|special needs)\b',
                r'\b(?:accommodation|accessible|wheelchair)\b',
            ],
        }

        # Protected demographic groups for analysis
        self.protected_groups = {
            'gender': ['male', 'female', 'non-binary'],
            'ethnicity': ['asian', 'black', 'white', 'hispanic', 'other'],
            'age_group': ['under_30', '30_50', 'over_50'],
            'disability_status': ['disabled', 'non_disabled']
        }

    def detect_demographic_indicators(self, text: str) -> Dict[str, List[str]]:
        """
        Detect potential demographic indicators in text.

        Returns:
            Dictionary of demographic categories and detected indicators
        """
        indicators = {category: [] for category in self.demographic_patterns}

        for category, patterns in self.demographic_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    indicators[category].extend(matches)

        return indicators

    def infer_demographic_group(self, indicators: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
        """
        Infer demographic group from indicators.

        Returns:
            Dictionary with inferred demographic values
        """
        inferred = {
            'gender': None,
            'ethnicity': None,
            'age_group': None,
            'disability_status': None
        }

        # Infer gender from pronouns and titles
        if indicators['gender']:
            gender_indicators = set([ind.lower() for ind in indicators['gender']])
            if any(ind in {'he', 'him', 'his', 'mr.'} for ind in gender_indicators):
                inferred['gender'] = 'male'
            elif any(ind in {'she', 'her', 'hers', 'mrs.', 'ms.', 'miss'} for ind in gender_indicators):
                inferred['gender'] = 'female'

        # Infer age group from age indicators
        if indicators['age']:
            for indicator in indicators['age']:
                if re.search(r'(?:fresh|recent|early)\s+graduate', indicator, re.IGNORECASE):
                    inferred['age_group'] = 'under_30'
                elif re.search(r'(?:senior|experienced|veteran|seasoned)', indicator, re.IGNORECASE):
                    inferred['age_group'] = 'over_50'

        # Note: Real implementation would use more sophisticated inference
        # This is a simplified version for demonstration

        return inferred

    def calculate_demographic_disparity(
        self,
        candidate_scores: List[CandidateScore],
        demographic_groups: Dict[int, Dict[str, Optional[str]]]
    ) -> Dict[str, float]:
        """
        Calculate demographic disparity in scores.

        Returns:
            Dictionary with disparity scores for each demographic dimension
        """
        if not candidate_scores:
            return {}

        # Group scores by demographic
        demographic_scores = defaultdict(list)

        for score in candidate_scores:
            groups = demographic_groups.get(score.resume_id, {})
            for dimension, group in groups.items():
                if group:
                    key = f"{dimension}_{group}"
                    demographic_scores[key].append(score.overall_score)

        # Calculate average scores per group
        group_averages = {}
        for group, scores in demographic_scores.items():
            if scores:
                group_averages[group] = statistics.mean(scores)

        # Calculate disparity metrics
        disparity_scores = {}

        # Calculate overall average
        all_scores = [score.overall_score for score in candidate_scores]
        overall_average = statistics.mean(all_scores) if all_scores else 0

        # Calculate disparity for each demographic dimension
        for dimension in self.protected_groups:
            dimension_groups = self.protected_groups[dimension]
            dimension_scores = {}

            # Get average for each group in this dimension
            for group in dimension_groups:
                key = f"{dimension}_{group}"
                if key in group_averages:
                    dimension_scores[group] = group_averages[key]

            if len(dimension_scores) >= 2:
                # Calculate range and ratio
                min_score = min(dimension_scores.values())
                max_score = max(dimension_scores.values())
                range_disparity = max_score - min_score
                ratio_disparity = max_score / min_score if min_score > 0 else float('inf')

                # Calculate disparity score (0-1, higher = more disparity)
                disparity_score = min(range_disparity, 1.0)  # Cap at 1.0
                disparity_scores[f"{dimension}_disparity"] = disparity_score

                # Calculate deviation from overall average
                for group, avg_score in dimension_scores.items():
                    deviation = abs(avg_score - overall_average)
                    disparity_scores[f"{dimension}_{group}_deviation"] = deviation

        return disparity_scores

    def analyze_anonymization_impact(
        self,
        scores_before: List[float],
        scores_after: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze the impact of anonymization on scores.

        Args:
            scores_before: Scores before anonymization
            scores_after: Scores after anonymization

        Returns:
            Analysis results
        """
        if len(scores_before) != len(scores_after):
            raise ValueError("Score lists must have same length")

        if not scores_before:
            return {}

        # Calculate basic statistics
        avg_before = statistics.mean(scores_before)
        avg_after = statistics.mean(scores_after)
        mean_difference = avg_after - avg_before
        relative_change = mean_difference / avg_before if avg_before != 0 else 0

        # Calculate individual changes
        individual_changes = [after - before for before, after in zip(scores_before, scores_after)]
        max_increase = max(individual_changes) if individual_changes else 0
        max_decrease = min(individual_changes) if individual_changes else 0
        std_change = statistics.stdev(individual_changes) if len(individual_changes) > 1 else 0

        # Count candidates with significant changes
        significant_threshold = avg_before * self.thresholds.anonymization_impact
        significant_increases = sum(1 for change in individual_changes if change > significant_threshold)
        significant_decreases = sum(1 for change in individual_changes if change < -significant_threshold)

        # Calculate fairness impact score (0-1, higher = more impact)
        fairness_impact = min(
            abs(relative_change) + (std_change / avg_before if avg_before != 0 else 0),
            1.0
        )

        return {
            'average_before': avg_before,
            'average_after': avg_after,
            'mean_difference': mean_difference,
            'relative_change': relative_change,
            'max_increase': max_increase,
            'max_decrease': max_decrease,
            'std_change': std_change,
            'significant_increases': significant_increases,
            'significant_decreases': significant_decreases,
            'fairness_impact_score': fairness_impact,
            'has_significant_impact': fairness_impact > 0.2  # Arbitrary threshold
        }

    def detect_text_pattern_bias(
        self,
        texts: List[str],
        scores: List[float]
    ) -> Dict[str, Any]:
        """
        Detect bias based on text patterns.

        Args:
            texts: List of resume texts
            scores: Corresponding scores

        Returns:
            Bias analysis results
        """
        if len(texts) != len(scores):
            raise ValueError("Text and score lists must have same length")

        if not texts:
            return {}

        # Group texts by score quartiles
        score_quartiles = np.percentile(scores, [25, 50, 75])
        quartile_groups = [[] for _ in range(4)]

        for text, score in zip(texts, scores):
            if score <= score_quartiles[0]:
                quartile_groups[0].append(text)
            elif score <= score_quartiles[1]:
                quartile_groups[1].append(text)
            elif score <= score_quartiles[2]:
                quartile_groups[2].append(text)
            else:
                quartile_groups[3].append(text)

        # Analyze pattern frequency across quartiles
        pattern_analysis = {}

        for category, patterns in self.demographic_patterns.items():
            pattern_frequencies = []

            for i, group_texts in enumerate(quartile_groups):
                if not group_texts:
                    pattern_frequencies.append(0)
                    continue

                # Count pattern occurrences
                total_count = 0
                for text in group_texts:
                    for pattern in patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        total_count += len(matches)

                # Normalize by text length
                total_chars = sum(len(text) for text in group_texts)
                frequency = total_count / max(total_chars, 1) * 1000  # per 1000 chars
                pattern_frequencies.append(frequency)

            # Calculate pattern bias score
            if len(pattern_frequencies) >= 2:
                # Compare top vs bottom quartile
                top_freq = pattern_frequencies[3]
                bottom_freq = pattern_frequencies[0]
                bias_ratio = top_freq / bottom_freq if bottom_freq > 0 else float('inf')

                pattern_bias_score = min(abs(1 - bias_ratio), 1.0)
                pattern_analysis[category] = {
                    'frequencies': pattern_frequencies,
                    'bias_ratio': bias_ratio,
                    'bias_score': pattern_bias_score
                }

        # Calculate overall text pattern bias score
        if pattern_analysis:
            overall_score = statistics.mean(
                [analysis['bias_score'] for analysis in pattern_analysis.values()]
            )
        else:
            overall_score = 0.0

        return {
            'pattern_analysis': pattern_analysis,
            'overall_pattern_bias_score': overall_score
        }

    def calculate_overall_bias_score(
        self,
        demographic_disparity: Dict[str, float],
        anonymization_impact: Dict[str, Any],
        text_pattern_bias: Dict[str, Any]
    ) -> float:
        """
        Calculate overall bias score from multiple metrics.

        Returns:
            Overall bias score (0-1, higher = more bias)
        """
        # Extract component scores
        demographic_score = 0.0
        if demographic_disparity:
            demographic_score = max(demographic_disparity.values(), default=0.0)

        anonymization_score = anonymization_impact.get('fairness_impact_score', 0.0)

        text_pattern_score = text_pattern_bias.get('overall_pattern_bias_score', 0.0)

        # Calculate weighted average
        overall_score = (
            demographic_score * self.weights.demographic_disparity +
            anonymization_score * self.weights.anonymization_impact +
            text_pattern_score * self.weights.text_pattern_bias
        )

        return min(overall_score, 1.0)

    def identify_bias_types(self, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Identify specific types of bias present.

        Returns:
            List of bias type descriptions
        """
        bias_types = []

        # Check demographic disparity
        demographic_disparity = analysis_results.get('demographic_disparity', {})
        for dimension, disparity in demographic_disparity.items():
            if disparity > self.thresholds.demographic_disparity:
                bias_types.append(f"Demographic disparity in {dimension}")

        # Check anonymization impact
        anonymization_impact = analysis_results.get('anonymization_impact', {})
        if anonymization_impact.get('has_significant_impact', False):
            bias_types.append("Significant anonymization impact")

        # Check text pattern bias
        text_pattern_bias = analysis_results.get('text_pattern_bias', {})
        pattern_analysis = text_pattern_bias.get('pattern_analysis', {})
        for category, analysis in pattern_analysis.items():
            if analysis.get('bias_score', 0) > 0.5:  # Arbitrary threshold
                bias_types.append(f"Text pattern bias in {category}")

        # Check overall bias score
        overall_bias_score = analysis_results.get('overall_bias_score', 0)
        if overall_bias_score > self.thresholds.individual_bias_score:
            bias_types.append("High overall bias score")

        return bias_types

    def analyze_candidate_fairness(
        self,
        candidate_scores: List[CandidateScore],
        texts_before_anonymization: Dict[int, str],
        texts_after_anonymization: Dict[int, str]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive fairness analysis.

        Args:
            candidate_scores: List of candidate scores
            texts_before_anonymization: Resume texts before anonymization
            texts_after_anonymization: Resume texts after anonymization

        Returns:
            Comprehensive fairness analysis results
        """
        if not candidate_scores:
            return {}

        # Extract scores and texts
        scores = [score.overall_score for score in candidate_scores]
        resume_ids = [score.resume_id for score in candidate_scores]

        # Detect demographic indicators
        demographic_indicators = {}
        demographic_groups = {}

        for score in candidate_scores:
            resume_id = score.resume_id
            text_before = texts_before_anonymization.get(resume_id, "")

            indicators = self.detect_demographic_indicators(text_before)
            demographic_indicators[resume_id] = indicators

            groups = self.infer_demographic_group(indicators)
            demographic_groups[resume_id] = groups

        # Analyze demographic disparity
        demographic_disparity = self.calculate_demographic_disparity(
            candidate_scores, demographic_groups
        )

        # Analyze anonymization impact
        scores_before = []
        scores_after = []
        for score in candidate_scores:
            resume_id = score.resume_id
            # Note: In real implementation, we'd have both before/after scores
            # For now, we'll use the same scores for demonstration
            scores_before.append(score.overall_score)
            scores_after.append(score.overall_score)

        anonymization_impact = self.analyze_anonymization_impact(scores_before, scores_after)

        # Detect text pattern bias
        text_list = [texts_before_anonymization.get(rid, "") for rid in resume_ids]
        text_pattern_bias = self.detect_text_pattern_bias(text_list, scores)

        # Calculate overall bias score
        overall_bias_score = self.calculate_overall_bias_score(
            demographic_disparity,
            anonymization_impact,
            text_pattern_bias
        )

        # Identify bias types
        bias_types = self.identify_bias_types({
            'demographic_disparity': demographic_disparity,
            'anonymization_impact': anonymization_impact,
            'text_pattern_bias': text_pattern_bias,
            'overall_bias_score': overall_bias_score
        })

        # Prepare results
        results = {
            'demographic_indicators': demographic_indicators,
            'demographic_groups': demographic_groups,
            'demographic_disparity': demographic_disparity,
            'anonymization_impact': anonymization_impact,
            'text_pattern_bias': text_pattern_bias,
            'overall_bias_score': overall_bias_score,
            'bias_types': bias_types,
            'has_potential_bias': len(bias_types) > 0,
            'summary': {
                'total_candidates': len(candidate_scores),
                'candidates_with_demographic_indicators': sum(
                    1 for indicators in demographic_indicators.values()
                    if any(indicators.values())
                ),
                'significant_bias_findings': len(bias_types)
            }
        }

        return results

    def generate_fairness_report(
        self,
        analysis_results: Dict[str, Any],
        candidate_score_id: int
    ) -> FairnessReportCreate:
        """
        Generate a fairness report from analysis results.

        Returns:
            FairnessReportCreate schema for database storage
        """
        # Extract anonymization impact data
        anonymization_impact = analysis_results.get('anonymization_impact', {})
        score_before = anonymization_impact.get('average_before')
        score_after = anonymization_impact.get('average_after')
        score_delta = anonymization_impact.get('mean_difference')

        # Calculate group fairness if demographic groups are available
        demographic_groups = analysis_results.get('demographic_groups', {})
        group_fairness_score = None
        demographic_group = None

        if demographic_groups:
            # For simplicity, use the first candidate's demographic group
            # In real implementation, this would be more sophisticated
            first_candidate_groups = next(iter(demographic_groups.values()), {})
            if first_candidate_groups:
                demographic_group = ", ".join(
                    f"{k}: {v}" for k, v in first_candidate_groups.items() if v
                )
                group_fairness_score = analysis_results.get('overall_bias_score', 0)

        # Create the report
        report = FairnessReportCreate(
            candidate_score_id=candidate_score_id,
            has_potential_bias=analysis_results.get('has_potential_bias', False),
            bias_types=analysis_results.get('bias_types', []),
            bias_score=analysis_results.get('overall_bias_score', 0.0),
            score_before_anonymization=score_before,
            score_after_anonymization=score_after,
            score_delta=score_delta,
            demographic_group=demographic_group,
            group_fairness_score=group_fairness_score,
            report_data=analysis_results
        )

        return report

    def batch_analyze_fairness(
        self,
        candidate_scores_list: List[List[CandidateScore]],
        texts_before_list: List[Dict[int, str]],
        texts_after_list: List[Dict[int, str]]
    ) -> List[Dict[str, Any]]:
        """
        Perform fairness analysis on multiple batches.

        Returns:
            List of analysis results for each batch
        """
        all_results = []

        for i, (candidate_scores, texts_before, texts_after) in enumerate(
            zip(candidate_scores_list, texts_before_list, texts_after_list)
        ):
            try:
                results = self.analyze_candidate_fairness(
                    candidate_scores,
                    texts_before,
                    texts_after
                )
                all_results.append(results)
                logger.info(f"Completed fairness analysis for batch {i}")
            except Exception as e:
                logger.error(f"Failed to analyze fairness for batch {i}: {e}")
                all_results.append({})

        return all_results


# Global analyzer instance
fairness_analyzer = FairnessAnalyzer()