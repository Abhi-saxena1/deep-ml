import torch

import math


def rubric_llm_judge_evaluation(
    judge_scores,
    criteria_weights,
    passing_threshold=0.6,
    max_score=5.0
):


    # Average score for each criterion across all judges
    num_judges = len(judge_scores)
    num_criteria = len(criteria_weights)

    criterion_scores = []

    for j in range(num_criteria):
        total = 0

        for i in range(num_judges):
            total += judge_scores[i][j]

        average = total / num_judges
        criterion_scores.append(average)

    # Weighted overall score
    weighted_score = 0

    for j in range(num_criteria):
        weighted_score += criteria_weights[j] * criterion_scores[j]

    # Normalize to 0-1
    normalized_score = weighted_score / max_score

    # Pass / fail
    pass_status = normalized_score >= passing_threshold

    # Calculate standard deviation for each criterion
    std_deviations = []

    for j in range(num_criteria):
        mean = criterion_scores[j]

        variance = 0

        for i in range(num_judges):
            difference = judge_scores[i][j] - mean
            variance += difference ** 2

        variance /= num_judges

        std = math.sqrt(variance)
        std_deviations.append(std)

    # Average standard deviation
    avg_std = sum(std_deviations) / num_criteria

    # Maximum possible standard deviation for scores [0, max_score]
    max_std = max_score / 2

    # Convert disagreement into agreement
    judge_agreement = 1 - (avg_std / max_std)

    return {
        "weighted_score": round(weighted_score, 4),
        "normalized_score": round(normalized_score, 4),
        "criterion_scores": [round(x, 4) for x in criterion_scores],
        "pass_status": pass_status,
        "judge_agreement": round(judge_agreement, 4)
    }

    
