import torch

def pairwise_preference_judge(comparisons, criteria_weights, tie_threshold):
    if not comparisons:
        return {
            "results": [],
            "win_rate_a": 0.0,
            "win_rate_b": 0.0,
            "tie_rate": 0.0,
            "avg_margin": 0.0
        }

    total_weight = sum(criteria_weights.values())
    results = []

    a_wins = 0
    b_wins = 0
    ties = 0
    total_margin = 0.0

    for comparison in comparisons:
        scores_a = comparison["scores_a"]
        scores_b = comparison["scores_b"]

        weighted_a = 0.0
        weighted_b = 0.0

        for criterion, weight in criteria_weights.items():
            weighted_a += weight * scores_a[criterion]
            weighted_b += weight * scores_b[criterion]

        score_a = weighted_a / total_weight
        score_b = weighted_b / total_weight

        difference = score_a - score_b
        margin = abs(difference)

        if margin <= tie_threshold:
            winner = "tie"
            ties += 1
        elif difference > 0:
            winner = "A"
            a_wins += 1
        else:
            winner = "B"
            b_wins += 1

        total_margin += margin

        results.append({
            "id": comparison["id"],
            "winner": winner,
            "margin": round(margin, 4)
        })

    n = len(comparisons)

    return {
        "results": results,
        "win_rate_a": round(a_wins / n, 4),
        "win_rate_b": round(b_wins / n, 4),
        "tie_rate": round(ties / n, 4),
        "avg_margin": round(total_margin / n, 4)
    }

 