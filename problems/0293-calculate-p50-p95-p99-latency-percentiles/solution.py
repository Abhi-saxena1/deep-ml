import torch
import math

def calculate_latency_percentiles(latencies: torch.Tensor) -> dict[str, float]:
    if latencies.numel() == 0:
        return {
            "P50": 0.0,
            "P95": 0.0,
            "P99": 0.0
        }

    latencies = torch.sort(latencies).values
    n = len(latencies)

    def percentile(p):
        position = p * (n - 1)

        lower = math.floor(position)
        upper = math.ceil(position)

        if lower == upper:
            return latencies[lower].item()

        fraction = position - lower

        return (
            latencies[lower]
            + fraction * (latencies[upper] - latencies[lower])
        ).item()

    return {
        "P50": round(percentile(0.50), 4),
        "P95": round(percentile(0.95), 4),
        "P99": round(percentile(0.99), 4)
    }
    
