import torch

def decompose_latency(stage_latencies : dict ,percentiles : list) -> dict:
    stage_tensors =list (stage_latencies.values())
    e2e_latencies = torch.stack(stage_tensors).sum(dim=0)

    e2e_mean = e2e_latencies.mean().item()

    e2e_percentiles ={
      p: torch.quantile(
          e2e_latencies,
        
          torch.tensor(p / 100.0, device = e2e_latencies.device
      )
      )
          for p in percentiles
      
    }

    stage_stats = {}

    for name , latencies in stage_latencies.items():
        stage_stats[name]= {
            
            "mean" : latencies.mean(),
            "std": round (latencies.std(unbiased=False).item(),2),
        
            "percentile" : {
                p: torch.quantile(
                    latencies,
                    torch.tensor(p /100.0, device =latencies.device)

            )
                for p in percentiles
        }  
    }

    stage_means ={
        name: latencies.mean()
        for name, latencies in stage_latencies.items()

    }
    bottleneck = max (
        stage_means,
        key = lambda name: stage_means[name].item()
    )
    total_mean = e2e_mean

    stage_pct ={
        name :round((mean.item() / total_mean) * 100, 2)
        for name , mean in stage_means.items()
    }

    return {
        "e2e_mean": e2e_mean,
        "e2e_percentiles": e2e_percentiles,
        "stage_stats": stage_stats,
        "bottleneck": bottleneck,
        "stage_pct": stage_pct
    }