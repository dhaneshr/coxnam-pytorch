import torch

def cox_loss(risk_scores, duration, event):
    if risk_scores.ndim > 1:
        risk_scores = risk_scores.squeeze()

    epsilon = 1e-9
    
    sorted_indices = torch.argsort(risk_scores, descending=True).long()
    sorted_risk_scores = risk_scores[sorted_indices]
    sorted_duration = duration[sorted_indices]
    sorted_event = event[sorted_indices]

    # Compute the log partial likelihood
    exp_risk_scores = torch.exp(sorted_risk_scores)
    cumsum_exp_risk_scores = torch.cumsum(exp_risk_scores, dim=0)
    log_partial_likelihood = torch.sum(sorted_risk_scores - torch.log(cumsum_exp_risk_scores + epsilon)) - \
        torch.sum(sorted_event * sorted_risk_scores)

    return -log_partial_likelihood
