# utils/unwrap_torch.py
import torch, math

def unwrap_phase_deg_torch(phase_deg: torch.Tensor, remove_offset: bool = True) -> torch.Tensor:
    """
    phase_deg: [..., 200] wrapped to [-180,180)
    returns:   [..., 200] unwrapped (degrees)
    """
    ph = phase_deg * (math.pi/180.0)
    d = torch.diff(ph, dim=-1)
    # wrap diffs to (-pi, pi]
    dmod = (d + math.pi) % (2*math.pi) - math.pi
    # special case to avoid -pi jumps when d>0
    mask = (dmod == -math.pi) & (d > 0)
    dmod = torch.where(mask, torch.full_like(dmod, math.pi), dmod)
    ph0 = ph[..., :1]
    phu = torch.cumsum(torch.cat([ph0, dmod], dim=-1), dim=-1)
    if remove_offset:
        phu = phu - phu[..., :1]
    return phu * (180.0/math.pi)
