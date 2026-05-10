import torch
import torch.nn as nn


class GroupWiseEvidenceAlignment(nn.Module):
    """

    Inputs:
        m_pred, m_gt: [B, K, H, W]
        y_pred, y_gt: [B, K] integer status vectors, where 1 means positive.
    Returns:
        total_loss, stats

    Notes:
    - TP uses IoU-style overlap loss.
    - FN uses NEGATIVE MSE.
    - FP uses energy suppression.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, m_pred, m_gt, y_pred, y_gt):
        assert m_pred.shape == m_gt.shape
        bsz, num_disease, _, _ = m_pred.shape
        device = m_pred.device

        l_tp_vals, l_fn_vals, l_fp_vals = [], [], []
        tp_counts = fn_counts = fp_counts = 0

        for b in range(bsz):
            yp = y_pred[b]
            yg = y_gt[b]

            g_tp = ((yg == 1) & (yp == 1)).nonzero(as_tuple=False).flatten()
            g_fn = ((yg == 1) & (yp == 0)).nonzero(as_tuple=False).flatten()
            g_fp = ((yg == 0) & (yp == 1)).nonzero(as_tuple=False).flatten()

            if g_tp.numel() > 0:
                pred = m_pred[b, g_tp]
                gt = m_gt[b, g_tp]
                inter = 2.0 * (pred * gt).sum(dim=(-1, -2)) + self.eps
                denom = pred.pow(2).sum(dim=(-1, -2)) + gt.pow(2).sum(dim=(-1, -2)) + self.eps
                l_tp_vals.append(1.0 - (inter / denom).mean())
                tp_counts += int(g_tp.numel())
            else:
                l_tp_vals.append(torch.zeros((), device=device))

            if g_fn.numel() > 0:
                pred = m_pred[b, g_fn]
                gt = m_gt[b, g_fn]
                mse = (pred - gt).pow(2).mean(dim=(-1, -2)).mean()
                l_fn_vals.append(-mse)
                fn_counts += int(g_fn.numel())
            else:
                l_fn_vals.append(torch.zeros((), device=device))

            if g_fp.numel() > 0:
                pred = m_pred[b, g_fp]
                energy = pred.pow(2).mean(dim=(-1, -2)).mean()
                l_fp_vals.append(energy)
                fp_counts += int(g_fp.numel())
            else:
                l_fp_vals.append(torch.zeros((), device=device))

        l_tp = torch.stack(l_tp_vals).mean()
        l_fn = torch.stack(l_fn_vals).mean()
        l_fp = torch.stack(l_fp_vals).mean()
        l_r = l_tp + l_fn + l_fp
        stats = {
            'gear_tp': float(l_tp.detach().cpu()),
            'gear_fn': float(l_fn.detach().cpu()),
            'gear_fp': float(l_fp.detach().cpu()),
            'gear_total': float(l_r.detach().cpu()),
            'gear_tp_count': tp_counts,
            'gear_fn_count': fn_counts,
            'gear_fp_count': fp_counts,
        }
        return l_r, stats
