import collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import os


_cfg  = None; _step = 0
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_net = _tgt = _opt = None
loss_hist = []        # Loss per step
risk_hist = []        # Risk per step
last_stats = {}
_enabled = True
_training = False        # True for online training; False for inference only
_converged = False      # Convergence/freeze flag
win_size   = 1000          # Moving window
_eps_loss  = 1e-2
loss_deque = collections.deque(maxlen=win_size)

class _Net(nn.Module):
    def __init__(self, dim, hid=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1), nn.Sigmoid())
            # nn.Linear(hid, 1))
    def forward(self, x): return self.mlp(x)



# ---------- Initialization ----------
def set_from_cfg(cfg):
    global _cfg, _net, _tgt, _opt, _enabled, _training, _converged, _step, last_stats
    _enabled = bool(cfg.get('enable', True))
    _training = bool(cfg.get('training', False))
    _converged = False
    _step = 0
    loss_hist.clear()
    risk_hist.clear()
    loss_deque.clear()
    last_stats = {}
    if not _enabled:          # Return early so callers bypass risk logic.
        return
    _cfg = cfg
    _net = _Net(int(cfg['in_dim'])).to(_device)
    _tgt = _Net(int(cfg['in_dim'])).to(_device)
    _tgt.load_state_dict(_net.state_dict())

    # Cast string config values to numeric types.
    lr     = float(cfg['lr'])
    _cfg['gamma']       = float(cfg['gamma'])
    _cfg['update_gap']  = int(cfg['update_gap'])
    _cfg['tau']         = float(cfg['tau'])
    _cfg['W_av']        = float(cfg['W_av'])
    _cfg['risk_penalty']= float(cfg['risk_penalty'])
    _cfg['k_inv_d']     = float(cfg['k_inv_d'])
    _cfg['eps_loss']    = float(cfg['eps_loss']) if 'eps_loss' in cfg else _eps_loss

    _opt = torch.optim.Adam(_net.parameters(), lr=lr)

    _ckpt_path = cfg.get('ckpt', None)
    has_ckpt = bool(_ckpt_path and os.path.isfile(_ckpt_path))
    if has_ckpt:
        _net.load_state_dict(torch.load(_ckpt_path, map_location=_device))
        _tgt.load_state_dict(_net.state_dict())
    elif not _training:
        raise FileNotFoundError(
            f"[AV-safe] inference mode requires a checkpoint, but none was found at {_ckpt_path!r}"
        )

    if _training:
        _converged = False
    else:
        _converged = True


# ---------- Inference ----------
@torch.no_grad()
def risk(x_np):                       # x_np: (in_dim,)
    if not _enabled:
        return 0.0                       # AV-safe is disabled, so risk is always 0.
    x = torch.tensor(x_np,dtype=torch.float32,device=_device)
    # return _tgt(x.unsqueeze(0)).item()
    v_prime = _tgt(x.unsqueeze(0)).item()        # V'(s)

    # ---------- Potential P(s) ----------
    k = _cfg.get('k_inv_d', 1.0)
    inv_d = float(x_np[-2])                      # Inverse-distance feature
    P = k * inv_d

    risk_val = max(0.0, min(1.0, v_prime + P))   # Clip to [0, 1]
    # risk_val   = torch.sigmoid(torch.tensor(v_prime + P)).item()

    # Optional debug print.
    # print(f"[risk] V'={v_prime: .3f}  P={P: .3f}  Risk={risk_val: .3f}")
    
    return risk_val

# ---------- Online TD-0 update ----------
def update(s, s_next, done_col, done):
    """
    Updates the risk network online when training=True.
    """
    global _converged, _step, last_stats
    if not _enabled or not _training:
        return 0.0                       # AV-safe is disabled, so risk is always 0.
    if _converged:
        return                         # Stop further gradient updates.
    
    # global _step
    s  = torch.tensor(s,      dtype=torch.float32, device=_device).unsqueeze(0)
    s_ = torch.tensor(s_next, dtype=torch.float32, device=_device).unsqueeze(0)
    dc = torch.tensor([[float(done_col)]],         device=_device)      # shape [1, 1], now uses dense reward.
    done_f = torch.tensor([[float(done)]],     device=_device)   # 1 if terminal
    

    # with torch.no_grad():
    #     tgt = torch.where(dc > 0.5,
    #                       torch.ones_like(dc),          # Collision maps to target 1.
    #                       _cfg['gamma'] * _tgt(s_))     # Otherwise use gamma * J_theta(s').
    with torch.no_grad():
        # if done_col:                     # Collision frame
        #     tgt = torch.ones_like(dc)    # 1
        # elif done:                       # Normal termination without collision
        #     tgt = torch.zeros_like(dc)   # 0
        # else:                            # Intermediate step
        #     tgt = _cfg['gamma'] * _tgt(s_)
        tgt = dc + (1. - done_f) * _cfg['gamma'] * _tgt(s_)
        tgt = torch.clamp(tgt, 0.0, 1.0)                      # Keep values in [0, 1].

        # Use the restored-scale risk target to set positive-sample weights:
        # Phi(s_t) = \hat{Phi}(s_t) + F(s_t)
        p_curr = _cfg['k_inv_d'] * s[:, -2:-1]

    pred = _net(s)

    # loss = F.mse_loss(pred, tgt)

    # pos_w  = 100.0
    # w = torch.where(tgt > 0.8, pos_w, 1.0).to(pred.dtype)   # Multiply positive samples by 100.
    pos_w  = 50
    tgt_risk = torch.clamp(tgt + p_curr, 0.0, 1.0)
    w = torch.where(tgt_risk > 0.85, pos_w, 1.0).to(pred.dtype)   # Weight high-risk samples.
    loss = F.binary_cross_entropy(pred, tgt, weight=w)

    # pos_w = torch.tensor(100.0, device=pred.device, dtype=pred.dtype)
    # loss = F.binary_cross_entropy_with_logits(pred, tgt,pos_weight=pos_w)


    _opt.zero_grad()
    loss.backward()
    _opt.step()

    # if _step % 500 == 0:        # Lower this value to log more frequently.
    # print(f'[AV-safe] step {_step:>6}  loss={loss.item():.4e}  Jπ={pred.item():.3f}')

    step_id = _step + 1

    print(f"[AV-safe] step {step_id:>6}  loss={loss.item():.3e}  "
      f"V'={pred.item():.3f}  P={(_cfg['k_inv_d']*float(s[-1,-2])):.3f}  "
      f"Risk={max(0,min(1,pred.item()+_cfg['k_inv_d']*float(s[-1,-2]))):.3f}")
        # f"Risk={torch.sigmoid(torch.tensor(pred + _cfg['k_inv_d']*s[0,-2])).item():.3f}"
        
    # Record monitoring curves.
    loss_hist.append(loss.item())
    risk_hist.append(pred.item())
    last_stats = {
        'avsafe_update_step': int(step_id),
        'avsafe_loss': float(loss.item()),
        'avsafe_pred': float(pred.item()),
        'avsafe_tgt': float(tgt.item()),
        'avsafe_tgt_risk': float(tgt_risk.item()),
        'avsafe_td_reward': float(dc.item()),
        'avsafe_weight': float(w.item()),
    }

    loss_deque.append(loss.item())
    # -------- Convergence check --------
    if len(loss_deque) == win_size:
        if np.mean(loss_deque) < _cfg['eps_loss']:
            _converged = True
            print(f'AV-safe is ready: step {step_id}, converged={_converged}')

    # -------- Soft-update the target network --------
    if step_id % _cfg['update_gap'] == 0:
        β = _cfg['tau']
        for p_t, p in zip(_tgt.parameters(), _net.parameters()):
            p_t.data.mul_(1-β).add_(β * p.data)

    _step = step_id


def converged():
    return _converged if _enabled else True

def latest_stats():
    return dict(last_stats)

def n_updates():                         # Convenience helper for progress logging
    return _step

def gamma(): 
    return _cfg['gamma']

def save(path: str):
    if _net is None: return
    torch.save(_net.state_dict(), path)

def ready():
    """Returns True in inference mode or after training convergence."""
    return _converged
