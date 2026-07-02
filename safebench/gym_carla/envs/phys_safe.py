# ========== safebench/gym_carla/envs/phys_safe.py ==========
import numpy as np

# ---------------- Physical safety parameters mirrored from YAML --------------
A_LON_DEC_AV   = 4.0   # AV maximum longitudinal deceleration |a_lon,max,AV|   (m/s^2)
A_LON_DEC_NPC  = 4.0   # NPC maximum longitudinal deceleration |a_lon,max,NPC|
A_LAT_DEC_AV   = 1.5   # AV maximum lateral deceleration |a_lat,max,AV|
A_LAT_DEC_NPC  = 1.5   # NPC maximum lateral deceleration |a_lat,max,NPC|
EPS = 1e-6             # Avoid division by zero
INF_DIST = 1e6     # 1000 km is far beyond the CARLA map scale

# ===========================================================
# 1) Opposite direction: formula d_safe,1^{lon}
# ===========================================================
def phys_safe_opposite(v_av: float, v_npc: float) -> float:
    """
    For two vehicles moving toward each other, the minimum longitudinal safety distance required so both can brake at maximum deceleration without collision is:
    Minimum required longitudinal safety distance:
        d_safe,1^{lon} = V_AV^2/(2 a_AV) + V_NPC^2/(2 a_NPC)
    """
    d = (v_av**2) / (2.0 * (A_LON_DEC_AV + EPS)) \
      + (v_npc**2) / (2.0 * (A_LON_DEC_NPC + EPS))
    return max(d, 0.0)

# ===========================================================
# 2) Same direction, rear-end, or cut-in: formula d_safe,2^{lon} = max(d_stop, d_equal)
# ===========================================================
def _d_equal(v_av: float, v_npc: float) -> float:
    """
    Position gap when both same-direction vehicles brake maximally and reach equal speed:
        d_equal^{lon} = (V_AV - V_NPC)^2 / (2 (a_AV - a_NPC))
    Valid only when a_AV > a_NPC and V_AV > V_NPC; otherwise returns 0.
    """
    if A_LON_DEC_AV <= A_LON_DEC_NPC + EPS or v_av <= v_npc + EPS:
        return 0.0
    # Near-equal speeds or decelerations are handled by the early return above to avoid unstable divisions.
    return (v_av - v_npc)**2 / (2.0 * (A_LON_DEC_AV - A_LON_DEC_NPC))

def _d_stop(v_av: float, v_npc: float) -> float:
    """
    NPC brakes to a stop first, then AV brakes to a stop at maximum deceleration.
    Required distance after both stop, with absolute value to keep it non-negative:
        d_stop^{lon} = | V_NPC^2 / (2 a_AV) - V_AV^2 / (2 a_NPC) |
    """
    return abs((v_npc**2) / (2.0 * (A_LON_DEC_NPC + EPS))
             - (v_av **2) / (2.0 * (A_LON_DEC_AV + EPS)))

def phys_safe_same_lane(v_av: float, v_npc: float) -> float:
    """
    Physical safety distance in the same lane; use the larger of d_stop and d_equal.
    If the rear vehicle is not faster than the front vehicle, no extra distance is required.
    """
    if v_av <= v_npc + EPS:
        return 0.0
    return max(_d_stop(v_av, v_npc), _d_equal(v_av, v_npc))



# =======================================================
# Lateral safety distance, same or opposite lateral direction
# =======================================================



def phys_safe_lat_same(v_av: float, v_npc: float) -> float:
    """
    Lateral safety distance when NPC cuts toward AV.

    If NPC's maximum lateral braking is no stronger than AV's, use the equal-brake formula.
    Otherwise NPC has stronger lateral authority and AV cannot fully counter it, so return infinity.
    This means no finite safety distance exists, forcing a large penalty.
    """
    v_rel = v_npc - v_av
    if v_rel <= EPS:                      # NPC is moving away; risk is zero
        return 0.0

    if A_LAT_DEC_AV > A_LAT_DEC_NPC + EPS:
        # AV has stronger lateral braking or reverse acceleration; use equal-brake.
        return v_rel**2 / (2.0 * (A_LAT_DEC_AV - A_LAT_DEC_NPC))
    else:
        # NPC is stronger; AV cannot defend, so the safety distance is infinite.
        return INF_DIST



def phys_safe_lat_opp(v_av: float, v_npc: float) -> float:
    """
    Opposite lateral motion, or one side near zero; sum both stopping distances:
        d = v_AV^2/(2 a_AV) + v_NPC^2/(2 a_NPC)
    """
    return (v_av**2)  / (2.0 * (A_LAT_DEC_AV  + EPS)) + \
           (v_npc**2) / (2.0 * (A_LAT_DEC_NPC + EPS))


def set_from_cfg(c):
    global A_LON_DEC_AV, A_LON_DEC_NPC, A_LAT_DEC_AV, A_LAT_DEC_NPC, EPS, INF_DIST
    A_LON_DEC_AV  = c['a_lon_dec_av']
    A_LON_DEC_NPC = c['a_lon_dec_npc']
    A_LAT_DEC_AV  = c['a_lat_dec_av']
    A_LAT_DEC_NPC = c['a_lat_dec_npc']
    EPS           = c['eps']
    INF_DIST      = c['INF_DIST']
