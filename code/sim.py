"""Slotted simulator: electric parcel couriers, battery-swap cabinets, and dispatch policies.

Time unit: minute (one dispatch batch per minute). Distances: km (planar, x detour factor).
Energy: Wh. Money: RMB. All physical/economic parameters are in CFG and are assumptions
documented in the paper (Table of parameters); experiments sweep the uncertain ones.
"""
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

NEG = -1e12
IDLE, BUSY, TOSTATION, WAITING, OFF, STRANDED = range(6)


@dataclass
class Cfg:
    speed: float = 0.30          # km/min (18 km/h)
    detour: float = 1.3          # road distance / straight-line distance
    t_service: int = 3           # min per pickup
    t_swap: int = 1              # min per swap
    t_unload: int = 3            # min at hub
    load_cap: int = 12           # parcels before a hub trip
    lead: int = 20               # request released `lead` min before the realized pickup time
    window: int = 45             # pickup deadline = release + window
    noise: float = 0.15          # lognormal sigma of travel-time multiplier
    pay_base: float = 3.0        # RMB per pickup
    pay_km: float = 0.5          # RMB per km between pickup and hub
    c_km: float = 0.3            # RMB per km travel cost in the efficiency utility
    c_wait: float = 0.2          # RMB per min expected swap wait in the utility
    c_age: float = 0.05          # RMB per min a request has waited (anti-starvation term)
    pack_wh: tuple = (1200.0, 1200.0)  # pack energy of type A, B (48V-25Ah class)
    share_b: float = 0.25        # share of couriers whose vehicle uses pack type B
    wh_km: tuple = (25.0, 35.0)  # consumption: regular vehicle, heavy/old vehicle
    share_heavy: float = 0.3
    soc0: tuple = (0.3, 0.9)     # initial SoC range
    reserve: float = 0.05        # reserve energy fraction kept after reaching a cabinet
    th_force: float = 0.20       # must swap below this SoC
    th_pro: float = 0.45         # proactive swap allowed below this SoC
    n_cab: int = 0               # 0 -> ceil(fleet/6)
    couriers_per_cab: float = 8.0
    km2_per_cab: float = 6.0
    cab_place: str = "kmeans"   # kmeans | random
    e_noise: float = 0.0         # lognormal sigma of actual vs. planned leg energy
    ctg: float = 1.0             # weight of the energy cost-to-go term (joint policies)
    cab_slots: int = 10
    b_cab_share: float = 0.4     # share of cabinets that also stock type-B packs (half their slots)
    p_slot: float = 600.0        # W per charging slot
    p_cab: float = 4800.0        # W cabinet power cap (shared)
    ext_rate: float = 2.0        # external (private-rider) swaps per cabinet-hour, daily mean
    ext_b: float = 0.2           # share of external riders using type B
    ret_soc: tuple = (0.1, 0.3)  # SoC of returned packs
    res_slack: int = 10          # reservation held this many min past predicted arrival
    kappa_s: float = 1.0         # value of a proactive swap, in mean-order-pay units
    n_res: int = 3               # reservable packs listed per cabinet and pack type
    b_force: float = 1e6         # weight bonus making forced swaps win
    fleet_per_orders: float = 40.0  # one courier per this many daily requests
    ext_policy: str = "yield"   # yield/equal: identical in practice (couriers queue only when no free pack exists); retry: try one other cabinet
    gamer_share: float = 0.0     # share of couriers who log on 60 min early and stay 60 min late (gaming the floor)
    gaming: str = "extend"       # extend: gamers log on 60 min early/late; inert: pad_min of logged-in presence before the
                                 # day's first request, invisible to dispatch; none: gamers behave honestly
    pad_min: int = 60
    carry: float = 0.0           # share of couriers who resume each morning with yesterday's end SoC and position
    res_keep: int = 0            # charged packs per cabinet/type never offered for reservation (walk-in buffer)
    shift_pick: str = "top"     # top | random: which real couriers' shift spans are used
    seed: int = 0


def ext_profile(t):
    """Relative external swap intensity by minute of day (commute and lunch peaks)."""
    h = t / 60.0
    return 0.4 + np.exp(-((h - 8.5) / 1.0) ** 2) + 0.6 * np.exp(-((h - 12.5) / 1.0) ** 2) + np.exp(-((h - 18.0) / 1.2) ** 2)


_PROF_MEAN = np.mean([ext_profile(t) for t in range(7 * 60, 19 * 60)])


@dataclass
class Policy:
    name: str
    dispatch: str = "eff"        # near | eff | ff | dpp
    swap: str = "thresh"         # thresh | seq | joint | joint_nores
    energy_filter: bool = True
    V: float = 10.0
    lam: float = 1.0             # FairFoody-style weight
    rho: float = 0.0             # earnings floor RMB/h (dpp)
    ctg: float = -1.0            # energy cost-to-go weight; negative = use Cfg.ctg
    qswap: bool = True           # weight reservation columns by the deficit (IFD); False = Q only in request weights
    credit: bool = False         # income signal uses credited time (working, or idle while a request is pending) instead of online time
    signal: str = "cum"          # IFD income signal: cum = cumulative deficit [rho*A - R]^+ (no forgiveness); queue = reflected Q


class Sim:
    def __init__(self, inst, day, cfg: Cfg, pol: Policy, fleet=None, persona=None):
        self.cfg, self.pol = cfg, pol
        rng = np.random.default_rng([cfg.seed, int(day)])
        self.day = int(day)
        self.rng_ext = np.random.default_rng([cfg.seed, int(day), 2])  # external riders
        self.arr_t = {}
        d = inst["days"][day]
        self.hub = np.array(inst["hub"])
        # requests
        rel = d["t"] - cfg.lead
        self.o_rel, self.o_dl = rel.astype(int), (rel + cfg.window).astype(int)
        self.o_xy = np.c_[d["x"], d["y"]]
        self.o_pay = cfg.pay_base + cfg.pay_km * self.dist(self.o_xy, self.hub)
        self.n_o = len(rel)
        # couriers: the most active real couriers of the day give the shift spans
        n = persona["n"] if persona else (fleet or max(3, int(np.ceil(self.n_o / cfg.fleet_per_orders))))
        sh = d["shifts"]
        if cfg.shift_pick == "random":  # any real courier of the day, not only the most active
            sh = sh[np.random.default_rng([cfg.seed, int(day), 5]).permutation(len(sh))]
        sh = sh[np.arange(n) % len(sh)]
        if persona:  # weekly roster: today's real shift spans dealt to couriers in random order
            sh = sh[rng.permutation(n)]
        self.n = n
        self.s_on = np.maximum(sh[:, 1] - 15, 7 * 60 - cfg.lead).astype(int)
        self.s_off = np.minimum(sh[:, 2] + 15, 19 * 60 + 30).astype(int)
        self.s_off = np.maximum(self.s_off, self.s_on + 120)
        self.gamer = persona["gamer"].copy() if persona and "gamer" in persona else np.zeros(n, bool)
        if persona and "gamer" in persona and cfg.gaming == "extend":  # gamers pad their online time in dead hours
            g = persona["gamer"]
            self.s_on = np.where(g, np.maximum(self.s_on - 60, 6 * 60), self.s_on)
            self.s_off = np.where(g, np.minimum(self.s_off + 60, 21 * 60), self.s_off)
        self.ptype = (rng.random(n) < cfg.share_b).astype(int)
        self.heavy = rng.random(n) < cfg.share_heavy
        if persona:
            self.ptype, self.heavy = persona["ptype"].copy(), persona["heavy"].copy()
        self.whkm = np.where(self.heavy, cfg.wh_km[1], cfg.wh_km[0])
        self.capw = np.array(cfg.pack_wh)[self.ptype]
        self.E = rng.uniform(*cfg.soc0, n) * self.capw
        self.xy = self.o_xy[rng.integers(0, self.n_o, n)] * 0.5 + self.hub * 0.5
        if persona and "E_end" in persona and cfg.carry > 0:  # no home charging: the energy filter keeps them in reach
            k = np.random.default_rng([cfg.seed, 33, int(day)]).random(n) < cfg.carry
            self.E = np.where(k, persona["E_end"], self.E)
            self.xy = np.where(k[:, None], persona["xy_end"], self.xy)
        self.load = np.zeros(n, int)
        self.status = np.full(n, OFF)
        self.free = np.zeros(n, int)      # minute when current activity ends
        self.dest = np.zeros((n, 2))
        self.leg_km = np.zeros(n)
        self.cur_order = np.full(n, -1)
        self.cur_cab = np.full(n, -1)
        self.earn = persona["earn"].copy() if persona else np.zeros(n)
        self.Q = persona["Q"].copy() if persona else np.zeros(n)
        self.online_min = persona["online"].copy() if persona else np.zeros(n)
        self.credit_min = persona["credit"].copy() if persona and "credit" in persona else np.zeros(n)
        self.demand_min = np.zeros(36 * 60, bool)  # minute t has demand if a request was released in (t - 30, t]
        for r in np.unique(self.o_rel.astype(int)):
            self.demand_min[r:r + 30] = True
        self.overtime_min = np.zeros(n)
        self.unfinished = 0
        self.blocked = np.zeros(n, bool)
        # cabinets: demand-weighted k-means placement on request locations
        x0, x1, y0, y1 = inst["bbox"]
        area = 0.6 * (x1 - x0) * (y1 - y0)  # km^2, rough service-area estimate
        k = cfg.n_cab or max(2, int(np.ceil(n / cfg.couriers_per_cab)), int(np.ceil(area / cfg.km2_per_cab)))
        if persona and "cab" in persona:  # fixed network for the whole week (sited on the calibration week)
            self.cab, has_b = persona["cab"], persona["has_b"]
            k = len(self.cab)
        else:
            self.cab, has_b = site_cabinets(self.o_xy, k, inst["bbox"], cfg, rng)
        self.k = k
        self.pk_type, self.pk_E, self.pk_res, self.pk_exp = [], [], [], []
        for c in range(k):
            m = cfg.cab_slots
            ty = np.zeros(m, int)
            if has_b[c]:
                ty[m // 2:] = 1
            cap = np.array(cfg.pack_wh)[ty]
            e = np.where(rng.random(m) < 0.8, cap, cap * rng.uniform(0.3, 0.9, m))
            self.pk_type.append(ty); self.pk_E.append(e)
            self.pk_res.append(np.full(m, -1)); self.pk_exp.append(np.zeros(m, int))
        self.has_type = np.array([[np.any(self.pk_type[c] == ty) for ty in (0, 1)] for c in range(k)])
        self.queue = [[[], []] for _ in range(k)]   # FCFS courier queues per cabinet and type
        self.n_packs = sum(len(x) for x in self.pk_E)
        # distance from every request to its nearest compatible cabinet, per pack type
        self.o_cab_d = np.stack([self.near_cab_dist(self.o_xy, ty) for ty in (0, 1)], 1)
        # Proposition 1 premise: every courier can reach a compatible cabinet at shift start
        start_d = np.array([self.near_cab_dist(self.xy[i:i + 1], self.ptype[i])[0] for i in range(n)])
        assert np.all(self.E >= start_d * self.whkm), "a courier starts out of reach of a compatible cabinet"
        self.hub_cab_d = np.array([self.near_cab_dist(self.hub[None], ty)[0] for ty in (0, 1)])
        # bookkeeping
        self.o_state = np.zeros(self.n_o, int)  # 0 future, 1 pending, 2 assigned, 3 expired, 4 failed
        self.o_assign_t = np.full(self.n_o, -1)
        self.o_late = np.zeros(self.n_o, bool)
        self.o_arrive = np.full(self.n_o, -1)
        self.o_who = np.full(self.n_o, -1)
        self.km_empty = 0.0
        self.km_total = 0.0
        self.km_i = np.zeros(self.n)
        self.swaps = 0
        self.swap_wait = []
        self.stockouts = 0
        self.res_fail = 0
        self.ext_lost = 0
        self.ext_served = 0
        self.strand = 0
        self.dec_ms = []
        self.trace = []
        self.mean_pay = float(self.o_pay.mean())

    # ---------- geometry ----------
    def dist(self, a, b):
        return self.cfg.detour * np.hypot(*(np.asarray(a) - np.asarray(b)).T)

    def near_cab_dist(self, xy, ty):
        ok = np.where(self.has_type[:, ty])[0] if hasattr(self, "has_type") else np.arange(self.k)
        return (self.cfg.detour * np.hypot(xy[:, None, 0] - self.cab[None, ok, 0], xy[:, None, 1] - self.cab[None, ok, 1])).min(1)

    def noise(self, sigma, *key):
        """Common random numbers: the disturbance depends only on (seed, day, decision identity)."""
        if sigma <= 0:
            return 1.0
        return np.random.default_rng([self.cfg.seed, self.day, *[int(k) for k in key]]).lognormal(0, sigma)

    def tt(self, km, key=None):
        f = self.noise(self.cfg.noise, *key) if key is not None else 1.0
        return max(1, int(np.ceil(km / self.cfg.speed * f)))

    # ---------- run ----------
    def run(self, t0=None, t1=22 * 60):
        t0 = t0 if t0 is not None else int(min(self.o_rel.min(), self.s_on.min()))
        if self.cfg.gaming == "inert" and self.gamer.any():  # physically inert padding in a zero-demand period
            r0 = int(self.o_rel.min())
            assert not self.demand_min[max(0, r0 - self.cfg.pad_min):r0].any()
            self.online_min[self.gamer] += self.cfg.pad_min
            if self.pol.dispatch in ("dpp", "dpp_lin") and not self.pol.credit:
                self.Q[self.gamer] += self.pol.rho * self.cfg.pad_min / 60.0
        for t in range(t0, t1):
            self.step(t)
        self.t_end = t1  # private riders are observed up to t1 under every policy
        t = t1
        while np.isin(self.status, (BUSY, TOSTATION, WAITING, STRANDED)).any():  # finish started work past the cutoff
            assert t < 36 * 60, "activities did not finish by 12:00 the next day"
            self.step(t)
            t += 1
        assert sum(len(x) for x in self.pk_E) == self.n_packs  # pack conservation
        self.unfinished = int(np.isin(self.status, (BUSY, TOSTATION, WAITING, STRANDED)).sum())
        return self.metrics()

    def step(self, t):
        c = self.cfg
        # shifts
        on = (self.s_on <= t) & (t < self.s_off)
        start = (self.status == OFF) & on & (self.free <= t)
        self.status[start] = IDLE
        # activity completions
        for i in np.where((self.free == t) & np.isin(self.status, (BUSY, TOSTATION)))[0]:
            self.finish(i, t)
        self.charge(t)
        if t < getattr(self, "t_end", 24 * 60):
            self.external(t)
        self.expire_res(t)
        # couriers going off shift once idle
        self.status[(self.status == IDLE) & ~on] = OFF
        working = np.isin(self.status, (BUSY, TOSTATION, WAITING, STRANDED))
        # working time counts until the last activity ends, also after the scheduled shift (overtime)
        online = (np.isin(self.status, (IDLE, BUSY, TOSTATION, WAITING, STRANDED)) & on) | (working & ~on)
        self.overtime_min += working & ~on
        self.online_min += online
        credited = online & (working | self.demand_min[t])  # padding with no recent demand earns no credit
        if self.pol.dispatch in ("dpp", "dpp_lin"):
            self.Q = np.maximum(self.Q + self.pol.rho / 60.0 * (credited if self.pol.credit else online), 0.0)
        # requests
        self.o_state[(self.o_state == 0) & (self.o_rel <= t)] = 1
        exp = (self.o_state == 1) & (self.o_dl <= t)
        self.o_state[exp] = 3
        self.credit_min += credited
        tic = time.perf_counter()
        self.decide(t)
        self.dec_ms.append((time.perf_counter() - tic) * 1e3)
        if t % 10 == 0:
            self.trace.append((t, float(self.Q.mean()), float(self.Q.max()), int(online.sum())))

    # ---------- physical processes ----------
    def finish(self, i, t):
        c = self.cfg
        km = self.leg_km[i]
        f = self.noise(c.e_noise, 9, i, t)
        self.E[i] -= km * self.whkm[i] * f
        self.km_total += km
        self.km_i[i] += km
        self.xy[i] = self.dest[i]
        if self.E[i] < 0:  # only reachable without the energy filter
            self.strand += 1
            j = self.cur_order[i]
            if j >= 0:
                self.o_state[j] = 4
                self.earn[i] -= self.o_pay[j]
                self.cur_order[i] = -1
            self.E[i] = self.capw[i]  # rescue van brings a full pack, courier loses 45 min
            self.status[i], self.free[i] = BUSY, t + 45
            self.leg_km[i] = 0.0
            self.cur_cab[i] = -1
            return
        if self.status[i] == TOSTATION:
            self.arrive_cab(i, t)
            return
        j = self.cur_order[i]
        if j >= 0:  # arrived at pickup
            self.cur_order[i] = -1
            self.load[i] += 1
            self.status[i], self.free[i] = BUSY, t + c.t_service
            self.leg_km[i] = 0.0
            if self.load[i] >= c.load_cap:  # then a hub trip
                self.go(i, t + c.t_service, self.hub, BUSY, extra=c.t_unload, key=(3, i, t))
                self.load[i] = 0
            return
        self.status[i] = IDLE
        self.leg_km[i] = 0.0

    def go(self, i, t, xy, st, extra=0, key=None):
        km = float(self.dist(self.xy[i], xy))
        self.dest[i], self.leg_km[i] = xy, km
        self.status[i], self.free[i] = st, t + self.tt(km, key if key is not None else (4, i, t)) + extra

    def charge(self, t):
        for c in range(self.k):
            e, ty = self.pk_E[c], self.pk_type[c]
            cap = np.array(self.cfg.pack_wh)[ty]
            ch = e < cap
            if ch.any():
                p = min(self.cfg.p_slot, self.cfg.p_cab / ch.sum())
                e[ch] = np.minimum(cap[ch], e[ch] + p / 60.0)
            full = e >= cap
            for m in np.where(full)[0]:
                r = self.pk_res[c][m]
                if r >= 0:
                    if self.status[r] == WAITING and self.cur_cab[r] == c:
                        self.do_swap(r, c, m, t)
                elif self.queue[c][ty[m]]:
                    self.do_swap(self.queue[c][ty[m]].pop(0), c, m, t)

    def external(self, t):
        if not hasattr(self, "_ext"):  # pre-draw the day's private riders so they are identical across policies
            T = 24 * 60
            lam = np.array([self.cfg.ext_rate * ext_profile(u) / _PROF_MEAN / 60.0 for u in range(T)])
            self._ext = (self.rng_ext.random((T, self.k)) < lam[:, None], self.rng_ext.random((T, self.k)) < self.cfg.ext_b,
                         self.rng_ext.uniform(*self.cfg.ret_soc, (T, self.k)))
        arr, typ, ret = self._ext
        for c in np.where(arr[t])[0]:
            ty = int(typ[t, c])
            m = self.free_pack(c, ty)
            if self.cfg.ext_policy == "retry" and (m is None or self.queue[c][ty]):
                alt = [k for k in np.argsort(np.hypot(*(self.cab - self.cab[c]).T)) if k != c][:1]
                for k in alt:
                    mk = self.free_pack(k, ty)
                    if mk is not None and not self.queue[k][ty]:
                        c, m = k, mk
            blocked = bool(self.queue[c][ty]) and self.cfg.ext_policy != "equal"
            if m is None or blocked:
                self.ext_lost += 1
                continue
            self.ext_served += 1
            self.pk_E[c][m] = self.cfg.pack_wh[ty] * ret[t, c]

    def free_pack(self, c, ty):
        ok = (self.pk_type[c] == ty) & (self.pk_res[c] < 0) & (self.pk_E[c] >= np.array(self.cfg.pack_wh)[self.pk_type[c]])
        idx = np.where(ok)[0]
        return int(idx[0]) if len(idx) else None

    def expire_res(self, t):
        for c in range(self.k):
            gone = (self.pk_res[c] >= 0) & (self.pk_exp[c] <= t)
            for m in np.where(gone)[0]:
                r = self.pk_res[c][m]
                if not (self.status[r] == WAITING and self.cur_cab[r] == c):
                    self.pk_res[c][m] = -1
                    self.res_fail += 1

    def arrive_cab(self, i, t):
        c, ty = self.cur_cab[i], self.ptype[i]
        self.leg_km[i] = 0.0
        self.arr_t[i] = t
        mine = np.where(self.pk_res[c] == i)[0]
        if len(mine):
            m = mine[0]
            if self.pk_E[c][m] >= self.cfg.pack_wh[ty]:
                self.do_swap(i, c, m, t)
            else:
                self.status[i] = WAITING
            return
        m = self.free_pack(c, ty)
        if m is not None and not self.queue[c][ty]:
            self.do_swap(i, c, m, t)
        else:
            self.stockouts += 1
            self.status[i] = WAITING
            self.queue[c][ty].append(i)

    def do_swap(self, i, c, m, t):
        self.swap_wait.append(t - self.arr_t.get(i, t))
        self.pk_E[c][m], self.E[i] = self.E[i], self.capw[i]
        self.pk_res[c][m] = -1
        self.swaps += 1
        self.status[i], self.free[i] = BUSY, t + self.cfg.t_swap
        self.leg_km[i] = 0.0
        self.dest[i] = self.xy[i]
        self.cur_cab[i] = -1

    # ---------- decisions ----------
    def decide(self, t):
        c, p = self.cfg, self.pol
        idle = np.where(self.status == IDLE)[0]
        pend = np.where(self.o_state == 1)[0]
        if len(idle) == 0:
            return
        soc = self.E[idle] / self.capw[idle]
        # time and energy feasibility of courier-request pairs
        if len(pend):
            D = self.cfg.detour * np.hypot(self.xy[idle, None, 0] - self.o_xy[None, pend, 0],
                                           self.xy[idle, None, 1] - self.o_xy[None, pend, 1])
            eta = t + np.ceil(D / c.speed)
            ok_t = eta <= self.o_dl[pend][None, :]
            tail = self.o_cab_d[pend][:, self.ptype[idle]].T  # to nearest compatible cabinet
            hubpart = np.where((self.load[idle] + 1 >= c.load_cap)[:, None],
                               self.dist(self.o_xy[pend], self.hub)[None, :] + self.hub_cab_d[self.ptype[idle]][:, None] - tail, 0.0)
            need = (D + tail + hubpart) * self.whkm[idle][:, None] + c.reserve * self.capw[idle][:, None]
            ok_e = self.E[idle][:, None] >= need
            ok = ok_t & ok_e if p.energy_filter else ok_t
            self.blocked[idle] = (ok_t & ~ok_e).any(1) & ~ok.any(1)
        else:
            D = ok = tail = np.zeros((len(idle), 0), bool)
            self.blocked[idle] = False
        forced = (soc < c.th_force) | self.blocked[idle]
        if not p.energy_filter:
            forced = soc < c.th_force
        ok = ok & ~forced[:, None]
        util = (self.o_pay[pend][None, :] - c.c_km * D + c.c_age * (t - self.o_rel[pend])[None, :]) if len(pend) else D
        # dispatch weights
        if p.dispatch == "near":
            self.dispatch_nearest(t, idle, pend, D, ok)
            assigned = set()
        else:
            if p.dispatch == "eff":
                W = util.copy()
            elif p.dispatch == "ff":
                rate = self.earn[idle] / np.maximum((self.credit_min if p.credit else self.online_min)[idle] / 60.0, 0.25)
                avg = self.earn.sum() / max((self.credit_min if p.credit else self.online_min).sum() / 60.0, 0.25)
                W = util + p.lam * (np.maximum(avg - rate, 0) / max(avg, 1e-6))[:, None] * self.o_pay[pend][None, :]
            elif p.dispatch == "dpp_lin":  # textbook per-slot drift-plus-penalty weight (Theorem 1 form)
                ctg = (p.ctg if p.ctg >= 0 else c.ctg) * c.c_km * tail * (1.0 - soc)[:, None]
                W = p.V * (util - ctg) + self.phi(idle)[:, None] * self.o_pay[pend][None, :]
            elif p.dispatch == "ff_rate":  # FairFoody-style signal inside the rate-normalized weight
                rate = self.earn[idle] / np.maximum((self.credit_min if p.credit else self.online_min)[idle] / 60.0, 0.25)
                avg = self.earn.sum() / max((self.credit_min if p.credit else self.online_min).sum() / 60.0, 0.25)
                ctg = (p.ctg if p.ctg >= 0 else c.ctg) * c.c_km * tail * (1.0 - soc)[:, None]
                W = (util - ctg + p.lam * (np.maximum(avg - rate, 0) / max(avg, 1e-6))[:, None] * self.o_pay[pend][None, :]) / self.job_minutes(idle, pend, D)
            else:  # dpp, renewal-ratio form: drift-plus-penalty per minute of courier time
                tau = self.job_minutes(idle, pend, D)
                # energy cost-to-go: the detour to a compatible cabinet this order leaves behind, weighted by depletion
                ctg = (p.ctg if p.ctg >= 0 else c.ctg) * c.c_km * tail * (1.0 - soc)[:, None]
                W = (p.V * (util - ctg) + self.phi(idle)[:, None] * self.o_pay[pend][None, :]) / tau
            W = np.where(ok, W, NEG)
            if p.swap in ("joint", "joint_nores"):
                Wc, cols = self.swap_cols(t, idle, soc, forced)
                W = np.hstack([W, Wc])
            else:
                cols = []
            assigned = self.match(t, idle, pend, W, cols)
        # swaps for couriers left idle
        rest = np.array([i for i in idle if i not in assigned and self.status[i] == IDLE], int)
        if len(rest) == 0:
            return
        if p.swap == "thresh" or p.swap in ("joint", "joint_nores"):
            # joint policies already decided swaps; a forced courier left unmatched still must go
            fr = rest[(self.E[rest] / self.capw[rest] < c.th_force) | self.blocked[rest]]
            for i in fr:
                self.send_nearest(i, t)
        elif p.swap in ("seq", "seq_nores"):
            s2 = self.E[rest] / self.capw[rest]
            f2 = (s2 < c.th_force) | self.blocked[rest]
            Wc, cols = self.swap_cols(t, rest, s2, f2)
            if cols:
                self.match(t, rest, np.array([], int), Wc, cols)
            for i in rest[f2]:
                if self.status[i] == IDLE:
                    self.send_nearest(i, t)

    def phi(self, idx):
        """IFD income signal: the cash top-up owed so far (cum) or the reflected virtual queue (queue)."""
        if self.pol.signal == "queue":
            return self.Q[idx]
        base = self.credit_min if self.pol.credit else self.online_min
        return np.maximum(self.pol.rho * base[idx] / 60.0 - self.earn[idx], 0.0)

    def job_minutes(self, idle, pend, D):
        """Planned minutes a request occupies the courier, including the hub trip when the load fills."""
        c = self.cfg
        hub = (self.load[idle] + 1 >= c.load_cap)[:, None] * (np.ceil(self.dist(self.o_xy[pend], self.hub) / c.speed)[None, :] + c.t_unload)
        return np.ceil(D / c.speed) + c.t_service + hub

    def swap_cols(self, t, idle, soc, forced):
        """Reservable-pack columns: up to `per` packs per cabinet and type."""
        c, p = self.cfg, self.pol
        cols, W = [], []
        want = forced | (soc < c.th_pro)
        if not want.any():
            return np.zeros((len(idle), 0)), cols
        Dk = self.cfg.detour * np.hypot(self.xy[idle, None, 0] - self.cab[None, :, 0], self.xy[idle, None, 1] - self.cab[None, :, 1])
        q = self.phi(idle) if (p.dispatch in ("dpp", "dpp_lin") and p.qswap) else np.zeros(len(idle))
        V = p.V if p.dispatch in ("dpp", "dpp_lin") else 1.0
        val = self.mean_pay * c.kappa_s * np.clip((c.th_pro - soc) / c.th_pro, 0, 1)
        for k in range(self.k):
            for ty in (0, 1):
                cand = np.where((self.pk_type[k] == ty) & (self.pk_res[k] < 0))[0]
                if len(cand) == 0:
                    continue
                cap = c.pack_wh[ty]
                cand = cand[np.argsort(-self.pk_E[k][cand])]
                n_full = int((self.pk_E[k][cand] >= cap).sum())
                cand = cand[min(c.res_keep, n_full):][:c.n_res]  # keep a walk-in buffer of charged packs
                n_ch = int((self.pk_E[k] < np.array(c.pack_wh)[self.pk_type[k]]).sum())
                p_pack = min(c.p_slot, c.p_cab / max(1, n_ch))  # shared cabinet power
                for m in cand:
                    ready = (cap - self.pk_E[k][m]) / (p_pack / 60.0)
                    eta = np.ceil(Dk[:, k] / c.speed)
                    wait = np.maximum(ready - eta, 0)
                    # same structure as the request weight: V x utility + income signal x protected earnings
                    w = V * (val - c.c_km * Dk[:, k] - c.c_wait * wait) + q * val
                    if p.dispatch in ("dpp", "ff_rate"):
                        w = w / (eta + wait + c.t_swap)
                    w = w + forced * c.b_force
                    reach = self.E[idle] >= Dk[:, k] * self.whkm[idle]
                    w = np.where(want & reach & (self.ptype[idle] == ty) & (w > 0), w, NEG)
                    W.append(w)
                    cols.append((k, m))
        if not cols:
            return np.zeros((len(idle), 0)), cols
        return np.array(W).T, cols

    def match(self, t, idle, pend, W, cols):
        nI = len(idle)
        if W.shape[1] == 0:
            return set()
        M = np.hstack([W, np.zeros((nI, nI))])  # dummy "stay idle" columns
        r, cidx = linear_sum_assignment(M, maximize=True)
        assigned = set()
        n_o = len(pend)
        for a, b in zip(r, cidx):
            if M[a, b] <= 0 or M[a, b] <= NEG / 2 or b >= W.shape[1]:
                continue
            i = idle[a]
            if b < n_o:
                self.assign(i, pend[b], t)
            else:
                k, m = cols[b - n_o]
                self.send_cab(i, k, m, t, reserve=(self.pol.swap not in ("joint_nores", "seq_nores")))
            assigned.add(i)
        return assigned

    def dispatch_nearest(self, t, idle, pend, D, ok):
        free = np.ones(len(idle), bool)
        for b in np.argsort(self.o_rel[pend], kind="stable"):
            cand = np.where(ok[:, b] & free)[0]
            if len(cand) == 0:
                continue
            a = cand[np.argmin(D[cand, b])]
            free[a] = False
            self.assign(idle[a], pend[b], t)

    def assign(self, i, j, t):
        self.o_state[j] = 2
        self.o_assign_t[j] = t
        self.cur_order[i] = j
        self.earn[i] += self.o_pay[j]
        if self.pol.dispatch in ("dpp", "dpp_lin"):
            self.Q[i] = max(self.Q[i] - self.o_pay[j], 0.0)
        self.go(i, t, self.o_xy[j], BUSY, key=(1, i, j))
        self.km_empty += self.leg_km[i]
        self.o_arrive[j] = self.free[i]
        self.o_who[j] = i
        if self.free[i] > self.o_dl[j]:
            self.o_late[j] = True

    def send_cab(self, i, k, m, t, reserve):
        self.cur_cab[i] = k
        self.go(i, t, self.cab[k], TOSTATION, key=(2, i, k, t))
        self.km_empty += self.leg_km[i]
        if reserve and m is not None:
            self.pk_res[k][m] = i
            self.pk_exp[k][m] = t + self.tt(self.leg_km[i]) + self.cfg.res_slack  # planned arrival + slack

    def send_nearest(self, i, t):
        ty = self.ptype[i]
        ok = np.where(self.has_type[:, ty])[0]
        d = self.dist(self.cab[ok], self.xy[i])
        if self.pol.energy_filter and (self.E[i] >= d * self.whkm[i]).any():
            ok, d = ok[self.E[i] >= d * self.whkm[i]], d[self.E[i] >= d * self.whkm[i]]  # reachable only
        avail = np.array([self.free_pack(k, ty) is not None for k in ok])
        pick = ok[np.argmin(np.where(avail, d, d + 1e6))] if avail.any() else ok[np.argmin(d)]
        self.send_cab(i, pick, None, t, reserve=False)

    # ---------- metrics ----------
    def metrics(self):
        hrs = np.maximum(self.online_min / 60.0, 1e-9)
        rate = self.earn / hrs
        served = int((self.o_state == 2).sum())
        r = np.sort(rate)
        n = len(r)
        gini = float((2 * np.arange(1, n + 1) - n - 1).dot(r) / (n * r.sum())) if r.sum() > 0 else 0.0
        jain = float(r.sum() ** 2 / (n * (r ** 2).sum())) if r.sum() > 0 else 0.0
        rho = self.pol.rho
        wait = (self.o_assign_t - self.o_rel)[self.o_state == 2]
        done = self.o_state == 2
        ontime = done & ~self.o_late
        dis = self.heavy | (self.ptype == 1)
        return dict(
            orders=self.n_o, served=served, service_rate=served / self.n_o, ontime=int(ontime.sum()),
            pickup_delay_sum=float((self.o_arrive - self.o_rel)[done].sum()),
            late_rate=float(self.o_late[self.o_state == 2].mean()) if served else 0.0,
            mean_wait=float(wait.mean()) if served else 0.0,
            earn_total=float(self.earn.sum()), km_per_order=self.km_total / max(served, 1),
            gini=gini, jain=jain, rate_min=float(r[0]), rate_p10=float(np.percentile(r, 10)),
            rate_mean=float(r.mean()),
            floor_viol=float((rate < rho - 1e-6).mean()) if rho > 0 else np.nan,
            shortfall=float(np.maximum(rho - rate, 0).mean()) if rho > 0 else np.nan,
            gap_heavy=float(rate[~self.heavy].mean() - rate[self.heavy].mean()) if self.heavy.any() and (~self.heavy).any() else np.nan,
            gap_typeB=float(rate[self.ptype == 0].mean() - rate[self.ptype == 1].mean()) if (self.ptype == 1).any() and (self.ptype == 0).any() else np.nan,
            gap_dis=float(rate[~dis].mean() - rate[dis].mean()) if dis.any() and (~dis).any() else np.nan,
            swaps=self.swaps, swap_wait=float(np.mean(self.swap_wait)) if self.swap_wait else 0.0,
            stockouts=self.stockouts, res_fail=self.res_fail, ext_lost=self.ext_lost,
            ext_served=self.ext_served, strand=self.strand, fleet=self.n, cabinets=self.k,
            dec_ms_mean=float(np.mean(self.dec_ms)), dec_ms_p99=float(np.percentile(self.dec_ms, 99)),
            dec_ms_max=float(np.max(self.dec_ms)),
            q_final_mean=float(self.Q.mean()), q_final_max=float(self.Q.max()),
            deficit_ok=bool(np.all(self.earn + 1e-6 >= rho * (self.credit_min if self.pol.credit else self.online_min) / 60.0 - self.Q)) if self.pol.dispatch in ("dpp", "dpp_lin") else True,
        ), rate


def site_cabinets(X, k, bbox, cfg, rng):
    """Cabinet sites (k-means on request locations, or uniform in the bounding box) and type-B stocking."""
    x0, x1, y0, y1 = bbox
    cab = kmeans(X, k, rng) if cfg.cab_place == "kmeans" else np.c_[rng.uniform(x0, x1, k), rng.uniform(y0, y1, k)]
    has_b = np.zeros(k, bool)
    has_b[rng.permutation(k)[:max(1, int(round(cfg.b_cab_share * k)))]] = True
    return cab, has_b


def kmeans(X, k, rng, it=30):
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(it):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
        C = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
    return C
