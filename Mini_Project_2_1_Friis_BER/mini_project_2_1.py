"""
Mini Project 2.1 - Free-space path loss and digital modulation performance.

Part (a): Friis received power versus distance for three carriers, with the 6 dB-per-doubling check.
Part (b): Monte Carlo BER of BPSK and Gray-coded 16-QAM over AWGN, with theory overlays,
          the Eb/N0 gap at BER = 1e-3 and its relation to Shannon-Hartley.
Lab tasks 1-6: QPSK, rate-1/2 convolutional code with soft Viterbi decoding, 64-QAM,
          log-distance path loss, log-normal shadowing, and a throughput proxy against capacity.

Run from the repository root:
    python Mini_Project_2_1_Friis_BER/mini_project_2_1.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc

np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
LOG = []


def say(s=""):
    print(s)
    LOG.append(s)


# =============================================================== Part (a): Friis
def friis_rx_power(Ptx_dBm, Gtx_dBi, Grx_dBi, d_km, f_MHz):
    return Ptx_dBm + Gtx_dBi + Grx_dBi - 20 * np.log10(d_km) - 20 * np.log10(f_MHz) - 32.44


PTX, GTX, GRX = 43.0, 15.0, 0.0
d = np.linspace(0.05, 10, 500)
freqs = [900, 1900, 3500]
say("=" * 70)
say(f"Part (a): Friis, P_TX = {PTX} dBm, G_TX = {GTX} dBi, G_RX = {GRX} dBi")
fig, ax = plt.subplots(figsize=(8, 4.5))
for f in freqs:
    p = friis_rx_power(PTX, GTX, GRX, d, f)
    ax.plot(d, p, label=f"{f} MHz")
    say(f"  {f:5d} MHz: P_RX(0.05 km) = {p[0]:7.2f} dBm, P_RX(1 km) = "
        f"{friis_rx_power(PTX, GTX, GRX, 1.0, f):7.2f} dBm, P_RX(10 km) = {p[-1]:7.2f} dBm")
ax.set_xscale("log")
ax.set_xlabel("distance (km, log scale)")
ax.set_ylabel("received power (dBm)")
ax.set_title("Friis received power, P_TX = 43 dBm, G_TX = 15 dBi, G_RX = 0 dBi", fontsize=10)
ax.grid(alpha=0.35, which="both")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_friis.png"), dpi=160)
plt.close(fig)

d_check = friis_rx_power(PTX, GTX, GRX, 1.0, 1900) - friis_rx_power(PTX, GTX, GRX, 2.0, 1900)
f_check = friis_rx_power(PTX, GTX, GRX, 1.0, 900) - friis_rx_power(PTX, GTX, GRX, 1.0, 1800)
say(f"  6 dB check: P_RX(1 km) - P_RX(2 km) at 1900 MHz = {d_check:.4f} dB")
say(f"  6 dB check: P_RX(900 MHz) - P_RX(1800 MHz) at 1 km = {f_check:.4f} dB")
say(f"  exact value 20*log10(2) = {20 * np.log10(2):.4f} dB")


# =============================================================== Part (b): modulation
def gray_levels(bits_per_axis):
    """PAM levels indexed by the Gray-coded bit pattern of one axis (MSB first)."""
    L = 2 ** bits_per_axis
    levels = np.arange(-(L - 1), L, 2, dtype=float)          # -3 -1 1 3 for L = 4
    gray = np.arange(L) ^ (np.arange(L) >> 1)               # Gray code of each position
    lut = np.empty(L)
    lut[gray] = levels                                      # pattern -> level
    return lut, levels, gray


class QAM:
    """Square Gray-coded M-QAM (M = 4, 16, 64) normalised to unit average symbol energy.
    M = 2 is BPSK on the real axis: bit 0 -> -1, bit 1 -> +1."""

    def __init__(self, M):
        self.M = M
        self.k = int(np.log2(M))
        if M == 2:
            self.scale = 1.0
            return
        self.ka = self.k // 2
        self.lut, self.levels, self.gray = gray_levels(self.ka)
        L = 2 ** self.ka
        self.scale = np.sqrt(2 * (L ** 2 - 1) / 3)          # average energy of square M-QAM
        self.w = 2 ** np.arange(self.ka - 1, -1, -1)

    def mod(self, bits):
        if self.M == 2:
            return (2.0 * bits - 1.0).astype(complex)
        b = bits.reshape(-1, self.k)
        iv = b[:, :self.ka] @ self.w
        qv = b[:, self.ka:] @ self.w
        return (self.lut[iv] + 1j * self.lut[qv]) / self.scale

    def demod(self, y):
        """Minimum-distance decision, done per axis (exact for square QAM)."""
        if self.M == 2:
            return (y.real > 0).astype(np.int8)
        L = 2 ** self.ka
        out = []
        for comp in (y.real * self.scale, y.imag * self.scale):
            pos = np.clip(np.round((comp + (L - 1)) / 2), 0, L - 1).astype(int)   # nearest level index
            pat = self.gray[pos]                                                   # its Gray pattern
            out.append(((pat[:, None] >> np.arange(self.ka - 1, -1, -1)) & 1).astype(np.int8))
        return np.hstack(out).reshape(-1)

    def theory(self, ebn0_lin):
        if self.M in (2, 4):
            return 0.5 * erfc(np.sqrt(ebn0_lin))
        M, k = self.M, self.k
        return (2 / k) * (1 - 1 / np.sqrt(M)) * erfc(np.sqrt(3 * k * ebn0_lin / (2 * (M - 1))))


def awgn(symbols, ebn0_dB, k, rate=1.0):
    """Complex AWGN for unit-energy symbols: Es = 1, Eb = Es / (k * rate), N0 = Eb / (Eb/N0)."""
    ebn0 = 10 ** (ebn0_dB / 10)
    N0 = 1.0 / (k * rate * ebn0)
    noise = np.sqrt(N0 / 2) * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
    return symbols + noise


def ber_curve(mod, ebn0_list, nbits):
    bits = np.random.randint(0, 2, size=nbits - nbits % mod.k).astype(np.int8)
    tx = mod.mod(bits)
    out = []
    for e in ebn0_list:
        rx = awgn(tx, e, mod.k)
        errors = np.count_nonzero(mod.demod(rx) != bits)
        out.append((errors, len(bits)))
    return out


def ebn0_at(ber_target, ebn0, ber):
    """Interpolate log10(BER) linearly between sweep points to find Eb/N0 at a target BER."""
    ber = np.asarray(ber, float)
    lb = np.log10(np.maximum(ber, 1e-12))
    for i in range(len(ebn0) - 1):
        if ber[i] >= ber_target > ber[i + 1] > 0:
            t = (np.log10(ber_target) - lb[i]) / (lb[i + 1] - lb[i])
            return ebn0[i] + t * (ebn0[i + 1] - ebn0[i])
    return np.nan


# unit test of the mappers: Gray property and round trip
for M in (4, 16, 64):
    q = QAM(M)
    pts = q.mod(np.array([(i >> np.arange(q.k - 1, -1, -1)) & 1 for i in range(M)]).reshape(-1).astype(np.int8))
    assert abs(np.mean(abs(pts) ** 2) - 1) < 1e-12
    dmin = min(abs(a - b) for i, a in enumerate(pts) for b in pts[i + 1:])
    for i in range(M):
        for j in range(M):
            if i != j and abs(abs(pts[i] - pts[j]) - dmin) < 1e-9:
                assert bin(i ^ j).count("1") == 1, "Gray mapping violated"
    rb = np.random.randint(0, 2, 6000).astype(np.int8)
    assert np.array_equal(q.demod(q.mod(rb)), rb)
say("  mapper self-test passed: unit energy, nearest neighbours differ in one bit, noiseless round trip exact")

NB = 1_000_000
ebn0 = np.arange(0, 17, 1)
schemes = {"BPSK": QAM(2), "QPSK": QAM(4), "16-QAM": QAM(16), "64-QAM": QAM(64)}
ber = {}
say("=" * 70)
say(f"Part (b) and lab tasks 1, 3: Monte Carlo BER, {NB:,} bits per point, Eb/N0 0-16 dB")
for name, m in schemes.items():
    res = ber_curve(m, ebn0, NB)
    ber[name] = np.array([e / n for e, n in res])
say("  Eb/N0  " + "".join(f"{n:>12s}" for n in schemes) + "   BPSK theory")
for i, e in enumerate(ebn0):
    say(f"  {e:5d}  " + "".join(f"{ber[n][i]:>12.2e}" for n in schemes)
        + f"   {schemes['BPSK'].theory(10 ** (e / 10)):.2e}")

target = 1e-3
need = {n: ebn0_at(target, ebn0, ber[n]) for n in schemes}
say(f"  Eb/N0 needed for BER = {target:g}:")
for n, m in schemes.items():
    eta = m.k if m.M > 2 else 1
    shannon_min = 10 * np.log10((2 ** eta - 1) / eta)
    esn0 = need[n] + 10 * np.log10(m.k)
    say(f"    {n:7s} Eb/N0 = {need[n]:5.2f} dB  (Es/N0 = SNR = {esn0:5.2f} dB),  "
        f"{eta} bit/s/Hz, Shannon minimum Eb/N0 at that efficiency = {shannon_min:5.2f} dB, "
        f"gap to Shannon = {need[n] - shannon_min:4.2f} dB")
gap = need["16-QAM"] - need["BPSK"]
say(f"  16-QAM - BPSK gap at BER 1e-3: {gap:.2f} dB in Eb/N0, "
    f"{(need['16-QAM'] + 10 * np.log10(4)) - need['BPSK']:.2f} dB in SNR per symbol (Es/N0)")

fig, ax = plt.subplots(figsize=(8, 5))
lin = np.linspace(0, 16, 200)
cols = {"BPSK": "#1f77b4", "QPSK": "#2ca02c", "16-QAM": "#ff7f0e", "64-QAM": "#d62728"}
mk = {"BPSK": "o", "QPSK": "s", "16-QAM": "^", "64-QAM": "D"}
for n, m in schemes.items():
    b = ber[n].copy()
    b[b == 0] = np.nan
    ax.semilogy(ebn0, b, mk[n], color=cols[n], ms=5, label=f"{n} simulated")
    ax.semilogy(lin, m.theory(10 ** (lin / 10)), "-", color=cols[n], lw=1, alpha=0.8,
                label=f"{n} theory" if n in ("BPSK", "16-QAM", "64-QAM") else None)
ax.axhline(target, color="#7f7f7f", ls=":", lw=1)
ax.annotate("", xy=(need["16-QAM"], target), xytext=(need["BPSK"], target),
            arrowprops=dict(arrowstyle="<->", color="black"))
ax.text((need["16-QAM"] + need["BPSK"]) / 2, target * 1.6, f"{gap:.1f} dB", ha="center", fontsize=9)
ax.set_ylim(1e-6, 1)
ax.set_xlim(0, 16)
ax.set_xlabel("Eb/N0 (dB)")
ax.set_ylabel("bit error rate")
ax.set_title(f"BER over AWGN, {NB:,} bits per point (QPSK overlaps BPSK)", fontsize=10)
ax.grid(alpha=0.35, which="both")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_ber.png"), dpi=160)
plt.close(fig)

# constellation picture for the report
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, (n, M, e) in zip(axes, (("16-QAM", 16, 12), ("16-QAM", 16, 6))):
    q = QAM(M)
    bb = np.random.randint(0, 2, 4 * 3000).astype(np.int8)
    y = awgn(q.mod(bb), e, q.k)
    ax.plot(y.real, y.imag, ".", ms=1.5, alpha=0.5)
    ax.plot((q.lut[:, None] + 0j + 1j * q.lut[None, :]).real.ravel() / q.scale,
            (q.lut[:, None] + 1j * q.lut[None, :]).imag.ravel() / q.scale, "r+", ms=10)
    ax.set_title(f"{n} at Eb/N0 = {e} dB", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.35)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_16qam_constellation.png"), dpi=160)
plt.close(fig)


# =============================================================== Lab task 2: convolutional code
G = (0b111, 0b101)          # (7,5) octal, constraint length 3, rate 1/2


def conv_encode(bits):
    s = 0
    out = np.empty(2 * len(bits), np.int8)
    for i, b in enumerate(bits):
        reg = (b << 2) | s
        out[2 * i] = bin(reg & G[0]).count("1") & 1
        out[2 * i + 1] = bin(reg & G[1]).count("1") & 1
        s = reg >> 1
    return out


# trellis: state = two previous bits (b_{t-1}, b_{t-2}); next state = (b_t, b_{t-1})
NEXT = np.zeros((4, 2), int)
OUT = np.zeros((4, 2, 2))
for s in range(4):
    for b in range(2):
        reg = (b << 2) | s
        NEXT[s, b] = reg >> 1
        OUT[s, b] = [2 * (bin(reg & G[0]).count("1") & 1) - 1, 2 * (bin(reg & G[1]).count("1") & 1) - 1]
PREV = [[(s, b) for s in range(4) for b in range(2) if NEXT[s, b] == ns] for ns in range(4)]


def viterbi_soft(rx_batch):
    """Soft-decision Viterbi for BPSK (bit 0 -> -1, 1 -> +1), all Eb/N0 points at once.
    rx_batch: shape (P, 2n) real received values. Branch metric = correlation (max)."""
    P, L2 = rx_batch.shape
    n = L2 // 2
    pm = np.full((P, 4), -1e18)
    pm[:, 0] = 0.0
    surv = np.empty((n, P, 4), np.int8)            # predecessor index into PREV list
    r0, r1 = rx_batch[:, 0::2], rx_batch[:, 1::2]
    for t in range(n):
        new = np.empty((P, 4))
        for ns in range(4):
            (s_a, b_a), (s_b, b_b) = PREV[ns]
            ma = pm[:, s_a] + OUT[s_a, b_a, 0] * r0[:, t] + OUT[s_a, b_a, 1] * r1[:, t]
            mb = pm[:, s_b] + OUT[s_b, b_b, 0] * r0[:, t] + OUT[s_b, b_b, 1] * r1[:, t]
            pick = mb > ma
            new[:, ns] = np.where(pick, mb, ma)
            surv[t, :, ns] = pick
        pm = new
    # trellis terminated in state 0 by two tail bits
    dec = np.empty((P, n), np.int8)
    state = np.zeros(P, int)
    for t in range(n - 1, -1, -1):
        choice = surv[t, np.arange(P), state]
        for p in range(P):
            s_prev, b = PREV[state[p]][choice[p]]
            dec[p, t] = b
            state[p] = s_prev
    return dec


say("=" * 70)
NI = 600_000
info = np.random.randint(0, 2, NI).astype(np.int8)
coded = conv_encode(np.concatenate([info, [0, 0]]).astype(np.int8))
txc = 2.0 * coded - 1.0
eb_c = np.arange(0, 7.01, 0.5)
rx = np.vstack([awgn(txc.astype(complex), e, 1, rate=0.5).real for e in eb_c])
dec = viterbi_soft(rx)[:, :NI]
ber_coded = np.array([np.count_nonzero(dec[i] != info) / NI for i in range(len(eb_c))])
say(f"Lab task 2: (7,5) rate-1/2 convolutional code, soft Viterbi, {NI:,} info bits per point")
for e, bc in zip(eb_c, ber_coded):
    unc = schemes['BPSK'].theory(10 ** (e / 10))
    say(f"  Eb/N0 {e:3.1f} dB: coded BER {bc:.2e}   uncoded BPSK (theory) {unc:.2e}")
need_c3 = ebn0_at(1e-3, eb_c, ber_coded)
need_c4 = ebn0_at(1e-4, eb_c, ber_coded)
need_u4 = ebn0_at(1e-4, ebn0, ber["BPSK"])
say(f"  coding gain at 1e-3: {need['BPSK'] - need_c3:.2f} dB, at 1e-4: {need_u4 - need_c4:.2f} dB")
fig, ax = plt.subplots(figsize=(7, 4.5))
b = ber_coded.copy()
b[b == 0] = np.nan
bu = ber["BPSK"][:11].copy()
bu[bu == 0] = np.nan
ax.semilogy(ebn0[:11], bu, "o-", label="uncoded BPSK (simulated)")
ax.semilogy(eb_c, b, "s-", label="rate-1/2 (7,5) code, soft Viterbi")
for tgt, nc, nu in ((1e-3, need_c3, need["BPSK"]), (1e-4, need_c4, need_u4)):
    ax.annotate("", xy=(nc, tgt), xytext=(nu, tgt), arrowprops=dict(arrowstyle="<->"))
    ax.text((nc + nu) / 2, tgt * 1.5, f"{nu - nc:.1f} dB", ha="center", fontsize=9)
ax.set_xlabel("Eb/N0 per information bit (dB)")
ax.set_ylabel("bit error rate")
ax.set_ylim(1e-6, 1)
ax.grid(alpha=0.35, which="both")
ax.legend(fontsize=8)
ax.set_title("Coding gain of a rate-1/2 convolutional code", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_coding_gain.png"), dpi=160)
plt.close(fig)

# =============================================================== Lab tasks 4-5: log-distance + shadowing
say("=" * 70)
f0 = 1900
d0 = 0.1                                             # km reference distance
PL0 = 20 * np.log10(d0) + 20 * np.log10(f0) + 32.44  # Friis loss at d0
dd = np.linspace(0.1, 10, 600)
say(f"Lab task 4: log-distance model at {f0} MHz, d0 = {d0} km, PL(d0) = {PL0:.2f} dB (Friis)")
fig, ax = plt.subplots(figsize=(8, 4.5))
for n_exp in (2.0, 2.7, 3.5, 4.0):
    prx = PTX + GTX + GRX - (PL0 + 10 * n_exp * np.log10(dd / d0))
    ax.plot(dd, prx, label=f"n = {n_exp}")
    say(f"  n = {n_exp}: P_RX(1 km) = {PTX + GTX + GRX - (PL0 + 10 * n_exp * np.log10(1 / d0)):7.2f} dBm, "
        f"P_RX(10 km) = {prx[-1]:7.2f} dBm, slope {10 * n_exp:.0f} dB/decade")
ax.axhline(-100, color="black", ls="--", lw=1, label="-100 dBm example sensitivity")
ax.set_xscale("log")
ax.set_xlabel("distance (km, log scale)")
ax.set_ylabel("received power (dBm)")
ax.set_title(f"Log-distance path loss at {f0} MHz (n = 2 is free space)", fontsize=10)
ax.grid(alpha=0.35, which="both")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_log_distance.png"), dpi=160)
plt.close(fig)
for n_exp in (2.0, 2.7, 3.5, 4.0):
    dmax = d0 * 10 ** ((PTX + GTX + GRX - PL0 + 100) / (10 * n_exp))
    say(f"    range to -100 dBm with n = {n_exp}: {dmax:8.2f} km")

sigma = 8.0
n_exp = 3.5
mean = PTX + GTX + GRX - (PL0 + 10 * n_exp * np.log10(dd / d0))
say(f"Lab task 5: log-normal shadowing, n = {n_exp}, sigma = {sigma} dB, 4 realisations")
fig, ax = plt.subplots(figsize=(8, 4.5))
for r in range(4):
    x = np.random.randn(len(dd)) * sigma
    ax.plot(dd, mean + x, lw=0.6, alpha=0.75, label=f"realisation {r + 1}")
ax.plot(dd, mean, "k", lw=2, label="mean (no shadowing)")
ax.plot(dd, mean - 1.645 * sigma, "k--", lw=1, label="mean - 1.645 sigma (95% coverage margin)")
ax.set_xscale("log")
ax.set_xlabel("distance (km, log scale)")
ax.set_ylabel("received power (dBm)")
ax.set_title(f"Log-normal shadowing on the n = {n_exp} model, sigma = {sigma:.0f} dB", fontsize=10)
ax.grid(alpha=0.35, which="both")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_shadowing.png"), dpi=160)
plt.close(fig)
samples = np.random.randn(200000) * sigma
say(f"  check: sample std of the shadowing term = {samples.std():.2f} dB; "
    f"fraction of locations below mean - 1.645 sigma = {np.mean(samples < -1.645 * sigma):.3f}")
for n2, s2 in ((3.5, 4.0), (3.5, 8.0)):
    dm = d0 * 10 ** ((PTX + GTX + GRX - PL0 + 100 - 1.645 * s2) / (10 * n2))
    say(f"  range to -100 dBm with a 95% shadowing margin (n = {n2}, sigma = {s2}): {dm:.2f} km")

# =============================================================== Lab task 6: throughput proxy
say("=" * 70)
say("Lab task 6: throughput proxy k(1 - BER) against Shannon capacity log2(1 + SNR), SNR = Es/N0")
fig, ax = plt.subplots(figsize=(8, 4.5))
snr_db = np.linspace(-5, 30, 300)
ax.plot(snr_db, np.log2(1 + 10 ** (snr_db / 10)), "k", lw=2, label="Shannon log2(1 + SNR)")
for n, m in schemes.items():
    s = ebn0 + 10 * np.log10(m.k)
    thr = m.k * (1 - ber[n])
    p_ = np.clip(ber[n], 1e-15, 0.5)
    h2 = -(p_ * np.log2(p_) + (1 - p_) * np.log2(1 - p_))
    info_rate = m.k * (1 - h2)                       # capacity of the bit pipe after hard decisions
    ax.plot(s, thr, marker=mk[n], color=cols[n], ms=4, label=f"{n}: {m.k}(1 - BER)")
    ax.plot(s, info_rate, ls="--", color=cols[n], lw=1)
    i10 = np.argmin(abs(s - 10))
    s99 = s[np.argmax(thr >= 0.99 * m.k)]
    say(f"  {n:7s} at SNR {s[i10]:5.1f} dB: k(1-BER) = {thr[i10]:.3f}, k(1-h2(BER)) = {info_rate[i10]:.3f}, "
        f"Shannon {np.log2(1 + 10 ** (s[i10] / 10)):.3f} bit/s/Hz; reaches 99% of {m.k} bits at SNR ~{s99:.1f} dB")
ax.set_xlabel("SNR per symbol, Es/N0 (dB)")
ax.set_ylabel("bits per symbol (per channel use)")
ax.plot([], [], "k--", lw=1, label="dashed: k(1 - h2(BER)), information rate")
ax.set_title("Throughput proxy versus the Shannon-Hartley ceiling", fontsize=10)
ax.set_xlim(-5, 30)
ax.grid(alpha=0.35)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_throughput_vs_shannon.png"), dpi=160)
plt.close(fig)

with open(os.path.join(HERE, "results_2_1.txt"), "w") as fh:
    fh.write("\n".join(LOG) + "\n")
print("figures written to", FIG)
