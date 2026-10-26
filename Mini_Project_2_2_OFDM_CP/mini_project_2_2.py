"""
Mini Project 2.2 - OFDM waveform generator with a cyclic prefix visualizer.

Chain: random bits -> Gray 16-QAM -> N = 64 subcarriers -> IFFT -> cyclic prefix ->
3-tap multipath FIR channel (delay spread 7 samples) over a stream of back-to-back symbols ->
remove CP -> FFT -> one-tap zero-forcing equaliser -> decisions.

The same channel realisation is used for every CP length, so the only thing that changes
between runs is the cyclic prefix.

    python Mini_Project_2_2_OFDM_CP/mini_project_2_2.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
LOG = []


def say(s=""):
    print(s)
    LOG.append(s)


N = 64                       # subcarriers (all used as data for clarity)
NSYM = 200                   # OFDM symbols in the stream (statistics); figures use the first 3
TAP_DELAYS = np.array([0, 3, 7])
TAP_GAINS = np.array([1.0, 0.6 * np.exp(1j * 0.9), 0.35 * np.exp(-1j * 2.1)])
DELAY_SPREAD = TAP_DELAYS.max()            # maximum excess delay in samples
h = np.zeros(DELAY_SPREAD + 1, complex)
h[TAP_DELAYS] = TAP_GAINS
H_true = np.fft.fft(h, N)                  # channel frequency response on the subcarrier grid

# ------------------------------------------------------------------ 16-QAM Gray mapper
LUT = np.array([-3, -1, 3, 1], float)      # 2-bit Gray pattern 00,01,10,11 -> -3,-1,+3,+1
SCALE = np.sqrt(10)
INV = {-3: (0, 0), -1: (0, 1), 1: (1, 1), 3: (1, 0)}


def qam16_mod(bits):
    b = bits.reshape(-1, 4)
    return (LUT[2 * b[:, 0] + b[:, 1]] + 1j * LUT[2 * b[:, 2] + b[:, 3]]) / SCALE


def qam16_demod(sym):
    out = []
    for comp in (sym.real * SCALE, sym.imag * SCALE):
        lv = np.clip(2 * np.round((comp + 3) / 2) - 3, -3, 3).astype(int)
        out.append(np.array([INV[v] for v in lv]))
    return np.hstack(out).reshape(-1)


def qam16_slice(sym):
    re = np.clip(2 * np.round((sym.real * SCALE + 3) / 2) - 3, -3, 3)
    im = np.clip(2 * np.round((sym.imag * SCALE + 3) / 2) - 3, -3, 3)
    return (re + 1j * im) / SCALE


# unit tests (suggested approach steps 1-3)
pts = qam16_mod(np.array([(i >> np.arange(3, -1, -1)) & 1 for i in range(16)]).reshape(-1))
assert abs(np.mean(abs(pts) ** 2) - 1) < 1e-12
dmin = 2 / SCALE
for i in range(16):
    for j in range(16):
        if i != j and abs(abs(pts[i] - pts[j]) - dmin) < 1e-9:
            assert bin(i ^ j).count("1") == 1
tb = np.random.randint(0, 2, 4 * 1000)
assert np.array_equal(qam16_demod(qam16_mod(tb)), tb)
X0 = qam16_mod(np.random.randint(0, 2, 4 * N))
x0 = np.fft.ifft(X0)
roundtrip = np.max(abs(np.fft.fft(x0) - X0))
say("=" * 70)
say("Self-checks")
say(f"  16-QAM mapper: unit average energy, Gray neighbours, noiseless round trip OK")
say(f"  IFFT -> FFT round trip, max |error| = {roundtrip:.2e}")


def add_cp(x, cp):
    return np.concatenate([x[-cp:], x]) if cp > 0 else x.copy()


xc = add_cp(x0, 16)
say(f"  CP insertion: prefix equals the last 16 samples? {np.allclose(xc[:16], x0[-16:])}")
say(f"Channel taps at delays {TAP_DELAYS.tolist()} samples, |gains| = "
    f"{np.round(abs(TAP_GAINS), 2).tolist()}, delay spread = {DELAY_SPREAD} samples")


# ------------------------------------------------------------------ transmitter / channel / receiver
def transmit(bits, cp):
    X = qam16_mod(bits).reshape(NSYM, N)
    x = np.fft.ifft(X, axis=1) * np.sqrt(N)              # unit average sample power
    stream = np.concatenate([add_cp(sym, cp) for sym in x])
    return X, stream


def channel(stream, snr_db=None, rng=None):
    y = np.convolve(stream, h)[:len(stream)]             # FIR multipath, symbols smear into each other
    if snr_db is not None:
        p = np.mean(abs(stream) ** 2)
        n0 = p / 10 ** (snr_db / 10)
        y = y + np.sqrt(n0 / 2) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
    return y


def receive(y, cp):
    L = N + cp
    Y = np.empty((NSYM, N), complex)
    for k in range(NSYM):
        seg = y[k * L + cp: k * L + cp + N]                # discard CP, keep N samples
        Y[k] = np.fft.fft(seg) / np.sqrt(N)
    return Y / H_true                                    # one-tap zero-forcing equaliser


bits = np.random.randint(0, 2, 4 * N * NSYM)


def run(cp, snr_db=None, seed=1):
    rng = np.random.default_rng(seed)                    # same noise realisation for every CP
    X, s = transmit(bits, cp)
    Xh = receive(channel(s, snr_db, rng), cp)
    rb = qam16_demod(Xh.reshape(-1))
    ber = np.mean(rb != bits)
    ser = np.mean(qam16_slice(Xh.reshape(-1)) != X.reshape(-1))
    mse = np.mean(abs(Xh - X) ** 2)
    return Xh, X, ser, ber, mse


say("=" * 70)
SNR = 30
say(f"CP sweep (same channel; noiseless and with AWGN at SNR = {SNR} dB)")
say(f"{'CP':>4s}{'overhead':>10s}{'SER (no noise)':>16s}{'BER (no noise)':>16s}{'MSE (no noise)':>16s}"
    f"{'SER (30 dB)':>13s}{'EVM (30 dB)':>13s}")
cps = list(range(0, 17))
tab = []
for cp in cps:
    _, _, ser0, ber0, mse0 = run(cp, None)
    _, _, ser1, ber1, mse1 = run(cp, SNR)
    evm_db = 10 * np.log10(mse1)
    tab.append((cp, ser0, ber0, mse0, ser1, evm_db))
    say(f"{cp:>4d}{100 * cp / (N + cp):>9.1f}%{ser0:>16.4f}{ber0:>16.4f}{mse0:>16.2e}{ser1:>13.4f}{evm_db:>11.1f} dB")

# ------------------------------------------------------------------ figure: waveform with CP
CPF = 16
Xd, sd = transmit(bits, CPF)
yd = channel(sd)
fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
L = N + CPF
n = np.arange(3 * L)
axes[0].plot(n, abs(sd[:3 * L]), lw=1, label="|x[n]| transmitted")
axes[1].plot(n, abs(yd[:3 * L]), lw=1, color="#ff7f0e", label="|y[n]| after the multipath channel")
for ax in axes:
    for k in range(3):
        ax.axvspan(k * L, k * L + CPF, color="#2ca02c", alpha=0.18,
                   label="cyclic prefix" if k == 0 else None)
        if ax is axes[1]:
            ax.axvspan(k * L, k * L + DELAY_SPREAD, color="#d62728", alpha=0.25,
                       label=f"smeared by previous symbol ({DELAY_SPREAD} samples)" if k == 0 else None)
        ax.axvline(k * L, color="black", lw=0.6)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
axes[0].set_title(f"Three back-to-back OFDM symbols, N = {N}, CP = {CPF} samples", fontsize=10)
axes[1].set_xlabel("sample index n")
for k in range(3):
    axes[0].text(k * L + CPF + N / 2, axes[0].get_ylim()[1] * 0.92, f"symbol {k + 1}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_ofdm_waveform_cp.png"), dpi=160)
plt.close(fig)

# CP copy check figure: tail of symbol 1 vs its prefix
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(range(CPF), sd[:CPF].real, "o-", label="prefix samples (start of symbol 1)")
ax.plot(range(CPF), sd[N:N + CPF].real, "x--", ms=8, label="last 16 samples of symbol 1")
ax.set_xlabel("sample within the prefix")
ax.set_ylabel("real part")
ax.set_title("The cyclic prefix is an exact copy of the symbol's tail", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_cp_copy.png"), dpi=160)
plt.close(fig)

# ------------------------------------------------------------------ figure: channel
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].stem(range(len(h)), abs(h))
axes[0].set_xlabel("delay (samples)")
axes[0].set_ylabel("|h|")
axes[0].set_title(f"Impulse response, delay spread {DELAY_SPREAD} samples", fontsize=10)
axes[0].grid(alpha=0.3)
axes[1].plot(range(N), 20 * np.log10(abs(H_true)), ".-")
axes[1].set_xlabel("subcarrier index k")
axes[1].set_ylabel("|H[k]| (dB)")
axes[1].set_title("Frequency response seen by the subcarriers", fontsize=10)
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_channel.png"), dpi=160)
plt.close(fig)

# ------------------------------------------------------------------ figure: constellations
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, cp in zip(axes, (16, 4, 0)):
    Xh, X, ser, ber, mse = run(cp, None)
    ax.plot(Xh.real.ravel(), Xh.imag.ravel(), ".", ms=1.5, alpha=0.5)
    ax.plot(pts.real, pts.imag, "r+", ms=10, mew=1.5)
    state = "sufficient" if cp >= DELAY_SPREAD else ("too short" if cp > 0 else "no CP")
    ax.set_title(f"CP = {cp} ({state})\nSER = {ser:.3f}, MSE = {mse:.1e}", fontsize=10)
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
fig.suptitle("Recovered 16-QAM constellations, same channel, no noise", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_constellations.png"), dpi=160)
plt.close(fig)

# ------------------------------------------------------------------ figure: sweep
arr = np.array(tab)
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.semilogy(arr[:, 0], np.maximum(arr[:, 1], 1e-5), "o-", label="SER, no noise")
ax.semilogy(arr[:, 0], np.maximum(arr[:, 4], 1e-5), "s-", label=f"SER, SNR {SNR} dB")
ax.axvline(DELAY_SPREAD, color="#d62728", ls="--", label=f"delay spread = {DELAY_SPREAD} samples")
ax.set_xlabel("cyclic prefix length (samples)")
ax.set_ylabel("symbol error rate (floored at 1e-5 for plotting)")
ax.set_xticks(cps)
ax.grid(alpha=0.35, which="both")
ax2 = ax.twinx()
ax2.plot(arr[:, 0], arr[:, 5], "^-", color="#9467bd", label=f"EVM, SNR {SNR} dB")
ax2.set_ylabel("EVM (dB)", color="#9467bd")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2[:1], l1 + l2[:1], fontsize=8, loc="center right")
ax.set_ylim(1e-5, 4)
ax.set_title("Error rate versus CP length: threshold at the delay spread, then a plateau", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_ser_vs_cp.png"), dpi=160)
plt.close(fig)

with open(os.path.join(HERE, "results_2_2.txt"), "w") as fh:
    fh.write("\n".join(LOG) + "\n")
print("figures written to", FIG)
