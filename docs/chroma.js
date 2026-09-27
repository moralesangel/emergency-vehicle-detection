// A port of librosa.feature.chroma_stft with its default settings
// (n_fft 2048, hop 512, Hann window, power 2, norm=inf per frame).
// Only what the demo needs is implemented, but the numbers match librosa.

const N_FFT = 2048;
const HOP = 512;
const BINS = 12;

/** Periodic Hann window, as scipy/librosa use for STFT. */
function hann(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

/** In-place iterative radix-2 FFT on split real/imaginary arrays. */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k];
        const ai = im[i + k];
        const br = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const bi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ar + br;
        im[i + k] = ai + bi;
        re[i + k + len / 2] = ar - br;
        im[i + k + len / 2] = ai - bi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/**
 * librosa's chroma filterbank, exported from
 * librosa.filters.chroma(sr=16000, n_fft=2048) and stored sparsely.
 * Reimplementing it here was error-prone: the octave folding and per-bin
 * widths have to match librosa exactly or every value shifts, so the real
 * matrix ships as data instead.
 */
function expandFilterbank(sparse, nBins) {
  return sparse.map((row) => {
    const dense = new Float32Array(nBins);
    for (let k = 0; k < row.i.length; k++) dense[row.i[k]] = row.v[k];
    return dense;
  });
}

/**
 * Chroma for a mono signal, returned as 12 rows of frames.
 * Centre-padded with reflection like librosa's default center=True.
 */
export function chromaStft(y, sr, filterbank) {
  const pad = N_FFT / 2;
  const padded = new Float32Array(y.length + 2 * pad);
  padded.set(y, pad);
  // numpy "reflect": the edge sample is not repeated, so the left pad runs
  // y[pad], y[pad-1], ... y[1] and the right pad mirrors the tail the same way.
  for (let i = 0; i < pad; i++) {
    padded[i] = y[Math.min(pad - i, y.length - 1)];
    padded[pad + y.length + i] = y[Math.max(y.length - 2 - i, 0)];
  }

  const win = hann(N_FFT);
  const frames = 1 + Math.floor((padded.length - N_FFT) / HOP);
  const nBins = N_FFT / 2 + 1;
  const fb = expandFilterbank(filterbank, nBins);
  const out = Array.from({ length: BINS }, () => new Float32Array(frames));

  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);
  const power = new Float64Array(nBins);

  for (let f = 0; f < frames; f++) {
    const off = f * HOP;
    for (let i = 0; i < N_FFT; i++) {
      re[i] = padded[off + i] * win[i];
      im[i] = 0;
    }
    fft(re, im);
    for (let i = 0; i < nBins; i++) power[i] = re[i] * re[i] + im[i] * im[i];

    let maxv = 0;
    const col = new Float64Array(BINS);
    for (let c = 0; c < BINS; c++) {
      let s = 0;
      const row = fb[c];
      for (let i = 0; i < nBins; i++) s += row[i] * power[i];
      col[c] = s;
      if (s > maxv) maxv = s;
    }
    // librosa normalises each frame by its max (norm=inf).
    for (let c = 0; c < BINS; c++) out[c][f] = maxv > 0 ? col[c] / maxv : 0;
  }

  return out.map((r) => Array.from(r));
}
