// Minimal forward pass for the encoder + CNN exported from Keras.
// Only the layer types these two models use are implemented; each mirrors the
// Keras semantics (channels-last, "same"/"valid" padding, ReLU/sigmoid).

const relu = (x) => (x > 0 ? x : 0);
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/** A dense tensor of shape [h][w][c] held flat, row-major. */
function tensor(h, w, c) {
  return { h, w, c, d: new Float32Array(h * w * c) };
}
const at = (t, y, x, k) => t.d[(y * t.w + x) * t.c + k];
const set = (t, y, x, k, v) => {
  t.d[(y * t.w + x) * t.c + k] = v;
};

/**
 * 2D convolution with stride 1.
 * W has shape [kh][kw][inC][outC] as a nested array, b has length outC.
 */
function conv2d(input, layer) {
  const [kh, kw] = layer.k;
  const W = layer.W;
  const b = layer.b;
  const outC = b.length;
  const same = layer.pad === 'same';
  // Keras "same" with odd kernels pads floor((k-1)/2) on the top/left.
  const padY = same ? Math.floor((kh - 1) / 2) : 0;
  const padX = same ? Math.floor((kw - 1) / 2) : 0;
  const outH = same ? input.h : input.h - kh + 1;
  const outW = same ? input.w : input.w - kw + 1;
  const out = tensor(outH, outW, outC);
  const act = layer.act === 'relu' ? relu : layer.act === 'sigmoid' ? sigmoid : (v) => v;

  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      for (let o = 0; o < outC; o++) {
        let sum = b[o];
        for (let i = 0; i < kh; i++) {
          const sy = y + i - padY;
          if (sy < 0 || sy >= input.h) continue;
          for (let j = 0; j < kw; j++) {
            const sx = x + j - padX;
            if (sx < 0 || sx >= input.w) continue;
            const Wij = W[i][j];
            for (let ic = 0; ic < input.c; ic++) {
              sum += at(input, sy, sx, ic) * Wij[ic][o];
            }
          }
        }
        set(out, y, x, o, act(sum));
      }
    }
  }
  return out;
}

function maxPool2d(input, layer) {
  const [ph, pw] = layer.pool;
  const [sy, sx] = layer.strides;
  const same = layer.pad === 'same';
  const outH = same ? Math.ceil(input.h / sy) : Math.floor((input.h - ph) / sy) + 1;
  const outW = same ? Math.ceil(input.w / sx) : Math.floor((input.w - pw) / sx) + 1;
  const out = tensor(outH, outW, input.c);

  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      for (let k = 0; k < input.c; k++) {
        let m = -Infinity;
        for (let i = 0; i < ph; i++) {
          const py = y * sy + i;
          if (py >= input.h) continue;
          for (let j = 0; j < pw; j++) {
            const px = x * sx + j;
            if (px >= input.w) continue;
            const v = at(input, py, px, k);
            if (v > m) m = v;
          }
        }
        set(out, y, x, k, m === -Infinity ? 0 : m);
      }
    }
  }
  return out;
}

/** Keras Flatten is row-major over [h][w][c]; the data is already in that order. */
const flatten = (t) => Array.from(t.d);

function dense(vec, layer) {
  const W = layer.W;
  const b = layer.b;
  const out = new Array(b.length);
  const act = layer.act === 'relu' ? relu : layer.act === 'sigmoid' ? sigmoid : (v) => v;
  for (let o = 0; o < b.length; o++) {
    let sum = b[o];
    for (let i = 0; i < vec.length; i++) sum += vec[i] * W[i][o];
    out[o] = act(sum);
  }
  return out;
}

/** Run a list of exported layers over a [h][w][c] tensor. */
function runLayers(input, layers) {
  let x = input;
  let v = null;
  for (const layer of layers) {
    switch (layer.type) {
      case 'conv':
        x = conv2d(x, layer);
        break;
      case 'pool':
        x = maxPool2d(x, layer);
        break;
      case 'flatten':
        v = flatten(x);
        break;
      case 'dense':
        v = dense(v, layer);
        break;
      // Dropout is inference-time identity and is not exported.
    }
  }
  return v ?? x;
}

/**
 * Chroma (12 x frames) to siren probability, mirroring backend/api.py:
 * standardise, drop the last frame, transpose to (frames, 12), encode, classify.
 */
export function predict(spec, chroma12xN) {
  const bins = chroma12xN.length;
  const frames = chroma12xN[0].length;

  let mean = 0;
  for (let b = 0; b < bins; b++) for (let f = 0; f < frames; f++) mean += chroma12xN[b][f];
  mean /= bins * frames;
  let varr = 0;
  for (let b = 0; b < bins; b++)
    for (let f = 0; f < frames; f++) varr += (chroma12xN[b][f] - mean) ** 2;
  const std = Math.sqrt(varr / (bins * frames)) || 1;

  // Drop the final frame, then transpose to (frames, bins) and fit 312 rows.
  const rows = 312;
  const x = tensor(rows, bins, 1);
  for (let r = 0; r < rows; r++) {
    const f = r % (frames - 1);
    for (let b = 0; b < bins; b++) {
      set(x, r, b, 0, (chroma12xN[b][f] - mean) / std);
    }
  }

  const z = runLayers(x, spec.encoder);
  const out = runLayers(z, spec.cnn);
  return Array.isArray(out) ? out[0] : out.d[0];
}

export { runLayers, tensor, set };
