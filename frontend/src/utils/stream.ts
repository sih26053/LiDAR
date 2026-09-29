/**
 * Minimal .npy reader (float64 '<f8', C-order, 2-D only) plus JSON-array
 * fallback, so an externally supplied point cloud can be pushed to
 * POST /stream/push without inventing data client-side.
 */

export async function readPointFile(file: File): Promise<number[][]> {
  const buf = new Uint8Array(await file.arrayBuffer());
  if (file.name.endsWith('.npy')) return parseNpy(buf);
  const text = new TextDecoder().decode(buf);
  const arr = JSON.parse(text) as unknown;
  if (!Array.isArray(arr) || arr.length === 0 || !Array.isArray(arr[0])) {
    throw new Error('JSON must be an N x 4 array of [x, y, z, intensity].');
  }
  return arr as number[][];
}

function parseNpy(buf: Uint8Array): number[][] {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (buf[0] !== 0x93 || String.fromCharCode(buf[1], buf[2], buf[3], buf[4], buf[5]) !== 'NUMPY') {
    throw new Error('Not a .npy file.');
  }
  const major = buf[6];
  let headerLen: number;
  let offset: number;
  if (major === 1) {
    headerLen = view.getUint16(8, true);
    offset = 10;
  } else {
    headerLen = view.getUint32(8, true);
    offset = 12;
  }
  const header = new TextDecoder().decode(buf.slice(offset, offset + headerLen));
  if (!header.includes("'descr': '<f8'") && !header.includes('"descr": "<f8"')) {
    throw new Error('Only float64 (<f8) .npy files are supported.');
  }
  if (header.includes('True')) throw new Error('Fortran-order .npy is not supported.');
  const shape = header.match(/\('shape': \(([^)]*)\)/) ?? header.match(/"shape": \(([^)]*)\)/);
  if (!shape) throw new Error('Cannot parse .npy shape.');
  const dims = shape[1].split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  if (dims.length !== 2 || dims[1] !== 4) throw new Error(`Expected N x 4, got (${dims.join(', ')}).`);
  const n = dims[0];
  if (n < 100 || n > 150000) throw new Error(`Point count ${n} outside [100, 150000].`);
  const data = new Float64Array(buf.buffer, buf.byteOffset + offset + headerLen, n * 4);
  const rows: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    rows[i] = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]];
  }
  return rows;
}
