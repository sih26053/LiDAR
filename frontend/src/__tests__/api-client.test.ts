import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, api } from '../api/client';

function mockFetchOnce(body: unknown, ok = true, status = 200) {
  (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
    ok,
    status,
    json: async () => body,
  });
}

describe('API client', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  it('calls the real endpoint paths with JSON bodies', async () => {
    const f = globalThis.fetch as unknown as ReturnType<typeof vi.fn>;
    mockFetchOnce({ status: 'ok', service: 'paradox-protocol-backend' });
    await api.checkHealth();
    expect(f).toHaveBeenCalledWith(expect.stringMatching(/\/health$/), expect.anything());

    mockFetchOnce({ frames: [], count: 0 });
    await api.getFrames();
    expect(f).toHaveBeenCalledWith(expect.stringMatching(/\/frames$/), expect.anything());

    mockFetchOnce({ frame_id: 'abc', load_status: 'loaded' });
    await api.loadFrame('abc');
    const [, init] = f.mock.calls[f.mock.calls.length - 1] as [string, RequestInit];
    expect(init.method).toBe('POST');
    expect(init.body).toContain('abc');

    mockFetchOnce({ result: { frame_id: 'abc' }, cache_hit: false });
    await api.runFrame('abc');
    const [url] = f.mock.calls[f.mock.calls.length - 1] as [string];
    expect(url).toMatch(/\/replay\/run$/);
  });

  it('surfaces backend unavailable instead of fake values', async () => {
    const f = globalThis.fetch as unknown as ReturnType<typeof vi.fn>;
    f.mockRejectedValueOnce(new Error('down'));
    await expect(api.getFrames()).rejects.toBeInstanceOf(ApiError);
    f.mockRejectedValueOnce(new Error('down'));
    try {
      await api.getFrames();
      expect.unreachable();
    } catch (e) {
      expect((e as ApiError).message).toMatch(/Backend unavailable/);
    }
  });

  it('propagates structured error payloads (frame not found)', async () => {
    mockFetchOnce(
      { stage: 'input', error_code: 'FRAME_NOT_FOUND', message: 'Frame not found: nope', frame_id: 'nope' },
      false,
      404,
    );
    try {
      await api.loadFrame('nope');
      expect.unreachable();
    } catch (e) {
      const err = e as ApiError;
      expect(err.status).toBe(404);
      expect(err.payload?.error_code).toBe('FRAME_NOT_FOUND');
    }
  });
});
