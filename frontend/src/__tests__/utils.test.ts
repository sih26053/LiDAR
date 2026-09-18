import { describe, expect, it } from 'vitest';
import { fmtFps, fmtInt, fmtMs, fmtTimestamp } from '../utils/formatting';
import { importanceColor, resolutionTier } from '../utils/visualization';
import { isValidResult } from '../utils/validation';
import { sampleResult } from './fixture';

describe('formatting (measured-or-Unavailable)', () => {
  it('renders Unavailable for null metrics, never fake zeros', () => {
    expect(fmtInt(null)).toBe('Unavailable');
    expect(fmtMs(undefined)).toBe('Unavailable');
    expect(fmtFps(NaN)).toBe('Unavailable');
    expect(fmtInt(34359)).toBe('34,359');
  });
  it('formats timestamps without crashing', () => {
    expect(fmtTimestamp(null)).toBe('Unavailable');
    expect(fmtTimestamp(1535385092150099.0)).toContain('2018');
  });
});

describe('visualization scales (display only, no ML)', () => {
  it('buckets backend resolution values to tiers', () => {
    expect(resolutionTier(0.05)).toBe('fine');
    expect(resolutionTier(0.1)).toBe('medium');
    expect(resolutionTier(0.2)).toBe('coarse');
    expect(resolutionTier(0.5)).toBe('very_coarse');
  });
  it('maps importance bands to distinct colors', () => {
    expect(importanceColor(0.85)).not.toBe(importanceColor(0.1));
  });
});

describe('validation', () => {
  it('accepts a valid backend result', () => {
    expect(isValidResult(sampleResult)).toBe(true);
  });
  it('rejects malformed payloads', () => {
    expect(isValidResult({})).toBe(false);
    expect(isValidResult({ frame_id: 'x' })).toBe(false);
  });
});
