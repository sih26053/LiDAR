/**
 * Settings — display-only demo settings. No algorithm parameters are
 * exposed here: importance weights and resolution thresholds are frozen.
 */
export function SettingsPanel({ speed, onSpeed, maxCells, onMaxCells }: {
  speed: number; onSpeed: (s: number) => void; maxCells: number; onMaxCells: (n: number) => void;
}) {
  return (
    <section className="panel" id="panel-settings" aria-label="Demo settings">
      <h2>Settings (display only)</h2>
      <label className="field">
        Replay speed (display cadence; measured latency unaffected)
        <select value={speed} onChange={(e) => onSpeed(Number(e.target.value))}>
          {[0.25, 0.5, 1, 2].map((s) => (
            <option key={s} value={s}>{s}x</option>
          ))}
        </select>
      </label>
      <label className="field">
        Max map cells per response (smaller = lighter payloads)
        <select value={maxCells} onChange={(e) => onMaxCells(Number(e.target.value))}>
          {[500, 1000, 2000, 4000].map((n) => (
            <option key={n} value={n}>{n.toLocaleString()} cells</option>
          ))}
        </select>
      </label>
      <p className="caption">Frozen pipeline parameters (weights, thresholds) are not adjustable.</p>
    </section>
  );
}
