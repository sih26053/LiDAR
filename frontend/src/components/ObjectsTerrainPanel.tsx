import { KIND_LABEL, groupObjects, terrainSplit } from '../utils/terrain';
import type { PipelineResult } from '../types/api';

/**
 * Panel 5 — Detected Objects & Terrain. Rows are annotation-sourced cell
 * groups from the backend result (centroid location, distance, cell count,
 * source). There is NO confidence column: the pipeline has no trained model,
 * so per-class ML confidence does not exist and is never shown.
 */
export function ObjectsTerrainPanel({ result }: { result: PipelineResult | null }) {
  const groups = result ? groupObjects(result.map_cells) : [];
  const split = result ? terrainSplit(result.map_cells) : null;
  return (
    <section className="panel" id="panel-objects" aria-label="Detected objects and terrain">
      <h2>5. Detected Objects &amp; Terrain</h2>
      {result && split ? (
        <>
          <table className="bench objs">
            <thead>
              <tr><th>Type</th><th>Location (X, Y)</th><th>Distance</th><th>Cells</th><th>Source</th></tr>
            </thead>
            <tbody>
              {groups.length === 0 && (
                <tr><td colSpan={5}>No annotation-sourced objects in this frame.</td></tr>
              )}
              {groups.map((g) => (
                <tr key={g.semantic_class}>
                  <td>{g.semantic_class} <span className="tag">{KIND_LABEL[g.kind]}</span></td>
                  <td className="mono">({g.centroidX.toFixed(1)}, {g.centroidY.toFixed(1)})</td>
                  <td>{g.distanceM.toFixed(0)} m</td>
                  <td>{g.cells.toLocaleString()}</td>
                  <td className="tag">{g.source} ref</td>
                </tr>
              ))}
            </tbody>
          </table>
          <ul className="kv">
            <li><span>Drivable surface cells</span><b>{split.drivable.toLocaleString()} (road_driveable)</b></li>
            <li><span>Non-drivable terrain cells</span><b>{split.nonDrivable.toLocaleString()} (static_manmade + vegetation)</b></li>
            <li><span>Dynamic-object cells</span><b>{split.dynamic.toLocaleString()} (vehicle + pedestrian_vru)</b></li>
            <li><span>Unclassified (fallback) cells</span><b>{split.unclassified.toLocaleString()}</b></li>
          </ul>
          <p className="caption">
            Locations/distances are geometry from backend cell positions. Classes are
            annotation references ({result.semantic.mode}), never model predictions;
            confidence is not applicable without a trained model.
          </p>
        </>
      ) : (
        <p className="state">No result available for this frame. Select a frame and press Run.</p>
      )}
    </section>
  );
}
