import { KIND_LABEL, groupObjects, terrainSplit } from '../utils/terrain';
import type { PipelineResult } from '../types/api';

/**
 * Panel 5 — Objects & Terrain (annotation or model channel). Rows are cell
 * groups from the backend result (centroid location, distance, cell count,
 * source). In the model channel, cells additionally carry measured softmax
 * confidence, summarized below; annotation cells never fabricate confidence.
 */
export function ObjectsTerrainPanel({ result }: { result: PipelineResult | null }) {
  const groups = result ? groupObjects(result.map_cells) : [];
  const split = result ? terrainSplit(result.map_cells) : null;
  return (
    <section className="panel" id="panel-objects" aria-label="Objects and terrain">
      <h2>5. Objects &amp; Terrain (model / annotation channel)</h2>
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
            {result.semantic.mode.startsWith('model')
              ? ' trained-classifier predictions with measured confidence (see Model panel).'
              : ` annotation references (${result.semantic.mode}), never model predictions;`}
            {result.model_eval?.available
              ? ` Model-vs-annotation agreement on box interiors this frame: ${result.model_eval.accuracy_vs_annotation_reference !== null ? `${(result.model_eval.accuracy_vs_annotation_reference * 100).toFixed(1)}%` : 'n/a'} (${result.model_eval.n_reference_points?.toLocaleString()} ref pts, eval-only).`
              : ''}
          </p>
        </>
      ) : (
        <p className="state">No result available for this frame. Select a frame and press Run.</p>
      )}
    </section>
  );
}
