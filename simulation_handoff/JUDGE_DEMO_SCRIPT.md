# JUDGE DEMO SCRIPT (~5 minutes, one continuous flow)

Setup first: `simulation_handoff/START_DEMO.md` (backend + frontend + browser).
Backup if live replay fails: `simulation_handoff/backup_demo/README.md`
(say explicitly: "pre-rendered backup, not a live run").

1. **Introduce problem** — uniform LiDAR maps waste detail on empty space and
   starve important regions; Paradox Protocol allocates resolution where it matters.
2. **Show original LiDAR** — Panel B. "This is the raw replayed sensor input
   entering our pipeline — not the adaptive map."
3. **Run the system** — Panel A: pick scenario frame (suggested default:
   `5991fad3…` scene-0655; alternatives in `results/final/demo_scenarios.json`),
   press **Run**. Status goes Loading → Processing → Ready.
4. **Explain importance** — Panel D: red = high importance (backend output;
   distance/semantic/terrain/dynamic/uncertainty, frozen weights W_BASE).
5. **Explain adaptive resolution** — Panel D: marker sizes = 0.05/0.10/0.20/0.50 m
   tiers from the frozen config (thresholds 0.70/0.45/0.20). Fine detail where
   needed, coarse elsewhere.
6. **Show final 2.5D map** — Panel C (centerpiece): x/y/elevation/occupancy +
   semantic overlay. Panel F shows the semantic source (annotation reference,
   never a model prediction).
7. **Show measured performance** — Panel E: input/processed points, map cells,
   fine/medium/coarse, mapping latency (core ML) vs API wall clock (overhead),
   FPS = 1000/total ms. All backend-measured.
8. **Show benchmark comparison** — stored Proposed Adaptive vs Uniform 5 cm vs
   Distance-based Adaptive (from `results/benchmark/`, not recalculated) next to
   the **Current replay result** row. Honest reading: uniform 5 cm has far more
   cells (7.45M mean); proposed keeps 873.9 mean cells at finer mean resolution
   (0.125 m) than distance-based (0.177 m).
9. **Explain USP** — Takeaway panel: sees real LiDAR → decides importance →
   adapts resolution → produces variable-resolution 2.5D map → detailed where
   needed, coarse elsewhere.
10. **Show another frame** — Prev/Next to a critical-object frame (`b26e7915…`,
    many vehicles) and re-run; same flow, fresh result (selecting a frame clears
    the old result — no stale panels).
11. **Finish with takeaway** — measured numbers on screen + stored benchmark;
    no unsupported performance claims.
