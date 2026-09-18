import { useCallback, useRef, useState } from 'react';

export interface CanvasView { zoom: number; ox: number; oy: number; }

/**
 * 2D camera for canvas map views: wheel = zoom at cursor, drag = pan,
 * double-click / reset() = return to fit. Pure view transform — backend
 * data is never altered. Resets automatically when resetSignal changes.
 */
export function useCanvasView(resetSignal: string | null) {
  const [view, setView] = useState<CanvasView>({ zoom: 1, ox: 0, oy: 0 });
  const drag = useRef<{ x: number; y: number } | null>(null);
  const lastSignal = useRef(resetSignal);

  if (lastSignal.current !== resetSignal) {
    lastSignal.current = resetSignal;
    if (view.zoom !== 1 || view.ox !== 0 || view.oy !== 0) setView({ zoom: 1, ox: 0, oy: 0 });
  }

  const onWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    setView((v) => ({ ...v, zoom: Math.min(8, Math.max(0.5, v.zoom * f)) }));
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    drag.current = { x: e.clientX, y: e.clientY };
  }, []);

  const onMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y;
    drag.current = { x: e.clientX, y: e.clientY };
    setView((v) => ({ ...v, ox: v.ox + dx, oy: v.oy + dy }));
  }, []);

  const endDrag = useCallback(() => { drag.current = null; }, []);
  const reset = useCallback(() => setView({ zoom: 1, ox: 0, oy: 0 }), []);

  return { view, onWheel, onMouseDown, onMouseMove, onMouseUp: endDrag, onMouseLeave: endDrag, onDoubleClick: reset, reset };
}

/** Apply a 2D camera to a fitted canvas point (zoom about centre + pan). */
export function applyView(cx: number, cy: number, size: number, view: CanvasView): { cx: number; cy: number } {
  const c = size / 2;
  return { cx: c + (cx - c) * view.zoom + view.ox, cy: c + (cy - c) * view.zoom + view.oy };
}
