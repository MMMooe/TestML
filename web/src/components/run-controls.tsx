import { Play, SlidersHorizontal } from "lucide-react";

type RunControlsProps = {
    adapter: string;
    setAdapter: (value: string) => void;
    batchSize: number;
    setBatchSize: (value: number) => void;
    confidenceThreshold: number;
    setConfidenceThreshold: (value: number) => void;
    hasTensorRTEngine: boolean;
    allowTensorRTFallback: boolean;
    setAllowTensorRTFallback: (value: boolean) => void;
    computedMode: string;
    disabled: boolean;
    onStart: () => void;
};

export function RunControls({
    adapter,
    setAdapter,
    batchSize,
    setBatchSize,
    confidenceThreshold,
    setConfidenceThreshold,
    hasTensorRTEngine,
    allowTensorRTFallback,
    setAllowTensorRTFallback,
    computedMode,
    disabled,
    onStart
}: RunControlsProps) {
    return (
        <section className="rounded-lg border border-stone-200 bg-white p-4 shadow-panel">
            <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-stone-900">
                <SlidersHorizontal size={17} aria-hidden="true" />
                Run setup
            </div>

            <div className="grid gap-3">
                <label className="grid gap-1 text-xs font-medium text-stone-600">
                    Adapter
                    <select
                        className="h-10 rounded-lg border border-stone-200 bg-white px-3 text-sm text-stone-900"
                        value={adapter}
                        onChange={(event) => setAdapter(event.target.value)}
                    >
                        <option value="classification">Classification</option>
                        <option value="custom/no-metrics">Inference adapter</option>
                        <option value="detection">Detection-shaped output</option>
                    </select>
                </label>

                <label className="grid gap-1 text-xs font-medium text-stone-600">
                    Batch size
                    <input
                        className="h-10 rounded-lg border border-stone-200 px-3 text-sm text-stone-900"
                        type="number"
                        min={1}
                        max={128}
                        value={batchSize}
                        onChange={(event) => setBatchSize(Number(event.target.value))}
                    />
                </label>

                <label className="grid gap-1 text-xs font-medium text-stone-600">
                    Confidence threshold
                    <input
                        className="accent-teal-700"
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={confidenceThreshold}
                        onChange={(event) => setConfidenceThreshold(Number(event.target.value))}
                    />
                    <span className="text-stone-500">{Math.round(confidenceThreshold * 100)}%</span>
                </label>

                {hasTensorRTEngine && (
                    <label className="flex items-center gap-3 rounded-lg border border-stone-200 px-3 py-2 text-xs font-medium text-stone-600">
                        <input
                            className="h-4 w-4 accent-teal-700"
                            type="checkbox"
                            checked={allowTensorRTFallback}
                            onChange={(event) => setAllowTensorRTFallback(event.target.checked)}
                        />
                        Allow .pt fallback if TensorRT fails
                    </label>
                )}
            </div>

            <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">{computedMode}</div>

            <button
                type="button"
                className="mt-4 inline-flex h-11 w-full items-center justify-center gap-2 rounded-lg bg-stone-900 px-4 text-sm font-semibold text-white hover:bg-stone-700 disabled:cursor-not-allowed disabled:bg-stone-300"
                disabled={disabled}
                onClick={onStart}
            >
                <Play size={17} aria-hidden="true" />
                Start run
            </button>
        </section>
    );
}