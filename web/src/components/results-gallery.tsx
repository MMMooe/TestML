import { AlertCircle, CheckCircle2, Download, Eye, Search, XCircle } from "lucide-react";

import { API_URL, JobStatus, MetricsResponse, ResultItem, ResultsResponse } from "@/lib/api";

type ResultsGalleryProps = {
    status: JobStatus | null;
    metrics: MetricsResponse | null;
    results: ResultsResponse | null;
    query: string;
    setQuery: (value: string) => void;
    correctness: string;
    setCorrectness: (value: string) => void;
    minConfidence: number;
    setMinConfidence: (value: number) => void;
    onRefresh: () => void;
    onSelect: (item: ResultItem) => void;
};

export function ResultsGallery({
    status,
    metrics,
    results,
    query,
    setQuery,
    correctness,
    setCorrectness,
    minConfidence,
    setMinConfidence,
    onRefresh,
    onSelect
}: ResultsGalleryProps) {
    const isEvaluation = status?.job_kind === "evaluation";

    return (
        <section className="min-h-0 rounded-lg border border-stone-200 bg-white shadow-panel">
            <header className="border-b border-stone-200 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                        <h2 className="text-base font-semibold text-stone-950">Results</h2>
                        <p className="mt-1 text-sm text-stone-500">{status ? `${status.state} · ${status.processed}/${status.total} images` : "No run started"}</p>
                    </div>
                    <button className="inline-flex items-center gap-2 rounded-lg border border-stone-200 px-3 py-2 text-sm font-medium text-stone-700 hover:bg-stone-50" type="button" onClick={onRefresh} disabled={!status}>
                        <Download size={16} aria-hidden="true" />
                        Refresh
                    </button>
                </div>

                {status && (
                    <div className="mt-4 h-2 overflow-hidden rounded-full bg-stone-100">
                        <div className="h-full bg-teal-600 transition-all" style={{ width: `${Math.round(status.progress * 100)}%` }} />
                    </div>
                )}
            </header>

            <div className="space-y-4 p-4">
                {status && (
                    <div className="grid gap-3 md:grid-cols-3">
                        <MetricTile label="Run kind" value={isEvaluation ? "Evaluation + inference" : "Inference only"} />
                        <MetricTile label="Images" value={String(status.total || 0)} />
                        {metrics?.available ? (
                            <MetricTile label="Accuracy" value={formatMetric(metrics.metrics.accuracy)} />
                        ) : (
                            <MetricTile label="Metrics" value={isEvaluation ? "Pending" : "No annotations"} />
                        )}
                    </div>
                )}

                <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_180px_180px]">
                    <label className="relative block">
                        <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-stone-400" size={16} aria-hidden="true" />
                        <input
                            className="h-10 w-full rounded-lg border border-stone-200 pl-9 pr-3 text-sm outline-none focus:border-teal-600"
                            placeholder="Search filename"
                            value={query}
                            onChange={(event) => setQuery(event.target.value)}
                        />
                    </label>
                    <select className="h-10 rounded-lg border border-stone-200 px-3 text-sm" value={correctness} onChange={(event) => setCorrectness(event.target.value)} disabled={!isEvaluation}>
                        <option value="">All outcomes</option>
                        <option value="correct">Correct</option>
                        <option value="incorrect">Incorrect</option>
                        <option value="error">Errors</option>
                    </select>
                    <label className="grid gap-1 text-xs text-stone-500">
                        Confidence {Math.round(minConfidence * 100)}%
                        <input className="accent-teal-700" type="range" min={0} max={1} step={0.05} value={minConfidence} onChange={(event) => setMinConfidence(Number(event.target.value))} />
                    </label>
                </div>

                {!results || results.items.length === 0 ? (
                    <div className="flex min-h-[360px] items-center justify-center rounded-lg border border-dashed border-stone-300 bg-stone-50 text-sm text-stone-500">No gallery items yet</div>
                ) : (
                    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
                        {results.items.map((item) => (
                            <button key={item.id} type="button" className="group overflow-hidden rounded-lg border border-stone-200 bg-white text-left shadow-panel hover:border-teal-500" onClick={() => onSelect(item)}>
                                <div className="aspect-[4/3] bg-stone-100">
                                    <img className="h-full w-full object-cover" src={`${API_URL}${item.image_url}`} alt={item.filename} />
                                </div>
                                <div className="space-y-2 p-3">
                                    <div className="flex items-start justify-between gap-2">
                                        <p className="min-w-0 truncate text-sm font-medium text-stone-900">{item.filename}</p>
                                        <OutcomeIcon item={item} />
                                    </div>
                                    <div className="flex items-center justify-between gap-3 text-xs text-stone-500">
                                        <span className="truncate">{item.prediction.label || "raw output"}</span>
                                        <span className="shrink-0">{formatConfidence(item.prediction.confidence)}</span>
                                    </div>
                                    <span className="inline-flex items-center gap-1 text-xs font-medium text-teal-700">
                                        <Eye size={14} aria-hidden="true" />
                                        Detail
                                    </span>
                                </div>
                            </button>
                        ))}
                    </div>
                )}
            </div>
        </section>
    );
}

function MetricTile({ label, value }: { label: string; value: string }) {
    return (
        <div className="rounded-lg border border-stone-200 bg-stone-50 px-3 py-2">
            <p className="text-xs font-medium text-stone-500">{label}</p>
            <p className="mt-1 truncate text-sm font-semibold text-stone-950">{value}</p>
        </div>
    );
}

function OutcomeIcon({ item }: { item: ResultItem }) {
    if (item.error) return <AlertCircle className="text-amber-600" size={17} aria-label="Error" />;
    if (item.is_correct === true) return <CheckCircle2 className="text-teal-700" size={17} aria-label="Correct" />;
    if (item.is_correct === false) return <XCircle className="text-red-600" size={17} aria-label="Incorrect" />;
    return null;
}

function formatConfidence(value?: number | null) {
    if (value === undefined || value === null) return "--";
    return `${Math.round(value * 100)}%`;
}

function formatMetric(value: unknown) {
    if (typeof value === "number") return `${Math.round(value * 1000) / 10}%`;
    if (value === null || value === undefined) return "n/a";
    return String(value);
}