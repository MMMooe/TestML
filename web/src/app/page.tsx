"use client";

import { ReactNode, useEffect, useMemo, useState } from "react";
import { AlertTriangle, Cpu, Database, Server, UploadCloud } from "lucide-react";

import { ResultDetail } from "@/components/result-detail";
import { ResultsGallery } from "@/components/results-gallery";
import { RunControls } from "@/components/run-controls";
import { UploadZone } from "@/components/upload-zone";
import {
    createJob,
    getJob,
    getMetrics,
    getResults,
    getRuntime,
    JobStatus,
    MetricsResponse,
    ResultItem,
    ResultsResponse,
    RuntimeInfo,
    uploadDataset,
    uploadModel
} from "@/lib/api";

export default function Home() {
    const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
    const [modelFiles, setModelFiles] = useState<File[]>([]);
    const [imageFiles, setImageFiles] = useState<File[]>([]);
    const [annotationFiles, setAnnotationFiles] = useState<File[]>([]);
    const [archiveFiles, setArchiveFiles] = useState<File[]>([]);
    const [adapter, setAdapter] = useState("classification");
    const [batchSize, setBatchSize] = useState(1);
    const [confidenceThreshold, setConfidenceThreshold] = useState(0);
    const [status, setStatus] = useState<JobStatus | null>(null);
    const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
    const [results, setResults] = useState<ResultsResponse | null>(null);
    const [query, setQuery] = useState("");
    const [correctness, setCorrectness] = useState("");
    const [minConfidence, setMinConfidence] = useState(0);
    const [selected, setSelected] = useState<ResultItem | null>(null);
    const [busy, setBusy] = useState(false);
    const [notice, setNotice] = useState<string | null>(null);

    useEffect(() => {
        getRuntime().then(setRuntime).catch((error) => setNotice(error.message));
    }, []);

    useEffect(() => {
        if (!status || status.state === "queued" || status.state === "running") return;
        refreshResults(status.id).catch((error) => setNotice(error.message));
    }, [query, correctness, minConfidence, status?.id, status?.state]);

    const computedMode = useMemo(() => {
        if (archiveFiles.length > 0 && annotationFiles.length === 0) {
            return "Archive contents decide: images only runs inference; images plus JSON runs evaluation + inference.";
        }
        if (annotationFiles.length > 0) return "Evaluation + inference: JSON annotations will be compared with predictions.";
        return "Inference only: no JSON annotations selected.";
    }, [annotationFiles.length, archiveFiles.length]);

    const canStart = modelFiles.length === 1 && (imageFiles.length > 0 || archiveFiles.length === 1) && !busy;

    async function startRun() {
        if (!canStart) return;
        setBusy(true);
        setNotice("Uploading files");
        setStatus(null);
        setMetrics(null);
        setResults(null);
        try {
            const model = await uploadModel(modelFiles[0]);
            const dataset = await uploadDataset({
                images: imageFiles,
                annotation: annotationFiles[0] || null,
                archive: archiveFiles[0] || null
            });
            if (dataset.warnings.length) setNotice(dataset.warnings.join(" "));
            const created = await createJob({
                model_id: model.id,
                dataset_id: dataset.id,
                adapter,
                batch_size: batchSize,
                confidence_threshold: confidenceThreshold
            });
            setStatus(created);
            setNotice("Job queued");
            await pollJob(created.id);
        } catch (error) {
            setNotice(error instanceof Error ? error.message : "Run failed");
        } finally {
            setBusy(false);
        }
    }

    async function pollJob(jobId: string) {
        let latest = await getJob(jobId);
        setStatus(latest);
        while (latest.state === "queued" || latest.state === "running") {
            await new Promise((resolve) => setTimeout(resolve, 1000));
            latest = await getJob(jobId);
            setStatus(latest);
        }
        await refreshResults(jobId);
        setNotice(latest.state === "completed" ? "Job completed" : latest.error || "Job failed");
    }

    async function refreshResults(jobId = status?.id) {
        if (!jobId) return;
        const [metricsResponse, resultsResponse] = await Promise.all([
            getMetrics(jobId),
            getResults(jobId, {
                page: 1,
                page_size: 48,
                q: query || undefined,
                correctness: correctness || undefined,
                min_confidence: minConfidence || undefined
            })
        ]);
        setMetrics(metricsResponse);
        setResults(resultsResponse);
    }

    return (
        <main className="grid min-h-screen grid-rows-[auto_minmax(0,1fr)] bg-[#f7f7f4]">
            <header className="border-b border-stone-200 bg-white px-4 py-3 md:px-6">
                <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-stone-900 text-white">
                            <UploadCloud size={20} aria-hidden="true" />
                        </div>
                        <div>
                            <h1 className="text-base font-semibold text-stone-950">Model Evaluation Gallery</h1>
                            <p className="text-sm text-stone-500">Ubuntu CUDA runtime with macOS dev-mock support</p>
                        </div>
                    </div>
                    <RuntimeBadge runtime={runtime} />
                </div>
            </header>

            <div className="grid min-h-0 gap-4 p-4 lg:grid-cols-[380px_minmax(0,1fr)] lg:p-6">
                <aside className="space-y-4 overflow-auto scrollbar-thin">
                    {runtime?.app_mode === "dev-mock" && (
                        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                            <div className="mb-1 flex items-center gap-2 font-semibold">
                                <AlertTriangle size={16} aria-hidden="true" />
                                Dev mock mode
                            </div>
                            Real inference and evaluation must run on the Ubuntu NVIDIA machine.
                        </div>
                    )}

                    <UploadZone label="Model" hint="Required .pt file" accept=".pt" files={modelFiles} onChange={(files) => setModelFiles(files.slice(0, 1))} kind="model" />
                    <UploadZone label="Images" hint="Required unless the zip contains images" accept="image/*" multiple files={imageFiles} onChange={setImageFiles} kind="images" />
                    <UploadZone label="Annotations" hint="Optional JSON for evaluation" accept=".json,application/json" files={annotationFiles} onChange={(files) => setAnnotationFiles(files.slice(0, 1))} kind="json" />
                    <UploadZone label="Dataset zip" hint="Optional archive with images and maybe JSON" accept=".zip,application/zip" files={archiveFiles} onChange={(files) => setArchiveFiles(files.slice(0, 1))} kind="archive" />

                    <RunControls
                        adapter={adapter}
                        setAdapter={setAdapter}
                        batchSize={batchSize}
                        setBatchSize={setBatchSize}
                        confidenceThreshold={confidenceThreshold}
                        setConfidenceThreshold={setConfidenceThreshold}
                        computedMode={computedMode}
                        disabled={!canStart}
                        onStart={startRun}
                    />

                    {notice && <div className="rounded-lg border border-stone-200 bg-white p-3 text-sm text-stone-700 shadow-panel">{notice}</div>}
                </aside>

                <section className="grid min-h-0 gap-4 lg:grid-rows-[auto_minmax(0,1fr)]">
                    <div className="grid gap-3 md:grid-cols-3">
                        <StatusPanel icon={<Server size={17} />} label="Job" value={status ? status.state : "Idle"} />
                        <StatusPanel icon={<Database size={17} />} label="Mode" value={status ? (status.job_kind === "evaluation" ? "Evaluation + inference" : "Inference only") : "Waiting"} />
                        <StatusPanel icon={<Cpu size={17} />} label="Device" value={runtime?.selected_device || "Unknown"} />
                    </div>

                    <ResultsGallery
                        status={status}
                        metrics={metrics}
                        results={results}
                        query={query}
                        setQuery={setQuery}
                        correctness={correctness}
                        setCorrectness={setCorrectness}
                        minConfidence={minConfidence}
                        setMinConfidence={setMinConfidence}
                        onRefresh={() => refreshResults().catch((error) => setNotice(error.message))}
                        onSelect={setSelected}
                    />
                </section>
            </div>

            <ResultDetail item={selected} onClose={() => setSelected(null)} />
        </main>
    );
}

function RuntimeBadge({ runtime }: { runtime: RuntimeInfo | null }) {
    const isCuda = runtime?.app_mode === "production-cuda";
    return (
        <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className={`rounded-lg px-3 py-2 font-semibold ${isCuda ? "bg-teal-50 text-teal-800" : "bg-amber-50 text-amber-900"}`}>
                {runtime?.app_mode || "loading"}
            </span>
            <span className="rounded-lg bg-stone-100 px-3 py-2 text-stone-600">{runtime?.device_name || runtime?.message || "Runtime check"}</span>
        </div>
    );
}

function StatusPanel({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
    return (
        <div className="rounded-lg border border-stone-200 bg-white p-3 shadow-panel">
            <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase text-stone-500">
                {icon}
                {label}
            </div>
            <p className="truncate text-sm font-semibold text-stone-950">{value}</p>
        </div>
    );
}