export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type RuntimeInfo = {
    app_mode: string;
    cuda_available: boolean;
    selected_device: string;
    torch_version?: string | null;
    cuda_version?: string | null;
    device_name?: string | null;
    message: string;
};

export type ModelRecord = {
    id: string;
    filename: string;
    size_bytes: number;
    created_at: string;
};

export type DatasetManifest = {
    id: string;
    image_count: number;
    has_annotations: boolean;
    job_kind: "inference" | "evaluation";
    warnings: string[];
};

export type JobStatus = {
    id: string;
    model_id: string;
    dataset_id: string;
    adapter: string;
    job_kind: "inference" | "evaluation";
    state: "queued" | "running" | "completed" | "failed";
    progress: number;
    processed: number;
    total: number;
    message: string;
    error?: string | null;
};

export type MetricsResponse = {
    job_id: string;
    available: boolean;
    job_kind: "inference" | "evaluation";
    metrics: Record<string, unknown>;
    message: string;
};

export type ResultItem = {
    id: string;
    filename: string;
    image_url: string;
    job_kind: "inference" | "evaluation";
    prediction: {
        label?: string | null;
        confidence?: number | null;
        top_k: Array<{ label: string; score: number; index?: number }>;
        raw?: unknown;
    };
    ground_truth?: unknown;
    is_correct?: boolean | null;
    error?: string | null;
};

export type ResultsResponse = {
    job_id: string;
    page: number;
    page_size: number;
    total: number;
    items: ResultItem[];
};

export async function getRuntime(): Promise<RuntimeInfo> {
    return request<RuntimeInfo>("/runtime");
}

export async function uploadModel(file: File): Promise<ModelRecord> {
    const form = new FormData();
    form.append("file", file);
    const response = await request<{ model: ModelRecord }>("/models", { method: "POST", body: form });
    return response.model;
}

export async function uploadDataset(args: {
    images: File[];
    annotation?: File | null;
    archive?: File | null;
}): Promise<DatasetManifest> {
    const form = new FormData();
    args.images.forEach((file) => form.append("images", file));
    if (args.annotation) form.append("annotation", args.annotation);
    if (args.archive) form.append("archive", args.archive);
    const response = await request<{ dataset: DatasetManifest }>("/datasets", { method: "POST", body: form });
    return response.dataset;
}

export async function createJob(args: {
    model_id: string;
    dataset_id: string;
    adapter: string;
    batch_size: number;
    confidence_threshold: number;
}): Promise<JobStatus> {
    return request<JobStatus>("/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(args)
    });
}

export async function getJob(jobId: string): Promise<JobStatus> {
    return request<JobStatus>(`/jobs/${jobId}`);
}

export async function getMetrics(jobId: string): Promise<MetricsResponse> {
    return request<MetricsResponse>(`/jobs/${jobId}/metrics`);
}

export async function getResults(jobId: string, query: Record<string, string | number | undefined>): Promise<ResultsResponse> {
    const params = new URLSearchParams();
    Object.entries(query).forEach(([key, value]) => {
        if (value !== undefined && value !== "") params.set(key, String(value));
    });
    return request<ResultsResponse>(`/jobs/${jobId}/results?${params.toString()}`);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${API_URL}${path}`, init);
    if (!response.ok) {
        let detail = response.statusText;
        try {
            const payload = await response.json();
            detail = payload.detail || detail;
        } catch {
            // Keep the HTTP status text if the body is not JSON.
        }
        throw new Error(Array.isArray(detail) ? detail.map((item) => item.msg).join(", ") : String(detail));
    }
    return response.json() as Promise<T>;
}