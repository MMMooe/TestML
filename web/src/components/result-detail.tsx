import { X } from "lucide-react";

import { API_URL, ResultItem } from "@/lib/api";

type ResultDetailProps = {
    item: ResultItem | null;
    onClose: () => void;
};

export function ResultDetail({ item, onClose }: ResultDetailProps) {
    if (!item) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-stretch justify-end bg-black/30" role="dialog" aria-modal="true">
            <section className="flex h-full w-full max-w-3xl flex-col bg-white shadow-2xl">
                <header className="flex items-center justify-between border-b border-stone-200 px-5 py-4">
                    <div className="min-w-0">
                        <h2 className="truncate text-base font-semibold text-stone-950">{item.filename}</h2>
                        <p className="text-xs text-stone-500">{item.job_kind === "evaluation" ? "Evaluation item" : "Inference item"}</p>
                    </div>
                    <button className="rounded-lg p-2 text-stone-500 hover:bg-stone-100 hover:text-stone-900" type="button" onClick={onClose} aria-label="Close detail">
                        <X size={18} aria-hidden="true" />
                    </button>
                </header>

                <div className="grid min-h-0 flex-1 gap-5 overflow-auto p-5 md:grid-cols-[minmax(0,1fr)_320px]">
                    <div className="flex min-h-[280px] items-center justify-center rounded-lg bg-stone-100">
                        <img className="max-h-[68vh] max-w-full object-contain" src={`${API_URL}${item.image_url}`} alt={item.filename} />
                    </div>

                    <div className="space-y-4">
                        <InfoBlock title="Prediction" value={JSON.stringify(item.prediction, null, 2)} />
                        {item.ground_truth !== undefined && item.ground_truth !== null && <InfoBlock title="Ground truth" value={JSON.stringify(item.ground_truth, null, 2)} />}
                        {item.error && <InfoBlock title="Error" value={item.error} />}
                    </div>
                </div>
            </section>
        </div>
    );
}

function InfoBlock({ title, value }: { title: string; value: string }) {
    return (
        <section className="rounded-lg border border-stone-200 bg-stone-50 p-3">
            <h3 className="mb-2 text-xs font-semibold uppercase text-stone-500">{title}</h3>
            <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words text-xs leading-5 text-stone-800">{value}</pre>
        </section>
    );
}