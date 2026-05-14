import { AlertTriangle, CheckCircle2, CircleDot, Clock3, XCircle } from "lucide-react";

export type RunLogLevel = "info" | "success" | "warning" | "error";

export type RunLogEntry = {
    id: string;
    level: RunLogLevel;
    time: string;
    title: string;
    detail?: string;
};

type RunLogProps = {
    entries: RunLogEntry[];
};

const levelStyles = {
    info: "border-stone-200 bg-stone-50 text-stone-600",
    success: "border-teal-200 bg-teal-50 text-teal-700",
    warning: "border-amber-200 bg-amber-50 text-amber-700",
    error: "border-red-200 bg-red-50 text-red-700"
};

const levelIcons = {
    info: CircleDot,
    success: CheckCircle2,
    warning: AlertTriangle,
    error: XCircle
};

export function RunLog({ entries }: RunLogProps) {
    return (
        <section className="rounded-lg border border-stone-200 bg-white p-4 shadow-panel">
            <div className="mb-3 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-sm font-semibold text-stone-900">
                    <Clock3 size={17} aria-hidden="true" />
                    Run log
                </div>
                <span className="rounded-lg bg-stone-100 px-2 py-1 text-xs font-medium text-stone-500">{entries.length}</span>
            </div>

            {entries.length === 0 ? (
                <div className="rounded-lg border border-dashed border-stone-300 bg-stone-50 px-3 py-4 text-sm text-stone-500">No run log yet</div>
            ) : (
                <div className="max-h-72 space-y-2 overflow-auto pr-1 scrollbar-thin">
                    {entries.map((entry) => {
                        const Icon = levelIcons[entry.level];
                        return (
                            <div key={entry.id} className={`rounded-lg border px-3 py-2 ${levelStyles[entry.level]}`}>
                                <div className="flex items-start gap-2">
                                    <Icon className="mt-0.5 shrink-0" size={15} aria-hidden="true" />
                                    <div className="min-w-0 flex-1">
                                        <div className="flex items-start justify-between gap-3">
                                            <p className="text-xs font-semibold text-stone-950">{entry.title}</p>
                                            <span className="shrink-0 text-[11px] font-medium text-stone-500">{entry.time}</span>
                                        </div>
                                        {entry.detail && <p className="mt-1 break-words text-xs leading-5 text-stone-600">{entry.detail}</p>}
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </section>
    );
}