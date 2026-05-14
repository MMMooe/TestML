import { FileArchive, FileJson, FileUp, Images, X } from "lucide-react";

type UploadZoneProps = {
    label: string;
    hint: string;
    accept: string;
    files: File[];
    onChange: (files: File[]) => void;
    multiple?: boolean;
    maxVisibleItems?: number;
    kind: "model" | "images" | "json" | "archive";
};

const icons = {
    model: FileUp,
    images: Images,
    json: FileJson,
    archive: FileArchive
};

export function UploadZone({ label, hint, accept, files, onChange, multiple = false, maxVisibleItems = 4, kind }: UploadZoneProps) {
    const Icon = icons[kind];
    const inputId = `upload-${kind}`;
    const visibleFiles = files.slice(0, maxVisibleItems);

    return (
        <section className="rounded-lg border border-stone-200 bg-white p-4 shadow-panel">
            <div className="flex items-start gap-3">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-teal-50 text-teal-700">
                    <Icon size={18} aria-hidden="true" />
                </div>
                <div className="min-w-0 flex-1">
                    <label htmlFor={inputId} className="block text-sm font-semibold text-stone-900">
                        {label}
                    </label>
                    <p className="mt-1 text-xs leading-5 text-stone-500">{hint}</p>
                    <input
                        id={inputId}
                        className="mt-3 block w-full cursor-pointer rounded-lg border border-stone-200 bg-stone-50 text-sm text-stone-700 file:mr-3 file:border-0 file:bg-stone-800 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white hover:bg-stone-100"
                        type="file"
                        accept={accept}
                        multiple={multiple}
                        onChange={(event) => onChange(Array.from(event.target.files || []))}
                    />
                </div>
            </div>

            {files.length > 0 && (
                <div className="mt-3 space-y-2">
                    {visibleFiles.map((file) => (
                        <div key={`${file.name}-${file.size}`} className="flex items-center justify-between gap-3 rounded-lg bg-stone-50 px-3 py-2 text-xs text-stone-700">
                            <span className="truncate">{file.name}</span>
                            <span className="shrink-0 text-stone-500">{formatSize(file.size)}</span>
                        </div>
                    ))}
                    {files.length > maxVisibleItems && <p className="text-xs text-stone-500">{files.length - maxVisibleItems} more files selected</p>}
                    <button
                        type="button"
                        className="inline-flex items-center gap-2 rounded-lg border border-stone-200 px-3 py-2 text-xs font-medium text-stone-700 hover:bg-stone-50"
                        onClick={() => onChange([])}
                    >
                        <X size={14} aria-hidden="true" />
                        Clear
                    </button>
                </div>
            )}
        </section>
    );
}

function formatSize(size: number) {
    if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`;
    return `${(size / 1024 / 1024).toFixed(1)} MB`;
}