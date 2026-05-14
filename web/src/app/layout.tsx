import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "Model Evaluation Gallery",
    description: "CUDA model inference and evaluation gallery"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
    return (
        <html lang="en">
            <body>{children}</body>
        </html>
    );
}