import type { Config } from "tailwindcss";

const config: Config = {
    content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
    theme: {
        extend: {
            boxShadow: {
                panel: "0 1px 2px rgba(15, 23, 42, 0.08)"
            }
        }
    },
    plugins: []
};

export default config;