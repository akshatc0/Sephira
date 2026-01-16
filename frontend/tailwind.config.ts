import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: {
          primary: "#0A0D1C",
          secondary: "#0F152F",
          card: {
            top: "#121834",
            bottom: "#090D20",
          },
        },
        text: {
          primary: "#FFFFFF",
          secondary: "rgba(255, 255, 255, 0.62)",
        },
      },
      fontFamily: {
        sans: ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
      },
      borderRadius: {
        card: "20px",
        button: "14px",
      },
    },
  },
  plugins: [],
};

export default config;

