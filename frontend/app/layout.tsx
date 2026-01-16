import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sephira - Sentiment Data Analysis",
  description: "LLM-powered sentiment data analysis and visualization",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
