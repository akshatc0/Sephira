"use client";

import { useState } from "react";
import ChatInterface from "@/components/ChatInterface";

export default function Home() {
  return (
    <main className="min-h-screen bg-background-primary">
      <div className="container mx-auto px-4 py-8">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-text-primary mb-2">Sephira</h1>
          <p className="text-text-secondary">
            LLM-powered sentiment data analysis and visualization
          </p>
        </header>

        <ChatInterface />
      </div>
    </main>
  );
}
