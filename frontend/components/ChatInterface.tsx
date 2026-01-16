"use client";

import { useState, useRef, useEffect } from "react";
import { sendChatMessage, generateChart } from "@/lib/api";
import type { ChatRequest, ChartRequest } from "@/lib/api";
import Button from "./Button";
import MessageList from "./MessageList";
import ChartDisplay from "./ChartDisplay";

export default function ChatInterface() {
  const [messages, setMessages] = useState<Array<{ role: "user" | "assistant"; content: string }>>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentChart, setCurrentChart] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput("");
    setError(null);
    setLoading(true);

    // Add user message to UI
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);

    try {
      // Build conversation history from messages
      const conversationHistory: Array<{ user: string; assistant: string }> = [];
      for (let i = 0; i < messages.length - 1; i++) {
        if (messages[i].role === "user" && messages[i + 1]?.role === "assistant") {
          conversationHistory.push({
            user: messages[i].content,
            assistant: messages[i + 1].content,
          });
          i++; // Skip the assistant message since we've processed it
        }
      }

      const request: ChatRequest = {
        message: userMessage,
        session_id: sessionId,
        conversation_history: conversationHistory,
      };

      const response = await sendChatMessage(request);

      // Update session ID
      if (response.session_id) {
        setSessionId(response.session_id);
      }

      // Add assistant response
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: response.response },
      ]);

      // Handle chart request if needed
      if (response.chart_request && response.chart_request.needs_chart) {
        try {
          const chartRequest: ChartRequest = {
            countries: response.chart_request.countries || [],
            date_range: response.chart_request.date_range || {
              start: "2020-01-01",
              end: "2024-12-31",
            },
            chart_type: response.chart_request.chart_type || "time_series",
            title: response.chart_request.title || "Sentiment Trends",
          };

          const chartResponse = await generateChart(chartRequest);
          setCurrentChart(chartResponse.base64_image);
        } catch (chartError) {
          console.error("Chart generation error:", chartError);
          setError("Chart generation failed. Please try again.");
        }
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "An error occurred";
      setError(errorMessage);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${errorMessage}. Please try again.`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="card-gradient rounded-card p-6 max-w-4xl mx-auto">
      <MessageList messages={messages} />
      {currentChart && <ChartDisplay base64Image={currentChart} />}
      
      {error && (
        <div className="mb-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400">
          {error}
        </div>
      )}

      <div className="flex gap-3 mt-4">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about sentiment data..."
          className="flex-1 bg-background-secondary border border-text-secondary/30 rounded-button px-4 py-3 text-text-primary placeholder:text-text-secondary resize-none focus:outline-none focus:border-text-primary"
          rows={3}
          disabled={loading}
        />
        <Button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          primary
        >
          {loading ? "Sending..." : "Send"}
        </Button>
      </div>

      <div ref={messagesEndRef} />
    </div>
  );
}

