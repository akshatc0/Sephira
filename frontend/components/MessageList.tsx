interface Message {
  role: "user" | "assistant";
  content: string;
}

interface MessageListProps {
  messages: Message[];
}

export default function MessageList({ messages }: MessageListProps) {
  if (messages.length === 0) {
    return (
      <div className="mb-6 p-6 text-center text-text-secondary">
        <p>Start a conversation about sentiment data</p>
        <p className="text-sm mt-2">
          Try asking: &quot;Show me sentiment trends for France over the last 5 years&quot;
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4 mb-6 max-h-96 overflow-y-auto">
      {messages.map((message, index) => (
        <div
          key={index}
          className={`flex ${
            message.role === "user" ? "justify-end" : "justify-start"
          }`}
        >
          <div
            className={`max-w-[80%] p-4 rounded-lg ${
              message.role === "user"
                ? "bg-white/10 text-text-primary"
                : "bg-background-secondary text-text-primary"
            }`}
          >
            <p className="whitespace-pre-wrap">{message.content}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

