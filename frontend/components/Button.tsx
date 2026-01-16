interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  primary?: boolean;
  type?: "button" | "submit" | "reset";
}

export default function Button({
  children,
  onClick,
  disabled = false,
  primary = false,
  type = "button",
}: ButtonProps) {
  if (primary) {
    return (
      <button
        type={type}
        onClick={onClick}
        disabled={disabled}
        className="bg-white text-black px-6 py-3 rounded-button font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100 transition-colors"
      >
        {children}
      </button>
    );
  }

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className="bg-transparent border border-white text-white px-6 py-3 rounded-button font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-white/10 transition-colors"
    >
      {children}
    </button>
  );
}

