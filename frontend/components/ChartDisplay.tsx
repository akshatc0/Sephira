interface ChartDisplayProps {
  base64Image: string;
}

export default function ChartDisplay({ base64Image }: ChartDisplayProps) {
  return (
    <div className="mb-6 p-4 bg-background-secondary rounded-lg">
      <img
        src={`data:image/png;base64,${base64Image}`}
        alt="Generated chart"
        className="w-full h-auto rounded-lg"
      />
    </div>
  );
}

