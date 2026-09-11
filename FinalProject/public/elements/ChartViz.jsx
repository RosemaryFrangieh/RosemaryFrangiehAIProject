import { useEffect, useRef } from "react";

export default function ChartJS() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const renderChart = () => {
      if (!canvasRef.current || !window.Chart || !props.config) {
        return;
      }
      const existing = window.Chart.getChart(canvasRef.current);
      if (existing) {
        existing.destroy();
      }
      new window.Chart(canvasRef.current, props.config);
    };

    if (window.Chart) {
      renderChart();
    } else {
      const script = document.createElement("script");
      script.src = "https://cdn.jsdelivr.net/npm/chart.js";
      script.onload = renderChart;
      document.head.appendChild(script);
    }
  }, [JSON.stringify(props.config)]);

  return (
    <div className="w-full rounded-lg border p-4" style={{ minHeight: "300px" }}>
      <canvas ref={canvasRef}></canvas>
    </div>
  );
}