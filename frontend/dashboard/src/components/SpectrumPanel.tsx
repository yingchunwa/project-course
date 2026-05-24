import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

import { getTaskSpectra } from "../services/api";
import type { AxisSpectrum, SpectrumAxis, WindowSpectraResponse } from "../types/api";

interface Props {
  taskId: string | null;
  windowIndex: number | null;
}

interface SensorBlock {
  key: "accel" | "vision";
  label: string;
  axes: SpectrumAxis[];
  colors: string[];
}

const BLOCKS: SensorBlock[] = [
  {
    key: "accel",
    label: "加速度计",
    axes: ["sensor_ax", "sensor_ay", "sensor_az"],
    colors: ["#1677ff", "#dc2626", "#16a34a"]
  },
  {
    key: "vision",
    label: "视觉",
    axes: ["vision_dx", "vision_dy"],
    colors: ["#2563eb", "#db2777"]
  }
];

function SpectrumPanel({ taskId, windowIndex }: Props): JSX.Element {
  const [spectra, setSpectra] = useState<WindowSpectraResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!taskId || windowIndex === null || windowIndex === undefined) {
      setSpectra(null);
      return;
    }
    let cancelled = false;
    getTaskSpectra(taskId, windowIndex)
      .then((data) => {
        if (!cancelled) {
          setSpectra(data);
          setError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [taskId, windowIndex]);

  return (
    <div style={panelStyle}>
      <div style={headerStyle}>
        <h3 style={{ margin: 0, fontSize: 15 }}>实时频谱</h3>
        <span style={metaStyle}>
          {windowIndex !== null && windowIndex !== undefined ? `窗口 #${windowIndex}` : "无窗口"}
        </span>
      </div>

      {error && <div style={errorStyle}>频谱加载失败: {error}</div>}

      <div style={gridStyle}>
        {BLOCKS.map((block) => (
          <SensorSpectrum
            key={block.key}
            block={block}
            spectra={spectra}
            hasError={error !== null}
          />
        ))}
      </div>
    </div>
  );
}

interface SensorSpectrumProps {
  block: SensorBlock;
  spectra: WindowSpectraResponse | null;
  hasError: boolean;
}

function SensorSpectrum({ block, spectra, hasError }: SensorSpectrumProps): JSX.Element {
  const option = useMemo(() => buildOption(spectra, block), [spectra, block]);
  const hasData = spectra !== null && block.axes.some((axis) => spectra[axis] !== null);

  return (
    <div style={subPanelStyle}>
      <div style={subHeaderStyle}>{block.label}</div>
      <div style={chartWrapStyle}>
        {!hasData && !hasError && (
          <div style={emptyStyle}>暂无频谱数据</div>
        )}
        {hasData && (
          <ReactECharts
            option={option}
            style={{ height: "100%", width: "100%" }}
            notMerge
          />
        )}
      </div>
    </div>
  );
}

function buildOption(
  spectra: WindowSpectraResponse | null,
  block: SensorBlock
): Record<string, unknown> {
  if (!spectra) return {};

  const series = block.axes
    .map((axis, idx) => {
      const data = spectra[axis] as AxisSpectrum | null;
      if (!data) return null;
      const peakIdx = data.power.indexOf(Math.max(...data.power));
      const peakHz = data.freq_hz[peakIdx];
      return {
        name: axis,
        type: "line" as const,
        data: data.freq_hz.map((f, i) => [f, data.power[i]]),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 1.6, color: block.colors[idx] },
        itemStyle: { color: block.colors[idx] },
        markLine: {
          symbol: "none",
          lineStyle: { color: block.colors[idx], type: "dashed" as const, width: 1 },
          label: {
            formatter: `${peakHz.toFixed(1)} Hz`,
            fontSize: 10,
            position: "insideEndBottom" as const,
            color: block.colors[idx],
            distance: 4
          },
          data: [{ xAxis: peakHz }]
        }
      };
    })
    .filter((s): s is NonNullable<typeof s> => s !== null);

  return {
    tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
    legend: { top: 2, textStyle: { fontSize: 11 }, itemHeight: 8, itemGap: 8 },
    grid: { left: 48, right: 16, bottom: 32, top: 40 },
    xAxis: { type: "value", name: "Hz", nameLocation: "end", nameTextStyle: { fontSize: 11 } },
    yAxis: { type: "value", name: "Power", nameTextStyle: { fontSize: 11 } },
    series
  };
}

const panelStyle: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #e8e8e8",
  borderRadius: 8,
  padding: 12,
  boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
  height: "100%",
  display: "flex",
  flexDirection: "column",
  minHeight: 0
};

const headerStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  marginBottom: 8
};

const gridStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
  gap: 10,
  flex: 1,
  minHeight: 0
};

const subPanelStyle: React.CSSProperties = {
  background: "#fafbfc",
  border: "1px solid #eef0f3",
  borderRadius: 6,
  padding: "6px 8px 8px",
  display: "flex",
  flexDirection: "column",
  minHeight: 0
};

const subHeaderStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  color: "#334155",
  marginBottom: 2
};

const chartWrapStyle: React.CSSProperties = {
  flex: 1,
  minHeight: 0,
  position: "relative"
};

const metaStyle: React.CSSProperties = { color: "#64748b", fontSize: 13 };

const errorStyle: React.CSSProperties = {
  padding: 10,
  background: "#fef2f2",
  border: "1px solid #fecaca",
  color: "#991b1b",
  borderRadius: 6,
  fontSize: 13,
  marginBottom: 10
};

const emptyStyle: React.CSSProperties = {
  padding: 24,
  textAlign: "center",
  color: "#94a3b8",
  background: "#f8fafc",
  borderRadius: 6,
  fontSize: 12
};

export default SpectrumPanel;
