// Types matching src/project_course/api/storage/models.py.
// Keep in sync with specs/001-dual-modal-monitoring/contracts/api.yaml.

export type TaskStatus = "pending" | "running" | "succeeded" | "failed";

export interface CreateTaskRequest {
  device_id: string;
  camera_mode?: string;
  imu_sample_rate_hz?: number;
  window_size_s?: number;
  window_hop_s?: number;
  roi_x?: number;
  roi_y?: number;
  roi_w?: number;
  roi_h?: number;
}

export interface TaskResponse {
  task_id: string;
  task_status: TaskStatus;
  created_at: string;
}

export interface SyncQuality {
  offset_ms_p95: number | null;
  drift_ppm: number | null;
  aligned_window_ratio: number | null;
}

export interface TaskDetailResponse {
  task_id: string;
  task_status: TaskStatus;
  device_id: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  camera_mode: string;
  imu_sample_rate_hz: number;
  window_size_s: number;
  window_hop_s: number;
  roi_x: number | null;
  roi_y: number | null;
  roi_w: number | null;
  roi_h: number | null;
  model_version: string | null;
  predicted_state: string | null;
  confidence_summary: number | null;
  effective_window_count: number;
  sync_quality: SyncQuality;
  error_message: string | null;
}

export interface WindowSample {
  sample_id: string;
  window_index: number;
  center_time_s: number;
  label?: string | null;
  modality?: string | null;
  imu_quality_flag?: string | null;
  cam_quality_flag?: string | null;
  sync_fit_failed?: boolean | null;
  seq_gap_count?: number | null;
  // arbitrary vision_* / sensor_* / fused_* fields
  [key: string]: string | number | boolean | null | undefined;
}

export interface TaskWindowsResponse {
  task_id: string;
  samples: WindowSample[];
}

export interface AxisSpectrum {
  freq_hz: number[];
  power: number[];
}

export type SpectrumAxis =
  | "vision_dx"
  | "vision_dy"
  | "sensor_ax"
  | "sensor_ay"
  | "sensor_az";

export interface WindowSpectraResponse {
  task_id: string;
  window_index: number;
  vision_dx: AxisSpectrum | null;
  vision_dy: AxisSpectrum | null;
  sensor_ax: AxisSpectrum | null;
  sensor_ay: AxisSpectrum | null;
  sensor_az: AxisSpectrum | null;
}

export interface DashboardOverview {
  latest_task_id: string | null;
  latest_status: TaskStatus | null;
  latest_predicted_state: string | null;
  latest_fused_frequency_hz: number | null;
  active_model_version: string | null;
  task_success_rate_24h: number;
  sync_offset_ms_p95: number | null;
  sync_drift_ppm: number | null;
  aligned_window_ratio: number | null;
  effective_window_count: number;
  latest_window_index: number | null;
}

// ---------------- offline history (renamed from /samples in v0) ------------

export interface HistoryMetadata {
  sample_id: string;
  label: string | null;
  captured_at: string | null;
  source_name: string | null;
  has_vision: boolean;
  has_sensor: boolean;
  file_path: string;
  window_count: number;
  ingested_at: string;
}

export type HistoryRow = Record<string, string | number | boolean | null>;

export interface HistoryDetail {
  metadata: HistoryMetadata;
  rows: HistoryRow[];
}

export interface HistoryTimeseries {
  sample_id: string;
  fields: string[];
  points: HistoryRow[];
}

export interface IngestErrorBody {
  detail: {
    message: string;
    missing_columns: string[];
  };
}
