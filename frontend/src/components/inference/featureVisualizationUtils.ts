import type { FeatureVisualizationData } from "../../types/inference";

/** Auto inner crop for images with black borders. */
export function getAutoCropBBox(
  imageData: ImageData,
  threshold = 15,
): [number, number, number, number] {
  const { width: w, height: h, data } = imageData;
  const mask = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
    mask[p] = gray > threshold ? 1 : 0;
  }

  const centerY = Math.floor(h / 2);
  const centerX = Math.floor(w / 2);

  const rowMask = (y: number) => {
    const xs: number[] = [];
    for (let x = 0; x < w; x += 1) {
      if (mask[y * w + x]) xs.push(x);
    }
    return xs;
  };
  const colMask = (x: number) => {
    const ys: number[] = [];
    for (let y = 0; y < h; y += 1) {
      if (mask[y * w + x]) ys.push(y);
    }
    return ys;
  };

  const validX = rowMask(centerY);
  const validY = colMask(centerX);

  let innerMinX = 0;
  let innerMaxX = w - 1;
  let innerMinY = 0;
  let innerMaxY = h - 1;

  if (validX.length > 0 && validY.length > 0) {
    innerMinX = validX[0];
    innerMaxX = validX[validX.length - 1];
    innerMinY = validY[0];
    innerMaxY = validY[validY.length - 1];
  } else {
    innerMinX = w;
    innerMaxX = 0;
    innerMinY = h;
    innerMaxY = 0;
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        if (mask[y * w + x]) {
          innerMinX = Math.min(innerMinX, x);
          innerMaxX = Math.max(innerMaxX, x);
          innerMinY = Math.min(innerMinY, y);
          innerMaxY = Math.max(innerMaxY, y);
        }
      }
    }
  }

  const pad = 5;
  const y1 = Math.min(innerMinY + pad, centerY);
  const y2 = Math.max(innerMaxY - pad, centerY);
  const x1 = Math.min(innerMinX + pad, centerX);
  const x2 = Math.max(innerMaxX - pad, centerX);
  return [y1, y2, x1, x2];
}

function gaussianKernel1D(sigma: number): Float32Array {
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const size = radius * 2 + 1;
  const kernel = new Float32Array(size);
  let sum = 0;
  for (let i = 0; i < size; i += 1) {
    const x = i - radius;
    const v = Math.exp(-(x * x) / (2 * sigma * sigma));
    kernel[i] = v;
    sum += v;
  }
  for (let i = 0; i < size; i += 1) kernel[i] /= sum;
  return kernel;
}

function convolve1D(
  src: Float32Array,
  width: number,
  height: number,
  kernel: Float32Array,
  horizontal: boolean,
): Float32Array {
  const dst = new Float32Array(src.length);
  const radius = (kernel.length - 1) / 2;

  if (horizontal) {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let acc = 0;
        for (let k = -radius; k <= radius; k += 1) {
          const sx = Math.min(width - 1, Math.max(0, x + k));
          acc += src[y * width + sx] * kernel[k + radius];
        }
        dst[y * width + x] = acc;
      }
    }
  } else {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let acc = 0;
        for (let k = -radius; k <= radius; k += 1) {
          const sy = Math.min(height - 1, Math.max(0, y + k));
          acc += src[sy * width + x] * kernel[k + radius];
        }
        dst[y * width + x] = acc;
      }
    }
  }
  return dst;
}

export function gaussianFilter2D(
  data: Float32Array,
  width: number,
  height: number,
  sigma: number,
): Float32Array {
  const kernel = gaussianKernel1D(Math.max(0.8, sigma));
  const tmp = convolve1D(data, width, height, kernel, true);
  return convolve1D(tmp, width, height, kernel, false);
}

/** Heatmap color: low=transparent dark, high=deep warm (confidence ↑ → color ↑). */
export function heatmapColor(t: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, t));
  const stops: [number, number, number, number][] = [
    [0.0, 20, 16, 48],
    [0.25, 68, 1, 84],
    [0.5, 180, 50, 20],
    [0.75, 240, 120, 20],
    [1.0, 255, 220, 80],
  ];
  for (let i = 0; i < stops.length - 1; i += 1) {
    const [t0, r0, g0, b0] = stops[i];
    const [t1, r1, g1, b1] = stops[i + 1];
    if (v >= t0 && v <= t1) {
      const f = (v - t0) / (t1 - t0);
      return [
        Math.round(r0 + (r1 - r0) * f),
        Math.round(g0 + (g1 - g0) * f),
        Math.round(b0 + (b1 - b0) * f),
      ];
    }
  }
  return [255, 220, 80];
}

/**
 * Heatmap: every keypoint contributes weight = confidence score.
 * Accumulate on grid → Gaussian smooth (sigma = max/image / 50) → normalize by max.
 */
export function buildResponseHeatmap(
  keypoints: [number, number][],
  scores: number[],
  imageWidth: number,
  imageHeight: number,
): { heatmap: Float32Array; width: number; height: number; displayMax: number } {
  const workScale = Math.min(1, 960 / Math.max(imageWidth, imageHeight));
  const w = Math.max(1, Math.round(imageWidth * workScale));
  const h = Math.max(1, Math.round(imageHeight * workScale));

  const raw = new Float32Array(w * h);
  const sx = w / imageWidth;
  const sy = h / imageHeight;

  for (let i = 0; i < keypoints.length; i += 1) {
    const score = scores[i];
    if (score <= 0) continue;
    const [kx, ky] = keypoints[i];
    const px = Math.min(w - 1, Math.max(0, Math.round(kx * sx)));
    const py = Math.min(h - 1, Math.max(0, Math.round(ky * sy)));
    raw[py * w + px] += score;
  }

  const sigma = (Math.max(imageWidth, imageHeight) / 50) * workScale;
  const smoothed = gaussianFilter2D(raw, w, h, sigma);

  let displayMax = 0;
  for (let i = 0; i < smoothed.length; i += 1) {
    displayMax = Math.max(displayMax, smoothed[i]);
  }
  if (displayMax <= 0) displayMax = 1;

  return { heatmap: smoothed, width: w, height: h, displayMax };
}

export interface FeatureVisualizationApiResponse {
  image_width: number;
  image_height: number;
  keypoints: [number, number][];
  scores: number[];
  crop_bbox: [number, number, number, number];
  keypoint_count: number;
  confidence_max: number;
  confidence_min: number;
  confidence_median: number;
  model_id: string;
  backend?: string;
  message?: string;
}

export function apiResponseToVisualization(
  api: FeatureVisualizationApiResponse,
  imageUrl: string,
): FeatureVisualizationData {
  return {
    imageUrl,
    imageWidth: api.image_width,
    imageHeight: api.image_height,
    keypoints: api.keypoints,
    scores: api.scores,
    cropBBox: api.crop_bbox,
    keypointCount: api.keypoint_count,
    confidenceMax: api.confidence_max,
    confidenceMin: api.confidence_min,
    confidenceMedian: api.confidence_median,
    modelId: api.model_id,
    backend: api.backend,
    message: api.message,
  };
}

export async function runFeatureVisualization(
  imageFile: File,
  imageUrl: string,
  modelId: string,
): Promise<FeatureVisualizationData> {
  const form = new FormData();
  form.append("model_id", modelId);
  form.append("image", imageFile);

  const response = await fetch("/api/v1/inference/feature-visualization", {
    method: "POST",
    body: form,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data));
  }
  return apiResponseToVisualization(data as FeatureVisualizationApiResponse, imageUrl);
}

export function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = url;
  });
}

/** @deprecated Use models with keypoint_detection capability from /models API */
export const KEYPOINT_EXTRACTOR_IDS = new Set(["superpoint", "sift"]);
