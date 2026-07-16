import type { SegmentationPredictData, SegmentationSessionData } from "../../types/inference";

export interface SegmentationSessionApiResponse {
  session_id: string;
  model_id: string;
  image_width: number;
  image_height: number;
  backend: string;
  message?: string;
}

export interface SegmentationPredictApiResponse {
  session_id: string;
  model_id: string;
  point: [number, number];
  image_width: number;
  image_height: number;
  mask: number[];
  contour: number[][];
  score: number;
  backend: string;
  message?: string;
}

export function sessionApiToData(
  api: SegmentationSessionApiResponse,
  imageUrl: string,
): SegmentationSessionData {
  return {
    sessionId: api.session_id,
    modelId: api.model_id,
    imageUrl,
    imageWidth: api.image_width,
    imageHeight: api.image_height,
    backend: api.backend,
    message: api.message,
  };
}

export function predictApiToData(api: SegmentationPredictApiResponse): SegmentationPredictData {
  return {
    sessionId: api.session_id,
    modelId: api.model_id,
    point: api.point,
    imageWidth: api.image_width,
    imageHeight: api.image_height,
    mask: api.mask,
    contour: api.contour,
    score: api.score,
    backend: api.backend,
    message: api.message,
  };
}

export async function startSegmentationSession(
  imageFile: File,
  modelId: string,
  imageUrl: string,
): Promise<SegmentationSessionData> {
  const form = new FormData();
  form.append("model_id", modelId);
  form.append("image", imageFile);

  const response = await fetch("/api/v1/inference/interactive-segmentation/session", {
    method: "POST",
    body: form,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data));
  }
  return sessionApiToData(data as SegmentationSessionApiResponse, imageUrl);
}

let predictRequestId = 0;

export async function predictSegmentationMask(
  sessionId: string,
  pointX: number,
  pointY: number,
): Promise<SegmentationPredictData> {
  const requestId = ++predictRequestId;
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("point_x", String(pointX));
  form.append("point_y", String(pointY));

  const response = await fetch("/api/v1/inference/interactive-segmentation/predict", {
    method: "POST",
    body: form,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data));
  }
  if (requestId !== predictRequestId) {
    throw new Error("stale");
  }
  return predictApiToData(data as SegmentationPredictApiResponse);
}

export function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = url;
  });
}
