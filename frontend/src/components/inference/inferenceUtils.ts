import type {
  CovisibilityRef,
  InferenceApiResult,
  MatchingVisualizationData,
  MockDataset,
  RetrievalReference,
} from "../../types/inference";
import type { ModelItem, PipelineStage } from "../../types";

export function placeholderThumb(hue: number, rank: number): string {
  const canvas = document.createElement("canvas");
  canvas.width = 160;
  canvas.height = 120;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  const g = ctx.createLinearGradient(0, 0, 160, 120);
  g.addColorStop(0, `hsl(${hue}, 55%, 32%)`);
  g.addColorStop(1, `hsl(${hue + 25}, 60%, 20%)`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 160, 120);
  ctx.strokeStyle = "rgba(34,211,238,0.55)";
  ctx.strokeRect(4, 4, 152, 112);
  ctx.fillStyle = "rgba(226,232,240,0.85)";
  ctx.font = "bold 14px sans-serif";
  ctx.fillText(`#${rank}`, 68, 64);
  return canvas.toDataURL();
}

export function retrievalThumbnail(imagePath: string | null | undefined, hue: number, rank: number): string {
  if (imagePath) {
    return `/api/v1/media/file?path=${encodeURIComponent(imagePath)}`;
  }
  return placeholderThumb(hue, rank);
}

export function formatModelLabel(model: Pick<ModelItem, "name" | "version">): string {
  return model.name;
}

export function modelsForStage(models: ModelItem[], stage: PipelineStage): ModelItem[] {
  return models.filter((m) => m.capabilities?.includes(stage));
}

export function matchingFromPipeline(
  result: InferenceApiResult,
  localImage0Url: string,
  localImage1Url: string,
): MatchingVisualizationData | null {
  const matching = result.pipeline?.image_matching;
  if (!matching || matching.status !== "completed") return null;
  if (!matching.keypoints0?.length || !matching.keypoints1?.length) return null;

  return {
    keypoints0: matching.keypoints0,
    keypoints1: matching.keypoints1,
    matches: matching.matches ?? [],
    image0Url: localImage0Url,
    image1Url: localImage1Url,
    matchCount: matching.match_count ?? matching.matches?.length ?? 0,
    message: matching.message,
  };
}

export function refsFromPipeline(result: InferenceApiResult): CovisibilityRef[] {
  const apiRefs = result.pipeline?.image_retrieval?.references ?? [];
  return apiRefs.map((ref: RetrievalReference, i: number) => ({
    id: ref.id,
    label: ref.label,
    score: ref.score,
    thumbnail: retrievalThumbnail(ref.image_path, 195 + i * 14, i + 1),
    image_path: ref.image_path ?? undefined,
  }));
}

export async function runInference(form: FormData): Promise<InferenceApiResult> {
  const response = await fetch("/api/v1/inference/run", {
    method: "POST",
    body: form,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data));
  }
  return data as InferenceApiResult;
}

export type DatasetWithFamily = MockDataset & { family?: string };
