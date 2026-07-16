import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { DatasetCatalog, DatasetSceneInfo, DatasetSplitInfo } from "../types/dataset";

export type DatasetSplitMode = "test" | "train";

export interface DatasetSceneSelection {
  catalog: DatasetCatalog[];
  loading: boolean;
  error: string | null;
  familyId: string;
  sceneId: string;
  split: string;
  family: DatasetCatalog | undefined;
  scene: DatasetSceneInfo | undefined;
  splitInfo: DatasetSplitInfo | undefined;
  testDatasetId: string;
  trainDatasetId: string;
  setFamilyId: (id: string) => void;
  setSceneId: (id: string) => void;
  setSplit: (split: string) => void;
}

interface UseDatasetSceneSelectionOptions {
  defaultFamilyId?: string;
  defaultSceneId?: string;
  defaultSplit?: string;
  showSplit?: boolean;
}

export function useDatasetSceneSelection(
  options: UseDatasetSceneSelectionOptions = {},
): DatasetSceneSelection {
  const {
    defaultFamilyId = "crossloc",
    defaultSceneId = "nature",
    defaultSplit = "test",
    showSplit = false,
  } = options;

  const [catalog, setCatalog] = useState<DatasetCatalog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [familyId, setFamilyId] = useState(defaultFamilyId);
  const [sceneId, setSceneId] = useState(defaultSceneId);
  const [split, setSplit] = useState(defaultSplit);

  useEffect(() => {
    api
      .get<DatasetCatalog[]>("/scenes/catalog")
      .then((data) => {
        setCatalog(data);
        setError(null);
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const family = useMemo(
    () => catalog.find((item) => item.id === familyId) ?? catalog[0],
    [catalog, familyId],
  );

  const scene = useMemo(
    () => family?.scenes.find((item) => item.scene_id === sceneId) ?? family?.scenes[0],
    [family, sceneId],
  );

  const splitInfo = useMemo(
    () => scene?.splits.find((item) => item.split === split) ?? scene?.splits[0],
    [scene, split],
  );

  const testDatasetId = useMemo(() => {
    return scene?.splits.find((item) => item.split === "test")?.dataset_id ?? "crossloc-nature-test";
  }, [scene]);

  const trainDatasetId = useMemo(() => {
    return scene?.splits.find((item) => item.split === "train")?.dataset_id ?? "crossloc-nature-train";
  }, [scene]);

  useEffect(() => {
    if (catalog.length === 0) return;
    if (!catalog.some((item) => item.id === familyId)) {
      setFamilyId(catalog[0].id);
    }
  }, [catalog, familyId]);

  useEffect(() => {
    if (!family) return;
    if (!family.scenes.some((item) => item.scene_id === sceneId)) {
      setSceneId(family.scenes[0].scene_id);
    }
  }, [family, sceneId]);

  useEffect(() => {
    if (!showSplit || !scene) return;
    if (!scene.splits.some((item) => item.split === split)) {
      setSplit(scene.splits[0].split);
    }
  }, [scene, split, showSplit]);

  return {
    catalog,
    loading,
    error,
    familyId,
    sceneId,
    split,
    family,
    scene,
    splitInfo,
    testDatasetId,
    trainDatasetId,
    setFamilyId,
    setSceneId,
    setSplit,
  };
}
