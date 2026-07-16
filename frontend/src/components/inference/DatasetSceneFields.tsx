import type { DatasetCatalog } from "../../types/dataset";

const selectClass =
  "rounded-md border border-slate-600/60 bg-slate-800/80 px-2 py-2 text-sm text-slate-100 outline-none focus:border-cyan-500/60";

export interface DatasetSceneFieldsProps {
  catalog: DatasetCatalog[];
  familyId: string;
  sceneId: string;
  split?: string;
  showSplit?: boolean;
  splitLabel?: string;
  onFamilyChange: (id: string) => void;
  onSceneChange: (id: string) => void;
  onSplitChange?: (split: string) => void;
  className?: string;
  disabled?: boolean;
}

export default function DatasetSceneFields({
  catalog,
  familyId,
  sceneId,
  split = "train",
  showSplit = false,
  splitLabel = "下级划分",
  onFamilyChange,
  onSceneChange,
  onSplitChange,
  className = selectClass,
  disabled = false,
}: DatasetSceneFieldsProps) {
  const family = catalog.find((item) => item.id === familyId) ?? catalog[0];
  const scene = family?.scenes.find((item) => item.scene_id === sceneId) ?? family?.scenes[0];

  return (
    <>
      <label className="flex flex-col gap-1 text-xs text-slate-300">
        数据集
        <select
          value={familyId}
          disabled={disabled || catalog.length === 0}
          onChange={(e) => onFamilyChange(e.target.value)}
          className={className}
        >
          {catalog.map((item) => (
            <option key={item.id} value={item.id}>
              {item.name}
            </option>
          ))}
        </select>
      </label>

      <label className="flex flex-col gap-1 text-xs text-slate-300">
        场景
        <select
          value={sceneId}
          disabled={disabled || !family}
          onChange={(e) => onSceneChange(e.target.value)}
          className={className}
        >
          {family?.scenes.map((item) => (
            <option key={item.scene_id} value={item.scene_id}>
              {item.name}
            </option>
          ))}
        </select>
      </label>

      {showSplit && (
        <label className="flex flex-col gap-1 text-xs text-slate-300">
          {splitLabel}
          <select
            value={split}
            disabled={disabled || !scene}
            onChange={(e) => onSplitChange?.(e.target.value)}
            className={className}
          >
            {scene?.splits.map((item) => (
              <option key={item.split} value={item.split}>
                {item.label}（{item.image_count} 张）
              </option>
            ))}
          </select>
        </label>
      )}
    </>
  );
}
