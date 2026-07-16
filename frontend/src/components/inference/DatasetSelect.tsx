import type { MockDataset } from "../../types/inference";
import type { DatasetWithFamily } from "./inferenceUtils";

interface DatasetSelectProps {
  datasets: MockDataset[];
  value: string;
  onChange: (id: string) => void;
  className?: string;
}

export default function DatasetSelect({ datasets, value, onChange, className }: DatasetSelectProps) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className={className}>
      {datasets.length === 0 ? (
        <option value="crossloc-nature-test">CrossLoc Nature Test</option>
      ) : (
        <>
          <optgroup label="CrossLoc（nature / natureoop / urban / urbanoop）">
            {datasets
              .filter((d) => (d as DatasetWithFamily).family === "crossloc" || d.id.startsWith("crossloc"))
              .map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.image_count} 张)
                </option>
              ))}
          </optgroup>
          <optgroup label="UAVD4L（inTraj / outTraj）">
            {datasets
              .filter((d) => (d as DatasetWithFamily).family === "uavd4l" || d.id.startsWith("uavd4l"))
              .map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.image_count} 张)
                </option>
              ))}
          </optgroup>
        </>
      )}
    </select>
  );
}
