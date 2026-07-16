import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Suspense, useMemo } from "react";
import type { Quaternion, Vector3 } from "three";
import CameraFrustum from "./CameraFrustum";
import MockPointCloud from "./MockPointCloud";

interface SceneViewportProps {
  cameraPose: { position: Vector3; quaternion: Quaternion } | null;
}

function SceneContent({ cameraPose }: SceneViewportProps) {
  return (
    <>
      <color attach="background" args={["#0a0f1a"]} />
      <fog attach="fog" args={["#0a0f1a", 18, 45]} />

      <ambientLight intensity={0.35} />
      <directionalLight position={[8, 12, 6]} intensity={0.85} color="#e0f2fe" />
      <pointLight position={[-6, 4, -4]} intensity={0.4} color="#38bdf8" />

      <axesHelper args={[2.5]} />
      <gridHelper args={[24, 24, "#1e3a5f", "#132337"]} position={[0, -3, 0]} />

      <MockPointCloud />

      {cameraPose && (
        <CameraFrustum position={cameraPose.position} quaternion={cameraPose.quaternion} />
      )}

      <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
    </>
  );
}

export default function SceneViewport({ cameraPose }: SceneViewportProps) {
  const hint = useMemo(
    () => (cameraPose ? "青色视锥体 = 预测相机位姿" : "上传并 Run 后显示预测相机"),
    [cameraPose],
  );

  return (
    <div className="relative h-full min-h-[480px] flex-1 overflow-hidden rounded-xl border border-cyan-900/50 bg-slate-950 shadow-[inset_0_0_60px_rgba(6,182,212,0.06)]">
      <div className="pointer-events-none absolute left-4 top-4 z-10 rounded-md border border-cyan-500/30 bg-slate-900/80 px-3 py-1.5 text-xs text-cyan-200/90 backdrop-blur">
        {hint}
      </div>
      <div className="pointer-events-none absolute right-4 top-4 z-10 rounded-md border border-slate-600/40 bg-slate-900/70 px-2 py-1 text-[10px] uppercase tracking-widest text-slate-400">
        Orbit · Zoom · Pan
      </div>

      <Canvas
        className="h-full w-full"
        camera={{ position: [6, 4, 8], fov: 50, near: 0.1, far: 200 }}
        gl={{ antialias: true, alpha: false }}
      >
        <Suspense fallback={null}>
          <SceneContent cameraPose={cameraPose} />
        </Suspense>
      </Canvas>
    </div>
  );
}
