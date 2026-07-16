import { useMemo } from "react";
import * as THREE from "three";

const POINT_COUNT = 1000;
const SPREAD = 12;

/** Random point cloud placeholder — replace with PLY loader later. */
export default function MockPointCloud() {
  const geometry = useMemo(() => {
    const positions = new Float32Array(POINT_COUNT * 3);
    const colors = new Float32Array(POINT_COUNT * 3);

    for (let i = 0; i < POINT_COUNT; i += 1) {
      positions[i * 3] = (Math.random() - 0.5) * SPREAD;
      positions[i * 3 + 1] = (Math.random() - 0.5) * SPREAD * 0.6;
      positions[i * 3 + 2] = (Math.random() - 0.5) * SPREAD;

      const c = new THREE.Color().setHSL(0.55 + Math.random() * 0.08, 0.7, 0.45 + Math.random() * 0.2);
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return geo;
  }, []);

  return (
    <points geometry={geometry}>
      <pointsMaterial size={0.06} vertexColors sizeAttenuation />
    </points>
  );
}
