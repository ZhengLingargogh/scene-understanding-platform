import { useMemo } from "react";
import type { Quaternion, Vector3 } from "three";
import * as THREE from "three";

interface CameraFrustumProps {
  position: Vector3;
  quaternion: Quaternion;
  /** Half-width of image plane at unit depth (scene units). */
  halfWidth?: number;
  /** Half-height of image plane at unit depth. */
  halfHeight?: number;
  depth?: number;
}

/**
 * Wireframe camera frustum (pyramid) in the camera's local frame,
 * then placed at the predicted world pose (already converted to Three.js).
 */
export default function CameraFrustum({
  position,
  quaternion,
  halfWidth = 0.35,
  halfHeight = 0.25,
  depth = 1.2,
}: CameraFrustumProps) {
  const geometry = useMemo(() => {
    const hw = halfWidth;
    const hh = halfHeight;
    const d = depth;

    // Camera looks down local -Z in Three.js
    const corners = [
      new THREE.Vector3(-hw, -hh, -d),
      new THREE.Vector3(hw, -hh, -d),
      new THREE.Vector3(hw, hh, -d),
      new THREE.Vector3(-hw, hh, -d),
    ];
    const apex = new THREE.Vector3(0, 0, 0);

    const segments: THREE.Vector3[] = [];
    for (const c of corners) {
      segments.push(apex.clone(), c.clone());
    }
    for (let i = 0; i < 4; i += 1) {
      segments.push(corners[i].clone(), corners[(i + 1) % 4].clone());
    }

    const geo = new THREE.BufferGeometry().setFromPoints(segments);
    return geo;
  }, [halfWidth, halfHeight, depth]);

  return (
    <group position={position} quaternion={quaternion}>
      <lineSegments geometry={geometry}>
        <lineBasicMaterial color="#22d3ee" linewidth={1} transparent opacity={0.95} />
      </lineSegments>
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.08, 12, 12]} />
        <meshBasicMaterial color="#f97316" />
      </mesh>
    </group>
  );
}
