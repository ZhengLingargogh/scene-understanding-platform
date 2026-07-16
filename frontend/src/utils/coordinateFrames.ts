import { Matrix4, Quaternion, Vector3 } from "three";

/**
 * Coordinate frame conventions
 * --------------------------
 * OpenCV (ACE / dsacstar backend):  X right, Y down,  Z forward (into scene)
 * Three.js (default world):         X right, Y up,    Z toward viewer (out of screen)
 *
 * Similarity transform S = diag(1, -1, -1) maps OpenCV vectors into Three.js vectors.
 * For a camera pose T_cv (4×4, camera → world in OpenCV frame):
 *
 *   T_three = S · T_cv · S
 *
 * (S is self-inverse, so S⁻¹ = S.)
 *
 * If your backend instead returns world → camera, invert before calling this helper.
 */
const OPENCV_TO_THREE = new Matrix4().set(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1);

export type Pose4x4 = number[][] | number[];

function toMatrix4(pose: Pose4x4): Matrix4 {
  const flat = Array.isArray(pose[0])
    ? (pose as number[][]).flat()
    : (pose as number[]);
  if (flat.length !== 16) {
    throw new Error(`Expected 4×4 pose (16 values), got ${flat.length}`);
  }
  return new Matrix4().fromArray(flat);
}

/** Convert backend OpenCV pose to Three.js position + quaternion. */
export function opencvPoseToThree(pose: Pose4x4): {
  position: Vector3;
  quaternion: Quaternion;
} {
  const tCv = toMatrix4(pose);
  const tThree = new Matrix4().copy(OPENCV_TO_THREE).multiply(tCv).multiply(OPENCV_TO_THREE);

  const position = new Vector3();
  const quaternion = new Quaternion();
  const scale = new Vector3();
  tThree.decompose(position, quaternion, scale);

  return { position, quaternion };
}

export function formatTranslation(t: number[]): string {
  return t.map((v) => v.toFixed(3)).join(", ");
}
