import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { PoseFrame, Status } from "../protocol";
import { MujocoScene } from "../MujocoScene";

// MuJoCo z-up world -> three.js y-up (same basis change the headset uses, without room placement).
const Z_UP_TO_Y_UP = new THREE.Matrix4().makeBasis(
  new THREE.Vector3(0, 0, -1), new THREE.Vector3(-1, 0, 0), new THREE.Vector3(0, 1, 0),
);

export interface ViewerHandle { push: (f: PoseFrame) => void; reload: () => void; status: (s: Status) => void }

export function Viewer3D({ handle }: { handle: React.MutableRefObject<ViewerHandle | null> }) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = host.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#0a0e18");
    scene.fog = new THREE.Fog("#0a0e18", 4, 12);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.02, 50);
    camera.position.set(-0.2, 1.75, 1.0);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.85, -0.45);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI * 0.49;

    const mj = new MujocoScene({ shadows: true });
    mj.setPlacement(Z_UP_TO_Y_UP.toArray());
    scene.add(mj.root);  // App calls reload() once the websocket connects

    handle.current = {
      push: (f) => mj.pushFrame(f, performance.now()),
      reload: () => { mj.load(); },
      status: (s) => mj.setOutOfReach(s.out_of_reach),
    };

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = el;
      renderer.setSize(w, h);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    renderer.setAnimationLoop(() => {
      controls.update();
      mj.update(performance.now());
      renderer.render(scene, camera);
    });
    return () => {
      renderer.setAnimationLoop(null);
      ro.disconnect();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      handle.current = null;
    };
  }, [handle]);

  return <div className="viewer" ref={host} />;
}
