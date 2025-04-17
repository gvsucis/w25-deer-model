"use client";

import { useRef, useEffect, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader";

export default function Viewer() {
  const [modelUrl, setModelUrl] = useState<string | null>(null);
  const [scanName, setScanName] = useState("Unnamed Scan"); // Add state for scan name

  const antlersRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Run only on the client
    const storedModelUrl = localStorage.getItem("matchModelUrl");
    const storedScanName = localStorage.getItem("scanname") || "Unnamed Scan";

    if (storedModelUrl) setModelUrl(storedModelUrl);
    if (storedScanName) setScanName(storedScanName); // Set the scan name state
  }, []);

  useEffect(() => {
    if (antlersRef.current && modelUrl) {
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xffffff);
      const camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
      );

      const hlight = new THREE.AmbientLight(0x404040, 100);
      scene.add(hlight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 10);
      directionalLight.position.set(0, 1, 0);
      directionalLight.castShadow = true;
      scene.add(directionalLight);

      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight - 120);
      antlersRef.current.appendChild(renderer.domElement);

      const loader = new STLLoader();
      loader.load(modelUrl, (geometry) => {
        const material = new THREE.MeshStandardMaterial({
          color: 0x595959,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.rotation.y = Math.PI;
        scene.add(mesh);

        const boundingBox = new THREE.Box3().setFromObject(mesh);
        const center = boundingBox.getCenter(new THREE.Vector3());
        const size = boundingBox.getSize(new THREE.Vector3());

        camera.position.set(center.x, center.y, center.z + size.z * 2.35);
        camera.lookAt(center);
        controls.target.copy(center);
      });

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.25;
      controls.screenSpacePanning = false;
      controls.enableZoom = true;

      const animate = () => {
        renderer.render(scene, camera);
        requestAnimationFrame(animate);
        controls.update();
      };

      animate();
    }
  }, [modelUrl]);

  return (
    <div className="bg-white">
      <div className="justify-start" ref={antlersRef}></div>
      <div className="items-start justify-end flex flex-col text-xl text-black font-semibold p-4">
        <div>Current File: {scanName}</div>
        <div>Pope &amp; Young Score: (placeholder)</div>
        <div>Boone &amp; Crockett Score: (placeholder)</div>
      </div>
    </div>
  );
}
