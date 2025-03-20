"use client";

import { useRef, useEffect, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader";

/*Loads scan into a 3D viewer using ThreeJS*/

export default function Viewer() {
  const [scanid, setScanid] = useState("");
  const scanUrl = localStorage.getItem("scanurl");

  useEffect(() => {
    const storedScanId = localStorage.getItem("scanid");
    if (storedScanId) setScanid(storedScanId);
  }, []);

  const antlersRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (antlersRef.current) {
      // create scene
      const scene = new THREE.Scene();
      // background color
      scene.background = new THREE.Color(0xffffff);
      // set camera constraints: FOV, AR, near and far planes
      const camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
      );
      // add ambient and direction light
      const hlight = new THREE.AmbientLight(0x404040, 100);
      scene.add(hlight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 10);
      directionalLight.position.set(0, 1, 0);
      directionalLight.castShadow = true;

      scene.add(directionalLight);

      // use entire window
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      antlersRef.current.appendChild(renderer.domElement);

      // load the stl scan
      const loader = new STLLoader();

      // EDIT THIS LATER FOR MODEL MATCH
      if (scanUrl) {
        loader.load(scanUrl, (geometry) => {
          const material = new THREE.MeshStandardMaterial({
            color: 0x595959,
          });
          const mesh = new THREE.Mesh(geometry, material);

          // Initial orientation
          mesh.rotation.y = Math.PI;
          scene.add(mesh);

          // Make a box around the model
          const boundingBox = new THREE.Box3().setFromObject(mesh);
          const center = boundingBox.getCenter(new THREE.Vector3());
          const size = boundingBox.getSize(new THREE.Vector3());

          // Center the camera on the model
          camera.position.set(center.x, center.y, center.z + size.z * 2.35);
          camera.lookAt(center);
          controls.target.copy(center);
        });
      } else {
        console.error("No scan URL found in localStorage");
      }

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
  }, []);

  return (
    <div className="bg-white">
      {/* Antler Scan showed on screen */}
      <div className="justify-start" ref={antlersRef}></div>
      {/* Current file and scoring text */}
      <div className="items-start justify-end flex flex-col text-xl text-black font-semibold p-4">
        <div className="">Current File: {scanid}</div>
        {/* Need to implement score values from database */}
        <div className="">Pope & Young Score: (placeholder)</div>
        <div className="">Boone & Crockett Score: (placeholder)</div>
      </div>
    </div>
  );
}
