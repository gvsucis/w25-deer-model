"use client";

import { useRef, useEffect } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader";

{
  /*Loads scan into a 3D viewer using ThreeJS*/
}

export default function Viewer() {
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
        .1,
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
      loader.load("/Antler6F.stl", (geometry) => {
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

  return <div ref={antlersRef}></div>;
}
