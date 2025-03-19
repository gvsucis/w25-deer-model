"use client"

import React, { useRef, useEffect } from "react";
import Image from "next/image";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader";

export default function Home() {
  const antlersRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (antlersRef.current) {
      // Create the 3D scene
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
      const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true});
      renderer.setSize(750, 950);
      renderer.setClearColor(0x000000, 0);
      antlersRef.current.appendChild(renderer.domElement);

      const light = new THREE.AmbientLight(0xffffff, 1);
      scene.add(light);
      
      // Load the antler model to the scene
      const loader = new STLLoader();
      loader.load("/Antler6F.stl", (geometry) => {
        const material = new THREE.MeshBasicMaterial({ color: 0x808080, wireframe: false });
        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);

        // Initial orientation
        mesh.rotation.y = Math.PI;

        // Make a box around the model
        const boundingBox = new THREE.Box3().setFromObject(mesh);
        const center = boundingBox.getCenter(new THREE.Vector3());
        const size = boundingBox.getSize(new THREE.Vector3());

        // Center the camera on the model
        camera.position.set(center.x, center.y, center.z + size.z * 2);
        camera.lookAt(center);
        controls.target.copy(center);
      });

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.25;
      controls.screenSpacePanning = false;
      controls.enableZoom = false;

      const animate = () => {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      };

      animate();
    }
  }, []);

  return (
    <div className="bg-white min-h-screen flex flex-row items-start justify-between pt-12">
      { /* Info Section */ }
      <div className="bg-orange-500 text-white p-4 mt-20 ml-6 max-w-lg shadow-lg rounded-lg">
        <p className="mb-2 font-bold">1. Upload a photo of your buck</p>
        <p className="mb-2 font-bold">2. View the 3D model of your antlers</p>
        <p className="mb-2 font-bold">3. Purchase a 3D printed replica or taxidermy</p>
        <div className="flex justify-center items-center gap-2 mt-4">
          <Image src="/whitetail-deer-.jpg" width={200} height={200} alt="Buck"/>
          <span className="text-2x1">→</span>
          <Image src="/3D_printed_photo.jpg" width={200} height={200} alt="Antlers"/>
        </div>
          <p className="font-bold text-lg mt-4 text-center">Subscription Available!</p>
      </div>

      { /* 3D Model */ }
      <div className="p-6 mt-6 mb-6 flex justify-center items-center w-[500px]">
        <div ref={antlersRef} className="w-full max-w-4xl mx-auto flex justify-center items-center"></div>
      </div>

      { /* Logo */ }
      <div className="bg-white flex justify-center" style={{ marginBottom: '12px' }}>
        <div className="flex justify-center items-center">
          <Image src="/logo.png" width={400} height={500} alt="Logo" style={{ objectFit: "contain", height: "500px" }}/>
        </div>
      </div>

    </div>
  );
};