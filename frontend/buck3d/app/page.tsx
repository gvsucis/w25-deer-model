"use client"

import React, { useRef, useEffect } from "react";
import Image from "next/image";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

export default function Home() {
  const antlersRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (antlersRef.current) {
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
      const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true});
      renderer.setSize(1500, 1000);
      renderer.setClearColor(0x000000, 0);
      antlersRef.current.appendChild(renderer.domElement);

      const light = new THREE.AmbientLight(0xffffff, 1);
      scene.add(light);

      const geometry = new THREE.BoxGeometry(2, 2, 2);
      const material = new THREE.MeshBasicMaterial({ color: 0xf97316 });
      const cube = new THREE.Mesh(geometry, material);
      scene.add(cube);

      const edges = new THREE.EdgesGeometry(geometry);
      const lineMaterial = new THREE.LineBasicMaterial({ color: 0x000000 });
      const lineSegments = new THREE.LineSegments(edges, lineMaterial);
      scene.add(lineSegments);

      camera.position.z = 5;

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

      <div className="p-6 mt-6 mb-6 flex justify-center items-center w-[500px]">
        <div ref={antlersRef} className="w-full max-w-4xl mx-auto flex justify-center items-center"></div>
      </div>

      <div className="bg-white flex justify-center" style={{ marginBottom: '12px' }}>
        <div className="flex justify-center items-center">
          <Image src="/logo.png" width={400} height={500} alt="Logo" style={{ objectFit: "contain", height: "500px" }}/>
        </div>
      </div>

    </div>
  );
};