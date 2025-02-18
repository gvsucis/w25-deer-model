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
      renderer.setSize(150, 150);
      renderer.setClearColor(0x000000, 0);
      antlersRef.current.appendChild(renderer.domElement);

      const light = new THREE.AmbientLight(0xffffff, 1);
      scene.add(light);

      const geometry = new THREE.BoxGeometry(3, 3, 3);
      const material = new THREE.MeshBasicMaterial({ color: 0x808080 });
      const cube = new THREE.Mesh(geometry, material);
      scene.add(cube);

      camera.position.z = 5;

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.25;
      controls.screenSpacePanning = false;

      const animate = () => {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      };

      animate();
    }
  }, []);

  return (
    <div className="bg-gray-100 min-h-screen flex flex-col items-center pt-24">
      <div className="bg-white p-6 mt-6 w-11/12 max-w-lg shadow-lg rounded-lg text-center">
        <div className="flex justify-center items-center gap-4 mt-4">
          <Image src="/3Dbuck-logo.png" width={275} height={275} alt="Logo"/>
        </div>
          <div ref={antlersRef} className="w-11/12 max-w-lg mt-4 mx-auto flex justify-center items-center"></div>
      </div>

      <div className="bg-orange-500 text-white p-6 mt-6 w-11/12 max-w-lg rounded-lg text-center">
        <p className="mb-2 font-bold">1. Upload a photo of your buck</p>
        <p className="mb-2 font-bold">2. View the 3D model of your antlers</p>
        <p className="mb-2 font-bold">3. Purchase a 3D printed replica or taxidermy</p>
        <div className="flex justify-center items-center gap-4 mt-4">
          <Image src="/whitetail-deer-.jpg" width={200} height={200} alt="Buck"/>
          <span className="text-2x1">→</span>
          <Image src="/3D_printed_photo.jpg" width={200} height={200} alt="Antlers"/>
        </div>
      </div>

      <p className="text-orange-700 font-bold text-lg mt-4">Subscription Available!</p>
    </div>
  );
};