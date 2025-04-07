"use client";

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
      const camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
      );
      const renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: true,
      });
      renderer.setSize(window.innerWidth, window.innerHeight);
      antlersRef.current.appendChild(renderer.domElement);

      const hlight = new THREE.AmbientLight(0x404040, 100);
      scene.add(hlight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 10);
      directionalLight.position.set(0, 1, 0);
      directionalLight.castShadow = true;
      scene.add(directionalLight);

      // Load the antler model to the scene
      const loader = new STLLoader();
      loader.load("/Antler6F.stl", (geometry) => {
        const material = new THREE.MeshStandardMaterial({ color: 0x595959 });
        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);

        // Initial orientation
        mesh.rotation.y = Math.PI;

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
    <div className="bg-white min-h-screen flex flex-row">
      {/* Info Section */}
      <div className="absolute left-6 bg-orange-500 text-white text-xl border-black border-4 whitespace-nowrap p-6 mt-[120px] w-[650px] h-[450px] shadow-lg rounded-lg">
        <p className="mb-2 font-bold">1. Upload a photo of your buck</p>
        <p className="mb-2 font-bold">2. View the 3D model of your antlers and its attributes</p>
        <p className="mb-2 font-bold">3. Purchase a 3D printed replica or taxidermy</p>
        <div className="flex justify-center items-center gap-2 mt-4">
          <Image
            src="/whitetail-deer-.jpg"
            width={275}
            height={200}
            alt="Buck"
            className="border-black border-2"
          />
          <span className="text-2x1">→</span>
          <Image
            src="/3D_printed_photo.jpg"
            width={275}
            height={200}
            alt="Antlers"
            className= "border-black border-2"
          />
        </div>
        <p className="font-bold text-2xl mt-4 text-center">
          Subscription Available!
        </p>
      </div>

      {/* 3D Model */}
      <div
        ref={antlersRef}
        className="justify-center items-center mx-auto"
      ></div>

      {/* Logo */}
      <div className="absolute right-6 bg-transparent mt-[120px]">
        <Image src="/logo.png" width={344} height={456} alt="Logo" />
      </div>
    </div>
  );
}
