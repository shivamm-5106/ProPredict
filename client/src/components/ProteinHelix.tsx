import React, { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, MeshDistortMaterial, Sphere } from "@react-three/drei";
import * as THREE from "three";

/**
 * Defensive R3F protein helix component for React 18 + TS.
 * - Use this in a TSX project (file extension .tsx)
 * - Ensure packages installed (see commands above)
 */

const ProteinStructure: React.FC = () => {
  // Use Group refs for accurate typings
  const groupRef = useRef<THREE.Group | null>(null);
  const innerGroupRef = useRef<THREE.Group | null>(null);

  useFrame((state) => {
    const t = state.clock.getElapsedTime();

    const group = groupRef.current;
    const inner = innerGroupRef.current;

    // safe checks
    if (group) {
      // rotation exists on Object3D
      group.rotation.y = t * 0.2;
      group.rotation.x = Math.sin(t * 0.3) * 0.1;
    }
    if (inner) {
      inner.rotation.y = -t * 0.3;
    }
  });

  const createHelix = (offset: number, phase: number) => {
    const points: { x: number; y: number; z: number }[] = [];
    const numPoints = 80;
    const radius = 1.5 + offset;
    const height = 6;

    for (let i = 0; i < numPoints; i++) {
      const angle = (i / numPoints) * Math.PI * 4 + phase;
      const y = (i / numPoints) * height - height / 2;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      points.push({ x, y, z });
    }
    return points;
  };

  const helices = useMemo(
    () => [
      { points: createHelix(0, 0), color: "#00b4d8", emissive: "#00b4d8" },
      { points: createHelix(0.8, Math.PI / 2), color: "#0077b6", emissive: "#0077b6" },
      { points: createHelix(1.6, Math.PI), color: "#90e0ef", emissive: "#90e0ef" },
    ],
    []
  );

  const coreSpheres = useMemo(() => {
    const arr: { x: number; y: number; z: number; scale: number }[] = [];
    const count = 12;
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2;
      arr.push({
        x: Math.cos(angle) * 0.8,
        y: Math.sin(angle * 2) * 2,
        z: Math.sin(angle) * 0.8,
        scale: 0.3 + Math.random() * 0.2,
      });
    }
    return arr;
  }, []);

  return (
    <group ref={groupRef} scale={1.2}>
      {helices.map((helix, idx) => (
        <group key={idx}>
          {helix.points.map((p, i) => (
            <Sphere key={i} args={[0.12, 16, 16]} position={[p.x, p.y, p.z]}>
              <meshStandardMaterial
                color={helix.color}
                emissive={helix.emissive}
                emissiveIntensity={0.6}
                metalness={0.8}
                roughness={0.2}
              />
            </Sphere>
          ))}
        </group>
      ))}

      <group ref={innerGroupRef}>
        {coreSpheres.map((s, i) => (
          <Sphere key={i} args={[s.scale, 32, 32]} position={[s.x, s.y, s.z]}>
            <MeshDistortMaterial
              color="#003566"
              emissive="#00b4d8"
              emissiveIntensity={0.4}
              distort={0.4}
              speed={2}
              metalness={0.9}
              roughness={0.1}
            />
          </Sphere>
        ))}
      </group>

      {[0, 1, 2, 3].map((i) => (
        <mesh key={i} position={[0, i * 2 - 3, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[2, 0.05, 16, 100]} />
          <meshStandardMaterial
            color="#001d3d"
            emissive="#0077b6"
            emissiveIntensity={0.3}
            transparent
            opacity={0.6}
          />
        </mesh>
      ))}

      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={2} color={"#00b4d8"} />
      <pointLight position={[-10, -10, -10]} intensity={1.5} color={"#90e0ef"} />
      <pointLight position={[0, 0, 10]} intensity={1} color={"#0077b6"} />
    </group>
  );
};

const ProteinHelix: React.FC = () => {
  return (
    <div className="w-full h-full">
      <Canvas camera={{ position: [0, 0, 12], fov: 50 }}>
        <ProteinStructure />
        <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.5} />
      </Canvas>
    </div>
  );
};

export default ProteinHelix;
