"use client";
import UploadFile from "../components/uploadDropbox/UploadFile";
import { useEffect, useState } from "react";
import axios from "axios";
import { useSession } from "next-auth/react";

interface Scan {
  scanid: string;
  userid: string;
  url: string;
  createdAt: string;
  name: string;
}
interface Match {
  scanid: string;
  matchid: string;
}

export default function Scans() {
  const { data: session, status } = useSession();
  const [scans, setScans] = useState<Scan[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadedImages, setLoadedImages] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (status === "authenticated" && session?.user?.id) {
      setIsLoading(true);

      const fetchScans = axios
        .get(`http://localhost:8000/api/scans?userid=${session.user.id}`)
        .then((response) => setScans(response.data))
        .catch((error) => console.error("internal error", error));

      const fetchMatches = axios
        .get(`http://localhost:8000/api/matches?userid=${session.user.id}`)
        .then((res) => setMatches(res.data))
        .catch((err) => console.error("match fetch error", err));

      // Wait for both requests to complete
      Promise.all([fetchScans, fetchMatches]).finally(() => {
        // Set loading to false only after data is fetched
        setIsLoading(false);
      });
      axios
        .get(`http://localhost:8000/api/matches?userid=${session.user.id}`)
        .then((res) => setMatches(res.data))
        .catch((err) => console.error("match fetch error", err));
    }
  }, [status, session]);

  // Track loaded images
  const handleImageLoad = (scanId: string) => {
    setLoadedImages((prev) => {
      const updated = new Set(prev);
      updated.add(scanId);
      return updated;
    });
  };


  const handleViewClick = async (scan: Scan) => {
    localStorage.setItem("scanid", scan.scanid);
    localStorage.setItem("scanurl", scan.url);
  const handleViewClick = async (scan: Scan) => {
    localStorage.setItem("scanurl", scan.url);
    localStorage.setItem("scanname", scan.name || "Unnamed Scan"); // Save name

    const existingMatch = matches.find((m) => m.scanid === scan.scanid);

    let modelUrl: string;

    if (existingMatch) {
      // Use existing matchid
      modelUrl = `https://buckview3d.s3.us-east-1.amazonaws.com/3dmodels/${existingMatch.matchid}.stl`;
    } else {
      try {
        const response = await axios.post(
          "http://localhost:8000/api/match-antler",
          {
            scanid: scan.scanid,
            userid: session?.user?.id,
            fileUrl: scan.url,
          }
        );

        modelUrl = response.data.modelUrl;
      } catch (error) {
        console.error("Error generating 3D match:", error);
        return;
      }
    }

    localStorage.setItem("matchModelUrl", modelUrl);
    window.location.href = "/viewer";
  };

  // NOTE: Make sure to delete from the S3 bucket as well with what ever key you use
  const handleDeleteClick = async (scan: Scan) => {
    const confirmDelete = window.confirm(`Are you sure you want to delete "${scan.name || 'this scan'}"?`);
  
    if (!confirmDelete) return;
  
    try {
      await axios.delete(`http://localhost:8000/api/scans?userid=${session?.user?.id}&scanid=${scan.scanid}`);
      // Optionally update the UI after deletion
      setScans((prev) => prev.filter((s) => s.scanid !== scan.scanid));
    } catch (error) {
      console.error("Error deleting scan:", error);
      alert("Failed to delete scan. Please try again.");
    }
  };

  const handleRenameClick = async (scan: Scan) => {
    const newName = prompt("Enter a new name for this scan:", scan.name || "");
    if (!newName || newName.trim() === "") return;
  
    try {
      const response = await axios.patch(
        `http://localhost:8000/api/scans?userid=${session?.user?.id}&scanid=${scan.scanid}`,
        { name: newName }
      );
  
      if (response.status === 200) {
        setScans((prev) =>
          prev.map((s) =>
            s.scanid === scan.scanid ? { ...s, name: newName } : s
          )
        );
      }
    } catch (err) {
      console.error("Rename failed", err);
      alert("Could not rename scan.");
    }
  };

  return (
    <main className="bg-white min-h-screen font-[family-name:var(--font-geist-sans)]">
      <div className="grid grid-cols-[300px_1fr] gap-4 p-4 w-full pt-[120px]">
        {/*Upload file box*/}
        <div className="flex flex-col items-center text-black text-lg justify-start box border-4 border-black bg-orange-500 p-1 h-[300px] w-[400px]">
          <UploadFile />
        </div>
      </div>

      {/*Grid for previous scans*/}
      <div className="text-black text-2xl font-bold pb-6 ml-6">
        Previous Scans
      </div>

      {isLoading ? (
        <div className="flex justify-center items-center p-10">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-orange-500"></div>
        </div>
      ) : (
        <div className="grid grid-cols-5 gap-10 w-full justify-evenly ml-6 mr-6 text-center">
          {scans.length > 0 ? (
            scans.map((scan, index) => (
              <div
                key={scan.scanid}
                className="relative rounded-lg w-32 h-32 border-4 border-black overflow-hidden"
              >
                {/* Load images in the background */}
                <div
                  className={!loadedImages.has(scan.scanid) ? "invisible" : ""}
                >
                  <img
                    src={scan.url}
                    alt={`Scan ${index + 1}`}
                    className="object-cover w-full h-full"
                    onLoad={() => handleImageLoad(scan.scanid)}
                    onError={() => handleImageLoad(scan.scanid)} // Handle error case too
                  />
                </div>

                {/* Show loading spinner while image loads */}
                {!loadedImages.has(scan.scanid) && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-orange-500"></div>
                  </div>
                )}

                <a
                  className="absolute bottom-2 right-2 rounded-full bg-orange-500 transition-colors flex items-center justify-center text-black hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm h-5 w-12 px-4 cursor-pointer"
                  onClick={() => handleViewClick(scan)}
                >
                  View
                </a>
              </div>
            ))
          ) : (
            <p className="col-span-5 text-black">No scans available yet.</p>
          )}
        </div>
      )}
      <div className="grid grid-cols-8 ml-6 mr-6 gap-6">
        {scans.length > 0 ? (
          scans.map((scan, index) => (
            <div
              key={scan.scanid}
              className="relative rounded-lg w-full h-[275px] border-4 mb-6 border-black overflow-hidden"
            >
              <img
                src={scan.url}
                alt={`Scan ${index + 1}`}
                className="items-start object-cover w-full h-[215px] border-b-4 border-black"
              />
              <a
                className="absolute bottom-2 left-[155px] rounded-full bg-orange-500 transition-colors flex items-center justify-center text-black font-medium hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm h-5 w-12 px-3"
                onClick={() => handleViewClick(scan)}
              >
                View
              </a>
              <a
                className="absolute bottom-2 left-2 rounded-full bg-black transition-colors flex items-center justify-center text-white font-medium hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm h-5 w-auto px-3"
                onClick={() => handleDeleteClick(scan)}
              >
                Delete
              </a>
              <a
                className="absolute bottom-2 left-[80px] rounded-full bg-blue-500 transition-colors flex items-center justify-center text-black font-medium hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm h-5 w-auto px-2"
                onClick={() => handleRenameClick(scan)}
              >
                Rename
              </a>
              <p className="text-black text-sm font-medium pl-2"> Name: {scan.name ? scan.name : 'Unnamed Scan'}</p>
            </div>
          ))
        ) : (
          <p className="m-6 text-black">No scans available yet.</p>
        )}
      </div>
    </main>
  );
}
