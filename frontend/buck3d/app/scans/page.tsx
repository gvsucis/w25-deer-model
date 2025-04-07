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
}
interface Match {
  scanid: string;
  matchid: string;
}

export default function Scans() {
  const { data: session, status } = useSession();
  const [scans, setScans] = useState<Scan[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);

  useEffect(() => {
    if (status === "authenticated" && session?.user?.id) {
      axios
        .get(`http://localhost:8000/api/scans?userid=${session.user.id}`)
        .then((response) => setScans(response.data))
        .catch((error) => console.error("internal eror", error));

       axios
         .get(`http://localhost:8000/api/matches?userid=${session.user.id}`)
         .then((res) => setMatches(res.data))
         .catch((err) => console.error("match fetch error", err));
    }
  }, [status, session]);

   const handleViewClick = async (scan: Scan) => {
     localStorage.setItem("scanid", scan.scanid);
     localStorage.setItem("scanurl", scan.url);

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
   //ADD functions for buttons
   //const handleDeleteClick = {}
   //const handleRenameClick = {}

  return (
    <main className="bg-white min-h-screen font-[family-name:var(--font-geist-sans)]">
      <div className="grid grid-cols-[300px_1fr] gap-4 p-4 w-full pt-[120px]">
        {/*Upload file box*/}
        <div className="flex flex-col items-center text-black text-lg justify-start box border-4 border-black bg-orange-500 p-1 h-[300px] w-[400px]">
          <UploadFile />
          {/*need to add action photo upload to cloud*/}
          {/* <form action="" className="text-sm p-3 space-y-12">
            <input
              type="file"
              name="buck_photo"
              className="text-black w-full"
            />
            <input
              type="submit"
              value="Upload Image"
              className="text-lg text-white border-solid border-black border-2 bg-blue-600 rounded hover:bg-[#383838] dark:hover:bg-[#ccc] px-2 h-8"
            />
          </form> */}
        </div>
      </div>
      {/*Grid for previous scans*/}
      <div className="text-black text-2xl font-bold pb-6 ml-6">
        Previous Scans
      </div>
      <div className="grid grid-cols-8 ml-6 mr-6 gap-6">
        {scans.length > 0 ? (
          scans.map((scan, index) => (
            <div
              key={scan.scanid}
              className="relative rounded-lg w-full h-64 border-4 mb-6 border-black overflow-hidden"
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
                //onclick
              >
                Delete
              </a>
              <a
                className="absolute bottom-2 left-[80px] rounded-full bg-blue-500 transition-colors flex items-center justify-center text-black font-medium hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm h-5 w-auto px-2"
                //onclick
              >
                Rename
              </a>
            </div>
          ))
        ) : (
          <p className="m-6 text-black">No scans available yet.</p>
        )}
      </div>
    </main>
  );
}
