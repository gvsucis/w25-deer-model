"use client"
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

export default function Scans() {
  const { data: session, status } = useSession();
  const [scans, setScans] = useState<Scan[]>([]);
  useEffect(() => {
    if (status === "authenticated" && session?.user?.id) {
      axios
        .get(`http://localhost:8000/api/scans?userid=${session.user.id}`)
        .then((response) => setScans(response.data))
        .catch((error) => console.error("internal eror", error));
    }
  }, [status, session]);
  return (
    <main className="bg-stone-300 min-h-screen font-[family-name:var(--font-geist-sans)]">
      <div className="grid grid-cols-[300px_1fr] gap-4 p-4 w-full pt-[120px]">
        {/*Upload file box*/}
        <div className="flex flex-col items-center text-black text-lg justify-center box border-4 border-black bg-white p-1 h-[200px] w-[300px]">
          <p>Upload Your Buck Photo Here</p>
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
        {/*Antler Viewing Box*/}
        <div className="whitespace-nowrap items-start column-start-2 grid grid-rows-[40px-1fr-40px]">
          <div className="p-4 text-2xl font-bold text-black items-start">
            3D Viewer
          </div>
          <div className="text-5xl text-black h-[900px] w-[1200px] border-4 border-black">
            3D antlers go here
          </div>
          {/*Current file and scoring text*/}
          <div className="flex flex-col text-xl text-black font-semibold p-4 items-start">
            <div className="">Current Scan: (placeholder)</div>
            <div className="">Pope & Young Score: (placeholder)</div>
            <div className="">Boone & Crockett Score: (placeholder)</div>
          </div>
        </div>
      </div>
      {/*Grid for previous scans*/}
      <div className="text-black text-2xl font-bold pb-6 ml-6">
        Previous Scans
      </div>
      <div className="grid grid-cols-5 gap-10 w-full justify-evenly ml-6 mr-6 text-center">
        {scans.length > 0 ? (
          scans.map((scan, index) => (
            <div
              key={scan.scanid}
              className="relative rounded-lg w-32 h-32 border-4 border-black overflow-hidden"
            >
              <img
                src={scan.url}
                alt={`Scan ${index + 1}`}
                className="object-cover w-full h-full"
              />
              <a
                className="absolute bottom-2 right-2 rounded-full bg-orange-500 transition-colors flex items-center justify-center text-black hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm h-5 w-12 px-4"
                href={scan.url}
                target="_self"
                rel="noopener noreferrer"
              >
                View
              </a>
            </div>
          ))
        ) : (
          <p className="col-span-5">No scans available yet.</p>
        )}
      </div>
    </main>
  );
}