"use client";
import React, { useState, useEffect } from "react";
import Image from "next/image";
import emailjs from "emailjs-com";
import axios from "axios";
import { useSession } from "next-auth/react";

interface Scan {
  scanid: string;
  userid: string;
  url: string;
  createdAt: string;
  name: string;
}

export default function Services() {
  const [serviceType, setServiceType] = useState<string[]>([]);
  const { data: session, status } = useSession();
  const [scans, setScans] = useState<Scan[]>([]);
  const [selectedScan, setSelectedScan] = useState<string | null>(null);

  useEffect(() => {
    if (status === "authenticated" && session?.user?.id) {
      axios
        .get(`http://localhost:8000/api/scans?userid=${session.user.id}`)
        .then((response) => setScans(response.data))
        .catch((error) => console.error("internal eror", error));
    }
  }, [status, session]);

  const handleScanSelect = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedScan(event.target.value);
  };

  const handleServiceTypeChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const { name, checked } = event.target;
    const updatedServices = checked
      ? [...serviceType, name]
      : serviceType.filter((type) => type !== name);
    setServiceType(updatedServices);
  };

  const sendEmail = () => {
    emailjs.send(
      'service_s752vvc', // REPLACE WITH YOUR SERVICE ID
      'template_yt796jk', // REPLACE WITH YOUR TEMPLATE ID
      {
        message: `Service Type: ${serviceType.join(", ")}\nPrice: $$$`,
      },
      'k7AWvwMDUA2tyGYhb' // REPLACE WITH YOUR USER API KEY
    ).then((response) => {
      console.log('Email sent successfully!', response.status, response.text);
    }).catch((err) => {
      console.error('Failed to send email:', err);
    });
  };

  return (
    <main className="bg-white min-h-screen font-[family-name:var(--font-geist-sans)]">
      <div className="grid grid-cols-[324px_1fr_1fr] max-[879px]:grid-cols-2 gap-4 p-4 w-full h-[1000px] pt-[120px] whitespace-nowrap">
        {/* First Column - Upload Box & Amount Due */}
        <div className="relative items-start border-r-4 border-black">
          {/* Upload File Box */}
          <div className="flex flex-col items-center text-black text-lg font-medium pt-6 border-4 border-black bg-gray-100 h-[200px] w-[300px] mt-8">
            Pick a Scan to Complete Service
            <form action="" className="text-md font-normal p-3 space-y-8 items-center h-[140px] w-[292px] border-b-4 border-orange-500">
              <select
                value={selectedScan || ""}
                onChange={handleScanSelect}
                className="text-black w-full border-2 border-black"
              >
                <option value="" disabled>
                  Select a scan
                </option>
                {scans.map((scan) => (
                  <option key={scan.scanid} value={scan.scanid}>
                    {scan.name || "Unnamed Scan"}
                  </option>
                ))}
              </select>
            </form>
          </div>

          {/* Purchase Options */}
          <div className="absolute bottom-24 mt-4 text-black">
            <h2 className="text-xl font-bold mb-2">Choose Purchase Options</h2>
            <form className="space-y-2">
              <div>
                <input
                  type="checkbox"
                  id="3dModel"
                  name="3dModel"
                  onChange={handleServiceTypeChange}
                />
                <label htmlFor="3dModel" className="ml-2">
                  3D Model
                </label>
              </div>
              <div>
                <input
                  type="checkbox"
                  id="taxidermy"
                  name="taxidermy"
                  onChange={handleServiceTypeChange}
                />
                <label htmlFor="taxidermy" className="ml-2">
                  Taxidermy
                </label>
              </div>
            </form>
          </div>

          {/* Purchase button */}
          <input
            type="button"
            value="Purchase"
            className="absolute bottom-10 block text-lg text-white border-solid border-black border-2 bg-orange-500 rounded hover:bg-[#383838] dark:hover:bg-[#ccc] px-2 h-10 mt-4"
            onClick={sendEmail}
          />

          {/* Amount Due Text (Now Positioned Under the Box) */}
          <div className="absolute bottom-0 text-start text-black text-2xl font-bold">
            Amount Due: $$$
          </div>
        </div>

        {/* Second Column */}
        <div className="text-black text-3xl font-bold border-r-4 border-black pr-4">
          3D Printed Model
          <Image
            className="border-black border-2 mb-4 mt-4"
            src="/3D_printed_photo.jpg"
            alt="3D printed example"
            width={296}
            height={222}
          />
          {/*scale selection*/}
          <div className="text-lg text-bold ml-2">
            Full Scale: (placeholder)
          </div>
          <div className="text-lg text-bold ml-2">
            Half Scale: (placeholder)
          </div>
          <div className="text-lg text-bold ml-2">
            Quarter Scale: (placeholder)
          </div>
          {/*color selection*/}
          <div className="text-2xl text-bold mt-4 mb-2">Color</div>
          <div className="text-lg text-bold ml-2">White: (placeholder)</div>
          <div className="text-lg text-bold ml-2">Black: (placeholder)</div>
          <div className="text-lg text-bold ml-2">Painted: (placeholder)</div>
          <div className="text-lg text-bold ml-2">Other: (placeholder)</div>
          {/*Payment selection*/}
          <div className="text-2xl text-bold mt-4 mb-2">Payment</div>
          <div className="text-lg text-bold ml-2">
            Single Use: (placeholder)
          </div>
          <div className="text-lg text-bold ml-2">
            Subscription: (placeholder)
          </div>
        </div>

        {/* Third Column */}
        <div className=" text-black text-3xl font-bold">
          Taxidermy
          <Image
            className="border-black border-2 mb-4 mt-4"
            src="/taxidermy_photo.jpg"
            alt="taxidermy example"
            width={296}
            height={222}
          />
          {/*scale selection*/}
          <div className="text-lg text-bold ml-2">Shoulder: (placeholder)</div>
          <div className="text-lg text-bold ml-2">Full: (placeholder)</div>
        </div>
      </div>
    </main>
  );
}
