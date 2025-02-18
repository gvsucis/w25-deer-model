import Image from "next/image";
export default function Services() {
  return (
    <main className="bg-stone-300 min-h-screen font-[family-name:var(--font-geist-sans)]">
      <div className="grid grid-cols-[324px_1fr_1fr] gap-4 p-4 w-full h-[1000px] pt-[120px] whitespace-nowrap">
        {/* First Column - Upload Box & Amount Due */}
        <div className="relative items-start border-r-4 border-black">
          {/* Upload File Box */}
          <div className="flex flex-col items-center text-black text-lg justify-center border-4 border-black bg-white p-1 h-[200px] w-[300px] mt-8">
            Pick a Scan to Complete Service
            <form action="" className="text-sm p-3 space-y-8 items-center">
              <input
                type="file"
                name="buck_scan"
                className="text-black w-full"
              />
              <input
                type="submit"
                value="Upload Scan"
                className="block text-lg text-white border-solid border-black border-2 bg-blue-600 rounded hover:bg-[#383838] dark:hover:bg-[#ccc] px-2 h-8"
              />
            </form>
          </div>

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
          <div className="text-lg text-bold ml-2">
            Shoulder: (placeholder)
          </div>
          <div className="text-lg text-bold ml-2">
            Full: (placeholder)
          </div>
        </div>
      </div>
    </main>
  );
}