export default function Scans() {
  return (
    <main className="bg-stone-300 min-h-screen font-[family-name:var(--font-geist-sans)]">
      <div className="grid grid-cols-[300px_1fr] gap-4 p-4 w-full pt-[120px]">
        {/*Upload file box*/}
        <div className="column-start-1 flex flex-col items-center text-black text-lg justify-center box border-4 border-black bg-white p-1 h-[200px] w-[300px]">
          Upload Your Buck Photo Here
          {/*need to add action photo upload to cloud*/}
          <form action="" className="text-sm p-3 space-y-12">
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
          </form>
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
        <div className="relative rounded-lg bg-yellow-950 w-32 h-32 border-4 border-black">
          1
          <a
            className="absolute bottom-2 right-2 rounded-full border border-solid border-transparent bg-orange-500 transition-colors flex items-center justify-center bg-foreground text-black hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm h-5 w-12 px-4"
            href=""
            target="_self"
            rel="noopener noreferrer"
          >
            View
          </a>
        </div>
        <div className="relative rounded-lg bg-yellow-950 w-32 h-32 border-4 border-black">
          2
          <a
            className="absolute bottom-2 right-2 rounded-full border border-solid border-transparent bg-orange-500 transition-colors flex items-center justify-center bg-foreground text-black hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm h-5 w-12 px-4"
            href=""
            target="_self"
            rel="noopener noreferrer"
          >
            View
          </a>
        </div>
        <div className="relative rounded-lg bg-yellow-950 w-32 h-32 border-4 border-black">
          3
          <a
            className="absolute bottom-2 right-2 rounded-full border border-solid border-transparent bg-orange-500 transition-colors flex items-center justify-center bg-foreground text-black hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm h-5 w-12 px-4"
            href=""
            target="_self"
            rel="noopener noreferrer"
          >
            View
          </a>
        </div>
        <div className="relative rounded-lg bg-yellow-950 w-32 h-32 border-4 border-black">
          4
          <a
            className="absolute bottom-2 right-2 rounded-full border border-solid border-transparent bg-orange-500 transition-colors flex items-center justify-center bg-foreground text-black hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm h-5 w-12 px-4"
            href=""
            target="_self"
            rel="noopener noreferrer"
          >
            View
          </a>
        </div>
        <div className="relative rounded-lg bg-yellow-950 w-32 h-32 border-4 border-black">
          5
          <a
            className="absolute bottom-2 right-2 rounded-full border border-solid border-transparent bg-orange-500 transition-colors flex items-center justify-center bg-foreground text-black hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm h-5 w-12 px-4"
            href=""
            target="_self"
            rel="noopener noreferrer"
          >
            View
          </a>
        </div>
      </div>
    </main>
  );
}