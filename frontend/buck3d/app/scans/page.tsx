export default function Scans() {
  return (
    <main className="bg-stone-300 min-h-screen font-[family-name:var(--font-geist-sans)]">
      <div className="text-2xl text-black"
      >
        3D Viewing of Antlers
      </div>
      {/*Upload file box*/}
      <div className="flex flex-col items-center text-black text-lg justify-center box border-4 border-black bg-white p-1 h-[200px] w-[300px] absolute top-40 left-6"
      >
        Upload Your Buck Photo Here
        {/*need to add action photo upload to cloud*/}
        <form action="" className="text-sm p-3 space-y-12">
          <input
            type="file"
            name="buck_photo"
            className="text-black w-full"
            id="file-input"
          />
          <input
            type="submit"
            value="Upload Image"
            className="text-lg text-white border-solid border-black border-2 bg-blue-600 rounded hover:bg-[#383838] dark:hover:bg-[#ccc] px-2 h-8"
          />
        </form>
      </div>
      {/*Antler Viewing Box*/}
      <div className="grid grid-rows-[40px-600px-40px] absolute w-[700px] top-[96px] right-6">
        <div className="row-span-1 col-span-full p-4 text-2xl text-bold text-black items-start"
        >
          3D Viewer
        </div>
        <div className="row-span-2 col-span-full text-5xl text-black text-bold h-[600px] border-4 border-black"
        >
          3D antlers go here
        </div>
        {/*Current file and scoring text*/}
        <div className="flex flex-col text-xl text-black text-bold p-4 items-start">
          <div className="flex"
          >
            Current Scan: file.obj
          </div>
          <div className="flex"
          >
            Pope & Young Score: 0
          </div>
          <div className="flex"
          >
            Boone & Crockett Score: 0
          </div>
        </div>
      </div>
    </main>
  );
}
