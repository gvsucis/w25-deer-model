export default function Services() {
    return (
        <main className="bg-stone-300 min-h-screen font-[family-name:var(--font-geist-sans)]">
            <div className="relative grid grid-cols-[315px_1fr_1fr] gap-4 p-4 w-full h-[1000px] pt-[120px]">
                {/* First Column - Upload Box & Amount Due */}
                <div className="flex flex-col items-start space-y-4 border-r-4 border-black pr-4">
                    {/* Upload File Box */}
                    <div className="flex flex-col items-center text-black text-lg justify-center border-4 border-black bg-white p-1 h-[200px] w-[300px]"
                    >
                        Pick a Scan to Complete Service
                        <form action="" className="text-sm p-3 space-y-4">
                            <input type="file" name="buck_scan" className="text-black w-full" />
                            <input
                                type="submit"
                                value="Upload Scan"
                                className="text-lg text-white border-solid border-black border-2 bg-blue-600 rounded hover:bg-[#383838] dark:hover:bg-[#ccc] px-2 h-8"
                            />
                        </form>
                    </div>

                    {/* Amount Due Text (Now Positioned Under the Box) */}
                    <div className="absolute bottom-0 text-start text-black text-2xl font-bold"
                    >
                        Amount Due: $$$
                    </div>
                </div>

                {/* Second Column */}
                <div className="grid text-black text-2xl font-bold border-r-4 border-black pr-4"
                >
                    3D Printed Model
                    {/*scale selection*/}
                    
                </div>

                {/* Third Column */}
                <div className="grid text-black text-2xl font-bold border-r-4 border-black pr-4"
                >
                    Taxidermy
                </div>
            </div>
        </main>
    );
}