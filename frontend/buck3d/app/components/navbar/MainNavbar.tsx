export default function MainNavbar() {
    return (
        <header className="flex items-center justify-between font-sans font-bold p-5 bg-white shadow-md w-full fixed top-0 left-0">
            <h1 className="text-black text-2xl">BuckView3D - Bring Your Buck Photos to Virtual Life</h1>

            {/* 3 buttons and profile to click at top of screen */}
            <div className="flex gap-8">
                <a
                    className="rounded-full border border-solid border-transparent bg-orange-500 transition-colors flex items-center justify-center bg-foreground text-black gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5"
                    href="./"
                    target="_self"
                    rel="noopener noreferrer"
                >
                    Home
                </a>
                <a
                    className="rounded-full border border-solid border-transparent bg-orange-500 transition-colors flex items-center justify-center bg-foreground text-black gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5"
                    href="./services"
                    target="_self"
                    rel="noopener noreferrer"
                >
                    Services
                </a>
                <a
                    className="rounded-full border border-solid border-transparent bg-orange-500 transition-colors flex items-center justify-center bg-foreground text-black gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5"
                    href="./scans"
                    target="_self"
                    rel="noopener noreferrer"
                >
                    Scans
                </a>
                {/*add profile page*/}
                <a
                    className="text-stone-600 flex items-center hover:underline hover:underline-offset-4 text-sm"
                    href=""
                    target="_self"
                    rel="noopener noreferrer"
                >
                    Profile
                </a>
            </div>
        </header>
    )
}