"use client"
import AuthButton from "../Buttons/AuthButton"

export default function MainNavbar() {
    return (
        <header className="flex items-center justify-between font-sans font-bold p-5 bg-white shadow-md w-full h-[100px] fixed top-0 left-0">
            <h1 className="whitespace-nowrap text-black text-2xl">BuckView3D - Bring Your Buck Photos to Virtual Life</h1>

            {/* 3 buttons and profile to click at top of screen */}
            <div className="flex p-5 gap-8">
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
                <a>
                <AuthButton/>
                </a>
            </div>
        </header>
    )
}