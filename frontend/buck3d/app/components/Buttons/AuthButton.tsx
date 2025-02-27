"use client";

import { signIn, signOut, useSession } from "next-auth/react";

export default function AuthButton() {
  const { data: session } = useSession();

  return session ? (
    <div className="flex items-center space-x-2 text-black">
      {session.user?.image && (
        <img
          src={session.user.image}
          alt="Profile Image"
          className="w-8 h-8 rounded-full"
        />
      )}
      <span>Profile</span>
      <button onClick={() => signOut()} className="ml-2">
        Sign Out
      </button>
    </div>
  ) : (
    <button onClick={() => signIn("google")} className="text-black">
      Sign In with Google
    </button>
  );
}
