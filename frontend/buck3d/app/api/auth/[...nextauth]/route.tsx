import NextAuth, { NextAuthOptions, Session } from "next-auth";
import GoogleProvider from "next-auth/providers/google";
import { prisma } from "../../../lib/prisma";
import { PrismaAdapter } from "@next-auth/prisma-adapter";
import { JWT } from "next-auth/jwt";
import type { Account, DefaultSession, User } from "next-auth";

declare module "next-auth" {
  interface Session extends DefaultSession {
    accessToken?: string;
    user: {
      id: string; // Ensure `id` is included
      name: string;
      email: string;
      image?: string;
    };
  }
  interface JWT {
    accessToken: string;
    id: string; // Store `id` inside JWT
  }
}

export const authOptions: NextAuthOptions = {
  adapter: PrismaAdapter(prisma),
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
  ],
  callbacks: {
    async jwt({
      token,
      account,
      user,
    }: {
      token: JWT;
      account?: Account | null;
      user?: User;
    }) {
      // When signing in, both account and user are defined.
      if (account && user) {
        token.accessToken = account.access_token!;
        token.id = user.id; // Directly assign the user ID at sign‑in
      }
      return token;
    },
    async session({ session, token }: { session: Session; token: JWT }) {
      // Use the token values we set in the jwt callback.
      session.accessToken = token.accessToken as string;
      session.user.id = token.sub as string;
      console.log("Session Callback - Session:", session);
      return session;
    },
  },
  secret: process.env.NEXTAUTH_SECRET,
};

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
