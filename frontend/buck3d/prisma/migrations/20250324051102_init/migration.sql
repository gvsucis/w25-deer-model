-- CreateTable
CREATE TABLE "Scan2DMatch" (
    "id" TEXT NOT NULL,
    "scanid" TEXT NOT NULL,
    "matchid" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Scan2DMatch_pkey" PRIMARY KEY ("id")
);
