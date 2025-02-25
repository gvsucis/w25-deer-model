-- CreateTable
CREATE TABLE "Scan2D" (
    "scanid" TEXT NOT NULL,
    "userid" TEXT NOT NULL,
    "url" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Scan2D_pkey" PRIMARY KEY ("scanid")
);
