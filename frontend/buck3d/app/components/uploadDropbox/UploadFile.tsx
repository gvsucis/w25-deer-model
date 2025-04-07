"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { useSession } from "next-auth/react";

export default function UploadFile() {
  const { data: session, status } = useSession();
  const [uploading, setUploading] = useState(false);
  const [fileUrl, setFileUrl] = useState<string | null>(null);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (status === "loading") {
        console.warn("Session is still loading. Please try again shortly.");
        return;
      }

      if (!session?.user?.id) {
        console.error("User is not authenticated or session is not available.");
        return;
      }

      const file = acceptedFiles[0];
      if (!file) return;

      setUploading(true);

      try {
        const { data } = await axios.post("/api/upload", {
          filename: file.name,
          filetype: file.type,
        });

        const { uploadUrl, fileUrl } = data;

        await fetch(uploadUrl, {
          method: "PUT",
          headers: { "Content-Type": file.type },
          body: file,
        });

        setFileUrl(fileUrl);
        console.log("File uploaded:", fileUrl);

        await axios.post("http://localhost:8000/api/scans", {
          userid: session.user.id,
          url: fileUrl,
        });
      } catch (error) {
        console.error("Upload failed", error);
      } finally {
        setUploading(false);
      }
    },
    [session, status]
  );

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  return (
    <div
      {...getRootProps()}
      className="w-[392px] h-[292px] p-5 cursor-pointer bg-gray-100 text-center font-bold"
    >
      <input {...getInputProps()} />
      {uploading ? (
        <p>Uploading...</p>
      ) : (
        <p>
          Upload Your Buck Photo Here
          <br />
          Drag & Drop a file or click to select
        </p>
      )}
      {fileUrl && (
        <div className="mt-4">
          <img
            src={fileUrl}
            alt="Uploaded file preview"
            className="max-w-full h-auto"
          />
        </div>
      )}
    </div>
  );
}
