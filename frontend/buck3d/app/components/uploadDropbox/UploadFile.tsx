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
      // Wait until the session has loaded
      if (status === "loading") {
        console.warn("Session is still loading. Please try again shortly.");
        return;
      }

      // Ensure the user is authenticated before proceeding
      if (!session?.user?.id) {
        console.error("User is not authenticated or session is not available.");
        return;
      }

      const file = acceptedFiles[0];
      if (!file) return;

      setUploading(true);

      try {
        // Request a signed URL for the upload
        const { data } = await axios.post("/api/upload", {
          filename: file.name,
          filetype: file.type,
        });

        const { uploadUrl, fileUrl } = data;

        await axios.put(uploadUrl, file, {
          headers: { "Content-Type": file.type },
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
    <div {...getRootProps()} className="border p-5 cursor-pointer bg-gray-100">
      <input {...getInputProps()} />
      {uploading ? (
        <p>Uploading...</p>
      ) : (
        <p>Drag & Drop a file or click to select</p>
      )}
      {fileUrl && (
        <p>
          Uploaded:{" "}
          <a href={fileUrl} target="_blank" rel="noreferrer">
            {fileUrl}
          </a>
        </p>
      )}
    </div>
  );
}
