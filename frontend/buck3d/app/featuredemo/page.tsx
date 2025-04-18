"use client";
import React, { useState, useRef } from "react";
import axios from "axios";

export default function FeatureDemo() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);

      // Create a preview URL
      const fileUrl = URL.createObjectURL(selectedFile);
      setPreviewUrl(fileUrl);

      // Reset any previous results
      setAnalysisResults(null);
      setError(null);
    }
  };

  const analyzeDeerParts = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Create FormData to send the file directly
      const formData = new FormData();
      formData.append("file", file);
      // Use fixed values for threshold and scale_factor
      formData.append("threshold", "0.3");
      formData.append("scale_factor", "1.0");

      // Send directly to the analyze-parts endpoint
      const response = await axios.post(
        "http://localhost:8000/api/analyze-parts",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      // Set results
      setAnalysisResults(response.data);

      // Log results for debugging
      console.log("Analysis results:", response.data);
    } catch (error) {
      console.error("Error analyzing image:", error);
      setError("Failed to analyze image. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFile(null);
    setPreviewUrl(null);
    setAnalysisResults(null);
    setError(null);

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold mb-4">Deer Antler Measurement Tool</h1>
      <p className="text-lg text-gray-700 mb-6">
        Upload a photo of a deer to detect and measure distances between key
        features such as antlers, eyes, ears, and more.
      </p>

      <div className="bg-white shadow-md rounded-lg p-6 mb-8">
        <div className="mb-4">
          <label
            className="block text-gray-700 text-sm font-bold mb-2"
            htmlFor="image-upload"
          >
            Upload a deer image
          </label>
          <input
            id="image-upload"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            ref={fileInputRef}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-md file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100"
          />
        </div>

        <div className="flex gap-3">
          <button
            onClick={analyzeDeerParts}
            disabled={!file || loading}
            className={`px-4 py-2 rounded-md text-white font-medium ${
              !file || loading
                ? "bg-blue-400 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700"
            }`}
          >
            {loading ? (
              <div className="flex items-center">
                <svg
                  className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Analyzing...
              </div>
            ) : (
              "Analyze Image"
            )}
          </button>
          <button
            onClick={resetForm}
            className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-md text-gray-800 font-medium"
          >
            Reset
          </button>
        </div>
      </div>

      {error && (
        <div
          className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6"
          role="alert"
        >
          <p>{error}</p>
        </div>
      )}

      {previewUrl && !analysisResults && (
        <div className="bg-white shadow-md rounded-lg overflow-hidden mb-8">
          <div className="border-b border-gray-200 bg-gray-50 px-4 py-2 text-gray-700 font-medium">
            Uploaded Image
          </div>
          <div className="p-4 flex justify-center">
            <img
              src={previewUrl}
              alt="Uploaded deer"
              className="max-h-[500px] object-contain"
            />
          </div>
        </div>
      )}

      {analysisResults && (
        <div className="grid lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2 bg-white shadow-md rounded-lg overflow-hidden">
            <div className="border-b border-gray-200 bg-gray-50 px-4 py-2 text-gray-700 font-medium">
              Analysis Result
            </div>
            <div className="p-4 flex justify-center">
              <img
                src={analysisResults.result_url}
                alt="Analysis result"
                className="max-h-[600px] object-contain"
              />
            </div>
          </div>
          <div className="bg-white shadow-md rounded-lg overflow-hidden">
            <div className="border-b border-gray-200 bg-gray-50 px-4 py-2 text-gray-700 font-medium">
              Detected Features
            </div>
            <div className="p-4">
              {analysisResults && (
                <>
                  <h3 className="font-bold text-lg mb-2">Summary</h3>
                  <div className="mb-4 flex flex-wrap">
                    {Object.entries(analysisResults.class_counts || {}).map(
                      ([cls, count]: [string, any]) => (
                        <span
                          key={cls}
                          className="bg-blue-100 text-blue-800 text-xs font-medium mr-2 mb-2 px-2.5 py-0.5 rounded"
                        >
                          {cls}: {count}
                        </span>
                      )
                    )}
                  </div>

                  <h3 className="font-bold text-lg mb-2">Measurements</h3>
                  {analysisResults.measurements &&
                  analysisResults.measurements.length > 0 ? (
                    <div className="relative overflow-x-auto">
                      <table className="w-full text-sm text-left text-gray-500">
                        <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                          <tr>
                            <th scope="col" className="px-6 py-3">
                              From
                            </th>
                            <th scope="col" className="px-6 py-3">
                              To
                            </th>
                            <th scope="col" className="px-6 py-3">
                              Distance
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {analysisResults.measurements
                            .sort((a: any, b: any) => a.distance - b.distance)
                            .map((m: any, index: number) => (
                              <tr
                                key={index}
                                className={
                                  index % 2 === 0 ? "bg-white" : "bg-gray-50"
                                }
                              >
                                <td className="px-6 py-2">{m.from}</td>
                                <td className="px-6 py-2">{m.to}</td>
                                <td className="px-6 py-2">{m.distance} px</td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="text-gray-500">No measurements detected</p>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      )}

     
    </div>
  );
}
