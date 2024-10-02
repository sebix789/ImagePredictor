import React, { useState } from "react";
import axios from "axios";
import { Button, Typography, CircularProgress, Box } from "@mui/material";

const FileUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleUploadFile = async () => {
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", formData);
      setPrediction(res.data.prediction);
      setLoading(false);
    } catch (error) {
      setError(error.message);
      setLoading(false);
    }
  };

  return (
    <Box sx={{ textAlign: "center", marginTop: "20px" }}>
      <Typography variant="h5" gutterBottom>
        Upload Image for Classification
      </Typography>
      <input
        accept="image/*"
        style={{ display: "none" }}
        id="raised-button-file"
        type="file"
        onChange={(e) => setSelectedFile(e.target.files[0])}
      />
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          gap: "10px",
          marginBottom: "20px",
        }}
      >
        <label htmlFor="raised-button-file">
          <Button variant="contained" component="span">
            Select Image
          </Button>
        </label>
        <Button
          variant="contained"
          color="primary"
          onClick={handleUploadFile}
          disabled={!selectedFile || loading}
        >
          {loading ? <CircularProgress size={24} /> : "Upload and Predict"}
        </Button>
      </Box>
      {selectedFile && (
        <Typography variant="body1" gutterBottom>
          Selected file: {selectedFile.name}
        </Typography>
      )}
      {error && (
        <Typography variant="body1" color="error" sx={{ marginTop: "20px" }}>
          {error}
        </Typography>
      )}
      {prediction !== null && (
        <Typography variant="h6" sx={{ marginTop: "20px" }}>
          Prediction Result:{" "}
          <Typography component="span" variant="h6" color="green">
            {prediction}
          </Typography>
        </Typography>
      )}
    </Box>
  );
};

export default FileUpload;
