import React, { useState, ChangeEvent } from 'react';
import axios from 'axios';
import Feedback from './Feedback';
import { Button, Typography, CircularProgress, Box } from '@mui/material';

const FileUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleUploadFile = async () => {
    if (!selectedFile) {
      setError('Please select a file first.');
      return;
    }

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await axios.post('http://127.0.0.1:5000/predict', formData);
      setPrediction(res.data.prediction);
      setLoading(false);
    } catch (error: any) {
      setError(error.message);
      setLoading(false);
    }
  };

  return (
    <Box sx={{ textAlign: 'center', marginTop: '20px' }}>
      <Typography variant="h5" gutterBottom>
        Upload Image for Classification
      </Typography>
      <input
        accept="image/*"
        style={{ display: 'none' }}
        id="raised-button-file"
        type="file"
        onChange={handleFileChange}
      />
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          gap: '10px',
          marginBottom: '20px',
          marginTop: '20px',
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
          {loading ? <CircularProgress size={24} /> : 'Upload and Predict'}
        </Button>
      </Box>
      {selectedFile && (
        <Typography variant="body1" gutterBottom>
          Selected file: {selectedFile.name}
        </Typography>
      )}
      {error && (
        <Typography variant="body1" color="error" sx={{ marginTop: '20px' }}>
          {error}
        </Typography>
      )}
      {prediction !== null && (
        <Typography variant="h6" sx={{ marginTop: '20px' }}>
          Prediction Result:{' '}
          <Typography component="span" variant="h6" color="green">
            {prediction}
          </Typography>
        </Typography>
      )}
      {prediction && selectedFile && (
        <Feedback imageName={selectedFile.name} prediction={prediction} />
      )}
    </Box>
  );
};

export default FileUpload;
