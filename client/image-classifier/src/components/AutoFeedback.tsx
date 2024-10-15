import React, { useState } from 'react';
import axios from 'axios';
import { Button, Box, CircularProgress, Typography } from '@mui/material';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const AutoFeedback: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [status, setStatus] = useState<string | null>('');

  const handleAutoFeedback = async () => {
    setLoading(true);
    setStatus('Processing...');

    try {
      const res = await axios.post('http://127.0.0.1:5000/auto_feedback');
      if (res.status === 200) {
        toast.success('All images submitted', {
          position: 'top-right',
        });
        setStatus('All images submitted for auto-feedback');
      }
    } catch (error: any) {
      toast.success('All images submitted', {
        position: 'top-right',
      });
      setStatus('Error submitting images for auto-feedback');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <ToastContainer />
      <Button onClick={handleAutoFeedback} variant="contained" color="primary">
        Start Auto Feedback
      </Button>
      {loading && <CircularProgress />}
      {status && <Typography>{status}</Typography>}
    </Box>
  );
};

export default AutoFeedback;
