import React, { useState } from 'react';
import axios from 'axios';
import {
  Button,
  Box,
  CircularProgress,
  Typography,
  List,
  ListItem,
} from '@mui/material';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const AutoFeedback: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [status, setStatus] = useState<string | null>('');
  const [results, setResults] = useState<any>({});

  const handleAutoFeedback = async () => {
    setLoading(true);
    setStatus('Processing...');

    try {
      const res = await axios.post('http://127.0.0.1:5000/auto-feedback');
      if (res.status === 200) {
        toast.success('All images submitted', {
          position: 'top-right',
        });
        setStatus('All images submitted for auto-feedback');
        setResults(res.data);
      }
    } catch (error: any) {
      toast.error('Error submitting images', {
        position: 'top-right',
      });
      setStatus('Error submitting images for auto-feedback');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        marginTop: '20px',
        display: 'flex',
        flexDirection: 'column',
        justifycontent: 'center',
        alignItems: 'center',
        gap: '10px',
      }}
    >
      <ToastContainer />
      <Button onClick={handleAutoFeedback} variant="contained" color="primary">
        Start Auto Feedback
      </Button>
      {loading && <CircularProgress />}
      {status && <Typography>{status}</Typography>}
      {!loading && results.length > 0 && (
        <Box sx={{ width: '100%', maxWidth: 360 }}>
          <List>
            <ListItem>
              <Typography variant="h6">
                Total Predictions: {results.summary.total_predictions}
              </Typography>
            </ListItem>
            <ListItem>
              <Typography variant="h6">
                Correct Predictions: {results.summary.correct_predictions}
              </Typography>
            </ListItem>
            <ListItem>
              <Typography variant="h6">
                Accuracy: {results.summary.accuracy.toFixed(2)}%
              </Typography>
            </ListItem>
          </List>
        </Box>
      )}
    </Box>
  );
};

export default AutoFeedback;
