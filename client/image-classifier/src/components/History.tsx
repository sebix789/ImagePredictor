import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Card,
  CardContent,
  Typography,
  Grid2 as Grid,
  Box,
  Button,
} from '@mui/material';
import Prediction from '../types/Prediction';
import HistoryProps from '../props/HistoryProps';

const History: React.FC<HistoryProps> = ({ onHide }) => {
  const [history, setHistory] = useState<Prediction[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await axios.get(`${process.env.API_URL}/history`);
        setHistory(res.data.predictions);
      } catch (error: any) {
        setError(error.response ? error.response.data.error : error.message);
      }
    };

    fetchHistory();
  }, []);

  return (
    <Box sx={{ marginTop: '20px' }}>
      <Typography variant="h5" gutterBottom>
        Prediction History
      </Typography>
      <Button
        variant="contained"
        color="secondary"
        onClick={onHide}
        sx={{ marginBottom: '20px' }}
      >
        Hide Prediction History
      </Button>
      {error && (
        <Typography variant="body1" color="error">
          {error}
        </Typography>
      )}
      <Grid container spacing={2}>
        {history.map((entry, index) => (
          // @ts-ignore
          <Grid item xs={12} md={6} lg={4} key={index}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="body1" color="textSecondary">
                  Image: {entry.image_name}
                </Typography>
                <Typography variant="body2">
                  Prediction: {entry.prediction}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default History;
