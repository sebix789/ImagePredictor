import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Card,
  CardContent,
  Typography,
  Grid2 as Grid,
  Box,
  Button,
} from "@mui/material";

const History = ({ onHide }) => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const fetchHistory = async () => {
      const res = await axios.get("http://127.0.0.1:5000/history");
      setHistory(res.data.predictions);
    };

    fetchHistory();
  }, []);

  return (
    <Box sx={{ marginTop: "20px" }}>
      <Typography variant="h5" gutterBottom>
        Prediction History
      </Typography>
      <Button
        variant="contained"
        color="secondary"
        onClick={onHide}
        sx={{ marginBottom: "20px" }}
      >
        Hide Prediction History
      </Button>
      <Grid container spacing={2}>
        {history.map((entry, index) => (
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
