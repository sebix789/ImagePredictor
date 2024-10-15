import React, { useState, useEffect } from 'react';
import axios from 'axios';
import FeedbackProps from '../props/FeedbackProps';
import {
  Button,
  Typography,
  Box,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const Feedback: React.FC<FeedbackProps> = ({ imageName, prediction }) => {
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  const [originalLabels, setOriginalLabels] = useState<string[]>([]);
  const [formattedLabels, setFormattedLabels] = useState<string[]>([]);
  const [selectedLabel, setSelectedLabel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchLabels = async () => {
      try {
        const res = await axios.get('http://127.0.0.1:5000/labels');
        const original = res.data.labels;
        const formatted = res.data.formatted_labels;

        const combinedLabels = original.map(
          (label: string, index: string | number) => ({
            original: label,
            formatted: formatted[index],
          }),
        );

        combinedLabels.sort((a: any, b: any) =>
          a.formatted.localeCompare(b.formatted),
        );

        setOriginalLabels(
          combinedLabels.map(
            (item: { original: string; formatted: string }) => item.original,
          ),
        );
        setFormattedLabels(
          combinedLabels.map(
            (item: { original: string; formatted: string }) => item.formatted,
          ),
        );
      } catch (error) {
        setError('Error fetching labels');
      }
    };

    fetchLabels();
  }, []);

  const handleFeedbackSubmit = async () => {
    if (isCorrect == null || (!isCorrect && !selectedLabel)) {
      setError('Please provide feedback whether the prediction is correct');
      return;
    }

    try {
      await axios.post('http://127.0.0.1:5000/feedback', {
        image_name: imageName,
        is_correct: isCorrect,
        true_label: !isCorrect ? selectedLabel : null,
      });

      setError(null);
      toast.success('Feedback submitted successfully', {
        position: 'top-right',
      });

      setTimeout(() => {
        window.location.reload();
      }, 2000);
    } catch (error) {
      toast.error('Failed to submitting feedback', {
        position: 'top-right',
      });
    }
  };

  return (
    <Box sx={{ marginTop: '20px', textAlign: 'center' }}>
      <ToastContainer />
      <Typography variant="body1" gutterBottom>
        Is the prediction correct for "{prediction}"?
      </Typography>
      <Button
        variant={isCorrect === true ? 'contained' : 'outlined'}
        color="success"
        onClick={() => setIsCorrect(true)}
      >
        Yes
      </Button>
      <Button
        variant={isCorrect === false ? 'contained' : 'outlined'}
        color="error"
        onClick={() => setIsCorrect(false)}
        sx={{ marginLeft: '10px' }}
      >
        No
      </Button>

      {isCorrect === false && (
        <Box
          sx={{
            marginTop: '20px',
            display: 'flex',
            justifyContent: 'center',
            marginBottom: '20px',
          }}
        >
          <FormControl fullWidth>
            <InputLabel id="label-select">Select Correct Label</InputLabel>
            <Select
              labelId="label-select"
              value={selectedLabel || ''}
              onChange={(event) => setSelectedLabel(event.target.value)}
            >
              {formattedLabels.map((label, index) => (
                <MenuItem key={label} value={originalLabels[index]}>
                  {label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
      )}

      <Button
        variant="contained"
        color="primary"
        onClick={handleFeedbackSubmit}
        disabled={isCorrect === null || (!isCorrect && !selectedLabel)}
        sx={{ marginLeft: '10px' }}
      >
        Submit Feedback
      </Button>

      {error && (
        <Typography variant="body1" color="error" sx={{ marginTop: '20px' }}>
          {error}
        </Typography>
      )}
    </Box>
  );
};

export default Feedback;
