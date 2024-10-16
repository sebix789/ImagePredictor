import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import History from './components/History';
import AutoFeedback from './components/AutoFeedback';
import {
  Container,
  Typography,
  Box,
  Button,
  createTheme,
  ThemeProvider,
} from '@mui/material';

const theme = createTheme();

function App() {
  const [showHistory, setShowHistory] = useState(false);

  return (
    <ThemeProvider theme={theme}>
      <Container>
        <Box sx={{ textAlign: 'center', marginTop: '40px' }}>
          <Typography variant="h3" gutterBottom>
            Image Classification
          </Typography>
          <FileUpload />
          <Button
            variant="contained"
            color="secondary"
            onClick={() => setShowHistory(true)}
            sx={{ marginTop: '20px' }}
          >
            Show Prediction History
          </Button>
          {showHistory && <History onHide={() => setShowHistory(false)} />}
          <AutoFeedback />
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
