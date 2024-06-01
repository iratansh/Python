import React, { useEffect, useState } from 'react';

const StockPredictionImage = ({ stock }) => {
  const [imageSrc, setImageSrc] = useState('');

  useEffect(() => {
    if (stock) {
      const newImageSrc = `./stock/public/${stock}_predictions.png`;
      setImageSrc(newImageSrc);
    }

    return () => {
      setImageSrc('');
    };
  }, [stock]);

  return (
    <div>
      {imageSrc && <img src={imageSrc} alt={`Stock Predictions for ${stock}`} />}
    </div>
  );
};

export default StockPredictionImage;

