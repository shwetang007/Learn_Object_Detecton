import React, { useState } from 'react';
import axios from 'axios';

const ImageUpload = () => {
    const [image, setImage] = useState(null);
    const [results, setResults] = useState([]);
    const [annotatedImageUrl, setAnnotatedImageUrl] = useState(null);

    const handleImageChange = (e) => {
        setImage(e.target.files[0]);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const formData = new FormData();
        formData.append('image', image);

        const response = await axios.post('http://localhost:5000/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });

        setResults(response.data.results);
        setAnnotatedImageUrl(`http://localhost:5000${response.data.annotated_image_url}`);
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input type="file" onChange={handleImageChange} />
                <button type="submit">Upload</button>
            </form>
            <div>
                {results.map((result, index) => (
                    <div key={index}>
                        <p>{result.name}: {result.confidence.toFixed(2)}</p>
                    </div>
                ))}
            </div>
            {annotatedImageUrl && (
                <img src={annotatedImageUrl} alt="Annotated" />
            )}
        </div>
    );
};

export default ImageUpload;
