import React from 'react';
import ImageUpload from './ImageUpload';
import axios from 'axios';

axios.get('/api/users')
    .then(response => {
        console.log(response.data);
    })
    .catch(error => {
        console.error(error);
    });

function App() {
    return (
        <div>
            <header>
                <h1>Object Detection App</h1>
            </header>
            <main>
                <ImageUpload />
            </main>
        </div>
    );
}

export default App;