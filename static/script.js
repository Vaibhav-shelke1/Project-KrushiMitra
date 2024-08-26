// script.js

// Your OpenWeatherMap API key
const apiKey = '9943f620b459fdcfd2d1e4386fc75bd6';

// Initialize the map
const map = L.map('map').setView([34.0479, 100.6197], 3); // Centered at (0,0) with zoom level 2

// Add a tile layer (base map)
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
}).addTo(map);

// Function to get precipitation data and add to map
async function getPrecipitationData() {
    const url = `https://tile.openweathermap.org/map/precipitation_new/{z}/{x}/{y}.png?appid=${apiKey}`;
    const precipitationLayer = L.tileLayer(url, {
        attribution: '&copy; <a href="https://openweathermap.org/">OpenWeatherMap</a>',
        opacity: 0.8,
        tileSize:256,
    });
    map.addLayer(precipitationLayer);
}

// Fetch and display precipitation data
getPrecipitationData();
