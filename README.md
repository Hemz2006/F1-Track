
# F1 Track

F1 Track is a web application for analyzing and replaying Formula 1 race data. It provides a dashboard for viewing race results, driver and team standings, circuit information, and detailed telemetry for races.

## Features

- **Race Replay**: Watch a replay of any race with live telemetry data including driver positions, lap times, tire usage, and more
- **Driver Standings**: See current World Drivers' Championship standings and detailed stats for each driver
- **Constructor Standings**: View current World Constructors' Championship standings 
- **Race Results**: Get full results for every race of the season including finishing positions, fastest laps, and pit stop data
- **Circuit Information**: Explore details on every track in the F1 calendar with map layouts and corner analysis
- **Data Analytics**: Dig into the data with visualization tools to compare teammate battles, analyze race pace, and spot trends

## Technical Overview

The F1 Track application consists of:

- **Backend**: Python Flask server that pulls live race data from the FastF1 API and stores it in a SQLite database
- **Frontend**: HTML templates with JavaScript to render the UI components and visualizations 
- **Data Analysis**: Jupyter notebooks to explore the data and build machine learning models

## Getting Started

1. Install the required Python packages: `pip install -r requirements.txt`
2. Start the Flask server: `python app.py`
3. Open a web browser and go to `http://localhost:5000` to access the application

## Data Sources

- [FastF1](https://theoehrly.github.io/Fast-F1/) - Python package for accessing F1 timing and telemetry data
- [Ergast API](http://ergast.com/mrd/) - Historical motor racing data for non-commercial use

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue on the GitHub repository. If you'd like to contribute code changes, you can fork the repository and open a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
