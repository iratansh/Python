"""
Weather App using PyQt5 and weatherbit api
Author: Ishaan Ratanshi
"""

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QLineEdit, QTextEdit
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt
import sys, requests

class WeatherApp(QWidget):
    def __init__(self):
        super().__init__()
        self.API_KEY = '725fc8d670f044b4b6cab371441f9d59'

        self.initUI()

    def initUI(self):
        self.font = QFont()
        self.font.setFamily("ComicSansMS")
        self.font.setPointSize(20)

        # Create Window
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)
        self.resize(750, 750)
        self.setStyleSheet("background-color: lightblue; border: 5px solid black;")
        self.setWindowTitle("Weather App")

        # Create and set up title label
        self.title = QLabel("Weather App", self)
        self.title.setStyleSheet("border: none; color: black; ")
        self.title.setFont(self.font)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.move(300, 30)

        # Create Background for Input Section
        self.input_background = QLabel(self)
        self.input_background.setStyleSheet('border-radius: 10px solid black; background-color: lightcyan;')
        self.input_background.resize(450, 100)
        self.input_background.move(160, 80)

        # Create Background for Output Section
        self.output_background = QLabel(self)
        self.output_background.setStyleSheet('border-radius: 10px solid black; background-color: lightcyan;')
        self.output_background.resize(450, 500)
        self.output_background.move(160, 250)

        # Create Input Box (LOCATION)
        self.input_textbox = QLineEdit(self)
        self.input_textbox.setStyleSheet('color: black; border: 3px solid black;')
        self.input_textbox.move(330, 101)
        self.input_textbox.resize(100, 25)

        # Create Input Box Label
        self.input_textbox_label = QLabel("Enter City:", self)
        self.input_textbox_label.setStyleSheet("border: 2px solid black; color: black;")
        self.input_textbox_label.setFont(self.font)
        self.input_textbox_label.move(180, 100)

        # Create Output Box for weather 
        self.output_textbox = QTextEdit(self)
        self.output_textbox.setReadOnly(True)
        self.output_textbox.setStyleSheet("border: 3px solid black; padding: 5px; background: lightblue; color: black")
        self.output_textbox.resize(400, 400)
        self.output_textbox.move(185, 300)

        # Create Output Box Label
        self.output_textbox_label = QLabel("Forecast:", self)
        self.output_textbox_label.setStyleSheet("border: 2px solid black; color: black;")
        self.output_textbox_label.setFont(self.font)
        self.output_textbox_label.resize(100, 30)
        self.output_textbox_label.move(330, 260)

        # Create Search Button 
        self.button = QPushButton(self)
        self.button.setIcon(QIcon('magnifying_glass.png'))
        self.button.clicked.connect(lambda: self.find_weather_for_location(self.input_textbox.text()))
        self.button.move(450, 100)
        self.button.resize(26, 25)

        # Create Search Button (state/country)
        self.state_country_button = QPushButton(self)
        self.state_country_button.setIcon(QIcon('magnifying_glass.png'))
        self.state_country_button.clicked.connect(lambda: self.find_weather_with_country_state())
        self.state_country_button.move(450, 100)
        self.state_country_button.resize(26, 25)
        self.state_country_button.setVisible(False)

    # Find Weather Function
    def find_weather_for_location(self, location):
        self.output_textbox.clear()
        location = location.lower().title()

        base_url = 'https://api.weatherbit.io/v2.0/current'
        params = {'city': location, 'key': self.API_KEY}

        if location != '':
            try:
                response = requests.get(base_url, params=params)
                data = response.json()
                list_of_locations = data.get('data', [])

                # If more than one location matches the input
                if len(list_of_locations) > 1:
                    self.button.setVisible(False)
                    self.state_country_button.setVisible(True)
                    self.output_textbox.append("Error: More than 1 location matches the Input.")
                # If there is only one location that matches the input       
                else:
                    response = requests.get(base_url, params=params)
                    data = response.json()
                    if response.status_code == 200:
                        temperature = data['data'][0]['temp']
                        datetime = data['data'][0]['datetime']
                        gust = data['data'][0]['gust']
                        precip = data['data'][0]['precip']
                        pressure = data['data'][0]['pres']
                        snow = data['data'][0]['snow']
                        sunrise = data['data'][0]['sunrise']
                        sunset = data['data'][0]['sunset']
                        uv = data['data'][0]['uv']
                        weather = data['data'][0]['weather']['description']
                        wind_speed = data['data'][0]['wind_spd']
            
                        self.output_textbox.append(f"Current temperature in {location}: {temperature}°C \nDatetime: {datetime}")
                        self.output_textbox.append(f"Gust: {gust}")
                        self.output_textbox.append(f"Precipitation: {precip}")
                        self.output_textbox.append(f"Pressure: {pressure}")
                        self.output_textbox.append(f"Snow: {snow}")
                        self.output_textbox.append(f"Sunrise + Sunset: {sunrise}, {sunset}")
                        self.output_textbox.append(f"UV: {uv}")
                        self.output_textbox.append(f"Weather: {weather}; Wind Speed: {wind_speed}")
                    else:
                        self.output_textbox.append(f"Error: {data.get('error', 'Unknown error')}")
            except Exception as e:
                self.output_textbox.append(f"Error: {e}")
        else:
            self.output_textbox.append("Error: No Input Detected.")

    # Function to find weather with country and state
    def find_weather_with_country_state(self):
        self.input_background.resize(450, 150)
        # Create input boxes for country and state
        self.input_textbox_country, self.input_textbox_label_country = self.create_input_box("Country Abbreviation:", 440, 140)
        self.input_textbox_state, self.input_textbox_label_state = self.create_input_box("State/Province Abbreviation:", 500, 185)

        # Set initial visibility
        self.input_textbox_country.setVisible(True)
        self.input_textbox_label_country.setVisible(True)
        self.input_textbox_state.setVisible(True)
        self.input_textbox_label_state.setVisible(True)

        location = self.input_textbox.text()
        country = self.input_textbox_country.text()
        state = self.input_textbox_state.text()

        base_url = 'https://api.weatherbit.io/v2.0/current'
        params = {'city': location, 'country': country, 'state': state, 'key': self.API_KEY}

        try:
            response = requests.get(base_url, params=params)
            data = response.json()

            if response.status_code == 200:
                temperature = data['data'][0]['temp']
                datetime = data['data'][0]['datetime']
                gust = data['data'][0]['gust']
                precip = data['data'][0]['precip']
                pressure = data['data'][0]['pres']
                snow = data['data'][0]['snow']
                sunrise = data['data'][0]['sunrise']
                sunset = data['data'][0]['sunset']
                uv = data['data'][0]['uv']
                weather = data['data'][0]['weather']['description']
                wind_speed = data['data'][0]['wind_spd']
            
                self.output_textbox.append(f"Current temperature in {location}: {temperature}°C \nDatetime: {datetime}")
                self.output_textbox.append(f"Gust: {gust}")
                self.output_textbox.append(f"Precipitation: {precip}")
                self.output_textbox.append(f"Pressure: {pressure}")
                self.output_textbox.append(f"Snow: {snow}")
                self.output_textbox.append(f"Sunrise + Sunset: {sunrise}, {sunset}")
                self.output_textbox.append(f"UV: {uv}")
                self.output_textbox.append(f"Weather: {weather} \nWind Speed: {wind_speed}")
            else:
                self.output_textbox.append(f"Error: {data.get('error', 'Unknown error')}")
        except Exception as e:
            self.output_textbox.append(f"Error: {e}")

    # Create Input box function
    def create_input_box(self, label_text, x, y):
        # Create Textbox
        input_textbox = QLineEdit(self)
        input_textbox.setStyleSheet("border: 3px solid black; color: black;")
        input_textbox.move(x, y)
        input_textbox.resize(100, 20)
        # Create Input Box Label
        input_textbox_label = QLabel(label_text, self)
        input_textbox_label.setStyleSheet("border: 2px solid black; color: black")
        input_textbox_label.setFont(self.font)
        input_textbox_label.move(180, y)
        return input_textbox, input_textbox_label

    def run(self):
        self.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    app = QApplication([])
    weather_app = WeatherApp()
    weather_app.run()