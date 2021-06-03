import requests
from bs4 import BeautifulSoup   


page = requests.get("https://forecast.weather.gov/MapClick.php?lat=37.777120000000025&lon=-122.41963999999996")


soup = BeautifulSoup(page.content, 'html.parser')



div1 = soup.find_all('div', class_="tombstone-container")[0]


print(div1)