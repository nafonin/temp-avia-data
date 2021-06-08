from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
import pandas as pd

driver=Chrome("/Users/afonin/PycharmProjects/temp-avia-data/chromedriver")

def get_info(icao):
    driver.get("https://www.avcodes.co.uk/aptcodesearch.asp")

    driver.find_element_by_name('icaoapt').send_keys(icao)
    driver.find_element_by_name('B1').click()

    lon = driver.find_element(By.XPATH, "//*[contains(text(), 'Longitude:')]").get_attribute("innerHTML").splitlines()
    lat = driver.find_element(By.XPATH, "//*[contains(text(), 'Latitude:')]").get_attribute("innerHTML").splitlines()
    name = driver.find_element_by_class_name('tablebg').get_attribute("innerHTML").splitlines()
    country = driver.find_element(By.XPATH, "//*[contains(text(), 'Country:')]").get_attribute("innerHTML").splitlines()

    lon = lon[0][14:]
    if lon[-1] == "W":
        lon = -(int(lon[:3]) + int(lon[4:6])/60 + int(lon[7:9])/3600)
    else:
        lon = int(lon[:3]) + int(lon[4:6]) / 60 + int(lon[7:9]) / 3600

    lat = lat[0][13:]
    if lat[-1] == "S":
        lat = -((int(lat[:3]) + int(lat[4:6])/60 + int(lat[7:9])/3600))
    else:
        lat = int(lat[:3]) + int(lat[4:6]) / 60 + int(lat[7:9]) / 3600

    city = name[0].strip('&nbsp;').split(' / ')[0]
    name = name[0][name[0].find('/')+2:].replace('&nbsp;', '')

    country = country[0].split('<br>')[1]

    return [lat, lon, city, name, country]

df = pd.read_csv("major_airports.csv", sep=';')
airports = df.airport.to_list()
df = pd.DataFrame(columns=['airport', 'lat', 'lon'])
cnt = 0

for airport in airports[1:]:
    try:
        lat, lon, city, name, country = get_info(airport)
        if (country == "United States of America") and (lon > 0):  # фиксю баг сайта
            lon = -lon
        df = df.append({'airport': airport, 'lat': lat, 'lon': lon, 'name': name, 'city': city, 'country': country},
                       ignore_index=True)
    except:
        cnt += 1
        print(f"{cnt}. {airport}")

df.to_csv("major_airports_data.csv", sep=';')