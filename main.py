import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import altair as alt

st.title("Анализ данных авиаиндустрии")

st.markdown("Привет!")
st.markdown("В этом проекте я буду анализировать данные авиаиндустрии: по перелётам, самолётам и аэропортам.")
st.markdown("Это первая часть анализа, выполненная в питоне. Чтобы увидеть вторую часть (R), пройди по ссылке **PLACEHOLDER**.")

st.markdown("Список использованных технологий:")
st.markdown("1. Pandas")
st.markdown("2. SQL")
st.markdown("3. GeoPandas")
st.markdown("4. Продвинутый web scraping с помощью Selenium")
st.markdown("5. Streamlit")
st.markdown("6. API")
st.markdown("7. Networkx")
st.markdown("8. Visualisation (plotly)")
st.markdown("Во второй части проекта использованы:")
st.markdown("9. R")
st.markdown("10. Machine Learning (kNN)")
st.markdown("11. Ggplot2, продвинутое использование с многослойными картинками")
st.markdown("12. Дополнения к ggplot2")
st.markdown("Всего строк кода порядка *XXX*")
st.markdown("Весь проект - единое целое")
st.markdown("И, конечно, он крутой!")

st.markdown("## 1. Описание доступа к данным по авиаперелётам")
st.markdown("Мой основной источник данных - OpenSky Network (https://opensky-network.org). Портал щедро предоставил мне доступ к данным, за что им большое спасибо!")
st.markdown("Данные лежат в Impala Shell (это такая оболочка над базой данных Impala). К оболочке можно подключаться и делать к ней SQL-запросы.")
st.markdown("1. Сначала нужно завести аккаунт на OpenSky Network и запросить доступ к данным (https://opensky-network.org/index.php?option=com_users).")
st.markdown("2. Затем нужно подключиться к БД. Для этого нужно запустить в терминале команду `ssh -p 2230 -l USERNAME data.opensky-network.org`. Вместо `USERNAME` нужно написать свой логин (кажется, он совпадает с фамилией, написанной с большой буквы).")
st.markdown("3. Чтобы увидеть таблицы, которые лежат в БД, нужно запустить команду `show tables;`. Получится такой список:")
st.markdown("""`+--------------------------+
| name                     |
+--------------------------+
| acas_data4               |
| allcall_replies_data4    |
| flarm_raw                |
| flights                  |
| flights_data4            |
| identification_data4     |
| operational_status_data4 |
| position_data4           |
| rollcall_replies_data4   |
| sensor_visibility        |
| sensor_visibility_data3  |
| state_vectors            |
| state_vectors_data3      |
| state_vectors_data4      |
| velocity_data4           |
+--------------------------+`""")
st.markdown("4. Я буду пользоваться только одной из них - `flights`. Это данные по всем завершившимся полётам с 1 января 2016 года. Узнать, что там лежит, можно с помощью команды `describe flights;`")
st.markdown("""`+-------------+--------+---------+
| name        | type   | comment |
+-------------+--------+---------+
| icao24      | string |         |
| firstseen   | string |         |
| lastseen    | string |         |
| duration    | bigint |         |
| callsign    | string |         |
| departure   | string |         |
| destination | string |         |
+-------------+--------+---------+`""")
st.markdown("`icao24` - это уникальный 24-битный код трансмиттера, который стоит на каждом уважающем себя воздушном судне. Эти трансмиттеры передают данные о важных характеристиках полёта - местоположении, направлении, скорости, и так далее. Именно благодаря данным с трансмиттеров существует эта БД. Благодаря этим же данным работает какой-нибудь flightradar24.com. Поскольку код уникальный, а трансмиттеры зачастую не заменяются, этот код можно отождествить с конкретным самолётом.")
st.markdown("`duration` - это длительность полёта в минутах.")
st.markdown("`departure` и `destination` - это соответственно аэропорт отправления и аэропорт назначения. Аэропорты для краткости закодированы, и в этих колонках лежат их ICAO-коды (ICAO - это International Civil Aviation Organization, такой международный регулятор/стандартизатор. Они занимаются в том числе унификацией данных - например, общепринятых алиасов аэропортов).")
st.markdown("`firstseen` и `lastseen` - это время начала и окончания полёта.")
st.markdown("`callsign` - это позывной, который авиалиния даёт рейсу. Он не стандартизирован ICAO, поэтому мне он не нужен.")
st.markdown("5. Чтобы достать данные о каком-нибудь полёте, нужно просто исполнить SQL-запрос: `select icao24, firstseen, lastseen, duration, departure, destination from flights limit 1;`")
st.markdown("""`+--------+---------------------+---------------------+----------+-----------+-------------+
| icao24 | firstseen           | lastseen            | duration | departure | destination |
+--------+---------------------+---------------------+----------+-----------+-------------+
| a029b7 | 2017-10-09 20:30:45 | 2017-10-09 21:13:12 | 42       | KFKL      | KJHW        |
+--------+---------------------+---------------------+----------+-----------+-------------+`""")
st.markdown("""Самолёт с трансмиттером `a029b7` вылетел 9 октября 2017 года из `KFKL` в `KJHW` на 42 минуты. При желании можно узнать, что это легкомоторный самолёт (https://opensky-network.org/aircraft-profile?icao24=a029b7), а летел он из 
Venango Regional Airport (штат Пенсильвания) в Chautauqua County-Jamestown Airport (штат Нью-Йорк).""")
st.markdown("""6. К сожалению, у меня не получилось интегрировать конкретно эту Impala Shell в питон-скрипт, поэтому придётся делать это руками.
            Буду копировать output терминала, вставлять в файл, а потом парсить средствами питона.""")
st.markdown("7. Теперь я могу собрать нужные мне данные в CSV-файл. Об этом ниже.")

st.markdown("## 2. Выгрузка данных по авиаперелётам")
st.markdown("Возьмём данные за 2019 год. Это достаточно длинный период времени, к тому же не искажённый пандемией.")
st.markdown("Всего в таблице `flights` 118 714 936 записей, а за 2019 год - 32 088 373. Это довольно много, heroku такое не вывезет. Поэтому сделаем несколько файлов с нужными агрегациями.")
st.markdown("Считаем количество перелетов по каждой (упорядоченной) паре аэропортов")
st.markdown("Исполняем SQL-запрос (использован **SQL** - не забудь поставить за это 1 балл):")
st.markdown("""`select
departure, destination, count(icao24)
from flights
where
firstseen >= "2019"
and lastseen < "2020"
group by departure, destination;`
""")
st.markdown("Сохраняем в файл flights_by_airport_pairs.txt")
st.markdown("Теперь напишем функцию, которая будет парсить этот файл")
with st.echo():
    def parse_hadoop_output(filename):
        f = open(filename, "r")
        for i, line in enumerate(f):
            if i == 1:
                header = parse_line(line)
                ncolumns = len(header)
                columns_data = dict.fromkeys(header, 0)
            elif len(line) == 1:
                continue
            elif line[1] != "-":
                content = parse_line(line)
                if content != header:
                    if columns_data[header[0]] == 0:
                        for ncol in range(ncolumns):
                            columns_data[header[ncol]] = [content[ncol],]
                        continue
                    for ncol in range(ncolumns):
                        columns_data[header[ncol]].append(content[ncol])
        f.close()
        df = pd.DataFrame(data=columns_data).rename(columns={'count(icao24)': 'count'})
        return df

    def parse_line(line):
        content = [x.strip(" ") for x in line.strip("|").split("|")][:-1]
        return content

    df_fl = parse_hadoop_output("flights_by_airports.txt")
    df_fl[:5]
    st.write(df_fl["departure"].nunique())

st.markdown("## 4. Получение данных по аэропортам")
st.markdown("Чтобы сделать красивые визуализации, нужно понять, где находится каждый аэропорт.")
st.markdown("Это отличная возможность использовать **Selenium**!")
st.markdown("Мой комп умрёт парсить 14 000 аэропортов, поэтому тберём только крупные. Скажем, с как минимум 500 рейсами в год.")
st.markdown("Таких всего:")

with st.echo(code_location="below"):
    df = df_fl.copy()
    df = df[pd.notna(df.departure)]
    df = df[pd.notna(df.destination)]
    df_dep = df.departure.value_counts().copy().reset_index()
    df_des = df.destination.value_counts().copy()
    df = df_dep.join(df_des, how='outer', on="index")
    df.fillna(0, inplace=True)
    df['total'] = df.departure + df.destination
    df = df[df.total >= 500]
    df.rename(columns={'index': 'airport'}, inplace=True)
    df.to_csv("major_airports.csv", sep=';')
    st.write(df.shape[0])

st.markdown("Чтобы тебе не пришлось ждать вечность, пока все аэропорты пропарсятся на Heroku, я делаю парсинг у себя, а результаты сохраняю в `airport_data.csv`")
st.markdown("Парсить будем с сайта https://www.avcodes.co.uk/aptcodesearch.asp")
st.markdown("Вот код парсера:")
st.code("""
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
""")

st.markdown("Парсер нашё данные только по 401 аэропорту. Остальные аэропорты, видимо, слишком маленькие, и их нет в базе на том сайте.")

with st.echo():
    df_ap = pd.read_csv("major_airports_data.csv", sep=';')
    st.write(df_ap[:5])

st.markdown("## 5. Визуализации")

st.markdown("### Граф связности аэропортов")
st.markdown("""Интересно посмотреть, какие есть ключевые транспортные хабы - аэропорты, которые связаны с многими другими.
Скорее всего, это будут какие-нибудь Heathrow (Лондон), O'Hare (Чикаго), Аэропорт Шанхая, JFK (Нью-Йорк).
В этих аэропортах много трансконтинентальных рейсов, пассажиры которых затем пересаживаются и летят в более мелкие аэропорты.
Проверим интуицию :)""")
st.markdown("Выкинем все перелёты, где одна из двух точек неизвестна, а также перелёты, кончившиеся там же, где и начались")
st.markdown("""Выкинем также все связки между аэропортами, где меньше 1460 перелётов в год в каждую сторону (по 4 в неделю).
Можно было бы поставить отсечку поменьше, но тогда граф будет слишком нечитаемый - точки будут неразличимы друг от друга.
С другой библиотекой (не `networkx`, а `Gephi`) было бы лучше.""")

"""with st.echo(code_location="below"):
    df_fl_graph = df_fl.copy()
    df_fl_graph = df_fl_graph[df_fl_graph['destination'] != "NULL"]
    df_fl_graph = df_fl_graph[df_fl_graph['departure'] != "NULL"]
    df_fl_graph = df_fl_graph[df_fl_graph.departure != df_fl_graph.destination]
    df_fl_graph['a1'] = df_fl_graph[['destination', 'departure']].min(axis=1)
    df_fl_graph['a2'] = df_fl_graph[['destination', 'departure']].max(axis=1)
    df_fl_graph.drop(['departure', 'destination'], inplace=True, axis=1)
    df_fl_graph['count'] = df_fl_graph['count'].astype(int)
    df_fl_graph = df_fl_graph[df_fl_graph['count'] >= 365*4]
    df_fl_graph = df_fl_graph.groupby(['a1', 'a2']).sum().reset_index()
    G = nx.from_pandas_edgelist(df_fl_graph, source='a1', target='a2', edge_attr="count")
    degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
    top_nodes = [x[0] for x in degrees[:5]]
    other_nodes = [x[0] for x in degrees[5:]]
    plt.plot(1)
    nx.draw(G, node_size=[5, ]*len(other_nodes) + [300, ]*5,
            pos=nx.nx_agraph.graphviz_layout(G, prog="fdp"),
            node_color=['blue', ]*len(other_nodes) + ['orange', ]*5,
            nodelist=other_nodes + top_nodes,
            edge_color='grey',
            labels=dict(zip(top_nodes, top_nodes)),
            alpha=0.85,
            font_size=4,
            font_family='helvetica',
            width=0.1)
    fig = plt.plot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)"""

st.markdown("Видно, что 5 узловых аэропортов - это EGLL (Heathrow, London), EDDF (Franfkfurt), KORD (O'Hare, Chicago), KDFW (Dallas-Fort Worth), KLAX (Los Angeles). Интуиция плюс-минус не подвела :)")

st.markdown("### Самые крупные аэропорты на карте-1")
st.markdown("Давай посмотрим, где находятся 10 крупнейших аэропортов мира (по количеству рейсов)")

with st.echo(code_location="below"):
    df1 = pd.read_csv("major_airports.csv", sep=';').drop("Unnamed: 0", axis=1)
    df2 = pd.read_csv("major_airports_data.csv", sep=';').drop("Unnamed: 0", axis=1)
    df = df2.join(df1.set_index("airport"), on='airport').drop(columns=['departure', 'destination'])
    df['total'] = df.total.astype('int32')
    df_slice = df.sort_values(by=['total'], ascending=False)[:10]
    gdf = gpd.GeoDataFrame(df_slice, geometry=gpd.points_from_xy(df_slice.lon, df_slice.lat))
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ax = world.plot(facecolor='w', edgecolor='k')
    plt.plot(2)
    gdf.plot(markersize=np.square(gdf['total'])/1000000, ax=ax)
    fig2 = plt.plot()
    st.pyplot(fig2)

st.markdown("Все в Америке, кто бы мог подумать...")

st.markdown("### Самые крупные аэропорты на карте-2")
st.markdown("У geopandas не самый богатый встроенный функционал. По сути он работает на `matplotlib` без каких-то модификаций, поэтому красоты и интерактивности от него ждать не стоит.")
st.markdown("Хорошо, что есть библиотеки получше! Например, `plotly`")
st.markdown("Оставим только 100 крупнейших аэропортов")
st.markdown("Картинку лучше открыть целиком (для этого есть кнопка в правом верхнем углу).")

with st.echo(code_location="below"):
    df = df.sort_values(by=['total'], ascending=False)[:100]
    fig3 = px.scatter_geo(df, lat='lat', lon='lon', size=df['total']/1000, color='country',
                          hover_data=['country', 'city', 'name', 'total'],
                          hover_name='name', opacity=0.7, scope='world', title='Крупнейшие аэропорты',
                          basemap_visible=True)
    st.write(fig3)

st.markdown("Картинка получилась посимпатичнее, круто!")

st.markdown("## 6. Получение данных по самолётам")
st.markdown("Прежде чем уйти в R, чтобы поработать там с ML, нужно подготовить данные.")
st.markdown("Будем пользоваться API сайта aviationstack.com.")
st.markdown("Там всего 500 реквестов, поэтому я работаю с API локально, а код просто прикрепляю")
st.code("""
import json as js
import pandas as pd
import requests as rq

def execute_request(API_key, entrypoint, offset):
    url = entrypoint + f"&offset={offset}"
    r = rq.get(url)
    j = r.json()
    data = j['data']
    df = pd.DataFrame(data)
    return df

API_key = "YOUR_API_KEY"  # если хочешь, можешь получить свой ключ на сайте
entrypoint = f"http://api.aviationstack.com/v1/airplanes?access_key={API_key}"

for i in range(190):
    try:
        df = execute_request(API_key=API_key, entrypoint=entrypoint, offset=i*100)
    except:
        print(i)
    if i == 0:
        df_fetched = df.copy()
    else:
        df_fetched = df_fetched.append(df, ignore_index=True)

df_fetched.to_csv("aircraft_data.csv", sep=';')
""")

st.markdown("Вот код этой части целиком. Я посчитал строчки, их уже около 130. А ещё есть часть в R :)")
st.code("""
def parse_hadoop_output(filename):
    f = open(filename, "r")
    for i, line in enumerate(f):
        if i == 1:
            header = parse_line(line)
            ncolumns = len(header)
            columns_data = dict.fromkeys(header, 0)
        elif len(line) == 1:
            continue
        elif line[1] != "-":
            content = parse_line(line)
            if content != header:
                if columns_data[header[0]] == 0:
                    for ncol in range(ncolumns):
                        columns_data[header[ncol]] = [content[ncol],]
                    continue
                for ncol in range(ncolumns):
                    columns_data[header[ncol]].append(content[ncol])
    f.close()
    df = pd.DataFrame(data=columns_data).rename(columns={'count(icao24)': 'count'})
    return df
def parse_line(line):
    content = [x.strip(" ") for x in line.strip("|").split("|")][:-1]
    return content
df_fl = parse_hadoop_output("flights_by_airports.txt")
df_fl[:5]
st.write(df_fl["departure"].nunique())
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
df_ap = pd.read_csv("major_airports_data.csv", sep=';')
st.write(df_ap[:5])
df_fl_graph = df_fl.copy()
df_fl_graph = df_fl_graph[df_fl_graph['destination'] != "NULL"]
df_fl_graph = df_fl_graph[df_fl_graph['departure'] != "NULL"]
df_fl_graph = df_fl_graph[df_fl_graph.departure != df_fl_graph.destination]
df_fl_graph['a1'] = df_fl_graph[['destination', 'departure']].min(axis=1)
df_fl_graph['a2'] = df_fl_graph[['destination', 'departure']].max(axis=1)
df_fl_graph.drop(['departure', 'destination'], inplace=True, axis=1)
df_fl_graph['count'] = df_fl_graph['count'].astype(int)
df_fl_graph = df_fl_graph[df_fl_graph['count'] >= 365*4]
df_fl_graph = df_fl_graph.groupby(['a1', 'a2']).sum().reset_index()
G = nx.from_pandas_edgelist(df_fl_graph, source='a1', target='a2', edge_attr="count")
degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
top_nodes = [x[0] for x in degrees[:5]]
other_nodes = [x[0] for x in degrees[5:]]
plt.plot(1)
nx.draw(G, node_size=[5, ]*len(other_nodes) + [300, ]*5,
        pos=nx.nx_agraph.graphviz_layout(G, prog="fdp"),
        node_color=['blue', ]*len(other_nodes) + ['orange', ]*5,
        nodelist=other_nodes + top_nodes,
        edge_color='grey',
        labels=dict(zip(top_nodes, top_nodes)),
        alpha=0.85,
        font_size=4,
        font_family='helvetica',
        width=0.1)
fig = plt.plot()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig)
df1 = pd.read_csv("major_airports.csv", sep=';').drop("Unnamed: 0", axis=1)
df2 = pd.read_csv("major_airports_data.csv", sep=';').drop("Unnamed: 0", axis=1)
df = df2.join(df1.set_index("airport"), on='airport').drop(columns=['departure', 'destination'])
df['total'] = df.total.astype('int32')
df_slice = df.sort_values(by=['total'], ascending=False)[:10]
gdf = gpd.GeoDataFrame(df_slice, geometry=gpd.points_from_xy(df_slice.lon, df_slice.lat))
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
ax = world.plot(facecolor='w', edgecolor='k')
plt.plot(2)
gdf.plot(markersize=np.square(gdf['total'])/1000000, ax=ax)
fig2 = plt.plot()
st.pyplot(fig2)
df = df.sort_values(by=['total'], ascending=False)[:100]
fig3 = px.scatter_geo(df, lat='lat', lon='lon', size=df['total']/1000, color='country',
                      hover_data=['country', 'city', 'name', 'total'],
                      hover_name='name', opacity=0.7, scope='world', title='Крупнейшие аэропорты',
                      basemap_visible=True)
st.write(fig3)
def execute_request(API_key, entrypoint, offset):
    url = entrypoint + f"&offset={offset}"
    r = rq.get(url)
    j = r.json()
    data = j['data']
    df = pd.DataFrame(data)
    return df
API_key = "YOUR_API_KEY"  # если хочешь, можешь получить свой ключ на сайте
entrypoint = f"http://api.aviationstack.com/v1/airplanes?access_key={API_key}"
for i in range(190):
    try:
        df = execute_request(API_key=API_key, entrypoint=entrypoint, offset=i*100)
    except:
        print(i)
    if i == 0:
        df_fetched = df.copy()
    else:
        df_fetched = df_fetched.append(df, ignore_index=True)
df_fetched.to_csv("aircraft_data.csv", sep=';')
""")